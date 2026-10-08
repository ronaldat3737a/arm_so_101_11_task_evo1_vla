"""Read-only audit of a LeRobotDataset (codebase v3.0) snapshot on disk.

Everything here works on the raw files (meta/info.json, meta/tasks.parquet, meta/episodes/*.parquet,
data/*.parquet, meta/stats.json) so the audit does not depend on the loader it is meant to validate.
Nothing is modified: problems are collected as issues and reported.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ERROR = "ERROR"
WARNING = "WARNING"
INFO = "INFO"
# Episode is intact but contains no demonstration; excluded from training, not a data-integrity error.
EXCLUDED = "EXCLUDED"

STATIC_RANGE_DEG = 1.0


@dataclass
class Issue:
    severity: str
    code: str
    message: str

    def to_dict(self) -> dict:
        return {"severity": self.severity, "code": self.code, "message": self.message}


@dataclass
class AuditContext:
    issues: list[Issue] = field(default_factory=list)

    def add(self, severity: str, code: str, message: str) -> None:
        self.issues.append(Issue(severity, code, message))


def read_parquet_dir(root: Path, pattern: str):
    files = sorted(root.glob(pattern))
    if not files:
        return None, []
    tables = [pq.read_table(f) for f in files]
    import pyarrow as pa

    return pa.concat_tables(tables, promote_options="default"), files


def feature_summary(features: dict) -> dict:
    """Split info.json features into cameras / vector features / bookkeeping columns."""
    cameras, vectors, other = {}, {}, {}
    for key, ft in features.items():
        dtype = ft.get("dtype")
        if dtype in ("video", "image"):
            names = ft.get("names") or []
            shape = ft.get("shape")
            hwc = dict(zip(names, shape, strict=False)) if names else {}
            vinfo = ft.get("info") or {}
            cameras[key] = {
                "dtype": dtype,
                "shape": shape,
                "height": hwc.get("height", vinfo.get("video.height")),
                "width": hwc.get("width", vinfo.get("video.width")),
                "channels": hwc.get("channels", vinfo.get("video.channels")),
                "codec": vinfo.get("video.codec"),
                "pix_fmt": vinfo.get("video.pix_fmt"),
                "video_fps": vinfo.get("video.fps"),
                "is_depth_map": vinfo.get("is_depth_map"),
            }
        elif key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            other[key] = {"dtype": dtype, "shape": ft.get("shape")}
        else:
            vectors[key] = {"dtype": dtype, "shape": ft.get("shape"), "names": ft.get("names")}
    return {"cameras": cameras, "vectors": vectors, "bookkeeping": other}


def _stack(column) -> np.ndarray:
    return np.asarray(column.to_pylist(), dtype=np.float64)


def episode_hash(state: np.ndarray, action: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(state, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(action, dtype=np.float32).tobytes())
    return h.hexdigest()


def recompute_stats(x: np.ndarray) -> dict:
    return {
        "min": x.min(axis=0).tolist(),
        "max": x.max(axis=0).tolist(),
        "mean": x.mean(axis=0).tolist(),
        "std": x.std(axis=0).tolist(),
    }


def compare_stats(stored: dict, recomputed: dict, atol: float = 1e-2, rtol: float = 1e-3) -> dict:
    out = {}
    for k in ("min", "max", "mean", "std"):
        if k not in stored:
            out[k] = "missing"
            continue
        a = np.asarray(stored[k], dtype=np.float64).reshape(-1)
        b = np.asarray(recomputed[k], dtype=np.float64).reshape(-1)
        if a.shape != b.shape:
            out[k] = f"shape_mismatch {a.shape} vs {b.shape}"
            continue
        out[k] = "ok" if np.allclose(a, b, atol=atol, rtol=rtol) else f"max_abs_diff={np.abs(a - b).max():.4g}"
    return out


def best_action_state_lag(state: np.ndarray, action: np.ndarray, ep_bounds, max_lag: int = 15) -> dict:
    """Mean |action[t] - state[t+k]| per lag k, within episodes. Smallest error suggests absolute targets."""
    errs = {}
    for k in range(0, max_lag + 1):
        tot, n = 0.0, 0
        for lo, hi in ep_bounds:
            if hi - lo <= k:
                continue
            d = np.abs(action[lo : hi - k, :5] - state[lo + k : hi, :5])  # body joints only (exclude gripper)
            tot += d.sum()
            n += d.size
        errs[k] = tot / max(n, 1)
    best = min(errs, key=errs.get)
    return {"mean_abs_err_by_lag": {str(k): round(v, 4) for k, v in errs.items()}, "best_lag": best}


def audit_snapshot(root: Path, repo_files: dict[str, int] | None = None) -> dict:
    """Audit one dataset snapshot. `repo_files` maps repo path -> size (from the Hub file listing)."""
    ctx = AuditContext()
    res: dict = {"local_path": str(root)}

    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        ctx.add(ERROR, "missing_info", "meta/info.json not found")
        res["issues"] = [i.to_dict() for i in ctx.issues]
        return res
    info = json.loads(info_path.read_text())
    fps = info.get("fps")
    res.update(
        {
            "codebase_version": info.get("codebase_version"),
            "robot_type": info.get("robot_type"),
            "fps": fps,
            "total_episodes_meta": info.get("total_episodes"),
            "total_frames_meta": info.get("total_frames"),
            "total_tasks_meta": info.get("total_tasks"),
            "splits_meta": info.get("splits"),
            "features_raw": info.get("features"),
        }
    )
    fs = feature_summary(info.get("features", {}))
    res["features"] = fs
    if info.get("codebase_version") != "v3.0":
        ctx.add(WARNING, "codebase_version", f"codebase_version={info.get('codebase_version')} (expected v3.0)")
    for vk in ("action", "observation.state"):
        if vk not in fs["vectors"]:
            ctx.add(ERROR, "missing_feature", f"feature '{vk}' missing from info.json")
    for cam, c in fs["cameras"].items():
        if c["video_fps"] is not None and fps is not None and abs(float(c["video_fps"]) - float(fps)) > 1e-6:
            ctx.add(ERROR, "video_fps_mismatch", f"{cam}: video.fps={c['video_fps']} != fps={fps}")

    # tasks
    tasks_tbl, _ = read_parquet_dir(root / "meta", "tasks.parquet")
    tasks = {}
    if tasks_tbl is None:
        ctx.add(ERROR, "missing_tasks", "meta/tasks.parquet not found")
    else:
        for row in tasks_tbl.to_pylist():
            tasks[int(row["task_index"])] = row["task"]
    res["tasks"] = {str(k): v for k, v in sorted(tasks.items())}

    # episodes metadata
    ep_tbl, ep_files = read_parquet_dir(root / "meta" / "episodes", "chunk-*/file-*.parquet")
    if ep_tbl is None:
        ctx.add(ERROR, "missing_episodes_meta", "meta/episodes/*.parquet not found")
        res["issues"] = [i.to_dict() for i in ctx.issues]
        return res
    ep_cols = [c for c in ep_tbl.column_names if not c.startswith("stats/")]
    eps = ep_tbl.select(ep_cols).to_pylist()
    eps.sort(key=lambda r: r["episode_index"])
    ep_idx = [int(e["episode_index"]) for e in eps]
    lengths = [int(e["length"]) for e in eps]
    res["num_episodes"] = len(eps)
    res["episode_lengths"] = {
        "min": int(min(lengths)),
        "max": int(max(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "per_episode": dict(zip(map(str, ep_idx), lengths, strict=True)),
    }
    if ep_idx != list(range(len(eps))):
        ctx.add(ERROR, "episode_index_gap", f"episode indices are not 0..N-1: {ep_idx[:10]}...")
    if info.get("total_episodes") != len(eps):
        ctx.add(ERROR, "total_episodes_mismatch", f"info.total_episodes={info.get('total_episodes')} != {len(eps)}")
    if info.get("total_frames") != sum(lengths):
        ctx.add(ERROR, "total_frames_mismatch", f"info.total_frames={info.get('total_frames')} != sum(lengths)={sum(lengths)}")
    if len(eps) != 50:
        ctx.add(INFO, "episode_count", f"{len(eps)} episodes (most datasets in the collection have 50)")
    short = [i for i, n in zip(ep_idx, lengths, strict=True) if fps and n < 2 * fps]
    if short:
        ctx.add(WARNING, "short_episodes", f"episodes shorter than 2 s: {short}")
    ep_tasks = {int(e["episode_index"]): e.get("tasks") for e in eps}
    multi_task = [i for i, t in ep_tasks.items() if not t or len(t) != 1]
    if multi_task:
        ctx.add(WARNING, "episode_task_count", f"episodes without exactly one task: {multi_task}")

    prev_to = 0
    for e in eps:
        if e["dataset_from_index"] != prev_to:
            ctx.add(ERROR, "index_range_gap", f"episode {e['episode_index']}: from_index {e['dataset_from_index']} != {prev_to}")
        if e["dataset_to_index"] - e["dataset_from_index"] != e["length"]:
            ctx.add(ERROR, "index_range_len", f"episode {e['episode_index']}: to-from != length")
        prev_to = e["dataset_to_index"]

    # video references (v3: several episodes share one mp4, addressed by timestamps)
    video_refs: dict[str, set[str]] = {}
    video_path_tpl = info.get("video_path")
    max_span_err = 0.0
    for cam in fs["cameras"]:
        files = set()
        for e in eps:
            ci = e.get(f"videos/{cam}/chunk_index")
            fi = e.get(f"videos/{cam}/file_index")
            t0 = e.get(f"videos/{cam}/from_timestamp")
            t1 = e.get(f"videos/{cam}/to_timestamp")
            if ci is None or fi is None or t0 is None or t1 is None:
                ctx.add(ERROR, "video_ref_missing", f"episode {e['episode_index']}: no video reference for {cam}")
                continue
            files.add(video_path_tpl.format(video_key=cam, chunk_index=ci, file_index=fi))
            span_err = abs((t1 - t0) - e["length"] / fps)
            max_span_err = max(max_span_err, span_err)
            if span_err > 1.5 / fps:
                ctx.add(ERROR, "video_span_mismatch", f"episode {e['episode_index']} {cam}: video span {t1 - t0:.3f}s vs {e['length'] / fps:.3f}s")
        video_refs[cam] = files
    res["video_files_referenced"] = {k: sorted(v) for k, v in video_refs.items()}
    res["max_video_span_error_s"] = max_span_err
    if repo_files is not None:
        for cam, files in video_refs.items():
            for f in files:
                if f not in repo_files:
                    ctx.add(ERROR, "video_file_missing", f"{f} referenced by episodes but not in repo")
                elif repo_files[f] < 1024:
                    ctx.add(ERROR, "video_file_tiny", f"{f} is only {repo_files[f]} bytes (corrupted / LFS pointer?)")

    # frame data
    data_tbl, data_files = read_parquet_dir(root / "data", "chunk-*/file-*.parquet")
    if data_tbl is None:
        ctx.add(ERROR, "missing_data", "data/*.parquet not found")
        res["issues"] = [i.to_dict() for i in ctx.issues]
        return res
    res["num_frames"] = data_tbl.num_rows
    res["data_columns"] = {f.name: str(f.type) for f in data_tbl.schema}
    if data_tbl.num_rows != sum(lengths):
        ctx.add(ERROR, "frame_count_mismatch", f"data rows {data_tbl.num_rows} != sum(episode lengths) {sum(lengths)}")

    order = np.argsort(np.asarray(data_tbl.column("index").to_pylist()), kind="stable")
    data_tbl = data_tbl.take(order)
    index = np.asarray(data_tbl.column("index").to_pylist())
    ep_col = np.asarray(data_tbl.column("episode_index").to_pylist())
    fi_col = np.asarray(data_tbl.column("frame_index").to_pylist())
    ts_col = np.asarray(data_tbl.column("timestamp").to_pylist(), dtype=np.float64)
    ti_col = np.asarray(data_tbl.column("task_index").to_pylist())
    if not np.array_equal(index, np.arange(len(index))):
        ctx.add(ERROR, "global_index", "global 'index' column is not 0..N-1")

    vec = {}
    for key in ("observation.state", "action"):
        if key not in data_tbl.column_names:
            continue
        arr = _stack(data_tbl.column(key))
        vec[key] = arr
        nan = int(np.isnan(arr).sum())
        inf = int(np.isinf(arr).sum())
        res.setdefault("missing_values", {})[key] = {"nan": nan, "inf": inf}
        if nan or inf:
            ctx.add(ERROR, "nan_inf", f"{key}: {nan} NaN, {inf} inf values")
        declared = fs["vectors"].get(key, {}).get("shape")
        if declared and arr.shape[1] != declared[0]:
            ctx.add(ERROR, "shape_mismatch", f"{key}: data dim {arr.shape[1]} != declared {declared}")
    for col in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        if col in data_tbl.column_names:
            n_null = data_tbl.column(col).null_count
            if n_null:
                ctx.add(ERROR, "null_values", f"{col}: {n_null} null values")

    unknown_tasks = sorted(set(ti_col.tolist()) - set(tasks))
    if unknown_tasks:
        ctx.add(ERROR, "unknown_task_index", f"task_index values not in tasks.parquet: {unknown_tasks}")
    res["task_index_values"] = sorted(set(int(t) for t in ti_col.tolist()))

    ep_bounds = []
    max_ts_err = 0.0
    hashes = {}
    for e in eps:
        lo, hi = e["dataset_from_index"], e["dataset_to_index"]
        ep_bounds.append((lo, hi))
        if not np.all(ep_col[lo:hi] == e["episode_index"]):
            ctx.add(ERROR, "episode_rows", f"episode {e['episode_index']}: rows {lo}:{hi} contain other episode_index values")
            continue
        if not np.array_equal(fi_col[lo:hi], np.arange(hi - lo)):
            ctx.add(ERROR, "frame_index", f"episode {e['episode_index']}: frame_index not 0..L-1")
        err = np.abs(ts_col[lo:hi] - fi_col[lo:hi] / fps).max()
        max_ts_err = max(max_ts_err, float(err))
        if err > 0.5 / fps:
            ctx.add(ERROR, "timestamp_drift", f"episode {e['episode_index']}: timestamp deviates {err:.4f}s from frame_index/fps")
        if len(set(ti_col[lo:hi].tolist())) != 1:
            ctx.add(WARNING, "task_changes_within_episode", f"episode {e['episode_index']} has several task_index values")
        if "observation.state" in vec and "action" in vec:
            hashes[int(e["episode_index"])] = episode_hash(vec["observation.state"][lo:hi], vec["action"][lo:hi])
    res["max_timestamp_error_s"] = max_ts_err

    dup_groups: dict[str, list[int]] = {}
    for i, h in hashes.items():
        dup_groups.setdefault(h, []).append(i)
    dups = [g for g in dup_groups.values() if len(g) > 1]
    res["duplicate_episode_groups"] = dups
    res["episode_hashes"] = {str(k): v for k, v in hashes.items()}
    if dups:
        ctx.add(WARNING, "duplicate_episodes", f"exact duplicate (state, action) episodes: {dups}")

    # idle frames at episode start/end (robot not moving): informational for trimming decisions
    if "action" in vec:
        a = vec["action"]
        head, tail = [], []
        for lo, hi in ep_bounds:
            d = np.abs(np.diff(a[lo:hi], axis=0)).max(axis=1) if hi - lo > 1 else np.zeros(0)
            moving = np.nonzero(d > 0.5)[0]
            if len(moving) == 0:
                head.append(hi - lo)
                tail.append(hi - lo)
            else:
                head.append(int(moving[0]))
                tail.append(int(len(d) - 1 - moving[-1]))
        res["idle_frames"] = {
            "head_mean": float(np.mean(head)),
            "head_max": int(np.max(head)),
            "tail_mean": float(np.mean(tail)),
            "tail_max": int(np.max(tail)),
        }
        # An episode whose action never spans more than STATIC_RANGE_DEG on any joint is a recording
        # with no demonstration in it (aborted / idle). Such episodes are excluded from the split.
        static = []
        for i, (lo, hi) in enumerate(ep_bounds):
            if float(np.ptp(a[lo:hi], axis=0).max()) < STATIC_RANGE_DEG:
                static.append(ep_idx[i])
        res["static_episodes"] = static
        if static:
            ctx.add(
                EXCLUDED,
                "static_episode",
                f"episodes with no motion (action range < {STATIC_RANGE_DEG} deg on every joint): {static}",
            )
        if "observation.state" in vec:
            first = np.stack([vec["observation.state"][lo] for lo, _ in ep_bounds])
            res["start_pose"] = {"median": np.median(first, 0).round(2).tolist(), "std": first.std(0).round(2).tolist()}

    # joint ranges & action convention
    if "observation.state" in vec and "action" in vec:
        res["joint_ranges"] = {
            key: {"min": vec[key].min(axis=0).round(3).tolist(), "max": vec[key].max(axis=0).round(3).tolist()}
            for key in vec
        }
        res["action_state_lag"] = best_action_state_lag(vec["observation.state"], vec["action"], ep_bounds)
        res["mean_abs_action_minus_state"] = np.abs(vec["action"] - vec["observation.state"]).mean(axis=0).round(4).tolist()

    # stored stats vs recomputed
    stats_path = root / "meta" / "stats.json"
    if not stats_path.exists():
        res["stats"] = None
        ctx.add(WARNING, "missing_stats", "meta/stats.json not found (LeRobot will need to compute it)")
    else:
        stats = json.loads(stats_path.read_text())
        res["stats"] = {
            "keys": sorted(stats),
            "stat_types": sorted({s for v in stats.values() for s in v}),
            "action": {k: stats.get("action", {}).get(k) for k in ("min", "max", "mean", "std", "q01", "q99")},
            "observation.state": {
                k: stats.get("observation.state", {}).get(k) for k in ("min", "max", "mean", "std", "q01", "q99")
            },
        }
        cmp = {}
        for key, arr in vec.items():
            if key in stats:
                cmp[key] = compare_stats(stats[key], recompute_stats(arr))
                bad = {k: v for k, v in cmp[key].items() if v != "ok"}
                if bad:
                    ctx.add(WARNING, "stats_mismatch", f"meta/stats.json[{key}] differs from data: {bad}")
            else:
                ctx.add(WARNING, "stats_key_missing", f"meta/stats.json has no entry for {key}")
        res["stats_check"] = cmp
        for cam in fs["cameras"]:
            if cam not in stats:
                ctx.add(WARNING, "stats_key_missing", f"meta/stats.json has no entry for {cam}")

    res["files"] = {
        "data": [str(f.relative_to(root)) for f in data_files],
        "episodes_meta": [str(f.relative_to(root)) for f in ep_files],
    }
    res["issues"] = [i.to_dict() for i in ctx.issues]
    return res


def cross_dataset_checks(results: dict[str, dict], expected_joints: list[str], expected_robot: str) -> dict:
    """Compare conventions across datasets."""
    issues: list[dict] = []
    ok = {k: v for k, v in results.items() if "features" in v}

    def collect(fn):
        out = {}
        for rid, r in ok.items():
            try:
                out[rid] = fn(r)
            except Exception as exc:  # noqa: BLE001 - report, do not crash the audit
                out[rid] = f"<error: {exc}>"
        return out

    def unique(values: dict) -> list:
        seen = []
        for v in values.values():
            if v not in seen:
                seen.append(v)
        return seen

    checks = {
        "codebase_version": collect(lambda r: r["codebase_version"]),
        "robot_type": collect(lambda r: r["robot_type"]),
        "fps": collect(lambda r: r["fps"]),
        "action_names": collect(lambda r: r["features"]["vectors"]["action"]["names"]),
        "action_shape": collect(lambda r: r["features"]["vectors"]["action"]["shape"]),
        "action_dtype": collect(lambda r: r["features"]["vectors"]["action"]["dtype"]),
        "state_names": collect(lambda r: r["features"]["vectors"]["observation.state"]["names"]),
        "state_shape": collect(lambda r: r["features"]["vectors"]["observation.state"]["shape"]),
        "state_dtype": collect(lambda r: r["features"]["vectors"]["observation.state"]["dtype"]),
        "camera_keys": collect(lambda r: sorted(r["features"]["cameras"])),
        "camera_resolutions": collect(
            lambda r: {k: [c["height"], c["width"]] for k, c in sorted(r["features"]["cameras"].items())}
        ),
        "video_codecs": collect(lambda r: {k: c["codec"] for k, c in sorted(r["features"]["cameras"].items())}),
        "extra_vector_features": collect(
            lambda r: sorted(set(r["features"]["vectors"]) - {"action", "observation.state"})
        ),
    }
    summary = {}
    for name, values in checks.items():
        u = unique(values)
        summary[name] = {"consistent": len(u) == 1, "unique_values": u}
        if len(u) != 1:
            issues.append({"severity": ERROR, "code": f"inconsistent_{name}", "message": f"{name} differs across datasets: {values}"})

    for rid, names in checks["action_names"].items():
        if names != expected_joints:
            issues.append({"severity": ERROR, "code": "joint_names", "message": f"{rid}: action names {names} != LeRobot so_follower {expected_joints}"})
    for rid, names in checks["state_names"].items():
        if names != expected_joints:
            issues.append({"severity": ERROR, "code": "joint_names", "message": f"{rid}: state names {names} != LeRobot so_follower {expected_joints}"})
    for rid, rt in checks["robot_type"].items():
        if rt != expected_robot:
            issues.append({"severity": WARNING, "code": "robot_type", "message": f"{rid}: robot_type={rt} (expected {expected_robot})"})

    # Joint-range consistency: large differences between datasets can indicate re-calibration.
    ranges = {rid: r.get("joint_ranges", {}).get("observation.state") for rid, r in ok.items()}
    ranges = {k: v for k, v in ranges.items() if v}
    if ranges:
        mins = np.array([v["min"] for v in ranges.values()])
        maxs = np.array([v["max"] for v in ranges.values()])
        summary["state_range_envelope"] = {
            "joint_names": expected_joints,
            "min_of_mins": mins.min(axis=0).round(2).tolist(),
            "max_of_mins": mins.max(axis=0).round(2).tolist(),
            "min_of_maxs": maxs.min(axis=0).round(2).tolist(),
            "max_of_maxs": maxs.max(axis=0).round(2).tolist(),
        }

    # Cross-dataset exact-duplicate episodes
    seen: dict[str, tuple[str, str]] = {}
    cross_dups = []
    for rid, r in ok.items():
        for ep, h in r.get("episode_hashes", {}).items():
            if h in seen and seen[h][0] != rid:
                cross_dups.append({"a": seen[h], "b": (rid, ep)})
            else:
                seen.setdefault(h, (rid, ep))
    summary["cross_dataset_duplicate_episodes"] = cross_dups
    if cross_dups:
        issues.append({"severity": WARNING, "code": "cross_dataset_duplicates", "message": f"{len(cross_dups)} episodes appear in more than one dataset"})

    tasks = {rid: list(r.get("tasks", {}).values()) for rid, r in ok.items()}
    flat = [t for ts in tasks.values() for t in ts]
    dup_instr = sorted({t for t in flat if flat.count(t) > 1})
    summary["task_strings"] = tasks
    if dup_instr:
        issues.append({"severity": ERROR, "code": "shared_instruction", "message": f"the same instruction is used by several datasets: {dup_instr}"})
    return {"summary": summary, "issues": issues}
