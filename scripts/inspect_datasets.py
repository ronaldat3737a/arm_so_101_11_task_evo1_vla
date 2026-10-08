#!/usr/bin/env python
"""Inspect all source datasets on the Hugging Face Hub and write an audit report.

Outputs:
    reports/dataset_audit.json
    reports/dataset_audit.md

Downloads meta/ + data/ (small) for every dataset at a pinned revision. Videos are large
(~600 MB per dataset); they are only downloaded and decoded with --probe-videos.
Nothing is modified: problems are written to the report.

Usage:
    python scripts/inspect_datasets.py
    python scripts/inspect_datasets.py --probe-videos          # also download + decode-check videos
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.audit import ERROR, EXCLUDED, WARNING, audit_snapshot, cross_dataset_checks  # noqa: E402


def probe_videos(root: Path, result: dict, fps: float) -> dict:
    """Open each referenced mp4 with PyAV and decode the first/last frame of every episode."""
    try:
        import av  # provided by lerobot[dataset]
    except ImportError:
        return {"status": "NOT_CHECKED", "reason": "PyAV (av) is not installed in this environment"}
    import pyarrow.parquet as pq

    out = {"status": "CHECKED", "files": {}, "errors": []}
    for cam, files in result.get("video_files_referenced", {}).items():
        for rel in files:
            path = root / rel
            entry = {}
            try:
                with av.open(str(path)) as c:
                    s = c.streams.video[0]
                    entry = {
                        "codec": s.codec_context.name,
                        "width": s.codec_context.width,
                        "height": s.codec_context.height,
                        "avg_rate": float(s.average_rate) if s.average_rate else None,
                        "frames_header": s.frames,
                        "duration_s": float(c.duration / 1e6) if c.duration else None,
                    }
                    cfg = result["features"]["cameras"][cam]
                    if (entry["width"], entry["height"]) != (cfg["width"], cfg["height"]):
                        out["errors"].append(f"{rel}: decoded size {entry['width']}x{entry['height']} != info.json {cfg['width']}x{cfg['height']}")
            except Exception as exc:  # noqa: BLE001
                out["errors"].append(f"{rel}: cannot open ({exc})")
            out["files"][rel] = entry

    # per-episode boundary decode
    ep_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    eps = [r for f in ep_files for r in pq.read_table(f).to_pylist()]
    tpl = result["features_raw"] and json.loads((root / "meta" / "info.json").read_text())["video_path"]
    n_checked = 0
    for cam in result.get("video_files_referenced", {}):
        for e in eps:
            rel = tpl.format(video_key=cam, chunk_index=e[f"videos/{cam}/chunk_index"], file_index=e[f"videos/{cam}/file_index"])
            t0 = e[f"videos/{cam}/from_timestamp"]
            t1 = e[f"videos/{cam}/to_timestamp"] - 1.0 / fps
            for t in (t0, t1):
                try:
                    with av.open(str(root / rel)) as c:
                        s = c.streams.video[0]
                        c.seek(int(max(t - 0.5, 0) / s.time_base), stream=s, any_frame=False, backward=True)
                        got = None
                        for frame in c.decode(s):
                            if frame.time is not None and frame.time >= t - 0.5 / fps:
                                got = frame.time
                                break
                        if got is None or abs(got - t) > 1.0 / fps:
                            out["errors"].append(f"episode {e['episode_index']} {cam}: frame at t={t:.3f}s not decodable (got {got})")
                except Exception as exc:  # noqa: BLE001
                    out["errors"].append(f"episode {e['episode_index']} {cam}: decode error at t={t:.3f}s ({exc})")
                n_checked += 1
    out["boundary_frames_checked"] = n_checked
    return out


def to_markdown(audit: dict) -> str:
    L = []
    L.append("# Dataset audit\n")
    L.append(f"Generated: {audit['generated_at']}  ")
    L.append(f"Collection: `{audit['collection']['slug']}` ({audit['collection']['n_items']} items)  ")
    L.append(f"Video probe: {'yes' if audit['video_probe_requested'] else 'no (run with --probe-videos)'}\n")
    v = audit["verdict"]
    L.append(f"## Verdict: **{v['status']}**\n")
    L.append(f"- errors: {v['n_errors']}, warnings: {v['n_warnings']}, excluded no-op episodes: {v.get('n_excluded_episodes', 0)}")
    for m in v["notes"]:
        L.append(f"- {m}")
    L.append("")

    L.append("## Per-dataset summary\n")
    L.append("| # | repo | revision | version | episodes | static (unusable) | frames | fps | ep len min/median/max | task string | errors | warnings |")
    L.append("|---|------|----------|---------|---------:|-------------------|-------:|----:|----------------------|-------------|------:|--------:|")
    for i, (rid, r) in enumerate(audit["datasets"].items(), 1):
        if "num_episodes" not in r:
            L.append(f"| {i} | `{rid}` | - | - | - | - | - | - | - | - | FAILED | |")
            continue
        el = r["episode_lengths"]
        ne = sum(1 for x in r["issues"] if x["severity"] == ERROR)  # EXCLUDED is listed separately
        nw = sum(1 for x in r["issues"] if x["severity"] == WARNING)
        tasks = "; ".join(r["tasks"].values())
        L.append(
            f"| {i} | `{rid.split('/')[-1]}` | `{r['revision'][:8]}` | {r['codebase_version']} | {r['num_episodes']} | "
            f"{r.get('static_episodes') or '-'} | {r['num_frames']} | {r['fps']} | {el['min']}/{el['median']:.0f}/{el['max']} | {tasks} | {ne} | {nw} |"
        )
    L.append("")

    s = audit["cross_dataset"]["summary"]
    L.append("## Cross-dataset conventions\n")
    L.append("| check | consistent | value(s) |")
    L.append("|-------|:----------:|----------|")
    for name, c in s.items():
        if not isinstance(c, dict) or "consistent" not in c:
            continue
        vals = c["unique_values"]
        L.append(f"| {name} | {'yes' if c['consistent'] else '**NO**'} | `{json.dumps(vals[0] if len(vals) == 1 else vals)}` |")
    L.append("")
    if "state_range_envelope" in s:
        e = s["state_range_envelope"]
        L.append("### observation.state range across datasets (degrees; gripper 0-100)\n")
        L.append("| joint | min of mins | max of mins | min of maxs | max of maxs |")
        L.append("|-------|-----------:|-----------:|-----------:|-----------:|")
        for j, n in enumerate(e["joint_names"]):
            L.append(f"| {n} | {e['min_of_mins'][j]} | {e['max_of_mins'][j]} | {e['min_of_maxs'][j]} | {e['max_of_maxs'][j]} |")
        L.append("")

    L.append("### Episode start pose (median of first frame), calibration consistency evidence\n")
    L.append("Same physical rest pose should read the same values if all datasets share one calibration.\n")
    L.append("| repo | start pose median | std across episodes |")
    L.append("|------|-------------------|---------------------|")
    for rid, r in audit["datasets"].items():
        if "start_pose" in r:
            L.append(f"| `{rid.split('/')[-1]}` | {r['start_pose']['median']} | {r['start_pose']['std']} |")
    L.append("")

    first = next((r for r in audit["datasets"].values() if "features" in r), None)
    if first:
        L.append("## Feature schema (first dataset)\n")
        L.append("| key | dtype | shape | details |")
        L.append("|-----|-------|-------|---------|")
        for k, c in first["features"]["cameras"].items():
            L.append(f"| {k} | {c['dtype']} | {c['shape']} | {c['height']}x{c['width']} {c['codec']} {c['pix_fmt']} @ {c['video_fps']} fps |")
        for k, c in first["features"]["vectors"].items():
            L.append(f"| {k} | {c['dtype']} | {c['shape']} | {c['names']} |")
        for k, c in first["features"]["bookkeeping"].items():
            L.append(f"| {k} | {c['dtype']} | {c['shape']} | |")
        L.append("")

    L.append("## Action convention evidence\n")
    L.append("Mean |action[t] − state[t+k]| over body joints (degrees). A minimum at small k > 0 means `action` is an")
    L.append("absolute joint-position target that the follower reaches a few frames later (LeRobot teleop convention).\n")
    L.append("| repo | best lag k | err(k=0) | err(best) | mean abs(action−state) per joint |")
    L.append("|------|-----------:|---------:|----------:|----------------------------------|")
    for rid, r in audit["datasets"].items():
        if "action_state_lag" not in r:
            continue
        lag = r["action_state_lag"]
        L.append(
            f"| `{rid.split('/')[-1]}` | {lag['best_lag']} | {lag['mean_abs_err_by_lag']['0']} | "
            f"{lag['mean_abs_err_by_lag'][str(lag['best_lag'])]} | {r['mean_abs_action_minus_state']} |"
        )
    L.append("")

    L.append("## Data quality\n")
    L.append("| repo | NaN/inf | max ts err (s) | max video span err (s) | idle head mean/max | idle tail mean/max | dup episodes | stats.json vs data |")
    L.append("|------|--------:|---------------:|-----------------------:|-------------------:|-------------------:|-------------:|--------------------|")
    for rid, r in audit["datasets"].items():
        if "missing_values" not in r:
            continue
        mv = sum(v["nan"] + v["inf"] for v in r["missing_values"].values())
        idle = r.get("idle_frames", {})
        sc = r.get("stats_check", {})
        sc_txt = "ok" if sc and all(v == "ok" for d in sc.values() for v in d.values()) else json.dumps(sc)
        L.append(
            f"| `{rid.split('/')[-1]}` | {mv} | {r['max_timestamp_error_s']:.2e} | {r['max_video_span_error_s']:.2e} | "
            f"{idle.get('head_mean', 0):.0f}/{idle.get('head_max', 0)} | {idle.get('tail_mean', 0):.0f}/{idle.get('tail_max', 0)} | "
            f"{r['duplicate_episode_groups'] or '-'} | {sc_txt} |"
        )
    L.append("")

    L.append("## Video integrity\n")
    for rid, r in audit["datasets"].items():
        vp = r.get("video_probe")
        if vp is None:
            continue
        if vp["status"] != "CHECKED":
            L.append(f"- `{rid}`: {vp['status']} — {vp.get('reason', '')}")
        else:
            L.append(f"- `{rid}`: {len(vp['files'])} files opened, {vp['boundary_frames_checked']} episode-boundary frames decoded, {len(vp['errors'])} errors")
            for err in vp["errors"][:10]:
                L.append(f"  - {err}")
    if not any("video_probe" in r for r in audit["datasets"].values()):
        L.append("- NOT CHECKED. Only Hub file presence/size and episode timestamp spans were verified. Run `--probe-videos`.")
    L.append("")

    L.append("## Issues\n")
    any_issue = False
    for x in audit["cross_dataset"]["issues"]:
        any_issue = True
        L.append(f"- **{x['severity']}** [cross-dataset] `{x['code']}`: {x['message']}")
    for rid, r in audit["datasets"].items():
        for x in r.get("issues", []):
            any_issue = True
            L.append(f"- **{x['severity']}** [`{rid.split('/')[-1]}`] `{x['code']}`: {x['message']}")
    if not any_issue:
        L.append("- none")
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "dataset.yaml"))
    ap.add_argument("--out-dir", default=str(ROOT / "reports"))
    ap.add_argument("--probe-videos", action="store_true", help="download all videos and decode-check them")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cache_dir = ROOT / cfg["cache_dir"]
    api = HfApi()

    audit: dict = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "video_probe_requested": args.probe_videos,
        "datasets": {},
    }
    try:
        coll = api.get_collection(cfg["collection"])
        coll_ids = [it.item_id for it in coll.items if it.item_type == "dataset"]
        audit["collection"] = {"slug": coll.slug, "title": coll.title, "n_items": len(coll_ids), "items": coll_ids}
        missing = sorted(set(cfg["repos"]) - set(coll_ids))
        extra = sorted(set(coll_ids) - set(cfg["repos"]))
        audit["collection"]["configured_not_in_collection"] = missing
        audit["collection"]["in_collection_not_configured"] = extra
    except Exception as exc:  # noqa: BLE001
        audit["collection"] = {"slug": cfg["collection"], "n_items": None, "error": str(exc)}

    for rid in cfg["repos"]:
        print(f"[inspect] {rid}", flush=True)
        entry: dict = {}
        try:
            info = api.dataset_info(rid, files_metadata=True)
            entry["revision"] = info.sha
            entry["last_modified"] = info.last_modified.isoformat() if info.last_modified else None
            entry["hub_tags"] = info.tags
            repo_files = {s.rfilename: (s.size or 0) for s in info.siblings}
            entry["repo_files"] = {
                k: {"size": v, "lfs_sha256": (s.lfs.sha256 if s.lfs else None)}
                for s in info.siblings
                for k, v in [(s.rfilename, s.size)]
            }
            patterns = ["meta/**", "data/**", "README.md"] + (["videos/**"] if args.probe_videos else [])
            local = Path(
                snapshot_download(rid, repo_type="dataset", revision=info.sha, allow_patterns=patterns, cache_dir=cache_dir)
            )
            entry.update(audit_snapshot(local, repo_files))
            if args.probe_videos and "features" in entry:
                entry["video_probe"] = probe_videos(local, entry, entry["fps"])
                for err in entry["video_probe"].get("errors", []):
                    entry["issues"].append({"severity": ERROR, "code": "video_decode", "message": err})
        except Exception as exc:  # noqa: BLE001
            entry.setdefault("issues", []).append({"severity": ERROR, "code": "inspect_failed", "message": repr(exc)})
        audit["datasets"][rid] = entry

    exp = cfg["expected"]
    audit["cross_dataset"] = cross_dataset_checks(audit["datasets"], exp["joint_names"], exp["robot_type"])

    n_err = sum(1 for r in audit["datasets"].values() for x in r.get("issues", []) if x["severity"] == ERROR)
    n_err += sum(1 for x in audit["cross_dataset"]["issues"] if x["severity"] == ERROR)
    n_warn = sum(1 for r in audit["datasets"].values() for x in r.get("issues", []) if x["severity"] == WARNING)
    n_warn += sum(1 for x in audit["cross_dataset"]["issues"] if x["severity"] == WARNING)
    notes = []
    if audit["collection"].get("configured_not_in_collection"):
        notes.append(f"configured repos missing from collection: {audit['collection']['configured_not_in_collection']}")
    if audit["collection"].get("in_collection_not_configured"):
        notes.append(f"collection repos not configured: {audit['collection']['in_collection_not_configured']}")
    if not args.probe_videos:
        notes.append("video decode integrity NOT checked (metadata-only audit)")
    excluded = {rid: r.get("static_episodes", []) for rid, r in audit["datasets"].items() if r.get("static_episodes")}
    n_excl = sum(len(v) for v in excluded.values())
    integrity_ok = n_err == 0
    if not integrity_ok:
        status = "FAIL"
    elif n_excl:
        status = "PASS_WITH_EXCLUDED_EPISODES"
    else:
        status = "PASS_WITH_WARNINGS" if n_warn else "PASS"
    notes.insert(0, f"dataset integrity = {'PASS' if integrity_ok else 'FAIL'}")
    notes.insert(1, f"dataset usable for training = {'PASS' if integrity_ok else 'FAIL'}")
    if n_excl:
        notes.insert(2, f"{n_excl} no-op episodes (no motion, not corrupted) excluded from training: {excluded}")
    audit["verdict"] = {
        "status": status,
        "dataset_integrity": "PASS" if integrity_ok else "FAIL",
        "usable_for_training": "PASS" if integrity_ok else "FAIL",
        "n_excluded_episodes": n_excl,
        "n_errors": n_err,
        "n_warnings": n_warn,
        "notes": notes,
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    (out / "dataset_audit.md").write_text(to_markdown(audit))
    print(f"[inspect] verdict={audit['verdict']['status']} errors={n_err} warnings={n_warn} excluded={n_excl}")
    print(f"[inspect] wrote {out / 'dataset_audit.json'} and {out / 'dataset_audit.md'}")
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
