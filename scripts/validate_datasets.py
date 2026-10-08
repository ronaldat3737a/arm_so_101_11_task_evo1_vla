#!/usr/bin/env python
"""Validate the audited datasets with the official LeRobot loader and freeze the episode split.

Requires reports/dataset_audit.json (scripts/inspect_datasets.py) and the dataset videos in the local cache
(`python scripts/inspect_datasets.py --probe-videos`).

Steps:
  1. audit gate: the only tolerated audit errors are the documented static episodes (excluded from the split)
  2. Hub revision unchanged since the audit
  3. configs/tasks.yaml consistent with the audit (instructions, exclusions)
  4. episode-level split per task -> reports/split_manifest.json
  5. official `lerobot.datasets.lerobot_dataset.LeRobotDataset` loads every dataset at the pinned revision; frames
     (incl. decoded video) are sampled from train/val/test episodes and checked for keys, shapes, dtype, NaN, task string
  6. reports/dataset_manifest.json + reports/validation.md

Exit code 0 only if everything passes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.split import check_disjoint, episode_offsets, split_task_episodes  # noqa: E402

TOLERATED_AUDIT_ERRORS = {"static_episode"}


def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-config", default=str(ROOT / "configs" / "dataset.yaml"))
    ap.add_argument("--tasks-config", default=str(ROOT / "configs" / "tasks.yaml"))
    ap.add_argument("--frames-per-episode", type=int, default=3, help="frames decoded per sampled episode (first/middle/last)")
    ap.add_argument("--episodes-per-split", type=int, default=2, help="episodes sampled per split per task for decoding")
    ap.add_argument("--offline", action="store_true", help="skip the Hub revision check")
    args = ap.parse_args()

    reports = ROOT / "reports"
    dcfg = load_yaml(Path(args.dataset_config))
    tasks = load_yaml(Path(args.tasks_config))["tasks"]
    audit_path = reports / "dataset_audit.json"
    checks: list[dict] = []
    failed = False

    def check(name: str, ok: bool, detail: str = "") -> bool:
        nonlocal failed
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{' — ' + detail if detail else ''}", flush=True)
        failed |= not ok
        return ok

    if not check("audit report exists", audit_path.exists(), str(audit_path)):
        return finish(reports, checks, None, None, failed)
    audit = json.loads(audit_path.read_text())

    # 1. audit gate
    unexpected = [
        (rid, i) for rid, r in audit["datasets"].items() for i in r.get("issues", [])
        if i["severity"] == "ERROR" and i["code"] not in TOLERATED_AUDIT_ERRORS
    ] + [("cross", i) for i in audit["cross_dataset"]["issues"] if i["severity"] == "ERROR"]
    check("audit has no errors besides documented static episodes", not unexpected, json.dumps(unexpected)[:500])
    conv = audit["cross_dataset"]["summary"]
    for key in ("codebase_version", "robot_type", "fps", "action_names", "action_shape", "state_names", "state_shape", "camera_keys", "camera_resolutions"):
        check(f"cross-dataset consistent: {key}", conv[key]["consistent"], json.dumps(conv[key]["unique_values"])[:200])

    # 2. revisions unchanged
    if not args.offline:
        from huggingface_hub import HfApi

        api = HfApi()
        for rid in dcfg["repos"]:
            sha = api.dataset_info(rid).sha
            check(f"revision unchanged: {rid}", sha == audit["datasets"][rid]["revision"], f"audit {audit['datasets'][rid]['revision'][:8]} hub {sha[:8]}")

    # 3. tasks.yaml vs audit
    check("tasks.yaml lists the configured repos in order", [t["dataset"] for t in tasks] == dcfg["repos"])
    for t in tasks:
        a = audit["datasets"][t["dataset"]]
        check(f"{t['task_id']} instruction == dataset task string", list(a["tasks"].values()) == [t["instruction"]])
        check(f"{t['task_id']} excluded == audit static episodes", sorted(t["excluded_episodes"]) == sorted(a["static_episodes"]))

    # 4. split
    sp = dcfg["split"]
    n_eps = [audit["datasets"][t["dataset"]]["num_episodes"] for t in tasks]
    offsets = episode_offsets(n_eps)
    split_manifest = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "policy": "episode-level split per task after removing static episodes; seeded permutation; "
                  "test/val taken first, train = remaining (capped). Real-data split is held-out EPISODES, not held-out "
                  "object positions: positions in the real recordings are not annotated.",
        "seed": sp["seed"],
        "requested": {"train": sp["train"], "val": sp["val"], "test": sp["test"]},
        "aggregated_dataset_order": dcfg["repos"],
        "tasks": {},
        "global": {"train": [], "val": [], "test": [], "excluded": []},
    }
    for i, t in enumerate(tasks):
        s = split_task_episodes(n_eps[i], t["excluded_episodes"], sp["val"], sp["test"], sp["train"], sp["seed"] + i)
        check_disjoint(s)
        off = offsets[i]
        split_manifest["tasks"][t["task_id"]] = {
            "dataset": t["dataset"], "revision": audit["datasets"][t["dataset"]]["revision"],
            "num_episodes": n_eps[i], "global_episode_offset": off,
            **{k: v for k, v in s.items()},
            "counts": {k: len(v) for k, v in s.items()},
        }
        for k in ("train", "val", "test", "excluded"):
            split_manifest["global"][k] += [off + e for e in s[k]]
        check(f"{t['task_id']} split sizes", len(s["test"]) == sp["test"] and len(s["val"]) == sp["val"] and len(s["train"]) >= 30,
              f"train {len(s['train'])} val {len(s['val'])} test {len(s['test'])} excluded {len(s['excluded'])}")
    g = split_manifest["global"]
    check("global split disjoint", not (set(g["train"]) & set(g["test"]) or set(g["train"]) & set(g["val"]) or set(g["val"]) & set(g["test"])))
    (reports / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2))

    # 5. official loader + frame decode
    try:
        import torch
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        check("lerobot importable", False, f"{exc}; run inside the project env (uv run ...)")
        return finish(reports, checks, split_manifest, None, failed)


    dataset_entries = []
    for i, t in enumerate(tasks):
        rid = t["dataset"]
        a = audit["datasets"][rid]
        # The audit downloaded meta/ + data/ (+ videos/ with --probe-videos) into this pinned snapshot directory.
        local = Path(a["local_path"])
        if not check(f"{t['task_id']} local snapshot at audited revision", local.name == a["revision"] and (local / "meta" / "info.json").exists(), str(local)):
            continue
        missing_videos = [f for fs in a["video_files_referenced"].values() for f in fs if not (local / f).exists()]
        if not check(f"{t['task_id']} videos present locally", not missing_videos, f"{len(missing_videos)} missing"):
            continue
        t0 = time.time()
        try:
            ds = LeRobotDataset(rid, root=local, revision=a["revision"], download_videos=False)
        except Exception as exc:  # noqa: BLE001
            check(f"{t['task_id']} LeRobotDataset loads", False, repr(exc)[:300])
            continue
        check(f"{t['task_id']} LeRobotDataset loads", True, f"{ds.num_episodes} episodes, {ds.num_frames} frames, {time.time() - t0:.1f}s")
        check(f"{t['task_id']} loader counts == audit", ds.num_episodes == a["num_episodes"] and ds.num_frames == a["num_frames"])
        cams = sorted(ds.meta.camera_keys)
        s = split_manifest["tasks"][t["task_id"]]
        bad = []
        n_frames = 0
        dt_decode = 0.0
        for split_name in ("train", "val", "test"):
            for ep in s[split_name][: args.episodes_per_split]:
                lo = int(ds.meta.episodes["dataset_from_index"][ep])
                hi = int(ds.meta.episodes["dataset_to_index"][ep])
                picks = sorted({lo, (lo + hi) // 2, hi - 1})[: args.frames_per_episode]
                for idx in picks:
                    t1 = time.time()
                    try:
                        item = ds[idx]
                    except Exception as exc:  # noqa: BLE001
                        bad.append(f"ep{ep} idx{idx}: {exc!r}"[:200])
                        continue
                    dt_decode += time.time() - t1
                    n_frames += 1
                    if item["task"] != t["instruction"]:
                        bad.append(f"ep{ep} idx{idx}: task {item['task']!r}")
                    if int(item["episode_index"]) != ep:
                        bad.append(f"ep{ep} idx{idx}: episode_index {int(item['episode_index'])}")
                    for key in ("observation.state", "action"):
                        v = item[key]
                        if tuple(v.shape) != (6,) or not torch.isfinite(v).all():
                            bad.append(f"ep{ep} idx{idx}: {key} shape {tuple(v.shape)} finite={bool(torch.isfinite(v).all())}")
                    for cam in cams:
                        img = item[cam]
                        h, w = a["features"]["cameras"][cam]["height"], a["features"]["cameras"][cam]["width"]
                        if tuple(img.shape) != (3, h, w):
                            bad.append(f"ep{ep} idx{idx}: {cam} shape {tuple(img.shape)} != (3,{h},{w})")
                        elif float(img.float().std()) < 1e-3:
                            bad.append(f"ep{ep} idx{idx}: {cam} is a constant image")
        check(f"{t['task_id']} sampled frames decode and match schema", not bad and n_frames > 0,
              f"{n_frames} frames, {dt_decode / max(n_frames, 1) * 1000:.0f} ms/frame" + (f"; {bad[:3]}" if bad else ""))
        dataset_entries.append({
            "task_id": t["task_id"], "repo_id": rid, "revision": a["revision"], "instruction": t["instruction"],
            "num_episodes": a["num_episodes"], "num_frames": a["num_frames"], "usable_episodes": a["num_episodes"] - len(t["excluded_episodes"]),
            "split_counts": s["counts"], "local_path": str(local),
        })

    first = next(iter(audit["datasets"].values()))
    manifest = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "num_tasks": len(tasks),
        "total_episodes": sum(n_eps),
        "total_usable_episodes": sum(n_eps) - sum(len(t["excluded_episodes"]) for t in tasks),
        "total_frames": sum(audit["datasets"][t["dataset"]]["num_frames"] for t in tasks),
        "fps": first["fps"],
        "robot_type": first["robot_type"],
        "codebase_version": first["codebase_version"],
        "cameras": {k: {"height": c["height"], "width": c["width"], "codec": c["codec"]} for k, c in first["features"]["cameras"].items()},
        "state": {"key": "observation.state", **first["features"]["vectors"]["observation.state"]},
        "action": {"key": "action", **first["features"]["vectors"]["action"]},
        "action_dim": first["features"]["vectors"]["action"]["shape"][0],
        "state_dim": first["features"]["vectors"]["observation.state"]["shape"][0],
        "split_counts_total": {k: len(v) for k, v in split_manifest["global"].items()},
        "split_manifest": "reports/split_manifest.json",
        "datasets": dataset_entries,
    }
    return finish(reports, checks, split_manifest, manifest, failed)


def finish(reports: Path, checks: list[dict], split_manifest, manifest, failed: bool) -> int:
    if manifest is not None:
        (reports / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    n_fail = sum(c["status"] == "FAIL" for c in checks)
    L = ["# Dataset validation\n", f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}\n",
         f"## Verdict: **{'FAIL' if failed else 'PASS'}** ({len(checks) - n_fail}/{len(checks)} checks passed)\n",
         "| status | check | detail |", "|--------|-------|--------|"]
    L += [f"| {c['status']} | {c['check']} | {c['detail'].replace('|', '/')} |" for c in checks]
    if split_manifest:
        L += ["\n## Split (episodes per task)\n", "| task | dataset | train | val | test | excluded |", "|------|---------|------:|----:|-----:|----------|"]
        for tid, s in split_manifest["tasks"].items():
            L.append(f"| {tid} | `{s['dataset'].split('/')[-1]}` | {s['counts']['train']} | {s['counts']['val']} | {s['counts']['test']} | {s['excluded'] or '-'} |")
        g = split_manifest["global"]
        L.append(f"| **total** | | {len(g['train'])} | {len(g['val'])} | {len(g['test'])} | {len(g['excluded'])} |")
    (reports / "validation.md").write_text("\n".join(L) + "\n")
    print(f"[validate] {'FAIL' if failed else 'PASS'} — wrote reports/validation.md")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
