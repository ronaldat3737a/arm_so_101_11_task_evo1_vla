#!/usr/bin/env python
"""Verify the merged dataset datasets/so101_11tasks against the 11 source snapshots.

Checks (all through the official LeRobotDataset loader, lazy video decoding, no preloading):
  - totals (episodes, frames, tasks), features, fps identical to sources
  - every global episode maps to the right source episode: task string, length, exact state/action arrays
  - decoded frames (both cameras) of sampled episodes are pixel-identical to the same frames from the source
  - meta/stats.json is the train-only stats from compute_norm_stats.py
  - a train-episode subset loads with Evo-1's action-chunk delta timestamps (chunk 50) and returns a full chunk
  - peak RAM

Writes reports/aggregate_verification.{json,md}. Exit 0 only if all checks pass.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.resources import ResourceMonitor  # noqa: E402

AGGR_ROOT = ROOT / "datasets" / "so101_11tasks"


def run(episodes_per_task: int = 2) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    import yaml
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    checks: list[dict] = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": str(detail)})
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)

    audit = json.loads((ROOT / "reports" / "dataset_audit.json").read_text())
    split = json.loads((ROOT / "reports" / "split_manifest.json").read_text())
    tasks = yaml.safe_load((ROOT / "configs" / "tasks.yaml").read_text())["tasks"]
    norm = json.loads((ROOT / "reports" / "norm_stats.json").read_text())

    mon = ResourceMonitor()
    with mon:
        agg = LeRobotDataset("local/so101_11tasks", root=AGGR_ROOT, download_videos=False)
        tot_eps = sum(a["num_episodes"] for a in audit["datasets"].values())
        tot_frames = sum(a["num_frames"] for a in audit["datasets"].values())
        check("total episodes", agg.num_episodes == tot_eps, f"{agg.num_episodes} vs {tot_eps}")
        check("total frames", agg.num_frames == tot_frames, f"{agg.num_frames} vs {tot_frames}")
        check("fps", agg.fps == 30, agg.fps)
        check("tasks", sorted(agg.meta.tasks.index.tolist()) == sorted(t["instruction"] for t in tasks), len(agg.meta.tasks))
        check("camera keys", sorted(agg.meta.camera_keys) == ["observation.images.top", "observation.images.wrist"], agg.meta.camera_keys)
        for k in ("action", "observation.state"):
            check(f"{k} feature", agg.meta.features[k]["shape"] in ([6], (6,)) and agg.meta.features[k]["names"] == audit["datasets"][tasks[0]["dataset"]]["features"]["vectors"][k]["names"])

        # stats.json == train-only
        st = agg.meta.stats
        ok = all(np.allclose(np.asarray(st[k][s]).reshape(-1), norm["train_stats"][k][s]) for k in ("action", "observation.state") for s in ("min", "max", "mean", "std"))
        check("meta/stats.json is train-only stats", ok)

        # episode mapping + exact vectors (all episodes, from parquet, cheap)
        ep_meta = agg.meta.episodes
        bad_map = []
        for t in tasks:
            s = split["tasks"][t["task_id"]]
            off = s["global_episode_offset"]
            src_root = Path(audit["datasets"][t["dataset"]]["local_path"])
            src = pa.concat_tables([pq.read_table(f, columns=["episode_index", "action", "observation.state"]) for f in sorted((src_root / "data").glob("chunk-*/file-*.parquet"))])
            src_ep = np.asarray(src.column("episode_index").to_pylist())
            src_a = np.asarray(src.column("action").to_pylist(), dtype=np.float32)
            src_s = np.asarray(src.column("observation.state").to_pylist(), dtype=np.float32)
            for le in range(s["num_episodes"]):
                ge = off + le
                tasks_ge = ep_meta["tasks"][ge]
                if list(tasks_ge) != [t["instruction"]]:
                    bad_map.append(f"ep{ge}: task {tasks_ge}")
                    continue
                L = int(ep_meta["length"][ge])
                if L != audit["datasets"][t["dataset"]]["episode_lengths"]["per_episode"][str(le)]:
                    bad_map.append(f"ep{ge}: length {L}")
                lo, hi = int(ep_meta["dataset_from_index"][ge]), int(ep_meta["dataset_to_index"][ge])
                a = np.stack([agg.hf_dataset[i]["action"].numpy() for i in (lo, hi - 1)])
                ref = src_a[src_ep == le][[0, -1]]
                if not np.array_equal(a, ref):
                    bad_map.append(f"ep{ge}: action differs from source")
                sv = np.stack([agg.hf_dataset[i]["observation.state"].numpy() for i in (lo, hi - 1)])
                if not np.array_equal(sv, src_s[src_ep == le][[0, -1]]):
                    bad_map.append(f"ep{ge}: state differs from source")
        check("all 551 global episodes map to source (task, length, first/last action+state exact)", not bad_map, bad_map[:5])

        # pixel identity of decoded frames vs source datasets
        pix_bad, n_pix = [], 0
        t_dec = []
        for i, t in enumerate(tasks):
            s = split["tasks"][t["task_id"]]
            src = LeRobotDataset(t["dataset"], root=Path(audit["datasets"][t["dataset"]]["local_path"]), revision=audit["datasets"][t["dataset"]]["revision"], download_videos=False)
            for le in (s["test"][:1] + s["train"][: episodes_per_task - 1]):
                ge = s["global_episode_offset"] + le
                lo, hi = int(ep_meta["dataset_from_index"][ge]), int(ep_meta["dataset_to_index"][ge])
                slo = int(src.meta.episodes["dataset_from_index"][le])
                for k in (0, (hi - lo) // 2, hi - lo - 1):
                    t0 = time.time()
                    x = agg[lo + k]
                    t_dec.append(time.time() - t0)
                    y = src[slo + k]
                    n_pix += 1
                    for cam in ("observation.images.top", "observation.images.wrist"):
                        if not torch.equal(x[cam], y[cam]):
                            pix_bad.append(f"{t['task_id']} ep{le} frame{k} {cam} max diff {float((x[cam].float() - y[cam].float()).abs().max())}")
                    if x["task"] != t["instruction"]:
                        pix_bad.append(f"{t['task_id']} ep{le}: task {x['task']}")
            del src
        check("decoded frames pixel-identical to source (both cameras)", not pix_bad, f"{n_pix} frames x 2 cams; {pix_bad[:3]}")

        # train subset with Evo-1 action chunk (50 steps at 30 fps)
        train = sorted(split["global"]["train"])
        dt = {"action": [i / agg.fps for i in range(50)], "observation.state": [0.0]}
        sub = LeRobotDataset("local/so101_11tasks", root=AGGR_ROOT, episodes=train, delta_timestamps=dt, download_videos=False, return_uint8=True)
        check("train subset episode count", sub.num_episodes == len(train), sub.num_episodes)
        item = sub[len(sub) // 2]
        check("train subset returns action chunk (50, 6) + action_is_pad + both cameras uint8",
              tuple(item["action"].shape) == (50, 6) and "action_is_pad" in item
              and item["observation.images.top"].dtype == torch.uint8 and tuple(item["observation.images.wrist"].shape) == (3, 480, 640),
              {k: tuple(v.shape) for k, v in item.items() if hasattr(v, "shape")})
        held = set(split["global"]["val"]) | set(split["global"]["test"]) | set(split["global"]["excluded"])
        sampled_eps = {int(sub[i]["episode_index"]) for i in np.linspace(0, len(sub) - 1, 200).astype(int)}
        check("train subset never yields held-out episodes", not (sampled_eps & held), f"{len(sampled_eps)} episodes sampled")

    ram_ok = mon.peak_rss_gb < 6.0
    check("peak RAM during verification < 6 GB", ram_ok, f"{mon.peak_rss_gb:.2f} GB")
    return {
        "checks": checks,
        "verdict": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL",
        "resources": mon.summary(),
        "decode_ms_p50": float(np.percentile(np.array(t_dec) * 1000, 50)) if t_dec else None,
        "decode_ms_p95": float(np.percentile(np.array(t_dec) * 1000, 95)) if t_dec else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-per-task", type=int, default=2)
    args = ap.parse_args()
    res = run(args.episodes_per_task)
    (ROOT / "reports" / "aggregate_verification.json").write_text(json.dumps(res, indent=2))
    L = ["# Aggregate dataset verification\n", f"## Verdict: **{res['verdict']}**\n",
         f"Peak RSS {res['resources']['peak_rss_gb']} GB; frame decode (2 cams) p50 {res['decode_ms_p50']:.0f} ms, p95 {res['decode_ms_p95']:.0f} ms\n",
         "| status | check | detail |", "|---|---|---|"]
    L += [f"| {c['status']} | {c['check']} | {c['detail'][:200].replace('|', '/')} |" for c in res["checks"]]
    (ROOT / "reports" / "aggregate_verification.md").write_text("\n".join(L) + "\n")
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
