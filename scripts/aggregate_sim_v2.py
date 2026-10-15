#!/usr/bin/env python
"""Aggregate the per-task v2 sim datasets (datasets/sim_so101_v2/task_XX, scripts/generate_sim_v2.sh) into one
LeRobotDataset with the official lerobot.datasets.aggregate.aggregate_datasets (file copies, no re-encoding), then
verify: episode/frame counts equal the sum of the parts, every episode decodes its first and last frame (both cameras),
and the instructions are the exact dataset strings. Writes reports/sim_v2_dataset_report.json.

    python scripts/aggregate_sim_v2.py
    python scripts/aggregate_sim_v2.py --src datasets/sim_so101_v3 --out datasets/sim_so101_v3_all --report sim_v3_dataset_report \
        --sim-config configs/simulation_v3.yaml      (v3: task_XX + task_XX_topupN parts)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from src.simulation.tasks import INSTRUCTIONS

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/sim_so101_v2")
    ap.add_argument("--out", default="datasets/sim_so101_v2_all")
    ap.add_argument("--report", default="sim_v2_dataset_report")
    ap.add_argument("--sim-config", default="configs/simulation_v2.yaml")
    a = ap.parse_args()
    parts = sorted(p for p in (ROOT / a.src).glob("task_*") if (p / "meta" / "info.json").exists())
    if len({p.name.split("_topup")[0] for p in parts}) != 11:
        print(f"STOP: expected datasets for 11 tasks, found {sorted(p.name for p in parts)}")
        return 1
    infos = [json.loads((p / "meta" / "info.json").read_text()) for p in parts]
    out = ROOT / a.out
    if out.exists():
        shutil.rmtree(out)
    aggregate_datasets(repo_ids=[f"local/sim_v2_{p.name}" for p in parts], aggr_repo_id=f"local/{out.name}", roots=parts,
                       aggr_root=out, concatenate_videos=False, concatenate_data=False)
    ds = LeRobotDataset(f"local/{out.name}", root=out, video_backend="pyav", download_videos=False)
    n_ep, n_fr = ds.meta.total_episodes, ds.meta.total_frames
    exp_ep, exp_fr = sum(i["total_episodes"] for i in infos), sum(i["total_frames"] for i in infos)
    bad, tasks = [], set()
    for e in range(n_ep):
        lo, hi = int(ds.meta.episodes["dataset_from_index"][e]), int(ds.meta.episodes["dataset_to_index"][e])
        for i in (lo, hi - 1):
            try:
                it = ds[i]
                assert it["observation.images.top"].shape[-2:] == (360, 640) and it["observation.images.wrist"].shape[-2:] == (480, 640)
                tasks.add(it["task"])
            except Exception as exc:  # noqa: BLE001
                bad.append({"episode": e, "frame": i, "error": repr(exc)[:200]})
    rep = {"root": str(out.relative_to(ROOT)), "sim_config": a.sim_config, "parts": [str(p.relative_to(ROOT)) for p in parts],
           "episodes": n_ep, "frames": n_fr, "expected_episodes": exp_ep, "expected_frames": exp_fr,
           "decode_checked_frames": 2 * n_ep, "decode_errors": bad, "instructions_exact": sorted(tasks) == sorted(set(INSTRUCTIONS.values())),
           "passed": n_ep == exp_ep and n_fr == exp_fr and not bad and sorted(tasks) == sorted(set(INSTRUCTIONS.values()))}
    (ROOT / "reports" / f"{a.report}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "decode_errors"} | {"n_decode_errors": len(bad)}, indent=2))
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
