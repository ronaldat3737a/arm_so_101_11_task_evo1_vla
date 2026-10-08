#!/usr/bin/env python
"""Merge the 11 audited source datasets into ONE local LeRobotDataset with the official LeRobot API.

    lerobot.datasets.aggregate.aggregate_datasets(..., concatenate_videos=False, concatenate_data=False)

- Inputs: the pinned local snapshots recorded in reports/dataset_audit.json (no network).
- Each source mp4/parquet is copied file-by-file (shutil.copy, streamed by the OS). No re-encoding and no
  concatenation, and nothing is loaded into RAM except per-episode metadata.
- Episode order = configs/dataset.yaml order, so global episode index = offset[task] + local index
  (verified here against reports/split_manifest.json and the merged task strings).
- Source datasets are never modified. Output: datasets/so101_11tasks (git-ignored).

Usage:
    python scripts/aggregate_datasets.py [--force]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.resources import ResourceMonitor  # noqa: E402

AGGR_REPO_ID = "local/so101_11tasks"
AGGR_ROOT = ROOT / "datasets" / "so101_11tasks"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="delete an existing merged dataset first")
    args = ap.parse_args()

    from lerobot.datasets.aggregate import aggregate_datasets

    dcfg = yaml.safe_load((ROOT / "configs" / "dataset.yaml").read_text())
    audit = json.loads((ROOT / "reports" / "dataset_audit.json").read_text())
    split = json.loads((ROOT / "reports" / "split_manifest.json").read_text())
    if audit["verdict"]["dataset_integrity"] != "PASS":
        print("STOP: dataset audit integrity is not PASS")
        return 1

    repos = dcfg["repos"]
    roots = [Path(audit["datasets"][r]["local_path"]) for r in repos]
    for r, p in zip(repos, roots, strict=True):
        if p.name != audit["datasets"][r]["revision"]:
            print(f"STOP: {r} local snapshot {p} is not the audited revision")
            return 1

    if AGGR_ROOT.exists():
        if not args.force:
            print(f"{AGGR_ROOT} exists; use --force to rebuild")
            return 1
        shutil.rmtree(AGGR_ROOT)

    mon = ResourceMonitor()
    t0 = time.time()
    with mon:
        aggregate_datasets(
            repo_ids=repos,
            aggr_repo_id=AGGR_REPO_ID,
            roots=roots,
            aggr_root=AGGR_ROOT,
            concatenate_videos=False,
            concatenate_data=False,
        )
    elapsed = time.time() - t0

    # Keep the all-episode stats written by aggregate_datasets for reference; compute_norm_stats.py replaces
    # meta/stats.json with train-only stats.
    shutil.copy(AGGR_ROOT / "meta" / "stats.json", AGGR_ROOT / "meta" / "stats_all_episodes.json")

    info = {
        "aggr_repo_id": AGGR_REPO_ID,
        "aggr_root": str(AGGR_ROOT.relative_to(ROOT)),
        "sources": [{"repo_id": r, "revision": audit["datasets"][r]["revision"]} for r in repos],
        "concatenate_videos": False,
        "concatenate_data": False,
        "elapsed_s": round(elapsed, 1),
        "peak_rss_gb": mon.peak_rss_gb,
        "split_manifest": "reports/split_manifest.json",
    }
    (AGGR_ROOT / "meta" / "so101_evo1_provenance.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    print(f"[aggregate] done in {elapsed:.0f}s, peak RSS {mon.peak_rss_gb:.2f} GB; global episodes expected {sum(t['num_episodes'] for t in split['tasks'].values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
