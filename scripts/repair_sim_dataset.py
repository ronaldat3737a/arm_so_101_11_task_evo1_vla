#!/usr/bin/env python
"""Repair a sim dataset whose generation was interrupted (SIGINT) after N complete episodes.

Keeps exactly the episodes listed in meta/episodes (the ones whose save_episode completed), removes data rows of the
partial episode, recomputes meta/stats.json from the per-episode stats with the official aggregate_stats, and verifies
the result with the official LeRobotDataset loader (sampled frames of every episode, both cameras).
The original data parquet is backed up next to the dataset (nothing is deleted).

    python scripts/repair_sim_dataset.py --root datasets/sim_so101
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(ROOT / "datasets" / "sim_so101"))
a = ap.parse_args()
root = Path(a.root)

from lerobot.datasets.compute_stats import aggregate_stats  # noqa: E402
from lerobot.datasets.io_utils import write_stats  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from scripts.compute_norm_stats import per_episode_stats  # noqa: E402

info = json.loads((root / "meta" / "info.json").read_text())
eps = pq.read_table(sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))[0]).to_pandas()
n_eps = len(eps)
n_frames = int(eps["length"].sum())
assert list(eps["episode_index"]) == list(range(n_eps)), "episode indices not contiguous"
assert info["total_episodes"] == n_eps and info["total_frames"] == n_frames, "info.json disagrees with meta/episodes"

report = {"root": str(root), "kept_episodes": n_eps, "kept_frames": n_frames, "data_files": {}}
backup = root.parent / f"{root.name}_repair_backup"
backup.mkdir(exist_ok=True)
for f in sorted((root / "data").glob("chunk-*/file-*.parquet")):
    t = pq.read_table(f)
    ei = np.asarray(t.column("episode_index").to_pylist())
    keep = ei < n_eps
    report["data_files"][str(f.relative_to(root))] = {"rows_before": t.num_rows, "rows_removed": int((~keep).sum())}
    if (~keep).any():
        dst = backup / f.relative_to(root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(f, dst)
        pq.write_table(t.filter(keep), f)
total_rows = sum(pq.read_metadata(f).num_rows for f in (root / "data").glob("chunk-*/file-*.parquet"))
assert total_rows == n_frames, f"data rows {total_rows} != {n_frames}"

st = per_episode_stats(root)
keys = sorted(json.loads((root / "meta" / "stats.json").read_text()))
write_stats(aggregate_stats([{k: v for k, v in st[e].items() if k in keys} for e in range(n_eps)]), root)

ds = LeRobotDataset("local/sim_so101", root=root, video_backend="pyav", download_videos=False)
assert ds.num_episodes == n_eps and ds.num_frames == n_frames
bad = 0
for e in range(n_eps):
    lo, hi = int(ds.meta.episodes["dataset_from_index"][e]), int(ds.meta.episodes["dataset_to_index"][e])
    for i in (lo, (lo + hi) // 2, hi - 1):
        it = ds[i]
        if tuple(it["observation.images.top"].shape) != (3, 360, 640) or tuple(it["observation.images.wrist"].shape) != (3, 480, 640) or int(it["episode_index"]) != e:
            bad += 1
report["verified_frames"] = 3 * n_eps
report["bad_frames"] = bad
report["tasks"] = ds.meta.tasks.index.tolist()
report["backup"] = str(backup)
report["status"] = "PASS" if bad == 0 else "FAIL"
(ROOT / "reports" / "sim_dataset_repair.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
sys.exit(0 if bad == 0 else 1)
