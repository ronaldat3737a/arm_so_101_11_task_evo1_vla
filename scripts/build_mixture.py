#!/usr/bin/env python
"""Build a training dataset for a data mixture (configs/data_mixture.yaml) with the official
lerobot.datasets.aggregate.aggregate_datasets (file copies, no re-encoding).

REAL_ONLY needs no new dataset (datasets/so101_11tasks + train episodes). SIM_ONLY / REAL_PLUS_SIM create
datasets/mixtures/<MIXTURE> and reports/mixture_<MIXTURE>.json listing the episodes to train on: real train episodes
(real val/test stay held out) + all sim episodes. Normalization stats are recomputed over exactly those episodes.

    python scripts/build_mixture.py --mixture REAL_PLUS_SIM
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--mixture", required=True)
ap.add_argument("--force", action="store_true")
a = ap.parse_args()
cfg = yaml.safe_load((ROOT / "configs" / "data_mixture.yaml").read_text())
srcs = cfg["mixtures"][a.mixture]["datasets"]
if srcs == ["real"]:
    print("REAL_ONLY uses datasets/so101_11tasks directly (train episodes from reports/split_manifest.json)")
    sys.exit(0)

from lerobot.datasets.aggregate import aggregate_datasets  # noqa: E402
from lerobot.datasets.compute_stats import aggregate_stats  # noqa: E402
from lerobot.datasets.io_utils import write_stats  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402

from scripts.compute_norm_stats import per_episode_stats  # noqa: E402
from src.training.launcher import train_episodes  # noqa: E402

out = ROOT / cfg["output_root"] / a.mixture
if out.exists():
    if not a.force:
        sys.exit(f"{out} exists; use --force")
    shutil.rmtree(out)
roots = [ROOT / cfg["sources"][s]["root"] for s in srcs]
ids = [f"local/{s}" for s in srcs]
metas = [LeRobotDatasetMetadata(i, root=r) for i, r in zip(ids, roots, strict=True)]
aggregate_datasets(repo_ids=ids, aggr_repo_id=f"local/{a.mixture}", roots=roots, aggr_root=out, concatenate_videos=False, concatenate_data=False)
episodes, off = [], 0
for s, m in zip(srcs, metas, strict=True):
    sel = train_episodes("all") if cfg["sources"][s]["episodes"] == "train_split" else list(range(m.total_episodes))
    episodes += [off + e for e in sel]
    off += m.total_episodes
st = per_episode_stats(out)
keys = sorted(json.loads((out / "meta" / "stats.json").read_text()))
write_stats(aggregate_stats([{k: v for k, v in st[e].items() if k in keys} for e in episodes]), out)
rep = {"mixture": a.mixture, "root": str(out.relative_to(ROOT)), "sources": {s: {"root": str(r.relative_to(ROOT)), "episodes": m.total_episodes} for s, r, m in zip(srcs, roots, metas, strict=True)},
       "train_episodes": episodes, "n_train_episodes": len(episodes), "stats": "recomputed over train_episodes only"}
(ROOT / "reports" / f"mixture_{a.mixture}.json").write_text(json.dumps(rep, indent=2))
print(json.dumps({k: v for k, v in rep.items() if k != "train_episodes"}, indent=2))
