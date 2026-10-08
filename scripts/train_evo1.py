#!/usr/bin/env python
"""Fine-tune Evo-1 (official lerobot.policies.evo1) on the merged SO-101 dataset.

    python scripts/train_evo1.py --config configs/evo1.yaml
    python scripts/train_evo1.py --config configs/evo1.yaml --set training.max_steps=20 --set policy.image_resolution=[256,256]

Pre-flight: aggregated dataset + train-only stats + split manifest must exist; GPU must be available; the
VLM must be in the local HF cache (or downloadable).
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def preflight(cfg: dict) -> list[str]:
    import torch

    errs = []
    agg = ROOT / "datasets" / "so101_11tasks"
    if not (agg / "meta" / "info.json").exists():
        errs.append("merged dataset missing: run scripts/aggregate_datasets.py")
    if not (ROOT / "reports" / "norm_stats.json").exists():
        errs.append("train-only norm stats missing: run scripts/compute_norm_stats.py")
    if not (ROOT / "reports" / "split_manifest.json").exists():
        errs.append("split manifest missing: run scripts/validate_datasets.py")
    if cfg["policy"].get("device", "cuda") == "cuda" and not torch.cuda.is_available():
        errs.append("CUDA not available")
    if cfg["policy"]["type"] != "evo1":
        errs.append("policy.type must be evo1")
    return errs


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--config", default=str(ROOT / "configs" / "evo1.yaml"))
ap.add_argument("--set", action="append", default=[], help="override, e.g. training.max_steps=20")
ap.add_argument("--resume", help="run dir to continue (outputs/<run>): restores weights, optimizer, scheduler, RNG, data position")
a = ap.parse_args()
cfg = yaml.safe_load(Path(a.config).read_text())
ov = {}
for s in a.set:
    k, v = s.split("=", 1)
    ov[k] = yaml.safe_load(v)
    if "." in k:
        sect, key = k.split(".", 1)
        cfg.setdefault(sect, {})[key] = ov[k]
    else:
        cfg[k] = ov[k]
errs = preflight(cfg)
if errs:
    print("PREFLIGHT FAIL:\n- " + "\n- ".join(errs))
    sys.exit(1)

from src.training.trainer import run  # noqa: E402

summary = run(a.config, ov, resume_dir=a.resume)
print(json.dumps({k: v for k, v in summary.items() if k not in ("log", "environment", "config")}, indent=2))
