#!/usr/bin/env python
"""ACT baseline on task_01 with the official lerobot-train (Phase 9).

    python scripts/train_act_baseline.py --config configs/act_baseline.yaml [--steps 200] [--dry-run]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.launcher import run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=str(ROOT / "configs" / "act_baseline.yaml"))
ap.add_argument("--steps", type=int)
ap.add_argument("--batch-size", type=int)
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
ov = {}
if a.steps:
    ov["steps"] = a.steps
    ov["save_freq"] = a.steps
if a.batch_size:
    ov["batch_size"] = a.batch_size
r = run(a.config, ov, a.dry_run)
print(json.dumps({k: r.get(k) for k in ("return_code", "wall_time_s", "peak_rss_gb_process_tree", "peak_vram_used_gb_nvidia_smi", "log", "output_dir")}, indent=2))
sys.exit(r.get("return_code", 0) or 0)
