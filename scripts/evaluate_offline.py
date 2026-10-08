#!/usr/bin/env python
"""Offline action-quality evaluation of a LeRobot checkpoint on held-out real episodes.

    python scripts/evaluate_offline.py --checkpoint outputs/<run>/checkpoints/last/pretrained_model --task task_01 --split test
Writes reports/offline_<name>_<split>.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.offline_eval import evaluate_offline  # noqa: E402
from src.simulation.tasks import TASKS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--task", default="all")
ap.add_argument("--split", default="test", choices=["val", "test"])
ap.add_argument("--frames-per-episode", type=int, default=8)
ap.add_argument("--horizon", type=int)
ap.add_argument("--name")
a = ap.parse_args()
tasks = sorted(TASKS) if a.task == "all" else a.task.split(",")
rep = evaluate_offline(a.checkpoint, tasks, a.split, a.frames_per_episode, a.horizon)
name = a.name or Path(a.checkpoint).resolve().parent.parent.parent.name
(ROOT / "reports" / f"offline_{name}_{a.split}.json").write_text(json.dumps(rep, indent=2))
print(json.dumps({k: rep[k] for k in ("mean_action_mae", "latency_ms", "resources")}, indent=2))
