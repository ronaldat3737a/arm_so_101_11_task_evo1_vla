#!/usr/bin/env python
"""Evaluate any LeRobot policy checkpoint (ACT, Evo-1, ...) in the MuJoCo simulation.

    python scripts/evaluate_policy.py --checkpoint outputs/<run>/checkpoints/last/pretrained_model \
        --task task_01 --splits train,test --profile position --episodes 10 [--video]

Writes reports/evaluation_<name>.json and .md (name = --name or run directory).
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

from src.evaluation.policy_eval import evaluate, to_markdown  # noqa: E402
from src.simulation.tasks import TASKS  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--task", default="all")
ap.add_argument("--splits", default="train,val,test")
ap.add_argument("--profile", default="position", choices=["fixed", "position", "position_visual"])
ap.add_argument("--episodes", type=int, default=10)
ap.add_argument("--seed0", type=int, default=1000)
ap.add_argument("--device", default="cuda")
ap.add_argument("--name")
ap.add_argument("--video", action="store_true")
ap.add_argument("--n-action-steps", type=int, help="override policy n_action_steps (chunk execution length)")
a = ap.parse_args()

tasks = sorted(TASKS) if a.task == "all" else a.task.split(",")
ov = {"n_action_steps": a.n_action_steps} if a.n_action_steps else None
name = a.name or Path(a.checkpoint).resolve().parent.parent.parent.name
rep = evaluate(a.checkpoint, tasks, a.splits.split(","), a.profile, a.episodes, a.seed0, a.device, ov, ROOT / "reports" / "videos" if a.video else None, name)
out = ROOT / "reports" / f"evaluation_{name}_{a.profile}"
out.with_suffix(".json").write_text(json.dumps(rep, indent=2))
out.with_suffix(".md").write_text(to_markdown(rep))
print(to_markdown(rep))
