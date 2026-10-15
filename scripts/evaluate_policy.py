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
ap.add_argument("--profile", default="position", choices=["fixed", "position", "position_visual", "position_wide"])
ap.add_argument("--episodes", type=int, default=10)
ap.add_argument("--seed0", type=int, default=1000)
ap.add_argument("--device", default="cuda")
ap.add_argument("--name")
ap.add_argument("--video", action="store_true")
ap.add_argument("--n-action-steps", type=int, help="override policy n_action_steps (chunk execution length)")
ap.add_argument("--num-inference-timesteps", type=int, help="Evo-1 flow-matching sampling steps (official config field)")
ap.add_argument("--quant", choices=["bf16", "w8a8_vision"], help="torchao post-training quantization (default: none = BF16)")
ap.add_argument("--compile", action="store_true", help="torch.compile vision tower, language model and action head")
ap.add_argument("--seed-map", help="JSON file {task: [seeds]} overriding seed0+k (e.g. layouts of generated sim data)")
ap.add_argument("--trace", action="store_true", help="save per-episode state/action traces to outputs/traces")
a = ap.parse_args()

tasks = sorted(TASKS) if a.task == "all" else a.task.split(",")
ov = {k: v for k, v in (("n_action_steps", a.n_action_steps), ("num_inference_timesteps", a.num_inference_timesteps)) if v} or None
name = a.name or Path(a.checkpoint).resolve().parent.parent.parent.name
rep = evaluate(a.checkpoint, tasks, a.splits.split(","), a.profile, a.episodes, a.seed0, a.device, ov, ROOT / "reports" / "videos" if a.video else None, name,
               quant=a.quant, compile=a.compile, trace_dir=ROOT / "outputs" / "traces" if a.trace else None,
               seed_map=json.loads(Path(a.seed_map).read_text()) if a.seed_map else None)
out = ROOT / "reports" / f"evaluation_{name}_{a.profile}"
out.with_suffix(".json").write_text(json.dumps(rep, indent=2))
out.with_suffix(".md").write_text(to_markdown(rep))
print(to_markdown(rep))
