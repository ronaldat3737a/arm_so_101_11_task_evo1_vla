#!/usr/bin/env python
"""Debug helper: run a task's oracle waypoint by waypoint and print object poses after each waypoint.

    python scripts/trace_oracle.py task_11 --profile position --seed 1 --split train
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

from src.simulation.env import SO101Env  # noqa: E402
from src.simulation.oracle import Oracle  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("task")
ap.add_argument("--profile", default="fixed")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--split", default="test")
a = ap.parse_args()
env = SO101Env(a.task, split=a.split, profile=a.profile, render=False)
env.reset(seed=a.seed)
o = Oracle(env)
names = [ob.name for ob in env.task.objects()]
print("start", {n: env.body_pos(n).round(3).tolist() for n in names})
for i, w in enumerate(env.task.oracle_plan(env)):
    try:
        ok, info, _ = o.run([w])
    except RuntimeError as e:
        print("wp", i, "ORACLE ERROR", e)
        break
    print(f"wp{i} tgt={np.round(w.pos, 3)} g={w.grip} t={env.t} gp={o.grasp_point().round(3)} grip={np.rad2deg(env.data.qpos[5]):.1f}",
          {n: env.body_pos(n).round(3).tolist() for n in names}, info.get("failure_reason"), "|", info.get("unmet"))
    if info.get("success") or info.get("failure_reason"):
        break
