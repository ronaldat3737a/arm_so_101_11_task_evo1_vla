#!/usr/bin/env python
"""Push test for the pots of every pot task, under a given scene config (SO101_SIM_CONFIG).

After a fixed-layout reset, a horizontal force is applied at the pot's rim height (on the side facing the robot,
pushing away from it), ramping linearly from 0 to --fmax N over --ramp s while the arm holds its rest pose. The first
event is recorded: `tip` (tilt > 30 deg, the task's pot_tipped_over criterion) or `slide` (horizontal displacement
> 2 cm with tilt < 10 deg), with the force at that moment. Also reports the pot's centre-of-mass height.

    SO101_SIM_CONFIG=configs/simulation_v3.yaml python scripts/check_pot_stability.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

from src.simulation.env import SO101Env  # noqa: E402

POT_TASKS = ("task_01", "task_03", "task_04", "task_05", "task_06")


def push(task: str, fmax: float, ramp: float) -> dict:
    env = SO101Env(task, split="train", profile="fixed", render=False)
    env.reset(seed=0)
    m, d = env.model, env.data
    bid = m.body("pot").id
    root = m.body_rootid[bid]
    com = d.subtree_com[root].copy()
    h = env.obj_info["pot"]["height"]
    r = env.obj_info["pot"]["inner_radius"]
    p0 = d.xpos[bid].copy()
    away = p0[:2] / (np.linalg.norm(p0[:2]) + 1e-9)  # radial direction from the robot base
    dt = m.opt.timestep
    n = int(ramp / dt)
    event, f_event, tilt = "none", None, 0.0
    for k in range(n):
        f = fmax * k / n
        d.qfrc_applied[:] = 0
        point = d.xpos[bid] + np.array([-away[0] * r, -away[1] * r, 0.0]) + d.xmat[bid].reshape(3, 3) @ np.array([0, 0, h])
        mujoco.mj_applyFT(m, d, np.array([away[0] * f, away[1] * f, 0.0]), np.zeros(3), point, bid, d.qfrc_applied)
        mujoco.mj_step(m, d)
        up = d.xmat[bid].reshape(3, 3)[:, 2]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, up[2]))))
        disp = float(np.linalg.norm(d.xpos[bid][:2] - p0[:2]))
        if tilt > 30:
            event, f_event = "tip", f
            break
        if disp > 0.02 and tilt < 10:
            event, f_event = "slide", f
            break
    out = {"task": task, "pot_mass_kg": float(m.body_subtreemass[bid]), "com_height_m": round(float(com[2] - p0[2]), 4),
           "pot_height_m": h, "event": event, "force_N": round(f_event, 2) if f_event is not None else None, "final_tilt_deg": round(tilt, 1)}
    env.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fmax", type=float, default=10.0)
    ap.add_argument("--ramp", type=float, default=2.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = os.environ.get("SO101_SIM_CONFIG", "configs/simulation.yaml")
    res = {"sim_config": cfg, "fmax_N": a.fmax, "ramp_s": a.ramp, "results": [push(t, a.fmax, a.ramp) for t in POT_TASKS]}
    for r in res["results"]:
        print(json.dumps(r))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
