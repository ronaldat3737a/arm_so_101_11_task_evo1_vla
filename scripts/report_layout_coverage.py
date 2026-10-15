#!/usr/bin/env python
"""Sampled object-placement coverage per task and object for two profiles (no physics stepping, no rendering):
radius / azimuth / yaw ranges actually produced by N resets on the train split.

    SO101_SIM_CONFIG=configs/simulation_v3.yaml python scripts/report_layout_coverage.py --profiles position_visual,position_wide
Writes reports/layout_coverage.{json,md}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

from src.simulation.env import SO101Env  # noqa: E402
from src.simulation.tasks import TASKS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profiles", default="position_visual,position_wide")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=900000)
    a = ap.parse_args()
    profs = a.profiles.split(",")
    rep = {"sim_config": os.environ.get("SO101_SIM_CONFIG", "configs/simulation.yaml"), "n_resets": a.n, "tasks": {}}
    for t in sorted(TASKS):
        rep["tasks"][t] = {}
        for prof in profs:
            env = SO101Env(t, split="train", profile=prof, render=False)
            vals: dict = {}
            for k in range(a.n):
                env.reset(seed=a.seed0 + k)
                for name in [it["name"] for it in env.task.randomized()]:
                    x, y, yaw = env.layout[name]
                    vals.setdefault(name, []).append((math.hypot(x, y), math.degrees(math.atan2(y, x)), yaw))
            env.close()
            rep["tasks"][t][prof] = {n: {"r_m": [round(float(np.min(v[:, 0])), 3), round(float(np.max(v[:, 0])), 3)],
                                         "theta_deg": [round(float(np.min(v[:, 1])), 1), round(float(np.max(v[:, 1])), 1)],
                                         "yaw_deg": [round(float(np.min(v[:, 2])), 1), round(float(np.max(v[:, 2])), 1)]}
                                     for n, v in ((n, np.asarray(v)) for n, v in vals.items())}
    (ROOT / "reports" / "layout_coverage.json").write_text(json.dumps(rep, indent=2))
    L = [f"# Placement coverage ({rep['sim_config']}, train split, {a.n} resets per task and profile)", "",
         "| task | object | " + " | ".join(f"{p} r (m) | {p} θ (deg) | {p} yaw (deg)" for p in profs) + " |",
         "|---|---|" + "---|" * (3 * len(profs))]
    for t, d in rep["tasks"].items():
        for name in d[profs[0]]:
            cells = []
            for p in profs:
                c = d[p][name]
                cells += [f"{c['r_m'][0]:.2f}–{c['r_m'][1]:.2f}", f"{c['theta_deg'][0]:.0f}…{c['theta_deg'][1]:.0f}", f"{c['yaw_deg'][0]:.0f}…{c['yaw_deg'][1]:.0f}"]
            L.append(f"| {t} | {name} | " + " | ".join(cells) + " |")
    (ROOT / "reports" / "layout_coverage.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
