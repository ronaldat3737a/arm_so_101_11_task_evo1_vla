#!/usr/bin/env python
"""Seed map {task: [seeds]} of the first N successful generation seeds per task (exact training layouts of a generated
sim dataset), verified by resetting the env with the same profile / split and comparing the recorded layouts.

    SO101_SIM_CONFIG=configs/simulation_v3.yaml python scripts/make_seed_map.py --report-prefix sim_v3_report_task_ \
        --profile position_wide --out reports/seed_map_sim_v3_train.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

from src.simulation.env import SO101Env  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report-prefix", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seed_map, maxdiff = {}, 0.0
    for i in range(1, 12):
        t = f"task_{i:02d}"
        r = json.loads((ROOT / "reports" / f"{a.report_prefix}{i:02d}.json").read_text())
        att = [x for x in r["tasks"][t]["attempts"] if x["success"]][: a.n]
        seed_map[t] = [x["seed"] for x in att]
        env = SO101Env(t, split="train", profile=a.profile, render=False)
        for x in att:
            _, info = env.reset(seed=x["seed"])
            for k, v in x["layout"].items():
                maxdiff = max(maxdiff, float(np.max(np.abs(np.array(info["layout"][k]) - np.array(v)))))
        env.close()
    Path(a.out).write_text(json.dumps(seed_map, indent=1))
    print(json.dumps({"out": a.out, "episodes": sum(len(v) for v in seed_map.values()), "max_layout_diff": maxdiff}))
    return 0 if maxdiff < 1e-9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
