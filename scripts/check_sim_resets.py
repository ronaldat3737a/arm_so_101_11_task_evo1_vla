#!/usr/bin/env python
"""Reset every task many times in every split/profile: placement must always succeed, the task must be neither
solved nor failed at reset, and test-split target objects must lie in test cells. Writes reports/sim_reset_check.json."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

from src.simulation.env import SO101Env  # noqa: E402
from src.simulation.tasks import TASKS  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
problems, total = [], 0
for t in sorted(TASKS):
    for sp in ("train", "val", "test"):
        e = SO101Env(t, split=sp, profile="position", render=False)
        split_items = [it["name"] for it in e.task.randomized() if it.get("split", True)]
        for seed in range(N):
            total += 1
            try:
                _, info = e.reset(seed=seed)
                ok, _ = e.task.success(e)
                f = e.task.failure(e)
                wrong = [n for n in split_items if info["split_of_layout"][n] != sp]
                if ok or f or wrong:
                    problems.append({"task": t, "split": sp, "seed": seed, "solved": ok, "failure": f, "wrong_split": wrong})
            except Exception as ex:  # noqa: BLE001
                problems.append({"task": t, "split": sp, "seed": seed, "error": str(ex)[:160]})
        e.close()
    for prof in ("fixed", "position_visual"):
        total += 1
        e = SO101Env(t, split="test", profile=prof, render=False)
        try:
            e.reset(seed=0)
            if e.task.success(e)[0] or e.task.failure(e):
                problems.append({"task": t, "profile": prof, "error": "solved/failed at reset"})
        except Exception as ex:  # noqa: BLE001
            problems.append({"task": t, "profile": prof, "error": str(ex)[:160]})
        e.close()
res = {"resets": total, "problems": len(problems), "details": problems}
(ROOT / "reports" / "sim_reset_check.json").write_text(json.dumps(res, indent=2))
print(f"{len(problems)} problems in {total} resets")
for p in problems[:20]:
    print(p)
sys.exit(1 if problems else 0)
