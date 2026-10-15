#!/usr/bin/env python
"""Generate synthetic demonstrations in MuJoCo with the scripted oracle and write them as a LeRobotDataset (v3.0) with
the SAME schema as the real data (observation.state / action float32[6] in canonical degrees, observation.images.top
640x360 and observation.images.wrist 640x480 AV1 videos, task = exact instruction, 30 fps).

Only TRAIN-region object positions are used, so sim data never covers val/test (unseen) positions.
Only successful oracle episodes are saved. Every attempted seed, its success flag and layout go to the report.
No learned model is used to create ground truth.

    python scripts/generate_sim_data.py --tasks all --episodes-per-task 20 --profile position_visual
Writes datasets/sim_so101 (or --out) and reports/sim_dataset_report.{md,json}
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

from src.simulation.action_adapter import FEATURE_NAMES  # noqa: E402
from src.simulation.env import SO101Env  # noqa: E402
from src.simulation.oracle import Oracle  # noqa: E402
from src.simulation.tasks import TASKS  # noqa: E402
from src.utils.resources import ResourceMonitor  # noqa: E402


def features() -> dict:
    vid = lambda h, w: {"dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "channels"]}  # noqa: E731
    return {
        "action": {"dtype": "float32", "shape": [6], "names": FEATURE_NAMES},
        "observation.state": {"dtype": "float32", "shape": [6], "names": FEATURE_NAMES},
        "observation.images.top": vid(360, 640),
        "observation.images.wrist": vid(480, 640),
    }


def main() -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--episodes-per-task", type=int, default=20, help="successful episodes to keep per task")
    ap.add_argument("--max-attempts-factor", type=float, default=2.0)
    ap.add_argument("--profile", default="position_visual", choices=["position", "position_visual", "position_wide"])
    ap.add_argument("--seed0", type=int, default=100000, help="seeds disjoint from evaluation seeds (1000+)")
    ap.add_argument("--out", default=str(ROOT / "datasets" / "sim_so101"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dart-prob", type=float, default=0.0, help="probability of a DART perturbation per approach (recovery demos)")
    ap.add_argument("--dart-dxy", type=float, nargs=2, default=(0.02, 0.04), help="lateral perturbation range (m)")
    ap.add_argument("--dart-dyaw", type=float, default=15.0, help="yaw perturbation range +-deg")
    ap.add_argument("--dart-steps", type=int, nargs=2, default=(15, 25), help="perturbation duration range (steps at 30 Hz)")
    ap.add_argument("--hover-steps", type=int, default=0, help="hover at the corrected pre-grasp pose above a container (tasks 05/06)")
    ap.add_argument("--drop-height", type=float, nargs=2, default=None, help="release height range above the rim (tasks 01/03), m")
    ap.add_argument("--episodes-override", default=None, help='JSON {task: episodes} overriding --episodes-per-task')
    ap.add_argument("--report-name", default="sim_dataset_report", help="reports/<name>.{json,md}")
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        if not a.force:
            print(f"{out} exists; use --force")
            return 1
        shutil.rmtree(out)
    tasks = sorted(TASKS) if a.tasks == "all" else a.tasks.split(",")
    ds = LeRobotDataset.create("local/sim_so101", fps=30, features=features(), root=out, robot_type="so_follower", use_videos=True)
    oracle_opts = {}
    if a.dart_prob > 0:
        oracle_opts.update(dart_prob=a.dart_prob, dart_dxy=tuple(a.dart_dxy), dart_dyaw=a.dart_dyaw, dart_steps=tuple(a.dart_steps))
    if a.hover_steps:
        oracle_opts["hover_steps"] = a.hover_steps
    if a.drop_height:
        oracle_opts["drop_height"] = tuple(a.drop_height)
    n_override = json.loads(a.episodes_override) if a.episodes_override else {}
    report = {"oracle_opts": oracle_opts, "sim_config": os.environ.get("SO101_SIM_CONFIG", "configs/simulation.yaml"), "profile": a.profile, "split": "train", "seed0": a.seed0, "tasks": {}}
    mon = ResourceMonitor()
    t_all = time.time()
    with mon:
        for t in tasks:
            env = SO101Env(t, split="train", profile=a.profile, render=True)
            attempts, kept = [], 0
            n_target = int(n_override.get(t, a.episodes_per_task))
            max_attempts = int(n_target * a.max_attempts_factor)
            for k in range(max_attempts):
                if kept >= n_target:
                    break
                seed = a.seed0 + 1000 * (int(t.split("_")[1])) + k
                obs, info0 = env.reset(seed=seed)
                # oracle-only randomness (perturbations, drop heights) from its own stream: env.rng is left untouched
                env.oracle_opts = oracle_opts
                env.oracle_rng = np.random.default_rng(seed + 10_000_019) if oracle_opts else None
                prev = obs
                n_frames = 0

                def cb(o, act, info):
                    # stream frames to the dataset writer (images go to temp files on disk, not RAM);
                    # observation BEFORE executing the action, as in the real recordings
                    nonlocal prev, n_frames
                    ds.add_frame({
                        "observation.state": prev["observation.state"].astype(np.float32),
                        "action": np.asarray(act, dtype=np.float32),
                        "observation.images.top": prev["observation.images.top"],
                        "observation.images.wrist": prev["observation.images.wrist"],
                        "task": env.task.instruction,
                    })
                    prev = o
                    n_frames += 1

                orc = Oracle(env)
                try:
                    ok, info, steps = orc.run(env.task.oracle_plan(env), obs_cb=cb)
                    reason = info.get("failure_reason")
                except RuntimeError as exc:
                    ok, steps, reason = False, env.t, f"oracle_error: {exc}"
                attempts.append({"seed": seed, "success": bool(ok), "steps": int(steps), "failure_reason": reason, "layout": info0["layout"],
                                 "perturbations": orc.perturbations})
                if not ok:
                    ds.clear_episode_buffer()  # official API: discard a failed episode
                    continue
                ds.save_episode(parallel_encoding=False)  # one AV1 encoder at a time: lower peak RAM
                kept += 1
                print(f"[simdata] {t} seed {seed}: saved episode ({n_frames} frames)", flush=True)
            env.close()
            report["tasks"][t] = {"instruction": TASKS[t]().instruction, "kept": kept, "attempted": len(attempts),
                                  "oracle_success_rate": sum(x["success"] for x in attempts) / max(len(attempts), 1), "attempts": attempts}
            print(f"[simdata] {t}: kept {kept}/{len(attempts)} attempts", flush=True)
        ds.finalize()
    report["total_episodes"] = sum(v["kept"] for v in report["tasks"].values())
    report["total_frames"] = int(ds.meta.total_frames)
    report["wall_time_s"] = round(time.time() - t_all, 1)
    report["resources"] = mon.summary()
    report["dataset_root"] = str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out)
    (ROOT / "reports" / f"{a.report_name}.json").write_text(json.dumps(report, indent=2))
    L = ["# Synthetic (simulation) dataset\n",
         f"Root: `{report['dataset_root']}` (LeRobotDataset v3.0, same schema as the real data). Profile `{a.profile}`, object positions from the TRAIN region only.",
         "Generated by the scripted oracle (privileged state + IK); only successful episodes are stored. Features: observation.images.top, observation.images.wrist, observation.state, action, task (exact instruction), episode_index, frame_index, timestamp; success = all stored episodes succeeded (per-seed flags below).\n",
         f"Total: {report['total_episodes']} episodes, {report['total_frames']} frames, {report['wall_time_s']} s, peak RSS {report['resources']['peak_rss_gb']} GB.\n",
         "| task | kept | attempted | oracle success rate | failure reasons |", "|---|---:|---:|---:|---|"]
    for t, v in report["tasks"].items():
        rs: dict = {}
        for x in v["attempts"]:
            if not x["success"]:
                k = (x["failure_reason"] or "").split(":")[0]
                rs[k] = rs.get(k, 0) + 1
        L.append(f"| {t} | {v['kept']} | {v['attempted']} | {v['oracle_success_rate']:.0%} | {', '.join(f'{k} ({n})' for k, n in rs.items())} |")
    (ROOT / "reports" / f"{a.report_name}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
