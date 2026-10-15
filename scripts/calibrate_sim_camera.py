#!/usr/bin/env python
"""Visual calibration aid for the simulated cameras: renders the reset scene of a task for candidate camera settings
next to a real frame of the same task (start of a train episode: the arm is in the rest pose in both).

Candidates are JSON dicts merged into the `cameras.<name>` entry of the base sim config, e.g.
    python scripts/calibrate_sim_camera.py --base configs/simulation_v2.yaml --camera top \
        --cand '{"pos":[-0.25,-0.30,0.40],"target":[0.25,0.0,0.0],"fovy":60}' --cand '{...}' --out /tmp/cmp.png
The output stacks [real | candidate 1 | candidate 2 | ...] per task. Nothing is written to the configs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")


def real_frame(task: str, camera: str) -> np.ndarray:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    sm = json.loads((ROOT / "reports" / "split_manifest.json").read_text())
    ge = sm["tasks"][task]["global_episode_offset"] + sm["tasks"][task]["train"][0]
    ds = LeRobotDataset("local/so101_11tasks", root=ROOT / "datasets" / "so101_11tasks", episodes=[ge], video_backend="pyav", download_videos=False)
    return (ds[0][f"observation.images.{camera}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=str(ROOT / "configs" / "simulation_v2.yaml"))
    ap.add_argument("--camera", default="top", choices=["top", "wrist"])
    ap.add_argument("--tasks", default="task_01,task_08")
    ap.add_argument("--cand", action="append", default=[], help="JSON dict merged into cameras.<camera>")
    ap.add_argument("--appearance", default="{}", help="JSON merged into the config's `appearance` (e.g. table_yaw_deg, robot_rgba)")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base = yaml.safe_load(Path(a.base).read_text())
    cands = [json.loads(c) for c in a.cand] or [{}]
    rows = []
    for task in a.tasks.split(","):
        tiles = [("real", real_frame(task, a.camera))]
        for i, c in enumerate(cands):
            cfg = json.loads(json.dumps(base))
            cfg["cameras"][a.camera].update(c)
            cfg["appearance"] = {**(cfg.get("appearance") or {}), **json.loads(a.appearance)}
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
                yaml.safe_dump(cfg, f)
            os.environ["SO101_SIM_CONFIG"] = f.name
            import importlib

            import src.simulation.env as envmod

            importlib.reload(envmod)
            env = envmod.SO101Env(task, split="train", profile="fixed", render=True)
            obs, _ = env.reset(seed=a.seed)
            tiles.append((f"cand {i + 1}", np.asarray(obs[f"observation.images.{a.camera}"])))
            env.close()
            os.unlink(f.name)
        ims = []
        for label, arr in tiles:
            im = Image.fromarray(arr).resize((int(arr.shape[1] * a.scale), int(arr.shape[0] * a.scale)))
            ImageDraw.Draw(im).text((4, 4), f"{task} {label}", fill=(255, 0, 255))
            ims.append(im)
        row = Image.new("RGB", (sum(i.width for i in ims), max(i.height for i in ims)))
        x = 0
        for im in ims:
            row.paste(im, (x, 0))
            x += im.width
        rows.append(row)
    out = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)))
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height
    out.save(a.out)
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
