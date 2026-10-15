#!/usr/bin/env python
"""Run the scripted oracle on simulated tasks: proves each task is solvable in the MuJoCo sim at train / val / test
(unseen) object positions, and exercises reset / observation / action / success / failure_reason end to end.

Usage:
    python scripts/run_oracle.py --task task_01 --episodes 20                 # all splits, profile 'position'
    python scripts/run_oracle.py --task all --profile fixed --episodes 1
    python scripts/run_oracle.py --task task_01 --episodes 1 --video          # MP4 of top+wrist cameras
Writes reports/oracle/<task>_<profile>.json and reports/oracle/summary.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

from src.simulation.env import SO101Env  # noqa: E402
from src.simulation.oracle import Oracle  # noqa: E402
from src.simulation.tasks import TASKS  # noqa: E402
from src.utils.resources import ResourceMonitor  # noqa: E402


def run_task(task_id: str, profile: str, splits: list[str], episodes: int, seed0: int, video: bool) -> dict:
    out = {"task": task_id, "profile": profile, "splits": {}}
    for split in splits:
        env = SO101Env(task_id, split=split, profile=profile, render=video)
        rows = []
        n = 1 if profile == "fixed" else episodes
        for k in range(n):
            seed = seed0 + k
            obs, info0 = env.reset(seed=seed)
            frames = []
            cb = (lambda o, a, i: frames.append(o)) if video else None
            t0 = time.time()
            try:
                ok, info, steps = Oracle(env).run(env.task.oracle_plan(env), obs_cb=cb)
                err = None
            except RuntimeError as exc:
                ok, info, steps, err = False, {"failure_reason": f"oracle_error: {exc}"}, env.t, str(exc)
            rows.append({
                "seed": seed, "success": bool(ok), "steps": int(steps), "failure_reason": info.get("failure_reason"),
                "unmet": info.get("unmet"), "collision_steps": info.get("collision_steps"), "layout": info0["layout"],
                "layout_split": info0["split_of_layout"], "sim_ms_per_step": (time.time() - t0) / max(steps, 1) * 1000, "oracle_error": err,
            })
            if video and frames:
                save_video(frames, ROOT / "reports" / "videos" / f"oracle_{task_id}_{profile}_{split}_seed{seed}.mp4", env.task.instruction, ok)
        env.close()
        succ = sum(r["success"] for r in rows)
        out["splits"][split] = {"episodes": len(rows), "success": succ, "success_rate": succ / max(len(rows), 1), "rows": rows}
        print(f"[oracle] {task_id} {profile} {split}: {succ}/{len(rows)}", flush=True)
    return out


def save_video(frames: list[dict], path: Path, instruction: str, success: bool) -> None:
    import av
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264" if "libx264" in av.codecs_available else "mpeg4", rate=30)
        s.width, s.height, s.pix_fmt = 1120, 480, "yuv420p"
        for o in frames:
            canvas = Image.new("RGB", (1120, 480))
            canvas.paste(Image.fromarray(o["observation.images.top"]), (0, 60))
            canvas.paste(Image.fromarray(o["observation.images.wrist"]).resize((480, 360)), (640, 60))
            d = ImageDraw.Draw(canvas)
            d.text((8, 8), instruction[:150], fill=(255, 255, 255))
            st = o["observation.state"]
            d.text((8, 440), "state " + " ".join(f"{v:6.1f}" for v in st), fill=(200, 200, 200))
            for p in s.encode(av.VideoFrame.from_image(canvas)):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    print(f"[oracle] wrote {path} (success={success})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="all")
    ap.add_argument("--profile", default="position", choices=["fixed", "position", "position_visual", "position_wide"])
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--outdir", default=None, help="report directory (default reports/oracle)")
    args = ap.parse_args()
    tasks = sorted(TASKS) if args.task == "all" else args.task.split(",")
    outdir = Path(args.outdir) if args.outdir else ROOT / "reports" / "oracle"
    outdir.mkdir(parents=True, exist_ok=True)
    mon = ResourceMonitor()
    with mon:
        results = [run_task(t, args.profile, args.splits.split(","), args.episodes, args.seed0, args.video) for t in tasks]
    for r in results:
        r["resources"] = mon.summary()
        (outdir / f"{r['task']}_{args.profile}.json").write_text(json.dumps(r, indent=2))
    # summary over all result files present
    L = ["# Oracle (scripted expert) success in MuJoCo\n", "Proves task solvability at train / val / test (unseen) positions. Not a policy result.\n",
         "| task | profile | train | val | test | main failure reasons |", "|---|---|---|---|---|---|"]
    for f in sorted(outdir.glob("task_*.json")):
        r = json.loads(f.read_text())
        cells = []
        reasons: dict = {}
        for sp in ("train", "val", "test"):
            x = r["splits"].get(sp)
            cells.append(f"{x['success']}/{x['episodes']}" if x else "-")
            for row in (x or {}).get("rows", []):
                if not row["success"]:
                    key = (row["failure_reason"] or f"unsolved: {row.get('unmet') or ''}").split("(")[0].strip()
                    key = key if key.startswith("unsolved") else key.split(":")[0]
                    reasons[key] = reasons.get(key, 0) + 1
        L.append(f"| {r['task']} | {r['profile']} | {' | '.join(cells)} | {', '.join(f'{k} ({v})' for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]))} |")
    L.append(f"\nPeak RSS of the last run: {mon.summary()['peak_rss_gb']} GB")
    (outdir / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
