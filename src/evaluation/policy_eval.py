"""Closed-loop evaluation of a LeRobot policy checkpoint in the MuJoCo SO-101 simulation.

Inference goes through the official `lerobot.common.control_utils.predict_action` (observation -> tensors ->
policy preprocessor -> select_action -> postprocessor), i.e. the same path `lerobot-record --policy.path` uses on the
real robot. The simulator receives the resulting canonical action (degrees, gripper 0..100) unchanged.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from src.simulation.env import SO101Env
from src.utils.resources import ResourceMonitor


def load_policy(checkpoint: str | Path, device: str = "cuda", overrides: dict | None = None):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    checkpoint = str(checkpoint)
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.device = device
    for k, v in (overrides or {}).items():
        setattr(cfg, k, v)
    policy = get_policy_class(cfg.type).from_pretrained(checkpoint, config=cfg)
    policy.to(device)
    policy.eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=checkpoint, preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post, cfg


def run_episode(env: SO101Env, policy, pre, post, device: torch.device, use_amp: bool, seed: int, frames: list | None = None) -> dict:
    from lerobot.common.control_utils import predict_action

    obs, info0 = env.reset(seed=seed)
    policy.reset()
    lat, actions = [], []
    info = info0
    done = False
    while not done:
        o = {k: v for k, v in obs.items() if k.startswith("observation.")}
        t0 = time.perf_counter()
        a = predict_action(o, policy, device, pre, post, use_amp, task=obs["task"], robot_type="so_follower")
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
        a = a.squeeze(0).float().cpu().numpy()[:6]
        actions.append(a)
        obs, _, term, trunc, info = env.step(a)
        if frames is not None:
            frames.append(obs)
        done = term or trunc
    acts = np.asarray(actions)
    jerk = np.abs(np.diff(acts, n=2, axis=0)).mean(axis=0) if len(acts) > 2 else np.zeros(6)
    return {
        "seed": seed,
        "success": bool(info["success"]),
        "failure_reason": info["failure_reason"],
        "timeout": bool(info["failure_reason"] and info["failure_reason"].startswith("timeout")),
        "steps": int(env.t),
        "collision_steps": int(info["collision_steps"]),
        "layout": info0["layout"],
        "layout_split": info0["split_of_layout"],
        "latency_ms": lat,
        "action_second_diff_mean": jerk.round(4).tolist(),
        "action_min": acts.min(0).round(2).tolist(),
        "action_max": acts.max(0).round(2).tolist(),
    }


def evaluate(checkpoint: str, tasks: list[str], splits: list[str], profile: str, episodes: int, seed0: int = 1000,
             device: str = "cuda", overrides: dict | None = None, video_dir: Path | None = None, name: str = "policy") -> dict:
    mon = ResourceMonitor()
    with mon:
        policy, pre, post, cfg = load_policy(checkpoint, device, overrides)
        dev = torch.device(device)
        use_amp = bool(getattr(cfg, "use_amp", False))
        results = {}
        for t in tasks:
            results[t] = {}
            for sp in splits:
                env = SO101Env(t, split=sp, profile=profile, render=True)
                rows = []
                for k in range(episodes):
                    frames = [] if (video_dir and k == 0) else None
                    rows.append(run_episode(env, policy, pre, post, dev, use_amp, seed0 + k, frames))
                    if frames:
                        from scripts.run_oracle import save_video

                        save_video(frames, video_dir / f"{name}_{t}_{profile}_{sp}_seed{seed0 + k}.mp4", env.task.instruction, rows[-1]["success"])
                env.close()
                results[t][sp] = rows
                s = sum(r["success"] for r in rows)
                print(f"[eval] {t} {profile} {sp}: {s}/{len(rows)}", flush=True)
    all_lat = [x for t in results.values() for rows in t.values() for r in rows for x in r["latency_ms"]]
    return {
        "checkpoint": str(checkpoint),
        "policy_type": cfg.type,
        "profile": profile,
        "episodes_per_split": episodes,
        "seed0": seed0,
        "results": results,
        "latency_ms": {"p50": float(np.percentile(all_lat, 50)), "p95": float(np.percentile(all_lat, 95)), "max": float(np.max(all_lat)), "n": len(all_lat)} if all_lat else None,
        "resources": mon.summary(),
    }


def to_markdown(rep: dict) -> str:
    L = [f"# Simulation evaluation: `{rep['checkpoint']}`\n", f"Policy: {rep['policy_type']}; profile: {rep['profile']}; {rep['episodes_per_split']} episodes per split; seeds from {rep['seed0']}.\n"]
    if rep["latency_ms"]:
        lt = rep["latency_ms"]
        L.append(f"Inference latency per step (incl. pre/post-processing): P50 {lt['p50']:.1f} ms, P95 {lt['p95']:.1f} ms, max {lt['max']:.1f} ms (n={lt['n']}). Peak RSS {rep['resources']['peak_rss_gb']} GB, peak VRAM {rep['resources']['peak_vram_alloc_gb']} GB.\n")
    L += ["| Task | Split | Episodes | Success | Success Rate | Timeout | Collision eps | Main failure reasons |", "|---|---|---:|---:|---:|---:|---:|---|"]
    tot = {"train": [0, 0], "val": [0, 0], "test": [0, 0]}
    for t, splits in rep["results"].items():
        for sp, rows in splits.items():
            s = sum(r["success"] for r in rows)
            to = sum(r["timeout"] for r in rows)
            co = sum(r["collision_steps"] > 0 for r in rows)
            reasons: dict = {}
            for r in rows:
                if not r["success"]:
                    k = (r["failure_reason"] or "").split(":")[0]
                    reasons[k] = reasons.get(k, 0) + 1
            tot.setdefault(sp, [0, 0])
            tot[sp][0] += s
            tot[sp][1] += len(rows)
            L.append(f"| {t} | {sp} | {len(rows)} | {s} | {s / max(len(rows), 1):.0%} | {to} | {co} | {', '.join(f'{k} ({v})' for k, v in reasons.items())} |")
    L.append("\n## Overall\n")
    for sp, (s, n) in tot.items():
        if n:
            L.append(f"- {sp}: {s}/{n} = {s / n:.0%}" + ("  ← unseen-position success rate" if sp == "test" else ""))
    L.append("\nNote: 'collision eps' counts episodes with any robot–table contact, including the resting gripper at the start pose.")
    return "\n".join(L) + "\n"
