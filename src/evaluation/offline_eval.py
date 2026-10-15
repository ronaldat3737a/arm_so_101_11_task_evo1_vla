"""Offline action-quality evaluation on REAL episodes (held-out val/test of reports/split_manifest.json; train only as an
underfitting diagnostic).

For sampled frames of each held-out episode, the policy predicts an action chunk from the real observation (both
cameras + state + instruction). The chunk is compared with the recorded future actions, in canonical units
(degrees / gripper 0..100) after the policy's own postprocessor.

Metrics: action MAE (all joints), per-joint MAE, gripper MAE, cosine similarity of action-delta vectors, first-step MAE,
temporal smoothness of the predicted chunk (mean |second difference|), inference latency.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from src.evaluation.policy_eval import load_policy
from src.utils.resources import ResourceMonitor

ROOT = Path(__file__).resolve().parents[2]
AGGR_ROOT = ROOT / "datasets" / "so101_11tasks"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def evaluate_offline(checkpoint: str, tasks: list[str], split: str = "test", frames_per_episode: int = 8, horizon: int | None = None, device: str = "cuda", overrides: dict | None = None, quant: str | None = None, compile: bool = False,
                     dataset_root: str | None = None, episodes_per_task: int | None = None) -> dict:
    """dataset_root (optional): evaluate on another LeRobotDataset with the same schema (e.g. generated sim data):
    its first `episodes_per_task` episodes of each task (by task_index order) are used instead of the real split."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    sm = json.loads((ROOT / "reports" / "split_manifest.json").read_text())
    mon = ResourceMonitor()
    with mon:
        policy, pre, post, cfg = load_policy(checkpoint, device, overrides)
        quant_info = None
        if quant:
            from src.quantization.torchao_quant import apply

            quant_info = apply(policy, quant, compile)
        chunk = int(getattr(cfg, "chunk_size", 1))
        H = min(horizon or chunk, chunk)
        dev = torch.device(device)
        per_task = {}
        lat = []
        root = Path(dataset_root) if dataset_root else AGGR_ROOT
        if dataset_root:
            from src.simulation.tasks import INSTRUCTIONS

            tmeta = LeRobotDataset("local/eval", root=root, download_videos=False, video_backend="pyav").meta
            ep_task = {int(e): tmeta.episodes["tasks"][e][0] for e in range(tmeta.total_episodes)}
        for t in tasks:
            if dataset_root:
                eps = [e for e, tt in ep_task.items() if tt == INSTRUCTIONS[t]][: episodes_per_task or None]
            else:
                s = sm["tasks"][t]
                eps = [s["global_episode_offset"] + e for e in s[split]]
            ds = LeRobotDataset("local/eval" if dataset_root else "local/so101_11tasks", root=root, episodes=eps, download_videos=False,
                                video_backend="pyav", delta_timestamps={"action": [i / 30 for i in range(H)]})
            errs, first, grip, cos, smooth, hold = [], [], [], [], [], []
            a2r = ds.absolute_to_relative_idx  # episode-filtered dataset: absolute frame index -> row
            for ge in eps:
                lo = int(ds.meta.episodes["dataset_from_index"][ge])
                hi = int(ds.meta.episodes["dataset_to_index"][ge])
                # map absolute dataset index -> index inside the episode-filtered dataset
                idxs = np.linspace(lo, hi - H - 1, frames_per_episode).astype(int) if hi - lo > H + 1 else [lo]
                for gi in idxs:
                    k = a2r[int(gi)] if a2r else int(gi)
                    item = ds[int(k)]
                    gt = item["action"].numpy()[:H]
                    pad = item.get("action_is_pad")
                    valid = ~pad.numpy()[:H] if pad is not None else np.ones(H, bool)
                    obs = {key: v.unsqueeze(0).to(dev) for key, v in item.items() if key.startswith("observation.")}
                    obs["task"] = [item["task"]]
                    obs["robot_type"] = ["so_follower"]
                    t0 = time.perf_counter()
                    with torch.inference_mode():
                        batch = pre(obs)
                        pred = policy.predict_action_chunk(batch)[:, :H]
                        pred = torch.stack([post(pred[:, i]) for i in range(pred.shape[1])], dim=1)
                    if dev.type == "cuda":
                        torch.cuda.synchronize()
                    lat.append((time.perf_counter() - t0) * 1000)
                    p = pred[0].float().cpu().numpy()[:, :6]
                    g = gt[valid]
                    pv = p[valid]
                    errs.append(np.abs(pv - g).mean(axis=0))
                    first.append(np.abs(pv[0] - g[0]))
                    grip.append(np.abs(pv[:, 5] - g[:, 5]).mean())
                    state = item["observation.state"].numpy().reshape(-1)[:6]
                    hold.append(np.abs(state[None, :] - g).mean())  # reference: predict "stay where you are"
                    dp, dg = (pv - state).reshape(-1), (g - state).reshape(-1)
                    cos.append(float(dp @ dg / (np.linalg.norm(dp) * np.linalg.norm(dg) + 1e-8)))
                    smooth.append(np.abs(np.diff(p, n=2, axis=0)).mean() if len(p) > 2 else 0.0)
            e = np.asarray(errs)
            per_task[t] = {
                "samples": len(errs),
                "action_mae": float(e.mean()),
                "joint_mae": dict(zip(JOINTS, e.mean(axis=0).round(3).tolist(), strict=True)),
                "first_step_mae": float(np.mean(first)),
                "gripper_mae": float(np.mean(grip)),
                "delta_cosine": float(np.mean(cos)),
                "pred_smoothness": float(np.mean(smooth)),
                "hold_state_baseline_mae": float(np.mean(hold)),
            }
            print(f"[offline] {t} {split}: action MAE {per_task[t]['action_mae']:.2f} deg (hold-state baseline {per_task[t]['hold_state_baseline_mae']:.2f}), gripper {per_task[t]['gripper_mae']:.2f}, cos {per_task[t]['delta_cosine']:.3f}", flush=True)
    allm = [v["action_mae"] for v in per_task.values()]
    return {
        "checkpoint": str(checkpoint), "dataset_root": str(root), "overrides": overrides or {}, "quantization": quant_info, "policy_type": cfg.type, "split": split, "horizon": H, "frames_per_episode": frames_per_episode,
        "per_task": per_task, "mean_action_mae": float(np.mean(allm)),
        "mean_hold_state_baseline_mae": float(np.mean([v["hold_state_baseline_mae"] for v in per_task.values()])),
        "latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)), "n": len(lat)},
        "resources": mon.summary(),
    }
