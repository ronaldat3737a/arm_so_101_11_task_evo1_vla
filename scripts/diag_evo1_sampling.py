#!/usr/bin/env python
"""Diagnostic: how much of Evo-1's offline action error is flow-matching sampling spread vs systematic bias?

For each sampled frame of real episodes, the frozen-VLM fused tokens are computed once (official
`_compute_fused_tokens`) and the action head is sampled N times from independent noise (official `policy.model`, as in
`Evo1Policy.predict_action_chunk`). Reported per task:
  single_sample_mae  mean MAE of one sample (what the robot would execute)
  mean_of_n_mae      MAE of the average of the N samples (approximates the conditional mean)
  sample_spread      mean absolute deviation of the samples around their average
  hold_mae           "stay where you are" reference
plus the error vs. horizon step curve for single samples, mean-of-N and hold.
Evaluation only: model and data are not changed.

    python scripts/diag_evo1_sampling.py --checkpoint checkpoints/evo1_so101/best --split test --frames-per-episode 2 --n 8
Writes reports/diag_sampling_<name>_<split>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.offline_eval import AGGR_ROOT  # noqa: E402
from src.evaluation.policy_eval import load_policy  # noqa: E402
from src.simulation.tasks import TASKS  # noqa: E402
from src.utils.resources import ResourceMonitor  # noqa: E402


@torch.no_grad()
def sample_chunks(pol, batch: dict, n: int) -> torch.Tensor:
    """N action chunks for one observation; same computation as Evo1Policy.predict_action_chunk (no RTC)."""
    prompts = pol._normalize_task_batch(batch)
    imgs, imask = pol._collect_image_batches(batch)
    states, _ = pol._prepare_state(batch)
    emb = pol._get_embodiment_ids(batch, states.shape[0])
    amask = pol._prepare_inference_action_mask(states.shape[0])
    with pol._maybe_autocast():
        fused, cmask = pol._compute_fused_tokens(prompts, imgs, imask)
        rep = lambda x: None if x is None else x.repeat_interleave(n, dim=0)  # noqa: E731
        acts = pol.model(rep(fused), state=rep(states), action_mask=rep(amask), embodiment_ids=rep(emb), context_mask=rep(cmask))
    return acts.view(n, pol.config.chunk_size, pol.config.max_action_dim).float()


def main() -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "evo1_so101" / "best"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--task", default="all")
    ap.add_argument("--frames-per-episode", type=int, default=2)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="evo1_stage1_cached_r2")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    sm = json.loads((ROOT / "reports" / "split_manifest.json").read_text())
    tasks = sorted(TASKS) if a.task == "all" else a.task.split(",")
    mon = ResourceMonitor()
    out = {"checkpoint": a.checkpoint, "split": a.split, "frames_per_episode": a.frames_per_episode, "n_samples": a.n, "per_task": {}}
    curves = {"single": [], "mean_of_n": [], "hold": []}
    with mon:
        pol, pre, post, cfg = load_policy(a.checkpoint, "cuda")
        H = cfg.chunk_size
        for t in tasks:
            s = sm["tasks"][t]
            eps = [s["global_episode_offset"] + e for e in s[a.split]]
            ds = LeRobotDataset("local/so101_11tasks", root=AGGR_ROOT, episodes=eps, download_videos=False, video_backend="pyav",
                                delta_timestamps={"action": [i / 30 for i in range(H)]})
            a2r = ds.absolute_to_relative_idx
            single, meanp, spread, hold = [], [], [], []
            for ge in eps:
                lo, hi = int(ds.meta.episodes["dataset_from_index"][ge]), int(ds.meta.episodes["dataset_to_index"][ge])
                idxs = np.linspace(lo, hi - H - 1, a.frames_per_episode).astype(int) if hi - lo > H + 1 else [lo]
                for gi in idxs:
                    item = ds[int(a2r[int(gi)] if a2r else int(gi))]
                    valid = ~item["action_is_pad"].numpy()[:H]
                    g = item["action"].numpy()[:H][valid]
                    obs = {k: v.unsqueeze(0).cuda() for k, v in item.items() if k.startswith("observation.")}
                    obs["task"], obs["robot_type"] = [item["task"]], ["so_follower"]
                    with torch.inference_mode():
                        chunks = sample_chunks(pol, pre(obs), a.n)
                        p = torch.stack([post(chunks[:, i]) for i in range(H)], dim=1).float().cpu().numpy()[:, :, :6]
                    p = p[:, valid]
                    pm = p.mean(axis=0)
                    st = item["observation.state"].numpy().reshape(-1)[:6]
                    e_single = np.abs(p - g[None]).mean(axis=2)  # (n, T)
                    e_mean = np.abs(pm - g).mean(axis=1)
                    e_hold = np.abs(st[None] - g).mean(axis=1)
                    single.append(e_single.mean())
                    meanp.append(e_mean.mean())
                    spread.append(np.abs(p - pm[None]).mean())
                    hold.append(e_hold.mean())
                    if valid.all():
                        curves["single"].append(e_single.mean(axis=0))
                        curves["mean_of_n"].append(e_mean)
                        curves["hold"].append(e_hold)
            out["per_task"][t] = {"samples": len(single), "single_sample_mae": float(np.mean(single)), "mean_of_n_mae": float(np.mean(meanp)),
                                  "sample_spread": float(np.mean(spread)), "hold_mae": float(np.mean(hold))}
            r = out["per_task"][t]
            print(f"[diag] {t} {a.split}: single {r['single_sample_mae']:.2f}  mean-of-{a.n} {r['mean_of_n_mae']:.2f}  spread {r['sample_spread']:.2f}  hold {r['hold_mae']:.2f}", flush=True)
    for k in ("single_sample_mae", "mean_of_n_mae", "sample_spread", "hold_mae"):
        out[f"mean_{k}"] = float(np.mean([v[k] for v in out["per_task"].values()]))
    steps = [0, 4, 9, 19, 29, 39, 49]
    out["error_vs_horizon_step"] = {"steps": steps, **{k: [float(np.mean(v, axis=0)[i]) for i in steps] for k, v in curves.items()}}
    out["resources"] = mon.summary()
    path = ROOT / "reports" / f"diag_sampling_{a.name}_{a.split}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_task"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
