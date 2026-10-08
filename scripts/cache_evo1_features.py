#!/usr/bin/env python
"""Cache the frozen-VLM fused tokens of Evo-1 to disk for fast stage-1 training of the action head.

In stage 1 the official Evo-1 policy runs the InternVL3 embedder in eval mode under no_grad, and our pipeline has no
image augmentation. The fused tokens of a frame are therefore a deterministic function of (images, instruction) and can
be computed once. Everything stored is produced by the official policy methods, i.e. exactly the tensors
`Evo1Policy.forward` feeds to the action head:
  fused tokens  <- Evo1Policy._compute_fused_tokens   (only the valid tokens are kept; the head's cross-attention has
                                                       no positional encoding, so dropping padded keys is exact)
  state, mask   <- Evo1Policy._prepare_state
  actions, mask <- Evo1Policy._prepare_actions         (normalized, padded to 24-D, action_is_pad applied)
after the checkpoint's own preprocessor (train-only MIN_MAX stats).

Frames: train episodes every `--stride` frames starting at `--offset` (a later run with another offset adds frames
without recomputing). Val episodes every `--val-stride` frames. Shards of `--shard-size` samples are written
atomically; finished shards are skipped, so the job is resumable.

    python scripts/cache_evo1_features.py --checkpoint checkpoints/evo1_so101/best --stride 10 --offset 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.policy_eval import load_policy  # noqa: E402
from src.training.launcher import AGGR_ROOT  # noqa: E402
from src.utils.resources import ResourceMonitor  # noqa: E402


def frame_list(ds, episodes: list[int], stride: int, offset: int) -> list[tuple[int, int]]:
    out = []
    for ge in episodes:
        lo, hi = int(ds.meta.episodes["dataset_from_index"][ge]), int(ds.meta.episodes["dataset_to_index"][ge])
        out += [(ge, i) for i in range(lo + offset, hi, stride)]
    return out


def main() -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "evo1_so101" / "best"))
    ap.add_argument("--out", default=str(ROOT / "datasets" / "evo1_feature_cache"))
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--val-stride", type=int, default=30)
    ap.add_argument("--shard-size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    a = ap.parse_args()

    pol, pre, _, cfg = load_policy(a.checkpoint, "cuda")
    pol.eval()
    chunk = cfg.chunk_size
    sm = json.loads((ROOT / "reports" / "split_manifest.json").read_text())
    ds = LeRobotDataset("local/so101_11tasks", root=AGGR_ROOT, video_backend="pyav", download_videos=False,
                        delta_timestamps={"action": [i / 30 for i in range(chunk)]})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "cache_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {
        "checkpoint": str(a.checkpoint), "vlm": cfg.vlm_model_name, "image_resolution": list(cfg.image_resolution), "chunk_size": chunk,
        "max_state_dim": cfg.max_state_dim, "max_action_dim": cfg.max_action_dim, "parts": {}}
    jobs = [("train", [t["global_episode_offset"] + e for t in sm["tasks"].values() for e in t["train"]], a.stride, a.offset),
            ("val", sorted(sm["global"]["val"]), a.val_stride, 0)]
    mon = ResourceMonitor()
    with mon:
        for split, eps, stride, offset in jobs:
            part = f"{split}_s{stride}_o{offset}"
            frames = frame_list(ds, eps, stride, offset)
            pdir = out / part
            pdir.mkdir(exist_ok=True)
            n_shards = (len(frames) + a.shard_size - 1) // a.shard_size
            meta["parts"][part] = {"split": split, "stride": stride, "offset": offset, "frames": len(frames), "shards": n_shards}
            meta_path.write_text(json.dumps(meta, indent=2))
            t0 = time.time()
            done_before = sum((pdir / f"shard_{k:05d}.safetensors").exists() for k in range(n_shards))
            for k in range(n_shards):
                fpath = pdir / f"shard_{k:05d}.safetensors"
                if fpath.exists():
                    continue
                sel = frames[k * a.shard_size : (k + 1) * a.shard_size]
                toks, lens, states, smasks, acts, amasks, epi, fidx, tasks = [], [], [], [], [], [], [], [], []
                for j in range(0, len(sel), a.batch):
                    items = [ds[i] for _, i in sel[j : j + a.batch]]
                    batch = torch.utils.data.default_collate(items)
                    batch = {kk: (v.cuda() if torch.is_tensor(v) else v) for kk, v in batch.items()}
                    with torch.inference_mode():
                        b = pre(batch)
                        imgs, imask = pol._collect_image_batches(b)
                        st, smask = pol._prepare_state(b)
                        ac, amask = pol._prepare_actions(b)
                        with pol._maybe_autocast():
                            fused, cmask = pol._compute_fused_tokens(pol._normalize_task_batch(b), imgs, imask)
                    for r in range(fused.shape[0]):
                        valid = cmask[r].bool() if cmask is not None else torch.ones(fused.shape[1], dtype=torch.bool, device=fused.device)
                        tk = fused[r][valid].to(torch.bfloat16).cpu()
                        toks.append(tk)
                        lens.append(tk.shape[0])
                    states.append(st.float().cpu())
                    smasks.append(smask.cpu())
                    acts.append(ac.float().cpu())
                    amasks.append(amask.cpu())
                    epi += [int(x) for x in batch["episode_index"].cpu()]
                    fidx += [int(x) for x in batch["frame_index"].cpu()]
                    tasks += [int(x) for x in batch["task_index"].cpu()]
                tensors = {
                    "tokens": torch.cat(toks), "lengths": torch.tensor(lens, dtype=torch.int32),
                    "state": torch.cat(states), "state_mask": torch.cat(smasks), "actions": torch.cat(acts), "action_mask": torch.cat(amasks),
                    "episode_index": torch.tensor(epi, dtype=torch.int32), "frame_index": torch.tensor(fidx, dtype=torch.int32),
                    "task_index": torch.tensor(tasks, dtype=torch.int32),
                }
                tmp = fpath.with_suffix(".tmp")
                save_file(tensors, str(tmp))
                os.replace(tmp, fpath)
                done = k + 1 - done_before
                rate = done * a.shard_size / max(time.time() - t0, 1e-6)
                left = (n_shards - k - 1) * a.shard_size / max(rate, 1e-6)
                print(json.dumps({"part": part, "shard": k + 1, "of": n_shards, "frames_per_s": round(rate, 2), "eta_h": round(left / 3600, 2),
                                  "rss_gb": round(mon._rss(), 2)}), flush=True)
    meta["resources"] = mon.summary()
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
