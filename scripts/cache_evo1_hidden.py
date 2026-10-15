#!/usr/bin/env python
"""Cache the hidden states entering language-model layer `--start` of Evo-1's frozen InternVL3 embedder, for the
stage-2-lite `selective_llm` profile (src/training/selective_llm.py).

Same frames, preprocessing and per-sample targets as scripts/cache_evo1_features.py (state / actions / masks from the
official `_prepare_state` / `_prepare_actions` after the checkpoint's preprocessor). Per sample it stores the valid
tokens' hidden states (bf16) and their original position ids instead of the final fused tokens.

Exactness check: for the first `--verify-batches` batches of every part, the cached hidden states are pushed through
layers start..end + norm (`run_tail`) and compared with the official `_compute_fused_tokens` output of the same
forward pass. The result is stored in cache_meta.json; the job stops if the relative error exceeds `--tol`.

    python scripts/cache_evo1_hidden.py --checkpoint checkpoints/evo1_so101/best --start 10 --stride 10 --offset 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cache_evo1_features import frame_list  # noqa: E402
from src.evaluation.policy_eval import load_policy  # noqa: E402
from src.training.launcher import AGGR_ROOT  # noqa: E402
from src.training.selective_llm import capture_layer_input, run_tail  # noqa: E402
from src.utils.resources import ResourceMonitor  # noqa: E402


def main() -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "evo1_so101" / "best"))
    ap.add_argument("--start", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--val-stride", type=int, default=30)
    ap.add_argument("--shard-size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--verify-batches", type=int, default=4)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--parts", default="val,train", help="which parts to build, in order (val, train, sim)")
    ap.add_argument("--sim-label", default="sim", help="part name prefix of the sim part (e.g. simv2 for the v2 scene data)")
    ap.add_argument("--sim-root", default=str(ROOT / "datasets" / "sim_so101_all"), help="oracle sim dataset (all episodes are train-region)")
    a = ap.parse_args()

    pol, pre, _, cfg = load_policy(a.checkpoint, "cuda")
    pol.eval()
    store, _h = capture_layer_input(pol, a.start)
    chunk = cfg.chunk_size
    sm = json.loads((ROOT / "reports" / "split_manifest.json").read_text())
    ds = LeRobotDataset("local/so101_11tasks", root=AGGR_ROOT, video_backend="pyav", download_videos=False,
                        delta_timestamps={"action": [i / 30 for i in range(chunk)]})
    out = Path(a.out or ROOT / "datasets" / f"evo1_hidden_cache_L{a.start}")
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "cache_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {
        "checkpoint": str(a.checkpoint), "start_layer": a.start, "vlm": cfg.vlm_model_name, "vlm_num_layers": cfg.vlm_num_layers,
        "image_resolution": list(cfg.image_resolution), "chunk_size": chunk, "parts": {}, "verification": {}}
    all_jobs = {"train": ("train", [t["global_episode_offset"] + e for t in sm["tasks"].values() for e in t["train"]], a.stride, a.offset),
                "val": ("val", sorted(sm["global"]["val"]), a.val_stride, 0)}
    datasets = {"train": ds, "val": ds}
    if "sim" in a.parts.split(","):
        sim_ds = LeRobotDataset("local/sim_so101_all", root=a.sim_root, video_backend="pyav", download_videos=False,
                                delta_timestamps={"action": [i / 30 for i in range(chunk)]})
        all_jobs["sim"] = (a.sim_label, list(range(sim_ds.meta.total_episodes)), a.stride, a.offset)
        datasets["sim"] = sim_ds
    mon = ResourceMonitor()
    with mon:
        for key in a.parts.split(","):
            split, eps, stride, offset = all_jobs[key]
            dsk = datasets[key]
            part = f"{split}_s{stride}_o{offset}"
            frames = frame_list(dsk, eps, stride, offset)
            pdir = out / part
            pdir.mkdir(exist_ok=True)
            n_shards = (len(frames) + a.shard_size - 1) // a.shard_size
            meta["parts"][part] = {"split": split, "stride": stride, "offset": offset, "frames": len(frames), "shards": n_shards}
            meta_path.write_text(json.dumps(meta, indent=2))
            t0 = time.time()
            done_before = sum((pdir / f"shard_{k:05d}.safetensors").exists() for k in range(n_shards))
            verified = 0
            for k in range(n_shards):
                fpath = pdir / f"shard_{k:05d}.safetensors"
                if fpath.exists():
                    continue
                sel = frames[k * a.shard_size : (k + 1) * a.shard_size]
                hs, ps, lens, states, acts, amasks, epi, fidx, tasks = [], [], [], [], [], [], [], [], []
                for j in range(0, len(sel), a.batch):
                    items = [dsk[i] for _, i in sel[j : j + a.batch]]
                    batch = torch.utils.data.default_collate(items)
                    batch = {kk: (v.cuda() if torch.is_tensor(v) else v) for kk, v in batch.items()}
                    with torch.inference_mode():
                        b = pre(batch)
                        imgs, imask = pol._collect_image_batches(b)
                        st, _sm = pol._prepare_state(b)
                        ac, amask = pol._prepare_actions(b)
                        with pol._maybe_autocast():
                            fused, cmask = pol._compute_fused_tokens(pol._normalize_task_batch(b), imgs, imask)
                        h, pos = store["h"], store["pos"].expand(fused.shape[0], -1)
                        valid = cmask.bool()
                        if verified < a.verify_batches:
                            with pol._maybe_autocast():
                                rec = run_tail(pol, h, pos, valid, a.start).float()
                            ref = fused.float()
                            err = float(((rec - ref)[valid]).norm() / ref[valid].norm())
                            v = meta["verification"].setdefault(part, [])
                            v.append(round(err, 6))
                            print(json.dumps({"verify": part, "batch": verified, "rel_err": err}), flush=True)
                            if err > a.tol:
                                meta_path.write_text(json.dumps(meta, indent=2))
                                print(f"STOP: reconstruction error {err:.4f} > tol {a.tol}")
                                return 1
                            verified += 1
                    for r in range(h.shape[0]):
                        hs.append(h[r][valid[r]].to(torch.bfloat16).cpu())
                        ps.append(pos[r][valid[r]].to(torch.int32).cpu())
                        lens.append(int(valid[r].sum()))
                    states.append(st.float().cpu())
                    acts.append(ac.float().cpu())
                    amasks.append(amask.cpu())
                    epi += [int(x) for x in batch["episode_index"].cpu()]
                    fidx += [int(x) for x in batch["frame_index"].cpu()]
                    tasks += [int(x) for x in batch["task_index"].cpu()]
                tensors = {
                    "hidden": torch.cat(hs), "position_ids": torch.cat(ps), "lengths": torch.tensor(lens, dtype=torch.int32),
                    "state": torch.cat(states), "actions": torch.cat(acts), "action_mask": torch.cat(amasks),
                    "episode_index": torch.tensor(epi, dtype=torch.int32), "frame_index": torch.tensor(fidx, dtype=torch.int32),
                    "task_index": torch.tensor(tasks, dtype=torch.int32),
                }
                tmp = fpath.with_suffix(".tmp")
                save_file(tensors, str(tmp))
                os.replace(tmp, fpath)
                done = k + 1 - done_before
                rate = done * a.shard_size / max(time.time() - t0, 1e-6)
                print(json.dumps({"part": part, "shard": k + 1, "of": n_shards, "frames_per_s": round(rate, 2),
                                  "eta_h": round((n_shards - k - 1) * a.shard_size / max(rate, 1e-6) / 3600, 2), "rss_gb": round(mon._rss(), 2)}), flush=True)
            meta_path.write_text(json.dumps(meta, indent=2))
    meta["resources"] = mon.summary()
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
