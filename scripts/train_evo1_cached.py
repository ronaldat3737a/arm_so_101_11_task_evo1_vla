#!/usr/bin/env python
"""Stage-1 Evo-1 training on cached frozen-VLM features (scripts/cache_evo1_features.py).

The loss is the official one, computed exactly as `Evo1Policy.forward` does after the fused-token step:
    pred_velocity, noise = policy.model(fused, state, actions_gt, action_mask, embodiment_ids, context_mask)
    target = (actions_gt - noise) * mask;  loss = policy._compute_masked_loss(...)
Only the trainable parameters of stage 1 (integration + action head) are optimized, with the policy's own optimizer and
scheduler presets. Checkpoints are full LeRobot Evo-1 policies (save_pretrained incl. the unchanged VLM + processors),
so evaluation / deployment use the normal image path. Resumable (weights, optimizer, scheduler, RNG, shard position).

    python scripts/train_evo1_cached.py --init checkpoints/evo1_so101/best --cache datasets/evo1_feature_cache \
        --parts train_s10_o0 --steps 20000 --batch 32 --lr 1e-5
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.policy_eval import load_policy  # noqa: E402
from src.training.launcher import environment_info  # noqa: E402
from src.utils.resources import ResourceMonitor  # noqa: E402


class ShardStream:
    """Random order over shards; one shard (~256 MB) in RAM at a time; samples shuffled within the shard."""

    def __init__(self, files: list[Path], seed: int, collate_fn=None):
        self.files, self.seed = files, seed
        self.collate_fn = collate_fn or collate
        self.epoch, self.shard_pos, self.sample_pos = 0, 0, 0
        self._load()

    def _order(self):
        g = np.random.default_rng(self.seed + self.epoch)
        return g.permutation(len(self.files))

    def _load(self):
        f = self.files[self._order()[self.shard_pos]]
        d = load_file(str(f))
        off = np.concatenate([[0], np.cumsum(d["lengths"].numpy())])
        self.cur = d
        self.off = off
        self.perm = np.random.default_rng(self.seed * 7919 + self.epoch * 1000 + self.shard_pos).permutation(len(d["lengths"]))

    def next_batch(self, n: int) -> dict:
        idx = []
        while len(idx) < n:
            if self.sample_pos >= len(self.perm):
                self.shard_pos += 1
                self.sample_pos = 0
                if self.shard_pos >= len(self.files):
                    self.epoch += 1
                    self.shard_pos = 0
                self._load()
            take = min(n - len(idx), len(self.perm) - self.sample_pos)
            idx += [(self.cur, self.off, int(j)) for j in self.perm[self.sample_pos : self.sample_pos + take]]
            self.sample_pos += take
        return self.collate_fn(idx)

    def state(self):
        return {"epoch": self.epoch, "shard_pos": self.shard_pos, "sample_pos": self.sample_pos}

    def restore(self, st):
        self.epoch, self.shard_pos, self.sample_pos = st["epoch"], st["shard_pos"], st["sample_pos"]
        self._load()


def collate(items) -> dict:
    lens = [int(off[j + 1] - off[j]) for d, off, j in items]
    L = max(lens)
    B = len(items)
    tok = torch.zeros(B, L, items[0][0]["tokens"].shape[1], dtype=torch.bfloat16)
    cmask = torch.zeros(B, L, dtype=torch.bool)
    for r, (d, off, j) in enumerate(items):
        tok[r, : lens[r]] = d["tokens"][off[j] : off[j + 1]]
        cmask[r, : lens[r]] = True
    pick = lambda k: torch.stack([d[k][j] for d, off, j in items])  # noqa: E731
    return {"tokens": tok, "context_mask": cmask, "state": pick("state"), "actions": pick("actions"), "action_mask": pick("action_mask")}


def loss_on(pol, b: dict, device) -> torch.Tensor:
    """Official Evo1Policy.forward computation after _compute_fused_tokens (modeling_evo1.py)."""
    dt_ = pol._compute_dtype
    fused = b["tokens"].to(device=device, dtype=dt_)
    states = b["state"].to(device=device, dtype=dt_)
    actions = b["actions"].to(device=device, dtype=dt_)
    amask = b["action_mask"].to(device=device)
    cmask = b["context_mask"].to(device=device)
    emb = torch.zeros(fused.shape[0], dtype=torch.long, device=device)
    with pol._maybe_autocast():
        pv, noise = pol.model(fused, state=states, actions_gt=actions, action_mask=amask.to(dtype=dt_), embodiment_ids=emb, context_mask=cmask)
    pv, noise = pv.float(), noise.float()
    flat = amask.view(amask.shape[0], -1).to(torch.float32)
    target = (actions.float() - noise).view(actions.shape[0], -1) * flat
    return pol._compute_masked_loss(pv, target, amask, "mean")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default=str(ROOT / "checkpoints" / "evo1_so101" / "best"))
    ap.add_argument("--cache", default=str(ROOT / "datasets" / "evo1_feature_cache"))
    ap.add_argument("--parts", default="train_s10_o0")
    ap.add_argument("--val-part", default="val_s30_o0")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--name", default="evo1_stage1_cached")
    ap.add_argument("--export", default="evo1_so101")
    ap.add_argument("--resume", help="run dir to continue")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    run_dir = Path(a.resume).resolve() if a.resume else ROOT / "outputs" / f"{a.name}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    if a.resume:
        a = argparse.Namespace(**{**vars(a), **json.loads((run_dir / "args.json").read_text()), "resume": a.resume})
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps({k: v for k, v in vars(a).items() if k != "resume"}, indent=2))

    pol, pre, post, cfg = load_policy(a.init, "cuda")
    cfg.optimizer_lr = a.lr
    cfg.scheduler_warmup_steps = a.warmup
    pol.train()
    device = torch.device("cuda")
    params = pol.get_optim_params()
    n_train = sum(p.numel() for g in params for p in g["params"])
    opt = cfg.get_optimizer_preset().build(params)
    sched = cfg.get_scheduler_preset().build(opt, a.steps)
    files = sorted(f for p in a.parts.split(",") for f in (Path(a.cache) / p).glob("shard_*.safetensors"))
    val_files = sorted((Path(a.cache) / a.val_part).glob("shard_*.safetensors"))
    if not files or not val_files:
        print(f"STOP: no cached shards in {a.cache}/{a.parts} or {a.val_part}")
        return 1
    stream = ShardStream(files, a.seed)
    val_batches = []
    for f in val_files:
        d = load_file(str(f))
        off = np.concatenate([[0], np.cumsum(d["lengths"].numpy())])
        idx = [(d, off, j) for j in range(len(d["lengths"]))]
        val_batches += [collate(idx[i : i + a.batch]) for i in range(0, len(idx), a.batch)]
    clip = cfg.optimizer_grad_clip_norm
    step, best, log = 0, math.inf, []
    if a.resume:
        from safetensors.torch import load_file as lf

        sd = lf(str(run_dir / "checkpoints" / "latest" / "pretrained_model" / "model.safetensors"))
        pol.load_state_dict(sd, strict=False)
        st = torch.load(run_dir / "checkpoints" / "latest" / "training_state.pt", weights_only=False)
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        stream.restore(st["stream"])
        torch.random.set_rng_state(st["rng"])
        torch.cuda.set_rng_state(st["cuda_rng"])
        step, best, log = st["step"], st["best"], st["log"]
        print(json.dumps({"resumed_from_step": step}), flush=True)

    @torch.no_grad()
    def validate() -> float:
        pol.eval()
        s, c = torch.random.get_rng_state(), torch.cuda.get_rng_state()
        torch.manual_seed(1234)
        vals = [float(loss_on(pol, b, device)) for b in val_batches]
        torch.random.set_rng_state(s)
        torch.cuda.set_rng_state(c)
        pol.train()
        return float(np.mean(vals))

    def save(name: str, full: bool):
        final, tmp = run_dir / "checkpoints" / name, run_dir / "checkpoints" / f".{name}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        (tmp / "pretrained_model").mkdir(parents=True)
        pol.save_pretrained(tmp / "pretrained_model")
        pre.save_pretrained(tmp / "pretrained_model")
        post.save_pretrained(tmp / "pretrained_model")
        (tmp / "training_state.json").write_text(json.dumps({"step": step, "best_val_loss": best}))
        if full:
            torch.save({"step": step, "best": best, "log": log, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                        "stream": stream.state(), "rng": torch.random.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, tmp / "training_state.pt")
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)

    mon = ResourceMonitor()
    t0 = time.time()
    start_step = step
    if step == 0:
        v0 = validate()
        log.append({"step": 0, "val_loss": v0})
        print(json.dumps({"step": 0, "val_loss": v0, "trainable_params": n_train, "train_shards": len(files), "val_samples": sum(len(b["state"]) for b in val_batches)}), flush=True)
    with mon:
        while step < a.steps:
            b = stream.next_batch(a.batch)
            loss = loss_on(pol, b, device)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % 100 == 0:
                rec = {"step": step, "loss": float(loss), "grad_norm": float(gn), "lr": opt.param_groups[0]["lr"], "epoch": stream.epoch,
                       "samples_per_s": round((step - start_step) * a.batch / (time.time() - t0), 1), "peak_vram_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3)}
                log.append(rec)
                print(json.dumps(rec), flush=True)
            if step % a.val_every == 0 or step == a.steps:
                vl = validate()
                log.append({"step": step, "val_loss": vl})
                print(json.dumps({"step": step, "val_loss": vl}), flush=True)
                if vl < best:
                    best = vl
                    save("best", False)
            if step % a.ckpt_every == 0 or step == a.steps:
                save("latest", True)
    summary = {"run_dir": str(run_dir.relative_to(ROOT)), "init": a.init, "steps": step, "batch": a.batch, "lr": a.lr, "best_val_loss": best,
               "samples_seen": step * a.batch, "wall_time_s": round(time.time() - t0, 1), "trainable_params": n_train, "resources": mon.summary(),
               "cache_parts": a.parts, "environment": environment_info(), "log": log}
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if a.export:
        exp = ROOT / "checkpoints" / a.export
        for n in ("best", "latest"):
            if (exp / n).exists():
                shutil.rmtree(exp / n)
            shutil.copytree(run_dir / "checkpoints" / n / "pretrained_model", exp / n)
        shutil.copy(run_dir / "training_summary.json", exp / "training_summary.json")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("log", "environment")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
