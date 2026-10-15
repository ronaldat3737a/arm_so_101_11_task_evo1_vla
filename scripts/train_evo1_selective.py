#!/usr/bin/env python
"""Stage-2-lite Evo-1 training, profile `selective_llm`: the last language-model layers of the InternVL3 embedder
(+ final norm) are trained together with the stage-1 trainable set (action head), on cached hidden states
(scripts/cache_evo1_hidden.py). Vision tower, projector and the lower language-model layers stay frozen.

Forward per batch: cached hidden state at layer `cache_start` -> frozen layers cache_start..train_from-1 -> trainable
layers train_from..end -> norm -> fused tokens -> official Evo-1 action-head loss (same as scripts/train_evo1_cached.py).
Step 0 reports the validation loss of the init checkpoint; with an unchanged init it must equal that checkpoint's
cached-token validation loss (end-to-end check of the cache + tail reconstruction).

Checkpoints are full LeRobot Evo-1 policies (the trained language-model layers are inside model.safetensors and are
loaded strictly by Evo1Policy.from_pretrained). A sidecar `stage2_lite.json` records the profile. Resumable.

    python scripts/train_evo1_selective.py --train-from 12 --steps 10000
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

from scripts.train_evo1_cached import ShardStream, loss_on  # noqa: E402
from src.evaluation.policy_eval import load_policy  # noqa: E402
from src.training.launcher import environment_info  # noqa: E402
from src.training.selective_llm import language_model, run_tail, set_trainable  # noqa: E402
from src.utils.resources import ResourceMonitor  # noqa: E402


def collate_hidden(items) -> dict:
    lens = [int(off[j + 1] - off[j]) for d, off, j in items]
    L, B = max(lens), len(items)
    h = torch.zeros(B, L, items[0][0]["hidden"].shape[1], dtype=torch.bfloat16)
    pos = torch.zeros(B, L, dtype=torch.long)
    kmask = torch.zeros(B, L, dtype=torch.bool)
    for r, (d, off, j) in enumerate(items):
        h[r, : lens[r]] = d["hidden"][off[j] : off[j + 1]]
        pos[r, : lens[r]] = d["position_ids"][off[j] : off[j + 1]].long()
        kmask[r, : lens[r]] = True
    pick = lambda k: torch.stack([d[k][j] for d, off, j in items])  # noqa: E731
    return {"hidden": h, "position_ids": pos, "context_mask": kmask, "state": pick("state"), "actions": pick("actions"),
            "action_mask": pick("action_mask")}


def split(b: dict, n: int) -> list[dict]:
    B = b["state"].shape[0]
    return [{k: v[i : i + n] for k, v in b.items()} for i in range(0, B, n)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default=str(ROOT / "checkpoints" / "evo1_so101" / "best"))
    ap.add_argument("--cache", default=str(ROOT / "datasets" / "evo1_hidden_cache_L10"))
    ap.add_argument("--train-from", type=int, default=12, help="first trainable language-model layer (of 14)")
    ap.add_argument("--parts", default="train_s10_o0")
    ap.add_argument("--val-part", default="val_s30_o0")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--name", default="evo1_s2lite_selective_llm")
    ap.add_argument("--export", default="evo1_so101_s2lite_selective_llm")
    ap.add_argument("--resume", help="run dir to continue")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    run_dir = Path(a.resume).resolve() if a.resume else ROOT / "outputs" / f"{a.name}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    if a.resume:
        a = argparse.Namespace(**{**vars(a), **json.loads((run_dir / "args.json").read_text()), "resume": a.resume})
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps({k: v for k, v in vars(a).items() if k != "resume"}, indent=2))
    cmeta = json.loads((Path(a.cache) / "cache_meta.json").read_text())
    cache_start = int(cmeta["start_layer"])
    assert cache_start <= a.train_from < int(cmeta["vlm_num_layers"]), (cache_start, a.train_from)

    pol, pre, post, cfg = load_policy(a.init, "cuda")
    cfg.optimizer_lr = a.lr
    cfg.scheduler_warmup_steps = a.warmup
    device = torch.device("cuda")
    # modules the cached forward never uses -> CPU (frees VRAM; they are still saved with the policy)
    vlm, lm = pol.model.embedder.model, language_model(pol)
    for m in [vlm.vision_tower, vlm.multi_modal_projector, lm.embed_tokens, *lm.layers[:cache_start]]:
        m.to("cpu")
    torch.cuda.empty_cache()
    lm_names = set_trainable(pol, a.train_from)
    pol.train()
    pol.model.embedder.eval()  # no dropout anywhere in the language model (attention_dropout = 0); keep eval semantics
    params = pol.get_optim_params()
    n_train = sum(p.numel() for g in params for p in g["params"])
    n_lm = sum(p.numel() for n, p in pol.named_parameters() if n in set(lm_names))
    opt = cfg.get_optimizer_preset().build(params)
    sched = cfg.get_scheduler_preset().build(opt, a.steps)
    files = sorted(f for p in a.parts.split(",") for f in (Path(a.cache) / p).glob("shard_*.safetensors"))
    val_files = sorted((Path(a.cache) / a.val_part).glob("shard_*.safetensors"))
    if not files or not val_files:
        print(f"STOP: no cached shards in {a.cache}/{a.parts} or {a.val_part}")
        return 1
    stream = ShardStream(files, a.seed, collate_hidden)
    val_batches = []
    for f in val_files:
        d = load_file(str(f))
        off = np.concatenate([[0], np.cumsum(d["lengths"].numpy())])
        idx = [(d, off, j) for j in range(len(d["lengths"]))]
        val_batches += [collate_hidden(idx[i : i + a.batch]) for i in range(0, len(idx), a.batch)]  # same batching as train_evo1_cached
    clip = cfg.optimizer_grad_clip_norm
    step, best, log = 0, math.inf, []
    if a.resume:
        sd = load_file(str(run_dir / "checkpoints" / "latest" / "pretrained_model" / "model.safetensors"))
        pol.load_state_dict(sd, strict=False)
        st = torch.load(run_dir / "checkpoints" / "latest" / "training_state.pt", weights_only=False)
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        stream.restore(st["stream"])
        torch.random.set_rng_state(st["rng"])
        torch.cuda.set_rng_state(st["cuda_rng"])
        step, best, log = st["step"], st["best"], st["log"]
        print(json.dumps({"resumed_from_step": step}), flush=True)

    def fused_loss(mb: dict) -> torch.Tensor:
        h = mb["hidden"].to(device)
        pos = mb["position_ids"].to(device)
        km = mb["context_mask"].to(device)
        with pol._maybe_autocast():
            fused = run_tail(pol, h, pos, km, cache_start)
        return loss_on(pol, {"tokens": fused, "state": mb["state"], "actions": mb["actions"], "action_mask": mb["action_mask"],
                             "context_mask": mb["context_mask"]}, device)

    @torch.no_grad()
    def validate() -> float:
        pol.eval()
        s, c = torch.random.get_rng_state(), torch.cuda.get_rng_state()
        torch.manual_seed(1234)
        vals = [float(fused_loss(b)) for b in val_batches]
        torch.random.set_rng_state(s)
        torch.cuda.set_rng_state(c)
        pol.train()
        pol.model.embedder.eval()
        return float(np.mean(vals))

    def save(name: str, full: bool):
        final, tmp = run_dir / "checkpoints" / name, run_dir / "checkpoints" / f".{name}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        (tmp / "pretrained_model").mkdir(parents=True)
        pol.save_pretrained(tmp / "pretrained_model")
        pre.save_pretrained(tmp / "pretrained_model")
        post.save_pretrained(tmp / "pretrained_model")
        (tmp / "pretrained_model" / "stage2_lite.json").write_text(json.dumps(
            {"profile": "selective_llm", "trainable_language_model_layers": list(range(a.train_from, len(lm.layers))),
             "trainable_language_model_params": n_lm, "init": a.init, "step": step,
             "note": "trained LM layers stored in fp32 in model.safetensors; loaded into the bf16 VLM at inference"}, indent=2))
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
        print(json.dumps({"step": 0, "val_loss": v0, "trainable_params": n_train, "trainable_lm_params": n_lm, "train_shards": len(files),
                          "val_samples": sum(len(b["state"]) for b in val_batches), "peak_vram_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3)}), flush=True)
    with mon:
        while step < a.steps:
            b = stream.next_batch(a.batch)
            mbs = split(b, a.micro_batch)
            tot = 0.0
            for mb in mbs:
                loss = fused_loss(mb) * (mb["state"].shape[0] / a.batch)
                loss.backward()
                tot += float(loss)
            gn = torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % 100 == 0:
                rec = {"step": step, "loss": tot, "grad_norm": float(gn), "lr": opt.param_groups[0]["lr"], "epoch": stream.epoch,
                       "samples_per_s": round((step - start_step) * a.batch / (time.time() - t0), 1), "peak_vram_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                       "rss_gb": round(mon._rss(), 2)}
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
    summary = {"run_dir": str(run_dir.relative_to(ROOT)), "profile": "selective_llm", "init": a.init, "train_from_layer": a.train_from,
               "steps": step, "batch": a.batch, "micro_batch": a.micro_batch, "lr": a.lr, "best_val_loss": best, "samples_seen": step * a.batch,
               "wall_time_s": round(time.time() - t0, 1), "trainable_params": n_train, "trainable_lm_params": n_lm, "resources": mon.summary(),
               "peak_vram_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3), "cache": a.cache, "cache_parts": a.parts,
               "environment": environment_info(), "log": log}
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if a.export:
        exp = ROOT / "checkpoints" / a.export
        exp.mkdir(parents=True, exist_ok=True)
        for n in ("best", "latest"):
            if (exp / n).exists():
                shutil.rmtree(exp / n)
            shutil.copytree(run_dir / "checkpoints" / n / "pretrained_model", exp / n)
        shutil.copy(run_dir / "training_summary.json", exp / "training_summary.json")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("log", "environment")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
