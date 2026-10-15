#!/usr/bin/env python
"""Where does Evo-1 inference time go? Times (CUDA-synchronized, median of N runs on one real val frame) of:
  vision   : InternViT vision tower + projector (via forward hooks on the official modules)
  language : truncated language model
  vlm      : whole `_compute_fused_tokens`
  head     : action head sampling (`policy.model(...)`, num_inference_timesteps flow steps)
  chunk    : `predict_action_chunk` end to end
Writes reports/latency_profile_<name>.json. Evaluation only.

    python scripts/profile_evo1_latency.py --checkpoint checkpoints/evo1_so101_s2lite_selective_llm/best
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.offline_eval import AGGR_ROOT  # noqa: E402
from src.evaluation.policy_eval import load_policy  # noqa: E402


def main() -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "evo1_so101_s2lite_selective_llm" / "best"))
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--name", default="evo1_s2lite_selective_llm")
    ap.add_argument("--steps", type=int, help="num_inference_timesteps override")
    ap.add_argument("--quant", help="torchao scheme (src/quantization/torchao_quant.py)")
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()

    pol, pre, _, cfg = load_policy(a.checkpoint, "cuda", {"num_inference_timesteps": a.steps} if a.steps else None)
    pol.eval()
    qinfo = None
    if a.quant or a.compile:
        from src.quantization.torchao_quant import apply

        qinfo = apply(pol, a.quant or "bf16", a.compile)
    sm = json.loads((ROOT / "reports" / "split_manifest.json").read_text())
    ge = sm["global"]["val"][0]
    ds = LeRobotDataset("local/so101_11tasks", root=AGGR_ROOT, video_backend="pyav", download_videos=False)
    item = ds[int(ds.meta.episodes["dataset_from_index"][ge]) + 100]
    obs = {k: v.unsqueeze(0).cuda() for k, v in item.items() if k.startswith("observation.")}
    obs["task"], obs["robot_type"] = [item["task"]], ["so_follower"]
    vlm = pol.model.embedder.model
    times: dict[str, list[float]] = {k: [] for k in ("vision", "language", "vlm", "head", "chunk")}
    marks: dict[str, float] = {}

    def pre_hook(name):
        def f(*_a, **_k):
            torch.cuda.synchronize()
            marks[name] = time.perf_counter()
        return f

    def post_hook(name):
        def f(*_a, **_k):
            torch.cuda.synchronize()
            times[name].append((time.perf_counter() - marks[name]) * 1000)
        return f

    vt = getattr(vlm.vision_tower, "_orig_mod", vlm.vision_tower)
    lmod = getattr(vlm.language_model, "_orig_mod", vlm.language_model)
    hooks = [vt.register_forward_pre_hook(pre_hook("vision")), vlm.multi_modal_projector.register_forward_hook(post_hook("vision")),
             lmod.register_forward_pre_hook(pre_hook("language")), lmod.register_forward_hook(post_hook("language"))]
    with torch.inference_mode():
        b = pre(obs)
        for i in range(a.runs + 3):
            prompts = pol._normalize_task_batch(b)
            imgs, imask = pol._collect_image_batches(b)
            st, _ = pol._prepare_state(b)
            emb = pol._get_embodiment_ids(b, st.shape[0])
            am = pol._prepare_inference_action_mask(st.shape[0])
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with pol._maybe_autocast():
                fused, cmask = pol._compute_fused_tokens(prompts, imgs, imask)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                pol.model(fused, state=st, action_mask=am, embodiment_ids=emb, context_mask=cmask)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            pol.predict_action_chunk(b)
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            if i >= 3:
                times["vlm"].append((t1 - t0) * 1000)
                times["head"].append((t2 - t1) * 1000)
                times["chunk"].append((t3 - t2) * 1000)
    for h in hooks:
        h.remove()
    med = {k: round(float(np.median(v[-a.runs:])), 1) for k, v in times.items()}
    rep = {"checkpoint": a.checkpoint, "runs": a.runs, "median_ms": med, "vlm_num_layers": cfg.vlm_num_layers,
           "num_inference_timesteps": cfg.num_inference_timesteps, "max_views": cfg.max_views, "fused_tokens": int(cmask.sum()), "quantization": qinfo, "steps_override": a.steps,
           "gpu": torch.cuda.get_device_name(0)}
    (ROOT / "reports" / f"latency_profile_{a.name}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
