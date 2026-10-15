#!/usr/bin/env python
"""Layer redundancy (CKA) and layer sensitivity of Evo-1's InternVL3 embedder, as input for CLP-inspired layer pruning
and for choosing which modules to keep at higher precision when quantizing.

Frames: real VAL episodes (never trained on), `--frames-per-task` frames per task spread over each episode.

CKA: linear CKA (Kornblith et al. 2019) between the input and the output token representations of every
language-model decoder layer and every vision-encoder layer, over all valid tokens (language model: tokens with
attention mask; vision: all patch tokens of the present camera views). High CKA = the layer barely changes its input
(candidate for pruning). Accumulated as centered (cross-)covariances, so no token features are stored.
Two variants are reported: `raw` (all tokens) and `robust` (tokens whose norm exceeds `--outlier-factor` x the batch
median at any layer are excluded). Transformers develop a few "massive activation" tokens whose huge, layer-invariant
values dominate the covariance and push raw linear CKA to ~1 for every layer after they appear.

Sensitivity: each layer is skipped in turn (forward hook returns the layer input = identity). With paired noise
(fixed seeds per batch, identical across configurations) the script measures
  - the official flow-matching loss (as in Evo1Policy.forward), and
  - the action MAE in degrees of one sampled 50-step chunk (official inference path + postprocessor) vs the recorded
    actions,
and reports the change against the unmodified model. Evaluation only: nothing is trained or saved besides the report.

    python scripts/analyze_cka_sensitivity.py --checkpoint checkpoints/evo1_so101_s2lite_selective_llm/best
Writes reports/cka_sensitivity_<name>.{json,md}
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
from src.simulation.tasks import TASKS  # noqa: E402
from src.utils.resources import ResourceMonitor  # noqa: E402


def _first(x):
    return x[0] if isinstance(x, tuple) else x


class CovAcc:
    """Accumulates sum x, sum x^T x (and cross terms) in float64 for linear CKA."""

    def __init__(self, n_rep: int, d: int, device):
        self.n = 0
        self.s = torch.zeros(n_rep, d, dtype=torch.float64, device=device)
        self.g = torch.zeros(n_rep, d, d, dtype=torch.float64, device=device)  # gram of representation k
        self.c = torch.zeros(n_rep - 1, d, d, dtype=torch.float64, device=device)  # cross gram (k+1)^T k

    def add(self, reps: list[torch.Tensor], keep: torch.Tensor | None = None):
        xs = [r.float() if keep is None else r[keep].float() for r in reps]
        self.n += xs[0].shape[0]
        for k, x in enumerate(xs):
            self.s[k] += x.sum(0).double()
            self.g[k] += (x.T @ x).double()
            if k:
                self.c[k - 1] += (x.T @ xs[k - 1]).double()

    def cka_in_out(self) -> list[float]:
        n = self.n
        out = []
        for k in range(len(self.c)):
            mx, my = self.s[k] / n, self.s[k + 1] / n
            cxx = self.g[k] - n * torch.outer(mx, mx)
            cyy = self.g[k + 1] - n * torch.outer(my, my)
            cyx = self.c[k] - n * torch.outer(my, mx)
            out.append(float((cyx**2).sum() / (torch.linalg.norm(cxx) * torch.linalg.norm(cyy))))
        return out


def main() -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "evo1_so101_s2lite_selective_llm" / "best"))
    ap.add_argument("--frames-per-task", type=int, default=12)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--name", default="evo1_s2lite_selective_llm")
    ap.add_argument("--skip-sensitivity", action="store_true")
    ap.add_argument("--sens-frames-per-task", type=int, default=6, help="subset of the frames used for the skip-layer sweep")
    ap.add_argument("--outlier-factor", type=float, default=10.0)
    a = ap.parse_args()

    mon = ResourceMonitor()
    t_start = time.time()
    with mon:
        pol, pre, post, cfg = load_policy(a.checkpoint, "cuda")
        pol.eval()
        H = cfg.chunk_size
        dev = torch.device("cuda")
        sm = json.loads((ROOT / "reports" / "split_manifest.json").read_text())
        ds = LeRobotDataset("local/so101_11tasks", root=AGGR_ROOT, video_backend="pyav", download_videos=False,
                            delta_timestamps={"action": [i / 30 for i in range(H)]})
        frames = []
        for t in sorted(TASKS):
            s = sm["tasks"][t]
            eps = [s["global_episode_offset"] + e for e in s["val"]]
            per_ep = max(1, int(np.ceil(a.frames_per_task / len(eps))))
            sel = []
            for ge in eps:
                lo, hi = int(ds.meta.episodes["dataset_from_index"][ge]), int(ds.meta.episodes["dataset_to_index"][ge])
                sel += [int(i) for i in np.linspace(lo, max(lo, hi - H - 1), per_ep + 2)[1:-1]]
            frames += sel[: a.frames_per_task]
        batches = []
        for i in range(0, len(frames), a.batch):
            items = [ds[j] for j in frames[i : i + a.batch]]
            batches.append(torch.utils.data.default_collate(items))
        print(json.dumps({"frames": len(frames), "batches": len(batches), "load_s": round(time.time() - t_start, 1)}), flush=True)

        vlm = pol.model.embedder.model
        lm_layers = list(vlm.language_model.layers)
        vis_layers = list(vlm.vision_tower.encoder.layer)

        def to_dev(b):
            return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}

        # ---------------- CKA ----------------
        d_lm, d_vis = lm_layers[0].self_attn.q_proj.in_features, vis_layers[0].layernorm_before.normalized_shape[0]
        acc = {(m, v): CovAcc((len(lm_layers) if m == "lm" else len(vis_layers)) + 1, d_lm if m == "lm" else d_vis, dev)
               for m in ("lm", "vis") for v in ("raw", "robust")}

        def robust_keep(reps):
            keep = torch.ones(reps[0].shape[0], dtype=torch.bool, device=reps[0].device)
            for r in reps:
                n = r.float().norm(dim=-1)
                keep &= n <= a.outlier_factor * n.median()
            return keep
        cap: dict = {}
        hooks = []
        for name, layers in (("lm", lm_layers), ("vis", vis_layers)):
            for k, layer in enumerate(layers):
                if k == 0:
                    hooks.append(layer.register_forward_pre_hook(
                        lambda m, args, kwargs, n=name: cap.__setitem__((n, 0), (args[0] if args else kwargs["hidden_states"]).detach()), with_kwargs=True))
                hooks.append(layer.register_forward_hook(lambda m, i, o, n=name, kk=k: cap.__setitem__((n, kk + 1), _first(o).detach())))
        t0 = time.time()
        excluded = {"lm": 0, "vis": 0}
        with torch.inference_mode():
            for b in batches:
                bb = pre(to_dev(b))
                imgs, imask = pol._collect_image_batches(bb)
                with pol._maybe_autocast():
                    _, cmask = pol._compute_fused_tokens(pol._normalize_task_batch(bb), imgs, imask)
                valid = cmask.bool()
                present = imask.bool().reshape(-1)  # (B * max_views), same order as the stacked pixel values
                reps = {"lm": [cap[("lm", k)][valid] for k in range(len(lm_layers) + 1)],
                        "vis": [cap[("vis", k)][present].reshape(-1, cap[("vis", k)].shape[-1]) for k in range(len(vis_layers) + 1)]}
                for m, r in reps.items():
                    acc[(m, "raw")].add(r)
                    keep = robust_keep(r)
                    acc[(m, "robust")].add(r, keep)
                    excluded[m] += int((~keep).sum())
        for h in hooks:
            h.remove()
        cka = {v: {"language_model": acc[("lm", v)].cka_in_out(), "vision_encoder": acc[("vis", v)].cka_in_out(),
                   "tokens": {"language_model": acc[("lm", v)].n, "vision_encoder": acc[("vis", v)].n}} for v in ("raw", "robust")}
        cka["robust"]["excluded_tokens"] = {"language_model": excluded["lm"], "vision_encoder": excluded["vis"]}
        cka["outlier_factor"] = a.outlier_factor
        cka["time_s"] = round(time.time() - t0, 1)
        for v in ("raw", "robust"):
            print(json.dumps({f"cka_lm_{v}": [round(x, 4) for x in cka[v]["language_model"]]}), flush=True)
            print(json.dumps({f"cka_vis_{v}": [round(x, 4) for x in cka[v]["vision_encoder"]]}), flush=True)
        print(json.dumps({"excluded": excluded, "tokens_raw": cka["raw"]["tokens"], "s": cka["time_s"]}), flush=True)
        del acc
        torch.cuda.empty_cache()

        # ---------------- sensitivity ----------------
        sens = None
        if not a.skip_sensitivity:

            sens_batches = []
            for ti in range(len(TASKS)):  # first sens-frames-per-task frames of every task (batches are task-ordered)
                per = a.frames_per_task // a.batch
                sens_batches += batches[ti * per : ti * per + max(1, a.sens_frames_per_task // a.batch)]

            def evaluate() -> dict:
                losses, maes = [], []
                with torch.inference_mode():
                    for bi, b in enumerate(sens_batches):
                        bb = pre(to_dev(b))
                        imgs, imask = pol._collect_image_batches(bb)
                        st, _ = pol._prepare_state(bb)
                        ac, amask = pol._prepare_actions(bb)
                        emb = pol._get_embodiment_ids(bb, st.shape[0])
                        with pol._maybe_autocast():
                            fused, cmask = pol._compute_fused_tokens(pol._normalize_task_batch(bb), imgs, imask)
                            torch.manual_seed(a.seed * 100003 + bi)
                            pv, noise = pol.model(fused, state=st, actions_gt=ac, action_mask=amask.to(dtype=pol._compute_dtype),
                                                  embodiment_ids=emb, context_mask=cmask)
                            flat = amask.view(amask.shape[0], -1).float()
                            tgt = (ac.float() - noise.float()).view(ac.shape[0], -1) * flat
                            losses.append(float(pol._compute_masked_loss(pv.float(), tgt, amask, "mean")))
                            torch.manual_seed(a.seed * 100003 + 50000 + bi)
                            pred = pol.model(fused, state=st, action_mask=pol._prepare_inference_action_mask(st.shape[0]),
                                             embodiment_ids=emb, context_mask=cmask)
                        pred = pred.view(st.shape[0], H, cfg.max_action_dim).float()
                        p = torch.stack([post(pred[:, i]) for i in range(H)], dim=1).float().cpu().numpy()[..., :6]
                        g = b["action"].numpy()[:, :H]
                        valid = ~b["action_is_pad"].numpy()[:, :H]
                        maes.append(float(np.abs(p - g)[valid].mean()))
                return {"loss": float(np.mean(losses)), "action_mae": float(np.mean(maes))}

            def skip(layer):
                return layer.register_forward_hook(lambda m, i, o: (i[0],) + tuple(o[1:]) if isinstance(o, tuple) else i[0])

            t0 = time.time()
            base = evaluate()
            print(json.dumps({"base": base, "s": round(time.time() - t0, 1)}), flush=True)
            sens = {"base": base, "frames": sum(len(b["action"]) for b in sens_batches), "language_model": [], "vision_encoder": []}
            for name, layers in (("language_model", lm_layers), ("vision_encoder", vis_layers)):
                for k, layer in enumerate(layers):
                    h = skip(layer)
                    r = evaluate()
                    h.remove()
                    r.update({"layer": k, "d_loss": r["loss"] - base["loss"], "d_action_mae": r["action_mae"] - base["action_mae"]})
                    sens[name].append(r)
                    print(json.dumps({"skip": name, **{kk: round(v, 4) if isinstance(v, float) else v for kk, v in r.items()}}), flush=True)
            sens["time_s"] = round(time.time() - t0, 1)

    rep = {"checkpoint": a.checkpoint, "frames": len(frames), "frames_per_task": a.frames_per_task, "split": "val", "seed": a.seed,
           "cka_in_out": cka, "sensitivity_skip_layer": sens, "resources": mon.summary(), "wall_time_s": round(time.time() - t_start, 1)}
    out = ROOT / "reports" / f"cka_sensitivity_{a.name}"
    out.with_suffix(".json").write_text(json.dumps(rep, indent=2))
    md = [f"# CKA / sensitivity: `{a.checkpoint}`", "",
          f"{len(frames)} real val frames ({a.frames_per_task}/task) for CKA; {sens['frames'] if sens else 0} of them for the skip sweep. "
          "CKA = linear CKA between layer input and output tokens (raw / robust = without massive-activation tokens, "
          f"factor {a.outlier_factor}). Skip = layer replaced by identity; deltas vs the unmodified model with paired noise.", ""]
    if sens:
        md += [f"Base: loss {sens['base']['loss']:.4f}, action MAE {sens['base']['action_mae']:.2f} deg", ""]
    for name, key in (("Language model (InternVL3 / Qwen2.5, truncated to 14 layers)", "language_model"), ("Vision encoder (InternViT-300M)", "vision_encoder")):
        md += [f"## {name}", "", "| layer | CKA raw | CKA robust | d_loss (skip) | d_action_MAE deg (skip) |", "|---:|---:|---:|---:|---:|"]
        for k, c in enumerate(cka["raw"][key]):
            s = sens[key][k] if sens else None
            md.append(f"| {k} | {c:.4f} | {cka['robust'][key][k]:.4f} | " + (f"{s['d_loss']:+.4f} | {s['d_action_mae']:+.2f} |" if s else "– | – |"))
        md.append("")
    out.with_suffix(".md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
