"""Post-training quantization of a loaded Evo-1 policy with torchao (no retraining, architecture unchanged).

Schemes (applied to plain nn.Linear layers only, not nn.MultiheadAttention internals; norms, embeddings, convolutions and the interfaces stay BF16):
  bf16            no quantization (reference; optionally compiled for a fair latency comparison)
  w8a8_vision     int8 weights + dynamic per-token int8 activations in the InternViT encoder layers
  w8a8_vis_proj   + the multi-modal projector
  w8a8_vis_lm     + the language-model decoder layers
  w8a8_all        + the action head (flow-matching transformer)
`compile=True` wraps the vision tower, the language model and the action head in torch.compile; on this GPU (sm_86)
torchao int8 matmuls are only faster than BF16 when compiled (measured: 1.85x on a 1024->4096->1024 MLP, 5.5x slower
in eager mode).
"""

from __future__ import annotations

import torch

SCHEMES = ("bf16", "w8a8_vision", "w8a8_vis_proj", "w8a8_vis_lm", "w8a8_all")


def _targets(policy, scheme: str) -> list[torch.nn.Module]:
    vlm = policy.model.embedder.model
    t = []
    if scheme in ("w8a8_vision", "w8a8_vis_proj", "w8a8_vis_lm", "w8a8_all"):
        t.append(vlm.vision_tower.encoder)
    if scheme in ("w8a8_vis_proj", "w8a8_vis_lm", "w8a8_all"):
        t.append(vlm.multi_modal_projector)
    if scheme in ("w8a8_vis_lm", "w8a8_all"):
        t.append(vlm.language_model.layers)
    if scheme == "w8a8_all":
        t.append(policy.model.action_head)
    return t


def apply(policy, scheme: str = "bf16", compile: bool = True) -> dict:
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

    assert scheme in SCHEMES, scheme
    # exact nn.Linear only: nn.MultiheadAttention's out_proj (NonDynamicallyQuantizableLinear) is consumed by
    # F.multi_head_attention_forward through its raw weight, which torchao's Int8Tensor does not support
    is_lin = lambda m, _fqn=None: type(m) is torch.nn.Linear  # noqa: E731
    n_q = 0
    for mod in _targets(policy, scheme):
        n_q += sum(is_lin(m) for m in mod.modules())
        quantize_(mod, Int8DynamicActivationInt8WeightConfig(), filter_fn=is_lin)
    if compile:
        vlm = policy.model.embedder.model
        vlm.vision_tower = torch.compile(vlm.vision_tower)
        vlm.language_model = torch.compile(vlm.language_model, dynamic=True)  # prompt length differs per task
        policy.model.action_head = torch.compile(policy.model.action_head, dynamic=True)
    return {"scheme": scheme, "compile": compile, "quantized_linear_layers": n_q, "library": "torchao",
            "config": "Int8DynamicActivationInt8WeightConfig" if scheme != "bf16" else None}


def quantized_summary(policy) -> dict:
    """Number of torchao int8 weight tensors per module group (evidence of what is really quantized)."""
    vlm = policy.model.embedder.model
    unwrap = lambda m: getattr(m, "_orig_mod", m)  # noqa: E731
    groups = {"vision_tower": unwrap(vlm.vision_tower), "multi_modal_projector": vlm.multi_modal_projector,
              "language_model": unwrap(vlm.language_model), "action_head": unwrap(policy.model.action_head)}
    return {g: sum("Int8" in type(p).__name__ for p in m.parameters()) for g, m in groups.items()}
