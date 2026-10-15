"""Quantization targeting: only the requested module groups are int8-quantized (toy model with the Evo-1 layout)."""

import pytest
import torch
from torch import nn

torchao = pytest.importorskip("torchao")

from src.quantization.torchao_quant import apply, quantized_summary  # noqa: E402


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        vlm = nn.Module()
        vlm.vision_tower = nn.Module()
        vlm.vision_tower.encoder = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 64))
        vlm.multi_modal_projector = nn.Sequential(nn.Linear(64, 64))
        vlm.language_model = nn.Module()
        vlm.language_model.layers = nn.ModuleList([nn.Linear(64, 64)])
        self.model = nn.Module()
        self.model.embedder = nn.Module()
        self.model.embedder.model = vlm
        head = nn.Module()
        head.proj = nn.Linear(64, 64)
        head.attn = nn.MultiheadAttention(64, 4)  # out_proj must never be quantized
        self.model.action_head = head


def _toy():
    return _Toy().to(torch.bfloat16)


def test_w8a8_vision_quantizes_only_vision():
    pol = _toy()
    info = apply(pol, "w8a8_vision", compile=False)
    assert info["quantized_linear_layers"] == 2
    s = quantized_summary(pol)
    assert s["vision_tower"] == 2
    assert s["multi_modal_projector"] == 0 and s["language_model"] == 0 and s["action_head"] == 0
    assert pol.model.embedder.model.multi_modal_projector[0].weight.dtype == torch.bfloat16


def test_bf16_scheme_quantizes_nothing():
    pol = _toy()
    apply(pol, "bf16", compile=False)
    assert sum(quantized_summary(pol).values()) == 0


def test_all_scheme_skips_multihead_attention_out_proj():
    pol = _toy()
    apply(pol, "w8a8_all", compile=False)
    assert "Int8" not in type(pol.model.action_head.attn.out_proj.weight).__name__
    assert quantized_summary(pol)["action_head"] == 1
