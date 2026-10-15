"""Stage-2-lite `selective_llm`: train only the last language-model layers of Evo-1's InternVL3 embedder (+ final norm
+ action head) on cached hidden states.

The layers below `start` are frozen, and the pipeline has no image augmentation, so the hidden state entering layer
`start` is a deterministic function of (images, instruction) and can be cached once
(`scripts/cache_evo1_hidden.py`). The remaining layers are then run with the official Qwen2 decoder-layer modules of
the loaded policy:

    h = layers[start](h, mask, position_ids, rotary(h, position_ids)); ...; fused = language_model.norm(h)

Only the valid tokens (attention mask True) are cached together with their original position ids. The padded keys of
the official forward are excluded by its attention mask, and a valid query only attends to valid keys at earlier
positions, so dropping the other tokens is exact (checked numerically against `Evo1Policy._compute_fused_tokens`
while caching).
"""

from __future__ import annotations

import torch


def language_model(policy):
    return policy.model.embedder.model.language_model


def capture_layer_input(policy, start: int):
    """Register a pre-hook on decoder layer `start`. Returns (store dict, handle). store['h'] / store['pos'] hold the
    hidden states and position ids of the last forward."""
    store: dict = {}

    def hook(_mod, args, kwargs):
        store["h"] = (args[0] if args else kwargs["hidden_states"]).detach()
        store["pos"] = kwargs["position_ids"].detach()

    handle = language_model(policy).layers[start].register_forward_pre_hook(hook, with_kwargs=True)
    return store, handle


def causal_mask(key_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Additive (B, 1, L, L) mask like transformers' eager causal mask: query i sees valid keys j <= i (tokens are kept
    in their original order, so index order == position order)."""
    L = key_mask.shape[1]
    allowed = torch.ones(L, L, dtype=torch.bool, device=key_mask.device).tril()[None] & key_mask[:, None, :]
    m = torch.zeros(allowed.shape, dtype=dtype, device=key_mask.device)
    m.masked_fill_(~allowed, torch.finfo(dtype).min)
    return m[:, None]


def run_tail(policy, h: torch.Tensor, pos: torch.Tensor, key_mask: torch.Tensor, start: int) -> torch.Tensor:
    """Layers start..end + final norm of the (truncated) language model; returns the fused tokens (B, L, H)."""
    lm = language_model(policy)
    mask = causal_mask(key_mask, h.dtype)
    pe = lm.rotary_emb(h, pos)
    for layer in lm.layers[start:]:
        h = layer(h, attention_mask=mask, position_ids=pos, position_embeddings=pe)
    return lm.norm(h)


def set_trainable(policy, start: int) -> list[str]:
    """Stage-1 trainable set (action head etc.) + language-model layers >= start + final norm. The newly trainable LM
    parameters are kept in fp32 (master weights) and run under Evo-1's bf16 autocast; bf16 weights would lose updates
    of size lr * grad to rounding."""
    lm = language_model(policy)
    names = []
    for mod in list(lm.layers[start:]) + [lm.norm]:
        mod.float()
        for p in mod.parameters():
            p.requires_grad = True
    for n, p in policy.named_parameters():
        if p.requires_grad and ".language_model." in n:
            names.append(n)
    return names
