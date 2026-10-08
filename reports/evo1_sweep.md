# Evo-1 stage1 local sweep (RTX 3050 6 GB, 2026-10-07)

5 optimizer updates per setting, grad_accum 1, both cameras at 448 px (the official evo1 rejects 256: InternVL3
image_seq_length is tied to its native 448), chunk 50, bf16 autocast, num_workers 0. 775.2M params total, 122.0M
trainable (VLM frozen).

| batch | grad ckpt | peak VRAM (alloc) | peak RSS | steady s/update | samples/s (steady) | result |
|---:|:---:|---:|---:|---:|---:|---|
| 1 | on | 3.52 GB | 2.15 GB | 0.66 | ~1.5 | PASS |
| 2 | on | 3.65 GB | 2.20 GB | 1.17 | ~1.7 | PASS |
| 4 | on | >5.8 GB | — | — | — | FAIL (CUDA OOM) |
| 1 | off | 3.52 GB | 2.13 GB | 0.65 | ~1.5 | PASS |
| 2 | off | 3.65 GB | 2.22 GB | 1.16 | ~1.7 | PASS |

Selected: batch 2 × grad_accum 4 (effective 8). Inference (checkpoint reload, offline): 816 ms P50 / 1087 ms P95 per
50-action chunk, peak RSS 3.4 GB.
