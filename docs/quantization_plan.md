# Compression plan: sensitivity → CLP → quantization (W8A8 → W4A8 → W4A4 → mixed)

Sources checked on 2026-10-07. Status values: PASS / FAIL / BLOCKED / NOT YET VERIFIED.

## FoldQuantVLA (official: github.com/cair-vinuni/FoldQuantVLA, arXiv 2609.24433)

| fact | value (quoted / from README) |
|---|---|
| supported models | GR00T N1.5 / N1.6 / N1.7, π₀.₅. **Evo-1 is not supported.** |
| method | one change of basis per shared activation site, inverse folded into every consuming weight, rounding after the basis is fixed. Uniform W4A4 projections + selective INT8 for sensitive language output projections |
| native kernels | "TensorRT plugins on CUTLASS integer GEMMs run W8A8 and W4A4 projections" |
| native INT4 targets | "Four-bit is native INT4 on Ada (sm_89) and Jetson AGX Orin (sm_87); H100 lowers four-bit operands to its INT8 datapath" |
| fake quant | "runs in PyTorch with fake-quant layers" |
| environments | each model directory pins its own Python / torch / TensorRT |

This machine: RTX 3050 Laptop, compute capability **8.6** (sm_86, Ampere). The Ampere ISA has INT4 tensor-core MMA,
but FoldQuantVLA does not list sm_86, and Evo-1 would need new folding sites (InternVL3 vision tower, Qwen2-style
language layers, Evo-1 integration module and flow-matching action head).

| item | status |
|---|---|
| FoldQuantVLA support for Evo-1 | BLOCKED upstream (not supported); requires our own adaptation, labelled "FoldQuant-style" |
| Native FoldQuant W4A4 on RTX 3050 (sm_86) | NOT YET VERIFIED (needs a CUTLASS/TensorRT build for sm_86). If the build fails: BLOCKED, and fake quant + W8A8 + W4A8 + mixed precision remain the deliverables |
| PyTorch fake-quant / functional low-bit simulation | planned (always runnable) |

## CLP (arXiv 2606.20246, "Finetuning Vision-Language-Action Models Requires Fewer Layers Than You Think")

CKA-guided Layer Pruning: training-free layer selection by CKA similarity, "prune first, then finetune normally".
Reported for π₀ (6/18 layers) and GR00T-N1.5 (7/12 VLM layers, 8/16 DiT layers). **Layer lists are model-specific;
the GR00T list is not reused.** For Evo-1 this project implements a *CLP-inspired adaptation to Evo-1*:
full model → CKA per layer on an 11-task representative subset (hidden states streamed to disk) → candidate redundant
layers → prune one → benchmark (offline action metrics + sim success + VRAM/RAM/latency) → prune more only if acceptable.

## Order (mandatory)

FP16/BF16 reference → W8A8 → W4A8 → W4A4 → mixed precision (sensitivity-guided). Never start from W4A4.

Initial candidates, each to be benchmarked rather than assumed:
- W8A8: vision FP16, projector FP16, language W8A8, action expert W8A8, interfaces FP16
- W4A8: vision FP16, language W4A8, projector FP16, action expert W8A8 (action W4A8 only if justified)
- mixed: vision FP16, language W4A4 with sensitive projections W8A8, projector FP16, action expert W4A4 with sensitive projections W8A8
