# System check

| item | value |
|------|-------|
| OS | Ubuntu 26.04.1 LTS (kernel 7.0.0-34-generic) |
| Python (this interpreter) | 3.12.14 — `/home/ncd/Desktop/VRB/task_week_2/so101_evo1/.venv/bin/python` |
| CPU | 12th Gen Intel(R) Core(TM) i5-12450HX (12 threads) |
| RAM total / available / swap | 10.9 / 4.1 / 37.2 GB |
| Disk free (project) | 231 of 385 GB |
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU, 6.0 GB VRAM, compute 8.6 |
| NVIDIA driver / driver CUDA | 595.91.07 / 13.2 |
| nvcc (CUDA toolkit) | not installed |
| PyTorch | 2.11.0+cu130 (CUDA build 13.0), cuda_available=True, bf16=True, matmul=True |
| pkg `lerobot` | 0.6.1 |
| pkg `transformers` | 5.5.4 |
| pkg `accelerate` | 1.15.0 |
| pkg `huggingface-hub` | 1.33.0 |
| pkg `datasets` | 4.8.5 |
| pkg `torchcodec` | 0.11.1 |
| pkg `av` | 15.1.0 |
| pkg `flash-attn` | not installed |
| pkg `feetech-servo-sdk` | 1.0.0 |
| pkg `isaacsim` | not installed |
| pkg `isaaclab` | not installed |
| pkg `leisaac` | not installed |
| isaacsim_importable | False |
| isaaclab_importable | False |
| leisaac_importable | False |
| ISAACSIM_PATH | None |
| tool `ffmpeg` | not found |
| tool `ffprobe` | not found |
| tool `git` | /usr/bin/git |
| tool `git-lfs` | not found |
| tool `uv` | /home/ncd/.local/bin/uv |
| tool `conda` | not found |

## Blockers

- BLOCKED: Isaac Sim 5.1 minimum spec not met (needs >= 32 GB RAM, >= 16 GB VRAM, Ubuntu 22.04/24.04; have 10.9 GB RAM, 6.0 GB VRAM, Ubuntu 26.04.1 LTS). IsaacLab/LeIsaac simulation must run on another machine, or a lighter simulator must be chosen.

## Warnings

- ffmpeg not on PATH. Not required for decoding (LeRobot uses torchcodec/PyAV wheels), but needed for MP4 encoding of evaluation videos via ffmpeg CLI.
- git-lfs not installed. Not required: datasets/models are fetched with huggingface_hub.

## Recommendations

- VRAM 6.0 GB: Evo-1 stage1 only (VLM frozen), batch_size 1-2 with gradient accumulation, bf16 autocast (policy.use_amp=true), gradient checkpointing on, flash-attn optional. Full stage2 VLM finetune (~0.8B params with AdamW) does not fit; run it on a >=24 GB GPU.
- RAM 10.9 GB: num_workers<=2, prefetch_factor<=2, never cache decoded frames in RAM.
