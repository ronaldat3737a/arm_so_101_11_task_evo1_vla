# so101_evo1

Train a Vision-Language-Action policy (**Evo-1**, official LeRobot integration) for the **SO-101** follower arm on
11 manipulation tasks, evaluate it in a **MuJoCo** simulation of SO-101 (including object positions never seen in
training), then deploy it on the real arm with a safety layer. Everything runs locally on an RTX 3050 6 GB / 12 GB RAM laptop.

Current status of every phase: [`reports/status.md`](reports/status.md). Design decisions: [`PROJECT_SPEC.md`](PROJECT_SPEC.md).

## 1. What is in here

| part | where |
|---|---|
| dataset audit / validation / split / merge / norm stats | `scripts/inspect_datasets.py`, `validate_datasets.py`, `aggregate_datasets.py`, `compute_norm_stats.py`, `verify_aggregate.py` |
| robot conventions (verified from LeRobot source) | `docs/so101_interface.md` |
| MuJoCo SO-101 simulation, 11 tasks, spatial train/val/test split | `src/simulation/`, `configs/simulation.yaml`, `scripts/run_oracle.py`, `scripts/check_sim_resets.py` |
| synthetic demonstrations / data mixtures | `scripts/generate_sim_data.py`, `scripts/build_mixture.py`, `configs/data_mixture.yaml` |
| ACT baseline | `configs/act_baseline.yaml`, `scripts/train_act_baseline.py` |
| Evo-1 fine-tuning | `configs/evo1.yaml`, `scripts/train_evo1.py`, `src/training/trainer.py` |
| evaluation (sim closed loop, offline on held-out real episodes) | `scripts/evaluate_policy.py`, `scripts/evaluate_offline.py` |
| real SO-101 (dry-run, safety) | `scripts/deploy_so101.py`, `configs/deployment.yaml`, `configs/camera.yaml`, `docs/real_robot_setup.md` |

## 2. Data

Hugging Face collection [hungho77/so101-leftarmonly-task](https://huggingface.co/collections/hungho77/so101-leftarmonly-task):
11 datasets (`configs/dataset.yaml`), LeRobot v3.0, 30 fps, 551 episodes.
- Cameras: `top` 640×360 and `wrist` 640×480 (AV1).
- State / action: 6-D absolute joint positions in degrees, plus the gripper as 0..100.
- Instructions: one per dataset, used verbatim (`configs/tasks.yaml`).
- 9 episodes without motion are excluded (`reports/dataset_audit.md`).
- Split per task by episode: 40/5/5 (432/55/55 overall), in `reports/split_manifest.json`.

## 3. Install

```bash
git clone <this repo> && cd so101_evo1
uv sync --extra dev                      # Python 3.12, lerobot[training,evo1,feetech]==0.6.1, mujoco (locked in uv.lock)
uv run python scripts/system_check.py    # -> reports/system_check.md
uv run python scripts/download_so101_model.py   # official SO-101 MJCF + meshes (TheRobotStudio/SO-ARM100 @ a758567)
```

Rendering is headless (EGL); set `MUJOCO_GL=egl` if your system defaults differently. System FFmpeg is optional (PyAV decodes the videos).

## 4. Inspect and validate the datasets

```bash
uv run python scripts/inspect_datasets.py --probe-videos   # ~3.8 GB download; -> reports/dataset_audit.{json,md}
uv run python scripts/validate_datasets.py                 # official LeRobotDataset; -> split + dataset manifests
uv run python scripts/aggregate_datasets.py                # merged dataset datasets/so101_11tasks (official aggregate)
uv run python scripts/compute_norm_stats.py                # train-only normalization stats
uv run python scripts/verify_aggregate.py                  # merged vs sources (pixel-identical frames, episode map)
uv run pytest -q
```

## 5. Train

```bash
# baseline (pipeline check, task_01 only)
uv run python scripts/train_act_baseline.py --config configs/act_baseline.yaml
# Evo-1 stage 1 (VLM frozen), local 6 GB profile: batch 1 x grad-accum 8, bf16, gradient checkpointing, 2 cameras, 256 px
uv run python scripts/train_evo1.py --config configs/evo1.yaml
uv run python scripts/train_evo1.py --config configs/evo1.yaml --set training.max_steps=20      # quick smoke
```

Evo-1 outputs go to `outputs/<run>/` and are exported to `checkpoints/evo1_so101/{best,latest,config.yaml,norm_stats.json,training_summary.json}`.
Every run writes `reports/experiment_<timestamp>.json` (config, versions, dataset revisions, peak RAM/VRAM).
W&B is off by default.

## 6. Evaluate in simulation

```bash
uv run python scripts/run_oracle.py --task all --profile fixed --episodes 1         # sanity: every task is solvable
uv run python scripts/evaluate_policy.py --checkpoint checkpoints/evo1_so101/best --task all --episodes 20
uv run python scripts/evaluate_policy.py --checkpoint checkpoints/evo1_so101/best --task task_01 --splits test --episodes 20 --video
uv run python scripts/evaluate_offline.py --checkpoint checkpoints/evo1_so101/best --task all --split test   # held-out REAL episodes
```

Profiles: `fixed` (nominal layout), `position` (randomized positions within the split's region) and `position_visual`
(plus moderate lighting / colour / camera noise). The `test` split uses object positions in cells never used for
training: this is the unseen-position success rate. Reports: `reports/evaluation_<name>_<profile>.{json,md}`, videos in `reports/videos/`.

## 7. Visualize

`--video` on `evaluate_policy.py` / `run_oracle.py` writes MP4s with both cameras, the instruction and the joint state.

## 8. Deploy on the real SO-101

Read `docs/real_robot_setup.md` first (ports, motor IDs, official calibration, checks).

```bash
uv run python scripts/deploy_so101.py --checkpoint checkpoints/evo1_so101/best --task task_01 --dry-run --source replay   # no hardware
uv run python scripts/deploy_so101.py --checkpoint checkpoints/evo1_so101/best --task task_01 --dry-run --source robot    # cameras + state, nothing sent
uv run python scripts/deploy_so101.py --checkpoint checkpoints/evo1_so101/best --task task_01 --real --speed-scale 0.1
```

## 9. Troubleshooting

| symptom | cause / fix |
|---|---|
| `Could not load libtorchcodec ... libavutil.so` | no system FFmpeg; LeRobot falls back to PyAV automatically (harmless). `sudo apt install ffmpeg` to use torchcodec |
| `mujoco ... EGL` errors | use `MUJOCO_GL=egl` (headless) or `MUJOCO_GL=glfw` with a display |
| CUDA OOM with Evo-1 | keep `batch_size: 1`, raise `grad_accum_steps`, keep `enable_gradient_checkpointing: true`, `image_resolution: [256, 256]`. Do not drop cameras/state/instruction to hide OOM |
| RAM pressure | `num_workers: 0`; sim data generation encodes one video at a time; never preload the dataset |
| `--real` refused | run `--dry-run --source robot` for the same checkpoint/calibration first |
| `calibration file ... not found` | `lerobot-calibrate --robot.type=so101_follower --robot.port=... --robot.id=<id>` |
| slow downloads | datasets / VLM are fetched once into the HF cache; `HF_TOKEN` raises rate limits |
