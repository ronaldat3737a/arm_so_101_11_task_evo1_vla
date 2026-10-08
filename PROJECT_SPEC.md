# PROJECT_SPEC — SO-101 × Evo-1, 11 tasks, sim → real

## Goal

```
HF datasets (11 × SO-101, LeRobot v3.0) → audit/validate → episode split → merged LeRobotDataset
→ Evo-1 (InternVL3-1B VLM + flow-matching head, official LeRobot policy) stage1 [→ stage2]
→ simulation eval on unseen object positions → real SO-101 (dry-run → low-speed real)
```

Primary metric: **task success rate on unseen object positions** (per task + overall), never loss alone.

## Verified facts (with sources)

| fact | value | source |
|------|-------|--------|
| datasets | 11 repos, all `codebase_version v3.0`, `robot_type so_follower`, 30 fps | reports/dataset_audit.md |
| state/action | float32[6] `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper].pos`, degrees + gripper 0–100, absolute joint targets (state lags action by ~4 frames) | audit + docs/so101_interface.md |
| cameras | `observation.images.top` 360×640, `observation.images.wrist` 480×640, AV1 | audit |
| episodes | 50 per dataset (task_05: 51). 9 episodes over 7 datasets contain no motion | audit |
| instructions | one task string per dataset, natural language, unique across datasets | audit, configs/tasks.yaml |
| Evo-1 robot pretraining | none: "strong performance without pretraining on robot data" (arXiv 2511.04555). "Pretrained Evo-1" = pretrained VLM `OpenGVLab/InternVL3-1B-hf` + official two-stage finetune | paper abstract, LeRobot docs evo1.mdx |
| Evo-1 in LeRobot | `lerobot.policies.evo1` in `lerobot==0.6.1` (PyPI 2026-08-03), `policy.type=evo1`, stage1/stage2, MIN_MAX norm from dataset stats, pad to 24-D, 448×448, chunk 50 | wheel source |
| standalone Evo-1 repo | `main`/`evo1-flash`: LeRobot **v2.1** datasets only, DeepSpeed, `arm_key`/`dataset_key` norm_stats. `evo1-lerobot`: vendored LeRobot fork (py3.10, v3 datasets) | github.com/MINT-SJTU/Evo-1 @ 5fd14b0 / fb177e4 |
| LeIsaac | py3.11, Isaac Sim 5.1.0, IsaacLab 2.3.0, torch 2.7.0, `lerobot==0.4.2` extra, SO-101 USD asset, single-arm template (front+wrist cams 640×480) | LeIsaac @ 24d3bcd |
| Isaac Sim 5.1 min spec | 32 GB RAM, 16 GB VRAM (RTX 4080), Ubuntu 22.04/24.04 | docs.isaacsim.omniverse.nvidia.com/5.1.0 |

## Decisions

- **D1 Evo-1 implementation = upstream LeRobot 0.6.1 `policy.type=evo1`.** Reasons: (a) our datasets are v3.0. The
  standalone Evo-1 trainer only reads v2.1, so it would need a lossy down-conversion. (b) It is the official
  integration (Evo-1 README, 2026-07-05: "added to the official LeRobot framework"). (c) It is a pinned PyPI release with
  SO-101 robot, calibration, recording and rollout in the same package. The `evo1-lerobot` branch is an
  older vendored fork of the same integration.
- **D2 Initialization.** Evo-1 has no robot-pretrained base. Start from `OpenGVLab/InternVL3-1B-hf` (pinned revision) and
  run stage1 (VLM frozen, action head trained) → optionally stage2 (full). The `MINT-SJTU/Evo1_SO100` checkpoint
  (author DeepSpeed format, SO-100, different camera names) is **not** loadable by the LeRobot policy (no converter
  exists). Writing one would mean re-implementing weight mapping, so it is not done.
- **D3 Multi-dataset training.** `lerobot-train` takes one `repo_id` (MultiLeRobotDataset is disabled in 0.6.1), so the
  11 datasets are merged with the official `lerobot.datasets.aggregate.aggregate_datasets` into one local
  LeRobotDataset (`datasets/so101_11tasks`). Train episodes are selected with `--dataset.episodes=[...]`.
- **D4 Split.** Per task, after removing static episodes: 40 train / 5 val / 5 test when ≥50 usable, otherwise
  val=test=5 and train = rest. Seeded shuffle, frozen in reports/split_manifest.json.
- **D5 Normalization.** Evo-1 uses MIN_MAX on `meta/stats.json`. After aggregation that file covers all episodes. To keep
  test episodes out of the stats, `compute_norm_stats.py` recomputes stats from the train episodes only and writes them
  where LeRobot reads them. No custom normalization scheme is introduced.
- **D6 Instructions.** The exact dataset task strings (configs/tasks.yaml). No paraphrasing.
- **D7 Two environments.** `.venv` (py3.12, lerobot 0.6.1): data, training, real deploy. Sim env (py3.11, Isaac/LeIsaac,
  lerobot 0.4.2): simulation only. The policy runs in `.venv` and is served to the sim through a process boundary.
- **D8 Sim joint mapping.** Body joints: rad = deg2rad(dataset degrees). Gripper: LeIsaac linear map (0..100 → −10..100°).
  LeIsaac's built-in converter assumes the −100..100 motor mode and is not used.

## Phase status

| phase | status | notes |
|-------|--------|-------|
| 0 inspect directory | DONE | empty directory, project created in `so101_evo1/` |
| 1 LeRobot SO-101 | DONE | docs/so101_interface.md |
| 2 Evo-1 + LeRobot integration | DONE (inspection) | D1, D2 |
| 3 HF collection, 11 datasets | DONE | metadata, parquet, 46 videos (3.8 GB, sizes match Hub), 2204 episode-boundary frames decoded, 0 errors |
| 4 system check | DONE | torch 2.11.0+cu130, CUDA OK, bf16 OK on RTX 3050 6 GB; reports/system_check.md |
| 5 dataset audit | DONE | reports/dataset_audit.{json,md}: FAIL due to 9 static episodes (documented, excluded at split) |
| 6 validation / conversion / split / manifest | PARTIAL | validation PASS 112/112 with official LeRobotDataset; split 432/55/55 (+9 excluded) in reports/split_manifest.json; dataset_manifest.json. TODO: merge with aggregate_datasets, train-only norm stats |
| 7 simulation env (11 tasks) | **BLOCKED** | Isaac Sim min spec not met on this machine; 3D assets for pot/lid/banana/board/hollow cube absent |
| 8 position randomization | TODO (depends on 7) | |
| 9 ACT baseline | TODO | `policy.type=act`, fits 6 GB |
| 10 Evo-1 fine-tune wrapper | TODO | stage1 feasible on 6 GB (to be measured); stage2 needs ≥24 GB |
| 11 sim evaluation | TODO (depends on 7) | |
| 12 real SO-101 inference | TODO | needs hardware |
| 13 end-to-end tests | TODO | |

## Known warnings

- torchcodec 0.11.1 cannot load (system FFmpeg libs `libavdevice.so.62` missing), so LeRobot falls back to the `pyav` backend
  (decode works, ~15-20 ms/frame). Optional speed-up: install system FFmpeg 8 (`sudo apt install ffmpeg`), which needs sudo.

## Open blockers

- **BLOCKED (simulation hardware):** Isaac Sim 5.1 needs ≥32 GB RAM / ≥16 GB VRAM / Ubuntu 22.04–24.04. This machine has
  10 GB / 6 GB / Ubuntu 26.04. Running Isaac Sim and Evo-1 inference together on 6 GB is not feasible.
- **BLOCKED (3D assets):** no meshes for steel pot, lid, banana, wooden board, orange hollow cube, height-varying blue cubes.
  The plan is primitives/approximations (reports/simulation_asset_report.md), unless real meshes are provided.
- **Stage2 Evo-1 full finetune** does not fit 6 GB (≈0.8 B trainable params × AdamW fp32 ≈ 13 GB states alone).
