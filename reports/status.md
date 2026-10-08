# Project status

Status vocabulary: PASS / FAIL / BLOCKED / NOT YET VERIFIED.
Target machine: RTX 3050 Laptop 6 GB (compute 8.6), 12 GB RAM (10.9 GB visible to the OS), i5-12450HX, Ubuntu 26.04, CUDA 13.0, torch 2.11.0+cu130, lerobot 0.6.1.

## Phase 0–5: environment, sources, audit

| item | status | evidence |
|---|---|---|
| system check (torch CUDA, bf16) | PASS | reports/system_check.md |
| dataset integrity (11 datasets, 551 episodes, 46 videos, 2204 boundary frames decoded) | PASS | reports/dataset_audit.md, verdict PASS_WITH_EXCLUDED_EPISODES |
| dataset usable for training | PASS | 9 no-op episodes (intact, no motion) excluded from training |
| official LeRobot validation | PASS (112/112) | reports/validation.md |

## Phase 6: aggregation, train-only normalization, manifest

- **STATUS:** PASS
- **IMPLEMENTED:**
  - `scripts/aggregate_datasets.py`: official `lerobot.datasets.aggregate.aggregate_datasets`
    (`concatenate_videos=False`, `concatenate_data=False`: file copies, no re-encode) → `datasets/so101_11tasks` (3.8 GB on disk).
  - `scripts/compute_norm_stats.py`: per-episode stats from meta/episodes aggregated over the 432 train episodes with
    official `aggregate_stats` → `meta/stats.json`. The all-episode stats are kept in `meta/stats_all_episodes.json`.
  - `scripts/verify_aggregate.py`: merged dataset checked against the sources.
  - `src/utils/resources.py`: peak RSS / VRAM monitor used by all benchmarks.
- **TESTS:**
  - aggregate verification 14/14 PASS (reports/aggregate_verification.md). All 551 episodes map to their source
    episode (task string, length, exact action/state). 66 frames × 2 cameras are pixel-identical to the source.
    The train subset yields (50, 6) action chunks and never yields held-out episodes.
  - train-only stats equal a direct recomputation from 200,257 train frames (min/max diff 0, mean/std diff < 3e-5).
  - unit tests 16/16 PASS.
- **PEAK VRAM:** n/a (CPU only)
- **PEAK RAM:** aggregation 0.96 GB, verification 1.28 GB
- **BENCHMARK:** aggregation 10 s. Lazy decode of one sample (2 cameras, PyAV): see reports/aggregate_verification.md.
- **LIMITATIONS:**
  - torchcodec cannot load (no system FFmpeg). PyAV is used and decodes correctly.
  - Test episodes reach shoulder_pan 81.7°, above the train max of 75.2°, so normalized test values can slightly exceed 1.
    This is genuine held-out behaviour.
- **NEXT STEP:** Phase 7, MuJoCo SO-101 + task_01.

## Phase 7–8: MuJoCo simulation, 11 tasks, position randomization

- **STATUS:** PASS for the environment (all 11 tasks reset/observe/act/detect success, spatial split enforced).
  Oracle solvability: PASS on the fixed layout for 11/11 tasks. NOT YET VERIFIED at randomized positions for task_02–11.
- **DECISION:** Isaac Sim not used. MuJoCo local with the official SO-101 MJCF (TheRobotStudio/SO-ARM100
  `so101_new_calib_camera.xml` @ a758567, `scripts/download_so101_model.py`).
- **IMPLEMENTED:**
  - `src/simulation/action_adapter.py`: canonical degrees ↔ sim radians (round-trip tested).
  - `scene.py`: robot, table, cameras `top` 640×360 and `wrist` 640×480.
  - `assets.py`: primitive object builders.
  - `tasks.py`: 11 tasks with exact dataset instructions, success predicates and failure reasons
    (`wrong_object_in_pot`, `wrong_cube_chosen`, `wrong_order`, `*_tipped_over`, `*_fell_off_table`, `timeout: …`).
  - `placement.py`: checkerboard polar split train/val/test plus a keep-out around the resting gripper.
  - `env.py`: reset(seed), step(canonical action) at 30 Hz, profiles `fixed` / `position` / `position_visual`.
  - `oracle.py` + `ik.py`: scripted expert with closed-loop IK.
  - `scripts/run_oracle.py`, `scripts/check_sim_resets.py`.
- **TESTS:**
  - unit tests 43/43 PASS (`tests/test_simulation.py`: joint order, determinism, split, observation schema, action
    tracking, success / failure / timeout detection, all tasks reset).
  - reset stress test 1012/1012 PASS.
  - oracle task_01 randomized: train 20/20, val 20/20, test 18/20. Oracle fixed layout: 11/11 tasks.
  - MP4: `reports/videos/oracle_task_01_*.mp4`, `oracle_task_08_*.mp4`.
- **PEAK RAM:** ~0.95 GB (oracle runs). **PEAK VRAM:** n/a (CPU rendering via EGL).
- **BENCHMARK:** ~40 ms per control step including rendering of both cameras.
- **LIMITATIONS:**
  - Objects are primitives (`reports/simulation_asset_report.md`). The top camera is placed qualitatively, not calibrated.
  - Finger contact uses pads instead of the official jaw collision meshes.
  - Gripper torque is set to 50 % (LeRobot gripper config).
  - task_04 lid is placed in any split cell.
  - Large visual domain gap vs. the real videos: a policy trained on real data only is not expected to transfer
    zero-shot. Offline evaluation on held-out real episodes is reported separately.
- **NEXT STEP:** oracle at randomized positions for task_02–11; sim dataset generation (REAL_PLUS_SIM).

## Phase 9: ACT baseline (task_01)

- **STATUS:** smoke PASS (training, checkpoint save, checkpoint load, sim closed loop, offline evaluation). Full run in progress.
- **IMPLEMENTED:**
  - `src/training/launcher.py`: official `lerobot-train`, train episodes from split_manifest, RSS/VRAM monitoring,
    `reports/experiment_<ts>.json`.
  - `configs/act_baseline.yaml`, `scripts/train_act_baseline.py`.
  - `src/evaluation/policy_eval.py` + `scripts/evaluate_policy.py`: sim closed loop via the official `predict_action`.
  - `src/evaluation/offline_eval.py` + `scripts/evaluate_offline.py`: held-out real episodes, action/joint/gripper MAE,
    delta cosine, smoothness, hold-state reference.
- **TESTS (200-step smoke):**
  - loss 14.6 → 3.9. grad norm NaN only at step 100 (initial fp16 GradScaler overflow), finite afterwards.
  - offline test MAE 19.7° vs hold-state 21.9°.
  - sim 0/1 (expected at 200 steps).
- **PEAK VRAM:** 4.0 GB (nvidia-smi, training, batch 8, AMP). **PEAK RAM:** 2.7 GB (process tree, num_workers=0).
- **BENCHMARK:** ~16 samples/s training. Inference P50 3.4 ms / P95 6.7 ms per step (ACT runs once per 100-step chunk;
  first call 544 ms).
- **NEXT STEP:** full 20k-step run → offline + sim evaluation.

## Phase 10: Evo-1 (preparation)

- **STATUS:** NOT YET VERIFIED.
- **FINDING:** the `evo1-flash` branch adds gradient checkpointing (+ use_reentrant option) and a prefetching data
  pipeline to the standalone trainer. The LeRobot 0.6.1 Evo-1 policy already exposes `enable_gradient_checkpointing`
  and `gradient_checkpointing_use_reentrant`. The standalone flash trainer needs LeRobot v2.1 data and DeepSpeed, so
  it is not used.
- VLM `OpenGVLab/InternVL3-1B-hf` @ 014c0583 (revision used in the LeRobot Evo-1 docs): download in progress.

## PAUSED (2026-10-07 18:45)

All training and simulation processes were stopped gracefully (SIGINT). State and exact resume plan: `reports/pause_state.json`.
- ACT: complete checkpoint at step 10000 (`outputs/act_task01_20261007_172212/checkpoints/010000`).
- Sim data: 80 complete episodes (task_01–04). The dataset is not finalized and needs repair or verification before use.

## RESUMED (2026-10-07 ~19:50)

- **Sim data repair** (`scripts/repair_sim_dataset.py`): PASS. Both videos were complete (23138 frames = 80 episodes).
  Only 218 data rows of the partial task_05 episode were removed (backup in `datasets/sim_so101_repair_backup`).
  80 episodes / 23138 frames verified with the official loader (240 frames, both cameras, 0 bad). Tasks 01–04 were
  NOT regenerated.
- **Sim data part 2**: task_05..11 → `datasets/sim_so101_part2`, with the same deterministic seeds as the original run
  (in progress).
- **ACT checkpoint 010000** (not retrained):
  - offline task_01: test MAE 12.46° (hold-state 20.88°), gripper 9.63, delta-cosine 0.81; val MAE 10.86°.
  - sim task_01: train 0/10, test 0/10 (all timeouts). Video: the arm rises, opens the gripper and hovers, no grasp.
    Actions are in-distribution joint targets, so this is the real→sim visual domain gap.
  - The sim robot is yellow (MJCF default) while the real arm is cyan. Recolouring is a cheap gap reduction; it will be
    done together with the next sim-data regeneration.
  - ACT resume to 20k: NOT done (not needed for pipeline verification).

## Phase 10: Evo-1 stage1

- **STATUS:** smoke PASS (train, checkpoint, reload, offline inference). Full stage1 run in progress.
- **FINDING:** the official `lerobot.policies.evo1` rejects `image_resolution=256`: "must match the InternVL checkpoint's
  native image size (448)". The requested 256 px local profile is not possible without changing the architecture, so
  448 (official) is used.
- **Sweep** (`reports/evo1_sweep.md`):
  - batch 4 OOM; batch 2 best (~1.7–1.8 samples/s, 3.65–4.11 GB allocated).
  - gradient checkpointing has no effect in stage1 (the frozen VLM runs under no_grad).
- **Trainer:** exact resume (weights, optimizer, scheduler, RNG, data position) verified. An uninterrupted and a resumed
  run give bit-identical losses.
- **Inference:** 816 ms P50 / 1087 ms P95 per 50-action chunk on RTX 3050. Peak RSS 3.4 GB.
- **PEAK VRAM:** 4.11 GB allocated / ~5.1 GB nvidia-smi (batch 2). **PEAK RAM:** ~2.5 GB.
- **Running:** `outputs/evo1_so101_stage1_20261007_201435` (3000 updates, effective batch 8, ckpt + val every 250).
  - Resume: `uv run python scripts/train_evo1.py --resume outputs/evo1_so101_stage1_20261007_201435`

## Phase 15–16: simulation data + data mixtures

- **STATUS:** PASS.
- **Sim dataset** `datasets/sim_so101_all`: 220 episodes / 62,998 frames, 20 per task, train-region positions,
  oracle-generated, successes only. Built with the official aggregate from part 1 (repaired) + part 2.
  Verified: 440 frames decode, 0 errors (`reports/sim_dataset_report.md`).
- **REAL_PLUS_SIM** `datasets/mixtures/REAL_PLUS_SIM`: 432 real train + 220 sim episodes = 652 train episodes,
  263,255 frames. Stats recomputed over exactly these episodes. Real val/test stay held out.
  Loader check PASS (`reports/mixture_REAL_PLUS_SIM.json`).
- **Issue found and fixed:** the first official aggregation (11 sources, LeRobot 0.6.1) wrote
  `meta/episodes/file_index` values 0..4 while all 551 rows are in `file-000.parquet`. The training loader ignores
  this column, but a second aggregation reads it and failed. The derived dataset's column was corrected to (0, 0);
  backup in `datasets/so101_11tasks_meta_backup/`. Re-verification `scripts/verify_aggregate.py`: 14/14 PASS.
  The 2-source mixture aggregate was written correctly (single index 0).

## Phase 10: Evo-1 stage1, first full run: FAIL (undertrained)

- Run `outputs/evo1_so101_stage1_20261007_201435`: 3000 updates × effective batch 8 = 24k samples, 3.67 h.
  val loss 0.435 → 0.338. Peak VRAM 4.11 GB, peak RSS 3.95 GB. Exported to `checkpoints/evo1_so101/`.
- **Offline, held-out REAL test episodes, 11 tasks: FAIL.**
  - mean action MAE 44.3°, worse than the hold-state reference (~12°); delta-cosine ≈ 0.1 (`reports/offline_evo1_stage1_test.json`).
- **Diagnosis:**
  - Not a pipeline bug: normalization stats, 24→6 crop and input formats were checked.
  - Outputs depend on the observation: 17.8° change for a different observation under the same noise seed.
  - But the across-noise-seed spread is ~50°. The flow-matching head barely denoises yet, so samples are dominated by
    noise. Even on TRAIN frames the error is large: underfitting.
  - An earlier diagnostic claiming "identical outputs for different observations" was my indexing error (local episode
    ids used without the global offset) and has been retracted.
- **LR sweep** (250 more updates from the checkpoint): 1e-4 → val 0.362 (worse), offline ≈ 46°; 1e-5 → val 0.333,
  offline ≈ 44°. LR is not the bottleneck; sample count is (24k samples vs 640k in the official stage1 recipe).
- **Remedy in progress: cached frozen-VLM features** (stage1 only; exact up to bf16 numerics).
  - In stage1 the VLM is frozen (eval, no_grad, no augmentation), so fused tokens are deterministic per frame.
  - `scripts/cache_evo1_features.py` stores the official `_compute_fused_tokens` / `_prepare_state` / `_prepare_actions`
    outputs (valid tokens only, ~1 MB/frame, resumable shards).
  - `scripts/train_evo1_cached.py` trains integration + action head with the official model call and masked loss.
  - Equivalence check vs `Evo1Policy.forward` on identical frames and noise: |Δloss| ≤ 1.7e-3 (≤0.5 % relative, bf16 storage).
  - Throughput: VLM extraction ~2 frames/s (one-off). Head-only training ~80 samples/s at batch 32 (vs 1.8 samples/s
    end-to-end): ~45× more samples per hour.
  - Running: train frames every 10th (~20k frames, ~2.8 h, ~21 GB) + val every 30th → `datasets/evo1_feature_cache`.
  - Next: 20k head updates × batch 32 (640k samples, comparable to the official stage1 sample count), continuing from the
    current checkpoint → offline + sim evaluation.

## Phase 10: Evo-1 stage1, cached-feature training: PARTIAL

- **Run** `outputs/evo1_stage1_cached_20261008_035605`: init from the first stage1 run
  (backup `checkpoints/evo1_so101_stage1_e2e_3000`), 20k updates × batch 32 = 640k samples (31 epochs over 20,213
  cached frames), 2.4 h, 73.7 samples/s. Peak VRAM 4.48 GB, peak RSS 3.73 GB.
  - val loss (871 real val frames): 0.327 → **0.035**, still decreasing. Exported to `checkpoints/evo1_so101/`.
- **Offline, held-out REAL test episodes, 11 tasks** (4 frames/episode, horizon 50):
  - mean action MAE 44.3° → **15.0°**; delta-cosine 0.23–0.56. Hold-state reference ≈ 12.0° on this protocol.
  - Better than the hold reference only on task_02: NOT yet better than doing nothing on most tasks.
- **Fair ACT comparison, task_01** (8 frames/episode, horizon 50, same frames):

  | model | test MAE | val MAE | gripper | cosine |
  |---|---:|---:|---:|---:|
  | hold-state | 11.1 | 11.2 | – | – |
  | ACT 10k (task_01 only, ~80k samples) | 8.9 | 7.5 | 6.8 | 0.77 |
  | Evo-1 stage1 cached (11 tasks) | 10.5 | 11.1 | 8.4 | 0.65 |

  Note: earlier ACT numbers (12.5°) used horizon 100 and are not comparable.
- **NEXT:** more distinct frames (cache offset 5 → 40k frames) and continue stage1. Then stage-2-lite, then sim eval.

## PAUSED (2026-10-08 06:40)

No project process is running. Full resume plan: `reports/pause_state.json`.
- Phase: 10 (Evo-1 stage1 via cached frozen-VLM features). Cached run #1 complete; caching extra frames for run #2.
- Feature cache `datasets/evo1_feature_cache` (21 GB):
  - train_s10_o0: 79/79 shards, 20,213 frames (complete).
  - val_s30_o0: 4/4 shards, 871 frames (complete).
  - **train_s10_o5: 4/79 shards (last completed shard_00003), 1,024 / 20,007 frames.** No temp files left.
- Latest Evo-1 checkpoint: `checkpoints/evo1_so101/best` (cached run #1, step 20000, val 0.0352, offline test MAE 15.0°).
  Backup of the end-to-end run: `checkpoints/evo1_so101_stage1_e2e_3000`.
- Simulation data (final, do not regenerate): `datasets/sim_so101_all` (220 eps) and mixture `datasets/mixtures/REAL_PLUS_SIM`.
- Next: resume caching train_s10_o5 (skips finished shards) → stage1 run #2 on both parts → offline eval.

## PAUSED (2026-10-08 12:21)

No project process is running (GPU idle). Full resume plan: `reports/pause_state.json`.
- Feature cache: **79/79 + 79/79 train shards (40,220 frames) + 4/4 val shards: COMPLETE** (39 GB, do not recompute).
- Stage1 cached **run 2** `outputs/evo1_stage1_cached_r2_20261008_121256`, stopped at **step 1000 / 20000**.
  - Latest complete checkpoint: `checkpoints/latest` at step 1000 (weights + optimizer/scheduler/RNG/data position).
  - val loss 0.0352 (init) → 0.0465 at step 1000 (new frames, LR warm-up).
  - Data: 158 shards, REAL_ONLY, 11 tasks. Precision: bf16 cached tokens + bf16 autocast, fp32 weights/loss.
  - Batch 32 (effective 32), lr 1e-5.
  - Resume: `uv run python scripts/train_evo1_cached.py --resume outputs/evo1_stage1_cached_r2_20261008_121256`.
- `checkpoints/evo1_so101` still holds run 1. Backups: `evo1_so101_stage1_cached_r1`, `evo1_so101_stage1_e2e_3000`.
