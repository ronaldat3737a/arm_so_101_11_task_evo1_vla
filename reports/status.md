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

## RESUMED (2026-10-08, after the 12:21 pause)

- Checked before resuming: `checkpoints/latest` is at step 1000 and `training_state.pt` is complete. No project process was
  running and the GPU was idle. The cache was not recomputed.
- Run 2 was resumed with `--resume outputs/evo1_stage1_cached_r2_20261008_121256`. The log confirms
  `{"resumed_from_step": 1000}`. Step 1100: loss 0.0435, lr 9.98e-6 (cosine schedule continues), 75.5 samples/s,
  VRAM alloc 4.48 GB.
- **STATUS:** run 2 training is IN PROGRESS (about 2.2 h left). Next: offline eval on all 11 tasks of the test split,
  compared with run 1 (15.0°) and the hold-state baseline.

## Phase 10: Evo-1 stage1, cached run 2 (REAL_ONLY, 40,220 cached frames): RESULT

- **Training: PASS.**
  - Resumed exactly from step 1000 (`resumed_from_step: 1000`, lr schedule continued) and finished at 20000/20000.
    Total 640k samples, about 15.9 epochs over 40,220 frames.
  - Speed 73.5 samples/s. Peak VRAM alloc 4.48 GB, taken from the training log.
  - Peak RSS of the training segment was not recorded: the script crashed while writing the summary (relative-path bug
    in `--resume`, now fixed). The checkpoints were already written. The summary and export were done afterwards by
    the same script with no further training step.
  - Exported to `checkpoints/evo1_so101/` (best = step 20000). Run 1 backup: `checkpoints/evo1_so101_stage1_cached_r1`.
- **Val loss** (871 real val frames): 0.0352 (init = run 1) → 0.0465 at step 1000 (lr warm-up) → **0.0280** at step 20000.
  It plateaued from step 18000.
- **Offline, real test episodes, 11 tasks** (`reports/offline_evo1_stage1_cached_r2_test.json`; 4 frames/episode,
  horizon 50, same protocol as run 1):

  | metric | run 1 | run 2 | hold-state |
  |---|---:|---:|---:|
  | mean action MAE (deg) | 15.03 | **11.95** | 11.96 |
  | delta cosine | 0.406 | 0.532 | – |
  | gripper MAE | 8.47 | 6.39 | – |
  | first-step MAE | 6.99 | 5.08 | – |
  | tasks better than hold | 1/11 | **5/11** | – |

  | task | hold | run 1 | run 2 | run 2 − hold | cos r2 |
  |---|---:|---:|---:|---:|---:|
  | task_01 | 8.92 | 12.85 | 10.35 | +1.43 | 0.580 |
  | task_02 | 13.54 | 12.99 | 11.25 | −2.29 | 0.655 |
  | task_03 | 8.07 | 10.76 | 8.57 | +0.50 | 0.520 |
  | task_04 | 14.36 | 18.16 | 12.49 | −1.87 | 0.614 |
  | task_05 | 11.22 | 15.83 | 12.31 | +1.08 | 0.476 |
  | task_06 | 9.07 | 11.64 | 9.52 | +0.45 | 0.416 |
  | task_07 | 11.31 | 14.80 | 9.60 | −1.71 | 0.557 |
  | task_08 | 18.35 | 20.47 | 16.73 | −1.61 | 0.446 |
  | task_09 | 10.67 | 16.50 | 11.47 | +0.80 | 0.466 |
  | task_10 | 14.43 | 14.91 | 14.17 | −0.26 | 0.544 |
  | task_11 | 11.67 | 16.42 | 15.00 | +3.33 | 0.574 |

  - Inference P50 796 ms / P95 801 ms per 50-action chunk. Eval peak RSS 3.21 GB, VRAM alloc 3.37 GB.
- **Policy quality: NOT ACHIEVED.** Run 2 is clearly better than run 1 on every metric, but the mean MAE equals the
  "stay still" reference, and only 5/11 tasks beat it. The lower val loss did not translate into a policy that clearly
  beats doing nothing.
- **NEXT:** diagnostic on train episodes (underfitting vs generalization), then decide what Stage 2 Lite trains.

### Diagnostics of run 2 (evaluation only; model unchanged)

- **Train episodes** (`reports/offline_evo1_stage1_cached_r2_train.json`, 2 frames/episode, all 432 train episodes):
  - MAE 11.29° vs hold 12.69°. Better than hold on 9/11 tasks. Cosine 0.56–0.72.
  - For comparison, test: 11.95° vs 11.96°, 5/11.
  - Train and test differ by only about 0.7°, so this is not overfitting. The model fits even its training episodes
    only moderately: **underfitting / limited precision.**
- **Sampling spread** (`scripts/diag_evo1_sampling.py` → `reports/diag_sampling_evo1_stage1_cached_r2_test.json`;
  test, 2 frames/episode, 8 flow-matching samples per frame from the same fused tokens):

  | | single sample | mean of 8 samples | spread between samples | hold |
  |---|---:|---:|---:|---:|
  | mean MAE (deg) | 12.61 | 10.62 | 7.35 | 12.56 |
  | tasks better than hold | 6/11 | 7/11 | | |

  - Error vs horizon step (single / mean-of-8 / hold): step 0 4.7 / 3.4 / 2.0; step 9 7.5 / 5.8 / 6.2;
    step 29 15.0 / 13.2 / 14.8; step 49 17.8 / 14.6 / 21.7.
  - Limitation: with 2 frames per episode, `linspace` picks the first and last frame of each episode (mostly a resting
    arm), so this hold value differs from the 4-frame protocol.
  - Reading: about 16% of the single-sample error is flow-matching sample spread. Evo-1 chunks are also much less
    smooth than ACT's (mean |2nd difference| 3.8–5.7 vs 0.84 on task_01).
  - The first-step offset of about 5° also exists for ACT (4.96°), so it is a data property (action leads state), not
    an Evo-1 defect.

## Phase 11: Stage 2 Lite

- Profiles from the spec: head_only, projector_action, selective_llm, qlora, clp_selective.
  - **head_only** = stage1 (run 2 above). Reference: test MAE 11.95°.
  - **selective_llm** (running first): train language-model layers 12–13 of 14, the final norm and the action head.
    Vision tower, projector and LM layers 0–11 stay frozen. Architecture is unchanged.
    - Implementation:
      - `src/training/selective_llm.py`
      - `scripts/cache_evo1_hidden.py`: hidden states entering LM layer 10, same frames as `train_s10_o0` plus val.
      - `scripts/train_evo1_selective.py`
    - Exactness: rebuilding the fused tokens from the cached layer-10 hidden states with the official Qwen2 layers
      gives relative error **0.0** against `_compute_fused_tokens` (4 batches, checked while caching).
    - Fair control: head_only continued for the same number of steps on the same frames (existing cache
      `train_s10_o0`).
  - **projector_action**: needs gradients through all 14 LM layers, and the pre-projector vision cache would be about
    4 MB/frame (about 80 GB for 20k frames). Without a cache it runs at about 1.8 samples/s. Decided after
    selective_llm.
  - **qlora**: `peft` / `bitsandbytes` are not installed. QLoRA is not needed for memory at this model size (the
    truncated LM fits in bf16). NOT YET RUN.
  - **clp_selective**: depends on CLP. NOT YET RUN.
- **STATUS:** IN PROGRESS. The hidden-state cache is being built (about 3 h, 21,084 frames, about 21 GB).

### Stage 2 Lite, selective_llm vs head_only control: RESULT (2026-10-09)

Both runs start from run 2 (`checkpoints/evo1_so101/best`) and use the same frames (`train_s10_o0`, 20,213 real
train frames, REAL_ONLY), 10,000 steps × batch 32, lr 1e-5 cosine, warm-up 500, seed 2000. Offline evaluation uses
the same protocol as run 1 and run 2 (test, 4 frames/episode, horizon 50, single sample).

| | run 2 (head_only, 20k) | head_only +10k (control) | **selective_llm +10k** | hold |
|---|---:|---:|---:|---:|
| trainable params | 122.0 M | 122.0 M | 151.9 M (+29.8 M LM layers 12–13 + norm) | |
| best val loss | 0.02796 | 0.02665 | **0.02618** | |
| test action MAE (deg) | 11.95 | 11.86 | **11.26** | 11.96 |
| tasks better than hold | 5/11 | 6/11 | **9/11** | |
| delta cosine | 0.532 | 0.553 | **0.587** | |
| gripper MAE | 6.39 | 6.05 | **5.87** | |
| first-step MAE | 5.08 | 4.73 | **4.53** | |
| smoothness (mean abs 2nd diff) | 4.19 | 3.63 | **3.60** | |
| train speed (samples/s) | 73.5 | 72.7 | 27.7 | |
| wall time | | 1.23 h | 3.22 h | |
| peak VRAM alloc (train) | 4.48 GB | 4.48 GB | 4.13 GB | |
| peak RSS (train) | n/a | 2.78 GB | 3.71 GB (5.25 GB in the 2-step smoke run) | |
| inference P50 per 50-step chunk | 796 ms | 797 ms | 798 ms | |

Per task (test MAE, deg):

| task | hold | run 2 | head_only +10k | selective_llm |
|---|---:|---:|---:|---:|
| task_01 | 8.92 | 10.35 | 10.42 | 10.05 |
| task_02 | 13.54 | 11.25 | 10.73 | 10.06 |
| task_03 | 8.07 | 8.57 | 7.55 | 7.96 |
| task_04 | 14.36 | 12.49 | 13.62 | 14.35 |
| task_05 | 11.22 | 12.31 | 10.85 | 11.45 |
| task_06 | 9.07 | 9.52 | 9.60 | 8.08 |
| task_07 | 11.31 | 9.60 | 9.26 | 10.06 |
| task_08 | 18.35 | 16.73 | 18.60 | 16.66 |
| task_09 | 10.67 | 11.47 | 11.81 | 10.64 |
| task_10 | 14.43 | 14.17 | 13.19 | 13.25 |
| task_11 | 11.67 | 15.00 | 14.83 | 11.28 |

- **STATUS selective_llm: PASS** (implemented, exactness-verified, trained, benchmarked).
  - Against the equal-budget control it is better on every aggregate metric: MAE −0.60°, 9/11 vs 6/11 tasks better
    than hold. So unfreezing the top 2 LM layers helps beyond simply training the head longer.
- **Policy quality: still NOT ACHIEVED.**
  - The best model is only 0.7° (6%) better than "stay still" on average.
  - Each configuration was evaluated once with single flow-matching samples. Per-task differences below about 1° are
    within evaluation noise; the aggregate trend is consistent across all metrics.
- Exports: `checkpoints/evo1_so101_s2lite_selective_llm/{best,latest}` (current best Evo-1, BF16 reference
  candidate) and `checkpoints/evo1_so101_s2lite_head_only/`.
  - Reports: `reports/offline_evo1_s2lite_{selective_llm,head_only}_test.json`.
  - The trained LM layers are stored in fp32 and loaded into the bf16 VLM; this was verified exact.
- Not run: projector_action (cost, see above), qlora (`peft` / `bitsandbytes` not installed), clp_selective (needs
  CLP).
  - Deeper selective_llm (train layers 10–13) is possible with the existing L10 cache.
- **NEXT:** simulation evaluation of selective_llm (closed loop, 11 tasks, unseen test positions).

## Phase 12: simulation evaluation of the best Evo-1 (stage-2-lite selective_llm, REAL_ONLY)

- **Command:** `scripts/evaluate_policy.py --checkpoint checkpoints/evo1_so101_s2lite_selective_llm/best --task all
  --splits test --profile position --episodes 5 --video`.
- **Reports:** `reports/evaluation_evo1_s2lite_selective_llm_position.{json,md}`.
- **Videos:** `reports/videos/evo1_s2lite_selective_llm_task_*_position_test_seed1000.mp4` (11 files).
- **Evaluation run: PASS** (all 11 tasks, unseen test positions, closed loop, official `predict_action` path).
- **Task success: 0/55 (0 %): NOT ACHIEVED.**
  - 53 timeouts, 1 `lid_fell_off_table` (task_04) and 1 `wrong_order` (task_11: a block was taken out of the box,
    but the wrong one).
  - Videos: the arm moves around the workspace but does not go to the target object. The wrist camera sometimes sees
    the target, but no grasp follows.
  - Same pattern as ACT (0/20): a REAL_ONLY policy does not transfer zero-shot to the MuJoCo rendering. The robot
    colour, primitive objects and the uncalibrated top camera all differ from the real videos.
- **Resources:** inference P50 5.2 ms per step (one 50-action chunk about every 1.7 s; chunk call about 0.8 s, max
  1.45 s). Peak RSS 4.80 GB, peak VRAM alloc 3.42 GB.
- **Consequence:** the BF16 reference is not yet "good" on task success. The cheapest lever in the plan is
  **REAL_PLUS_SIM**: the 220 oracle sim episodes (`datasets/sim_so101_all`, train-region positions) can be added to
  the cached selective_llm training at about 1 h of extra caching (stride 10 → about 6.3k frames).

## Phase 13: CKA / sensitivity (on the BF16 selective_llm model)

- **STATUS: PASS** (measured, evaluation only).
- **Script:** `scripts/analyze_cka_sensitivity.py`.
- **Reports:** `reports/cka_sensitivity_evo1_s2lite_selective_llm.{json,md}`.
- **Frames:** 132 real VAL frames (12/task) for CKA; 66 of them (6/task) for the skip-layer sweep.
  - The sweep replaces a layer by identity and measures the official flow-matching loss and the MAE of one sampled
    chunk, with paired noise across configurations.
  - Base on these frames: loss 0.0291, MAE 11.84°.
- **Resources:** 2.0 h. Peak RSS 4.42 GB, VRAM alloc 4.40 GB.
- **CKA finding:** raw linear CKA is unusable for the language model. One "massive activation" (attention-sink) token
  per frame appears at layer 2 and dominates the covariance (raw CKA: 0.058 at layer 2, then 1.0000 for layers 4–13).
  Excluding tokens whose norm is more than 10× the median removes exactly 1 token/frame (132 of 73,056). Robust CKA is
  then 0.975–0.995 for all LM layers. Vision robust CKA: layer 23 0.644, layer 3 0.885, others 0.93–0.98.
- **CKA alone does not identify prunable layers.** The lowest robust CKA in the LM (layers 4, 5) belongs to the least
  sensitive layers. The skip-layer sweep is therefore the primary criterion for CLP, and CKA is reported alongside.
- **Sensitivity** (ΔMAE in deg when the layer is skipped):
  - Language model:
    - critical: layer 0 (+16.2), 1 (+4.4), 2 (+3.9), 10 (+2.5).
    - least sensitive: **3 (+0.03), 4 (−0.19), 5 (+0.13)**.
    - then 7 (+0.52), 6 (+0.56), 8 (+0.60), 13 (+0.62).
  - Vision encoder:
    - critical: layer 0 (+10.6).
    - least sensitive: **1 (+0.52, Δloss +0.0002)**, 20 (+0.81), 2 (+0.88), 4 (+1.02).
    - middle and late layers 8–17 are the most sensitive (+2.3 to +3.6).
  - Noise: 66 frames, one sample each. Paired noise removes most sampling variance, but differences below about 0.5°
    are not significant.
- **CLP candidates** (prune one at a time, re-benchmark, continue only if acceptable):
  - LM: layer 4 → 3 → 5.
  - Vision: layer 1.
- **Quantization hint:** vision encoder layer 0, LM layers 0–2 and the vision middle layers are the most sensitive, so
  they are the first candidates to keep at higher precision.

## Phase 14: REAL_PLUS_SIM, selective_llm (training done)

- **Sim hidden cache:** `sim_s10_o0`, 6,400 frames of `datasets/sim_so101_all`, 25/25 shards, verification
  rel_err 0.0.
- **Training: PASS.**
  - Run `outputs/evo1_s2lite_selective_llm_rps_20261009_114711`: init run 2, layers 12–13 + head, 10,000 steps × 32,
    104 shards (79 real + 25 sim).
  - Best val loss on real val frames 0.02629 (REAL_ONLY selective_llm: 0.02618).
  - 3.24 h. Peak RSS 4.84 GB, VRAM 4.13 GB.
  - Export: `checkpoints/evo1_so101_s2lite_selective_llm_rps/`.
- **Offline and sim evaluation: NOT YET VERIFIED.** They were stopped by the pause; the commands are in
  pause_state.json.

## PAUSED (2026-10-09 15:05)

No project process is running (GPU 12 MiB). Resume plan: `reports/pause_state.json`.
- Complete and not to be redone: all caches, stage1 run 2, selective_llm, the head_only control, CKA/sensitivity,
  and REAL_PLUS_SIM training.
- Next on resume: offline eval + sim eval of `checkpoints/evo1_so101_s2lite_selective_llm_rps/best`, then CLP.

## RESUMED (2026-10-10T00:58:35+07:00)

- Checked: REAL_PLUS_SIM checkpoint present, no project process running, GPU idle. Nothing retrained or recomputed.
- Running: offline test eval, then sim eval of `checkpoints/evo1_so101_s2lite_selective_llm_rps/best`.

### REAL_PLUS_SIM vs REAL_ONLY (selective_llm, same init / steps / hyper-parameters)

| | REAL_ONLY | REAL_PLUS_SIM (+6,400 sim frames, 24 %) | hold |
|---|---:|---:|---:|
| best val loss (real val) | 0.02618 | 0.02629 | |
| offline test MAE, real (deg) | **11.26** | 12.05 | 11.96 |
| tasks better than hold (real) | **9/11** | 5/11 | |
| delta cosine / gripper MAE | 0.587 / 5.87 | 0.556 / 6.33 | |
| sim success, unseen test positions | 0/55 | 0/55 | |
| sim episodes ending by object interaction (not timeout) | 2/55 | **26/55** | |

- REAL_PLUS_SIM sim failures:
  - pot_tipped_over 18;
  - wrong_order 5 (task_11: a block taken out of the box, wrong one);
  - wrong_object_in_pot 1;
  - banana_fell_off_table 1;
  - timeouts 30.
- Video `reports/videos/evo1_s2lite_selective_llm_rps_task_05_*`: the arm goes straight to the pot and reaches
  inside towards the blue cube, then knocks the pot over.
- The REAL_ONLY policy wanders around the workspace.
- **Reading:**
  - Adding sim data makes the policy task-directed in simulation: it finds and approaches the right objects.
  - It is not yet precise enough to complete a task.
  - It costs about 0.8° of offline accuracy on real test episodes (11.26 → 12.05).
- **STATUS:**
  - REAL_ONLY vs REAL_PLUS_SIM comparison: **PASS** (measured).
  - Sim task success: **NOT ACHIEVED** for both.
- **BF16 reference:**
  - REAL_ONLY selective_llm for real-robot action quality.
  - REAL_PLUS_SIM is the only one with task-directed sim behaviour.
  - Neither is yet a "good" reference by the task-success criterion.

### REAL_PLUS_SIM v2: more sim frames (sim_s10_o0 + sim_s10_o5, 12,800 frames, about 39 %)

- **Cache and training: PASS.**
  - Cache `sim_s10_o5`: 25/25 shards, rel_err 0.0.
  - Training 10,000 steps on 129 shards. Best val 0.02669. 3.24 h, peak RSS 4.69 GB, VRAM 4.13 GB.
  - Export: `checkpoints/evo1_so101_s2lite_selective_llm_rps2/`.

| | REAL_ONLY | REAL_PLUS_SIM (24 % sim) | REAL_PLUS_SIM v2 (39 % sim) |
|---|---:|---:|---:|
| offline test MAE, real (deg) | **11.26** | 12.04 | 12.40 |
| tasks better than hold | **9/11** | 5/11 | 4/11 |
| sim success, unseen positions | 0/55 | 0/55 | 0/55 |
| sim episodes ending by object interaction | 2/55 | 25/55 | 24/55 |

- v2 sim failures:
  - pot_tipped_over 19 (task_01 5/5, task_05 5/5, task_06 5/5);
  - wrong_order 5 (task_11);
  - timeouts 31.
- **Result:** more frames from the same 220 sim episodes (offset 5 adds new frames, not new demonstrations) do not
  help in sim and cost more real accuracy. **NOT ACHIEVED.**
- **Reading:**
  - The sim bottleneck is demonstration diversity and precision: there are only 20 oracle episodes per task, all in
    the train region, while evaluation uses unseen test positions.
  - The policy finds the right object but contact is too coarse; the pot is hit and tipped.
  - The pot-transfer tasks are the closest to working.
- **Decision needed:** options are more NEW sim episodes (the existing 220 stay untouched), or CLP / quantization on
  the REAL_ONLY reference.

## Phase 15: CLP (CLP-inspired adaptation to Evo-1, not official Evo-1 CLP), started

- User decision (2026-10-10): go to CLP on the REAL_ONLY selective_llm reference (11.26°).
- `scripts/clp_prune.py`: removes language-model layers, renumbers the remaining ones and sets `vlm_num_layers`.
  The official `Evo1Policy.from_pretrained` loads the result strictly without code changes (verified).
  Vision-encoder layers cannot be pruned this way: their count comes from the base VLM config.
- `checkpoints/evo1_clp/lm_drop4` created: 13 LM layers, −14.9 M params. Benchmark NOT YET VERIFIED.
- Acceptance per prune step: offline test MAE ≤ reference + 0.3°.

## PAUSED (2026-10-10)

No project process is running (GPU idle). Resume plan: `reports/pause_state.json` → `clp.next`.

## RESUMED (2026-10-10): CLP loop running

- `scripts/clp_run.py --order 4,3,5` (resumable). Each step prunes the reference with the cumulative drop list, runs
  the offline test eval and writes `reports/clp_progress.json` immediately.
- On interruption, rerun the same command: finished steps are skipped.

### CLP step 1 result + latency profile

- **CLP drop LM layer 4** (no recovery fine-tune; `reports/clp_progress.json`, `reports/offline_evo1_clp_lm_drop4_test.json`):
  - test MAE 11.80° vs reference 11.26° (**+0.54° > tol 0.3 → rejected**). 7/11 tasks better than hold (vs 9/11).
  - Latency 791 vs 798 ms (−1 %). The 66-frame val skip sweep (−0.19°) did not hold on the full test set.
  - The loop stopped as designed; layers 3 and 5 were not tried.
- **Latency profile** (`scripts/profile_evo1_latency.py` → `reports/latency_profile_evo1_s2lite_selective_llm.json`;
  RTX 3050, median of 20 runs, one 50-action chunk):

  | component | ms | share |
  |---|---:|---:|
  | vision tower + projector | 409 | 52 % |
  | action head, 32 flow steps | 310 | 39 % |
  | language model, 14 layers | 67 | 8 % |
  | full chunk | 791 | |

  - **LM-layer CLP cannot give a meaningful speed-up** (about 5 ms/layer). On this model, **CLP on the LM: NOT
    WORTHWHILE** (accuracy cost without recovery, ≤ 1 % speed).
  - The vision tower runs **3 views (`max_views=3`) although only 2 cameras exist**. The third is a zero image whose
    tokens are masked, so about 1/3 of vision time is wasted. `max_views=2` changes the prompt and needs retraining
    (not done).
  - Vision-layer pruning is not possible via the official config (BLOCKED without an architecture change).
  - The head cost scales with `num_inference_timesteps` (official inference-time config field): sweep below.

### Inference sampling steps (`num_inference_timesteps`, official config field, no retraining)

Reports: `reports/offline_evo1_s2lite_selective_llm{,_steps16,_steps8}_test.json`.

| steps | test MAE | tasks > hold | cosine | smoothness | latency P50 per chunk |
|---:|---:|---:|---:|---:|---:|
| 32 (default) | 11.26 | 9/11 | 0.587 | 3.60 | 798 ms |
| 16 | 11.61 | 6/11 | 0.562 | 2.70 | 642 ms |
| **8** | **11.24** | 7/11 | 0.569 | **1.77** | **565 ms (−29 %)** |

- **PASS (measured).** With 8 steps the MAE is unchanged within evaluation noise (16 steps scoring worse than 8 shows
  the noise level is about 0.3–0.4°). Chunks are 2× smoother and latency is 29 % lower.
- `num_inference_timesteps=8` is the inference-setting candidate for BEST_LOCAL. Sim confirmation: NOT YET VERIFIED.
- **Quantization prerequisites:**
  - `torchao` / `bitsandbytes` are not installed. PyTorch's built-in dynamic int8 quantization is CPU-only.
  - GPU W8A8 / W4A8 therefore needs a new dependency (e.g. `torchao`).
  - FoldQuantVLA supports only GR00T / π0.5 and native INT4 targets sm_87/sm_89 (this GPU is sm_86): W4A4 stays
    BLOCKED unless verified otherwise.

## Phase 16: quantization (W8A8), in progress

- User approved adding `torchao` (0.18.0, `uv add torchao`). torch 2.11 and lerobot 0.6.1 are unchanged.
- **Op-level probe on sm_86** (1024→4096→1024 MLP, 1100 tokens):

  | config | time | rel. error |
  |---|---:|---:|
  | BF16 eager | 1.60 ms | |
  | BF16 compiled | 1.60 ms | |
  | W8A8 (`Int8DynamicActivationInt8WeightConfig`) eager | 8.77 ms | 1.7 % |
  | **W8A8 compiled** | **0.87 ms** | 1.7 % |
  | int8 weight-only compiled | 133 ms (broken kernel path) | |
  | W4A8 (`Int8DynamicActivationIntxWeightConfig`, int4) compiled | 1.85 ms | 11.4 % |
  | int4 weight-only | needs `mslk` (not installed) | |

- **W4A8:** the available torchao path is slower than BF16 with 11 % layer error on this GPU. NOT a useful candidate.
  Still to be benchmarked end to end on a component, or declared BLOCKED.
- **End-to-end latency profile** (8 flow steps, `reports/latency_profile_prof_*.json`):
  - BF16 eager 560 ms/chunk (vision 409).
  - BF16 compiled 399 ms (vision 264).
  - **W8A8 vision + compile 332 ms (vision 196)**.
- `src/quantization/torchao_quant.py` schemes: bf16, w8a8_vision, w8a8_vis_proj, w8a8_vis_lm, w8a8_all.
  - Plain `nn.Linear` only: `nn.MultiheadAttention.out_proj` in the action head is incompatible with Int8Tensor.
- Running: full offline test benchmark per scheme, all compiled, 8 steps. Resumable loop; command in pause_state.json.
- Note: peak RSS during torch.compile reached 8.1 GB in a smoke test (10 GB machine).

### W8A8 end-to-end results (selective_llm REAL_ONLY, 8 flow steps, offline test, 220 chunks)

| config | test MAE | tasks > hold | cosine | gripper | latency P50 / P95 | peak RSS | quantized Linear |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 eager (reference) | 11.24 | 7/11 | 0.569 | 5.73 | 565 / 568 ms | 3.24 GB | 0 |
| BF16 + compile | 11.55 | 7/11 | 0.575 | 5.89 | 402 / 404 ms | 8.33 GB | 0 |
| **W8A8 vision + compile** | 11.81 | 5/11 | 0.535 | 6.23 | **334 / 337 ms** | 8.60 GB | 144 |
| W8A8 vision + LM + compile | 11.92 | 5/11 | 0.559 | 6.01 | 351 / 359 ms | 8.13 GB | 244 |
| W8A8 all (incl. action head) + compile | 12.04 | 6/11 | 0.561 | 6.02 | 555 / 581 ms | 8.37 GB | 268 |

- Reports: `reports/offline_evo1_q_{bf16,w8a8_vision,w8a8_vis_lm,w8a8_all}_s8_test.json`.
- **W8A8: PASS (benchmarked).**
  - Best speed/accuracy point: **vision W8A8, language model and action head BF16** (= the spec's first W8A8
    candidate without LM quantization): −41 % latency vs BF16 eager, +0.58° MAE (+0.26° vs BF16 compiled).
  - The MAE differences are at the scale of the evaluation noise (BF16 eager vs compiled already differs by 0.32°).
  - Quantizing the LM adds latency (small token count) and error. Quantizing the action head makes it slower (int8 +
    dynamic compile on small matrices).
- **Memory:** torch.compile raises peak process RSS to about 8.1–8.6 GB (fits 12 GB, tight on this 10 GB machine).
  VRAM unchanged (3.42 GB).
- **W4A8: NOT USEFUL / BLOCKED on this stack.** torchao int4-weight paths are slower than BF16 on sm_86 (11 % layer
  error), and int4 weight-only needs `mslk`.
- **W4A4 (FoldQuantVLA): BLOCKED.** It supports only GR00T / π0.5 and native INT4 kernels target sm_87/sm_89; this GPU
  is sm_86 (`docs/quantization_plan.md`).
- **Mixed precision chosen:** vision W8A8, projector / LM / head BF16, compiled, 8 flow steps.

## Phase 17: BEST_LOCAL closed-loop MuJoCo evaluation (2026-10-10)

- **Integration:** `scripts/evaluate_policy.py` gained `--quant {bf16,w8a8_vision}`, `--compile`,
  `--num-inference-timesteps` and `--trace`. Without them it keeps the old BF16 eager behaviour.
  - Results now record `model_config`, `overrides`, `quantization` (incl. int8 tensor count per module group) and the
    per-chunk network latency.
  - Traces go to `outputs/traces/*.npz`.
- **Tests:** `pytest` 57/57 PASS. New `tests/test_quantization.py` checks that w8a8_vision quantizes only vision and
  that MHA out_proj is never quantized. Sim smoke PASS.
- **Runs:** same checkpoint, seeds 1000–1004, unseen test positions, 11 tasks × 5, 8 flow steps, compiled:
  - `reports/evaluation_evo1_bl_{bf16,w8a8_vision}_c_s8_position.{json,md}`
  - videos `reports/videos/evo1_bl_*_task_*`

| | BF16 + compile | W8A8 vision + compile |
|---|---:|---:|
| success | **0/55** | **0/55** |
| int8 tensors vision / projector / LM / head | 0/0/0/0 | 144/0/0/0 |
| network call per chunk P50 / P95 | 406 / 418 ms | 339 / 354 ms |
| chunk calls > 0.5 s (compile warm-up) | 3 / 1050 | 3 / 1033 |
| peak RSS / VRAM | 9.67 / 3.42 GB | 9.23 / 3.42 GB |
| failures | 55 timeouts | 52 timeouts, 2 pot_tipped_over, 1 wrong_order |

- **Failure analysis** (traces, videos, real-vs-sim comparison):
  - **Verified consistent:**
    - Joint order and units.
    - Gripper convention: the policy closes to about 3 in 49/55 episodes and opens to 52.
    - Start pose: sim within 2–4° of the real task_01 start mean.
    - Image preprocessing: sim gives HWC uint8; the official `prepare_observation_for_inference` divides by 255 and
      permutes to CHW, matching the training tensors.
    - Actuator tracking: |state[t+5] − action[t]| is 2–5°, so control is not the bottleneck.
  - **Concrete mismatches (evidence, not assumed domain gap):**
    1. Top camera viewpoint. Real: low, side view, robot base bottom-left, leader arm at the right edge. Sim:
       `pos [-0.10,-0.20,0.42]`, high, steep, robot centred and rotated. Side-by-side frame:
       real vs sim top / wrist at reset.
    2. Object scale: in the real wrist view the cubes look much larger than the sim 25 mm cubes. The size was only
       estimated from video and never measured.
    3. Robot colour (yellow vs cyan) and table texture.
  - **Policy behaviour in sim:**
    - Joint motion is 3–6× larger per step than in real recordings (e.g. shoulder_lift 2.40 vs 0.88°/step).
    - Actions leave the training range (shoulder_lift −118.6° < train min −105.5°; elbow 114.8° > 99.9°).
    - Action jumps of 9.5° at chunk boundaries vs 2.1°/step inside a chunk.
    - This is consistent with out-of-distribution visual input plus open-loop 50-step chunks.
  - **Chunk-boundary jumps also appear on real images:** replay gives 14.3° at boundaries vs 1.0°/step inside
    (recorded 0.5°/step). This is a policy property, not only a sim artefact.
- **STATUS:**
  - W8A8 vision speed / offline accuracy: **PASS** (339 ms/chunk; offline MAE 11.81 vs BF16 compiled 11.55).
  - MuJoCo task success: **FAIL (0/55)** for both BF16 and W8A8. W8A8 is not worse than BF16 in sim.
  - Real robot: replay dry-run **PASS**, no hardware. Hardware dry-run **NOT YET VERIFIED**. Real motion test
    **NOT DONE / NOT READY** (`reports/real_robot_readiness.md`).
- **W8A8 vision is the speed choice, not a validated final policy.**
- **Next candidates (not run):**
  - shorter execution horizon (`--n-action-steps` 10–25) to reduce open-loop chunk jumps;
  - re-rendering the sim top camera / object scale to match the real setup (needs new sim data; the existing 220
    episodes stay untouched).

## Phase 18 (user request 2026-10-10): (1) shorter execution horizon, (2) sim scene matched to the real setup

### (1) n_action_steps sweep: RUNNING
- W8A8 vision + compile, 8 flow steps, same seeds and positions as the 50-step baseline.
- `--n-action-steps 25` then `10` → `reports/evaluation_evo1_bl_w8a8_vision_c_s8_n{25,10}_position.*`.
- Resumable loop (pause_state.json → n_action_steps_sweep).

### (2) Sim scene v2: config DONE, data generation queued
- **Correction of an earlier claim:** the real cubes are **not** larger than the sim cubes.
  - Evidence: the gripper reading while an object blocks the closing jaw (commanded < measured − 4).
  - Real median 14.0–15.2 (tasks 01, 02, 03, 06) and 17.5 (07). Sim with 25 mm cubes: 15.9–16.2.
  - The larger look in the real wrist frames comes from the wrist-camera viewpoint, not from object size.
  - Object sizes therefore stay unchanged.
  - Assumption: the real and sim gripper share the 0–100 calibration mapping.
- **Real top-camera geometry**, read from frames of 4 tasks:
  - The far table edge / backdrop is horizontal while the folded arm projects vertically at the left. So the real
    robot stands at an angle to the table and the camera is behind and to the right of it.
  - The sim table / backdrop were aligned with the robot, so no camera pose alone could reproduce the view.
- **`configs/simulation_v2.yaml`** (selected with `SO101_SIM_CONFIG`; the default config and the v1 data are
  unchanged):
  - top camera pos (−0.22, −0.24, 0.52) → target (0.18, −0.36, 0), fovy 60;
  - `appearance`:
    - robot_rgba cyan (printed parts; STS3215 motors keep their colour);
    - table_yaw_deg −20 (visual only, table top at the same height);
    - white non-colliding leader-arm stand-in at the right edge.
- New code (backward compatible):
  - `src/simulation/scene.py` `build(..., appearance=...)`;
  - `src/simulation/env.py` `SO101_SIM_CONFIG`;
  - `scripts/calibrate_sim_camera.py` (side-by-side real vs candidate renders);
  - `scripts/generate_sim_data.py --report-name`;
  - `scripts/generate_sim_v2.sh` (resumable, per task).
  - pytest `tests/test_simulation.py` 21/21 PASS.
- **Match quality: qualitative only.** Real camera intrinsics are unknown and nothing was measured.
  - Wrist camera: ±10–20° pitch / FOV changes did not reproduce the real view (real sees only the table). It is left
    unchanged: NOT calibrated.
- **Plan:**
  - 40 new oracle episodes per task (seeds 200000+, disjoint from v1 and evaluation) → `datasets/sim_so101_v2/task_XX`;
  - hidden cache → selective_llm REAL_PLUS_SIM(v2) from run 2 → sim eval on the v2 scene;
  - plus the REAL_ONLY policy on the v2 scene, to isolate the effect of the scene change alone.

### n_action_steps = 25 result
- `reports/evaluation_evo1_bl_w8a8_vision_c_s8_n25_position.*`: **0/55** (55 timeouts). Chunk call P50 335 ms.
- Chunk-boundary jump 7.4° (vs 9.5° at 50). Within-chunk 2.0°/step (vs 2.1). Joint motion 1.93°/step (vs 1.91).
- A shorter horizon reduces the boundary jumps by about 20 % but does not change the overall over-motion (3–6× real).
- The jitter comes mostly from the predicted actions themselves on sim images, not from chunking. **NOT ACHIEVED.**

## PAUSED (2026-10-10 13:00)

No project process is running (GPU idle). Resume plan: `reports/pause_state.json` → `next_steps_in_order`.
- Done: n=25 sweep; `configs/simulation_v2.yaml` and the tools for it.
- Not done: n=10 sweep (stopped early; partial output deleted, must be rerun); v2 sim data generation (not started).

## RESUMED (2026-10-10T13:14:05+07:00)

- Checked: no project process, GPU idle, n25 report and v2 config present. Running: n=10 sim eval → v2 generator smoke → v2 generation.

### n_action_steps = 10 result
- `reports/evaluation_evo1_bl_w8a8_vision_c_s8_n10_position.*`: **0/55** (55 timeouts). Chunk call P50 335 ms.
  Peak RSS **10.78 GB** (swap used on the 10 GB machine).
- Chunk-boundary jump 6.6° (n50 9.5, n25 7.4). Within-chunk 2.04°/step. Joint motion 2.16°/step (n50 1.91).

| n_action_steps | success | boundary jump | within-chunk | joint motion / step |
|---:|---:|---:|---:|---:|
| 50 | 0/55 | 9.5° | 2.1° | 1.91° |
| 25 | 0/55 | 7.4° | 2.0° | 1.93° |
| 10 | 0/55 | 6.6° | 2.0° | 2.16° |

- **Direction 1 conclusion: NOT ACHIEVED.**
  - Shorter execution horizons reduce the chunk-boundary jumps (−30 % at n=10) but do not reduce the over-motion
    (3–6× the real recordings).
  - No task succeeds.
  - The jitter is in the per-chunk predictions themselves on sim images.

## PAUSED (2026-10-10 15:40)

No project process is running (GPU idle). Resume: `reports/pause_state.json` → `next_steps_in_order`.
- Direction 1: complete (n_action_steps 50 / 25 / 10: 0/55 each).
- Direction 2: `datasets/sim_so101_v2/task_01..03` complete (40 episodes each). task_04 stopped at 33 episodes, not
  finalized; the script regenerates it. task_05..11 not started.
- Next: `./scripts/generate_sim_v2.sh`, then `./scripts/run_sim_v2_pipeline.sh`.

## RESUMED (2026-10-10T21:51:03+07:00)

- Checked: no project process, GPU idle; v2 task_01..03 complete. Running: generate_sim_v2.sh (task_04 regenerated) then run_sim_v2_pipeline.sh.

### Direction 2 result: sim scene v2 data + REAL_PLUS_SIM(v2) (2026-10-11)

- **v2 data: PASS.**
  - `datasets/sim_so101_v2_all`: 440 episodes / 123,056 frames, 40 per task, train-region positions, oracle
    successes only, seeds 200000+.
  - Official aggregate verified (`reports/sim_v2_dataset_report.json`): counts match, 880 frames decoded (first + last
    of every episode, both cameras), 0 errors, exact instructions.
  - Per-task reports: `reports/sim_v2_report_task_XX.json`.
  - Generation stats: task_04 took 132 min (long episodes, about 500 frames). The machine was suspended for about 6 h
    in between; the job resumed.
- **Cache:** `simv2_s10_o0`, 49 shards, 12,497 frames, rel_err 0.0.
- **Training: PASS.**
  - Run `outputs/evo1_s2lite_selective_llm_rpsv2_20261011_054754`: selective_llm from run 2, 10k steps × 32,
    train_s10_o0 + simv2 (about 38 % sim).
  - Best val 0.02678. 3.3 h, peak RSS 4.99 GB, VRAM 4.13 GB.
  - Export: `checkpoints/evo1_so101_s2lite_selective_llm_rpsv2/`.
- **Offline, real test** (4 frames/ep):

  | model | MAE | tasks > hold |
  |---|---:|---:|
  | REAL_ONLY | 11.26 | 9/11 |
  | REAL_PLUS_SIM v1 scene | 12.04 | 5/11 |
  | REAL_PLUS_SIM v2 scene | 11.46 | 7/11 |

  - The v2 sim data costs much less real accuracy than v1.
- **MuJoCo on the v2 scene** (`SO101_SIM_CONFIG=configs/simulation_v2.yaml`): BEST_LOCAL inference (W8A8 vision,
  compile, 8 steps), seeds 1000–1004, unseen test positions.
  - Reports: `reports/evaluation_evo1_v2scene_{realonly,rpsv2}_w8vis_s8_position.*`, videos and traces.

  | policy | success | ended by object interaction | joint motion / step | peak RSS |
  |---|---:|---:|---:|---:|
  | REAL_ONLY on v2 scene | 0/55 | 0/55 (all timeouts) | 1.89° (v1 scene 1.91°) | 11.44 GB |
  | REAL_PLUS_SIM(v2) on v2 scene | 0/55 | 19/55 (pot_tipped_over 14, wrong_order 5) | **1.47°** | 10.12 GB |

  - **The scene change alone does not make the real-only policy transfer** (0/55, motion unchanged).
  - With v2 sim data the policy is task-directed again. Video task_05: it goes to the pot and centres the wrist camera
    on the blue cube inside, then tips the pot. Motion is about 23 % calmer than before.
  - No task is completed. **Task success: FAIL (0/55).**
- **Memory warning:** sim evaluation with torch.compile peaks at 10–11.4 GB RSS on a 10.9 GB machine (swap).

## Phase 19 (user request 2026-10-11): MuJoCo physics fix for pots/lids (scene v3) + re-evaluation without compile

- **Check of the existing values:**
  - Masses are already realistic: pots 0.3–0.6 kg, lid 0.15 kg, cubes 15–20 g (a solid 25 mm ABS cube is about
    16 g). They were not increased.
  - Two physical errors found:
    1. The pot builder spread the mass uniformly over bottom + 20 wall plates, so about 9 % sat in the bottom and the
       centre of mass was at about half the height. Real stainless pots have a thick base.
    2. Pot / lid friction against the table was effectively 1.0. MuJoCo combines two geoms' friction with max(), and
       the table is 1.0. Steel on wood is about 0.3–0.5.
- **Correction of my first reasoning:** for a horizontal push at the rim, static tipping (F·h > m·g·r) does not
  depend on the centre-of-mass height.
  - Whether a pot slides or tips is set by μ vs r/h ≈ 1.0. With μ = 1.0 the pots sat exactly on the tip/slide
    boundary.
  - The first v3 attempt (friction 0.4 without priority) changed nothing, because the table's 1.0 still won.
- **`configs/simulation_v3.yaml`** = v2 visuals + `physics:` (code `src/simulation/env.py::apply_physics`,
  `assets.py`, backward compatible):
  - container_bottom_mass_fraction 0.6 (same total mass);
  - container_friction 0.4 with contact priority 1, so the container's own friction applies against the table.
    Finger pads also have priority 1 → max rule → 1.5, so grasps are unaffected.
  - Cubes, sizes and solref/solimp unchanged: stiffer object contacts previously made cubes dropped into the pot
    rebound and launch it.
  - Values are literature-typical, not measured on the real objects.
- **Push test** (`scripts/check_pot_stability.py` → `reports/pot_stability_{v2,v3}.json`; horizontal force ramp at
  rim height, away from the robot):
  - v2: task_01 / 05 / 06 pots **tip** at 6.7 / 8.4 / 4.0 N; task_03 / 04 slide at 6.6 / 7.3 N.
  - v3: **all 5 slide** (tilt ≤ 0.5°) at 2.2–3.7 N.
- **Oracle check** (unseen test positions, seeds 1000–1009; `reports/oracle_v2/`, `reports/oracle_v3/`):
  - v2 47/50, **v3 49/50** (task_04 7 → 9/10; the other pot tasks 10/10 in both).
  - The physics change does not break the tasks.
- **Memory:** no `--compile` for the following sim evals.
  - W8A8 without compile is slower than BF16 (eager int8 5.5× slower per op), so the no-compile evaluation uses
    **BF16 eager, 8 flow steps** (RSS about 5 GB).
  - The sim is synchronous, so latency does not change task success.
- **Running:** `scripts/run_physics_v3_eval.sh`. REAL_PLUS_SIM(v2) policy, same seeds:
  - A = v2 scene (old physics, same inference config), the control;
  - B = v3 scene.

### Phase 19 result: re-evaluation of REAL_PLUS_SIM(v2) after the physics fix

Same checkpoint, BF16 eager (no compile), 8 flow steps, seeds 1000–1004, unseen test positions.
Reports: `reports/evaluation_evo1_rpsv2_{v2,v3}scene_bf16_s8_position.*`, videos, traces `outputs/traces/`.

| | A: v2 scene (old physics) | B: v3 scene (fixed physics) |
|---|---:|---:|
| success | 0/55 | **1/55** (task_11, seed 1000) |
| pot_tipped_over | 14 | 11 |
| wrong_order / wrong_cube_chosen | 5 / 2 | 4 / 0 |
| timeouts | 34 | 39 |
| chunk call P50 | 565 ms | 565 ms |
| peak RSS / VRAM | 5.61 / 3.42 GB | 4.29 / 3.42 GB |

- **No compile → no swap:** peak RSS 4.3–5.6 GB, vs 10–11.4 GB compiled. A vs the earlier compiled W8A8 run of the same
  policy (14 tips, 0/55) shows the inference change does not alter behaviour.
- **The task_11 success is real** (video: the cubes are taken out of the sorter and placed on the table). But
  task_11 has no pot or lid, so the physics change cannot have caused it.
  - The same seed failed in run A, and the policy's flow-matching noise is not seeded.
  - This is a single stochastic success, **not evidence of a physics effect**.
- **Pot tipping went 14 → 11, within run-to-run noise.**
  - The push test shows a horizontal push now slides the pots. The remaining tips come from contacts while the
    gripper descends into the pot (video task_06: the small cup is tipped from its rim while the gripper enters it).
  - Trace at the tip moment: in 9/11 tips the gripper tracks its command, i.e. it is not grasping; 2/11 are blocked
    grasps on the rim.
  - These are collisions a real pot would also suffer: a **policy precision problem, not an unrealistic-physics
    problem**.
  - Making the pots heavier or stickier than real objects would only hide it, so no further physics change was made.
- **STATUS:**
  - Physics fix: **PASS** (push test, oracle 49/50).
  - Task success: **1/55**. Above 0, but not attributable to the physics fix. Pot tipping not solved.

## Phase 20 (user request 2026-10-11): train-distribution test of REAL_PLUS_SIM(v2)

- **Layouts:** the first 5 successful generation seeds per task of the v2 data (`reports/seed_map_sim_v2_train.json`;
  new `--seed-map` option of `scripts/evaluate_policy.py`).
  - Reset with profile position_visual / split train reproduces the recorded layouts exactly (max |diff| 0.0 over 55),
    and the visual randomization is seeded the same way.
  - v2 scene (= data physics), BF16 eager, no compile, 8 flow steps.
- **Closed loop** (`reports/evaluation_evo1_rpsv2_trainlayouts_v2scene_bf16_s8_position_visual.*`):
  - **0/55**: pot_tipped_over 16, wrong_order 5, wrong_cube_chosen 1, timeouts 33.
  - Joint motion 1.48°/step (test positions 1.47°). Peak RSS 5.68 GB.
  - Unseen test positions for comparison: 0–1/55, 11–14 tips.
  - **→ not position overfitting.** The policy fails as badly on the exact training layouts.
- **Open loop on the same trained sim episodes** (`scripts/evaluate_offline.py --dataset-root datasets/sim_so101_v2_all
  --episodes-per-task 5` → `reports/offline_evo1_rpsv2_simv2train_test.json`; the "_test" suffix is the default split
  label, the frames are sim train frames):
  - action MAE 7.0° vs hold 17.5°, cosine 0.88, first-step error 3.9°.
  - Per joint: wrist_roll 9.6°, pan 7.6°, elbow 7.8°, lift 7.0°, wrist_flex 5.9°, gripper 4.2.
  - The policy reproduces the direction of the oracle motion but not its precision. A 4° pan error is about 1.7 cm at
    25 cm reach, more than the grasp tolerance (a 25 mm cube, about 1 cm margin), which matches the rim hits and
    missed grasps.
  - On top of this comes closed-loop error compounding.
- **Conclusion: scenario 2 (underfitting at the precision the contact phase needs). STATUS:** test PASS (measured);
  task success FAIL.
- Note: the hidden cache holds every 10th frame of the v2 sim episodes (12,497 of 123,056). More steps and / or more
  cached frames (offsets 3/5/7) are the direct levers.

## Phase 21 (user approved 2026-10-11): continued training with twice the cached v2 sim frames

- `scripts/run_rpsv2c_pipeline.sh` (resumable): cache simv2_s10_o5 → train from rpsv2 best, 20k steps, real + simv2 o0 + o5 → evals: offline real, offline sim-train, closed loop on train layouts and test positions (BF16 eager, no compile).
- Note: the init checkpoint stores LM layers 12–13 in fp32, but they are loaded into the bf16 VLM before being re-cast to fp32 for training (one bf16 rounding of those layers). Sim share of the training frames rises from about 38 % to about 55 %.
- STATUS: RUNNING.

### Phase 21 result: rpsv2c (continued training, 2× cached v2 sim frames)

- **Cache:** simv2_s10_o5, 49/49 shards, rel_err 0.0.
- **Training: PASS.**
  - Run `outputs/evo1_s2lite_selective_llm_rpsv2c_20261011_154956`: init rpsv2, 20k steps × 32, real + simv2 o0 + o5.
  - Best val (real val frames) **0.02457** (rpsv2 0.02678). 6.7 h, peak RSS 4.98 GB, VRAM 4.13 GB.
  - Export: `checkpoints/evo1_so101_s2lite_selective_llm_rpsv2c/`.

| metric | rpsv2 | **rpsv2c** |
|---|---:|---:|
| offline real test MAE (hold 11.96) | 11.46 | **10.52** (best of the project) |
| real test tasks better than hold | 7/11 | **10/11** |
| real test first-step error | 4.68° | 3.72° |
| offline on trained sim episodes: MAE (hold 17.53) | 7.01 | **3.92** |
| … first-step error / cosine | 3.91° / 0.88 | **2.22° / 0.96** |
| closed loop, exact training layouts | 0/55 (16 tips) | **2/55** (task_02, task_03; 14 tips) |
| closed loop, unseen test positions | 0/55 (14 tips) | 0/55 (15 tips) |
| joint motion / step (closed loop) | 1.48° | 1.24° |

- Reports:
  - `reports/offline_evo1_s2lite_selective_llm_rpsv2c_test.json`
  - `reports/offline_evo1_rpsv2c_simv2train_test.json`
  - `reports/evaluation_evo1_rpsv2c_{trainlayouts_v2scene_bf16_s8_position_visual,v2scene_bf16_s8_position}.*`
  - videos and traces
- **Reading:**
  - More steps + more frames clearly reduce underfitting: the open-loop sim error is about halved, and real offline
    accuracy improves too.
  - Closed loop improves only slightly:
    - first two successes on training layouts;
    - task_04 reaches "banana in pot" (2 episodes fail only at the lid stage);
    - calmer motion.
  - Pot tipping is unchanged (14–15). Unseen positions are still 0/55.
- The remaining gap between good open-loop accuracy (2.2° first step) and closed-loop failure points to error
  compounding in the contact phase, and to the open-loop 50-action execution.
- **STATUS:**
  - Underfitting reduction: **PASS** (measured).
  - Task success: **FAIL** (2/55 train layouts, 0/55 unseen).
  - rpsv2c is the best checkpoint so far, offline and in closed loop.

## Phase 22 (user request 2026-10-12)

### (1) Action-drift test: rpsv2c, n_action_steps 10, exact training layouts. RUNNING
- Output `reports/evaluation_evo1_rpsv2c_trainlayouts_v2scene_bf16_s8_n10_position_visual.*`.
- Command in pause_state.json → drift_test.

### (2) Stronger placement randomization for the next sim data (code only, no generation)
- There is no `generate_sim_oracle_data.py`. Placement is sampled in `src/simulation/tasks.py`
  (`Task.sample_layout`, per-object r / θ ranges) and `placement.py` (train cells of the checkerboard split + keep-out),
  with the yaw range from the scene-config profile. `scripts/generate_sim_data.py` only selects the profile.
- **New profile `position_wide`**, only in `configs/simulation_v3.yaml`:
  - = position_visual (same visual randomization) + `widen: {dr: 0.02 m, dth: 15°, container_r_max: 0.25 m}` +
    `yaw_range_deg: 45`.
  - Code: `tasks.py::_widened` (ranges extended and clipped to the workspace r 0.13–0.29 m, θ −60…70°).
  - The pot radius is capped at 0.25 m: the oracle needs a vertical gripper above the pot.
  - Yaw ±45° covers every distinct cube orientation (90° symmetry).
  - **Still restricted to the train cells** of the split, so the unseen-position test stays valid.
- Backward compatible:
  - without `widen` the original ranges are used; the 55 v2 training layouts still reproduce exactly (max |diff| 0.0);
  - `position_wide` added to the `--profile` choices of `generate_sim_data.py`, `evaluate_policy.py`, `run_oracle.py`;
  - pytest 57/57 PASS.
- **Coverage actually sampled** (`scripts/report_layout_coverage.py` → `reports/layout_coverage.{json,md}`; 200
  resets per task, train split).
  - **Finding:** with the old profile, the split leaves much narrower coverage than the declared ranges for the
    anchor objects:
    - pots: θ only 45…65° (declared 25…65°);
    - task_08 board: θ 45…50°;
    - task_02 / 07 / 09 board: θ 30…50°.
  - So for these tasks the unseen test positions of the anchor object lie outside the training coverage. This is
    extrapolation and supports the generalization concern for these tasks.
  - With position_wide:
    - pots θ 29/30…70°, r 0.16/0.17–0.25 m;
    - boards θ 29/30…65°;
    - cubes / bases r 0.13–0.28/0.29 m, θ −60…10/46°;
    - yaw ±45° (was ±20°).
- **Oracle feasibility on position_wide** (5 episodes / task, train split, v3 physics; no data generation):
  RUNNING → `reports/oracle_v3_wide/`.
- **Oracle on position_wide: PASS (feasible)**, 44/55 = 80 % (`reports/oracle_v3_wide/`; v3 physics, train split,
  seeds 950000+, 5 episodes per task).
  - Per task: 01 4/5, 02 5/5, 03 4/5, 04 4/5, 05 4/5, 06 **2/5** (1 pot tipped by the oracle itself), 07 4/5,
    08 5/5, 09 4/5, 10 5/5, 11 3/5.
  - v2 data generation with the old profile: about 90 % (440 / 490 attempts).
  - For a v3 generation run use `--max-attempts-factor 3` (only successes are kept; 2.0 may not reach 40 for task_06).
- Not generated yet (as requested).

### (1) Result: action-drift test (rpsv2c, exact training layouts, BF16 eager, no compile)

| n_action_steps | success | pot_tipped_over | wrong order / cube | timeouts | chunk-boundary jump | joint motion / step | peak RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 2/55 (task_02, task_03) | 14 | 5 | 34 | 3.4° | 1.24° | 5.84 GB |
| **10** | 2/55 (task_01, task_11) | 13 | 5 | 35 | 3.6° | 1.43° | 5.35 GB |

- Report: `reports/evaluation_evo1_rpsv2c_trainlayouts_v2scene_bf16_s8_n10_position_visual.*`.
- **The hypothesis is not confirmed.** Re-planning every 10 steps (every 0.33 s) does not reduce pot tipping (14 → 13)
  or raise success (2 → 2).
  - Different tasks succeed in the two runs, consistent with stochastic single successes.
  - Motion is slightly larger with n=10.
  - Since frequent re-planning from fresh observations does not prevent the tips, the failure is not open-loop drift
    between re-plans.
  - The policy's approach / contact trajectory near the pot itself is off.
- task_04: 2 episodes again reach "banana in pot" and fail only at the lid. task_09: 1 episode places the red cube but
  not the hollow cube.
- **STATUS:** test PASS (measured); drift hypothesis FAIL; task success FAIL (2/55).
- **Suggested next diagnostic** (not run): on the training layouts the oracle demonstration for the identical layout
  exists in `datasets/sim_so101_v2_all`. Comparing the policy's closed-loop joint / gripper trajectory with that demo
  would show where (approach, descent, closing) and by how much it diverges before a tip.

## Phase 23 (user request 2026-10-12): policy-vs-oracle trajectory diagnosis

- **STATUS: PASS** (measured). Report: `reports/diag_policy_vs_oracle.md`. Data: `reports/diag_policy_vs_oracle.json`.
  Script: `scripts/diag_policy_vs_oracle.py`.
- **Findings:**
  - Divergence starts in the approach in 48 / 98 diverging episodes.
  - The policy closes the gripper at the right height but 3.1 cm laterally off the target (oracle 0.06 cm).
  - All 27 pot tips reproduce on replay. The jaws hit the pot in 24–25 / 27, the wrist / camera mount in 11–12.
  - Tasks 05 / 06 tip 20 / 20 while entering the container: 1–3 cm off-axis, |Δ wrist_roll| about 16°, gripper
    vertical, open, not grasping.
  - Tasks 01 / 03 tip after a missed grasp, when the empty gripper over-descends about 5 cm into the pot.
- A first wrist-tilt hypothesis was checked and rejected (gripper tilt only 3–5°).

## Phase 24 (user request 2026-10-12): oracle v3 (recovery / hover / safe drop) + v3 sim data generation

- **Oracle changes** (`src/simulation/oracle.py`, `tasks.py`). Off by default, so v1 / v2 reproduce; enabled per
  episode via `env.oracle_opts` with the oracle's own rng (env.rng untouched).
  - **DART recovery:** the approach waypoint of every pick is `perturbable`. With probability p, the EXECUTED command
    at 30–70 % of the approach is pushed 2–4 cm sideways (random direction) and ±15° yaw for 15–25 steps.
    - The RECORDED action in that window is the expert's correction: one rate-limited step from the executed command
      toward the true approach pose.
    - Then the oracle re-plans from the measured grasp point; the recovery is recorded normally.
    - Recording the drift itself as demonstration would teach drifting, so the drift is never used as a label.
  - **Hover & align (tasks 05 / 06):** `pre_grasp_hold` steps at the closed-loop-corrected pre-grasp pose above the
    container (centring + jaw yaw from the existing `correct=True` step), then descend.
  - **Safe drop (tasks 01 / 03):** release height sampled in rim + 2–3 cm.
    - The existing oracle already released at rim + 2 cm and retreated immediately; the over-descent seen in the
      diagnosis is the policy's, not the oracle's.
- `scripts/generate_sim_data.py` options: `--dart-prob`, `--dart-dxy`, `--dart-dyaw`, `--dart-steps`,
  `--hover-steps`, `--drop-height`, `--episodes-override`. Per-attempt perturbation logs go into the report.
- **Tests:** `tests/test_oracle_dart.py`:
  - labels ≠ executed only during the perturbation, within one rate-limited step;
  - the task is still solved;
  - with no options, labels equal the executed commands.
  - pytest 59/59 PASS.
- **Feasibility** (position_wide + DART p 0.5 + hover 15 + drop 2–3 cm, v3 scene, 6 / task, no data written):
  - oracle **53/66 (80 %)** (position_wide alone 44/55 = 80 %);
  - perturbed episodes solved 32/43 (74 %);
  - task_11 2/6 (3 picks, almost always perturbed).
  - Visual check: perturbed task_05 episode (3.9 cm, −6°) recovers and succeeds.
- **Note:** with position_wide, pots at θ up to 70° can be partly outside the top-camera frame (left edge); the wrist
  camera still sees them.
- **Generation RUNNING:** `scripts/generate_sim_v3.sh` (resumable per task) → `datasets/sim_so101_v3/task_XX`.
  - 1000 episodes: tasks 05 / 06 122 each, others 84 each; `--max-attempts-factor 3`; seeds 300000+.
  - Estimate about 18–20 h CPU.
- **Disk:** 17 GB free. Generation (about 5 GB) fits, but the v3 hidden cache (about 28 GB) will not. Old artifacts
  must be removed first (user approval needed).

### Disk cleanup (user approved 2026-10-12)

- Deleted:
  - `datasets/evo1_feature_cache` (39 GB). It was the stage-1 frozen-VLM token cache, used only by `cache_evo1_features.py` / `train_evo1_cached.py` and not by the current selective_llm pipeline. Rebuilding it would take about 11 h.
  - `outputs/smoke_s2lite_selective_20261009_033322` (4.6 GB, 2-step smoke run).
- User decisions:
  - task_11 may be topped up with `--max-attempts-factor 5+` until 84 episodes.
  - The partial top-camera cut-off of pots at θ ≈ 70° is kept on purpose (it pushes the model toward the wrist camera).

## Phase 25 (user approved 2026-10-12): v3 round with a stop criterion

- `scripts/run_v3_pipeline.sh` (resumable) runs automatically after generation + top-up:
  - aggregate (+ decode check) → seed map of the v3 training layouts (verified) → hidden cache simv3 → train rpsv3 (init rpsv2c, 20k steps, real + simv3);
  - evals (BF16 eager, no compile, v3 scene): offline real, offline v3 sim-train, closed loop on the v3 training layouts and on unseen test positions, for both rpsv3 and the rpsv2c baseline.
- **Stop criterion:** on the v3 training layouts, success ≥ 10/55 AND clearly fewer pot tips than rpsv2c. Otherwise: stop adding sim data, record the limitation, move to the real-robot dry-run with the best checkpoint.
- Helpers: `scripts/aggregate_sim_v2.py` now takes `--src/--out/--report/--sim-config` (includes top-up parts); `scripts/make_seed_map.py`.

## PAUSED (2026-10-12 14:40)

No project process is running (GPU idle). Resume: `reports/pause_state.json` → `next_steps_in_order`.
- v3 data: task_01–08 complete (748 episodes). task_09 stopped at 24 (not finalized; regenerated on resume). task_10, task_11 not started.
- Top-up and the v3 pipeline (cache → train → eval) had not started.

## RESUMED (2026-10-12T21:40:33+07:00)

- Checked: no project process, GPU idle, v3 task_01–08 complete. Running: generate_sim_v3.sh (task_09 regenerated, 10, 11) → topup_sim_v3.sh → run_v3_pipeline.sh.

## PAUSED (2026-10-13 12:16)

No project process is running (GPU idle). Resume: `./scripts/run_v3_pipeline.sh` (pause_state.json → next_steps_in_order).
- v3 data COMPLETE: 1000 episodes / 332,710 frames, aggregate PASS (2000 frames decoded, 0 errors), no top-up needed (task_11 84/116 attempts).
- Seed map of the v3 training layouts PASS (layout diff 0.0). Hidden cache simv3_s10_o0 COMPLETE (132 shards, 33,722 frames, rel_err 0.0).
- rpsv3 training: latest complete checkpoint at step 14000 / 20000 (best val 0.02554). Stopped at about 14300; the steps after 14000 will be redone.
- Evaluations not started.

## RESUMED (2026-10-13T21:06:58+07:00)

- Checked: no project process, GPU idle, rpsv3 latest checkpoint at step 14000. Running: run_v3_pipeline.sh (resume training → evals).

## PAUSED (2026-10-13 22:28)

No project process is running (GPU idle). rpsv3 training: latest complete checkpoint at step 18000 / 20000 (best val 0.02420), stopped right after it was saved. Evaluations not started. Resume: `./scripts/run_v3_pipeline.sh`.

## RESUMED (2026-10-13T23:29:41+07:00)

- Checked: no project process, GPU idle, rpsv3 latest checkpoint at step 18000. Running: run_v3_pipeline.sh (finish training → evals).

### Phase 25 result: v3 round (rpsv3)

- **Training: PASS.**
  - Run `outputs/evo1_s2lite_selective_llm_rpsv3_20261013_073750`: init rpsv2c, 20k steps, real + simv3 (33,722
    frames).
  - Best val 0.02405 (rpsv2c 0.02457). Peak RSS 5.23 GB, VRAM 4.14 GB.
  - Export: `checkpoints/evo1_so101_s2lite_selective_llm_rpsv3/`.
- **Offline:**
  - real test MAE 10.44 (rpsv2c 10.52, within evaluation noise); 9/11 tasks better than hold; first-step error
    3.33° (rpsv2c 3.72°).
  - On trained v3 sim episodes: MAE 6.15 vs hold 17.21, first step 1.97°. These episodes include DART perturbation
    segments.
- **Closed loop, v3 scene** (fixed physics), BF16 eager, no compile, 8 flow steps:

| policy / layouts | success | pot_tipped_over | wrong order / object | timeouts | joint motion / step |
|---|---:|---:|---:|---:|---:|
| rpsv2c, exact v3 training layouts | 2/55 (task_02, task_03) | 9 | 3 | 41 | 1.25° |
| **rpsv3, exact v3 training layouts** | **1/55** (task_06) | 6 | 6 | 42 | 1.17° |
| rpsv3, unseen test positions | 2/55 (task_02, task_03) | 8 | 5 | 40 | 1.14° |
| rpsv2c, unseen test positions | 1/55 (task_08) | 10 | 5 | 39 | 1.23° |

- **Stop criterion (agreed 2026-10-12): NOT MET.** It needed success ≥ 10/55 on the v3 training layouts AND clearly
  fewer pot tips than rpsv2c; rpsv3 reached 1/55 with tips 6 vs 9.
- The recovery / hover / wider-position data did not translate into task success.
- **Decision as agreed:**
  - stop adding sim data;
  - record the limitation;
  - move to the real-robot dry-run with the best checkpoint.
- rpsv3 and rpsv2c are equivalent within noise offline. rpsv3 has a slightly lower first-step error and calmer motion.

- v3 round **COMPLETE** (`scripts/run_v3_pipeline.sh` PIPELINE_DONE).
  - The rpsv2c baseline on unseen test positions: 1/55, 10 tips.
  - rpsv3 vs rpsv2c on the same conditions: train layouts 1 vs 2 / 55 (tips 6 vs 9); test positions 2 vs 1 / 55
    (tips 8 vs 10).
  - The differences are within single-episode noise. Stop criterion: NOT MET.
