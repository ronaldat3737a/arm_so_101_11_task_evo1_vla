# Real SO-101 readiness: BEST_LOCAL candidate (selective_llm, W8A8 vision, compile, 8 flow steps)

Status: **NOT READY for a real motion test.** Only the replay dry-run, with no hardware and no command sent, has
been done.

## Done (verified)

| item | result | evidence |
|---|---|---|
| Replay dry-run (held-out real episode task_01, no hardware) | PASS, 792 steps, finite actions | `reports/deploy_dry-run_task_01_20261010_105115.json` |
| Joint order / units | identical: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll (deg), gripper (0–100) | dataset `meta/info.json`, `docs/so101_interface.md` |
| Quantization actually applied | int8 tensors: vision 144, projector / LM / head 0 | dry-run summary `quantization` |
| Warm-up before the control loop | 26.9 s, 17.0 s, 0.34 s (compile + one recompile); nothing sent | dry-run summary `warmup_ms` |
| Inference within the watchdog (0.5 s) after warm-up | chunk call 339–340 ms, one per 50 steps (1.67 s at 30 Hz) | dry-run summary |
| Safety layer active | joint box = data envelope ± 5° (no calibration file in replay), rate limit 0.5°/step at speed 0.1: 792/792 steps rate-limited; workspace check: 24 rejects | dry-run summary |

## Open issues before any motion

1. **Hardware dry-run** (`--dry-run --source robot`, torque off, read-only) has not been run: no calibration file,
   no camera check. It is the only thing that can write `reports/dryrun_pass.json`, which `--real` requires.
2. **Camera mapping:** confirm the physical top / wrist cameras match `configs/camera.yaml` (resolution 640×360 /
   640×480, same viewpoint as the training videos).
3. **Chunk-boundary jumps:**
   - 50 actions are executed open-loop. On real images (replay), the policy action jumps on average 14.3° at chunk
     boundaries vs 1.0°/step inside a chunk. Recorded human actions: 0.5°/step.
   - In closed-loop sim: 9.5° vs 2.1°.
   - The rate limiter absorbs this, but at speed_scale 0.1 the command lags the policy by up to 68°. The robot will
     not follow the policy's intent.
4. **Blocking inference:** the control loop pauses about 340 ms at every chunk call (synchronous inference).
5. **RAM:** torch.compile raises process RSS to about 9.2–9.7 GB (sim eval) on a 10 GB machine.
6. **Policy quality:** MuJoCo task success is 0/55. Offline test MAE is 11.8° vs a "stay still" baseline of 12.0°.
   No evidence yet that the policy completes a task.

## Required before the first supervised real test (not done)

- [ ] `lerobot-calibrate` for the follower; calibration file present.
- [ ] `--dry-run --source robot` PASS with this checkpoint, quantization and calibration (writes the pass file).
- [ ] Physical e-stop verified: power switch within reach; `touch /tmp/so101_STOP` and Ctrl+C tested while the
      motors are powered.
- [ ] Workspace cleared; only the task_01 objects present; a person supervising at the robot.
- [ ] Speed scale ≤ 0.1, max_relative_target 8°, episode limit 40 s (configs/deployment.yaml).
- [ ] Written approval from the user for one supervised task_01 trial.
