# Real SO-101 setup and deployment

Use only after the simulation evaluation passes. All commands are official LeRobot 0.6.1 CLIs plus `scripts/deploy_so101.py`.

## 1. Hardware and ports

```bash
uv run lerobot-find-port            # unplug/replug the follower USB to identify e.g. /dev/ttyACM0
uv run lerobot-find-cameras opencv  # lists cameras; use stable /dev/v4l/by-id/... paths in configs/camera.yaml
```

Edit `configs/deployment.yaml` (`robot.port`, `robot.id`) and `configs/camera.yaml`.
Cameras must be named `top` and `wrist`, with top 640×360 and wrist 640×480 at 30 fps, matching the training data.
Mount the top camera at the overhead/front viewpoint of the training videos (`reports/frames/*.jpg`). Policies are
sensitive to viewpoint.

## 2. Motor IDs (only for a new or reassembled arm)

```bash
uv run lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACM0
```

Expected IDs: shoulder_pan=1, shoulder_lift=2, elbow_flex=3, wrist_flex=4, wrist_roll=5, gripper=6 (`docs/so101_interface.md`).

## 3. Calibration (official procedure; no custom calibration)

```bash
uv run lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_so101_follower
```

- Move the arm to the middle of its range when asked, then sweep every joint except wrist_roll through its full range.
- File: `$HF_LEROBOT_CALIBRATION/robots/so_follower/my_so101_follower.json` (default `~/.cache/huggingface/lerobot/calibration`).
- The policy outputs degrees relative to the calibrated mid-range (`use_degrees=True`). Calibrate carefully. The training
  data reads shoulder_lift ≈ −102.6° and elbow_flex ≈ 97° in the folded rest pose (`reports/dataset_audit.md`, start pose).

## 4. Pre-deployment checks

Check | how | expected
---|---|---
calibration present & matches motors | `deploy_so101.py --dry-run --source robot` refuses to run otherwise | runs
motor IDs | step 2 output | 1..6 as above
joint direction | dry-run with torque off: move each joint by hand, watch the printed state | positive shoulder_pan turns the arm toward the robot's right (−y); values change smoothly
joint limits | dry-run log `joint_box` | calibration ∩ train-action envelope ± 5°
gripper | close by hand → ≈0, open → large | 0 = closed, 100 = open
rest pose | fold the arm | ≈ [−3, −102, 97, 50, 0, x] like the training data

## 5. Dry-run (mandatory)

```bash
# no hardware at all: replay a held-out real test episode through the model and safety layer
uv run python scripts/deploy_so101.py --checkpoint checkpoints/evo1_so101/best --task task_01 --dry-run --source replay
# real cameras + robot state, torque disabled, NOTHING is sent; a pass writes reports/dryrun_pass.json
uv run python scripts/deploy_so101.py --checkpoint checkpoints/evo1_so101/best --task task_01 --dry-run --source robot
```

The dry-run prints the predicted joint action, the gripper action, the command after the safety layer and the inference
latency. `reports/deploy_dry-run_*.json` holds the full log, the action range and the intervention counts.

## 6. Real run (low speed)

```bash
uv run python scripts/deploy_so101.py --checkpoint checkpoints/evo1_so101/best --task task_01 --real --speed-scale 0.1
```

- `--real` is refused unless a `--source robot` dry-run passed for the same checkpoint and calibration within 24 h.
- At start the arm moves slowly (reset speed 0.05) to the rest pose.
- Safety layers, in order:
  1. joint clipping (calibration ∩ data envelope);
  2. per-step rate limit `max_joint_delta × speed_scale`;
  3. workspace box on the FK gripper point;
  4. LeRobot `max_relative_target`;
  5. motor firmware position limits.
- Emergency stop: Ctrl+C, space or esc (with a display), or `touch /tmp/so101_STOP`. On stop the arm holds its current
  position, then torque is disabled on disconnect.
- Inference timeout 0.5 s and communication timeout 0.2 s stop the run.
- Increase `--speed-scale` only gradually, after successful runs.
