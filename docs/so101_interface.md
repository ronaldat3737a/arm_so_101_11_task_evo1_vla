# SO-101 interface (source of truth: LeRobot 0.6.1)

Everything below was read from the `lerobot==0.6.1` wheel (PyPI, uploaded 2026-08-03) and from the
datasets themselves. Nothing here comes from an unofficial robot model. File references are paths inside that wheel.

## Robot class

| item | value | source |
|------|-------|--------|
| config types | `so101_follower`, `so100_follower` (same class) | `robots/so_follower/config_so_follower.py` |
| class | `SOFollower` (aliases `SO101Follower`, `SO100Follower`), `name = "so_follower"` | `robots/so_follower/so_follower.py` |
| `robot_type` written into datasets | `so_follower` | matches all 11 datasets (`meta/info.json`) |
| bus | `FeetechMotorsBus`, 6 × Feetech `sts3215` (4096 ticks / turn) | `so_follower.py`, `motors/feetech/tables.py` |
| gearing (follower) | 6 × STS3215 1/345 | `docs/source/so101.mdx` |

## Joints, IDs, units

| motor id | name | feature key | norm mode (use_degrees=True, default) | unit |
|---------:|------|-------------|----------------------------------------|------|
| 1 | shoulder_pan | `shoulder_pan.pos` | `DEGREES` | degrees, 0 = middle of calibrated range |
| 2 | shoulder_lift | `shoulder_lift.pos` | `DEGREES` | degrees |
| 3 | elbow_flex | `elbow_flex.pos` | `DEGREES` | degrees |
| 4 | wrist_flex | `wrist_flex.pos` | `DEGREES` | degrees |
| 5 | wrist_roll | `wrist_roll.pos` | `DEGREES` | degrees |
| 6 | gripper | `gripper.pos` | `RANGE_0_100` | 0–100 % of calibrated range |

- `use_degrees: bool = True` is the default ("for backward compatibility with previous policies/dataset").
  The datasets confirm degrees: e.g. `shoulder_lift.pos` rests at −102.6, the gripper stays within 0–100.
- DEGREES conversion (`motors/motors_bus.py::_normalize`): `deg = (raw − mid) · 360 / 4095`, with
  `mid = (range_min + range_max) / 2` from the calibration file.
- **The DEGREES path does not clamp** in `_unnormalize` (only `RANGE_*` modes clamp). Limits are enforced by
  the motor firmware: `write_calibration` writes `Min_Position_Limit` / `Max_Position_Limit` = calibrated
  `range_min` / `range_max`. The deployment layer (Phase 12) will *also* clip in software.

## Observation / action convention

- `observation_features` = `{<motor>.pos: float for 6 motors} ∪ {<camera_name>: (H, W, 3)}`.
- `action_features` = `{<motor>.pos: float for 6 motors}`. An action is an **absolute goal position** per joint
  (`send_action` → `sync_write("Goal_Position", …)`).
- In datasets the 6 values are packed in this order into `observation.state` (float32[6]) and `action` (float32[6]):
  `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`.
- Evidence from the data (reports/dataset_audit.md, "Action convention evidence"): `|action[t] − state[t+k]|`
  is minimal at k = 4 frames (133 ms at 30 fps) for all 11 datasets. The action is the leader-arm position
  (absolute target) and the follower reaches it ~4 frames later. Actions are **not** deltas.
- Robot-side safety primitive: `max_relative_target` (float or per-motor dict). When set, `send_action` reads the
  present position and caps `|goal − present|` (`robots/utils.py::ensure_safe_goal_position`).
- Gripper protection written at connect (`configure`): `Max_Torque_Limit=500` (50 %), `Protection_Current=250`,
  `Overload_Torque=25`. PID: P=16, I=0, D=32 (configurable).
- `disable_torque_on_disconnect: True` by default.

## Motor limits

There are **no hard-coded joint limits in degrees** in LeRobot. Limits are per-robot and come from calibration:

- `calibrate()` records `range_min` / `range_max` per motor while the user moves each joint through its full
  range. `wrist_roll` is treated as full-turn: `range_min = 0`, `range_max = 4095`.
- In degrees: `limit_deg = (range_{min,max} − mid) · 360 / 4095`.
- Observed ranges in the training data (union across the 11 datasets, reports/dataset_audit.md):
  shoulder_pan ≈ [−71, 81], shoulder_lift ≈ [−103, 80], elbow_flex ≈ [−96, 104],
  wrist_flex ≈ [−89, 100], wrist_roll ≈ [−127, 100], gripper ≈ [0.4, 89]. These are the
  operating envelope of the demonstrations, not hardware limits. `deploy_so101.py` (Phase 12, not yet implemented) will use
  intersect(calibration limits, data envelope + margin) as the software clip box.

## Calibration mechanism

- Command: `lerobot-calibrate --robot.type=so101_follower --robot.port=<port> --robot.id=<id>`.
- File: `$HF_LEROBOT_CALIBRATION/robots/so_follower/<id>.json`; default
  `HF_LEROBOT_CALIBRATION = $HF_LEROBOT_HOME/calibration` (`utils/constants.py`, `robots/robot.py`).
  Note the directory uses the class `name` (`so_follower`), not the config type.
- Stored per motor: `id, drive_mode (0), homing_offset, range_min, range_max` (`MotorCalibration`).
- Procedure: move the arm to the middle of its range → `set_half_turn_homings()`, then sweep all joints except
  `wrist_roll` → `record_ranges_of_motion()`.
- `connect(calibrate=True)`: if the motors' stored calibration does not match the file, it prompts.
  The deploy script (Phase 12) will run `connect(calibrate=False)` and refuse to continue if `is_calibrated` is False.
- Motor ID / baudrate setup: `lerobot-setup-motors --robot.type=so101_follower --robot.port=<port>`.
  Port discovery: `lerobot-find-port`. Cameras: `lerobot-find-cameras opencv`.

**Implication for this project:** the dataset was recorded with the dataset author's calibration. All 11 datasets
share the same rest-pose readings (shoulder_lift −102.59, elbow ≈ 97), which is consistent with one calibration.
Your physical robot has its own calibration. The policy outputs degrees relative to *your* calibrated mid-range,
so the real arm must be calibrated with the official procedure, carefully, so that the same physical pose gives
approximately the same degrees. Verify this before any `--real` run.

## Camera convention

| dataset key | resolution (H×W) | codec | fps | role (from name) |
|-------------|------------------|-------|-----|------------------|
| `observation.images.top` | 360 × 640 | AV1 yuv420p | 30 | fixed overhead / scene camera |
| `observation.images.wrist` | 480 × 640 | AV1 yuv420p | 30 | gripper-mounted camera |

- In LeRobot the dataset key is `observation.images.<camera_name>`, where `<camera_name>` is the key in
  `--robot.cameras={top: {...}, wrist: {...}}`. A real deployment **must name the cameras `top` and `wrist`** and
  match the resolutions above (configs/camera.yaml).
- Camera configs: `lerobot.cameras.opencv.OpenCVCameraConfig(index_or_path, width, height, fps, ...)`.
- Images are delivered HWC uint8 RGB from the robot. The policy processors convert them to CHW float.

## Evo-1 side (lerobot.policies.evo1)

- Pads state/action to `max_state_dim = max_action_dim = 24`; postprocessor crops back to the dataset dim (6).
- `gripper_index` default is 6 (LIBERO layout). It is only used when `binarize_gripper=true`, and we keep that **false**
  (SO-101 gripper is continuous 0–100).
- Normalization: `STATE`/`ACTION` = `MIN_MAX` using dataset `meta/stats.json`. The stats are saved inside the checkpoint
  (`policy_preprocessor_step_*_normalizer_processor.safetensors`). The `arm_key` / `dataset_key` of the standalone
  Evo-1 server do **not** exist in the LeRobot integration.
- Images: resized inside the embedder to `image_resolution = (448, 448)` (square only). Up to `max_views = 3`.

## LeIsaac (simulation) convention vs. our datasets

Read from LeIsaac `main` @ `24d3bcd` (2026-04-15), `source/leisaac/leisaac/assets/robots/lerobot.py`,
`utils/robot_utils.py`, `tasks/template/single_arm_env_cfg.py`.

| item | LeIsaac | our datasets | consequence |
|------|---------|--------------|-------------|
| robot asset | `assets/robots/so101_follower.usd` (downloaded separately), joints `shoulder_pan … gripper` | same joint names | reuse asset |
| sim joint unit | radians (USD), limits in degrees: pan ±110, lift ±100, elbow −100..90, wrist_flex ±95, roll ±160, gripper −10..100 | — | — |
| LeRobot ↔ sim mapping | `convert_lerobot_action_to_leisaac`: **linear map of motor range (−100..100, gripper 0..100) onto USD limits**, i.e. assumes `use_degrees=False` (`RANGE_M100_100`) | body joints in **degrees** (`use_degrees=True`) | **LeIsaac's converter must NOT be used for our policy**. Body joints: sim_rad = deg2rad(dataset_deg) (+ per-joint offset if verified). Gripper (0..100) keeps the linear map |
| rest pose (sim degrees) | lift −100, elbow 90, wrist_flex 50, pan 0, roll 0 (±30 tolerance) | lift −102.6, elbow ≈ 97, wrist_flex ≈ 42–61, pan ≈ −3…−12 | sim joint zero ≈ LeRobot-degrees zero; residual offsets of a few degrees are expected and must be measured (sim2real checklist) |
| cameras | `front` 640×480 (78° FOV, on base), `wrist` 640×480 (75° FOV, on gripper) | `top` **640×360**, `wrist` 640×480 | rename `front`→`top`, change the top camera to 640×360 and place it to match the real overhead view |
| actuator | implicit PD, stiffness 17.8, damping 0.6, effort 10 | real STS3215 P=16 D=32 | approximate dynamics, documented as approximation |
| LeRobot dependency | `leisaac[lerobot]` pins `lerobot==0.4.2`. Policy inference over gRPC (`lerobot-async` extra) | we need `lerobot==0.6.1` (Evo-1) | sim env and train env are **separate**. The policy must run in the 0.6.1 env and be served to the sim over a process boundary |
| Python / Isaac | Python 3.11, Isaac Sim 5.1.0, IsaacLab 2.3.0, torch 2.7.0 cu128 | Python 3.12 (lerobot 0.6.1) | two environments |
