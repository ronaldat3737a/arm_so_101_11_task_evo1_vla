# Simulation asset report (MuJoCo)

The simulation is **not** pixel-identical to the real scene. It uses the official SO-101 kinematic model and simple
primitive objects. Everything below is an approximation unless marked "official".

## Robot (official model, small documented changes)

| item | source / value | status |
|---|---|---|
| kinematics, joint limits, visual meshes, link inertias | TheRobotStudio/SO-ARM100 `Simulation/SO101/so101_new_calib_camera.xml` @ `a758567` | official, unchanged |
| joint zero | "new calibration" = middle of range = LeRobot `use_degrees=True` zero | official |
| actuators | official position actuators (kp 998.22, kv 2.731, forcerange ±3.35 N·m) | official |
| gripper force | forcerange halved to ±1.675 N·m: LeRobot `SOFollower.configure()` writes `Max_Torque_Limit=500` (50 %) to the gripper "to avoid burnout". At full torque, releasing a squeezed thin knob flung the lid | **modified** to mirror the real config |
| wrist camera body/mount | official `_camera` variant. The MuJoCo camera is added on body `wrist_camera`, offset 25 mm along the lens axis (outside its own housing), 75° fovy | camera element added by us |
| **finger collision** | The official fingertips collide via the convex hulls of `wrist_roll_follower_so101_v1` / `moving_jaw_so101_v1`. Their inner faces taper toward the palm and wedged a 25 mm cube out of the grasp. These two meshes no longer collide. Each finger gets a 35 × 13 × 4 mm (fixed) / 31 × 13 × 4 mm (moving) box pad placed on the inner face measured from the mesh vertices. Pads: friction 1.5, condim 6, stiff contact (solref 0.002), priority 1. Visuals unchanged | **modified** (`src/simulation/scene.py::FINGER_PADS`) |
| gripper mapping | canonical 0..100 → hinge −10°..100° linearly (the upstream README says the LeRobot linear gripper mapping "is not yet reflected" in the model) | approximation |
| base placement | base at z = +2 mm so the base meshes rest on the table | measured |

Measured consequences that also apply to the real arm:
- `wrist_flex` is limited to ±95°, so the gripper cannot stay vertical above ~10 cm.
- The shoulder actuator saturates at 3.35 N·m and sags ~2° on an extended arm.

Some dataset values exceed the MJCF limits by a few degrees: shoulder_lift reaches −105.5° (limit −100°), elbow_flex
104° (limit 96.8°). These are rest poses at the hard stops. The real calibration differs slightly from the CAD model
(sim2real checklist item); the simulator clamps them.

## Scene

| item | approximation |
|---|---|
| table | box, wood colour, top at z = 0, friction 1.0 |
| backdrop | green box behind the workspace (video: green backdrop), no collision |
| top camera | fixed, 640 × 360, fovy 68°, behind/right of the arm looking forward-down. It is placed to match the real view qualitatively: arm at bottom-left, workspace toward the backdrop. **Not calibrated** against the real camera (no intrinsics/extrinsics available) |
| wrist camera | 640 × 480 on the official camera mount, jaws visible at the bottom as in the real wrist view |
| lighting | one directional + one fill light. `position_visual` profile scales intensity ±25 % |
| right (white) arm visible in the real top view | not modelled |

## Objects (all primitives; builders in `src/simulation/assets.py`)

Dimensions are estimated from the video frames (`reports/frames/top_rows_*.jpg`) relative to the 25 mm cubes; nothing was measured.

| object | tasks | primitive approximation | size (m) | mass |
|---|---|---|---|---|
| cube | 01–03, 05–08 | box | 0.025 (yellow distractor 0.020–0.022) | 20 g |
| steel pot / bowl | 01, 03 | flat bottom (8 mm) + ring of 20 wall boxes (6 mm) + 2 small handles | Ø 0.13, h 0.06–0.065 | 0.5 kg |
| saucepan with long handle | 04, 05 | same pot, Ø 0.16, h 0.07 + black box handle 0.10 long | Ø 0.16 | 0.6 kg |
| small steel cup | 06 | same pot builder | Ø 0.096, h 0.055 | 0.3 kg |
| lid | 04 | disk 6 mm + knob cylinder Ø 18 mm × 30 mm (knob taller than the 2 cm the fingertips reach below the grasp point) | r 0.083 | 0.15 kg |
| banana | 04, 05 | 3 capsules on an arc, r 0.016 | tip-to-tip 0.133 | 0.12 kg |
| yellow triangular prism | 01 | box | 0.035 × 0.035 × 0.02 | 20 g |
| light wooden block / board | 02, 07, 09 | box | 0.12–0.15 × 0.10–0.12 × 0.03 | 0.3 kg |
| square board with corner cubes | 08 | box 0.15 × 0.15 × 0.02; cubes at (±5, ±5) cm, 3 of 4 corners filled; minority colour randomized per episode | — | 0.3 kg |
| orange hollow cube | 09 | open-top shell: bottom + 4 walls (4 mm) | 0.042 outer | 30 g |
| orange blocks | 10 | box | 0.03 × 0.03 × 0.045 | 30 g |
| shape-sorter box | 11 | solid platform 0.15 × 0.08 × 0.02 + 6 mm collar around each of 3 square holes (2.8 cm). The real box is closed with holes in its top and the pieces stick out above it; the collar keeps the pieces upright | — | 0.6 kg |
| blue pieces of different heights | 11 | boxes 0.022 × 0.022 × {0.04, 0.055, 0.07} | — | 15 g |

Semantics kept from the video but simplified:
- task_02 / task_07 / task_09: the "wooden board" is a block; the reference object is placed on it with a small random offset.
- task_04: the lid lies on the table; the pot handle points away from the robot (±20°); stage order (banana before lid) is enforced.
- task_04: stage-2 object (lid) is placed in any split cell; banana and pot carry the spatial split (see configs/simulation.yaml).
- task_08: the reasoning element (which colour is the minority) is randomized per episode; a wrong colour on the board is `wrong_cube_chosen`.
- task_11: removal order is tracked; a wrong first removal is `wrong_order`.
- The right (white) arm, cables and other clutter visible in the real frames are not modelled.

Contact parameters: objects solref 0.01 (soft, MuJoCo default solimp), condim 4, friction 1.0. Thin (3 mm) pot walls
with stiff contacts let a dropped cube tunnel and rebound, so walls and bottom have minimum thicknesses.

All primitives are created by named builders. A real mesh can replace a builder without touching task logic.
Tasks only use body names, poses and the semantic dimensions returned by the builder (inner radius, height, ...).

## Not available

No CAD meshes or measured dimensions of the real objects were provided. Sizes are estimated from the video frames
(reports/frames/*.jpg) relative to the 25 mm cubes and the robot.
