# Dataset audit

Generated: 2026-10-07T04:36:45.553729+00:00  
Collection: `hungho77/so101-leftarmonly-task-6ac4b58bcea46f327127da54` (11 items)  
Video probe: yes

## Verdict: **PASS_WITH_EXCLUDED_EPISODES**

- errors: 0, warnings: 5, excluded no-op episodes: 9
- dataset integrity = PASS
- dataset usable for training = PASS
- 9 no-op episodes (no motion, not corrupted) excluded from training: {'hungho77/so101_left_arm_blue_cube_next_to_blue_cube': [21], 'hungho77/so101_left_arm_pick_blue_cube_into_pot': [25], 'hungho77/so101_left_arm_pick_banana_close_lid': [3], 'hungho77/so101_left_arm_blue_cube_out_of_pot_onto_blue_cube': [12, 28, 47], 'hungho77/so101_left_arm_stack_yellow_on_orange_cube': [2], 'hungho77/so101_left_arm_red_cube_into_orange_hollow_cube': [22], 'hungho77/so101_left_arm_blue_cubes_shortest_to_tallest': [26]}

## Per-dataset summary

| # | repo | revision | version | episodes | static (unusable) | frames | fps | ep len min/median/max | task string | errors | warnings |
|---|------|----------|---------|---------:|-------------------|-------:|----:|----------------------|-------------|------:|--------:|
| 1 | `so101_left_arm_pick_red_cube_into_pot` | `62d3cb12` | v3.0 | 50 | - | 37536 | 30 | 628/741/887 | Pick up the red cube and place it into the steel pot. | 0 | 0 |
| 2 | `so101_left_arm_blue_cube_next_to_blue_cube` | `1d32c7fa` | v3.0 | 50 | [21] | 14841 | 30 | 10/296/429 | Pick up the blue cube and place it next to the blue cube. | 0 | 1 |
| 3 | `so101_left_arm_pick_blue_cube_into_pot` | `98192ba8` | v3.0 | 50 | [25] | 29688 | 30 | 3/597/779 | Pick up the blue cube and place it into the steel pot | 0 | 1 |
| 4 | `so101_left_arm_pick_banana_close_lid` | `63a2c901` | v3.0 | 50 | [3] | 22278 | 30 | 17/446/653 | Take the banana and place it into the steel pot then close the lid. | 0 | 1 |
| 5 | `so101_left_arm_blue_cube_out_of_pot_onto_blue_cube` | `ace616ca` | v3.0 | 51 | [12, 28, 47] | 23103 | 30 | 271/415/897 | Take the blue cube out of the steel pot and place it on the blue cube. | 0 | 0 |
| 6 | `so101_left_arm_red_cube_out_of_pot_onto_red_cube` | `9363738d` | v3.0 | 50 | - | 21837 | 30 | 244/418/777 | Take the red cube out of the steel pot and place it on the red cube | 0 | 0 |
| 7 | `so101_left_arm_stack_yellow_on_orange_cube` | `ef6cb56c` | v3.0 | 50 | [2] | 23691 | 30 | 184/464/759 | Stack the yellow cube on top of the orange cube. | 0 | 0 |
| 8 | `so101_left_arm_balance_red_blue_on_wooden_board` | `460f5d1a` | v3.0 | 50 | - | 14217 | 30 | 208/282/412 | Pick up the cube that makes the number of red and blue cubes on the wooden board equal, and place it in the empty corner of the wooden board. | 0 | 0 |
| 9 | `so101_left_arm_red_cube_into_orange_hollow_cube` | `cf2c68df` | v3.0 | 50 | [22] | 20642 | 30 | 9/384/897 | Pick up the red cube from the wooden board, put it into the orange hollow cube, then pick up the orange hollow cube and place it on the wooden board | 0 | 1 |
| 10 | `so101_left_arm_stack_orange_blocks` | `b932726e` | v3.0 | 50 | - | 15682 | 30 | 188/299/668 | Pick up one orange block and stack it neatly on top of the other orange block | 0 | 0 |
| 11 | `so101_left_arm_blue_cubes_shortest_to_tallest` | `629eba59` | v3.0 | 50 | [26] | 30230 | 30 | 14/603/897 | Take out the blue cubes from shortest to tallest and place it on the table | 0 | 1 |

## Cross-dataset conventions

| check | consistent | value(s) |
|-------|:----------:|----------|
| codebase_version | yes | `"v3.0"` |
| robot_type | yes | `"so_follower"` |
| fps | yes | `30` |
| action_names | yes | `["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]` |
| action_shape | yes | `[6]` |
| action_dtype | yes | `"float32"` |
| state_names | yes | `["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]` |
| state_shape | yes | `[6]` |
| state_dtype | yes | `"float32"` |
| camera_keys | yes | `["observation.images.top", "observation.images.wrist"]` |
| camera_resolutions | yes | `{"observation.images.top": [360, 640], "observation.images.wrist": [480, 640]}` |
| video_codecs | yes | `{"observation.images.top": "av1", "observation.images.wrist": "av1"}` |
| extra_vector_features | yes | `[]` |

### observation.state range across datasets (degrees; gripper 0-100)

| joint | min of mins | max of mins | min of maxs | max of maxs |
|-------|-----------:|-----------:|-----------:|-----------:|
| shoulder_pan.pos | -70.81 | -46.81 | 44.0 | 81.01 |
| shoulder_lift.pos | -103.03 | -102.59 | 53.63 | 80.18 |
| elbow_flex.pos | -96.4 | -71.78 | 99.47 | 104.04 |
| wrist_flex.pos | -89.01 | 18.86 | 99.82 | 100.44 |
| wrist_roll.pos | -92.18 | -55.08 | 27.03 | 100.18 |
| gripper.pos | 1.4 | 13.77 | 46.89 | 89.28 |

### Episode start pose (median of first frame), calibration consistency evidence

Same physical rest pose should read the same values if all datasets share one calibration.

| repo | start pose median | std across episodes |
|------|-------------------|---------------------|
| `so101_left_arm_pick_red_cube_into_pot` | [-2.64, -102.59, 93.67, 61.41, -0.35, 2.0] | [2.29, 1.02, 2.62, 7.13, 3.15, 2.5] |
| `so101_left_arm_blue_cube_next_to_blue_cube` | [-12.0, -102.59, 97.54, 50.68, -9.49, 32.42] | [7.33, 1.26, 1.96, 5.04, 5.92, 8.49] |
| `so101_left_arm_pick_blue_cube_into_pot` | [-2.68, -102.33, 96.48, 53.67, 1.27, 1.9] | [2.83, 5.43, 3.15, 7.05, 2.8, 4.35] |
| `so101_left_arm_pick_banana_close_lid` | [-7.25, -102.59, 97.63, 48.84, -6.51, 29.92] | [6.65, 7.6, 1.5, 11.76, 6.87, 9.64] |
| `so101_left_arm_blue_cube_out_of_pot_onto_blue_cube` | [-4.26, -102.59, 97.89, 42.07, -2.07, 27.09] | [8.05, 10.75, 11.09, 15.25, 8.23, 7.48] |
| `so101_left_arm_red_cube_out_of_pot_onto_red_cube` | [-3.34, -102.59, 98.29, 41.63, -3.91, 27.44] | [7.11, 1.11, 3.16, 13.91, 8.98, 8.36] |
| `so101_left_arm_stack_yellow_on_orange_cube` | [-4.04, -102.59, 96.92, 51.38, -2.86, 29.38] | [11.68, 16.6, 17.94, 14.59, 9.35, 9.09] |
| `so101_left_arm_balance_red_blue_on_wooden_board` | [-8.7, -102.59, 97.54, 50.15, -8.18, 29.57] | [7.41, 3.29, 2.81, 6.46, 4.84, 8.35] |
| `so101_left_arm_red_cube_into_orange_hollow_cube` | [-11.6, -102.59, 97.98, 51.87, -13.85, 31.03] | [8.66, 6.25, 1.5, 10.02, 6.84, 12.59] |
| `so101_left_arm_stack_orange_blocks` | [-9.36, -102.59, 96.66, 54.33, -14.64, 40.58] | [8.6, 9.26, 8.24, 8.31, 7.29, 12.17] |
| `so101_left_arm_blue_cubes_shortest_to_tallest` | [-11.43, -102.59, 98.29, 51.12, -8.57, 23.35] | [10.59, 11.96, 6.95, 7.18, 7.34, 9.94] |

## Feature schema (first dataset)

| key | dtype | shape | details |
|-----|-------|-------|---------|
| observation.images.wrist | video | [480, 640, 3] | 480x640 av1 yuv420p @ 30 fps |
| observation.images.top | video | [360, 640, 3] | 360x640 av1 yuv420p @ 30 fps |
| action | float32 | [6] | ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos'] |
| observation.state | float32 | [6] | ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos'] |
| timestamp | float32 | [1] | |
| frame_index | int64 | [1] | |
| episode_index | int64 | [1] | |
| index | int64 | [1] | |
| task_index | int64 | [1] | |

## Action convention evidence

Mean |action[t] − state[t+k]| over body joints (degrees). A minimum at small k > 0 means `action` is an
absolute joint-position target that the follower reaches a few frames later (LeRobot teleop convention).

| repo | best lag k | err(k=0) | err(best) | mean abs(action−state) per joint |
|------|-----------:|---------:|----------:|----------------------------------|
| `so101_left_arm_pick_red_cube_into_pot` | 4 | 2.4112 | 1.1596 | [1.454, 3.2111, 3.9701, 1.7221, 1.6987, 4.0586] |
| `so101_left_arm_blue_cube_next_to_blue_cube` | 4 | 2.4575 | 0.8037 | [1.5266, 3.8454, 3.9467, 1.7096, 1.2592, 4.5577] |
| `so101_left_arm_pick_blue_cube_into_pot` | 4 | 2.0536 | 0.9671 | [1.2351, 3.0113, 3.5563, 1.4636, 1.0016, 3.413] |
| `so101_left_arm_pick_banana_close_lid` | 4 | 2.6968 | 0.8605 | [1.8881, 4.0606, 3.8195, 2.2335, 1.4823, 2.68] |
| `so101_left_arm_blue_cube_out_of_pot_onto_blue_cube` | 4 | 1.9187 | 0.7707 | [1.1, 2.9045, 2.9778, 1.6727, 0.9385, 4.063] |
| `so101_left_arm_red_cube_out_of_pot_onto_red_cube` | 4 | 1.9582 | 0.735 | [1.2447, 2.9535, 2.7987, 1.8074, 0.9867, 4.6248] |
| `so101_left_arm_stack_yellow_on_orange_cube` | 4 | 1.9137 | 0.8569 | [0.977, 2.9198, 3.4475, 1.3831, 0.8414, 2.549] |
| `so101_left_arm_balance_red_blue_on_wooden_board` | 4 | 2.6548 | 0.8294 | [1.6158, 4.0283, 4.2158, 1.9366, 1.4778, 3.9552] |
| `so101_left_arm_red_cube_into_orange_hollow_cube` | 4 | 2.2763 | 0.835 | [1.4356, 3.5378, 3.5594, 1.571, 1.2779, 5.5084] |
| `so101_left_arm_stack_orange_blocks` | 4 | 2.5322 | 0.8414 | [1.472, 3.7656, 3.6893, 1.639, 2.0951, 7.4805] |
| `so101_left_arm_blue_cubes_shortest_to_tallest` | 4 | 2.4436 | 0.881 | [2.0496, 3.6711, 3.0398, 2.1032, 1.3542, 4.0042] |

## Data quality

| repo | NaN/inf | max ts err (s) | max video span err (s) | idle head mean/max | idle tail mean/max | dup episodes | stats.json vs data |
|------|--------:|---------------:|-----------------------:|-------------------:|-------------------:|-------------:|--------------------|
| `so101_left_arm_pick_red_cube_into_pot` | 0 | 8.90e-07 | 5.33e-14 | 33/104 | 16/29 | - | ok |
| `so101_left_arm_blue_cube_next_to_blue_cube` | 0 | 4.45e-07 | 2.49e-14 | 26/65 | 10/35 | - | ok |
| `so101_left_arm_pick_blue_cube_into_pot` | 0 | 8.90e-07 | 5.33e-14 | 34/79 | 15/62 | - | ok |
| `so101_left_arm_pick_banana_close_lid` | 0 | 8.90e-07 | 5.33e-14 | 26/59 | 7/26 | - | ok |
| `so101_left_arm_blue_cube_out_of_pot_onto_blue_cube` | 0 | 8.90e-07 | 2.66e-14 | 74/897 | 66/897 | - | ok |
| `so101_left_arm_red_cube_out_of_pot_onto_red_cube` | 0 | 8.90e-07 | 2.66e-14 | 21/50 | 11/31 | - | ok |
| `so101_left_arm_stack_yellow_on_orange_cube` | 0 | 8.90e-07 | 5.33e-14 | 33/184 | 18/184 | - | ok |
| `so101_left_arm_balance_red_blue_on_wooden_board` | 0 | 4.45e-07 | 2.66e-14 | 28/52 | 8/28 | - | ok |
| `so101_left_arm_red_cube_into_orange_hollow_cube` | 0 | 8.90e-07 | 5.33e-14 | 26/69 | 12/43 | - | ok |
| `so101_left_arm_stack_orange_blocks` | 0 | 8.90e-07 | 4.62e-14 | 27/127 | 9/32 | - | ok |
| `so101_left_arm_blue_cubes_shortest_to_tallest` | 0 | 8.90e-07 | 4.62e-14 | 26/56 | 11/31 | - | ok |

## Video integrity

- `hungho77/so101_left_arm_pick_red_cube_into_pot`: 4 files opened, 200 episode-boundary frames decoded, 0 errors
- `hungho77/so101_left_arm_blue_cube_next_to_blue_cube`: 4 files opened, 200 episode-boundary frames decoded, 0 errors
- `hungho77/so101_left_arm_pick_blue_cube_into_pot`: 4 files opened, 200 episode-boundary frames decoded, 0 errors
- `hungho77/so101_left_arm_pick_banana_close_lid`: 2 files opened, 200 episode-boundary frames decoded, 0 errors
- `hungho77/so101_left_arm_blue_cube_out_of_pot_onto_blue_cube`: 10 files opened, 204 episode-boundary frames decoded, 0 errors
- `hungho77/so101_left_arm_red_cube_out_of_pot_onto_red_cube`: 8 files opened, 200 episode-boundary frames decoded, 0 errors
- `hungho77/so101_left_arm_stack_yellow_on_orange_cube`: 2 files opened, 200 episode-boundary frames decoded, 0 errors
- `hungho77/so101_left_arm_balance_red_blue_on_wooden_board`: 2 files opened, 200 episode-boundary frames decoded, 0 errors
- `hungho77/so101_left_arm_red_cube_into_orange_hollow_cube`: 4 files opened, 200 episode-boundary frames decoded, 0 errors
- `hungho77/so101_left_arm_stack_orange_blocks`: 2 files opened, 200 episode-boundary frames decoded, 0 errors
- `hungho77/so101_left_arm_blue_cubes_shortest_to_tallest`: 4 files opened, 200 episode-boundary frames decoded, 0 errors

## Issues

- **WARNING** [`so101_left_arm_blue_cube_next_to_blue_cube`] `short_episodes`: episodes shorter than 2 s: [21]
- **EXCLUDED** [`so101_left_arm_blue_cube_next_to_blue_cube`] `static_episode`: episodes with no motion (action range < 1.0 deg on every joint): [21]
- **WARNING** [`so101_left_arm_pick_blue_cube_into_pot`] `short_episodes`: episodes shorter than 2 s: [25]
- **EXCLUDED** [`so101_left_arm_pick_blue_cube_into_pot`] `static_episode`: episodes with no motion (action range < 1.0 deg on every joint): [25]
- **WARNING** [`so101_left_arm_pick_banana_close_lid`] `short_episodes`: episodes shorter than 2 s: [3]
- **EXCLUDED** [`so101_left_arm_pick_banana_close_lid`] `static_episode`: episodes with no motion (action range < 1.0 deg on every joint): [3]
- **INFO** [`so101_left_arm_blue_cube_out_of_pot_onto_blue_cube`] `episode_count`: 51 episodes (most datasets in the collection have 50)
- **EXCLUDED** [`so101_left_arm_blue_cube_out_of_pot_onto_blue_cube`] `static_episode`: episodes with no motion (action range < 1.0 deg on every joint): [12, 28, 47]
- **EXCLUDED** [`so101_left_arm_stack_yellow_on_orange_cube`] `static_episode`: episodes with no motion (action range < 1.0 deg on every joint): [2]
- **WARNING** [`so101_left_arm_red_cube_into_orange_hollow_cube`] `short_episodes`: episodes shorter than 2 s: [22]
- **EXCLUDED** [`so101_left_arm_red_cube_into_orange_hollow_cube`] `static_episode`: episodes with no motion (action range < 1.0 deg on every joint): [22]
- **WARNING** [`so101_left_arm_blue_cubes_shortest_to_tallest`] `short_episodes`: episodes shorter than 2 s: [26]
- **EXCLUDED** [`so101_left_arm_blue_cubes_shortest_to_tallest`] `static_episode`: episodes with no motion (action range < 1.0 deg on every joint): [26]
