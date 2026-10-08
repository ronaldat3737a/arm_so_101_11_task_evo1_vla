# Dataset validation

Generated: 2026-10-07T04:24:53.584359+00:00

## Verdict: **PASS** (112/112 checks passed)

| status | check | detail |
|--------|-------|--------|
| PASS | audit report exists | /home/ncd/Desktop/VRB/task_week_2/so101_evo1/reports/dataset_audit.json |
| PASS | audit has no errors besides documented static episodes | [] |
| PASS | cross-dataset consistent: codebase_version | ["v3.0"] |
| PASS | cross-dataset consistent: robot_type | ["so_follower"] |
| PASS | cross-dataset consistent: fps | [30] |
| PASS | cross-dataset consistent: action_names | [["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]] |
| PASS | cross-dataset consistent: action_shape | [[6]] |
| PASS | cross-dataset consistent: state_names | [["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]] |
| PASS | cross-dataset consistent: state_shape | [[6]] |
| PASS | cross-dataset consistent: camera_keys | [["observation.images.top", "observation.images.wrist"]] |
| PASS | cross-dataset consistent: camera_resolutions | [{"observation.images.top": [360, 640], "observation.images.wrist": [480, 640]}] |
| PASS | revision unchanged: hungho77/so101_left_arm_pick_red_cube_into_pot | audit 62d3cb12 hub 62d3cb12 |
| PASS | revision unchanged: hungho77/so101_left_arm_blue_cube_next_to_blue_cube | audit 1d32c7fa hub 1d32c7fa |
| PASS | revision unchanged: hungho77/so101_left_arm_pick_blue_cube_into_pot | audit 98192ba8 hub 98192ba8 |
| PASS | revision unchanged: hungho77/so101_left_arm_pick_banana_close_lid | audit 63a2c901 hub 63a2c901 |
| PASS | revision unchanged: hungho77/so101_left_arm_blue_cube_out_of_pot_onto_blue_cube | audit ace616ca hub ace616ca |
| PASS | revision unchanged: hungho77/so101_left_arm_red_cube_out_of_pot_onto_red_cube | audit 9363738d hub 9363738d |
| PASS | revision unchanged: hungho77/so101_left_arm_stack_yellow_on_orange_cube | audit ef6cb56c hub ef6cb56c |
| PASS | revision unchanged: hungho77/so101_left_arm_balance_red_blue_on_wooden_board | audit 460f5d1a hub 460f5d1a |
| PASS | revision unchanged: hungho77/so101_left_arm_red_cube_into_orange_hollow_cube | audit cf2c68df hub cf2c68df |
| PASS | revision unchanged: hungho77/so101_left_arm_stack_orange_blocks | audit b932726e hub b932726e |
| PASS | revision unchanged: hungho77/so101_left_arm_blue_cubes_shortest_to_tallest | audit 629eba59 hub 629eba59 |
| PASS | tasks.yaml lists the configured repos in order |  |
| PASS | task_01 instruction == dataset task string |  |
| PASS | task_01 excluded == audit static episodes |  |
| PASS | task_02 instruction == dataset task string |  |
| PASS | task_02 excluded == audit static episodes |  |
| PASS | task_03 instruction == dataset task string |  |
| PASS | task_03 excluded == audit static episodes |  |
| PASS | task_04 instruction == dataset task string |  |
| PASS | task_04 excluded == audit static episodes |  |
| PASS | task_05 instruction == dataset task string |  |
| PASS | task_05 excluded == audit static episodes |  |
| PASS | task_06 instruction == dataset task string |  |
| PASS | task_06 excluded == audit static episodes |  |
| PASS | task_07 instruction == dataset task string |  |
| PASS | task_07 excluded == audit static episodes |  |
| PASS | task_08 instruction == dataset task string |  |
| PASS | task_08 excluded == audit static episodes |  |
| PASS | task_09 instruction == dataset task string |  |
| PASS | task_09 excluded == audit static episodes |  |
| PASS | task_10 instruction == dataset task string |  |
| PASS | task_10 excluded == audit static episodes |  |
| PASS | task_11 instruction == dataset task string |  |
| PASS | task_11 excluded == audit static episodes |  |
| PASS | task_01 split sizes | train 40 val 5 test 5 excluded 0 |
| PASS | task_02 split sizes | train 39 val 5 test 5 excluded 1 |
| PASS | task_03 split sizes | train 39 val 5 test 5 excluded 1 |
| PASS | task_04 split sizes | train 39 val 5 test 5 excluded 1 |
| PASS | task_05 split sizes | train 38 val 5 test 5 excluded 3 |
| PASS | task_06 split sizes | train 40 val 5 test 5 excluded 0 |
| PASS | task_07 split sizes | train 39 val 5 test 5 excluded 1 |
| PASS | task_08 split sizes | train 40 val 5 test 5 excluded 0 |
| PASS | task_09 split sizes | train 39 val 5 test 5 excluded 1 |
| PASS | task_10 split sizes | train 40 val 5 test 5 excluded 0 |
| PASS | task_11 split sizes | train 39 val 5 test 5 excluded 1 |
| PASS | global split disjoint |  |
| PASS | task_01 local snapshot at audited revision | /home/ncd/Desktop/VRB/task_week_2/so101_evo1/datasets/hf_cache/datasets--hungho77--so101_left_arm_pick_red_cube_into_pot/snapshots/62d3cb12cc705fd39c851833853ec49b2ca20fe4 |
| PASS | task_01 videos present locally | 0 missing |
| PASS | task_01 LeRobotDataset loads | 50 episodes, 37536 frames, 0.1s |
| PASS | task_01 loader counts == audit |  |
| PASS | task_01 sampled frames decode and match schema | 18 frames, 16 ms/frame |
| PASS | task_02 local snapshot at audited revision | /home/ncd/Desktop/VRB/task_week_2/so101_evo1/datasets/hf_cache/datasets--hungho77--so101_left_arm_blue_cube_next_to_blue_cube/snapshots/1d32c7fa37f245c6acc9dce944515beba65867db |
| PASS | task_02 videos present locally | 0 missing |
| PASS | task_02 LeRobotDataset loads | 50 episodes, 14841 frames, 0.1s |
| PASS | task_02 loader counts == audit |  |
| PASS | task_02 sampled frames decode and match schema | 18 frames, 15 ms/frame |
| PASS | task_03 local snapshot at audited revision | /home/ncd/Desktop/VRB/task_week_2/so101_evo1/datasets/hf_cache/datasets--hungho77--so101_left_arm_pick_blue_cube_into_pot/snapshots/98192ba8b697e80f811f68137c7ed2e042ddd64c |
| PASS | task_03 videos present locally | 0 missing |
| PASS | task_03 LeRobotDataset loads | 50 episodes, 29688 frames, 0.1s |
| PASS | task_03 loader counts == audit |  |
| PASS | task_03 sampled frames decode and match schema | 18 frames, 14 ms/frame |
| PASS | task_04 local snapshot at audited revision | /home/ncd/Desktop/VRB/task_week_2/so101_evo1/datasets/hf_cache/datasets--hungho77--so101_left_arm_pick_banana_close_lid/snapshots/63a2c9016f3d15ec3ef5de244f0ecd677afc27ce |
| PASS | task_04 videos present locally | 0 missing |
| PASS | task_04 LeRobotDataset loads | 50 episodes, 22278 frames, 0.1s |
| PASS | task_04 loader counts == audit |  |
| PASS | task_04 sampled frames decode and match schema | 18 frames, 15 ms/frame |
| PASS | task_05 local snapshot at audited revision | /home/ncd/Desktop/VRB/task_week_2/so101_evo1/datasets/hf_cache/datasets--hungho77--so101_left_arm_blue_cube_out_of_pot_onto_blue_cube/snapshots/ace616ca5651dbcd2a6b90c356dfbbe83e75c89d |
| PASS | task_05 videos present locally | 0 missing |
| PASS | task_05 LeRobotDataset loads | 51 episodes, 23103 frames, 0.1s |
| PASS | task_05 loader counts == audit |  |
| PASS | task_05 sampled frames decode and match schema | 18 frames, 15 ms/frame |
| PASS | task_06 local snapshot at audited revision | /home/ncd/Desktop/VRB/task_week_2/so101_evo1/datasets/hf_cache/datasets--hungho77--so101_left_arm_red_cube_out_of_pot_onto_red_cube/snapshots/9363738dafae5bd65f9a7cd73d63fc04c0c0ff66 |
| PASS | task_06 videos present locally | 0 missing |
| PASS | task_06 LeRobotDataset loads | 50 episodes, 21837 frames, 0.1s |
| PASS | task_06 loader counts == audit |  |
| PASS | task_06 sampled frames decode and match schema | 18 frames, 15 ms/frame |
| PASS | task_07 local snapshot at audited revision | /home/ncd/Desktop/VRB/task_week_2/so101_evo1/datasets/hf_cache/datasets--hungho77--so101_left_arm_stack_yellow_on_orange_cube/snapshots/ef6cb56c979b4c60c50f9582e2c055df1b2f7a92 |
| PASS | task_07 videos present locally | 0 missing |
| PASS | task_07 LeRobotDataset loads | 50 episodes, 23691 frames, 0.1s |
| PASS | task_07 loader counts == audit |  |
| PASS | task_07 sampled frames decode and match schema | 18 frames, 15 ms/frame |
| PASS | task_08 local snapshot at audited revision | /home/ncd/Desktop/VRB/task_week_2/so101_evo1/datasets/hf_cache/datasets--hungho77--so101_left_arm_balance_red_blue_on_wooden_board/snapshots/460f5d1a012109e1ecaef7f24f852c8abd0a3b74 |
| PASS | task_08 videos present locally | 0 missing |
| PASS | task_08 LeRobotDataset loads | 50 episodes, 14217 frames, 0.1s |
| PASS | task_08 loader counts == audit |  |
| PASS | task_08 sampled frames decode and match schema | 18 frames, 14 ms/frame |
| PASS | task_09 local snapshot at audited revision | /home/ncd/Desktop/VRB/task_week_2/so101_evo1/datasets/hf_cache/datasets--hungho77--so101_left_arm_red_cube_into_orange_hollow_cube/snapshots/cf2c68df347b69612da0a0efd1f294b08ebf6533 |
| PASS | task_09 videos present locally | 0 missing |
| PASS | task_09 LeRobotDataset loads | 50 episodes, 20642 frames, 0.1s |
| PASS | task_09 loader counts == audit |  |
| PASS | task_09 sampled frames decode and match schema | 18 frames, 15 ms/frame |
| PASS | task_10 local snapshot at audited revision | /home/ncd/Desktop/VRB/task_week_2/so101_evo1/datasets/hf_cache/datasets--hungho77--so101_left_arm_stack_orange_blocks/snapshots/b932726e629f81cb142c4545d9d958340d28ec31 |
| PASS | task_10 videos present locally | 0 missing |
| PASS | task_10 LeRobotDataset loads | 50 episodes, 15682 frames, 0.1s |
| PASS | task_10 loader counts == audit |  |
| PASS | task_10 sampled frames decode and match schema | 18 frames, 15 ms/frame |
| PASS | task_11 local snapshot at audited revision | /home/ncd/Desktop/VRB/task_week_2/so101_evo1/datasets/hf_cache/datasets--hungho77--so101_left_arm_blue_cubes_shortest_to_tallest/snapshots/629eba595368fb7040db4ee8ea0b3da382ffde03 |
| PASS | task_11 videos present locally | 0 missing |
| PASS | task_11 LeRobotDataset loads | 50 episodes, 30230 frames, 0.1s |
| PASS | task_11 loader counts == audit |  |
| PASS | task_11 sampled frames decode and match schema | 18 frames, 15 ms/frame |

## Split (episodes per task)

| task | dataset | train | val | test | excluded |
|------|---------|------:|----:|-----:|----------|
| task_01 | `so101_left_arm_pick_red_cube_into_pot` | 40 | 5 | 5 | - |
| task_02 | `so101_left_arm_blue_cube_next_to_blue_cube` | 39 | 5 | 5 | [21] |
| task_03 | `so101_left_arm_pick_blue_cube_into_pot` | 39 | 5 | 5 | [25] |
| task_04 | `so101_left_arm_pick_banana_close_lid` | 39 | 5 | 5 | [3] |
| task_05 | `so101_left_arm_blue_cube_out_of_pot_onto_blue_cube` | 38 | 5 | 5 | [12, 28, 47] |
| task_06 | `so101_left_arm_red_cube_out_of_pot_onto_red_cube` | 40 | 5 | 5 | - |
| task_07 | `so101_left_arm_stack_yellow_on_orange_cube` | 39 | 5 | 5 | [2] |
| task_08 | `so101_left_arm_balance_red_blue_on_wooden_board` | 40 | 5 | 5 | - |
| task_09 | `so101_left_arm_red_cube_into_orange_hollow_cube` | 39 | 5 | 5 | [22] |
| task_10 | `so101_left_arm_stack_orange_blocks` | 40 | 5 | 5 | - |
| task_11 | `so101_left_arm_blue_cubes_shortest_to_tallest` | 39 | 5 | 5 | [26] |
| **total** | | 432 | 55 | 55 | 9 |
