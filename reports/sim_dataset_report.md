# Synthetic (simulation) dataset

`datasets/sim_so101_all`: LeRobotDataset v3.0, same schema as the real data. 220 episodes, 62,998 frames, 20 per task.
Built with the official `aggregate_datasets` from:
- part 1 `datasets/sim_so101` (task_01–04): generation interrupted by the pause and repaired (`reports/sim_dataset_repair.json`);
- part 2 `datasets/sim_so101_part2` (task_05–11): `reports/sim_dataset_report_part2.{md,json}`.

Generation:
- scripted oracle (privileged object poses + IK). No learned model creates ground truth.
- profile `position_visual` (positions + moderate lighting / colour / camera noise), TRAIN-region positions only;
- deterministic seeds `100000 + 1000·task + k`. Only successful episodes are stored, so the success flag is 1 for every stored episode.

Each frame: observation.images.top (640×360), observation.images.wrist (640×480), observation.state, action
(canonical degrees, gripper 0..100), task (exact instruction), episode_index, frame_index, timestamp.
Verified: 440 frames (first/last of every episode) decode with the official loader; 0 errors.

| task | kept | attempts | oracle success | source |
|---|---:|---:|---:|---|
| task_01 | 20 | 20 | 100% | part 1 log |
| task_02 | 20 | 21 | 95% | part 1 log |
| task_03 | 20 | 20 | 100% | part 1 log |
| task_04 | 20 | 25 | 80% | part 1 log |
| task_05 | 20 | 23 | 87% | part 2 |
| task_06 | 20 | 23 | 87% | part 2 |
| task_07 | 20 | 21 | 95% | part 2 |
| task_08 | 20 | 21 | 95% | part 2 |
| task_09 | 20 | 21 | 95% | part 2 |
| task_10 | 20 | 20 | 100% | part 2 |
| task_11 | 20 | 25 | 80% | part 2 |

Per-seed attempt details (layouts, failure reasons) are only available for part 2: part 1's report was not written
because of the interruption, and its log lists the saved seeds.

Known limitation: the sim robot uses the MJCF default yellow while the real arm is cyan (sim2real checklist).
