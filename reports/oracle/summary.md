# Oracle (scripted expert) success in MuJoCo

Proves task solvability at train / val / test (unseen) positions. Not a policy result.

| task | profile | train | val | test | main failure reasons |
|---|---|---|---|---|---|
| task_01 | fixed | - | - | 1/1 |  |
| task_01 | position | 20/20 | 20/20 | 18/20 | unsolved: (1), unsolved: red_cube still moving (1) |
| task_02 | fixed | - | - | 1/1 |  |
| task_02 | position | 4/5 | 5/5 | 5/5 | unsolved: blue_cube not next to the other cube (1) |
| task_03 | fixed | - | - | 1/1 |  |
| task_03 | position | 5/5 | 5/5 | 5/5 |  |
| task_04 | fixed | - | - | 1/1 |  |
| task_04 | position | 4/5 | 3/5 | 3/5 | unsolved: banana not in pot (2), unsolved: lid not on pot (2), unsolved: lid still touching gripper (1) |
| task_05 | fixed | - | - | 1/1 |  |
| task_05 | position | 5/5 | 5/5 | 4/5 | unsolved: blue_cube not on blue_cube_base (1) |
| task_06 | fixed | - | - | 1/1 |  |
| task_06 | position | 4/5 | 5/5 | 5/5 | pot_tipped_over (1) |
| task_07 | fixed | - | - | 1/1 |  |
| task_07 | position | 4/5 | 5/5 | 5/5 | unsolved: yellow_cube not on orange_cube (1) |
| task_08 | fixed | - | - | 1/1 |  |
| task_08 | position | 5/5 | 5/5 | 5/5 |  |
| task_09 | fixed | - | - | 1/1 |  |
| task_09 | position | 5/5 | 4/5 | 5/5 | unsolved: red cube not inside the hollow cube (1) |
| task_10 | fixed | - | - | 1/1 |  |
| task_10 | position | 5/5 | 5/5 | 5/5 |  |
| task_11 | fixed | - | - | 1/1 |  |
| task_11 | position | 3/5 | 3/5 | 5/5 | wrong_order (3), unsolved: 1 blocks still in the box (1) |

Peak RSS of the last run: 0.846 GB
