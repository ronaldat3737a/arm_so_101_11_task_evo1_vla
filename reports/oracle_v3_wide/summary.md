# Oracle (scripted expert) success in MuJoCo

Proves task solvability at train / val / test (unseen) positions. Not a policy result.

| task | profile | train | val | test | main failure reasons |
|---|---|---|---|---|---|
| task_01 | position_wide | 4/5 | - | - | timeout (1) |
| task_02 | position_wide | 5/5 | - | - |  |
| task_03 | position_wide | 4/5 | - | - | timeout (1) |
| task_04 | position_wide | 4/5 | - | - | unsolved: banana not in pot (1) |
| task_05 | position_wide | 4/5 | - | - | unsolved: blue_cube not on blue_cube_base (1) |
| task_06 | position_wide | 2/5 | - | - | unsolved: red_cube not on red_cube_base (2), pot_tipped_over (1) |
| task_07 | position_wide | 4/5 | - | - | unsolved: yellow_cube not on orange_cube (1) |
| task_08 | position_wide | 5/5 | - | - |  |
| task_09 | position_wide | 4/5 | - | - | unsolved: red cube not inside the hollow cube (1) |
| task_10 | position_wide | 5/5 | - | - |  |
| task_11 | position_wide | 3/5 | - | - | wrong_order (2) |

Peak RSS of the last run: 1.03 GB
