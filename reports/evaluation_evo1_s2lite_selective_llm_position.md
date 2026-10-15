# Simulation evaluation: `checkpoints/evo1_so101_s2lite_selective_llm/best`

Policy: evo1; profile: position; 5 episodes per split; seeds from 1000.

Inference latency per step (incl. pre/post-processing): P50 5.2 ms, P95 7.1 ms, max 1454.4 ms (n=50780). Peak RSS 4.798 GB, peak VRAM 3.42 GB.

| Task | Split | Episodes | Success | Success Rate | Timeout | Collision eps | Main failure reasons |
|---|---|---:|---:|---:|---:|---:|---|
| task_01 | test | 5 | 0 | 0% | 5 | 0 | timeout (5) |
| task_02 | test | 5 | 0 | 0% | 5 | 0 | timeout (5) |
| task_03 | test | 5 | 0 | 0% | 5 | 0 | timeout (5) |
| task_04 | test | 5 | 0 | 0% | 4 | 3 | timeout (4), lid_fell_off_table (1) |
| task_05 | test | 5 | 0 | 0% | 5 | 0 | timeout (5) |
| task_06 | test | 5 | 0 | 0% | 5 | 1 | timeout (5) |
| task_07 | test | 5 | 0 | 0% | 5 | 0 | timeout (5) |
| task_08 | test | 5 | 0 | 0% | 5 | 1 | timeout (5) |
| task_09 | test | 5 | 0 | 0% | 5 | 1 | timeout (5) |
| task_10 | test | 5 | 0 | 0% | 5 | 0 | timeout (5) |
| task_11 | test | 5 | 0 | 0% | 4 | 1 | timeout (4), wrong_order (1) |

## Overall

- test: 0/55 = 0%  ← unseen-position success rate

Note: 'collision eps' counts episodes with any robot–table contact, including the resting gripper at the start pose.
