# Simulation evaluation: `checkpoints/evo1_so101_s2lite_selective_llm_rps/best`

Policy: evo1; profile: position; 5 episodes per split; seeds from 1000.

Inference latency per step (incl. pre/post-processing): P50 5.8 ms, P95 7.4 ms, max 1479.4 ms (n=33342). Peak RSS 4.917 GB, peak VRAM 3.42 GB.

| Task | Split | Episodes | Success | Success Rate | Timeout | Collision eps | Main failure reasons |
|---|---|---:|---:|---:|---:|---:|---|
| task_01 | test | 5 | 0 | 0% | 1 | 5 | pot_tipped_over (4), timeout (1) |
| task_02 | test | 5 | 0 | 0% | 5 | 4 | timeout (5) |
| task_03 | test | 5 | 0 | 0% | 2 | 5 | wrong_object_in_pot (1), timeout (2), pot_tipped_over (2) |
| task_04 | test | 5 | 0 | 0% | 2 | 3 | banana_fell_off_table (1), timeout (2), pot_tipped_over (2) |
| task_05 | test | 5 | 0 | 0% | 0 | 0 | pot_tipped_over (5) |
| task_06 | test | 5 | 0 | 0% | 0 | 1 | pot_tipped_over (5) |
| task_07 | test | 5 | 0 | 0% | 5 | 5 | timeout (5) |
| task_08 | test | 5 | 0 | 0% | 5 | 5 | timeout (5) |
| task_09 | test | 5 | 0 | 0% | 5 | 2 | timeout (5) |
| task_10 | test | 5 | 0 | 0% | 5 | 5 | timeout (5) |
| task_11 | test | 5 | 0 | 0% | 0 | 1 | wrong_order (5) |

## Overall

- test: 0/55 = 0%  ← unseen-position success rate

Note: 'collision eps' counts episodes with any robot–table contact, including the resting gripper at the start pose.
