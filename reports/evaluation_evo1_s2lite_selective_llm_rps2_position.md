# Simulation evaluation: `checkpoints/evo1_so101_s2lite_selective_llm_rps2/best`

Policy: evo1; profile: position; 5 episodes per split; seeds from 1000.

Inference latency per step (incl. pre/post-processing): P50 5.2 ms, P95 6.8 ms, max 1200.0 ms (n=35184). Peak RSS 5.163 GB, peak VRAM 3.42 GB.

| Task | Split | Episodes | Success | Success Rate | Timeout | Collision eps | Main failure reasons |
|---|---|---:|---:|---:|---:|---:|---|
| task_01 | test | 5 | 0 | 0% | 0 | 4 | pot_tipped_over (5) |
| task_02 | test | 5 | 0 | 0% | 5 | 5 | timeout (5) |
| task_03 | test | 5 | 0 | 0% | 3 | 2 | timeout (3), pot_tipped_over (2) |
| task_04 | test | 5 | 0 | 0% | 3 | 2 | pot_tipped_over (2), timeout (3) |
| task_05 | test | 5 | 0 | 0% | 0 | 0 | pot_tipped_over (5) |
| task_06 | test | 5 | 0 | 0% | 0 | 1 | pot_tipped_over (5) |
| task_07 | test | 5 | 0 | 0% | 5 | 5 | timeout (5) |
| task_08 | test | 5 | 0 | 0% | 5 | 5 | timeout (5) |
| task_09 | test | 5 | 0 | 0% | 5 | 4 | timeout (5) |
| task_10 | test | 5 | 0 | 0% | 5 | 3 | timeout (5) |
| task_11 | test | 5 | 0 | 0% | 0 | 1 | wrong_order (5) |

## Overall

- test: 0/55 = 0%  ← unseen-position success rate

Note: 'collision eps' counts episodes with any robot–table contact, including the resting gripper at the start pose.
