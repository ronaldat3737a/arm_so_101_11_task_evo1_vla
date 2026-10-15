# Simulation evaluation: `checkpoints/evo1_so101_s2lite_selective_llm_rpsv2c/best`

Policy: evo1; profile: position; 5 episodes per split; seeds from 1000.

Inference latency per step (incl. pre/post-processing): P50 5.4 ms, P95 7.0 ms, max 933.4 ms (n=39422). Peak RSS 5.693 GB, peak VRAM 3.42 GB.

Network call per action chunk: P50 565.4 ms, P95 569.3 ms, max 933.4 ms (n=797; max includes compile warm-up).

Model config: {'chunk_size': 50, 'n_action_steps': 50, 'num_inference_timesteps': 8, 'vlm_num_layers': 14, 'use_amp': True, 'image_resolution': (448, 448), 'max_views': 3, 'training_stage': 'stage1'}; overrides: {'num_inference_timesteps': 8}; quantization: None

| Task | Split | Episodes | Success | Success Rate | Timeout | Collision eps | Main failure reasons |
|---|---|---:|---:|---:|---:|---:|---|
| task_01 | test | 5 | 0 | 0% | 3 | 4 | timeout (3), pot_tipped_over (2) |
| task_02 | test | 5 | 0 | 0% | 5 | 4 | timeout (5) |
| task_03 | test | 5 | 0 | 0% | 5 | 4 | timeout (5) |
| task_04 | test | 5 | 0 | 0% | 5 | 3 | timeout (5) |
| task_05 | test | 5 | 0 | 0% | 2 | 2 | timeout (2), pot_tipped_over (3) |
| task_06 | test | 5 | 0 | 0% | 0 | 2 | pot_tipped_over (5) |
| task_07 | test | 5 | 0 | 0% | 5 | 4 | timeout (5) |
| task_08 | test | 5 | 1 | 20% | 4 | 5 | timeout (4) |
| task_09 | test | 5 | 0 | 0% | 5 | 0 | timeout (5) |
| task_10 | test | 5 | 0 | 0% | 5 | 4 | timeout (5) |
| task_11 | test | 5 | 0 | 0% | 0 | 0 | wrong_order (5) |

## Overall

- test: 1/55 = 2%  ← unseen-position success rate

Note: 'collision eps' counts episodes with any robot–table contact, including the resting gripper at the start pose.
