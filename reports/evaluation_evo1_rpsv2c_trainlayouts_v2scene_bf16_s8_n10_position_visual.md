# Simulation evaluation: `checkpoints/evo1_so101_s2lite_selective_llm_rpsv2c/best`

Policy: evo1; profile: position_visual; 5 episodes per split; seeds from 1000.

Inference latency per step (incl. pre/post-processing): P50 5.2 ms, P95 564.6 ms, max 973.3 ms (n=37832). Peak RSS 5.347 GB, peak VRAM 3.42 GB.

Network call per action chunk: P50 564.6 ms, P95 568.0 ms, max 973.3 ms (n=3792; max includes compile warm-up).

Model config: {'chunk_size': 50, 'n_action_steps': 10, 'num_inference_timesteps': 8, 'vlm_num_layers': 14, 'use_amp': True, 'image_resolution': (448, 448), 'max_views': 3, 'training_stage': 'stage1'}; overrides: {'n_action_steps': 10, 'num_inference_timesteps': 8}; quantization: None

| Task | Split | Episodes | Success | Success Rate | Timeout | Collision eps | Main failure reasons |
|---|---|---:|---:|---:|---:|---:|---|
| task_01 | train | 5 | 1 | 20% | 3 | 4 | timeout (3), pot_tipped_over (1) |
| task_02 | train | 5 | 0 | 0% | 5 | 5 | timeout (5) |
| task_03 | train | 5 | 0 | 0% | 4 | 5 | timeout (4), pot_tipped_over (1) |
| task_04 | train | 5 | 0 | 0% | 4 | 3 | pot_tipped_over (1), timeout (4) |
| task_05 | train | 5 | 0 | 0% | 0 | 0 | pot_tipped_over (5) |
| task_06 | train | 5 | 0 | 0% | 0 | 0 | pot_tipped_over (5) |
| task_07 | train | 5 | 0 | 0% | 5 | 5 | timeout (5) |
| task_08 | train | 5 | 0 | 0% | 4 | 4 | timeout (4), wrong_cube_chosen (1) |
| task_09 | train | 5 | 0 | 0% | 5 | 1 | timeout (5) |
| task_10 | train | 5 | 0 | 0% | 5 | 4 | timeout (5) |
| task_11 | train | 5 | 1 | 20% | 0 | 0 | wrong_order (4) |

## Overall

- train: 2/55 = 4%

Note: 'collision eps' counts episodes with any robot–table contact, including the resting gripper at the start pose.
