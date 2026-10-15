# Simulation evaluation: `checkpoints/evo1_so101_s2lite_selective_llm_rpsv2c/best`

Policy: evo1; profile: position_visual; 5 episodes per split; seeds from 1000.

Inference latency per step (incl. pre/post-processing): P50 5.2 ms, P95 6.8 ms, max 924.6 ms (n=37069). Peak RSS 5.841 GB, peak VRAM 3.42 GB.

Network call per action chunk: P50 564.6 ms, P95 581.9 ms, max 924.6 ms (n=748; max includes compile warm-up).

Model config: {'chunk_size': 50, 'n_action_steps': 50, 'num_inference_timesteps': 8, 'vlm_num_layers': 14, 'use_amp': True, 'image_resolution': (448, 448), 'max_views': 3, 'training_stage': 'stage1'}; overrides: {'num_inference_timesteps': 8}; quantization: None

| Task | Split | Episodes | Success | Success Rate | Timeout | Collision eps | Main failure reasons |
|---|---|---:|---:|---:|---:|---:|---|
| task_01 | train | 5 | 0 | 0% | 4 | 3 | timeout (4), pot_tipped_over (1) |
| task_02 | train | 5 | 1 | 20% | 4 | 5 | timeout (4) |
| task_03 | train | 5 | 1 | 20% | 1 | 4 | pot_tipped_over (3), timeout (1) |
| task_04 | train | 5 | 0 | 0% | 5 | 5 | timeout (5) |
| task_05 | train | 5 | 0 | 0% | 0 | 0 | pot_tipped_over (5) |
| task_06 | train | 5 | 0 | 0% | 0 | 3 | pot_tipped_over (5) |
| task_07 | train | 5 | 0 | 0% | 5 | 5 | timeout (5) |
| task_08 | train | 5 | 0 | 0% | 5 | 5 | timeout (5) |
| task_09 | train | 5 | 0 | 0% | 5 | 2 | timeout (5) |
| task_10 | train | 5 | 0 | 0% | 5 | 2 | timeout (5) |
| task_11 | train | 5 | 0 | 0% | 0 | 0 | wrong_order (5) |

## Overall

- train: 2/55 = 4%

Note: 'collision eps' counts episodes with any robot–table contact, including the resting gripper at the start pose.
