# Simulation evaluation: `checkpoints/evo1_so101_s2lite_selective_llm_rpsv2/best`

Policy: evo1; profile: position; 5 episodes per split; seeds from 1000.

Inference latency per step (incl. pre/post-processing): P50 5.3 ms, P95 6.9 ms, max 69856.4 ms (n=37053). Peak RSS 10.117 GB, peak VRAM 3.42 GB.

Network call per action chunk: P50 335.0 ms, P95 341.0 ms, max 69856.4 ms (n=750; max includes compile warm-up).

Model config: {'chunk_size': 50, 'n_action_steps': 50, 'num_inference_timesteps': 8, 'vlm_num_layers': 14, 'use_amp': True, 'image_resolution': (448, 448), 'max_views': 3, 'training_stage': 'stage1'}; overrides: {'num_inference_timesteps': 8}; quantization: {'scheme': 'w8a8_vision', 'compile': True, 'quantized_linear_layers': 144, 'library': 'torchao', 'config': 'Int8DynamicActivationInt8WeightConfig', 'int8_weight_tensors': {'vision_tower': 144, 'multi_modal_projector': 0, 'language_model': 0, 'action_head': 0}}

| Task | Split | Episodes | Success | Success Rate | Timeout | Collision eps | Main failure reasons |
|---|---|---:|---:|---:|---:|---:|---|
| task_01 | test | 5 | 0 | 0% | 5 | 5 | timeout (5) |
| task_02 | test | 5 | 0 | 0% | 5 | 4 | timeout (5) |
| task_03 | test | 5 | 0 | 0% | 2 | 4 | timeout (2), pot_tipped_over (3) |
| task_04 | test | 5 | 0 | 0% | 4 | 3 | timeout (4), pot_tipped_over (1) |
| task_05 | test | 5 | 0 | 0% | 0 | 0 | pot_tipped_over (5) |
| task_06 | test | 5 | 0 | 0% | 0 | 0 | pot_tipped_over (5) |
| task_07 | test | 5 | 0 | 0% | 5 | 4 | timeout (5) |
| task_08 | test | 5 | 0 | 0% | 5 | 5 | timeout (5) |
| task_09 | test | 5 | 0 | 0% | 5 | 2 | timeout (5) |
| task_10 | test | 5 | 0 | 0% | 5 | 3 | timeout (5) |
| task_11 | test | 5 | 0 | 0% | 0 | 0 | wrong_order (5) |

## Overall

- test: 0/55 = 0%  ← unseen-position success rate

Note: 'collision eps' counts episodes with any robot–table contact, including the resting gripper at the start pose.
