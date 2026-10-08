# Simulation evaluation: `outputs/act_task01_20261007_172212/checkpoints/010000/pretrained_model`

Policy: act; profile: position; 10 episodes per split; seeds from 1000.

Inference latency per step (incl. pre/post-processing): P50 3.6 ms, P95 22.6 ms, max 489.3 ms (n=18000). Peak RSS 3.901 GB, peak VRAM 0.26 GB.

| Task | Split | Episodes | Success | Success Rate | Timeout | Collision eps | Main failure reasons |
|---|---|---:|---:|---:|---:|---:|---|
| task_01 | train | 10 | 0 | 0% | 10 | 0 | timeout (10) |
| task_01 | test | 10 | 0 | 0% | 10 | 0 | timeout (10) |

## Overall

- train: 0/10 = 0%
- test: 0/10 = 0%  ← unseen-position success rate

Note: 'collision eps' counts episodes with any robot–table contact, including the resting gripper at the start pose.
