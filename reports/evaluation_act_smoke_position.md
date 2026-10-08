# Simulation evaluation: `outputs/act_task01_20261007_171824/checkpoints/last/pretrained_model`

Policy: act; profile: position; 1 episodes per split; seeds from 1000.

Inference latency per step (incl. pre/post-processing): P50 3.4 ms, P95 6.7 ms, max 544.3 ms (n=900). Peak RSS 2.091 GB, peak VRAM 0.26 GB.

| Task | Split | Episodes | Success | Success Rate | Timeout | Collision eps | Main failure reasons |
|---|---|---:|---:|---:|---:|---:|---|
| task_01 | test | 1 | 0 | 0% | 1 | 0 | timeout (1) |

## Overall

- test: 0/1 = 0%  ← unseen-position success rate

Note: 'collision eps' counts episodes with any robot–table contact, including the resting gripper at the start pose.
