# Normalization stats (train episodes only)

Train episodes: 432, frames: 200257. Held-out (val/test/excluded) episodes not used: 119.

Evo-1 uses MIN_MAX for state/action (lerobot.policies.evo1 default), so min/max define [-1, 1].

Verification vs. direct recomputation from train frames (max abs diff): {"action": {"min": "0.00e+00", "max": "0.00e+00", "mean": "5.46e-06", "std": "8.90e-06", "count": 200257}, "observation.state": {"min": "0.00e+00", "max": "0.00e+00", "mean": "7.55e-06", "std": "2.54e-05", "count": 200257}}


## action

| joint | train min | train max | train mean | train std | all-episode min | all-episode max |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -71.12 | 75.16 | -2.71 | 21.72 | -71.12 | 81.67 |
| shoulder_lift | -105.54 | 76.09 | -26.23 | 46.33 | -105.54 | 79.16 |
| elbow_flex | -99.25 | 99.87 | 27.67 | 45.90 | -99.25 | 100.04 |
| wrist_flex | -90.20 | 101.19 | 56.12 | 22.14 | -90.20 | 101.19 |
| wrist_roll | -126.73 | 100.62 | -8.74 | 19.83 | -126.73 | 100.62 |
| gripper | 0.00 | 89.64 | 20.29 | 15.60 | 0.00 | 89.64 |

## observation.state

| joint | train min | train max | train mean | train std | all-episode min | all-episode max |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -70.81 | 74.33 | -2.88 | 21.68 | -70.81 | 81.01 |
| shoulder_lift | -103.03 | 77.54 | -25.02 | 46.81 | -103.03 | 80.18 |
| elbow_flex | -96.40 | 104.04 | 29.37 | 45.50 | -96.40 | 104.04 |
| wrist_flex | -89.01 | 100.44 | 56.45 | 21.94 | -89.01 | 100.44 |
| wrist_roll | -92.18 | 100.18 | -8.61 | 19.50 | -92.18 | 100.18 |
| gripper | 1.40 | 89.28 | 23.46 | 12.29 | 1.40 | 89.28 |
