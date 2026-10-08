# Aggregate dataset verification

## Verdict: **PASS**

Peak RSS 1.228 GB; frame decode (2 cams) p50 15 ms, p95 23 ms

| status | check | detail |
|---|---|---|
| PASS | total episodes | 551 vs 551 |
| PASS | total frames | 253745 vs 253745 |
| PASS | fps | 30 |
| PASS | tasks | 11 |
| PASS | camera keys | ['observation.images.wrist', 'observation.images.top'] |
| PASS | action feature |  |
| PASS | observation.state feature |  |
| PASS | meta/stats.json is train-only stats |  |
| PASS | all 551 global episodes map to source (task, length, first/last action+state exact) | [] |
| PASS | decoded frames pixel-identical to source (both cameras) | 66 frames x 2 cams; [] |
| PASS | train subset episode count | 432 |
| PASS | train subset returns action chunk (50, 6) + action_is_pad + both cameras uint8 | {'observation.images.wrist': (3, 480, 640), 'observation.images.top': (3, 360, 640), 'action': (50, 6), 'observation.state': (1, 6), 'timestamp': (), 'frame_index': (), 'episode_index': (), 'index': ( |
| PASS | train subset never yields held-out episodes | 200 episodes sampled |
| PASS | peak RAM during verification < 6 GB | 1.23 GB |
