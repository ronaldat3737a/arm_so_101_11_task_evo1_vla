# Sim2Real checklist

Status: PASS / FAIL / NOT YET VERIFIED. Evidence in brackets. Items that need the physical robot cannot be PASS yet.

| item | status | evidence / notes |
|---|---|---|
| robot geometry matches | PASS (model level) | official SO-101 MJCF from TheRobotStudio/SO-ARM100 @ a758567. The physical arm build itself is NOT YET VERIFIED |
| joint convention matches | PASS | joint names and order identical in dataset, LeRobot `so_follower` and MJCF (`tests/test_simulation.py::test_model_joint_and_actuator_order`) |
| action convention matches | PASS | absolute joint targets in degrees (LeRobot `use_degrees=True`), state follows action by ~4 frames in the real data. The sim consumes the same canonical action (`tests/test_action_adapter.py`) |
| zero / calibration offset sim vs real | NOT YET VERIFIED | "new calibration" zero = mid-range in both. Real rest pose exceeds the MJCF limits by ≤5° (shoulder_lift −105.5 vs −100, elbow 104 vs 96.8): small offsets exist and must be measured on the physical robot |
| camera viewpoint approximately matches | NOT YET VERIFIED | sim top camera placed qualitatively from the video frames; no intrinsics/extrinsics of the real camera available |
| image resolution matches | PASS | sim renders top 640×360, wrist 640×480 = dataset (`test_observation_matches_dataset_schema`) |
| action normalization matches | PASS | the same checkpoint preprocessor/postprocessor (train-only MIN_MAX stats) is used in sim, offline and real inference via the official `predict_action` |
| state normalization matches | PASS | as above |
| gripper convention matches | PARTIAL | canonical 0..100 in all three. Sim maps 0..100 linearly onto hinge −10°..100° (upstream says LeRobot's gripper mapping "is not yet reflected" in the MJCF). Real-arm check NOT YET VERIFIED |
| gripper force | PARTIAL | sim gripper torque = 50 % of STS3215, mirroring LeRobot's `Max_Torque_Limit=500` |
| workspace matches | NOT YET VERIFIED | sim workspace r 0.13–0.29 m around the base; the real table layout relative to the base is unknown |
| object scale matches | NOT YET VERIFIED | estimated from video (reports/simulation_asset_report.md) |
| object colors match | PARTIAL | approximate RGB; no colour calibration |
| object physical behaviour reasonable | PARTIAL | primitives with tuned contacts; oracle solves 11/11 tasks on the fixed layout and task_01 at 58/60 randomized positions |
| latency measured | PARTIAL | ACT inference P50 3.4 ms / P95 6.7 ms on RTX 3050 (sim loop). Evo-1 NOT YET VERIFIED. Real camera + bus latency NOT YET VERIFIED |
| safety limits active | NOT YET VERIFIED | deployment script (Phase 12) not implemented yet |
