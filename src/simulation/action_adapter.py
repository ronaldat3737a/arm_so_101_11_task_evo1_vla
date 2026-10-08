"""Canonical SO-101 action/state representation <-> MuJoCo joint angles.

Canonical representation (dataset, policy, real robot): LeRobot `so_follower` with `use_degrees=True`,
float[6] in this order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll (degrees, 0 = middle of the
calibrated range), gripper (0..100, 0 = closed, 100 = open).

Simulation: TheRobotStudio SO-ARM100 `so101_new_calib*.xml`, whose joint zero is also the middle of the range
("new calibration"), joints in radians. Hence for the 5 body joints:  sim_rad = deg2rad(canonical_deg).

Gripper: the MJCF gripper hinge spans [-10°, 100°] (range -0.1745..1.7453 rad). The upstream README states the
LeRobot 0..100 linear gripper mapping "is not yet reflected" in the model, so we map 0..100 linearly onto the
hinge range (the same choice LeIsaac makes for the gripper). This is an approximation, recorded in the sim2real checklist.

NOT used: LeIsaac's convert_lerobot_action_to_leisaac. Its body joints assume the -100..100 motor mode and are
wrong for degree data.
"""

from __future__ import annotations

import numpy as np

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FEATURE_NAMES = [f"{j}.pos" for j in JOINT_NAMES]
N_BODY = 5
GRIPPER_MIN_DEG = -10.0  # hinge angle at canonical gripper = 0 (closed)
GRIPPER_MAX_DEG = 100.0  # hinge angle at canonical gripper = 100 (open)


def canonical_to_sim(a: np.ndarray) -> np.ndarray:
    """(..., 6) canonical degrees/percent -> (..., 6) joint radians."""
    a = np.asarray(a, dtype=np.float64)
    if a.shape[-1] != 6:
        raise ValueError(f"expected 6 values, got shape {a.shape}")
    out = np.empty_like(a)
    out[..., :N_BODY] = np.deg2rad(a[..., :N_BODY])
    g = a[..., N_BODY] / 100.0
    out[..., N_BODY] = np.deg2rad(GRIPPER_MIN_DEG + g * (GRIPPER_MAX_DEG - GRIPPER_MIN_DEG))
    return out


def sim_to_canonical(q: np.ndarray) -> np.ndarray:
    """(..., 6) joint radians -> (..., 6) canonical degrees/percent."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1] != 6:
        raise ValueError(f"expected 6 values, got shape {q.shape}")
    out = np.empty_like(q)
    out[..., :N_BODY] = np.rad2deg(q[..., :N_BODY])
    g_deg = np.rad2deg(q[..., N_BODY])
    out[..., N_BODY] = (g_deg - GRIPPER_MIN_DEG) / (GRIPPER_MAX_DEG - GRIPPER_MIN_DEG) * 100.0
    return out


def clip_canonical(a: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(a, dtype=np.float64), low, high)
