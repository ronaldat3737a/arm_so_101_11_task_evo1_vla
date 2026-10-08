import numpy as np
import pytest

from src.simulation.action_adapter import (
    FEATURE_NAMES,
    canonical_to_sim,
    sim_to_canonical,
)


def test_feature_order_matches_lerobot_so_follower():
    assert FEATURE_NAMES == ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]


def test_round_trip_random():
    rng = np.random.default_rng(0)
    a = np.concatenate([rng.uniform(-120, 120, (1000, 5)), rng.uniform(0, 100, (1000, 1))], axis=1)
    np.testing.assert_allclose(sim_to_canonical(canonical_to_sim(a)), a, atol=1e-9)


def test_body_joints_are_plain_degrees():
    q = canonical_to_sim(np.array([90.0, -45.0, 30.0, 0.0, 180.0, 0.0]))
    np.testing.assert_allclose(q[:5], np.deg2rad([90, -45, 30, 0, 180]))


def test_gripper_endpoints_map_to_hinge_limits():
    q = canonical_to_sim(np.array([[0, 0, 0, 0, 0, 0.0], [0, 0, 0, 0, 0, 100.0]]))
    np.testing.assert_allclose(q[:, 5], np.deg2rad([-10.0, 100.0]))


def test_not_the_m100_100_converter():
    # LeIsaac's converter maps motor 100 onto the USD limit (shoulder_pan 110 deg). Ours keeps degrees.
    assert np.rad2deg(canonical_to_sim(np.array([100.0, 0, 0, 0, 0, 0]))[0]) == pytest.approx(100.0)


def test_shape_validation():
    with pytest.raises(ValueError):
        canonical_to_sim(np.zeros(5))
