"""Safety layer tests: no motors are touched."""

import json

import numpy as np
import pytest

from src.deployment.safety import JOINTS, SafetyConfig, SafetyLayer, Watchdog, WorkspaceChecker, calibration_limits

LO = np.array([-100.0, -100, -95, -95, -150, 0])
HI = np.array([100.0, 100, 95, 95, 150, 100])


def layer(**kw):
    return SafetyLayer(SafetyConfig(joint_low=LO, joint_high=HI, **kw))


def test_clip_to_joint_limits_and_never_bypass():
    s = layer(speed_scale=1.0, max_joint_delta_deg=1000, max_gripper_delta=1000)
    out, rep = s.filter(np.array([500.0, -500, 0, 0, 0, 150]), np.zeros(6))
    assert np.all(out <= HI) and np.all(out >= LO)
    assert set(rep["clipped_joints"]) == {"shoulder_pan", "shoulder_lift", "gripper"}


def test_rate_limit_scales_with_speed():
    s = layer(speed_scale=0.1, max_joint_delta_deg=5.0, max_gripper_delta=10.0)
    out, rep = s.filter(np.array([50.0, 0, 0, 0, 0, 80]), np.zeros(6))
    assert out[0] == pytest.approx(0.5) and out[5] == pytest.approx(1.0)
    assert "shoulder_pan" in rep["rate_limited"]


def test_non_finite_action_holds_position_and_requests_estop():
    s = layer()
    cur = np.array([1.0, 2, 3, 4, 5, 6])
    out, rep = s.filter(np.array([np.nan, 0, 0, 0, 0, 0]), cur)
    np.testing.assert_array_equal(out, cur)
    assert rep["estop"] == "non-finite action"


def test_speed_scale_validated():
    with pytest.raises(ValueError):
        layer(speed_scale=1.5)


def test_workspace_rejects_target_below_table():
    ws = WorkspaceChecker()
    s = SafetyLayer(SafetyConfig(joint_low=LO, joint_high=HI, speed_scale=1.0, max_joint_delta_deg=1000, max_gripper_delta=1000), ws)
    cur = np.array([0.0, -15, 35, 65, 0, 50])  # gripper vertical above the table
    assert ws.tip(cur)[2] > 0
    down = np.array([0.0, 40, 30, 65, 0, 50])  # drives the tip through the table
    assert ws.tip(down)[2] < -0.005
    out, rep = s.filter(down, cur)
    assert rep["workspace_reject"] and np.allclose(out, cur)


def test_calibration_limits_follow_lerobot_degrees_formula(tmp_path):
    cal = {j: {"id": i + 1, "drive_mode": 0, "homing_offset": 0, "range_min": 1000, "range_max": 3000} for i, j in enumerate(JOINTS)}
    cal["wrist_roll"].update(range_min=0, range_max=4095)
    f = tmp_path / "arm.json"
    f.write_text(json.dumps(cal))
    lo, hi = calibration_limits(f)
    assert hi[0] == pytest.approx(1000 * 360 / 4095) and lo[0] == pytest.approx(-1000 * 360 / 4095)
    assert hi[4] == pytest.approx(180.0, abs=0.05)
    assert (lo[5], hi[5]) == (0.0, 100.0)


def test_watchdog_raises_on_timeout():
    import time

    with pytest.raises(TimeoutError):
        with Watchdog("x", 0.01):
            time.sleep(0.03)
