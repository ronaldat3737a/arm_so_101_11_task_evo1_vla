"""MuJoCo simulation tests (no real hardware). Rendering is only enabled where image shapes are checked."""

import math
import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

from src.simulation.action_adapter import canonical_to_sim  # noqa: E402
from src.simulation.env import SO101Env  # noqa: E402
from src.simulation.placement import SplitGrid, sample_xy  # noqa: E402
from src.simulation.tasks import INSTRUCTIONS, TASKS  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


@pytest.fixture(scope="module")
def env():
    e = SO101Env("task_01", split="train", profile="position", render=False)
    yield e
    e.close()


def test_model_joint_and_actuator_order(env):
    m = env.model
    assert [m.joint(i).name for i in range(6)] == JOINTS
    assert [m.actuator(i).name for i in range(6)] == JOINTS


def test_reset_is_deterministic_per_seed(env):
    _, a = env.reset(seed=3)
    s1 = env.data.qpos.copy()
    _, b = env.reset(seed=3)
    assert a["layout"] == b["layout"]
    np.testing.assert_allclose(env.data.qpos, s1)
    _, c = env.reset(seed=4)
    assert c["layout"] != a["layout"]


def test_randomized_target_objects_respect_split():
    for split in ("train", "val", "test"):
        e = SO101Env("task_01", split=split, profile="position", render=False)
        for seed in range(5):
            _, info = e.reset(seed=seed)
            assert info["split_of_layout"]["red_cube"] == split
            assert info["split_of_layout"]["pot"] == split
        e.close()


def test_split_cells_are_disjoint_and_cover_workspace():
    import yaml

    cfg = yaml.safe_load(open("configs/simulation.yaml"))
    g = SplitGrid.from_config(cfg)
    rng = np.random.default_rng(0)
    seen = {"train": 0, "val": 0, "test": 0}
    for _ in range(3000):
        r, th = rng.uniform(g.r_min, g.r_max), math.radians(rng.uniform(g.theta_min_deg, g.theta_max_deg))
        seen[g.split_of(r * math.cos(th), r * math.sin(th))] += 1
    assert all(v > 0 for v in seen.values())
    assert 0.5 < seen["train"] / 3000 < 0.7  # 3 of 5 residues
    x, y = sample_xy(rng, g, "test", [], 0.02, 0.01)
    assert g.split_of(x, y) == "test" and not g.in_keepout(x, y, 0.02)


def test_observation_matches_dataset_schema():
    e = SO101Env("task_01", split="train", profile="fixed", render=True)
    obs, _ = e.reset(seed=0)
    assert obs["observation.state"].shape == (6,) and obs["observation.state"].dtype == np.float32
    assert obs["observation.images.top"].shape == (360, 640, 3) and obs["observation.images.top"].dtype == np.uint8
    assert obs["observation.images.wrist"].shape == (480, 640, 3)
    assert obs["task"] == INSTRUCTIONS["task_01"] == "Pick up the red cube and place it into the steel pot."
    assert obs["observation.images.top"].std() > 5 and obs["observation.images.wrist"].std() > 5
    e.close()


def test_action_drives_joints_to_canonical_targets(env):
    env.reset(seed=0)
    target = np.array([10.0, -20.0, 40.0, 50.0, 15.0, 40.0])  # canonical degrees / gripper %
    for _ in range(45):
        obs, *_ = env.step(target)
    np.testing.assert_allclose(obs["observation.state"][[0, 4]], target[[0, 4]], atol=1.0)
    np.testing.assert_allclose(obs["observation.state"][1:4], target[1:4], atol=3.5)  # gravity sag (shoulder saturates)
    assert abs(obs["observation.state"][5] - target[5]) < 3.0
    np.testing.assert_allclose(env.data.ctrl[:6], canonical_to_sim(target), atol=1e-9)


def test_non_finite_action_rejected(env):
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step([np.nan, 0, 0, 0, 0, 0])


def test_success_detector_positive_and_negative(env):
    env.reset(seed=0)
    ok, why = env.task.success(env)
    assert not ok and "not above pot" in why
    # teleport the cube into the pot and let it settle
    pot = env.body_pos("pot")
    j = env.model.joint("red_cube_free")
    env.data.qpos[j.qposadr[0] : j.qposadr[0] + 7] = [pot[0], pot[1], 0.03, 1, 0, 0, 0]
    env.data.qvel[j.dofadr[0] : j.dofadr[0] + 6] = 0
    hold = env.start_pose
    success = False
    for _ in range(40):
        _, r, term, _, info = env.step(hold)
        if term:
            success = info["success"]
            break
    assert success and r == 1.0


def test_failure_reason_on_object_off_table(env):
    env.reset(seed=0)
    j = env.model.joint("red_cube_free")
    env.data.qpos[j.qposadr[0] : j.qposadr[0] + 3] = [0.25, -0.15, -0.2]
    _, _, term, _, info = env.step(env.start_pose)
    assert term and info["failure_reason"] == "red_cube_fell_off_table"


def test_timeout_reports_reason():
    e = SO101Env("task_01", split="train", profile="fixed", render=False)
    e.reset(seed=0)
    e.max_steps = 5
    for _ in range(5):
        _, _, term, trunc, info = e.step(e.start_pose)
    assert trunc and not term and info["failure_reason"].startswith("timeout")
    e.close()


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_every_registered_task_resets(task_id):
    e = SO101Env(task_id, split="test", profile="position", render=False)
    obs, info = e.reset(seed=1)
    assert obs["task"] == INSTRUCTIONS[task_id]
    ok, _ = e.task.success(e)
    assert not ok, "task must not be solved at reset"
    assert e.task.failure(e) is None
    e.close()
