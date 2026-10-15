"""DART recovery demos: perturbed executed commands, corrective recorded labels, task still solved; off by default."""

import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from src.simulation.env import SO101Env  # noqa: E402
from src.simulation.oracle import MAX_JOINT_SPEED_DEG, Oracle  # noqa: E402

OPTS = {"dart_prob": 1.0, "dart_dxy": (0.02, 0.04), "dart_dyaw": 15.0, "dart_steps": (15, 25)}


def _run(opts):
    env = SO101Env("task_01", split="train", profile="fixed", render=False)
    env.reset(seed=0)
    env.oracle_opts = opts
    env.oracle_rng = np.random.default_rng(123) if opts else None
    executed, labels = [], []
    step = env.step

    def spy(a):
        executed.append(np.array(a, dtype=float))
        return step(a)

    env.step = spy
    orc = Oracle(env)
    ok, _, _ = orc.run(env.task.oracle_plan(env), obs_cb=lambda o, a, i: labels.append(np.array(a, dtype=float)))
    env.close()
    return ok, orc.perturbations, np.array(executed), np.array(labels)


def test_dart_perturbation_labels_and_success():
    ok, pert, ex, lab = _run(OPTS)
    assert ok
    assert len(pert) == 1 and 2.0 <= pert[0]["dxy_cm"] <= 4.0 and abs(pert[0]["dyaw_deg"]) <= 15.0
    assert len(ex) == len(lab)
    diff = np.abs(ex - lab).max(axis=1)
    n = pert[0]["steps"]
    assert (diff > 1e-6).sum() >= n // 2  # labels differ from the executed (perturbed) commands during the perturbation
    lim = MAX_JOINT_SPEED_DEG / 30 + 1e-6
    assert np.all(np.abs(ex - lab)[:, :5] <= lim)  # a label is one rate-limited corrective step from the executed command


def test_default_oracle_unchanged():
    ok, pert, ex, lab = _run(None)
    assert ok and pert == []
    np.testing.assert_allclose(ex, lab)  # without options the recorded action is the executed command
