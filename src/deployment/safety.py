"""Safety layer between the policy and the SO-101 follower. Pure numpy + MuJoCo FK; no hardware access here.

Order of checks for every action (canonical: degrees for the 5 body joints, 0..100 for the gripper):
  1. finite check (non-finite -> emergency stop)
  2. joint box: intersection of the calibrated motor range (from the LeRobot calibration file) and the training-data
     envelope widened by a margin. The calibration range is the hard limit; the data envelope keeps the arm in poses it
     was trained on. Values are clipped, never bypassed.
  3. rate limit: |target - current| per control step <= max_joint_delta * speed_scale (gripper separately)
  4. workspace: FK of the clipped target (official SO-101 MJCF). If the gripper point leaves the workspace box, the
     command is rejected and the arm holds its current position.
Additional runtime guards (timeouts, e-stop) live in `Watchdog` / `EStop`.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
STS3215_RES = 4096


def calibration_limits(calib_file: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Canonical limits from a LeRobot calibration json (`MotorCalibration` fields per motor).

    Body joints (MotorNormMode.DEGREES): deg = (raw - mid) * 360 / (res - 1), mid = (range_min + range_max) / 2.
    Gripper (RANGE_0_100): 0..100 by definition. Mirrors lerobot.motors.motors_bus._normalize.
    """
    cal = json.loads(Path(calib_file).read_text())
    lo, hi = np.zeros(6), np.zeros(6)
    for i, j in enumerate(JOINTS[:5]):
        c = cal[j]
        mid = (c["range_min"] + c["range_max"]) / 2
        a = (c["range_min"] - mid) * 360 / (STS3215_RES - 1)
        b = (c["range_max"] - mid) * 360 / (STS3215_RES - 1)
        lo[i], hi[i] = min(a, b), max(a, b)
    lo[5], hi[5] = 0.0, 100.0
    return lo, hi


def data_envelope(norm_stats_json: str | Path, margin_deg: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    st = json.loads(Path(norm_stats_json).read_text())["train_stats"]["action"]
    lo = np.asarray(st["min"], dtype=np.float64) - margin_deg
    hi = np.asarray(st["max"], dtype=np.float64) + margin_deg
    lo[5], hi[5] = max(lo[5], 0.0), min(hi[5], 100.0)
    return lo, hi


@dataclass
class SafetyConfig:
    joint_low: np.ndarray
    joint_high: np.ndarray
    max_joint_delta_deg: float = 5.0  # per control step, before speed scaling
    max_gripper_delta: float = 10.0  # canonical units per step
    speed_scale: float = 0.1
    workspace_low: np.ndarray = field(default_factory=lambda: np.array([-0.05, -0.35, -0.005]))
    workspace_high: np.ndarray = field(default_factory=lambda: np.array([0.40, 0.35, 0.40]))


class WorkspaceChecker:
    """FK of the gripper point with the official SO-101 MJCF (same model as the simulator)."""

    def __init__(self):
        import mujoco

        from src.simulation.action_adapter import canonical_to_sim
        from src.simulation.scene import ROBOT_XML

        self._mj = mujoco
        self._to_sim = canonical_to_sim
        self.m = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
        self.d = mujoco.MjData(self.m)
        self.site = self.m.site("gripperframe").id

    def tip(self, a: np.ndarray) -> np.ndarray:
        self.d.qpos[:6] = self._to_sim(a)
        self._mj.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy()


class SafetyLayer:
    def __init__(self, cfg: SafetyConfig, workspace: WorkspaceChecker | None = None):
        if not 0 < cfg.speed_scale <= 1.0:
            raise ValueError("speed_scale must be in (0, 1]")
        self.cfg = cfg
        self.ws = workspace
        self.events: list[dict] = []

    def filter(self, target: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, dict]:
        """Return the command to send and a report of the interventions."""
        c = self.cfg
        target = np.asarray(target, dtype=np.float64).reshape(6)
        current = np.asarray(current, dtype=np.float64).reshape(6)
        rep = {"clipped_joints": [], "rate_limited": [], "workspace_reject": False, "estop": None}
        if not np.all(np.isfinite(target)):
            rep["estop"] = "non-finite action"
            return current.copy(), rep
        clipped = np.clip(target, c.joint_low, c.joint_high)
        rep["clipped_joints"] = [JOINTS[i] for i in np.nonzero(np.abs(clipped - target) > 1e-9)[0]]
        step = np.full(6, c.max_joint_delta_deg * c.speed_scale)
        step[5] = c.max_gripper_delta * c.speed_scale
        delta = clipped - current
        limited = current + np.clip(delta, -step, step)
        rep["rate_limited"] = [JOINTS[i] for i in np.nonzero(np.abs(delta) > step + 1e-9)[0]]
        if self.ws is not None:
            tip = self.ws.tip(limited)
            if np.any(tip < c.workspace_low) or np.any(tip > c.workspace_high):
                rep["workspace_reject"] = True
                rep["tip"] = tip.round(3).tolist()
                limited = current.copy()
        if rep["clipped_joints"] or rep["rate_limited"] or rep["workspace_reject"]:
            self.events.append(rep)
        return limited, rep


class EStop:
    """Emergency stop: Ctrl+C (SIGINT), a STOP file, or the space / esc key (pynput, if a display is available)."""

    def __init__(self, stop_file: str | Path | None = None, keyboard: bool = True):
        self._ev = threading.Event()
        self.reason: str | None = None
        self.stop_file = Path(stop_file) if stop_file else None
        signal.signal(signal.SIGINT, lambda *_: self.trigger("SIGINT"))
        self._listener = None
        if keyboard and os.environ.get("DISPLAY"):
            try:
                from pynput import keyboard as kb

                def on_press(key):
                    if key in (kb.Key.space, kb.Key.esc):
                        self.trigger(f"key {key}")

                self._listener = kb.Listener(on_press=on_press)
                self._listener.start()
            except Exception:  # noqa: BLE001 - keyboard e-stop optional; SIGINT and STOP file always work
                self._listener = None

    def trigger(self, reason: str) -> None:
        if not self._ev.is_set():
            self.reason = reason
            self._ev.set()

    @property
    def active(self) -> bool:
        if self.stop_file is not None and self.stop_file.exists():
            self.trigger(f"stop file {self.stop_file}")
        return self._ev.is_set()


class Watchdog:
    """Times a block; raises TimeoutError if it took longer than `limit_s` (inference / bus / camera)."""

    def __init__(self, name: str, limit_s: float):
        self.name, self.limit_s = name, limit_s
        self.last_s = 0.0

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, exc_type, *_):
        self.last_s = time.perf_counter() - self._t
        if exc_type is None and self.last_s > self.limit_s:
            raise TimeoutError(f"{self.name} took {self.last_s * 1000:.0f} ms > {self.limit_s * 1000:.0f} ms")
