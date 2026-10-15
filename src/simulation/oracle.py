"""Scripted oracle controller (privileged object poses + IK). Used to prove each task is solvable in sim and to
generate synthetic demonstrations later. It emits the SAME canonical actions (degrees, gripper 0..100) as the policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .action_adapter import canonical_to_sim, sim_to_canonical
from .ik import GripperIK

OPEN = 45.0  # canonical gripper opening used for grasping (0 = closed, 100 = fully open)
CLOSED = 0.0
RELEASE = None  # release = open by RELEASE_DELTA beyond the current (object-dependent) grasp opening
RELEASE_DELTA = 15.0  # canonical units; a wide-opening moving jaw swings ~7 cm sideways and hits container walls
KEEP = "keep"  # keep the current gripper command (retreat after a release without opening further)
MAX_JOINT_SPEED_DEG = 90.0  # deg/s per joint during scripted motion (grasps are firm since the finger pads)


@dataclass
class Waypoint:
    pos: tuple | None  # gripper tip target (world); None = keep arm
    grip: float | str | None  # canonical 0..100; None = release (current opening + RELEASE_DELTA); KEEP = unchanged
    yaw: float = 0.0
    hold_steps: int = 0
    joints: np.ndarray | None = None  # explicit canonical joint target (overrides pos)
    correct: bool = False  # closed-loop correction of the grasp point after reaching the waypoint (free space only)
    cartesian: bool = False  # move the grasp point along a straight line (IK every <= 1 cm) instead of joint interpolation
    lowerable: bool = False  # IK may lower this target by up to 4 cm if unreachable (transport waypoints only)
    perturbable: bool = False  # approach waypoint: may receive a DART perturbation (only when env.oracle_opts enables it)


def _polar_lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Interpolate radius, azimuth (around the robot base) and height: horizontal moves follow an arc around the base
    instead of a straight chord that can pass over the shoulder (unreachable)."""
    ra, rb = math.hypot(a[0], a[1]), math.hypot(b[0], b[1])
    ta, tb = math.atan2(a[1], a[0]), math.atan2(b[1], b[0])
    r = ra + (rb - ra) * t
    th = ta + (tb - ta) * t
    return np.array([r * math.cos(th), r * math.sin(th), a[2] + (b[2] - a[2]) * t])


class Oracle:
    def __init__(self, env):
        self.env = env
        self.ik = GripperIK(env.model)
        # optional data-generation options (scripts/generate_sim_data.py sets env.oracle_opts / env.oracle_rng per episode);
        # absent -> the original deterministic oracle
        self.opts = getattr(env, "oracle_opts", None) or {}
        self.prng = getattr(env, "oracle_rng", None)
        self.perturbations: list[dict] = []
        # World-frame offset added to IK targets. The official model's shoulder actuator saturates (forcerange 3.35 N m)
        # and sags ~2 deg under gravity on an extended arm, so the oracle closes the loop on the measured grasp point.
        self.bias = np.zeros(3)

    def grasp_point(self) -> np.ndarray:
        s = self.env.data.site("gripperframe")
        return s.xpos + s.xmat.reshape(3, 3) @ self.ik.offset

    def _correct(self, wp: Waypoint, cur: np.ndarray, q_seed: np.ndarray, obs_cb, iters: int = 4, steps: int = 8, tol: float = 0.003):
        env = self.env
        info = {}
        for _ in range(iters):
            err = np.asarray(wp.pos) - self.grasp_point()
            if np.linalg.norm(err) < tol:
                break
            self.bias = np.clip(self.bias + 0.8 * err, -0.03, 0.03)
            tgt = self._target_joints(wp, q_seed)
            for k in range(steps):
                a = cur + (tgt - cur) * min(1.0, (k + 1) / 3)
                obs, _, term, trunc, info = env.step(a)
                if obs_cb:
                    obs_cb(obs, a, info)
                if term or trunc:
                    return tgt, info, True
            cur = tgt
        return cur, info, False

    def _grip(self, wp: Waypoint) -> float:
        if wp.grip is None:  # relative release
            return float(min(100.0, self.env.joint_canonical()[5] + RELEASE_DELTA))
        if wp.grip == KEEP:
            return float(self._last_grip)
        return wp.grip

    def _target_joints(self, wp: Waypoint, q_now_rad: np.ndarray) -> np.ndarray:
        if wp.joints is not None:
            a = np.array(wp.joints, dtype=np.float64)
            a[5] = self._grip(wp)
            return a
        q, err = self.ik.solve(np.asarray(wp.pos) + self.bias, q_now_rad, yaw=wp.yaw)
        if err > 0.006 and wp.lowerable:
            # transport heights are not semantically important: step down until the pose is reachable
            for dz in (0.01, 0.02, 0.03, 0.04):
                p = np.asarray(wp.pos) + self.bias - np.array([0, 0, dz])
                q, err = self.ik.solve(p, q_now_rad, yaw=wp.yaw)
                if err <= 0.006:
                    break
        if err > 0.01:
            raise RuntimeError(f"IK failed for {wp.pos}: err {err:.3f}")
        a = sim_to_canonical(q)
        a[5] = self._grip(wp)
        return a

    def _perturb(self, wp: Waypoint, cur: np.ndarray, q_seed: np.ndarray, obs_cb):
        """DART-style perturbation during an approach: the EXECUTED command is pushed 2-4 cm sideways (random direction)
        and +-yaw for a few steps, while the RECORDED action is the expert's correction from the executed command toward
        the true approach target (rate limited). Teaches recovery without teaching the drift itself."""
        env, o, rng = self.env, self.opts, self.prng
        dxy = float(rng.uniform(*o["dart_dxy"]))
        ang = float(rng.uniform(0, 2 * math.pi))
        dyaw = float(rng.uniform(-o["dart_dyaw"], o["dart_dyaw"]))
        steps = int(rng.integers(o["dart_steps"][0], o["dart_steps"][1] + 1))
        gp = self.grasp_point()
        p_pert = (gp[0] + dxy * math.cos(ang), gp[1] + dxy * math.sin(ang), gp[2])
        tgt_p = self._target_joints(Waypoint(p_pert, wp.grip if wp.grip not in (None, KEEP) else self._last_grip, wp.yaw + math.radians(dyaw), lowerable=True), q_seed)
        tgt_clean = self._target_joints(Waypoint(wp.pos, wp.grip if wp.grip not in (None, KEEP) else self._last_grip, wp.yaw, lowerable=wp.lowerable), q_seed)
        dt = 1.0 / env.fps
        lim = np.array([MAX_JOINT_SPEED_DEG * dt] * 5 + [6.0])
        delta = tgt_p - cur
        steps = max(steps, int(math.ceil(np.abs(delta[:5]).max() / (MAX_JOINT_SPEED_DEG * dt))))
        info, a = {}, cur
        for k in range(1, steps + 1):
            a = cur + delta * (k / steps)
            label = a + np.clip(tgt_clean - a, -lim, lim)  # what the expert would command from here
            obs, _, term, trunc, info = env.step(a)
            if obs_cb:
                obs_cb(obs, label, info)
            if term or trunc:
                return a, canonical_to_sim(tgt_p), info, True
        self.perturbations.append({"dxy_cm": round(dxy * 100, 2), "dir_deg": round(math.degrees(ang), 1), "dyaw_deg": round(dyaw, 1),
                                   "steps": steps, "t_start": env.t - steps})
        return a, canonical_to_sim(tgt_p), info, False

    def run(self, waypoints: list[Waypoint], obs_cb=None) -> tuple[bool, dict, int]:
        """Execute waypoints; returns (success, last info, steps)."""
        env = self.env
        cur = env.joint_canonical()
        cur[5] = env.last_action[5]
        info = {}
        dt = 1.0 / env.fps
        q_seed = env.data.qpos[:6].copy()
        self._last_grip = cur[5]
        def cart_segments(wp):
            start = self.grasp_point()  # measured position
            end = np.asarray(wp.pos)
            n_seg = max(1, int(math.ceil(np.linalg.norm(end - start) / 0.01)))
            g = self._grip(wp)
            return [Waypoint(tuple(_polar_lerp(start, end, i / n_seg)), g, wp.yaw, lowerable=wp.lowerable) for i in range(1, n_seg + 1)]

        for wp in waypoints:
            segments = [wp]
            if wp.cartesian and wp.pos is not None:
                segments = cart_segments(wp)
            k_pert = None
            if wp.perturbable and wp.cartesian and self.prng is not None and self.opts.get("dart_prob", 0) > 0 and len(segments) >= 4 \
                    and self.prng.random() < self.opts["dart_prob"]:
                k_pert = int(len(segments) * self.prng.uniform(0.3, 0.7))
            si = 0
            while si < len(segments):
                seg = segments[si]
                if k_pert is not None and si == k_pert:
                    k_pert = None
                    cur, q_seed, info_p, done = self._perturb(wp, cur, q_seed, obs_cb)
                    info = info_p or info
                    if done:
                        return info["success"], info, env.t
                    segments, si = cart_segments(wp), 0  # recover: re-plan from the measured (perturbed) grasp point
                    continue
                si += 1
                tgt = self._target_joints(seg, q_seed)
                q_seed = canonical_to_sim(tgt)
                delta = tgt - cur
                n = max(1, int(math.ceil(np.abs(delta[:5]).max() / (MAX_JOINT_SPEED_DEG * dt))), int(abs(delta[5]) / 6) + 1)  # noqa: E501
                for k in range(1, n + 1):
                    a = cur + delta * min(1.0, k / n)
                    obs, _, term, trunc, info = env.step(a)
                    if obs_cb:
                        obs_cb(obs, a, info)
                    if term or trunc:
                        return info["success"], info, env.t
                cur = tgt
                self._last_grip = tgt[5]
            if wp.pos is not None and wp.joints is None and wp.correct:
                cur, info_c, done = self._correct(wp, cur, q_seed, obs_cb)
                info = info_c or info
                if done:
                    return info["success"], info, env.t
            for _ in range(wp.hold_steps):
                obs, _, term, trunc, info = env.step(cur)
                if obs_cb:
                    obs_cb(obs, cur, info)
                if term or trunc:
                    return info["success"], info, env.t
        # let things settle and give the success detector time to confirm
        for _ in range(env.hold_steps + 15):
            obs, _, term, trunc, info = env.step(cur)
            if obs_cb:
                obs_cb(obs, cur, info)
            if term or trunc:
                break
        return info.get("success", False), info, env.t


# Heights chosen from the reachability table of the official model (wrist_flex is limited to +-95 deg): the gripper
# can stay vertical up to z ~0.105 for r in [0.18, 0.24] m and tilts 20-30 deg at z = 0.13.
PRE_GRASP_HEIGHT = 0.07  # above the object centre
TRANSPORT_Z = 0.13  # absolute grasp-point height while carrying


def pick_place_waypoints(
    env,
    obj: str,
    obj_half_h: float,
    place_xyz,
    yaw_obj_deg: float = 0.0,
    release_drop: float = 0.0,
    grasp_depth: float = 0.0,
    transport_z: float = TRANSPORT_Z,
    grasp_point=None,
    place_yaw_deg: float | None = None,
    pre_grasp_height: float = PRE_GRASP_HEIGHT,
    open_pct: float = OPEN,
    half_width: float = 0.0125,
    pre_grasp_hold: int = 0,
) -> list[Waypoint]:
    """Pick `obj` and release it with the grasp point at `place_xyz` (+ release_drop).

    grasp_point: explicit world grasp point (default: object centre + grasp_depth, e.g. a lid knob or a block's top).
    place_yaw_deg: jaw orientation at release (default: keep the grasp yaw). Matters for stacking aligned blocks.
    open_pct: jaw opening before grasping; use a small opening (~25) to reach into containers.
    half_width: object half-width along the jaw closing axis. The IK grasp point centres a 25 mm object; wider objects
        are centred by shifting the target along the closing axis (moving-jaw side = +closing direction, measured).
    pre_grasp_hold: steps to hover at the (closed-loop corrected) pre-grasp pose before descending (hover & align).
    The approach waypoint is marked perturbable (DART recovery demos when enabled in env.oracle_opts).
    """
    p = np.asarray(grasp_point, dtype=np.float64) if grasp_point is not None else env.body_pos(obj) + np.array([0, 0, grasp_depth])
    yaw = math.radians(yaw_obj_deg)
    pyaw = yaw if place_yaw_deg is None else math.radians(place_yaw_deg)
    shift = half_width - 0.0125
    p = p - shift * np.array([math.cos(yaw), math.sin(yaw), 0.0])
    place_xyz = np.asarray(place_xyz, dtype=np.float64) - shift * np.array([math.cos(pyaw), math.sin(pyaw), 0.0])
    grasp_z = p[2]
    px, py, pz = place_xyz
    tz = max(transport_z, pz + 0.02, grasp_z + 0.03)
    return [
        Waypoint((p[0], p[1], grasp_z + pre_grasp_height), open_pct, yaw, hold_steps=pre_grasp_hold, correct=True, lowerable=True,
                 cartesian=True, perturbable=True),
        Waypoint((p[0], p[1], grasp_z), open_pct, yaw, hold_steps=5, cartesian=True),
        Waypoint((p[0], p[1], grasp_z), CLOSED, yaw, hold_steps=10),
        Waypoint((p[0], p[1], tz), CLOSED, yaw, cartesian=True, lowerable=True),
        Waypoint((px, py, tz), CLOSED, pyaw, cartesian=True, lowerable=True),
        Waypoint((px, py, pz + release_drop), CLOSED, pyaw, hold_steps=5, cartesian=True),
        Waypoint((px, py, pz + release_drop), RELEASE, pyaw, hold_steps=10),
        Waypoint((px, py, tz), KEEP, pyaw, cartesian=True, lowerable=True),
    ]
