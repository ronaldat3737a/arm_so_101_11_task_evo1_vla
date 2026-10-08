"""Damped-least-squares IK for the SO-101 gripper (used only by the scripted oracle, never by the policy).

Target: the GRASP POINT (a fixed offset from the gripperframe site, between the fingertips) + approach axis
(site +x) pointing straight down. Measured on the official meshes: the jaws close along the site z axis, the fixed
fingertip is at site z = -0.003, the moving fingertip at z = +0.005 (closed) .. +0.088 (60 % open). A 25 mm cube is
centred at site z=+0.017 (4.5 mm clearance to the fixed pad for tracking error); x=-0.020 puts ~2 cm of finger around it. Solved over
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex. wrist_roll is set analytically from the desired grasp yaw.
Works on a private MjData so the simulation state is untouched.
"""

from __future__ import annotations

import mujoco
import numpy as np

ARM = [0, 1, 2, 3]  # qpos / dof indices of pan, lift, elbow, wrist_flex (robot joints come first in the model)
ROLL = 4


class GripperIK:
    GRASP_OFFSET = np.array([-0.020, 0.0, 0.017])

    def __init__(self, model: mujoco.MjModel, site: str = "gripperframe", offset=None):
        self.offset = self.GRASP_OFFSET if offset is None else np.asarray(offset, dtype=np.float64)
        self.m = model
        self.d = mujoco.MjData(model)
        self.site = model.site(site).id
        self.body = model.site_bodyid[self.site]
        self.lo = model.jnt_range[:6, 0].copy()
        self.hi = model.jnt_range[:6, 1].copy()

    # nominal "reach forward, gripper down" posture (canonical deg [0,-15,35,65]) used as a fallback seed
    REACH_SEED = np.deg2rad([0.0, -15.0, 35.0, 65.0, 0.0, 0.0])

    def solve(self, target_pos, q_init, yaw: float = 0.0, iters: int = 300, tol: float = 1e-3) -> tuple[np.ndarray, float]:
        # The approach-axis constraint is soft: wrist_flex is limited to +-95 deg, so the gripper cannot stay vertical
        # above ~8 cm. Try decreasing orientation weights; position accuracy has priority.
        best = None
        for w_rot in (0.3, 0.1, 0.03, 0.0):
            for seed in (np.asarray(q_init[:6], dtype=np.float64), self._aim(self.REACH_SEED, target_pos)):
                q, e = self._solve(target_pos, seed, yaw, iters, tol, w_rot)
                if best is None or e < best[1] - 1e-4:
                    best = (q, e)
                if e < tol:
                    return q, e
        return best

    @staticmethod
    def _aim(seed, target_pos):
        s = seed.copy()
        # measured on the official model: positive shoulder_pan turns the arm toward -y
        s[0] = -np.arctan2(target_pos[1], target_pos[0])
        return s

    def _solve(self, target_pos, q_init, yaw, iters, tol, w_rot=0.3):
        m, d = self.m, self.d
        q = np.array(q_init[:6], dtype=np.float64)
        d.qpos[:] = 0
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        down = np.array([0.0, 0.0, -1.0])
        err_norm = np.inf
        for _ in range(iters):
            d.qpos[:6] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            R = d.site_xmat[self.site].reshape(3, 3)
            x = d.site_xpos[self.site] + R @ self.offset
            e_pos = np.asarray(target_pos) - x
            e_rot = np.cross(R[:, 0], down)  # rotate approach axis onto down
            err = np.concatenate([e_pos, w_rot * e_rot])
            err_norm = float(np.linalg.norm(e_pos))
            if err_norm < tol and (w_rot == 0.0 or np.linalg.norm(e_rot) < 2e-2):
                break
            mujoco.mj_jac(m, d, jp, jr, x, self.body)
            J = np.vstack([jp[:, ARM], w_rot * jr[:, ARM]])
            lam = 1e-3
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), err)
            q[ARM] = np.clip(q[ARM] + np.clip(dq, -0.2, 0.2), self.lo[ARM], self.hi[ARM])
        # wrist roll (measured, gripper vertical): the jaws close along the radial direction at roll = 0; the closing axis
        # angle in the world is (-pan + roll). Align it with the requested yaw (mod pi, the closing axis is symmetric).
        roll = yaw + q[0]
        roll = (roll + np.pi / 2) % np.pi - np.pi / 2
        q[ROLL] = np.clip(roll, self.lo[ROLL], self.hi[ROLL])
        return q, err_norm
