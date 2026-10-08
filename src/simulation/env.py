"""SO-101 MuJoCo environment with the dataset's observation/action interface.

    obs = {"observation.state": float32[6] canonical (degrees, gripper 0..100),
           "observation.images.top": uint8[360, 640, 3], "observation.images.wrist": uint8[480, 640, 3],
           "task": instruction}
    step(action: float[6] canonical absolute joint targets) at 30 Hz

Reset: robot starts in the dataset's median start pose of the task (clamped to the model's joint limits), objects
are placed by the task (profile fixed / position / position_visual, restricted to the split's cells) and settled.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import mujoco
import numpy as np
import yaml

from .action_adapter import canonical_to_sim, sim_to_canonical
from .placement import SplitGrid
from .scene import CameraSpec, build
from .tasks import make_task

ROOT = Path(__file__).resolve().parents[2]
ROBOT_BODIES = ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "wrist_camera_mount", "wrist_camera", "moving_jaw_so101_v1")
GRIPPER_BODIES = ("gripper", "moving_jaw_so101_v1")


def _load_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "simulation.yaml").read_text())


def _start_pose(task_id: str) -> np.ndarray:
    audit = json.loads((ROOT / "reports" / "dataset_audit.json").read_text())
    tasks = yaml.safe_load((ROOT / "configs" / "tasks.yaml").read_text())["tasks"]
    repo = next(t["dataset"] for t in tasks if t["task_id"] == task_id)
    return np.asarray(audit["datasets"][repo]["start_pose"]["median"], dtype=np.float64)


class SO101Env:
    def __init__(self, task_id: str, split: str = "train", profile: str = "position", render: bool = True, cameras: tuple[str, ...] = ("top", "wrist")):
        os.environ.setdefault("MUJOCO_GL", "egl")
        self.cfg = _load_cfg()
        self.task = make_task(task_id)
        self.split = split
        self.profile_name = profile
        self.profile = self.cfg["profiles"][profile]
        self.grid = SplitGrid.from_config(self.cfg)
        self.fps = self.cfg["control"]["fps"]
        cams = []
        for name in cameras:
            c = self.cfg["cameras"][name]
            cams.append(CameraSpec(name, c["width"], c["height"], c["fovy"], tuple(c["pos"]), target=tuple(c["target"]) if "target" in c else None, body=c.get("body"), quat=tuple(c["quat"]) if "quat" in c else None))
        self.camera_specs = cams
        self.model, self.obj_info = build(self.task.objects(), cams, timestep=self.cfg["control"]["physics_timestep"])
        self.board_size = {o.name: tuple(o.size) for o in self.task.objects() if o.kind in ("board", "sorter_box")}
        self.data = mujoco.MjData(self.model)
        self.n_substeps = max(1, round((1.0 / self.fps) / self.model.opt.timestep))
        self.jlo = self.model.jnt_range[:6, 0].copy()
        self.jhi = self.model.jnt_range[:6, 1].copy()
        self.ctrl_lo = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = self.model.actuator_ctrlrange[:, 1].copy()
        self.start_pose = sim_to_canonical(np.clip(canonical_to_sim(_start_pose(task_id)), self.jlo, self.jhi))
        self._robot_body_ids = {self.model.body(b).id for b in ROBOT_BODIES}
        self._gripper_body_ids = {self.model.body(b).id for b in GRIPPER_BODIES}
        self._table_geom = self.model.geom("table").id
        self.render_enabled = render
        self._renderers: dict[str, mujoco.Renderer] = {}
        if render:
            for c in cams:
                self._renderers[c.name] = mujoco.Renderer(self.model, c.height, c.width)
        self.max_steps = int((self.task.max_seconds or self.cfg["episode"]["max_seconds"]) * self.fps)
        self.hold_steps = self.cfg["episode"]["success_hold_steps"]
        self._base_cam = {c.name: (self.model.cam(c.name).pos.copy(), float(self.model.cam(c.name).fovy[0])) for c in cams}
        self._base_light = self.model.light_diffuse.copy()
        self._base_rgba = self.model.geom_rgba.copy()
        self.rng = np.random.default_rng(0)
        self.layout: dict = {}

    # ---------------- state accessors used by tasks ----------------
    def body_pos(self, name: str) -> np.ndarray:
        return self.data.body(name).xpos.copy()

    def body_up_z(self, name: str) -> float:
        return float(self.data.body(name).xmat.reshape(3, 3)[2, 2])

    def body_linvel(self, name: str) -> np.ndarray:
        j = self.model.joint(f"{name}_free")
        return self.data.qvel[j.dofadr[0] : j.dofadr[0] + 3].copy()

    def in_gripper_contact(self, name: str) -> bool:
        bid = self.model.body(name).id
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if (b1 == bid and b2 in self._gripper_body_ids) or (b2 == bid and b1 in self._gripper_body_ids):
                return True
        return False

    def robot_table_contacts(self) -> int:
        n = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g = {c.geom1, c.geom2}
            if self._table_geom in g:
                other = c.geom2 if c.geom1 == self._table_geom else c.geom1
                if self.model.geom_bodyid[other] in self._robot_body_ids and self.model.geom_bodyid[other] != self.model.body("base").id:
                    n += 1
        return n

    def joint_canonical(self) -> np.ndarray:
        return sim_to_canonical(self.data.qpos[:6])

    # ---------------- core API ----------------
    def _set_free(self, name: str, x: float, y: float, yaw_deg: float) -> None:
        info = self.obj_info[name]
        j = self.model.joint(f"{name}_free")
        a = j.qposadr[0]
        z = info["half_height"] + 0.0005
        yaw = math.radians(yaw_deg)
        self.data.qpos[a : a + 7] = [x, y, z, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.data.qvel[j.dofadr[0] : j.dofadr[0] + 6] = 0

    def set_object_pose(self, name: str, x: float, y: float, z: float, yaw_deg: float = 0.0) -> None:
        j = self.model.joint(f"{name}_free")
        a = j.qposadr[0]
        yaw = math.radians(yaw_deg)
        self.data.qpos[a : a + 7] = [x, y, z, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.data.qvel[j.dofadr[0] : j.dofadr[0] + 6] = 0

    def support_top(self, support: str) -> float:
        """Height of the surface an object rests on, for a support placed by the task (board top / container bottom)."""
        info = self.obj_info[support]
        pos = self.layout_pos(support)
        if info["kind"] in ("pot", "sorter_box"):
            return float(info.get("bottom", 0.008))
        if info["kind"] == "hollow_cube":
            return float(pos[2] - info["half_height"] + 0.004)
        return float(pos[2] + info["half_height"])

    def layout_pos(self, name: str) -> np.ndarray:
        j = self.model.joint(f"{name}_free")
        return self.data.qpos[j.qposadr[0] : j.qposadr[0] + 3].copy()

    def place_on(self, name: str, support: str, dx: float = 0.0, dy: float = 0.0, yaw_deg: float = 0.0) -> None:
        sp = self.layout_pos(support)
        self.set_object_pose(name, sp[0] + dx, sp[1] + dy, self.support_top(support) + self.obj_info[name]["half_height"] + 0.0005, yaw_deg)

    def _apply_visual_randomization(self) -> None:
        self.model.light_diffuse[:] = self._base_light
        self.model.geom_rgba[:] = self._base_rgba
        for name, (pos, fovy) in self._base_cam.items():
            self.model.cam(name).pos[:] = pos
            self.model.cam(name).fovy[0] = fovy
        if not self.profile.get("visual"):
            return
        p, r = self.profile, self.rng
        self.model.light_diffuse[:] = self._base_light * r.uniform(*p["light_intensity_scale"])
        t = self._table_geom
        self.model.geom_rgba[t, :3] = np.clip(self._base_rgba[t, :3] + r.normal(0, p["table_color_jitter"], 3), 0, 1)
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] not in self._robot_body_ids and g != t and self.model.geom_bodyid[g] != 0:
                self.model.geom_rgba[g, :3] = np.clip(self._base_rgba[g, :3] + r.normal(0, p["object_color_jitter"], 3), 0, 1)
        for name, (pos, fovy) in self._base_cam.items():
            if name == "top":  # wrist camera is rigidly mounted
                self.model.cam(name).pos[:] = pos + r.normal(0, p["camera_pos_noise"], 3)
                self.model.cam(name).fovy[0] = fovy + r.normal(0, p["camera_fovy_noise"])

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        q0 = canonical_to_sim(self.start_pose)
        self.data.qpos[:6] = q0
        self.data.ctrl[:] = np.clip(q0, self.ctrl_lo, self.ctrl_hi)
        ws = self.cfg["workspace"]
        if self.profile["randomize_positions"]:
            self.layout = self.task.sample_layout(self.rng, self.grid, self.split, self.profile["yaw_range_deg"], ws["min_separation"], self.obj_info)
        else:
            self.layout = self.task.nominal_layout()
        for name, (x, y, yaw) in self.layout.items():
            self._set_free(name, x, y, yaw)
        self._apply_visual_randomization()  # before post_place: tasks may recolour objects (task_08)
        if hasattr(self.task, "post_place"):
            self.task.post_place(self)  # objects resting on / inside other objects
        if hasattr(self.task, "on_reset"):
            self.task.on_reset(self)  # per-episode task state (stage / order tracking)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(int(0.5 / self.model.opt.timestep)):  # settle 0.5 s
            mujoco.mj_step(self.model, self.data)
        self.t = 0
        self._hold = 0
        self.collision_steps = 0
        self.last_action = self.start_pose.copy()
        return self._obs(), self._info(False, None)

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        a = np.asarray(action, dtype=np.float64).reshape(6)
        if not np.all(np.isfinite(a)):
            raise ValueError(f"non-finite action {a}")
        self.last_action = a
        self.data.ctrl[:] = np.clip(canonical_to_sim(a), self.ctrl_lo, self.ctrl_hi)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self.t += 1
        if self.robot_table_contacts() > 0:
            self.collision_steps += 1
        ok, why = self.task.success(self)
        self._hold = self._hold + 1 if ok else 0
        success = self._hold >= self.hold_steps
        fail = self.task.failure(self)
        truncated = (self.t >= self.max_steps) and not success
        reason = None
        if not success:
            if fail:
                reason = fail
            elif truncated:
                reason = f"timeout: {why}"
        terminated = success or fail is not None
        return self._obs(), float(success), terminated, truncated, self._info(success, reason, why)

    def _obs(self) -> dict:
        obs = {"observation.state": self.joint_canonical().astype(np.float32), "task": self.task.instruction}
        if self.render_enabled:
            for name, r in self._renderers.items():
                r.update_scene(self.data, name)
                img = r.render()
                if self.profile.get("visual") and self.profile.get("image_noise_std"):
                    img = np.clip(img.astype(np.float32) + self.rng.normal(0, self.profile["image_noise_std"], img.shape), 0, 255).astype(np.uint8)
                obs[f"observation.images.{name}"] = img
        return obs

    def _info(self, success: bool, reason: str | None, why: str = "") -> dict:
        return {
            "success": success,
            "failure_reason": reason,
            "unmet": why,
            "t": getattr(self, "t", 0),
            "collision_steps": getattr(self, "collision_steps", 0),
            "layout": {k: [round(float(v), 4) for v in p] for k, p in self.layout.items()},
            "split_of_layout": {k: self.grid.split_of(p[0], p[1]) for k, p in self.layout.items()},
        }

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
