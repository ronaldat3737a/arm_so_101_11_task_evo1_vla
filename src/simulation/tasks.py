"""Task definitions for the 11 SO-101 tasks (semantics from configs/tasks.yaml + the dataset videos).

Each task declares its objects (primitive approximations, see reports/simulation_asset_report.md), a nominal layout
(profile `fixed`), how positions are randomized (profile `position*`, restricted to the split's cells), a success
predicate and terminal failure conditions. Instructions are read verbatim from configs/tasks.yaml.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

from .assets import ObjectSpec
from .placement import SplitGrid, sample_xy

ROOT = Path(__file__).resolve().parents[2]
_TASKS_YAML = yaml.safe_load((ROOT / "configs" / "tasks.yaml").read_text())["tasks"]
INSTRUCTIONS = {t["task_id"]: t["instruction"] for t in _TASKS_YAML}

CUBE = 0.025


class Task:
    task_id: str = ""
    max_seconds: float | None = None

    @property
    def instruction(self) -> str:
        return INSTRUCTIONS[self.task_id]

    def objects(self) -> list[ObjectSpec]:
        raise NotImplementedError

    def nominal_layout(self) -> dict[str, tuple[float, float, float]]:
        """name -> (x, y, yaw_deg); z is computed from the object geometry."""
        raise NotImplementedError

    def randomized(self) -> list[dict]:
        """Placement order for randomization: [{name, r, th, split: bool}] (split=False -> any cell)."""
        raise NotImplementedError

    def sample_layout(self, rng: np.random.Generator, grid: SplitGrid, split: str, yaw_range: float, min_sep: float, info: dict) -> dict:
        # Rejection sampling of the WHOLE layout: if one object cannot be placed given the earlier ones, start over.
        for attempt in range(50):
            layout = {}
            placed = []
            try:
                for item in self.randomized():
                    rad = info[item["name"]]["footprint_radius"]
                    r_rng, th_rng = _widened(item, grid, getattr(self, "widen", None))
                    x, y = sample_xy(rng, grid, split if item.get("split", True) else None, placed, rad, min_sep, r_rng, th_rng, max_tries=300)
                    yaw = float(rng.uniform(-yaw_range, yaw_range)) if item.get("yaw", True) else 0.0
                    layout[item["name"]] = (x, y, yaw)
                    placed.append((x, y, rad))
                break
            except RuntimeError:
                if attempt == 49:
                    raise
        for name, pose in self.nominal_layout().items():
            layout.setdefault(name, pose)
        return layout

    def success(self, env) -> tuple[bool, str]:
        raise NotImplementedError

    def oracle_plan(self, env) -> list:
        """Scripted solution (list of oracle Waypoints) computed from privileged object poses after reset."""
        raise NotImplementedError

    def failure(self, env) -> str | None:
        for o in self.objects():
            p = env.body_pos(o.name)
            if p[2] < -0.03 or abs(p[0] - 0.25) > 0.58 or abs(p[1] + 0.15) > 0.68:
                return f"{o.name}_fell_off_table"
        return None


def _inside_container(env, obj: str, container: str, inner_radius: float, height: float) -> tuple[bool, str]:
    p, c = env.body_pos(obj), env.body_pos(container)
    dxy = math.hypot(p[0] - c[0], p[1] - c[1])
    if dxy > inner_radius:
        return False, f"{obj} not above {container} (dxy={dxy:.3f} > {inner_radius:.3f})"
    if p[2] > c[2] + height:
        return False, f"{obj} above rim of {container}"
    return True, ""


def _resting(env, obj: str) -> tuple[bool, str]:
    if env.in_gripper_contact(obj):
        return False, f"{obj} still touching gripper"
    if np.linalg.norm(env.body_linvel(obj)) > 0.05:
        return False, f"{obj} still moving"
    return True, ""


def _drop_height(env) -> float:
    """Release height of the grasp point above a container rim (tasks 01 / 03). Default 2 cm (v1 / v2 data);
    env.oracle_opts["drop_height"] = (lo, hi) samples it from the oracle's own rng (safe-drop demos, v3)."""
    o = getattr(env, "oracle_opts", None) or {}
    if "drop_height" in o and getattr(env, "oracle_rng", None) is not None:
        return float(env.oracle_rng.uniform(*o["drop_height"]))
    return 0.02


def _widened(item: dict, grid: SplitGrid, widen: dict | None) -> tuple:
    """Per-object (r, theta) sampling ranges. `widen` (profile option, e.g. position_wide) extends them by dr / dth,
    clipped to the workspace; pots keep r <= container_r_max because the oracle needs a vertical gripper above them.
    None = the original ranges (v1/v2 data reproduce exactly)."""
    r, th = item.get("r"), item.get("th")
    if not widen:
        return r, th
    r0, r1 = r or (grid.r_min, grid.r_max)
    t0, t1 = th or (grid.theta_min_deg, grid.theta_max_deg)
    r0, r1 = max(grid.r_min, r0 - widen["dr"]), min(grid.r_max, r1 + widen["dr"])
    if "pot" in item["name"]:
        r1 = min(r1, widen.get("container_r_max", r1))
    t0, t1 = max(grid.theta_min_deg, t0 - widen["dth"]), min(grid.theta_max_deg, t1 + widen["dth"])
    return (r0, r1), (t0, t1)


def _upright(env, name: str, max_tilt_deg: float = 30.0) -> bool:
    return env.body_up_z(name) > math.cos(math.radians(max_tilt_deg))


class Task01(Task):
    """Pick up the red cube and place it into the steel pot. Scene (video): steel pot top-left, red cube near the arm,
    distractors (yellow cube, orange cube, yellow triangular prism -> box approximation)."""

    task_id = "task_01"

    def objects(self):
        return [
            ObjectSpec("red_cube", "box", "red", (CUBE,), mass=0.02),
            ObjectSpec("pot", "pot", "steel", (0.065, 0.065, 0.003), mass=0.5),  # ~13 cm stainless pot
            ObjectSpec("yellow_cube", "box", "yellow", (0.02,), mass=0.015),
            ObjectSpec("orange_cube", "box", "orange", (CUBE,), mass=0.02),
            ObjectSpec("yellow_prism", "box", "yellow", (0.035, 0.035, 0.02), mass=0.02),
        ]

    def nominal_layout(self):
        return {
            "red_cube": (0.23, 0.07, 0.0),
            "pot": (0.17, 0.20, 0.0),
            "yellow_cube": (0.26, 0.10, 10.0),
            "orange_cube": (0.27, -0.06, -15.0),
            "yellow_prism": (0.18, -0.14, 30.0),
        }

    def randomized(self):
        return [
            {"name": "pot", "r": (0.18, 0.24), "th": (25, 65), "yaw": False},  # gripper stays vertical over the pot (oracle reachability table)
            {"name": "red_cube", "r": (0.14, 0.26), "th": (-45, 45)},
            {"name": "yellow_cube", "split": False},
            {"name": "orange_cube", "split": False},
            {"name": "yellow_prism", "split": False},
        ]

    def success(self, env):
        inner = env.obj_info["pot"]["inner_radius"] - CUBE * 0.35
        ok, why = _inside_container(env, "red_cube", "pot", inner, env.obj_info["pot"]["height"])
        if not ok:
            return False, why
        return _resting(env, "red_cube")

    def failure(self, env):
        if not _upright(env, "pot"):
            return "pot_tipped_over"
        return super().failure(env)

    def oracle_plan(self, env):
        from .oracle import pick_place_waypoints

        pot = env.body_pos("pot")
        # release the cube ~1 cm above the rim: higher drops make it bounce out of the pot
        return pick_place_waypoints(env, "red_cube", CUBE / 2, (pot[0], pot[1], env.obj_info["pot"]["height"] + _drop_height(env)), yaw_obj_deg=env.layout["red_cube"][2])




# ----------------------------------------------------------------------------------------------------------------
# Shared helpers for tasks 02-11
# ----------------------------------------------------------------------------------------------------------------

def _stacked_on(env, top: str, bottom: str, xy_tol: float, expect_dz: float, dz_tol: float = 0.008) -> tuple[bool, str]:
    a, b = env.body_pos(top), env.body_pos(bottom)
    dxy = math.hypot(a[0] - b[0], a[1] - b[1])
    if dxy > xy_tol:
        return False, f"{top} not on {bottom} (dxy={dxy:.3f} > {xy_tol:.3f})"
    dz = a[2] - b[2]
    if abs(dz - expect_dz) > dz_tol:
        return False, f"{top} height above {bottom} {dz:.3f} != {expect_dz:.3f}"
    return True, ""


def _on_board(env, obj: str, board: str, margin: float = 0.0) -> bool:
    p, b = env.body_pos(obj), env.body_pos(board)
    sx, sy, sz = env.board_size[board]
    # board frame (yaw) -> local coordinates
    yaw = math.atan2(env.data.body(board).xmat.reshape(3, 3)[1, 0], env.data.body(board).xmat.reshape(3, 3)[0, 0])
    dx, dy = p[0] - b[0], p[1] - b[1]
    lx, ly = math.cos(-yaw) * dx - math.sin(-yaw) * dy, math.sin(-yaw) * dx + math.cos(-yaw) * dy
    on_top = p[2] > b[2] + sz / 2
    return abs(lx) <= sx / 2 - margin and abs(ly) <= sy / 2 - margin and on_top


def _board(name, size, color="light_wood", mass=0.3):
    return ObjectSpec(name, "board", color, size, mass=mass)


def _place_target_on(env, support_or_obj: str, obj_half_h: float, extra: float = 0.006, dxy=(0.0, 0.0)):
    """Grasp-point target (object centre) to release `obj` resting on top of `support_or_obj`."""
    p = env.body_pos(support_or_obj)
    info = env.obj_info[support_or_obj]
    top = p[2] + info["half_height"] if info["kind"] in ("box", "board") else env.support_top(support_or_obj)
    return (p[0] + dxy[0], p[1] + dxy[1], top + obj_half_h + extra)


class Task02(Task):
    """Pick up the blue cube and place it next to the blue cube. Video: a light wooden block (~12 x 10 x 3 cm) carries
    one blue cube; a second blue cube lies on the table near the arm; it ends up beside the first one on the block."""

    task_id = "task_02"

    def objects(self):
        return [
            _board("board", (0.12, 0.10, 0.03)),
            ObjectSpec("blue_cube_ref", "box", "blue", (CUBE,)),
            ObjectSpec("blue_cube", "box", "blue", (CUBE,)),
        ]

    def nominal_layout(self):
        return {"board": (0.22, 0.10, 10.0), "blue_cube": (0.20, -0.08, 0.0)}

    def randomized(self):
        return [
            {"name": "board", "r": (0.19, 0.25), "th": (10, 50), "yaw": True},
            {"name": "blue_cube", "r": (0.14, 0.26), "th": (-45, 0)},
        ]

    def post_place(self, env):
        env.place_on("blue_cube_ref", "board", dx=env.rng.uniform(-0.02, 0.02) if env.profile["randomize_positions"] else 0.0, dy=0.025)

    def success(self, env):
        if not _on_board(env, "blue_cube", "board"):
            return False, "blue_cube not on the board"
        a, b = env.body_pos("blue_cube"), env.body_pos("blue_cube_ref")
        d = math.hypot(a[0] - b[0], a[1] - b[1])
        if d > 0.055:
            return False, f"blue_cube not next to the other cube (d={d:.3f})"
        if abs(a[2] - b[2]) > 0.01:
            return False, "blue_cube stacked instead of next to"
        if not _on_board(env, "blue_cube_ref", "board"):
            return False, "reference cube knocked off the board"
        return _resting(env, "blue_cube")

    def oracle_plan(self, env):
        from .oracle import pick_place_waypoints

        b, r = env.body_pos("board"), env.body_pos("blue_cube_ref")
        v = np.array([b[0] - r[0], b[1] - r[1]])
        v = v / (np.linalg.norm(v) + 1e-9)  # beside the reference cube, toward the board centre
        tgt = (r[0] + v[0] * 0.035, r[1] + v[1] * 0.035, r[2] + 0.006)
        return pick_place_waypoints(env, "blue_cube", CUBE / 2, tgt, yaw_obj_deg=env.layout["blue_cube"][2])


class _CubeIntoContainer(Task):
    target = "blue_cube"
    container = "pot"

    def success(self, env):
        inner = env.obj_info[self.container]["inner_radius"] - CUBE * 0.35
        ok, why = _inside_container(env, self.target, self.container, inner, env.obj_info[self.container]["height"])
        if not ok:
            return False, why
        return _resting(env, self.target)

    def failure(self, env):
        if not _upright(env, self.container):
            return f"{self.container}_tipped_over"
        inner = env.obj_info[self.container]["inner_radius"]
        for o in self.objects():
            if o.kind == "box" and o.name != self.target:
                if _inside_container(env, o.name, self.container, inner, env.obj_info[self.container]["height"])[0]:
                    return f"wrong_object_in_{self.container}: {o.name}"
        return super().failure(env)

    def oracle_plan(self, env):
        from .oracle import pick_place_waypoints

        c = env.body_pos(self.container)
        return pick_place_waypoints(env, self.target, CUBE / 2, (c[0], c[1], env.obj_info[self.container]["height"] + _drop_height(env)),
                                    yaw_obj_deg=env.layout[self.target][2])


class Task03(_CubeIntoContainer):
    """Pick up the blue cube and place it into the steel pot. Video: steel bowl top-left, a cluster of red, orange,
    yellow cubes and one blue cube."""

    task_id = "task_03"

    def objects(self):
        return [
            ObjectSpec("blue_cube", "box", "blue", (CUBE,)),
            ObjectSpec("pot", "pot", "steel", (0.065, 0.06, 0.006), mass=0.5),
            ObjectSpec("red_cube_1", "box", "red", (CUBE,)),
            ObjectSpec("red_cube_2", "box", "red", (CUBE,)),
            ObjectSpec("orange_cube", "box", "orange", (CUBE,)),
            ObjectSpec("yellow_cube", "box", "yellow", (0.022,)),
        ]

    def nominal_layout(self):
        return {"pot": (0.17, 0.20, 0.0), "blue_cube": (0.24, 0.02, 0.0), "red_cube_1": (0.27, 0.08, 15.0), "red_cube_2": (0.22, -0.09, 0.0),
                "orange_cube": (0.27, -0.04, 30.0), "yellow_cube": (0.18, -0.16, 0.0)}

    def randomized(self):
        return [{"name": "pot", "r": (0.18, 0.24), "th": (25, 65), "yaw": False}, {"name": "blue_cube", "r": (0.14, 0.26), "th": (-45, 45)},
                {"name": "red_cube_1", "split": False}, {"name": "red_cube_2", "split": False}, {"name": "orange_cube", "split": False}, {"name": "yellow_cube", "split": False}]


class Task04(Task):
    """Take the banana and place it into the steel pot then close the lid. Video: saucepan with a long black handle,
    its lid lying on the table, a banana. Two stages: banana in pot, then lid on pot."""

    task_id = "task_04"
    max_seconds = 40  # two stages; longest demonstrations are 21.8 s

    def objects(self):
        return [
            ObjectSpec("banana", "banana", "banana", (0.14, 0.016), mass=0.12),
            # saucepan in the video is clearly wider than the task_01 pot and holds the banana flat: ~16 cm
            ObjectSpec("pot", "pot", "steel", (0.08, 0.07, 0.006), mass=0.6, extra={"handle_length": 0.10}),
            ObjectSpec("lid", "lid", "steel", (0.083, 0.006), mass=0.15),
        ]

    def nominal_layout(self):
        return {"pot": (0.20, 0.17, 150.0), "banana": (0.22, -0.03, 90.0), "lid": (0.22, -0.16, 0.0)}

    def randomized(self):
        return [{"name": "pot", "r": (0.19, 0.24), "th": (25, 60), "yaw": False}, {"name": "banana", "r": (0.21, 0.28), "th": (-55, 20)},
                {"name": "lid", "r": (0.17, 0.28), "th": (-70, 5), "yaw": False, "split": False}]  # stage-2 object: any cell (banana + pot carry the split)

    def sample_layout(self, rng, grid, split, yaw_range, min_sep, info):
        lay = super().sample_layout(rng, grid, split, yaw_range, min_sep, info)
        x, y, _ = lay["pot"]
        lay["pot"] = (x, y, math.degrees(math.atan2(y, x)) + rng.uniform(-20, 20))  # handle points away from the robot
        bx, by, byaw = lay["banana"]
        lay["banana"] = (bx, by, math.degrees(math.atan2(by, bx)) + 90 + byaw)  # long axis tangential: graspable across
        return lay

    def on_reset(self, env):
        self.banana_in_pot_first = False

    def _banana_in(self, env):
        return _inside_container(env, "banana", "pot", env.obj_info["pot"]["inner_radius"] - 0.01, env.obj_info["pot"]["height"] + 0.01)[0]

    def success(self, env):
        if self._banana_in(env) and not self._lid_on(env):
            self.banana_in_pot_first = True
        if not self._banana_in(env):
            return False, "banana not in pot"
        if not self._lid_on(env):
            return False, "lid not on pot"
        if not self.banana_in_pot_first:
            return False, "lid placed before the banana"
        return _resting(env, "lid")

    def _lid_on(self, env):
        a, b = env.body_pos("lid"), env.body_pos("pot")
        return math.hypot(a[0] - b[0], a[1] - b[1]) < 0.025 and a[2] > b[2] + env.obj_info["pot"]["height"] - 0.015 and env.body_up_z("lid") > 0.9

    def failure(self, env):
        if not _upright(env, "pot"):
            return "pot_tipped_over"
        return super().failure(env)

    def oracle_plan(self, env):
        from .oracle import pick_place_waypoints

        pot, lid = env.body_pos("pot"), env.body_pos("lid")
        h = env.obj_info["pot"]["height"]
        banana_yaw = env.layout["banana"][2]
        plan = pick_place_waypoints(env, "banana", 0.016, (pot[0], pot[1], h + 0.03), yaw_obj_deg=banana_yaw + 90)
        # lid: grasp the top of the knob; release ~2 cm above the rim (pushing it down onto a blocked contact stores
        # energy in the contact that flings it)
        kz = env.obj_info["lid"]["knob_grasp_z"]
        plan += pick_place_waypoints(env, "lid", 0.0, (pot[0], pot[1], h + 0.003 + 0.02 + kz), grasp_point=(lid[0], lid[1], lid[2] + kz))
        return plan


class _CubeOutOfContainerOntoCube(Task):
    color = "blue"
    container_size = (0.065, 0.06, 0.006)
    handle = 0.0

    @property
    def moving(self):
        return f"{self.color}_cube"

    @property
    def base(self):
        return f"{self.color}_cube_base"

    def post_place(self, env):
        env.place_on(self.moving, "pot", yaw_deg=env.rng.uniform(-30, 30) if env.profile["randomize_positions"] else 0.0)

    def success(self, env):
        ok, why = _stacked_on(env, self.moving, self.base, 0.014, CUBE)
        if not ok:
            return False, why
        return _resting(env, self.moving)

    def failure(self, env):
        if not _upright(env, "pot"):
            return "pot_tipped_over"
        return super().failure(env)

    def oracle_plan(self, env):
        from .oracle import pick_place_waypoints

        p = env.body_pos(self.moving)
        q = env.data.body(self.moving).xquat
        yaw = math.degrees(2 * math.atan2(q[3], q[0]))
        return pick_place_waypoints(env, self.moving, CUBE / 2, _place_target_on(env, self.base, CUBE / 2), yaw_obj_deg=yaw,
                                    pre_grasp_height=env.obj_info["pot"]["height"] + 0.05 - p[2], open_pct=25.0,
                                    pre_grasp_hold=int((getattr(env, "oracle_opts", None) or {}).get("hover_steps", 0)))


class Task05(_CubeOutOfContainerOntoCube):
    """Take the blue cube out of the steel pot and place it on the blue cube. Video: saucepan with handle holding a
    blue cube; another blue cube on the table; banana and lid as distractors."""

    task_id = "task_05"
    color = "blue"

    def objects(self):
        return [
            ObjectSpec("pot", "pot", "steel", (0.08, 0.07, 0.006), mass=0.6, extra={"handle_length": 0.10}),
            ObjectSpec("blue_cube", "box", "blue", (CUBE,)),
            ObjectSpec("blue_cube_base", "box", "blue", (CUBE,)),
            ObjectSpec("banana", "banana", "banana", (0.14, 0.016), mass=0.12),
        ]

    def nominal_layout(self):
        return {"pot": (0.21, 0.13, 150.0), "blue_cube_base": (0.22, -0.08, 0.0), "banana": (0.12, 0.20, 30.0)}

    def randomized(self):
        return [{"name": "pot", "r": (0.19, 0.24), "th": (10, 55), "yaw": False}, {"name": "blue_cube_base", "r": (0.15, 0.26), "th": (-50, 0)},
                {"name": "banana", "split": False, "r": (0.15, 0.29), "th": (-70, 75)}]

    def sample_layout(self, rng, grid, split, yaw_range, min_sep, info):
        lay = super().sample_layout(rng, grid, split, yaw_range, min_sep, info)
        x, y, _ = lay["pot"]
        lay["pot"] = (x, y, math.degrees(math.atan2(y, x)) + rng.uniform(-20, 20))
        return lay


class Task06(_CubeOutOfContainerOntoCube):
    """Take the red cube out of the steel pot and place it on the red cube. Video: small steel cup (~9 cm) holding a
    red cube, a second red cube on the table."""

    task_id = "task_06"
    color = "red"

    def objects(self):
        return [
            ObjectSpec("pot", "pot", "steel", (0.048, 0.055, 0.006), mass=0.3),
            ObjectSpec("red_cube", "box", "red", (CUBE,)),
            ObjectSpec("red_cube_base", "box", "red", (CUBE,)),
        ]

    def nominal_layout(self):
        return {"pot": (0.19, 0.15, 0.0), "red_cube_base": (0.24, -0.06, 0.0)}

    def randomized(self):
        return [{"name": "pot", "r": (0.18, 0.24), "th": (15, 60), "yaw": False}, {"name": "red_cube_base", "r": (0.15, 0.26), "th": (-50, 5)}]


class Task07(Task):
    """Stack the yellow cube on top of the orange cube. Video: a wooden block carries the orange cube; the yellow
    cube lies on the table."""

    task_id = "task_07"

    def objects(self):
        return [_board("board", (0.15, 0.12, 0.03)), ObjectSpec("orange_cube", "box", "orange", (CUBE,)), ObjectSpec("yellow_cube", "box", "yellow", (CUBE,))]

    def nominal_layout(self):
        return {"board": (0.22, 0.10, 0.0), "yellow_cube": (0.22, -0.09, 0.0)}

    def randomized(self):
        return [{"name": "board", "r": (0.19, 0.25), "th": (10, 50)}, {"name": "yellow_cube", "r": (0.14, 0.26), "th": (-45, 0)}]

    def post_place(self, env):
        r = env.profile["randomize_positions"]
        env.place_on("orange_cube", "board", dx=env.rng.uniform(-0.03, 0.03) if r else 0.0, dy=env.rng.uniform(-0.02, 0.02) if r else 0.0,
                     yaw_deg=env.rng.uniform(-20, 20) if r else 0.0)

    def success(self, env):
        ok, why = _stacked_on(env, "yellow_cube", "orange_cube", 0.014, CUBE)
        if not ok:
            return False, why
        return _resting(env, "yellow_cube")

    def oracle_plan(self, env):
        from .oracle import pick_place_waypoints

        q = env.data.body("orange_cube").xquat
        return pick_place_waypoints(env, "yellow_cube", CUBE / 2, _place_target_on(env, "orange_cube", CUBE / 2), yaw_obj_deg=env.layout["yellow_cube"][2],
                                    place_yaw_deg=math.degrees(2 * math.atan2(q[3], q[0])))


class Task08(Task):
    """Pick up the cube that makes the number of red and blue cubes on the wooden board equal, and place it in the
    empty corner of the wooden board. Video: square board with cubes in 3 of its 4 corners; one red and one blue cube
    on the table. The correct cube is the colour that is in the minority on the board (randomized per episode)."""

    task_id = "task_08"
    corners = ((1, 1), (1, -1), (-1, 1), (-1, -1))

    def objects(self):
        return [_board("board", (0.15, 0.15, 0.02)), *[ObjectSpec(f"board_cube_{i}", "box", "red", (CUBE,)) for i in range(3)],
                ObjectSpec("table_red", "box", "red", (CUBE,)), ObjectSpec("table_blue", "box", "blue", (CUBE,))]

    def nominal_layout(self):
        return {"board": (0.21, 0.11, 0.0), "table_red": (0.21, -0.07, 0.0), "table_blue": (0.25, -0.13, 0.0)}

    def randomized(self):
        return [{"name": "board", "r": (0.19, 0.24), "th": (15, 50)}, {"name": "table_red", "r": (0.15, 0.27), "th": (-55, -5)}, {"name": "table_blue", "r": (0.15, 0.27), "th": (-55, -5)}]

    def post_place(self, env):
        rnd = env.profile["randomize_positions"]
        self.empty = int(env.rng.integers(4)) if rnd else 0
        minority = ("red", "blue")[int(env.rng.integers(2))] if rnd else "blue"
        majority = "blue" if minority == "red" else "red"
        colors = [majority, majority, minority]
        if rnd:
            env.rng.shuffle(colors)
        b = env.layout_pos("board")
        yaw = math.radians(env.layout["board"][2])
        k = 0
        for i, (cx, cy) in enumerate(self.corners):
            if i == self.empty:
                continue
            lx, ly = cx * 0.05, cy * 0.05
            name = f"board_cube_{k}"
            env.model.geom_rgba[env.model.geom(f"{name}_geom").id] = COLORS_RGBA[colors[k]]
            env.set_object_pose(name, b[0] + math.cos(yaw) * lx - math.sin(yaw) * ly, b[1] + math.sin(yaw) * lx + math.cos(yaw) * ly,
                                env.support_top("board") + CUBE / 2 + 0.0005, env.layout["board"][2])
            k += 1
        self.correct = f"table_{minority}"
        self.wrong = f"table_{majority}"
        self.board_colors = colors

    def corner_xy(self, env):
        b = env.body_pos("board")
        R = env.data.body("board").xmat.reshape(3, 3)
        cx, cy = self.corners[self.empty]
        return b[:2] + R[:2, 0] * cx * 0.05 + R[:2, 1] * cy * 0.05

    def success(self, env):
        if _on_board(env, self.wrong, "board"):
            return False, "wrong cube placed on the board"
        c = self.corner_xy(env)
        p = env.body_pos(self.correct)
        d = math.hypot(p[0] - c[0], p[1] - c[1])
        if not _on_board(env, self.correct, "board") or d > 0.03:
            return False, f"{self.correct} not in the empty corner (d={d:.3f})"
        return _resting(env, self.correct)

    def failure(self, env):
        if _on_board(env, self.wrong, "board") and not env.in_gripper_contact(self.wrong):
            return f"wrong_cube_chosen: {self.wrong}"
        return super().failure(env)

    def oracle_plan(self, env):
        from .oracle import pick_place_waypoints

        c = self.corner_xy(env)
        top = env.body_pos("board")[2] + env.obj_info["board"]["half_height"]
        return pick_place_waypoints(env, self.correct, CUBE / 2, (c[0], c[1], top + CUBE / 2 + 0.006), yaw_obj_deg=env.layout[self.correct][2],
                                    place_yaw_deg=env.layout["board"][2])


class Task09(Task):
    """Pick up the red cube from the wooden board, put it into the orange hollow cube, then pick up the orange hollow
    cube and place it on the wooden board. Two stages."""

    task_id = "task_09"

    def objects(self):
        return [_board("board", (0.14, 0.12, 0.03)), ObjectSpec("red_cube", "box", "red", (0.022,)),
                ObjectSpec("hollow_cube", "hollow_cube", "orange", (0.042, 0.004), mass=0.03)]

    def nominal_layout(self):
        return {"board": (0.22, 0.11, 0.0), "hollow_cube": (0.22, -0.08, 0.0)}

    def randomized(self):
        return [{"name": "board", "r": (0.19, 0.25), "th": (10, 50)}, {"name": "hollow_cube", "r": (0.15, 0.26), "th": (-50, 0)}]

    def post_place(self, env):
        r = env.profile["randomize_positions"]
        env.place_on("red_cube", "board", dx=env.rng.uniform(-0.03, 0.03) if r else 0.0, dy=env.rng.uniform(-0.02, 0.02) if r else 0.0,
                     yaw_deg=env.rng.uniform(-20, 20) if r else 0.0)

    def on_reset(self, env):
        self.stage1_done = False

    def _red_in_hollow(self, env):
        a, h = env.body_pos("red_cube"), env.body_pos("hollow_cube")
        inner = env.obj_info["hollow_cube"]["inner_half"]
        return abs(a[0] - h[0]) < inner and abs(a[1] - h[1]) < inner and a[2] < h[2] + 0.021

    def success(self, env):
        if not self._red_in_hollow(env):
            return False, "red cube not inside the hollow cube"
        if not _on_board(env, "hollow_cube", "board") or env.body_pos("hollow_cube")[2] < env.body_pos("board")[2]:
            self.stage1_done = True
            return False, "hollow cube not on the board"
        if not self.stage1_done:
            return False, "hollow cube on the board before the red cube was put in"
        return _resting(env, "hollow_cube")

    def oracle_plan(self, env):
        from .oracle import pick_place_waypoints

        h = env.body_pos("hollow_cube")
        b = env.body_pos("board")
        plan = pick_place_waypoints(env, "red_cube", 0.011, (h[0], h[1], h[2] + 0.021 + 0.02), yaw_obj_deg=0.0)
        top = b[2] + env.obj_info["board"]["half_height"]
        plan += pick_place_waypoints(env, "hollow_cube", 0.021, (b[0], b[1], top + 0.021 + 0.008), yaw_obj_deg=env.layout["hollow_cube"][2],
                                     grasp_point=(h[0], h[1], h[2] + 0.008), half_width=0.021, place_yaw_deg=env.layout["board"][2])
        return plan


class Task10(Task):
    """Pick up one orange block and stack it neatly on top of the other orange block. Video: two orange rectangular
    blocks (~3 x 3 x 4.5 cm) standing on the table."""

    task_id = "task_10"
    block = (0.03, 0.03, 0.045)

    def objects(self):
        return [ObjectSpec("orange_block_a", "box", "orange", self.block, mass=0.03), ObjectSpec("orange_block_b", "box", "orange", self.block, mass=0.03)]

    def nominal_layout(self):
        return {"orange_block_a": (0.22, -0.07, 0.0), "orange_block_b": (0.23, 0.08, 0.0)}

    def randomized(self):
        return [{"name": "orange_block_b", "r": (0.17, 0.25), "th": (5, 45)}, {"name": "orange_block_a", "r": (0.15, 0.26), "th": (-50, -5)}]

    def success(self, env):
        h = self.block[2]
        for top, bot in (("orange_block_a", "orange_block_b"), ("orange_block_b", "orange_block_a")):
            ok, why = _stacked_on(env, top, bot, 0.012, h)
            if ok:
                if env.body_up_z(top) < 0.95:
                    return False, "top block not upright"
                return _resting(env, top)
        return False, why

    def oracle_plan(self, env):
        from .oracle import pick_place_waypoints

        h = self.block[2]
        return pick_place_waypoints(env, "orange_block_a", h / 2, _place_target_on(env, "orange_block_b", h / 2), yaw_obj_deg=env.layout["orange_block_a"][2],
                                    place_yaw_deg=env.layout["orange_block_b"][2], grasp_depth=h / 2 - 0.012)


class Task11(Task):
    """Take out the blue cubes from shortest to tallest and place it on the table. Video: a wooden shape-sorter box
    holds three blue blocks of different heights standing in its holes (sticking out above the top); they are taken
    out one by one, shortest first, and set on the table. Sorter: platform + low collars (see build_sorter_box)."""

    task_id = "task_11"
    heights = {"blue_short": 0.04, "blue_mid": 0.055, "blue_tall": 0.07}
    max_seconds = 40  # three pick-and-places; longest demonstrations are 29.9 s

    def objects(self):
        return [ObjectSpec("sorter", "sorter_box", "light_wood", (0.15, 0.08, 0.02, 0.004), mass=0.6, extra={"n_slots": 3}),
                *[ObjectSpec(n, "box", "blue", (0.022, 0.022, h), mass=0.015) for n, h in self.heights.items()]]

    def nominal_layout(self):
        return {"sorter": (0.15, 0.13, 131.0)}

    def randomized(self):
        return [{"name": "sorter", "r": (0.18, 0.23), "th": (30, 60)}]  # outer slots stay within ~0.24 m reach

    def sample_layout(self, rng, grid, split, yaw_range, min_sep, info):
        lay = super().sample_layout(rng, grid, split, yaw_range, min_sep, info)
        x, y, _ = lay["sorter"]
        lay["sorter"] = (x, y, math.degrees(math.atan2(y, x)) + 90 + rng.uniform(-15, 15))  # long side tangential
        return lay

    def post_place(self, env):
        s = env.layout_pos("sorter")
        yaw = math.radians(env.layout["sorter"][2])
        order = list(self.heights)
        if env.profile["randomize_positions"]:
            env.rng.shuffle(order)
        for name, sx in zip(order, env.obj_info["sorter"]["slot_x"], strict=True):
            env.set_object_pose(name, s[0] + math.cos(yaw) * sx, s[1] + math.sin(yaw) * sx, env.obj_info["sorter"]["bottom"] + self.heights[name] / 2 + 0.0005, env.layout["sorter"][2])

    def on_reset(self, env):
        self.removed: list[str] = []

    def _in_sorter(self, env, name):
        p, s = env.body_pos(name), env.body_pos("sorter")
        R = env.data.body("sorter").xmat.reshape(3, 3)
        lx, ly = R[:, 0] @ (p - s), R[:, 1] @ (p - s)
        return abs(lx) < 0.075 and abs(ly) < 0.04

    def success(self, env):
        for n in self.heights:
            if n not in self.removed and not self._in_sorter(env, n) and not env.in_gripper_contact(n):
                self.removed.append(n)
        if len(self.removed) < 3:
            return False, f"{3 - len(self.removed)} blocks still in the box"
        if self.removed != sorted(self.heights, key=self.heights.get):
            return False, f"wrong order: {self.removed}"
        for n in self.heights:
            if env.body_pos(n)[2] > self.heights[n] / 2 + 0.01 and env.body_pos(n)[2] > 0.025:  # standing or lying on the table
                return False, f"{n} not resting on the table"
        return True, ""

    def failure(self, env):
        expected = sorted(self.heights, key=self.heights.get)
        if self.removed and self.removed != expected[: len(self.removed)]:
            return f"wrong_order: {self.removed}"
        if not _upright(env, "sorter"):
            return "sorter_tipped_over"
        return super().failure(env)

    def oracle_plan(self, env):
        from .oracle import pick_place_waypoints

        plan = []
        s = env.body_pos("sorter")
        r_s, th_s = math.hypot(s[0], s[1]), math.atan2(s[1], s[0])
        tallest = env.obj_info["sorter"]["bottom"] + max(self.heights.values())
        safe_z = tallest + 0.02 + 0.03  # fingertips reach 2 cm below the grasp point; +3 cm clearance over the pieces
        for k, n in enumerate(sorted(self.heights, key=self.heights.get)):
            h = self.heights[n]
            # row on the table at the box's radius, on the robot-front side of the box (smaller azimuth)
            th = th_s - (0.075 + 0.05 + 0.045 * k) / r_s
            spot = (r_s * math.cos(th), r_s * math.sin(th))
            gz = env.body_pos(n)[2] + h / 2 - 0.012
            # close across the box width (yaw + 90): along the row the jaws would touch the neighbouring pieces
            plan += pick_place_waypoints(env, n, h / 2, (spot[0], spot[1], h - 0.012 + 0.008), yaw_obj_deg=env.layout["sorter"][2] + 90,
                                         grasp_depth=h / 2 - 0.012, pre_grasp_height=max(0.05, safe_z - gz), transport_z=safe_z,
                                         open_pct=30.0, half_width=0.011)
        return plan


COLORS_RGBA = {"red": np.array([0.80, 0.08, 0.08, 1]), "blue": np.array([0.08, 0.20, 0.75, 1])}

TASKS: dict[str, type[Task]] = {
    "task_01": Task01, "task_02": Task02, "task_03": Task03, "task_04": Task04, "task_05": Task05, "task_06": Task06,
    "task_07": Task07, "task_08": Task08, "task_09": Task09, "task_10": Task10, "task_11": Task11,
}


def make_task(task_id: str) -> Task:
    if task_id not in TASKS:
        raise KeyError(f"{task_id} not implemented; available: {sorted(TASKS)}")
    return TASKS[task_id]()
