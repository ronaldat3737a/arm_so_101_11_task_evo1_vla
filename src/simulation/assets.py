"""Object asset abstraction: every object is built by a named builder from primitives.

Replacing a primitive with a real mesh later means writing a new builder (e.g. one that adds a mesh geom) and
pointing the object's `kind` at it. Task logic only uses body names, poses and the declared semantic dimensions.

All dimensions in metres. Approximations are listed in reports/simulation_asset_report.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

COLORS = {
    "red": (0.80, 0.08, 0.08, 1),
    "blue": (0.08, 0.20, 0.75, 1),
    "yellow": (0.95, 0.85, 0.10, 1),
    "orange": (0.98, 0.50, 0.08, 1),
    "steel": (0.72, 0.74, 0.76, 1),
    "wood": (0.78, 0.62, 0.42, 1),
    "light_wood": (0.90, 0.80, 0.64, 1),
    "banana": (0.98, 0.86, 0.20, 1),
    "white": (0.92, 0.92, 0.90, 1),
}


@dataclass
class ObjectSpec:
    name: str
    kind: str  # builder name
    color: str = "red"
    size: tuple = (0.025,)  # meaning depends on kind (documented per builder)
    mass: float = 0.02
    friction: tuple = (1.0, 0.005, 0.0001)
    extra: dict = field(default_factory=dict)


def _geom(body, gtype, size, pos=(0, 0, 0), quat=(1, 0, 0, 0), rgba=(1, 1, 1, 1), mass=None, friction=(1.0, 0.005, 0.0001), name=None):
    g = body.add_geom()
    g.type = gtype
    g.size = list(size) + [0] * (3 - len(size))
    g.pos = list(pos)
    g.quat = list(quat)
    g.rgba = list(rgba)
    g.friction = list(friction)
    g.condim = 4
    # moderately soft object contacts (MuJoCo-default solimp). Stiff contacts on every object made a cube dropped
    # 5 cm into the pot rebound and launch the pot; only the finger pads are stiff (they win via geom priority).
    g.solref = [0.01, 1.0]
    if mass is not None:
        g.mass = mass
    if name:
        g.name = name
    return g


def build_box(body, spec: ObjectSpec):
    """size = (edge,) for a cube or (sx, sy, sz) full extents."""
    s = spec.size if len(spec.size) == 3 else (spec.size[0],) * 3
    _geom(body, mujoco.mjtGeom.mjGEOM_BOX, [x / 2 for x in s], rgba=COLORS[spec.color], mass=spec.mass, friction=spec.friction, name=f"{spec.name}_geom")
    return {"half_height": s[2] / 2, "footprint_radius": math.hypot(s[0], s[1]) / 2}


def build_pot(body, spec: ObjectSpec):
    """Open cylindrical pot: size = (outer_radius, height, wall_thickness). Bottom disk + ring of wall boxes + 2 handles.
    extra["bottom_mass_fraction"] (optional): share of the mass in the bottom disk (default: uniform over the parts).
    Collision walls/bottom are at least 6 mm / 8 mm thick: thinner shells let a falling cube tunnel and get ejected."""
    r, h, t = spec.size
    t = max(t, 0.006)
    rgba = COLORS[spec.color]
    n = 20
    per = spec.mass / (n + 2)
    m_bottom = per * 2
    fb = spec.extra.get("bottom_mass_fraction")
    if fb is not None:  # real stainless pots have a thick (encapsulated) base: most of the mass sits in the bottom
        m_bottom = fb * spec.mass
        per = (1.0 - fb) * spec.mass / (n + 2)  # walls + handles share the rest (same total mass)
    tb = 0.008
    _geom(body, mujoco.mjtGeom.mjGEOM_CYLINDER, [r, tb / 2], pos=(0, 0, tb / 2), rgba=rgba, mass=m_bottom, friction=spec.friction, name=f"{spec.name}_bottom")
    seg = 2 * r * math.tan(math.pi / n) * 1.05
    for i in range(n):
        a = 2 * math.pi * i / n
        q = [math.cos(a / 2), 0, 0, math.sin(a / 2)]
        _geom(body, mujoco.mjtGeom.mjGEOM_BOX, [t / 2, seg / 2, h / 2], pos=((r - t / 2) * math.cos(a), (r - t / 2) * math.sin(a), h / 2), quat=q, rgba=rgba, mass=per, friction=spec.friction)
    handle = spec.extra.get("handle_length", 0.0)
    if handle:  # saucepan: one long black handle along -x of the pot frame (yaw sets its direction)
        _geom(body, mujoco.mjtGeom.mjGEOM_BOX, [handle / 2, 0.009, 0.005], pos=(-(r + handle / 2), 0, h - 0.008), rgba=(0.08, 0.08, 0.08, 1), mass=per * 2)
    else:
        for sgn in (-1, 1):
            _geom(body, mujoco.mjtGeom.mjGEOM_BOX, [0.012, 0.006, 0.004], pos=(0, sgn * (r + 0.01), h - 0.01), rgba=rgba, mass=per)
    return {"half_height": 0.0, "footprint_radius": r + 0.02, "inner_radius": r - t, "height": h, "bottom": 0.008, "handle_length": handle}


def build_lid(body, spec: ObjectSpec):
    """Pot lid: size = (radius, thickness). Thin disk + knob on top."""
    r, t = spec.size
    rgba = COLORS[spec.color]
    _geom(body, mujoco.mjtGeom.mjGEOM_CYLINDER, [r, t / 2], rgba=rgba, mass=spec.mass * 0.8, friction=spec.friction, name=f"{spec.name}_disk")
    knob_h = 0.03  # the fingertips reach 2 cm below the grasp point, so the knob must be taller than that
    _geom(body, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.009, knob_h / 2], pos=(0, 0, t / 2 + knob_h / 2), rgba=(0.15, 0.15, 0.15, 1), mass=spec.mass * 0.2, friction=spec.friction, name=f"{spec.name}_knob")
    return {"half_height": t / 2, "footprint_radius": r, "knob_grasp_z": t / 2 + knob_h - 0.006}


def build_banana(body, spec: ObjectSpec):
    """Curved banana from 3 capsules along an arc: size = (length, radius). `length` is the overall tip-to-tip extent."""
    length, rad = spec.size
    rgba = COLORS[spec.color]
    seg = (length - 2 * rad) / 3  # capsule segment length so that the 3 segments + end caps span `length`
    for i, ang in enumerate((-0.35, 0.0, 0.35)):
        cx = (i - 1) * seg * math.cos(0.35)
        cy = -abs(i - 1) * seg * math.sin(0.35) * 0.5
        q = [math.cos((math.pi / 2) / 2), 0, math.sin((math.pi / 2) / 2), 0]  # capsule axis along x
        qz = [math.cos(ang / 2), 0, 0, math.sin(ang / 2)]
        qq = np.zeros(4)
        mujoco.mju_mulQuat(qq, np.array(qz), np.array(q))
        _geom(body, mujoco.mjtGeom.mjGEOM_CAPSULE, [rad, seg / 2], pos=(cx, cy, 0), quat=qq, rgba=rgba, mass=spec.mass / 3, friction=spec.friction)
    # separation radius: a full half-length circle is far too conservative for a thin curved object
    return {"half_height": rad, "footprint_radius": length * 0.35}


def build_board(body, spec: ObjectSpec):
    """Flat board: size = (sx, sy, thickness)."""
    sx, sy, sz = spec.size
    _geom(body, mujoco.mjtGeom.mjGEOM_BOX, [sx / 2, sy / 2, sz / 2], rgba=COLORS[spec.color], mass=spec.mass, friction=spec.friction, name=f"{spec.name}_geom")
    return {"half_height": sz / 2, "footprint_radius": math.hypot(sx, sy) / 2, "top_z_offset": sz / 2}


def build_hollow_cube(body, spec: ObjectSpec):
    """Open-top hollow cube shell: size = (outer_edge, wall_thickness). Bottom + 4 walls."""
    e, t = spec.size
    rgba = COLORS[spec.color]
    per = spec.mass / 5
    _geom(body, mujoco.mjtGeom.mjGEOM_BOX, [e / 2, e / 2, t / 2], pos=(0, 0, -e / 2 + t / 2), rgba=rgba, mass=per, friction=spec.friction, name=f"{spec.name}_bottom")
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        half = [t / 2, e / 2, e / 2] if sx else [e / 2, t / 2, e / 2]
        _geom(body, mujoco.mjtGeom.mjGEOM_BOX, half, pos=(sx * (e / 2 - t / 2), sy * (e / 2 - t / 2), 0), rgba=rgba, mass=per, friction=spec.friction)
    return {"half_height": e / 2, "footprint_radius": e * 0.71, "inner_half": e / 2 - t}


def build_sorter_box(body, spec: ObjectSpec):
    """Shape-sorter box: a closed wooden box whose top has holes; the pieces stand in the holes and stick out well above
    the top (video). Approximation: solid platform (height `h`) + a low collar (6 mm) around each of `n_slots` square
    holes that keeps the pieces upright. size = (sx, sy, h, collar_wall)."""
    sx, sy, h, t = spec.size
    n = spec.extra.get("n_slots", 3)
    hole = spec.extra.get("hole", 0.028)
    collar_h = 0.006
    rgba = COLORS[spec.color]
    _geom(body, mujoco.mjtGeom.mjGEOM_BOX, [sx / 2, sy / 2, h / 2], pos=(0, 0, h / 2), rgba=rgba, mass=spec.mass * 0.9, friction=spec.friction, name=f"{spec.name}_platform")
    slot_x = [-sx / 2 + (i + 0.5) * sx / n for i in range(n)]
    per = spec.mass * 0.1 / (4 * n)
    for x in slot_x:
        for dx, dy, half in ((hole / 2 + t / 2, 0, [t / 2, hole / 2 + t, collar_h / 2]), (-(hole / 2 + t / 2), 0, [t / 2, hole / 2 + t, collar_h / 2]),
                             (0, hole / 2 + t / 2, [hole / 2, t / 2, collar_h / 2]), (0, -(hole / 2 + t / 2), [hole / 2, t / 2, collar_h / 2])):
            _geom(body, mujoco.mjtGeom.mjGEOM_BOX, half, pos=(x + dx, dy, h + collar_h / 2), rgba=(0.55, 0.40, 0.25, 1), mass=per)
    return {"half_height": 0.0, "footprint_radius": math.hypot(sx, sy) / 2, "slot_x": slot_x, "height": h + collar_h, "bottom": h}


BUILDERS = {
    "box": build_box,
    "pot": build_pot,
    "lid": build_lid,
    "banana": build_banana,
    "board": build_board,
    "hollow_cube": build_hollow_cube,
    "sorter_box": build_sorter_box,
}


def add_object(spec_root: mujoco.MjSpec, obj: ObjectSpec) -> dict:
    body = spec_root.worldbody.add_body()
    body.name = obj.name
    body.pos = [0, 0, 0.5]
    j = body.add_freejoint()
    j.name = f"{obj.name}_free"
    info = BUILDERS[obj.kind](body, obj)
    if obj.extra.get("contact_priority"):
        # MuJoCo combines two geoms' friction with max(); a higher priority makes this object's own friction apply
        # (e.g. a steel pot on the wooden table) without changing the table for every other object
        for g in body.geoms:
            g.priority = int(obj.extra["contact_priority"])
    info["kind"] = obj.kind
    return info
