"""Object placement with a spatial train/val/test split (checkerboard on a polar grid around the robot base)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SplitGrid:
    cell_dr: float
    cell_dtheta_deg: float
    labels: dict  # split -> list of residues
    r_min: float
    r_max: float
    theta_min_deg: float
    theta_max_deg: float
    keepout: dict | None = None

    @classmethod
    def from_config(cls, cfg: dict) -> SplitGrid:
        ps, ws = cfg["position_split"], cfg["workspace"]
        return cls(ps["cell_dr"], ps["cell_dtheta"], ps["labels"], ws["r_min"], ws["r_max"], ws["theta_min"], ws["theta_max"], ws.get("keepout"))

    def residue(self, x: float, y: float) -> int:
        r = math.hypot(x, y)
        th = math.degrees(math.atan2(y, x))
        i = math.floor((r - self.r_min) / self.cell_dr)
        j = math.floor((th - self.theta_min_deg) / self.cell_dtheta_deg)
        return (i + 2 * j) % 5

    def split_of(self, x: float, y: float) -> str:
        res = self.residue(x, y)
        for name, vals in self.labels.items():
            if res in vals:
                return name
        raise ValueError(res)

    def in_keepout(self, x: float, y: float, radius: float) -> bool:
        if not self.keepout:
            return False
        (x0, x1), (y0, y1) = self.keepout["x"], self.keepout["y"]
        return x0 - radius <= x <= x1 + radius and y0 - radius <= y <= y1 + radius

    def in_workspace(self, x: float, y: float, r_min=None, r_max=None, th_min=None, th_max=None) -> bool:
        r = math.hypot(x, y)
        th = math.degrees(math.atan2(y, x))
        return (r_min or self.r_min) <= r <= (r_max or self.r_max) and (th_min if th_min is not None else self.theta_min_deg) <= th <= (th_max if th_max is not None else self.theta_max_deg)


def sample_xy(
    rng: np.random.Generator,
    grid: SplitGrid,
    split: str | None,
    placed: list[tuple[float, float, float]],
    radius: float,
    min_sep: float,
    r_range: tuple[float, float] | None = None,
    th_range: tuple[float, float] | None = None,
    max_tries: int = 2000,
) -> tuple[float, float]:
    """Uniform sample in the (r, theta) annulus sector; accept if in `split` cells (None = any) and not overlapping."""
    r0, r1 = r_range or (grid.r_min, grid.r_max)
    t0, t1 = th_range or (grid.theta_min_deg, grid.theta_max_deg)
    for _ in range(max_tries):
        r = math.sqrt(rng.uniform(r0**2, r1**2))
        th = math.radians(rng.uniform(t0, t1))
        x, y = r * math.cos(th), r * math.sin(th)
        if split is not None and grid.split_of(x, y) != split:
            continue
        if grid.in_keepout(x, y, min(radius, 0.06)):  # keep-out protects the resting fingertips; cap for long thin objects
            continue
        if all(math.hypot(x - px, y - py) >= radius + pr + min_sep for px, py, pr in placed):
            return x, y
    raise RuntimeError(f"could not place object (split={split}, r={r_range}, th={th_range}) after {max_tries} tries")
