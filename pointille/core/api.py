"""Entry point: ``distribute(mesh, options) -> DroneResult``."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from . import color as _color
from . import outline as _outline
from .facade import Audience
from .mesh import MeshData
from .relax import SurfaceProjector, relax
from .sampling import eliminate, poisson_radius, spacing_scale, surface_candidates, triangle_areas_normals
from .spatial import nearest
from .stats import scale_factor, scale_pivot, spacing

COLOR_MODES = ("area", "point", "none")
OUTLINE_AMOUNTS = _outline.AMOUNTS


@dataclass
class DistributeOptions:
    count: int = 500
    seed: int = 0
    oversample: float = 8.0
    relax_iterations: int = 20
    # facade: drones go where the audience sees the model; view_dir points model -> audience
    facade: bool = False
    view_dir: Tuple[float, float, float] = (0.0, -1.0, 0.0)
    # a camera instead of a far-away audience: sight lines converge on this point
    view_point: Optional[Tuple[float, float, float]] = None
    facade_angle: float = 80.0
    # degrees the audience spans around the vertical axis, seen from the model
    audience_spread: float = 0.0
    # drone density where nobody in the audience sees the model, relative to the facade
    hidden_density: float = 0.0
    occlusion: bool = True
    # "area": mean color of the drone's surface patch, "point": color under the drone
    color_mode: str = "area"
    # LEDs cannot be black: the darkest drone glows at this sRGB brightness 0..1 (0: the model's colours as they are)
    min_brightness: float = 0.0
    # outline: some drones right on the lines that draw the model, closer together than the fill
    outline: bool = False
    outline_silhouette: bool = True  # the contour the middle viewer sees (``view_dir`` / ``view_point``)
    outline_creases: bool = True  # edges sharper than ``crease_angle``
    outline_borders: bool = True  # edges of open surfaces: text, logos, planes
    outline_relief: bool = True  # what a line drawing shows inside the silhouette: eyes, mouth, folds
    outline_colors: bool = True  # borders between colours of the texture
    # "density": lines ``outline_density`` times denser than the fill; "share": that share of the drones
    outline_amount: str = "density"
    outline_density: float = 1.5
    outline_share: float = 0.3
    crease_angle: float = 40.0  # degrees between two faces that make a sharp edge
    relief_angle: float = 40.0  # degrees the surface must bend within a short stretch to draw a relief line
    color_contrast: float = 0.15  # OKLab difference that makes a colour border (black-white is 1)
    # ``layout.compose`` (``count`` is the most drones it may use, the corners ``target_min_distance`` apart):
    # the lines are this many times and the fill this many times that far apart
    line_ratio: float = 1.5
    fill_ratio: float = 3.0
    # ``around`` figures: a wave from the core through the inside tells the parts apart; upright parts keep
    # horizontal rows, sideways ones (a tail, wings, a lying body) get rings across, bright columns on every
    # row draw the outline from every side, the ends of parts are tips. False: horizontal rows everywhere
    around_parts: bool = True
    # safety scaling: "none" | "min_distance" | "height"
    scale_mode: str = "none"
    target_min_distance: float = 1.5
    target_height: float = 20.0
    up_axis: int = 2


@dataclass
class DroneResult:
    positions: np.ndarray  # (N, 3) in the model's world space, unscaled
    normals: np.ndarray  # (N, 3)
    triangle_ids: np.ndarray  # (N,)
    barycentrics: np.ndarray  # (N, 3)
    colors_linear: np.ndarray  # (N, 3)
    min_distance: float  # nearest-neighbour spacing, unscaled
    mean_distance: float
    area: float  # surface area the drones were spread over (weighted by density in facade mode)
    scale: float = 1.0  # suggested uniform scale (see ``scale_mode``)
    pivot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    options: Optional[DistributeOptions] = None
    visibility: Optional[np.ndarray] = None  # (N,) share of the audience that sees each drone
    outline: Optional[np.ndarray] = None  # (N,) True for the drones on the outline (they come first)
    outline_spacing: float = 0.0  # distance between neighbouring outline drones, unscaled
    # ``layout.compose`` only
    kind: Optional[np.ndarray] = None  # (N,) 0 in a corner, 1 on a line, 2 in the fill (in this order)
    part: Optional[np.ndarray] = None  # (N,) which of the parts each drone is on
    budget: int = 0  # the most drones the layout could use
    needed: int = 0  # how many it would have used at the spacing asked for
    spread: float = 1.0  # how much wider than asked the lines (and corners) stand, to stay within the budget
    fill_spread: float = 1.0  # ... and the fill (it gives way first)

    @property
    def outline_count(self) -> int:
        return 0 if self.outline is None else int(self.outline.sum())

    @property
    def count(self) -> int:
        return len(self.positions)

    @property
    def colors_srgb8(self) -> np.ndarray:
        return _color.to_srgb8(self.colors_linear)

    def scaled_positions(self) -> np.ndarray:
        return self.pivot + (self.positions - self.pivot) * self.scale

    @property
    def bbox(self):
        p = self.positions
        return p.min(axis=0), p.max(axis=0)

    @property
    def size(self) -> np.ndarray:
        lo, hi = self.bbox
        return hi - lo

    def summary(self) -> dict:
        size = self.size * self.scale
        return {
            "count": self.count,
            "scale": self.scale,
            "min_distance": self.min_distance * self.scale,
            "mean_distance": self.mean_distance * self.scale,
            "size": [float(v) for v in size],
            "outline_count": self.outline_count,
            "outline_spacing": self.outline_spacing * self.scale,
            "budget": self.budget,
            "needed": self.needed,
            "spread": self.spread,
            "fill_spread": self.fill_spread,
        }


def distribute(mesh: MeshData, options: DistributeOptions = None) -> DroneResult:
    """Spread exactly ``options.count`` drones evenly over the mesh surface.

    The mesh is only read; nothing in it is modified.
    """
    opt = options or DistributeOptions()
    n = int(opt.count)
    if n < 1:
        raise ValueError("count must be at least 1")
    if opt.color_mode not in COLOR_MODES:
        raise ValueError(f"unknown color mode {opt.color_mode!r}")
    if opt.outline_amount not in OUTLINE_AMOUNTS:
        raise ValueError(f"unknown outline amount {opt.outline_amount!r}")
    rng = np.random.default_rng(int(opt.seed))

    tris = mesh.triangles
    areas, normals = triangle_areas_normals(tris)
    weights, audience, density = _audience(tris, areas, normals, opt)
    total_area = float(weights.sum())

    wanted = max(int(math.ceil(max(opt.oversample, 2.0) * n)), n + 1)
    pos, tri, bary, rho, mean_rho = _gather_candidates(tris, weights, wanted, rng, density)
    if len(pos) < n:
        raise ValueError(
            f"the visible surface only fits about {len(pos)} drones; "
            "lower the count or disable facade mode"
        )
    # densities relative to the densest candidate: there the spacing is ``ideal``
    top = float(rho.max())
    area = total_area * mean_rho / top
    stretch = None if rho.min() == top else spacing_scale(rho / top)
    relative = None if density is None else (lambda p, t: density(p, t) / top)  # noqa: E731

    # the outline first: its drones stand still while the fill makes room around them
    lines, n_line, line_spacing = None, 0, 0.0
    fill, fill_area = np.ones(len(pos), bool), area
    if opt.outline and n > 3:
        lines = _outline.find_lines(mesh, areas, normals, weights, opt, 2.0 * poisson_radius(area, n), rng,
                                    audience)
    if lines is not None:
        n_line, line_spacing, fill, fill_area = _outline.split(lines, pos, area, n, opt)
    if n_line:
        l_pos, l_tri, l_bary = _outline.place(lines, n_line, line_spacing, opt.relax_iterations, tris)
        n_line = len(l_pos)
    n_fill = n - n_line
    if fill.sum() < n_fill:  # the lines took fewer drones than planned and left no room for the rest
        fill, fill_area = np.ones(len(pos), bool), area

    f_pos, f_tri, f_bary = pos[fill], tri[fill], bary[fill]
    f_stretch = None if stretch is None else stretch[fill]
    sel = eliminate(f_pos, n_fill, fill_area, scale=f_stretch)
    radius = poisson_radius(fill_area, n_fill)
    ideal = 2.0 * radius  # neighbour distance of a perfect hexagonal layout
    points, p_tri, p_bary = f_pos[sel], f_tri[sel], f_bary[sel]
    if opt.relax_iterations > 0 and n > 1:
        allowed = np.flatnonzero(weights > 0)
        widest = 1.0 if f_stretch is None else float(f_stretch.max())
        projector = SurfaceProjector(tris[allowed], allowed, 0.5 * ideal, 0.15 * ideal * widest)
        points, p_tri, p_bary = relax(points, p_tri, p_bary, projector, ideal, opt.relax_iterations,
                                      density=relative, max_scale=widest, fixed=l_pos if n_line else None)
    if n_line:
        points = np.concatenate([l_pos, points])
        p_tri = np.concatenate([l_tri, p_tri])
        p_bary = np.concatenate([l_bary, p_bary])

    if opt.color_mode == "area":
        cand_colors = _color.surface_colors(mesh, tri, bary)
        _, owner = nearest(pos, points, radius)
        colors = _color.average_by_owner(cand_colors, owner, n)
    elif opt.color_mode == "point":
        colors = _color.surface_colors(mesh, p_tri, p_bary)
    else:
        colors = np.ones((n, 3))
    if n_line and opt.color_mode != "none":
        # an outline drone shows the colour right under it, or the side of a colour border it marks
        under = _color.surface_colors(mesh, p_tri[:n_line], p_bary[:n_line])
        marked = lines.colors_at(points[:n_line])
        colors[:n_line] = np.where(np.isnan(marked), under, marked)
    colors = _color.glow(colors, opt.min_brightness)

    dmin, dmean = spacing(points, radius)
    lo, hi = points.min(axis=0), points.max(axis=0)
    scale = scale_factor(opt.scale_mode, dmin, hi - lo, opt.up_axis,
                         opt.target_min_distance, opt.target_height)
    return DroneResult(
        positions=points,
        normals=normals[p_tri],
        triangle_ids=p_tri,
        barycentrics=p_bary,
        colors_linear=colors,
        min_distance=dmin,
        mean_distance=dmean,
        area=area,
        scale=scale,
        pivot=scale_pivot(lo, hi, opt.up_axis),
        options=opt,
        visibility=audience.visibility(points, p_tri) if audience is not None else np.ones(n),
        outline=np.arange(n) < n_line,
        outline_spacing=line_spacing if n_line else 0.0,
    )


def _audience(tris, areas, normals, opt):
    """(triangle weights, audience, density) of the surface: the area of each triangle, in facade mode
    zero where nobody may see it; the audience, None without facade mode; the density of drones at a
    point, None for an even one."""
    weights = areas.copy()
    audience = density = None
    if opt.facade:
        audience = Audience(tris, normals, view_dir=opt.view_dir, view_point=opt.view_point,
                            spread=opt.audience_spread, facade_angle=opt.facade_angle,
                            occlusion=opt.occlusion, up_axis=opt.up_axis)
        hidden = min(max(float(opt.hidden_density), 0.0), 1.0)
        if hidden <= 0:
            weights *= audience.triangle_mask()
            if weights.sum() <= 0:
                raise ValueError("no part of the surface faces the audience")
        if hidden < 1:
            # full density where the whole audience sees the surface, ``hidden`` where nobody does
            density = lambda p, t: hidden + (1.0 - hidden) * audience.visibility(p, t)  # noqa: E731
    if float(weights.sum()) <= 0:
        raise ValueError("the model has no surface area")
    return weights, audience, density


def _gather_candidates(tris, weights, wanted, rng, density):
    """Candidates over the surface, as many per area as ``density`` says
    (uniform without it): each is kept with probability ``density``.

    Returns (positions, triangle_ids, barycentrics, densities, mean_density);
    the mean is over the area the candidates were drawn from.
    """
    if density is None:
        pos, tri, bary = surface_candidates(tris, weights, wanted, rng)
        return pos, tri, bary, np.ones(len(pos)), 1.0
    parts, have, generated, total, batch = [], 0, 0, 0.0, wanted
    for _ in range(6):
        pos, tri, bary = surface_candidates(tris, weights, batch, rng)
        rho = density(pos, tri)
        generated += batch
        total += float(rho.sum())
        keep = rng.random(batch) < rho
        parts.append((pos[keep], tri[keep], bary[keep], rho[keep]))
        have += int(keep.sum())
        if have >= wanted:
            break
        batch = int(math.ceil((wanted - have) / max(total / generated, 0.02) * 1.1))
    pos, tri, bary, rho = (np.concatenate([p[k] for p in parts])[:wanted] for k in range(4))
    return pos, tri, bary, rho, total / generated
