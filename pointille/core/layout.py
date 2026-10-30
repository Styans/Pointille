"""Layout: the drones that draw a figure best, as few as it needs, at most a budget.

``distribute`` spreads exactly N drones. A drawing of a figure needs a
different order: its corners first, then its lines, then a light fill. Here
the spacings are asked for instead of a count:

* corners - where lines end, meet or bend sharply, ``d`` apart (the minimum
  distance), the spacing growing along each line away from the corner;
* lines   - silhouette, relief, creases, borders, colour borders (see
  ``outline``), ``line_ratio * d`` apart;
* fill    - the rest of the surface, ``fill_ratio * d`` apart, as a lattice of
  triangles or squares the audience sees straight, or as blue noise.

The layout takes only as many drones as that gives, the budget is an upper
limit: when the figure needs more, every spacing grows by the same factor
until it fits. The model may be made of several parts (the objects of a
scene), each with its own role:

* "full"    - lines and fill;
* "outline" - only the lines (a rock under a tiger);
* "pattern" - the lines and a fill of only some of the colours of the model
  (the black stripes of a tiger, not its orange coat);
* "fill"    - only the fill.

Which colours a pattern fills is picked by colour groups (k-means in OKLab,
the user ticks groups), by an eyedropper (colours and a tolerance) or
automatically (every group that stands out from the main colour).

In facade mode the figure is drawn from the picture the audience sees
(``picture``): every silhouette as smooth even lines (a rod along its middle),
the most salient inner lines that do not crowd them, real corners only, and a
lattice the audience sees straight. The spacing factor is estimated from a
small picture first, so a figure that needs far more drones than the budget
is never laid out at the spacing asked for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence

import numpy as np

from . import color as _color
from . import outline as _outline
from . import picture as _pic
from . import shapes as _shapes
from .api import DroneResult, DistributeOptions, _audience, _gather_candidates
from .facade import normalize, rotate, view_basis
from .mesh import MeshData
from .relax import SurfaceProjector, relax
from .sampling import eliminate, spacing_scale, surface_candidates, triangle_areas_normals
from .spatial import Grid, nearest
from .stats import scale_pivot, spacing

ROLES = ("full", "outline", "pattern", "fill")
PATTERNS = ("hex", "square", "noise", "rings")
PICKS = ("groups", "eyedropper", "auto")
HEX = math.sqrt(3.0) / 2.0  # area of a lattice cell of spacing 1 (triangular lattice)
LINE_DETAIL = 1.5  # lines are looked for at this many line spacings: finer detail draws no line
CLEAR = 1.02  # drones next to a corner stand a hair farther than the minimum distance
GAP = 0.45  # the fill keeps this many fill spacings clear of the lines (and half a spacing from their drones)
SEAM = 0.7  # fill drones of two projections of the lattice keep this many fill spacings apart
PHASES = 3  # the lattice is tried at this many shifts along each of its axes
VOTE = 0.5  # a fill drone takes the colour group most of the surface within this many fill spacings has
CANDIDATES = 8.0  # candidates per drone that would fit at the minimum distance (times ``oversample / 8``)
MAX_CANDIDATES = 1_000_000
ROUNDS = 8
FIT = 0.95  # a layout made sparser to fit the budget uses at least this share of it, when it can
_RASTER_PAIRS = 1 << 22
# facade mode (``picture``)
ESTIMATE_PIXELS = 250_000  # pixels of the quick look that estimates how many drones the figure needs
EXACT_NEED = 4.0  # the figure is laid out at the spacing asked for when it needs at most this many budgets
INNER_USE = 0.5  # share of their allowance inner lines take, as the quick look guesses
HIDDEN_SEEN = 0.6  # share of the area facing the audience the middle viewer sees, as the quick guess has it
REFINE = 6  # bisection steps that fit the budget with the best reading of the picture
REDETECT = 1.3  # the picture is read again when the spacing changes more than this factor
RING_FACING = 0.3  # a ring drone where the surface turns away from the audience more than this (cos) is left out
SPOT_OCTAVES = 3  # with simple shapes, spots (eyes) are looked for at 1, 2 and 4 times the line scale
DETECT_PIXELS = 1_500_000  # the picture the lines are read from has at most about this many pixels
VIEWS_360 = 4  # without facade mode, the lines are read from this many views round the vertical axis
REDETECT_360 = 1.6  # ... read again only when their spacing changes more than this factor (four views cost)
FILL_FIRST = 2.0  # short of drones, the fill grows this much sparser before the lines do
ROUND_MOSTLY = 0.7  # without facade mode, a part this much round bodies (by area) draws no relief lines
CREASE_LONG = 3.0  # without facade mode, sharp edges of the mesh this many line spacings long are lines too
FILL_GLOW = 0.45  # without facade mode, the fill glows this much of its colour: a background to the lines


@dataclass
class PartOptions:
    role: str = "full"
    pattern: str = "hex"  # how the fill is laid out: "hex" | "square" | "noise" | "rings" (round its vertical axis)
    # simple shapes (see ``shapes``): eyes as rings, near-ellipses exact, straight edges straight, round
    # bodies filled with rings from the bottom up
    shapes: bool = True
    silhouette: bool = True
    relief: bool = True
    creases: bool = True
    borders: bool = True
    colors: bool = True
    # role "pattern": which colours get the fill
    pick: str = "groups"
    groups: Optional[np.ndarray] = None  # (K, 3) linear colours of the groups (``color_groups``); None: found here
    group_count: int = 4
    selected: Sequence[bool] = ()  # which groups; none ticked: as "auto"
    picks: Sequence = ()  # eyedropper: linear colours
    tolerance: float = 0.15  # eyedropper: OKLab distance to a picked colour


@dataclass
class Part:
    mesh: MeshData
    options: PartOptions = field(default_factory=PartOptions)

    @property
    def has_lines(self) -> bool:
        o = self.options
        return o.role != "fill" and (o.silhouette or o.relief or o.creases or o.borders or o.colors)

    @property
    def has_fill(self) -> bool:
        return self.options.role != "outline"


@dataclass
class ColorGroups:
    linear: np.ndarray  # (K, 3) mean colour of each group
    lab: np.ndarray  # (K, 3) its centre in OKLab
    share: np.ndarray  # (K,) share of the surface, the largest first


def color_groups(mesh: MeshData, count: int = 4, samples: int = 20000, seed: int = 0) -> ColorGroups:
    """The main colours of the model: k-means in OKLab of colours all over its surface."""
    tris = mesh.triangles
    areas, _ = triangle_areas_normals(tris)
    rng = np.random.default_rng(seed)
    _, tri, bary = surface_candidates(tris, areas, int(samples), rng)
    lin = _color.surface_colors(mesh, tri, bary)
    lab = _color.linear_to_oklab(lin)
    centres, label = _kmeans(lab, max(int(count), 1), rng)
    k = len(centres)
    share = np.bincount(label, minlength=k) / len(label)
    mean = np.stack([np.bincount(label, weights=lin[:, c], minlength=k) for c in range(3)], 1)
    mean /= np.maximum(np.bincount(label, minlength=k), 1)[:, None]
    order = np.argsort(-share, kind="stable")
    return ColorGroups(mean[order], centres[order], share[order])


def auto_selected(groups: ColorGroups, contrast: float) -> np.ndarray:
    """The groups that stand out from the main colour (the largest group) by ``contrast`` or more."""
    far = np.linalg.norm(groups.lab - groups.lab[:1], axis=1) >= max(float(contrast), 0.05)
    far[0] = False
    if not far.any():
        far[1:] = True
    return far


def compose(parts: List[Part], options: DistributeOptions = None) -> DroneResult:
    """Drones on ``parts`` (each a mesh and its role) by corners, lines and
    fill, at most ``options.count``; ``options.target_min_distance`` apart in
    the corners (in the model's units, or in the figure's with ``scale_mode``
    "height"). The meshes are only read.
    """
    opt = options or DistributeOptions()
    budget = int(opt.count)
    if budget < 1:
        raise ValueError("count must be at least 1")
    if not parts:
        raise ValueError("no parts to lay out")
    for p in parts:
        o = p.options
        if o.role not in ROLES or o.pattern not in PATTERNS or o.pick not in PICKS:
            raise ValueError(f"unknown part options {o.role!r} / {o.pattern!r} / {o.pick!r}")
        if not p.mesh.triangle_count:
            raise ValueError("a part has no triangles")
    if opt.color_mode not in ("area", "point", "none"):
        raise ValueError(f"unknown color mode {opt.color_mode!r}")
    rng = np.random.default_rng(int(opt.seed))

    mesh = MeshData.concatenate([p.mesh for p in parts])
    offsets = np.cumsum([0] + [p.mesh.triangle_count for p in parts])
    tris = mesh.triangles
    areas, normals = triangle_areas_normals(tris)
    weights, audience, density = _audience(tris, areas, normals, opt)
    total_area = float(weights.sum())
    lo, hi = tris.reshape(-1, 3).min(axis=0), tris.reshape(-1, 3).max(axis=0)
    scale = 1.0
    if opt.scale_mode == "height":
        h = float(hi[opt.up_axis] - lo[opt.up_axis])
        if h <= 0:
            raise ValueError("cannot scale: the model is flat along the up axis")
        scale = float(opt.target_height) / h
    d0 = float(opt.target_min_distance) / scale
    if not d0 > 0:
        raise ValueError("the minimum distance must be positive")
    bodies = _bodies(parts, tris, offsets, areas, opt.fill_ratio * d0, opt.up_axis)
    if opt.facade:
        return _compose_picture(parts, opt, mesh, offsets, normals, weights, audience, d0, scale, budget, bodies)

    found = {}  # the lines are read again only when their spacing changes much
    s1, sf1 = opt.line_ratio * d0, opt.fill_ratio * d0
    creases = _creases(parts, opt, mesh, offsets, s1, rng)
    # a quick look from the middle viewer: the drones the figure needs, so a figure needing far more than
    # the budget is never laid out at the spacing asked for
    ext = np.sort(np.ptp(tris.reshape(-1, 3), axis=0))
    quick = _pic.Picture(mesh, offsets, opt, max(s1 / _pic.PX, math.sqrt(float(ext[1] * ext[2]) / ESTIMATE_PIXELS)))
    _, q_length = _pic.estimate(quick, len(parts), max(0.5 * s1, 2.0 * quick.h))
    del quick
    fill_area = np.array([float(weights[offsets[k]:offsets[k + 1]].sum()) if p.has_fill else 0.0
                          for k, p in enumerate(parts)])
    share = np.array([_pic.SHARE[p.options.role] if p.has_lines else 0.0 for p in parts])
    sil_on = np.array([p.has_lines and p.options.silhouette for p in parts])

    def guess(f):
        n_sil = np.where(sil_on, q_length / (s1 * lines_factor(f)), 0.0)
        n_fill = fill_area / (HEX * (sf1 * f) ** 2)
        return float((n_sil + n_fill + INNER_USE * share / (1.0 - share) * (n_sil + n_fill)).sum())

    lo_f, hi_f = 1.0, 2.0
    while guess(hi_f) > budget and hi_f < 1e6:
        lo_f, hi_f = hi_f, hi_f * 2.0
    for _ in range(40):
        mid = math.sqrt(lo_f * hi_f)
        lo_f, hi_f = (mid, hi_f) if guess(mid) > budget else (lo_f, mid)
    f_first = 1.0 if guess(1.0) <= EXACT_NEED * budget else hi_f

    # candidates: area of the surface, its colours, room for a blue-noise fill
    # drones the surface holds at the minimum distance, as far apart as the quick look expects
    most = total_area / (HEX * (d0 * lines_factor(f_first)) ** 2)
    wanted = int(min(max(CANDIDATES * max(opt.oversample, 2.0) / 8.0 * most, 4000), MAX_CANDIDATES))
    pos, tri, bary, rho, mean_rho = _gather_candidates(tris, weights, wanted, rng, density)
    top = float(rho.max())
    area = total_area * mean_rho / top  # density-weighted area: each candidate stands for area / len(pos)
    stretch = None if rho.min() == top else spacing_scale(rho / top)
    relative = None if density is None else (lambda p, t: density(p, t) / top)  # noqa: E731
    owner = np.searchsorted(offsets, tri, side="right") - 1  # the part of each candidate
    cand_lin = _color.surface_colors(mesh, tri, bary) if opt.color_mode != "none" or any(
        p.options.role == "pattern" for p in parts) else None
    regions = [_region(p.options, cand_lin[owner == k] if cand_lin is not None else None, opt.color_contrast)
               if p.options.role == "pattern" else None for k, p in enumerate(parts)]
    in_region = np.ones(len(pos), bool)
    for k, sel in enumerate(regions):
        if sel is not None:
            in_region[owner == k] = sel


    def plan(f):
        """Lines and fill at ``f`` times the spacings asked for."""
        fl = lines_factor(f)
        near, s_line, s_fill = d0 * fl * CLEAR, opt.line_ratio * d0 * fl, opt.fill_ratio * d0 * f
        if "at" not in found or not 1.0 / REDETECT_360 <= fl / found["at"] <= REDETECT_360:
            found["reading"] = _views_read(parts, opt, mesh, offsets, normals, s_line, bodies)
            found["at"] = fl
        lines, line_parts = _views_place(found["reading"], parts, s_line, near, s_fill)
        lines, line_parts = _with_creases(creases, lines, line_parts, near, s_line, tris)
        fills = [_plan_fill(k, p, ln, lp, s_fill, opt, tris, normals, weights, offsets, audience, pos, tri, owner,
                            in_region, area, bodies.get(k))
                 if p.has_fill else None for k, (p, ln, lp) in enumerate(zip(parts, lines, line_parts))]
        n_line = sum(len(lp[0]) for lp in line_parts if lp is not None)
        n_fill = sum(fl["count"] for fl in fills if fl is not None)
        return dict(f=f, s_fill=s_fill, lines=lines, line_parts=line_parts, fills=fills, n_line=n_line,
                    n_fill=n_fill, total=n_line + n_fill)

    # the spacing factor: 1 when the figure fits the budget, else the smallest that fits (closely)
    best, last, needed = None, None, None
    f, over, under = f_first, None, None
    for round_ in range(ROUNDS):
        last = plan(f)
        if f == 1.0:
            needed = last["total"]
        if last["total"] <= budget:
            if best is None or last["total"] > best["total"]:
                best = last
            if f == 1.0 or last["total"] >= FIT * budget:
                break
            under = f
        else:
            over = f
        if over is not None and under is not None:
            if under / over < 1.01:
                break
            f = math.sqrt(over * under)
        else:  # lines take ~1/f drones and the fill ~1/f^2: the factor where they would fit
            a, b = last["n_line"] * f, last["n_fill"] * f * f
            g = (a + math.sqrt(a * a + 4.0 * b * budget)) / (2.0 * budget)
            f = max(g, f * 1.02) if last["total"] > budget else max(min(g, f / 1.02), 1.0)
    if needed is None:
        needed = int(round(guess(1.0)))
    plan_ = best or last  # none fitted: the last, trimmed below
    f, s_fill, lines, line_parts, fills = (plan_[k] for k in ("f", "s_fill", "lines", "line_parts", "fills"))

    # the drones, corners first, then the lines, then the fill
    p_pos, p_tri, p_bary, p_kind, p_part, p_line_color = [], [], [], [], [], []
    for k, lp in enumerate(line_parts):
        if lp is None:
            continue
        lpos, ltri, lbary, lkind, _ = lp
        p_pos.append(lpos)
        p_tri.append(ltri)
        p_bary.append(lbary)
        p_kind.append(lkind)
        p_part.append(np.full(len(lpos), k))
        p_line_color.append(lines[k].colors_at(lpos))
    fixed = np.concatenate(p_pos) if p_pos else np.zeros((0, 3))
    for k, fl in enumerate(fills):
        if fl is None or not fl["count"]:
            continue
        fpos, ftri, fbary = _make_fill(fl, s_fill, opt, tris, weights, offsets, k, fixed, stretch, relative,
                                       (pos, tri, bary))
        p_pos.append(fpos)
        p_tri.append(ftri)
        p_bary.append(fbary)
        p_kind.append(np.full(len(fpos), 2, np.int8))
        p_part.append(np.full(len(fpos), k))
        p_line_color.append(np.full((len(fpos), 3), np.nan))
    if not p_pos or not sum(len(x) for x in p_pos):
        raise ValueError("nothing to place: no lines were found and no part is filled")
    points, p_tri, p_bary = np.concatenate(p_pos), np.concatenate(p_tri), np.concatenate(p_bary)
    kind, part, marked = np.concatenate(p_kind), np.concatenate(p_part), np.concatenate(p_line_color)
    order = np.argsort(kind, kind="stable")  # corners, lines, fill: the order in which they keep their places
    points, p_tri, p_bary, kind, part, marked = (x[order] for x in (points, p_tri, p_bary, kind, part, marked))
    keep = _outline.park(points, d0 * lines_factor(f))  # nobody closer than the minimum distance, whatever happened
    keep = keep[:budget]
    points, p_tri, p_bary, kind, part, marked = (x[keep] for x in (points, p_tri, p_bary, kind, part, marked))
    n = len(points)
    if f == 1.0:
        needed = n

    if opt.color_mode == "area" and cand_lin is not None:
        _, nearest_drone = nearest(pos, points, 0.5 * opt.fill_ratio * d0 * f)
        colors = _color.average_by_owner(cand_lin, nearest_drone, n)
    elif opt.color_mode != "none":
        colors = _color.surface_colors(mesh, p_tri, p_bary)
    else:
        colors = np.ones((n, 3))
    if opt.color_mode != "none":
        on_line = kind < 2
        under = _color.surface_colors(mesh, p_tri[on_line], p_bary[on_line])
        colors[on_line] = np.where(np.isnan(marked[on_line]), under, marked[on_line])
    fill = kind == 2
    if fill.any() and (~fill).any():  # the lines draw the figure, the fill is a background to them
        colors[fill] = _color.srgb_to_linear(_color.linear_to_srgb(colors[fill]) * FILL_GLOW)
    colors = _color.glow(colors, opt.min_brightness)

    dmin, dmean = spacing(points, d0 * f * opt.fill_ratio)
    plo, phi = points.min(axis=0), points.max(axis=0)
    line_spacing = opt.line_ratio * d0 * lines_factor(f) if (kind < 2).any() else 0.0
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
        pivot=scale_pivot(plo, phi, opt.up_axis),
        options=opt,
        visibility=audience.visibility(points, p_tri) if audience is not None else np.ones(n),
        outline=kind < 2,
        outline_spacing=line_spacing,
        kind=kind,
        part=part,
        budget=budget,
        needed=needed,
        spread=lines_factor(f),
        fill_spread=f,
    )


def read_picture(parts, opt, mesh, offsets, normals, s: float, sf: float, round_tri, simple, eyes: float = 0.0):
    """The picture the middle viewer sees, read at line spacing ``s`` (fill spacing ``sf``): its
    silhouettes, features, inner lines, dots and the regions the fill may take (see ``picture``). With
    ``eyes`` (a line spacing) also eyeballs sculpted in the colour of the face (pairs of bumps of about that
    scale), on the parts with simple shapes."""
    tris = mesh.triangles
    lines_on = [p.has_lines and p.options.silhouette for p in parts]
    fill_on = [p.has_fill for p in parts]
    pic = _pic.Picture(mesh, offsets, opt, max(s / _pic.PX, _detail_floor(tris)))
    kinds = set()
    for p in parts:
        o = p.options
        if p.has_lines:
            if o.relief or o.creases:
                kinds.add("relief")
            if o.colors:
                kinds.update(("colors", "spots"))
            if o.silhouette or o.borders:
                kinds.add("occlusion")
    if eyes and any(k in simple and (p.options.relief or p.options.creases) for k, p in enumerate(parts)):
        kinds.add("bumps")
    found = _pic.fields(pic, mesh, normals, opt, s, kinds, SPOT_OCTAVES if simple else 1, eyes) if kinds else {}
    lin = np.ones(pic.shape + (3,))
    lin[pic.fg] = _color.surface_colors(mesh, pic.tri[pic.fg], pic.bary[pic.fg])
    label = _pic.color_regions(pic, lin, opt, s) if "colors" in kinds else None
    sil, inner, dots, allowed, spots, pending, feats = [], [], [], [], [], {}, []
    for k, p in enumerate(parts):
        o = p.options
        if lines_on[k]:
            st, dt = _pic.silhouettes(pic, k, s)
            sil += st
            dots += dt
        if p.has_lines and (o.colors or o.relief or o.creases):
            feats += _pic.feature_loops(pic, k, label if o.colors else None, lin, s, opt.color_contrast,
                                        depth=o.relief or o.creases)
        if p.has_lines:
            mine = {kind: found[kind] for kind in found
                    if (kind == "relief" and (o.relief or o.creases)) or (kind in ("colors", "spots") and o.colors)
                    or (kind == "occlusion" and (o.silhouette or o.borders))
                    or (kind == "bumps" and k in simple and (o.relief or o.creases))}
            st, dt = _pic.inner_lines(pic, k, mine, s, label, opt.color_contrast, lin)
            inner += st
            dots += dt
        mask = raw = pic.part == k
        if fill_on[k] and o.role == "pattern" and mask.any():
            sel = _region(o, lin[mask], opt.color_contrast)
            region = np.zeros(pic.shape)
            region[mask] = sel
            raw = region > 0.5  # a spot of the colour too small for the lattice still takes a drone
            sigma = max(0.5 * VOTE * sf / pic.h, 0.5)
            mask = mask & (_outline._blur(region, sigma) > 0.5 * _outline._blur(mask.astype(np.float64), sigma))
        allowed.append(mask if fill_on[k] else None)
        spots.append(raw if fill_on[k] else None)
    for k in round_tri:  # a round body is drawn by its rings: its relief (ribs, seams, facets) is no line
        body = np.isin(pic.tri, round_tri[k])
        inner = [st for st in inner if not (st.part == k and st.kind == "relief"
                                            and pic.sample(body, st.xy, False).mean() > 0.5)]
        pending[("round", k)] = body
    sil.sort(key=lambda x: -x.length)
    inner.sort(key=lambda x: -x.score)
    out = dict(pic=pic, lin=lin, sil=sil, inner=inner, dots=dots, allowed=allowed, spots=spots, feats=feats)
    out.update(pending)
    return out


def drawing(figure, parts, s: float, sf: float, near: float, simple):
    """The strokes a reading of the picture is drawn with at line spacing ``s`` (corners marked) and its
    dots as (picture point, part): the silhouettes, the features, the inner lines that fit their share."""
    n_parts = len(parts)
    pic = figure["pic"]
    fill_on = [p.has_fill for p in parts]
    share = np.array([_pic.SHARE[p.options.role] if p.has_lines else 0.0 for p in parts])
    area = np.bincount(pic.part[pic.fg], minlength=n_parts) * pic.h * pic.h
    sil_len = np.zeros(n_parts)
    for st in figure["sil"]:
        sil_len[st.part] += st.length
    n_fill = np.where(fill_on, np.maximum(area - sil_len * _pic.FILL_GAP * sf, 0.0) / (HEX * sf * sf), 0.0)
    allow = dict(enumerate((share / np.maximum(1.0 - share, 1e-9) * (sil_len / s + n_fill)).tolist()))
    taken, dots = _pic.choose(figure["sil"], figure["inner"], figure["dots"], s, sf, allow, simple, figure["feats"])
    if simple:
        taken, dots = _pic.simplify(taken, dots, pic, figure["lin"], near, s, simple)
    else:
        dots = [(xy, part) for xy, part, _ in dots]
    _pic.corners(taken, s)
    return taken, dots


def _compose_picture(parts, opt, mesh, offsets, normals, weights, audience, d0, scale, budget, bodies):
    """``compose`` in facade mode: the picture the audience sees, drawn by structure and salience
    (see ``picture``). The spacing factor is estimated from a small picture first, so a figure that
    needs far more drones than the budget is never laid out at the spacing asked for."""
    n_parts = len(parts)
    tris = mesh.triangles
    s1, sf1 = opt.line_ratio * d0, opt.fill_ratio * d0
    simple = {k for k, p in enumerate(parts) if p.options.shapes}
    round_tri = {k: np.concatenate([b.tri for b in bs]) for k, bs in bodies.items()}
    lines_on = [p.has_lines and p.options.silhouette for p in parts]
    fill_on = [p.has_fill for p in parts]
    share = np.array([_pic.SHARE[p.options.role] if p.has_lines else 0.0 for p in parts])
    # where nobody sees the model, a lattice ``hidden_density`` as dense as the fill (it stays whole when it turns)
    hidden = min(max(float(opt.hidden_density), 0.0), 1.0) if audience is not None else 0.0
    hid_ids, hid_area = np.zeros(0, np.int64), 0.0
    if hidden > 0:
        fill_tri = np.zeros(len(tris), bool)
        for k, p in enumerate(parts):
            fill_tri[offsets[k]:offsets[k + 1]] = p.has_fill
        areas = triangle_areas_normals(tris)[0]
        for t in round_tri.values():  # round bodies take rings all round
            fill_tri[t] = False
        hid_ids = np.flatnonzero(fill_tri & (areas > 0))
        hid_area = float(areas[fill_tri].sum()) - float(weights[fill_tri].sum()) * HIDDEN_SEEN

    def n_hidden(f):
        return hidden * hid_area / (HEX * (sf1 * f) ** 2)

    # a quick look: area and silhouette of every part in a small picture, and the drones they need
    ext = np.sort(np.ptp(tris.reshape(-1, 3), axis=0))
    h0 = max(s1 / _pic.PX, math.sqrt(float(ext[1] * ext[2]) / ESTIMATE_PIXELS))
    quick = _pic.Picture(mesh, offsets, opt, h0)
    q_area, q_length = _pic.estimate(quick, n_parts, max(0.5 * s1, 2.0 * quick.h))
    del quick

    def guess(f):
        s, sf = s1 * lines_factor(f), sf1 * f
        n_sil = np.where(lines_on, q_length / s, 0.0)
        band = np.where(lines_on, q_length * _pic.FILL_GAP * sf, 0.0)
        n_fill = np.where(fill_on, np.maximum(q_area - band, 0.0) / (HEX * sf * sf), 0.0)
        n_inner = INNER_USE * share / (1.0 - share) * (n_sil + n_fill)
        return float((n_sil + n_fill + n_inner).sum()) + n_hidden(f)

    lo_f, hi_f = 1.0, 2.0
    while guess(hi_f) > budget and hi_f < 1e6:
        lo_f, hi_f = hi_f, hi_f * 2.0
    for _ in range(40):
        mid = math.sqrt(lo_f * hi_f)
        lo_f, hi_f = (mid, hi_f) if guess(mid) > budget else (lo_f, mid)
    f_guess = 1.0 if guess(1.0) <= budget else hi_f
    exact = guess(1.0) <= EXACT_NEED * budget  # the spacing asked for is cheap to lay out

    figure = {}

    def detail(s):
        return max(s / _pic.PX, _detail_floor(tris))

    def detect(f):
        """Lines and fill regions of the picture read at ``f`` times the spacings asked for."""
        reading = read_picture(parts, opt, mesh, offsets, normals, s1 * lines_factor(f), sf1 * f, round_tri, simple)
        figure.clear()  # a new reading: the plans made from the old one keep theirs
        figure.update(reading, f=f)

    def plan(f, reading=None):
        """Lines and fill at ``f`` times the spacings asked for, from ``reading`` of the picture (by
        default the last one, read again when the spacing changed much)."""
        nonlocal figure
        if reading is not None:
            figure = reading
        elif "f" not in figure or not (1.0 / REDETECT <= lines_factor(f) / lines_factor(figure["f"]) <= REDETECT):
            figure = {}
            detect(f)
        pic = figure["pic"]
        s, sf, near = s1 * lines_factor(f), sf1 * f, d0 * lines_factor(f) * CLEAR
        taken, dots = drawing(figure, parts, s, sf, near, simple)
        xy, kind, st_idx, sigma = _pic.place(taken, near, s, simple)
        l_part = np.array([taken[i].part for i in st_idx], np.int64)
        if dots:
            d_xy = np.array([d[0] for d in dots])
            keep = ~_pic._near(d_xy, xy, max(near, _pic.LINE_GAP * s)) if len(xy) else np.ones(len(d_xy), bool)
            xy = np.vstack([xy, d_xy[keep]])
            kind = np.r_[kind, np.ones(int(keep.sum()), np.int8)]
            st_idx = np.r_[st_idx, np.full(int(keep.sum()), -1)]
            l_part = np.r_[l_part, np.array([d[1] for d in dots], np.int64)[keep]]
        dense = _pic._dense(taken)
        fill_xy, fill_part = [], []
        for k in range(n_parts):
            if figure["allowed"][k] is None:
                continue
            clear = np.vstack([xy] + fill_xy) if fill_xy else xy
            allowed, spots = figure["allowed"][k], figure["spots"][k]
            r = np.zeros((0, 2))
            if k in round_tri:  # round bodies: rings from the bottom up, the lattice on the rest
                key = ("round", k)
                if key not in figure:
                    figure[key] = np.isin(pic.tri, round_tri[k])
                allowed, spots = allowed & ~figure[key], spots & ~figure[key]
                r = _ring_fill(pic, bodies[k], k, figure["allowed"][k], s, sf, tris, normals, opt, dense, clear)
                clear = np.vstack([clear, r])
            p = _pic.lattice(pic, allowed, sf, parts[k].options.pattern, dense, _pic.FILL_GAP * sf,
                             clear, _pic.FILL_GAP * sf) if allowed.any() else np.zeros((0, 2))
            if parts[k].has_lines:
                p = np.vstack([p, _pic.pockets(pic, spots, sf, dense, _pic.FILL_GAP * sf, np.vstack([clear, p]))])
            p = np.vstack([r, p])
            fill_xy.append(p)
            fill_part.append(np.full(len(p), k))
        fill_xy = np.vstack(fill_xy) if fill_xy else np.zeros((0, 2))
        fill_part = np.concatenate(fill_part) if fill_part else np.zeros(0, np.int64)
        return dict(f=f, reading=figure, taken=taken, xy=xy, kind=kind, st_idx=st_idx, part=l_part, fill_xy=fill_xy,
                    fill_part=fill_part, n_line=len(xy), n_fill=len(fill_xy) + n_hidden(f),
                    total=len(xy) + len(fill_xy) + int(round(n_hidden(f))))

    # the spacing factor: 1 when the figure fits the budget, else the smallest that fits (closely)
    best, last = None, None
    f = 1.0 if exact else f_guess
    needed = None
    over, under = None, None
    for round_ in range(ROUNDS):
        last = plan(f)
        if f == 1.0:
            needed = last["total"]
        if last["total"] <= budget:
            if best is None or last["total"] > best["total"]:
                best = last
            if f == 1.0 or last["total"] >= FIT * budget:
                break
            under = f
        else:
            over = f
        if over is not None and under is not None:
            if under / over < 1.01:
                break
            f = math.sqrt(over * under)
        else:  # lines take ~1/f drones and the fill ~1/f^2: the factor where they would fit
            a, b = last["n_line"] * f, last["n_fill"] * f * f
            g = (a + math.sqrt(a * a + 4.0 * b * budget)) / (2.0 * budget)
            f = max(g, f * 1.02) if last["total"] > budget else max(min(g, f / 1.02), 1.0)
    if best is not None and best["total"] < FIT * budget and best["f"] > 1.0:
        # reading the picture again at another spacing may change the drawing (a comb of thin rays
        # merges into a disc): the best reading, a little denser, may fill the budget better
        lo_, hi_ = max(best["f"] / REDETECT, 1.0), best["f"]
        for _ in range(REFINE):
            mid = math.sqrt(lo_ * hi_)
            trial = plan(mid, best["reading"])
            if trial["total"] <= budget:
                hi_ = mid
                if trial["total"] > best["total"]:
                    best = trial
                if trial["total"] >= FIT * budget:
                    break
            else:
                lo_ = mid
    if needed is None:
        needed = int(round(guess(1.0)))
    plan_ = best or last
    f = plan_["f"]
    figure = plan_["reading"]
    pic, taken = figure["pic"], plan_["taken"]
    s, sf = s1 * lines_factor(f), sf1 * f

    # onto the surface: corners, lines, then the fill
    l_pos, l_tri, l_bary, l_found = pic.lift(plan_["xy"], plan_["part"], max(3.0, 0.6 * _pic.THIN * s / pic.h))
    f_pos, f_tri, f_bary, f_found = pic.lift(plan_["fill_xy"], plan_["fill_part"], 1.5)
    xy = np.vstack([plan_["xy"][l_found], plan_["fill_xy"][f_found]])
    points = np.vstack([l_pos, f_pos])
    p_tri, p_bary = np.r_[l_tri, f_tri], np.vstack([l_bary, f_bary])
    kind = np.r_[plan_["kind"][l_found], np.full(len(f_pos), 2, np.int8)]
    part = np.r_[plan_["part"][l_found], plan_["fill_part"][f_found]]
    st_idx = np.r_[plan_["st_idx"][l_found], np.full(len(f_pos), -1)]
    radius = np.full(len(points), d0 * lines_factor(f))
    if hidden > 0 and (len(hid_ids) or bodies):
        hp, ht, hb, _, phase = _lattice(hid_ids, tris, normals, replace(opt, facade=False), sf / math.sqrt(hidden),
                                        "hex", None)
        unseen = ~pic.shows(hp, ht, normals) if len(hp) else np.zeros(0, bool)
        far = ~_outline._near(hp, points, SEAM * sf) if len(hp) and len(points) else unseen
        ph = int(np.argmax(np.bincount(phase[unseen & far], minlength=PHASES * PHASES))) if len(hp) else 0
        sel = unseen & (phase == ph)  # the shift of the lattice that keeps the most points
        hp, ht, hb = hp[sel], ht[sel], hb[sel]
        for k, bs in bodies.items():  # the rings go on round the back, sparser
            for b in bs:
                rp, rt, rb, _ = _shapes.rings(tris, b.tri, normals, sf, pic.toward, opt.up_axis, s / hidden)
                if len(rp):
                    back = ~pic.shows(rp, rt, normals)
                    hp, ht, hb = np.vstack([hp, rp[back]]), np.r_[ht, rt[back]], np.vstack([hb, rb[back]])
        xy = np.vstack([xy, pic.project(hp)])
        points, p_tri, p_bary = np.vstack([points, hp]), np.r_[p_tri, ht], np.vstack([p_bary, hb])
        kind = np.r_[kind, np.full(len(hp), 2, np.int8)]
        part = np.r_[part, np.searchsorted(offsets, ht, side="right") - 1]
        st_idx = np.r_[st_idx, np.full(len(hp), -2)]  # -2: nobody sees it
        radius = np.r_[radius, np.full(len(hp), max(d0 * lines_factor(f), SEAM * sf))]
    if not len(points):
        raise ValueError("nothing to place: no lines were found and no part is filled")
    order = np.argsort(kind, kind="stable")
    xy, points, p_tri, p_bary, kind, part, st_idx, radius = (a[order] for a in (xy, points, p_tri, p_bary, kind,
                                                                                 part, st_idx, radius))
    keep = _outline.park(points, radius)[:budget]  # nobody closer than the minimum distance, whatever happened
    xy, points, p_tri, p_bary, kind, part, st_idx = (a[keep] for a in (xy, points, p_tri, p_bary, kind, part,
                                                                       st_idx))
    n = len(points)
    if f == 1.0:
        needed = n

    # colours: the patch of the picture around each drone, a colour border's drone the side that stands out
    lin = figure["lin"]
    if opt.color_mode == "none":
        colors = np.ones((n, 3))
    else:
        under = _color.surface_colors(mesh, p_tri, p_bary)
        colors = under.copy()
        if opt.color_mode == "area":
            for kk in range(n_parts):
                mine = part == kk
                fill = mine & (kind == 2) & (st_idx != -2)
                if fill.any():
                    colors[fill] = _pic.colors_near(pic, lin, xy[fill], kk, 0.5 * sf)
                line = mine & (kind < 2)
                if line.any():
                    colors[line] = _pic.colors_near(pic, lin, xy[line], kk, 0.4 * s)
        on_border = np.array([i >= 0 and taken[i].kind == "colors" and taken[i].color is None for i in st_idx],
                             bool)
        if on_border.any():
            for kk in np.unique(part[on_border]):
                sel = on_border & (part == kk)
                mask = pic.part == kk
                mean = _color.linear_to_oklab(lin[mask]).mean(axis=0)
                colors[sel] = _pic.colors_near(pic, lin, xy[sel], kk, 0.5 * s, np.broadcast_to(mean, (sel.sum(), 3)))
        own = np.array([i >= 0 and taken[i].color is not None for i in st_idx], bool)
        if own.any():  # a band shows its own colour
            colors[own] = np.array([taken[i].color for i in st_idx[own]])
        colors = np.where(np.isnan(colors), under, colors)
    colors = _color.glow(colors, opt.min_brightness)

    dmin, dmean = spacing(points, sf)
    plo, phi = points.min(axis=0), points.max(axis=0)
    return DroneResult(
        positions=points,
        normals=normals[p_tri],
        triangle_ids=p_tri,
        barycentrics=p_bary,
        colors_linear=colors,
        min_distance=dmin,
        mean_distance=dmean,
        area=float(weights.sum()),
        scale=scale,
        pivot=scale_pivot(plo, phi, opt.up_axis),
        options=opt,
        visibility=audience.visibility(points, p_tri) if audience is not None else np.ones(n),
        outline=kind < 2,
        outline_spacing=s if (kind < 2).any() else 0.0,
        kind=kind,
        part=part,
        budget=budget,
        needed=int(needed),
        spread=lines_factor(f),
        fill_spread=f,
    )


class _Drawn:
    """The lines of a part drawn from views all round: their points on the surface (``pos``, the fill keeps
    clear of them) and the colour of each of their drones."""

    def __init__(self, pos, drones, colors):
        self.pos, self.drones, self.colors = pos, drones, colors

    def colors_at(self, points):
        if not len(self.drones):
            return np.full((len(points), 3), np.nan)
        _, idx = nearest(points, self.drones, 1e-6)
        return self.colors[idx]


def _views_read(parts, opt, mesh, offsets, normals, s: float, bodies, eyes: float = 0.0, real: bool = False):
    """The drawing of a figure seen from all round: read (as in ``picture``) from ``VIEWS_360`` views round
    the vertical axis, each drawing the lines on the surface that faces it most; the silhouette is the
    middle viewer's (``eyes`` as in ``read_picture``; with ``real`` the first view is the middle viewer
    as the options have it, from ``view_point``). Returns the readings, for ``_views_place``."""
    tris = mesh.triangles
    up = np.eye(3)[opt.up_axis]
    corners = tris.reshape(-1, 3)
    center = 0.5 * (corners.min(axis=0) + corners.max(axis=0))
    middle = np.asarray(opt.view_point, np.float64) - center if opt.view_point is not None else normalize(opt.view_dir)
    middle = middle - (middle @ up) * up
    middle = normalize(middle) if np.linalg.norm(middle) > 1e-9 else view_basis(up)[1]
    dirs = [rotate(middle, up, 2.0 * math.pi * v / VIEWS_360) for v in range(VIEWS_360)]
    owner = np.argmax(np.stack([normals @ d for d in dirs], 1), axis=1)
    simple = {k for k, p in enumerate(parts) if p.options.shapes}
    round_tri = {k: np.concatenate([b.tri for b in bs]) for k, bs in bodies.items()}
    kinds = set()
    for p in parts:
        o = p.options
        if p.has_lines:
            if o.relief or o.creases:
                kinds.add("relief")
            if o.colors:
                kinds.update(("colors", "spots"))
            if o.silhouette or o.borders:
                kinds.add("occlusion")
    if eyes and any(k in simple and (p.options.relief or p.options.creases) for k, p in enumerate(parts)):
        kinds.add("bumps")
    views = []
    for v, d in enumerate(dirs):
        vopt = replace(opt, facade=True) if real and v == 0 else             replace(opt, view_dir=tuple(d), view_point=None, facade=True)
        try:
            pic = _pic.Picture(mesh, offsets, vopt, max(s / _pic.PX, _detail_floor(tris)))
        except ValueError:
            continue
        own = pic.fg & (owner[np.where(pic.fg, pic.tri, 0)] == v)
        found = _pic.fields(pic, mesh, normals, vopt, s, kinds, SPOT_OCTAVES if simple else 1, eyes) if kinds else {}
        lin = np.ones(pic.shape + (3,))
        lin[pic.fg] = _color.surface_colors(mesh, pic.tri[pic.fg], pic.bary[pic.fg])
        label = _pic.color_regions(pic, lin, vopt, s) if "colors" in kinds else None

        def owned(xy):
            return pic.sample(own, xy, False).mean() > 0.5 if len(xy) else False

        sil, inner, dots, feats = [], [], [], []
        for k, p in enumerate(parts):
            o = p.options
            if not p.has_lines:
                continue
            if v == 0 and o.silhouette:
                st, dt = _pic.silhouettes(pic, k, s)
                sil += st
                dots += dt
            if o.colors or o.relief or o.creases:
                feats += [x for x in _pic.feature_loops(pic, k, label if o.colors else None, lin, s,
                                                        opt.color_contrast, depth=o.relief or o.creases)
                          if owned(x.xy)]
            mine = {kind: found[kind] for kind in found
                    if (kind == "relief" and (o.relief or o.creases)) or (kind in ("colors", "spots") and o.colors)
                    or (kind == "occlusion" and (o.silhouette or o.borders))
                    or (kind == "bumps" and k in simple and (o.relief or o.creases))}
            st, dt = _pic.inner_lines(pic, k, mine, s, label, opt.color_contrast, lin)
            inner += [x for x in st if owned(x.xy)]
            dots += [x for x in dt if pic.sample(own, x[0][None], False)[0]]
        for k in round_tri:  # a round body is drawn by its rings: its relief is no line
            body = np.isin(pic.tri, round_tri[k])
            inner = [x for x in inner if not (x.part == k and x.kind == "relief"
                                              and pic.sample(body, x.xy, False).mean() > 0.5)]
        sil.sort(key=lambda x: -x.length)
        inner.sort(key=lambda x: -x.score)
        # the lines a view may take go by the whole figure it sees (only the middle view draws its outline)
        area, outline = _pic.estimate(pic, len(parts), s)
        views.append(dict(pic=pic, lin=lin, sil=sil, inner=inner, dots=dots, feats=feats, area=area,
                          outline=outline))
    return dict(views=views, simple=simple)


def _views_place(reading, parts, s: float, near: float, sf: float):
    """The line drones of a reading from views all round, placed at line spacing ``s`` and lifted onto
    the surface. Returns (lines, line drones) of each part, None for a part with no lines."""
    n = len(parts)
    simple = reading["simple"]
    share = np.array([_pic.SHARE[p.options.role] if p.has_lines else 0.0 for p in parts])
    acc = [dict(pos=[], tri=[], bary=[], kind=[], sigma=[], dense=[], color=[]) for _ in range(n)]
    for view in reading["views"]:
        pic, lin = view["pic"], view["lin"]
        sil_len = view["outline"]
        n_fill = np.maximum(view["area"] - sil_len * _pic.FILL_GAP * sf, 0.0) / (HEX * sf * sf)
        allow = dict(enumerate((share / np.maximum(1.0 - share, 1e-9) * (sil_len / s + n_fill)).tolist()))
        taken, dots = _pic.choose(view["sil"], view["inner"], view["dots"], s, sf, allow, simple, view["feats"])
        if simple:
            taken, dots = _pic.simplify(taken, dots, pic, lin, near, s, simple)
        else:
            dots = [(xy, part) for xy, part, _ in dots]
        _pic.corners(taken, s)
        xy, kind, st_idx, sigma = _pic.place(taken, near, s, simple)
        part = np.array([taken[i].part for i in st_idx], np.int64)
        if dots:
            d_xy = np.array([d[0] for d in dots])
            keep = ~_pic._near(d_xy, xy, max(near, _pic.LINE_GAP * s)) if len(xy) else np.ones(len(d_xy), bool)
            xy = np.vstack([xy, d_xy[keep]])
            kind = np.r_[kind, np.ones(int(keep.sum()), np.int8)]
            st_idx = np.r_[st_idx, np.full(int(keep.sum()), -1)]
            sigma = np.r_[sigma, np.full(int(keep.sum()), s)]
            part = np.r_[part, np.array([d[1] for d in dots], np.int64)[keep]]
        if not len(xy):
            continue
        pos, tri, bary, ok = pic.lift(xy, part, max(3.0, 0.6 * _pic.THIN * s / pic.h))
        xy, kind, st_idx, sigma, part = xy[ok], kind[ok], st_idx[ok], sigma[ok], part[ok]
        colors = np.full((len(xy), 3), np.nan)
        border = np.array([i >= 0 and taken[i].kind == "colors" and taken[i].color is None for i in st_idx], bool)
        for k in np.unique(part):
            sel = part == k
            colors[sel & ~border] = _pic.colors_near(pic, lin, xy[sel & ~border], k, 0.4 * s)
            if (sel & border).any():  # a colour border shows the side that stands out from the part's colour
                mean = _color.linear_to_oklab(lin[pic.part == k]).mean(axis=0)
                colors[sel & border] = _pic.colors_near(pic, lin, xy[sel & border], k, 0.5 * s,
                                                        np.broadcast_to(mean, (int((sel & border).sum()), 3)))
        own = np.array([i >= 0 and taken[i].color is not None for i in st_idx], bool)
        if own.any():  # a band, a ring shows its own colour
            colors[own] = np.array([taken[i].color for i in st_idx[own]])
        for st in taken:  # the lines themselves, for the fill to keep clear of
            dp, _, _, dok = pic.lift(st.xy, st.part, 3.0)
            acc[st.part]["dense"].append(dp)
        for k in np.unique(part):
            sel = part == k
            a = acc[k]
            for key, val in (("pos", pos[sel]), ("tri", tri[sel]), ("bary", bary[sel]), ("kind", kind[sel]),
                             ("sigma", sigma[sel]), ("color", colors[sel])):
                a[key].append(val)
    lines, line_parts = [None] * n, [None] * n
    for k, a in enumerate(acc):
        if not a["pos"]:
            continue
        pos, tri, bary = np.vstack(a["pos"]), np.concatenate(a["tri"]), np.vstack(a["bary"])
        kind, sigma, color = np.concatenate(a["kind"]).astype(np.int8), np.concatenate(a["sigma"]), np.vstack(a["color"])
        order = np.argsort(kind, kind="stable")  # the corners first
        pos, tri, bary, kind, sigma, color = (x[order] for x in (pos, tri, bary, kind, sigma, color))
        dense = np.vstack(a["dense"]) if a["dense"] else pos
        lines[k] = _Drawn(dense, pos, color)
        line_parts[k] = (pos, tri, bary, kind, sigma)
    return lines, line_parts


class _Shifted:
    """The audience of the whole scene seen from one part: its triangles are numbered from ``offset``."""

    def __init__(self, audience, offset: int):
        self.audience, self.offset = audience, int(offset)

    def visibility(self, points, tri):
        return self.audience.visibility(points, np.asarray(tri) + self.offset)


def _find(part: Part, opt, mesh: MeshData, offset: int, spacing: float, rng, audience):
    """The lines of one part, its triangles numbered as in the whole scene."""
    o = part.options
    popt = replace(opt, outline_silhouette=o.silhouette, outline_relief=o.relief, outline_creases=o.creases,
                   outline_borders=o.borders, outline_colors=o.colors)
    sub = part.mesh
    areas, normals = triangle_areas_normals(sub.triangles)
    lines = _outline.find_lines(sub, areas, normals, areas, popt, spacing, rng,
                                None if audience is None else _Shifted(audience, offset))
    if lines is None:
        return None
    lines.tri = lines.tri + offset
    for chain in lines.chains:
        chain.node_tri = chain.node_tri + offset
        if chain.tri is not None:
            chain.tri = chain.tri + offset
    node, c_pos, c_tri, c_bary = lines.corners
    lines.corners = (node, c_pos, c_tri + offset, c_bary)
    if lines.surface is not None:
        lines.surface.parent = lines.surface.parent + offset
    return lines


def _creases(parts, opt, mesh: MeshData, offsets, s: float, rng):
    """The sharp edges of each part's mesh (its faces meet at more than ``crease_angle``) at least
    ``CREASE_LONG`` line spacings ``s`` long, with the corners where they meet: the edges of a box are its
    lines from every side, though every view but one sees them as its outline. None for a part without."""
    out = []
    for k, p in enumerate(parts):
        lines = None
        if p.has_lines and p.options.creases:
            only = Part(p.mesh, replace(p.options, silhouette=False, relief=False, borders=False, colors=False))
            lines = _find(only, opt, mesh, int(offsets[k]), s, rng, None)
        if lines is not None:
            lines.chains = [c for c in lines.chains if c.length >= CREASE_LONG * s]  # a nick is no edge
            if lines.chains:
                nodes = np.vstack([c.p for c in lines.chains])
                on = _outline._near(lines.pos, nodes, 2.0 * lines.step)
                lines.pos, lines.tri, lines.bary, lines.color = (x[on] for x in (lines.pos, lines.tri, lines.bary,
                                                                                 lines.color))
                node, c_pos, c_tri, c_bary = lines.corners
                on = _outline._near(c_pos, nodes, 2.0 * lines.step)
                lines.corners = (node[on], c_pos[on], c_tri[on], c_bary[on])
                lines.strength = lines.strength[on]
            else:
                lines = None
        out.append(lines)
    return out


def _with_creases(creases, lines, line_parts, near: float, s: float, tris):
    """The lines read from the views with the sharp edges of the mesh (``_creases``) first: the line drones
    of the views on a sharp edge give way to its own (their corners stay)."""
    lines, line_parts = list(lines), list(line_parts)
    for k, c in enumerate(creases):
        if c is None:
            continue
        cpos, ctri, cbary, ckind, csigma = _place_lines(c, near, s, tris, None)
        dense, colors = c.pos, np.full((len(cpos), 3), np.nan)
        if line_parts[k] is not None:
            pos, tri, bary, kind, sigma = line_parts[k]
            off = (kind == 0) | ~_outline._near(pos, c.pos, _outline.LINE_GAP * s)
            cpos, ctri, cbary = np.vstack([cpos, pos[off]]), np.r_[ctri, tri[off]], np.vstack([cbary, bary[off]])
            ckind, csigma = np.r_[ckind, kind[off]], np.r_[csigma, sigma[off]]
            dense = np.vstack([dense, lines[k].pos])
            colors = np.vstack([colors, lines[k].colors_at(pos[off])])
        order = np.argsort(ckind, kind="stable")  # the corners first
        line_parts[k] = tuple(x[order] for x in (cpos, ctri, cbary, ckind, csigma))
        lines[k] = _Drawn(dense, line_parts[k][0], colors[order])
    return lines, line_parts


def _place_lines(lines, near: float, s_line: float, tris, audience):
    pos, tri, bary, kind, sigma = _outline.place_graded(lines, near, s_line, tris)
    if audience is not None and len(pos):  # another part may stand in front of the lines of this one
        seen = audience.visibility(pos, tri) > 0
        pos, tri, bary, kind, sigma = pos[seen], tri[seen], bary[seen], kind[seen], sigma[seen]
    return pos, tri, bary, kind, sigma


def _region(o: PartOptions, lin, contrast: float):
    """Which candidates of a pattern part get the fill (their colours ``lin``)."""
    if lin is None or not len(lin):
        return None if lin is None else np.zeros(0, bool)
    lab = _color.linear_to_oklab(lin)
    if o.pick == "eyedropper":
        if not len(o.picks):
            return np.zeros(len(lab), bool)
        picks = _color.linear_to_oklab(np.asarray(o.picks, np.float64).reshape(-1, 3))
        d = np.linalg.norm(lab[:, None, :] - picks[None, :, :], axis=2).min(axis=1)
        return d <= o.tolerance
    if o.groups is not None and len(o.groups):
        centres = _color.linear_to_oklab(np.asarray(o.groups, np.float64).reshape(-1, 3))
        label = _nearest_centre(lab, centres)
        share = np.bincount(label, minlength=len(centres)) / len(label)
        groups = ColorGroups(np.asarray(o.groups, np.float64).reshape(-1, 3), centres, share)
    else:
        centres, label = _kmeans(lab, max(int(o.group_count), 1), np.random.default_rng(0))
        share = np.bincount(label, minlength=len(centres)) / len(label)
        order = np.argsort(-share, kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        centres, label, share = centres[order], rank[label], share[order]
        groups = ColorGroups(centres, centres, share)
    ticked = np.zeros(len(groups.lab), bool)
    ticked[: len(o.selected)] = np.asarray(o.selected, bool)[: len(ticked)]
    if o.pick == "auto" or not ticked.any():
        if o.groups is not None and len(o.groups) and o.pick == "auto":
            # the main colour is the one most of the surface has, wherever the groups came from
            main = int(np.argmax(groups.share))
            far = np.linalg.norm(groups.lab - groups.lab[main], axis=1) >= max(float(contrast), 0.05)
            far[main] = False
            ticked = far if far.any() else np.arange(len(ticked)) != main
        else:
            ticked = auto_selected(groups, contrast)
    return ticked[label]


def _plan_fill(k, part, lines, line_part, s_fill, opt, tris, normals, weights, offsets, audience, pos, tri, owner,
               in_region, area, bodies=None):
    """Where the fill of part ``k`` goes and how many drones it takes; its round ``bodies`` take rings."""
    mine = owner == k
    if part.options.pattern == "noise":
        free = mine & in_region
        if lines is not None:
            free &= ~_outline._near(pos, lines.pos, GAP * s_fill)
        share = float(free.sum()) / len(pos)
        fill_area = area * share
        count = int(round(fill_area / (HEX * s_fill * s_fill))) if free.sum() >= 2 else 0
        return dict(kind="noise", free=free, area=fill_area, count=min(count, int(free.sum())))
    ids = np.arange(offsets[k], offsets[k + 1])
    ids = ids[weights[ids] > 0]
    if bodies:
        ids = ids[~np.isin(ids, np.concatenate([b.tri for b in bodies]))]
    # the lattice PHASES times finer: each of its PHASES^2 sub-lattices is the lattice shifted, and the
    # shift that keeps the most points clear of the lines is taken
    gp, gt, gb, gq, phase = _lattice(ids, tris, normals, opt, s_fill, part.options.pattern, audience)
    keep = np.ones(len(gp), bool)
    if len(gp) and lines is not None:
        keep &= ~_outline._near(gp, lines.pos, GAP * s_fill)
    if len(gp) and line_part is not None and len(line_part[0]):
        keep &= ~_outline._near(gp, line_part[0], 0.5 * s_fill)
    if keep.any() and part.options.role == "pattern":
        keep[keep] = _vote(gp[keep], pos[mine], in_region[mine], VOTE * s_fill)
    best = np.zeros(0, np.int64)
    for ph in range(PHASES * PHASES):
        at = np.flatnonzero(keep & (phase == ph))
        # a point on an edge between two triangles is found twice; two projections meet at a seam
        at = at[np.argsort(-gq[at], kind="stable")]
        at = np.sort(at[_outline.park(gp[at], SEAM * s_fill)])
        if len(at) > len(best):
            best = at
    gp, gt, gb = gp[best], gt[best], gb[best]
    for b in bodies or ():
        rp, rt, rb, _ = _shapes.rings(tris, b.tri, normals, s_fill, opt.view_dir, opt.up_axis,
                                      s_fill * opt.line_ratio / opt.fill_ratio)
        keep = weights[rt] > 0
        if keep.any() and lines is not None:
            keep &= ~_outline._near(rp, lines.pos, GAP * s_fill)
        if keep.any() and line_part is not None and len(line_part[0]):
            keep &= ~_outline._near(rp, line_part[0], 0.5 * s_fill)
        if keep.any() and part.options.role == "pattern":
            keep[keep] = _vote(rp[keep], pos[mine], in_region[mine], VOTE * s_fill)
        if keep.any() and audience is not None:
            keep[keep] = audience.visibility(rp[keep], rt[keep]) > 0
        gp, gt, gb = np.vstack([gp, rp[keep]]), np.r_[gt, rt[keep]], np.vstack([gb, rb[keep]])
    return dict(kind="grid", pos=gp, tri=gt, bary=gb, count=len(gp))


def _detail_floor(tris) -> float:
    """The smallest pixel of the picture the lines are read from: at most ``DETECT_PIXELS`` of them."""
    ext = np.sort(np.ptp(tris.reshape(-1, 3), axis=0))
    return math.sqrt(float(ext[1] * ext[2]) / DETECT_PIXELS)


def lines_factor(f: float) -> float:
    """How much wider than asked the lines stand when the whole layout is ``f`` times sparser: the fill
    gives way first (up to ``FILL_FIRST`` times), then both grow alike."""
    return max(1.0, f / FILL_FIRST)


def _bodies(parts, tris, offsets, areas, s_fill: float, up_axis: int):
    """{part: its round bodies}: of the parts filled with "rings", every piece; with simple shapes and a
    lattice, the pieces that are round (``shapes.round_bodies``)."""
    out = {}
    for k, p in enumerate(parts):
        o = p.options
        if not p.has_fill:
            continue
        ids = np.arange(offsets[k], offsets[k + 1])
        ids = ids[areas[ids] > 0]
        if o.pattern == "rings":
            found = [_shapes.Body(piece, np.zeros(3), 0.0, 0.0, 0.0) for piece in _shapes.pieces(tris, ids)]
        elif o.shapes and o.pattern in ("hex", "square"):
            found = _shapes.round_bodies(tris, ids, areas, _shapes.ROUND_SIZE * s_fill, up_axis)
        else:
            found = []
        if found:
            out[k] = found
    return out


def _ring_fill(pic, bodies, k, allowed, s: float, sf: float, tris, normals, opt, dense, clear):
    """Picture points of the rings on the round ``bodies`` of part ``k`` the audience sees: rows a fill
    spacing ``sf`` apart, drones a line spacing ``s`` apart on them (they read as rings), where the fill
    may go (``allowed``), clear of the lines (``dense``) and of the drones placed (``clear``)."""
    out = []
    for b in bodies:
        pos, tri, _, _ = _shapes.rings(tris, b.tri, normals, sf, pic.toward, opt.up_axis, s)
        if not len(pos):
            continue
        seen = pic.shows(pos, tri, normals)
        if pic.eye is None:
            facing = normals[tri] @ pic.d
        else:
            to_eye = pic.eye - pos
            facing = np.einsum("ij,ij->i", normals[tri], to_eye) / np.maximum(np.linalg.norm(to_eye, axis=1), 1e-300)
        xy = pic.project(pos[seen & (facing > RING_FACING)])
        if not len(xy):
            continue
        keep = pic.sample(allowed, xy, False) & (pic.sample(pic.part, xy, -1) == k)
        if keep.any() and len(dense):
            keep[keep] = ~_pic._near(xy[keep], dense, _pic.FILL_GAP * sf)
        if keep.any() and len(clear):
            keep[keep] = ~_pic._near(xy[keep], clear, _pic.FILL_GAP * sf)
        out.append(xy[keep])
    return np.vstack(out) if out else np.zeros((0, 2))


def _make_fill(fl, s_fill, opt, tris, weights, offsets, k, fixed, stretch, relative, cand):
    """The fill drones of part ``k``: the lattice points as planned, or blue noise as in
    ``distribute`` (elimination, then relaxation round the line drones, which stand still)."""
    if fl["kind"] == "grid":
        return fl["pos"], fl["tri"], fl["bary"]
    pos, tri, bary = cand
    free, n = fl["free"], fl["count"]
    f_pos, f_tri, f_bary = pos[free], tri[free], bary[free]
    f_stretch = None if stretch is None else stretch[free]
    sel = eliminate(f_pos, n, fl["area"], scale=f_stretch)
    points, p_tri, p_bary = f_pos[sel], f_tri[sel], f_bary[sel]
    if opt.relax_iterations > 0 and n > 1:
        allowed = np.arange(offsets[k], offsets[k + 1])
        allowed = allowed[weights[allowed] > 0]
        widest = 1.0 if f_stretch is None else float(f_stretch.max())
        projector = SurfaceProjector(tris[allowed], allowed, 0.5 * s_fill, 0.15 * s_fill * widest)
        points, p_tri, p_bary = relax(points, p_tri, p_bary, projector, s_fill, opt.relax_iterations,
                                      density=relative, max_scale=widest, fixed=fixed if len(fixed) else None)
    return points, p_tri, p_bary


def _vote(points, cand_pos, cand_in, radius):
    """True where most candidates within ``radius`` are in the region (the nearest one, with none there)."""
    if not len(cand_pos):
        return np.zeros(len(points), bool)
    qi, pj, _ = Grid(cand_pos, radius).query_radius(points, radius)
    yes = np.bincount(qi, weights=cand_in[pj].astype(np.float64), minlength=len(points))
    total = np.bincount(qi, minlength=len(points))
    out = 2.0 * yes > total
    empty = total == 0
    if empty.any():
        _, idx = nearest(points[empty], cand_pos, radius)
        out[empty] = cand_in[idx]
    return out


def _directions(opt, tris):
    """Directions the lattice is projected along: towards the middle viewer in facade mode, the six
    sides of the scene otherwise (each triangle takes the one it faces most)."""
    if opt.facade:
        if opt.view_point is not None:
            corners = tris.reshape(-1, 3)
            center = (corners.min(axis=0) + corners.max(axis=0)) * 0.5
            return [normalize(np.asarray(opt.view_point, np.float64) - center)]
        return [normalize(opt.view_dir)]
    out = []
    for axis in range(3):
        for sign in (1.0, -1.0):
            v = np.zeros(3)
            v[axis] = sign
            out.append(v)
    return out


def _lattice(ids, tris, normals, opt, s, pattern, audience):
    """Points of a lattice of spacing ``s / PHASES`` on the triangles ``ids``,
    as the lattice in a plane looks projected onto the surface along a
    direction: the audience sees it straight.

    Returns (pos, tri, bary, how squarely the triangle faces the direction,
    phase); the points of one phase make a lattice of spacing ``s``."""
    e = np.zeros(0, np.int64)
    empty = np.zeros((0, 3)), e, np.zeros((0, 3)), np.zeros(0), e
    if not len(ids):
        return empty
    s = s / PHASES
    dirs = _directions(opt, tris)
    n = normals[ids]
    facing = np.stack([n @ d for d in dirs], 1)
    best = np.argmax(facing, axis=1)
    cos = math.cos(math.radians(opt.facade_angle)) if opt.facade else 0.0
    corners = tris[ids].reshape(-1, 3)
    center = (corners.min(axis=0) + corners.max(axis=0)) * 0.5
    if pattern == "square":
        a1, a2 = np.array([s, 0.0]), np.array([0.0, s])
    else:
        a1, a2 = np.array([s, 0.0]), np.array([0.5 * s, HEX * s])
    inv = np.linalg.inv(np.stack([a1, a2], 1))
    out_p, out_t, out_b, out_q, out_ph = [], [], [], [], []
    for which, d in enumerate(dirs):
        sel = ids[(best == which) & (facing[np.arange(len(ids)), best] > cos)]
        if not len(sel):
            continue
        # opposite sides share one lattice: seen from the front, a drone at the back stands behind one in front
        _, e1, e2 = view_basis(-d if d[int(np.argmax(np.abs(d)))] < 0 else d)
        P = tris[sel]
        xy = np.stack([P @ e1, P @ e2], -1) - np.array([center @ e1, center @ e2])  # (T, 3, 2)
        uv = xy @ inv.T  # lattice coordinates: points at whole numbers
        u0 = np.ceil(uv[..., 0].min(axis=1) - 1e-9).astype(np.int64)
        u1 = np.floor(uv[..., 0].max(axis=1) + 1e-9).astype(np.int64)
        v0 = np.ceil(uv[..., 1].min(axis=1) - 1e-9).astype(np.int64)
        v1 = np.floor(uv[..., 1].max(axis=1) + 1e-9).astype(np.int64)
        nu, nv = np.maximum(u1 - u0 + 1, 0), np.maximum(v1 - v0 + 1, 0)
        count = nu * nv
        todo = np.flatnonzero(count)
        if not len(todo):
            continue
        total = np.cumsum(count[todo])
        start = 0
        while start < len(todo):
            before = total[start - 1] if start else 0
            stop = max(int(np.searchsorted(total, before + _RASTER_PAIRS, side="right")), start + 1)
            t = todo[start:stop]
            c = count[t]
            owner = np.repeat(np.arange(len(t)), c)
            k = np.arange(int(c.sum())) - np.repeat(np.cumsum(c) - c, c)
            t = t[owner]
            gu = (u0[t] + k % nu[t]).astype(np.float64)
            gv = (v0[t] + k // nu[t]).astype(np.float64)
            a, b, q = uv[t, 0], uv[t, 1], uv[t, 2]
            den = (b[:, 1] - q[:, 1]) * (a[:, 0] - q[:, 0]) + (q[:, 0] - b[:, 0]) * (a[:, 1] - q[:, 1])
            good = np.abs(den) > 1e-18
            den = np.where(good, den, 1.0)
            l0 = ((b[:, 1] - q[:, 1]) * (gu - q[:, 0]) + (q[:, 0] - b[:, 0]) * (gv - q[:, 1])) / den
            l1 = ((q[:, 1] - a[:, 1]) * (gu - q[:, 0]) + (a[:, 0] - q[:, 0]) * (gv - q[:, 1])) / den
            l2 = 1.0 - l0 - l1
            inside = good & (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
            lam = np.clip(np.stack([l0, l1, l2], 1)[inside], 0.0, 1.0)
            lam /= lam.sum(axis=1, keepdims=True)
            tt = sel[t[inside]]
            out_p.append(np.einsum("ij,ijk->ik", lam, tris[tt]))
            out_t.append(tt)
            out_b.append(lam)
            out_q.append(facing[np.searchsorted(ids, tt), which])
            out_ph.append((np.mod(gu[inside], PHASES) * PHASES + np.mod(gv[inside], PHASES)).astype(np.int64))
            start = stop
    if not out_p:
        return empty
    p, t, b, q, ph = (np.concatenate(x) for x in (out_p, out_t, out_b, out_q, out_ph))
    if audience is not None and len(p):  # what nobody sees (behind another part) takes no drone
        seen = audience.visibility(p, t) > 0
        p, t, b, q, ph = p[seen], t[seen], b[seen], q[seen], ph[seen]
    return p, t, b, q, ph


def _kmeans(x: np.ndarray, k: int, rng, iterations: int = 30):
    """k-means (k-means++ start). Returns (centres (k', 3), label of each point); fewer centres
    when there are fewer different points."""
    n = len(x)
    centres = [x[int(rng.integers(n))]]
    d2 = np.einsum("ij,ij->i", x - centres[0], x - centres[0])
    for _ in range(1, min(k, n)):
        total = float(d2.sum())
        if total <= 1e-12:
            break
        c = x[int(rng.choice(n, p=d2 / total))]
        centres.append(c)
        d2 = np.minimum(d2, np.einsum("ij,ij->i", x - c, x - c))
    c = np.array(centres)
    label = _nearest_centre(x, c)
    for _ in range(iterations):
        counts = np.bincount(label, minlength=len(c))
        new = np.stack([np.bincount(label, weights=x[:, i], minlength=len(c)) for i in range(x.shape[1])], 1)
        new = np.where(counts[:, None] > 0, new / np.maximum(counts, 1)[:, None], c)
        moved = float(np.abs(new - c).max())
        c = new
        label = _nearest_centre(x, c)
        if moved < 1e-6:
            break
    used = np.bincount(label, minlength=len(c)) > 0
    remap = np.cumsum(used) - 1
    return c[used], remap[label]


def _nearest_centre(x, centres):
    d = np.stack([np.einsum("ij,ij->i", x - c, x - c) for c in centres], 1)
    return np.argmin(d, axis=1)
