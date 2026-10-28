"""Cage: drones on a net of rings and columns that wraps a round body (a fruit, a pumpkin), seen from every side.

Horizontal planes cut the model at even steps along its surface (a steep wall
and a flat top get the same spacing); every cut is a set of loops (a body, two
legs, an arm). The drones sit on the loops, and from loop to loop they keep
their place, so they stand in columns too, like the lines of a globe: a
column is carried from one ring to the next by its nearest point, two columns
that crowd merge (every other one ends: the meridians of a globe near its
pole), a gap that opens gets a new column.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import List, Optional

import numpy as np

from . import color as _color
from . import outline as _outline
from .api import DistributeOptions, DroneResult
from .mesh import MeshData
from .sampling import surface_candidates, triangle_areas_normals
from .spatial import nearest
from .stats import scale_pivot, spacing as _spacing

SLOPE_FLOOR = 0.15  # where the surface is level, rings stand at most 1 / this closer in height
LEVELS = 1024  # height bins of the arc-length profile
EDGE_SECTORS = 8  # an edge ring is checked all round, in this many sectors


@dataclass
class Style:
    ring: float = 1.0  # ring spacing along the surface, times the base spacing
    column: float = 1.0  # drone spacing along a ring, times the base spacing
    merge: float = 0.68  # two columns closer than this many column spacings: one ends
    split: float = 1.45  # a gap wider than this many column spacings gets new columns
    even: float = 0.3  # each ring moves its drones this far towards even spacing
    smooth: float = 0.2  # loops are smoothed over this many column spacings (fur, facets)
    dot: float = 2.2  # a loop shorter than this many column spacings is one drone at its middle
    pinch: float = 0.3  # a loop narrower than this many column spacings somewhere is cut in two there
    reach: float = 1.6  # a column goes on to the next ring when it is less than this many ring spacings away
    thin: float = 1.6  # a loop less than this many column spacings across (length / pi) is a stem: one drone
    speck: float = 0.12  # loops shorter than this many column spacings are specks (fur): no drone
    contrast: float = 0.6  # how much the brightness range of the model is stretched (0: as color.glow)
    features: bool = True  # draw the borders between the model's colours (eyes, crown band, stem)
    feature: float = 0.6  # drone spacing along a feature line, times the base spacing
    feature_min: float = 4.0  # feature lines shorter than this many of their spacings are left out
    clear: float = 0.7  # net drones this close to a feature drone (times the column spacing) give way
    recess: float = 0.25  # ... and lies this deep (times the base spacing) inside the convex hull of its cut
    rim_color: tuple = (1.0, 0.82, 0.35)  # sRGB of the rims of the dark holes
    auto: bool = True  # with no style given, a round body is drawn as a cage (``cage``), a figure as a net
    cage: tuple = (0.6, 1.7)  # (ring, column) spacing of a cage
    twist: float = 0.0  # round bodies: rings wound into a helix (0: level rings)
    stem: float = 0.42  # drone spacing up a stem (a thin part), times the base spacing
    stem_clear: float = 0.5  # net drones this close to a stem (times the column spacing) give way
    rim: float = 0.42  # drone spacing round a carved hole, times the base spacing
    carve_body: float = 0.75  # a hole sits where the body is at least this share of its widest
    carve_sure: float = 2.0  # a dent this many times ``carve_depth`` deep on average is a hole even if not dark
    carve_depth: float = 0.015  # a dark dent deeper than this share of the height in a round body is a carved hole
    corner_tol: float = 0.3  # hole outlines are simplified to polygons within this many feature spacings
    hole_margin: float = 1.2  # net drones keep this many column spacings off a hole (and off the wall behind it):
    # the carved face reads as bright rims on dark, from the front too
    light: float = 0.2  # on a figure, a colour this much lighter than the median (a face on dark fur) is outlined
    accent_share: float = 0.5  # ... or this share of the main hue's own chroma, if more
    accent: float = 0.035  # a hue this far (OKLab a, b) from the model's main hue is an accent: its border is drawn
    hole: float = 0.2  # a colour this much darker than the model's median is a hole: no net drones on it
    voxel: float = 0.25  # the colour grid, times the base spacing
    net_level: float = 0.6  # net drones glow this bright, feature lines and stems at full
    min_spacing: float = 1.6  # the net is never denser than this many minimum distances (a big budget is not all used)
    rib_min: int = 6  # fewer ribs than this are lobes (an apple): ``meridians`` columns instead
    smooth_rings: bool = False  # a smooth fruit (no ribs): True draws it by whole latitude rings (a beehive), else meridians
    smooth_style: tuple = (1.25, 1.0)  # (ring, column) spacing of a smooth fruit
    lat_dot: float = 0.65  # drone spacing on a latitude ring, times the base spacing
    meridians: int = 11  # columns of a smooth fruit
    caps: bool = True  # a round body gets a closed ring line at its base, nothing below the base
    top_cap: bool = False  # ... and one at its shoulder too
    flat_top: float = 0.75  # the net leaves the top of a round body where it faces up more than this (cos)
    cap_share: float = 0.3  # ... on the lowest and highest rings at least this share of the widest one: a small
    # ring round the stem and at the base (a wide one seen edge-on is a flat lid on the fruit)
    cap_share_smooth: float = 0.3  # ... of a smooth fruit (its latitude rings reach higher)
    cap: float = 0.45  # drone spacing on a cap ring, times the base spacing
    stem_from: float = 0.6  # a round body's stem is a thin part above this share of its height
    collar: float = 0.6  # the collar ring stands this many minimum distances out from the stem
    stem_chroma: float = 1.8  # a stem's colour is this much more saturated ...
    stem_light: float = 0.62  # ... at this OKLab lightness
    rib_level: float = 0.75  # brightness of a fruit's ribs (its column drones)
    rib_level_carved: float = 0.55  # ... when it has a carved face
    cap_level: float = 0.6  # brightness of the cap rings (no bright bar)
    edge_turn: float = 15.0  # a round body's outline turning this many degrees at one place (a rim, a neck, a
    # step: a turned shape, a spool, a vase) is an edge: a ring line there
    edge_sharp: float = 0.72  # ... at one place: over twice the length it turns less than 1 / this as much more
    edge_window: float = 2.0  # ... measured over this many minimum distances along the outline
    edge_wide: float = 0.3  # ... on a ring at least this share of the widest one
    edge_votes: float = 0.75  # ... in at least this share of the sectors round the axis
    edge_gap: float = 3.0  # two edges farther apart along the outline than this many base spacings: rings between
    edge_dot: float = 0.45  # drone spacing on an edge ring, times the base spacing
    edge_fill_level: float = 0.8  # brightness of the rings between the edges
    edge_rib_level: float = 0.5  # ... and of the columns, when the body has edges (the rings draw it)
    fit: float = 0.97  # use at least this share of the budget


# ---------------------------------------------------------------- the model, cut by planes

class _Model:
    def __init__(self, T: np.ndarray, normals: np.ndarray, areas: np.ndarray):
        self.T = T  # (T, 3, 3), z up
        self.normals = normals
        self.areas = areas
        self.z = T[..., 2]
        self.zmin, self.zmax = self.z.min(axis=1), self.z.max(axis=1)
        self.lo, self.hi = float(self.zmin.min()), float(self.zmax.max())
        self.vid, self.verts = _outline._weld(T)
        self.nv = int(self.vid.max()) + 1
        # arc length along the surface as a function of height
        zc = self.z.mean(axis=1)
        steep = np.sqrt(np.clip(1.0 - normals[:, 2] ** 2, 0.0, 1.0))
        bins = np.clip(((zc - self.lo) / (self.hi - self.lo) * LEVELS).astype(int), 0, LEVELS - 1)
        num = np.bincount(bins, weights=areas * steep, minlength=LEVELS)
        den = np.bincount(bins, weights=areas, minlength=LEVELS)
        have = np.flatnonzero(den > 0)
        slope = np.maximum(np.interp(np.arange(LEVELS), have, num[have] / den[have]), SLOPE_FLOOR)
        dz = (self.hi - self.lo) / LEVELS
        self.s_of_z = np.r_[0.0, np.cumsum(dz / slope)]
        self.z_grid = np.linspace(self.lo, self.hi, LEVELS + 1)
        self._cache = {}

    def levels(self, step: float) -> np.ndarray:
        total = float(self.s_of_z[-1])
        n = max(int(round(total / step)), 1)
        return np.interp((np.arange(n) + 0.5) * (total / n), self.s_of_z, self.z_grid)

    def cut(self, level: float):
        """The loops where the plane at height ``level`` cuts the surface: a list of
        (points (m, 3), triangle ids (m,), barycentrics (m, 3), closed)."""
        key = round(level, 9)
        if key in self._cache:
            return self._cache[key]
        level = float(level) + 1e-7 * (self.hi - self.lo)
        tt = np.flatnonzero((self.zmin < level) & (self.zmax > level))
        out = []
        if len(tt) >= 3:
            T, z, vid, nv = self.T, self.z, self.vid, self.nv
            above = z[tt] > level
            crossing = above != np.roll(above, -1, axis=1)
            which = np.argsort(~crossing, axis=1, kind="stable")[:, :2]
            rows = np.arange(len(tt))
            ends = []
            for q in range(2):
                i = which[:, q]
                j = (i + 1) % 3
                zi, zj = z[tt, i], z[tt, j]
                f = (level - zi) / (zj - zi)
                p = T[tt, i] + f[:, None] * (T[tt, j] - T[tt, i])
                b = np.zeros((len(tt), 3))
                b[rows, i], b[rows, j] = 1.0 - f, f
                va, vb = vid[tt, i], vid[tt, j]
                ends.append((np.minimum(va, vb) * nv + np.maximum(va, vb), p, b))
            (k0, p0, b0), (k1, p1, b1) = ends
            walks, _ = _outline._chains(k0, k1, p0, p1, np.zeros(2 * len(k0)))
            out = _stitch(walks, p0, p1, b0, b1, tt)
        self._cache[key] = out
        return out


def _stitch(walks, p0, p1, b0, b1, tt):
    """Chains joined into loops (a mesh that is not manifold everywhere breaks them at its pinches)."""
    pieces = []
    for edges, fw, closed, a, b in walks:
        f = fw[:, None]
        pts = np.where(f, p0[edges], p1[edges])
        bar = np.where(f, b0[edges], b1[edges])
        tri = tt[edges]
        if closed:
            pieces.append((pts, tri, bar, True, a, b))
        else:
            # an open chain ends at its last edge's far end
            e = edges[-1]
            end_p = p1[e] if fw[-1] else p0[e]
            end_b = b1[e] if fw[-1] else b0[e]
            pieces.append((np.vstack([pts, end_p]), np.r_[tri, tt[e]], np.vstack([bar, end_b]), False, a, b))
    loops = [(p, t, b, True) for p, t, b, c, _, _ in pieces if c]
    open_ = [x for x in pieces if not x[3]]
    if not open_:
        return loops
    at = {}
    for k, (_, _, _, _, a, b) in enumerate(open_):
        at.setdefault(a, []).append(k)
        at.setdefault(b, []).append(k)
    used = [False] * len(open_)
    for k0 in range(len(open_)):
        if used[k0]:
            continue
        used[k0] = True
        p, t, b, _, start, node = open_[k0]
        seq = [(p, t, b)]
        closed = False
        while True:
            nxt = [k for k in at.get(node, ()) if not used[k]]
            if not nxt:
                break
            k = nxt[0]
            used[k] = True
            q, tq, bq, _, a, e = open_[k]
            if a == node:
                seq.append((q[1:], tq[1:], bq[1:]))
                node = e
            else:
                seq.append((q[::-1][1:], tq[::-1][1:], bq[::-1][1:]))
                node = a
            if node == start:
                closed = True
                break
        P = np.vstack([s[0] for s in seq])
        Tq = np.concatenate([s[1] for s in seq])
        Bq = np.vstack([s[2] for s in seq])
        if closed or np.linalg.norm(P[-1] - P[0]) < 1e-6 * (1 + np.ptp(P, axis=0).max()):
            loops.append((P[:-1], Tq[:-1], Bq[:-1], True))
        elif len(P) >= 2:
            loops.append((P, Tq, Bq, False))
    return loops


# ---------------------------------------------------------------- loops as smooth lines

@dataclass
class Loop:
    xy: np.ndarray  # (m, 2) smoothed, evenly resampled; closed loops do not repeat the first point
    raw: np.ndarray  # (m, 3) the surface points they came from
    tri: np.ndarray  # (m,)
    bary: np.ndarray  # (m, 3)
    z: float
    closed: bool
    arc: np.ndarray = field(default=None)  # (m + closed,) arc length at each point (and the end)
    length: float = 0.0
    centre: np.ndarray = field(default=None)

    def __post_init__(self):
        q = np.vstack([self.xy, self.xy[:1]]) if self.closed else self.xy
        seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
        self.arc = np.r_[0.0, np.cumsum(seg)]
        self.length = float(self.arc[-1])
        self.centre = _polygon_centre(self.xy) if self.closed else self.xy.mean(axis=0)

    def at(self, t: np.ndarray):
        """Points at arc lengths ``t``: (xy, index of the nearest sample)."""
        q = np.vstack([self.xy, self.xy[:1]]) if self.closed else self.xy
        L = max(self.length, 1e-12)
        t = np.mod(t, L) if self.closed else np.clip(t, 0.0, L)
        i = np.clip(np.searchsorted(self.arc, t, side="right") - 1, 0, len(q) - 2)
        seg = self.arc[i + 1] - self.arc[i]
        u = ((t - self.arc[i]) / np.maximum(seg, 1e-300))[:, None]
        xy = q[i] * (1.0 - u) + q[i + 1] * u
        near = np.where(u[:, 0] < 0.5, i, i + 1) % len(self.xy)
        return xy, near


def _polygon_centre(xy: np.ndarray) -> np.ndarray:
    x, y = xy[:, 0], xy[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cr = x * yn - xn * y
    a = cr.sum() / 2.0
    if abs(a) < 1e-12:
        return xy.mean(axis=0)
    return np.array([((x + xn) * cr).sum(), ((y + yn) * cr).sum()]) / (6.0 * a)


def _signed_area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum())


def _inside(pt: np.ndarray, poly: np.ndarray) -> bool:
    x, y = pt
    xa, ya = poly[:, 0], poly[:, 1]
    xb, yb = np.roll(xa, -1), np.roll(ya, -1)
    cross = (ya > y) != (yb > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = xa + (y - ya) * (xb - xa) / (yb - ya)
    return bool((cross & (x < xi)).sum() % 2)


def _resample(P: np.ndarray, closed: bool, h: float):
    """Evenly spaced points ``h`` or a little closer along the polyline ``P``; the index of the
    nearest original point of each."""
    q = np.vstack([P, P[:1]]) if closed else P
    seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(seg)]
    L = float(arc[-1])
    if L <= 0:
        return P[:1].copy(), np.zeros(1, np.int64)
    m = max(int(math.ceil(L / h)), 3 if closed else 2)
    x = np.arange(m) * (L / m) if closed else np.linspace(0.0, L, m)
    i = np.clip(np.searchsorted(arc, x, side="right") - 1, 0, len(seg) - 1)
    u = ((x - arc[i]) / np.maximum(seg[i], 1e-300))[:, None]
    pts = q[i] * (1.0 - u) + q[i + 1] * u
    near = np.where(u[:, 0] < 0.5, i, i + 1) % len(P)
    return pts, near


def _smooth(xy: np.ndarray, closed: bool, sigma: float) -> np.ndarray:
    """Gaussian smoothing along the line, ``sigma`` in samples."""
    if sigma < 0.3 or len(xy) < 4:
        return xy
    r = int(min(math.ceil(3.0 * sigma), (len(xy) - 1) // 2))
    if r < 1:
        return xy
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(xy, ((r, r), (0, 0)), mode="wrap" if closed else "edge")
    out = np.stack([np.convolve(pad[:, c], k, mode="valid") for c in range(xy.shape[1])], 1)
    return out


def _pinches(xy: np.ndarray, h: float, width: float, least: float, depth: int = 0) -> List[np.ndarray]:
    """Index runs of the closed line ``xy`` (evenly ``h`` apart) cut where it nearly touches itself
    (closer than ``width`` across, both sides at least ``least`` long)."""
    m = len(xy)
    idx = np.arange(m)
    if depth > 6 or m * h < 2.0 * least or m > 4000:
        return [idx]
    step = max(1, int(width / (2.0 * h)))
    sub = idx[::step]
    d = np.linalg.norm(xy[sub, None, :] - xy[None, sub, :], axis=2)
    gap = np.abs(sub[:, None] - sub[None, :])
    along = np.minimum(gap, m - gap) * h
    cand = (d < width) & (along > least)
    if not cand.any():
        return [idx]
    a, b = np.unravel_index(int(np.argmin(np.where(cand, d, np.inf))), d.shape)
    i, j = sorted((int(sub[a]), int(sub[b])))
    first = idx[i:j + 1]
    second = np.r_[idx[j:], idx[:i + 1]]
    out = []
    for run in (first, second):
        for r in _pinches(xy[run], h, width, least, depth + 1):
            out.append(run[r])
    return out


def _loops(model: _Model, level: float, h: float, st: Style, sc: float) -> List[Loop]:
    """The loops of the cut at ``level``, resampled, cut at their pinches, the walls of cavities
    (loops inside an odd number of others) left out, and smoothed."""
    raw = model.cut(level)
    pieces = []
    for P, tri, bary, closed in raw:
        if len(P) < 2:
            continue
        pts, near = _resample(P, closed, h)
        if closed and len(pts) >= 3:
            for run in _pinches(pts[:, :2], h, st.pinch * sc, max(2.0 * sc, 4.0 * st.pinch * sc)):
                if len(run) >= 3:
                    pieces.append((pts[run], tri[near[run]], bary[near[run]], True))
        else:
            pieces.append((pts, tri[near], bary[near], closed))
    # cavities: a loop inside an odd number of other loops is the wall of a hollow
    keep = []
    for k, (p, t, b, c) in enumerate(pieces):
        if not c:
            keep.append(k)
            continue
        probe = p[0, :2]
        depth = sum(1 for j, q in enumerate(pieces) if j != k and q[3] and len(q[0]) >= 3 and _inside(probe, q[0][:, :2]))
        if depth % 2 == 0:
            keep.append(k)
    loops = []
    sig = st.smooth * sc / h
    for k in keep:
        p, t, b, c = pieces[k]
        xy = _smooth(p[:, :2], c, sig)
        loops.append(Loop(xy, p, t, b, float(level), c))
    return loops




# ---------------------------------------------------------------- the shape: round or not, its ribs

@dataclass
class _Shape:
    round: bool = False
    axis: np.ndarray = field(default_factory=lambda: np.zeros(2))
    ribs: int = 0  # how many ribs (lobes) a round body has, 0: none found
    groove: float = -0.5 * math.pi  # the angle of a groove round the axis (the front without ribs)
    seed_length: float = 0.0


def _main(loops: List[Loop], thin: float) -> Optional[Loop]:
    big = [l for l in loops if l.closed and l.length / math.pi >= thin]
    return max(big, key=lambda l: l.length) if big else None


def _shape(model: _Model, step: float, st: Style) -> _Shape:
    """Is the model a round body (one loop round a vertical axis at nearly every height), and its ribs."""
    levels = model.levels(step)
    h = step / 6.0
    mains, totals = [], []
    for lv in levels:
        loops = _loops(model, lv, h, st, 2.0 * step)
        big = [l for l in loops if l.closed and l.length / math.pi >= 2.0 * step]
        mains.append(max(big, key=lambda l: l.length) if big else None)
        totals.append(sum(l.length for l in big))
    n = len(levels)
    mid = [k for k in range(n) if 0.1 * n <= k <= 0.9 * n and mains[k] is not None]
    out = _Shape()
    if not mid:
        return out
    cen = np.array([mains[k].centre for k in mid])
    axis = np.median(cen, axis=0)
    out.axis = axis
    out.seed_length = max(totals)
    ok = 0
    for k in mid:
        lp = mains[k]
        r = math.sqrt(abs(_signed_area(lp.xy)) / math.pi)
        single = lp.length >= 0.85 * totals[k]
        near = np.linalg.norm(lp.centre - axis) < 0.3 * max(r, 1e-9)
        ok += bool(single and near and _inside(axis, lp.xy))
    out.round = ok >= 0.75 * len(mid)
    if not out.round:
        return out
    # ribs: the strongest angular frequency of the radius, 4..24 times round
    power = np.zeros(25)
    coef = np.zeros(25, complex)
    votes = []
    nb = 360
    phi = -math.pi + (np.arange(nb) + 0.5) * (2 * math.pi / nb)
    for k in mid:
        d = mains[k].xy - axis
        a = np.arctan2(d[:, 1], d[:, 0])
        r = np.hypot(d[:, 0], d[:, 1])
        bins = np.floor((a + math.pi) / (2 * math.pi) * nb).astype(int) % nb
        cnt = np.bincount(bins, minlength=nb)
        have = np.flatnonzero(cnt > 0)
        if len(have) < nb // 4:
            continue
        rr = np.bincount(bins, weights=r, minlength=nb)[have] / cnt[have]
        rr = np.interp(np.arange(nb), have, rr, period=nb)
        rr = rr / rr.mean() - 1.0
        F = np.exp(-1j * np.outer(np.arange(25), phi)) @ rr
        power += np.abs(F) ** 2
        coef += F
        votes.append(int(np.argmax(np.abs(F[4:25]))) + 4)
    if votes:
        f = int(np.bincount(votes).argmax())
        if np.mean(np.asarray(votes) == f) >= 0.35:
            out.ribs = f
            crest = -np.angle(coef[f]) / f
            out.groove = crest + math.pi / f
    return out


def _profile_edges(r, z, lo: float, H: float, w: float, wide: float, top: Optional[float], st: Style):
    """The profile of points (radius ``r``, height ``z``): the widest at each height, then on inwards to the
    axis at both ends (a flat top's rim is a turn too), resampled every ``w`` / 8 along it, and the points
    where it turns sharply: more than ``st.edge_turn`` degrees over ``w``, and over twice that not much more.
    Returns (arc length, points, edge indices)."""
    nb = 512
    b = np.clip(((z - lo) / H * nb).astype(int), 0, nb - 1)
    order = np.lexsort((r, b))
    last = order[np.r_[np.flatnonzero(np.diff(b[order])), len(order) - 1]]  # the widest vertex of each height
    prof = np.c_[r[last], z[last]]
    prof = np.vstack([[0.0, prof[0, 1]], prof, [0.0, prof[-1, 1]]])
    h = w / 8.0
    seg = np.linalg.norm(np.diff(prof, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(seg)]
    u = np.arange(0.0, arc[-1], h)
    P = np.c_[np.interp(u, arc, prof[:, 0]), np.interp(u, arc, prof[:, 1])]
    n = len(P)
    k1 = max(int(round(0.5 * w / h)), 1)

    def turn(k):
        i = np.arange(k, n - k)
        a, c = P[i] - P[i - k], P[i + k] - P[i]
        cos = (a * c).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(c, axis=1), 1e-12)
        out = np.zeros(n)
        out[i] = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        return out

    t1, t2 = turn(k1), turn(2 * k1)
    # a big ring (not the foot of a stem), below the stem of a fruit (where it meets the stem is no edge)
    sharp = (t1 >= st.edge_turn) & (t1 >= st.edge_sharp * t2) & (P[:, 0] > wide)
    if top is not None:
        sharp &= P[:, 1] < top - w
    edges = []
    for i in np.flatnonzero(sharp)[np.argsort(-t1[sharp])]:
        if all(abs(u[i] - u[j]) > w for j in edges):
            edges.append(int(i))
    return u, P, sorted(edges)


def _edge_rings(model: _Model, shape: _Shape, st: Style, d0: float, s: float, top: Optional[float] = None):
    """The edges of a round body's outline (a turned shape: a spool, a vase, a goblet): the heights where its
    profile, the widest radius at each height, turns sharply (a rim, a neck, a step), and between two edges
    far apart along it, rings at even steps between. A fruit (its profile bends smoothly) has none; nothing
    from ``top`` up (its stem). Returns [(height, 0 an edge or 1 a ring between)], low to high."""
    if not shape.round:
        return []
    H = model.hi - model.lo
    d = model.verts[:, :2] - shape.axis
    r = np.hypot(d[:, 0], d[:, 1])
    z = model.verts[:, 2]
    w = max(st.edge_window * d0, 0.02 * H)
    wide = st.edge_wide * float(r.max())
    u, P, edges = _profile_edges(r, z, model.lo, H, w, wide, top, st)
    if not edges:
        return []
    # an edge of a turned shape goes all round: the profile of most sectors turns there too (not one lobe
    # of a fruit, not the rim of a carved hole)
    sector = (np.floor((np.arctan2(d[:, 1], d[:, 0]) + math.pi) / (2 * math.pi) * EDGE_SECTORS).astype(int)
              % EDGE_SECTORS)
    votes = np.zeros(len(edges))
    for q in range(EDGE_SECTORS):
        m = sector == q
        if m.sum() < 4:
            continue
        _, Pq, eq = _profile_edges(r[m], z[m], model.lo, H, w, wide, top, st)
        if eq:
            zq = Pq[eq, 1]
            votes += [float(np.abs(zq - P[i, 1]).min() <= 0.5 * w) for i in edges]
    edges = sorted(i for i, v in zip(edges, votes) if v >= st.edge_votes * EDGE_SECTORS)
    if len(edges) < 2:  # a turned shape has two at least (its rims, a neck); a flat base alone is the base ring
        return []
    e = 0.02 * d0
    out = []
    for q, i in enumerate(edges):
        out.append((float(np.clip(P[i, 1], model.lo + e, model.hi - e)), 0))
        if q + 1 < len(edges):  # a long stretch between two edges: rings at even steps
            j = edges[q + 1]
            m = int(math.ceil((u[j] - u[i]) / (st.edge_gap * s))) - 1
            for f in range(1, m + 1):
                out.append((float(np.interp(u[i] + (u[j] - u[i]) * f / (m + 1), u, P[:, 1])), 1))
    return sorted(out)


def _rays(loop: Loop, centre: np.ndarray, angles: np.ndarray):
    """Arc length on ``loop`` of the outermost point the ray from ``centre`` at each angle hits
    (NaN where it misses)."""
    a = loop.xy
    b = np.roll(a, -1, axis=0)
    u = np.stack([np.cos(angles), np.sin(angles)], 1)  # (n, 2)
    ab = b - a  # (m, 2)
    ac = a - centre  # (m, 2)
    den = u[:, None, 0] * ab[None, :, 1] - u[:, None, 1] * ab[None, :, 0]  # (n, m)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ac[None, :, 0] * ab[None, :, 1] - ac[None, :, 1] * ab[None, :, 0]) / den
        sp = (ac[None, :, 0] * u[:, None, 1] - ac[None, :, 1] * u[:, None, 0]) / den
    hit = (np.abs(den) > 1e-12) & (t > 0) & (sp >= 0) & (sp < 1)
    t = np.where(hit, t, -np.inf)
    j = np.argmax(t, axis=1)
    rows = np.arange(len(angles))
    seg = np.diff(loop.arc)
    arc = loop.arc[j] + np.clip(np.nan_to_num(sp[rows, j]), 0, 1) * seg[j]
    return np.where(np.isfinite(t[rows, j]), arc, np.nan)


# ---------------------------------------------------------------- features: the borders between the colours

@dataclass
class _Palette:
    keys: np.ndarray  # sorted voxel keys
    dims: np.ndarray
    lo: np.ndarray
    h: float
    lab: np.ndarray  # (V, 3) smoothed OKLab of each voxel
    lin: np.ndarray  # (V, 3) smoothed linear colour of each voxel
    median: float  # median lightness of the surface
    ab: np.ndarray  # (2,) the main hue (median a, b)
    recess: Optional[np.ndarray] = None  # (V,) how deep each voxel lies inside the convex hull of its cut

    def voxel(self, p: np.ndarray) -> np.ndarray:
        """The voxel of each point, or of an occupied neighbour (a smoothed line strays a little); -1: none."""
        q = np.floor((p - self.lo) / self.h).astype(np.int64) + 1
        base = (q[:, 0] * self.dims[1] + q[:, 1]) * self.dims[2] + q[:, 2]
        out = np.full(len(p), -1)
        for dx, dy, dz in [(0, 0, 0)] + [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
                                         if (x, y, z) != (0, 0, 0)]:
            todo = out < 0
            if not todo.any():
                break
            key = base[todo] + (dx * self.dims[1] + dy) * self.dims[2] + dz
            j = np.clip(np.searchsorted(self.keys, key), 0, len(self.keys) - 1)
            out[np.flatnonzero(todo)] = np.where(self.keys[j] == key, j, -1)
        return out

    def masks(self, st: Style):
        """(dark, accent) per voxel: much darker than the model (a carved hole), or of another hue (a gold
        crown on grey fur, a green stem on an orange pumpkin)."""
        dark = self.lab[:, 0] < self.median - st.hole
        if self.recess is not None:
            dark &= self.recess > st.recess * float(self.h / st.voxel)
        far = max(st.accent, st.accent_share * float(np.linalg.norm(self.ab)))
        accent = (np.linalg.norm(self.lab[:, 1:] - self.ab, axis=1) > far) & ~dark
        return dark, accent

    def light(self, st: Style) -> np.ndarray:
        """Voxels much lighter than the model and of its own hue (a face and a belly on dark fur)."""
        dark, accent = self.masks(st)
        return (self.lab[:, 0] > self.median + st.light) & ~accent & ~dark

    def fill_recess(self, model: "_Model") -> None:
        """Depth of every voxel below the convex hull of the horizontal cut through it."""
        rec = np.zeros(len(self.keys))
        for level in np.arange(model.lo + 0.5 * self.h, model.hi, self.h):
            raw = model.cut(level)
            if not raw:
                continue
            P = np.vstack([r[0] for r in raw])
            if len(P) < 3:
                continue
            hull = _hull(P[:, :2])
            if len(hull) < 3:
                continue
            a, b = hull, np.roll(hull, -1, axis=0)
            ab = b - a
            ap = P[:, None, :2] - a[None]
            u = np.clip((ap * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12)[None], 0.0, 1.0)
            d = np.linalg.norm(ap - u[..., None] * ab[None], axis=2).min(axis=1)
            vx = self.voxel(P)
            ok = vx >= 0
            np.maximum.at(rec, vx[ok], d[ok])
        self.recess = rec


def _hull(xy: np.ndarray) -> np.ndarray:
    """Convex hull of 2D points, counter-clockwise (monotone chain)."""
    pts = np.unique(np.round(xy, 6), axis=0)
    if len(pts) < 3:
        return pts
    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(p)
        return h
    lst = pts.tolist()
    lower, upper = half(lst), half(lst[::-1])
    return np.asarray(lower[:-1] + upper[:-1])


def _palette(mesh: MeshData, T: np.ndarray, areas: np.ndarray, h: float, st: Style, rng) -> _Palette:
    """The model's colours on a grid of voxels ``h`` wide, smoothed over a voxel round (fur, noise)."""
    n = int(min(600_000, max(60_000, 8.0 * float(areas.sum()) / (h * h))))
    p, t, b = surface_candidates(T, areas, n, rng)
    lin = np.clip(_color.surface_colors(mesh, t, b), 0.0, None)
    lab = _color.linear_to_oklab(lin)
    lo = p.min(axis=0) - h
    q = np.floor((p - lo) / h).astype(np.int64) + 1
    dims = q.max(axis=0) + 2
    key = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
    keys, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    both = np.c_[lab, lin]
    sums = np.stack([np.bincount(inv, weights=both[:, c]) for c in range(6)], 1)
    acc, acn = np.zeros_like(sums), np.zeros(len(keys))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                nk = keys + (dx * dims[1] + dy) * dims[2] + dz
                j = np.clip(np.searchsorted(keys, nk), 0, len(keys) - 1)
                ok = keys[j] == nk
                acc[ok] += sums[j[ok]]
                acn[ok] += cnt[j[ok]]
    v = acc / np.maximum(acn, 1)[:, None]
    return _Palette(keys, dims, lo, h, v[:, :3], v[:, 3:], float(np.median(lab[:, 0])),
                    np.median(lab[:, 1:], axis=0))


@dataclass
class _Feature:
    xyz: np.ndarray  # (m, 3) the smoothed line, evenly resampled
    closed: bool
    color: np.ndarray  # linear colour its drones show
    length: float
    dark: bool  # it rings a dark hole (a carved eye)


def _features(model: "_Model", pal: _Palette, a: float, st: Style, figure: bool = True) -> List[_Feature]:
    """The rims of the dark holes and the borders of the accent colours on the surface, as smooth 3D
    lines at least ``st.feature_min`` line spacings ``a`` long."""
    T = model.T
    vx = pal.voxel(T.mean(axis=1))
    dark_v, accent_v = pal.masks(st)
    edge, va, vb = _outline._edges(model.vid)
    order = np.argsort(edge, kind="stable")
    e_sorted = edge[order]
    starts = np.flatnonzero(np.r_[True, e_sorted[1:] != e_sorted[:-1]])
    sizes = np.diff(np.r_[starts, len(order)])
    two = starts[sizes == 2]
    h1, h2 = order[two], order[two + 1]
    out = []
    hl = a / 4.0
    V = model.verts
    todo = [(accent_v, False)] + ([(pal.light(st), False)] if figure and st.light > 0 else [])
    for mask_v, dark in todo:
        tm = np.zeros(len(T), bool)
        tm[vx >= 0] = mask_v[vx[vx >= 0]]
        if not tm.any():
            continue
        border = tm[h1 // 3] != tm[h2 // 3]
        hh = np.where(tm[h1 // 3], h1, h2)[border]  # the half-edge on the masked side
        if not len(hh):
            continue
        k0, k1 = va[hh], vb[hh]
        p0, p1 = V[k0], V[k1]
        # colour of the line: the bright side for a hole (its rim), the accent itself otherwise
        side = np.where(tm[h1 // 3], h2, h1)[border] if dark else hh
        sv = pal.voxel(T[side // 3].mean(axis=1))
        walks, _ = _outline._chains(k0, k1, p0, p1, np.zeros(2 * len(k0)))
        for edges, fw, closed, _, _ in walks:
            f = fw[:, None]
            pts = np.where(f, p0[edges], p1[edges])
            if not closed:
                e = edges[-1]
                pts = np.vstack([pts, p1[e] if fw[-1] else p0[e]])
            seg = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]) if closed else pts, axis=0), axis=1)
            length = float(seg.sum())
            if length < st.feature_min * a or len(pts) < 3:
                continue
            xyz, _ = _resample(pts, closed, hl)
            xyz = _smooth3(xyz, closed, 0.5 * a / hl)
            got = sv[edges]
            got = got[got >= 0]
            col = pal.lin[got].mean(axis=0) if len(got) else np.ones(3)
            if dark:  # the rim of a carved hole glows warm, like candle light
                col = _color.srgb_to_linear(np.asarray(st.rim_color))
            out.append(_Feature(xyz, closed, col, length, dark))
    return out


def _smooth3(p: np.ndarray, closed: bool, sigma: float) -> np.ndarray:
    return _smooth(p, closed, sigma) if len(p) >= 4 else p


def _place_features(feats: List[_Feature], a: float, d0: float):
    """Drones evenly along every feature line, ``a`` apart (closed loops at least ``RING_MIN``)."""
    pos, col = [], []
    for f in feats:
        q = np.vstack([f.xyz, f.xyz[:1]]) if f.closed else f.xyz
        seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
        arc = np.r_[0.0, np.cumsum(seg)]
        L = float(arc[-1])
        if L <= 0:
            continue
        if f.closed:
            n = max(int(round(L / a)), min(6, int(L / (1.05 * d0))))
            if n < 3:
                continue
            x = np.arange(n) * (L / n)
        else:
            n = max(int(round(L / a)), 1) + 1
            x = np.linspace(0.0, L, n)
        i = np.clip(np.searchsorted(arc, x, side="right") - 1, 0, len(seg) - 1)
        u = ((x - arc[i]) / np.maximum(seg[i], 1e-300))[:, None]
        pos.append(q[i] * (1 - u) + q[i + 1] * u)
        col.append(np.repeat(f.color[None], n, axis=0))
    if not pos:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.vstack(pos), np.vstack(col)


# ---------------------------------------------------------------- carvings: dark holes in a round body

@dataclass
class _Carving:
    theta0: float
    dth: float
    z0: float
    hm: float
    radius: float  # of the widest ring: map units along a ring
    centres: np.ndarray  # (rows, 2) the centre of each row's cut (the body may lean)
    holes: np.ndarray  # (rows, cols) the holes, cleaned
    renv: np.ndarray  # (rows, cols) radius of the outer skin (the convex hull of the cut)
    polys: list  # simplified outlines, each (m, 2) in (theta, z), closed

    def cell(self, xyz: np.ndarray):
        i = np.clip(np.floor((xyz[:, 2] - self.z0) / self.hm).astype(int), 0, self.holes.shape[0] - 1)
        d = xyz[:, :2] - self.centres[i]
        th = np.mod(np.arctan2(d[:, 1], d[:, 0]) - self.theta0, 2 * math.pi)
        j = np.floor(th / self.dth).astype(int) % self.holes.shape[1]
        return i, j

    def inside(self, xyz: np.ndarray, grow: float) -> np.ndarray:
        """True for points in a hole or within ``grow`` of one (along the surface)."""
        k = max(0, int(round(grow / self.hm)))
        m = _dilate(self.holes, k)
        i, j = self.cell(xyz)
        return m[i, j]

    def behind(self, xyz: np.ndarray, grow: float) -> np.ndarray:
        """True for points the viewer of the holes sees through a hole (or within ``grow`` of one): the
        far wall of a hollow lantern straight behind its carved face, which would muddle the face."""
        rows, cols = self.holes.shape
        hi, hj = np.nonzero(self.holes)
        if not len(hi):
            return np.zeros(len(xyz), bool)
        th = self.theta0 + (hj + 0.5) * self.dth
        face = math.atan2(float(np.sin(th).mean()), float(np.cos(th).mean()))
        u = np.array([math.cos(face), math.sin(face)])
        v = np.array([-u[1], u[0]])
        i = np.clip(np.floor((xyz[:, 2] - self.z0) / self.hm).astype(int), 0, rows - 1)
        side = np.einsum("ij,j->i", xyz[:, :2] - self.centres[i], v)
        r = self.renv.max(axis=1)[i]
        on = np.abs(side) < r
        front = face + np.arcsin(np.clip(side / np.maximum(r, 1e-9), -1.0, 1.0))
        j = np.floor(np.mod(front - self.theta0, 2 * math.pi) / self.dth).astype(int) % cols
        m = _dilate(self.holes, max(0, int(round(grow / self.hm))))
        return on & m[i, j]

    def to3d(self, th: np.ndarray, z: np.ndarray) -> np.ndarray:
        i = np.clip(np.round((z - self.z0) / self.hm - 0.5).astype(int), 0, self.holes.shape[0] - 1)
        j = np.floor(np.mod(th - self.theta0, 2 * math.pi) / self.dth).astype(int) % self.holes.shape[1]
        r = self.renv[i, j]
        c = self.centres[i]
        return np.c_[c[:, 0] + r * np.cos(th), c[:, 1] + r * np.sin(th), z]

    def place(self, a: float, d0: float):
        """Drones round every hole: on its corners, then evenly along its sides, ``a`` apart at most.
        Returns (pos, corner)."""
        pos, corner = [], []
        for P in self.polys:
            # corners closer than the minimum distance merge (the sharper one stays)
            xy = np.c_[P[:, 0] * self.radius, P[:, 1]]
            keep = [0]
            for i in range(1, len(P)):
                if np.linalg.norm(xy[i] - xy[keep[-1]]) >= 1.1 * d0:
                    keep.append(i)
            while len(keep) > 3 and np.linalg.norm(xy[keep[-1]] - xy[keep[0]]) < 1.1 * d0:
                keep.pop()
            P = P[keep]
            m = len(P)
            if m < 3:
                continue
            xy = np.c_[P[:, 0] * self.radius, P[:, 1]]
            seg = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
            if seg.sum() < 3.0 * d0:
                continue
            for k in range(m):
                n = max(1, int(seg[k] // a))
                f = np.arange(n) / n
                q = P[k] + f[:, None] * (P[(k + 1) % m] - P[k])
                pos.append(self.to3d(q[:, 0], q[:, 1]))
                c = np.zeros(n, bool)
                c[0] = True
                corner.append(c)
        if not pos:
            return np.zeros((0, 3)), np.zeros(0, bool)
        return np.vstack(pos), np.concatenate(corner)


def _dilate(m: np.ndarray, k: int) -> np.ndarray:
    out = m.copy()
    for _ in range(k):
        g = out.copy()
        g |= np.roll(out, 1, axis=1) | np.roll(out, -1, axis=1)
        g[1:] |= out[:-1]
        g[:-1] |= out[1:]
        out = g
    return out


def _erode(m: np.ndarray, k: int) -> np.ndarray:
    return ~_dilate(~m, k)


def _douglas(P: np.ndarray, tol: float) -> np.ndarray:
    """Indices of the points of the open polyline ``P`` kept by Douglas-Peucker at ``tol``."""
    keep = np.zeros(len(P), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(P) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        d = P[b] - P[a]
        L = float(np.hypot(*d))
        rel = P[a + 1:b] - P[a]
        dist = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0]) / L if L > 0 else np.hypot(rel[:, 0], rel[:, 1])
        i = int(np.argmax(dist))
        if dist[i] > tol:
            keep[a + 1 + i] = True
            stack += [(a, a + 1 + i), (a + 1 + i, b)]
    return np.flatnonzero(keep)


def _outlines(mask: np.ndarray):
    """Closed boundary loops (lists of lattice corners (x = column, y = row)) of the pixels of ``mask``."""
    H, W = mask.shape
    edges = {}
    ii, jj = np.nonzero(mask)
    for i, j in zip(ii.tolist(), jj.tolist()):
        if i == 0 or not mask[i - 1, j]:
            edges.setdefault((j, i), []).append((j + 1, i))
        if j == W - 1 or not mask[i, j + 1]:
            edges.setdefault((j + 1, i), []).append((j + 1, i + 1))
        if i == H - 1 or not mask[i + 1, j]:
            edges.setdefault((j + 1, i + 1), []).append((j, i + 1))
        if j == 0 or not mask[i, j - 1]:
            edges.setdefault((j, i + 1), []).append((j, i))
    loops = []
    while edges:
        start = next(iter(edges))
        node, loop = start, [start]
        while True:
            outs = edges.get(node)
            if not outs:
                break
            nxt = outs.pop()
            if not outs:
                del edges[node]
            node = nxt
            if node == start:
                break
            loop.append(node)
        if len(loop) >= 4:
            loops.append(np.asarray(loop, np.float64))
    return loops


def _carvings(model: _Model, pal: _Palette, shape: _Shape, st: Style, s: float) -> Optional[_Carving]:
    """The dark holes carved into a round body (a jack-o'-lantern's eyes, nose and mouth), mapped round
    its axis, cleaned, and outlined by simple polygons."""
    if not shape.round or pal is None:
        return None
    size = model.hi - model.lo
    hm = max(size / 140.0, 0.15)
    radius = max(shape.seed_length / (2 * math.pi), hm)
    cols = max(int(round(2 * math.pi * radius / hm)), 16)
    dth = 2 * math.pi / cols
    theta0 = shape.groove
    th = theta0 + (np.arange(cols) + 0.5) * dth
    rows = np.arange(model.lo + 0.5 * hm, model.hi, hm)
    dark = np.zeros((len(rows), cols), bool)
    renv = np.zeros((len(rows), cols))
    light = np.full((len(rows), cols), pal.median)
    rec = np.zeros((len(rows), cols))
    depth = st.carve_depth * size
    # the centre of each row: its main cut's middle, smoothed up the body
    cen = np.full((len(rows), 2), np.nan)
    for i, z in enumerate(rows):
        raw = [r for r in model.cut(z) if r[3] and len(r[0]) >= 3]
        if raw:
            P = max(raw, key=lambda r: float(np.linalg.norm(np.diff(r[0][:, :2], axis=0), axis=1).sum()))[0]
            c = _polygon_centre(P[:, :2])
            if _inside(c, P[:, :2]):
                cen[i] = c
    have = np.flatnonzero(~np.isnan(cen[:, 0]))
    if not len(have):
        return None
    for c in range(2):
        cen[:, c] = np.interp(np.arange(len(rows)), have, cen[have, c])
    w = max(1, int(round(0.06 * size / hm)))
    ker = np.ones(2 * w + 1) / (2 * w + 1)
    cen = np.stack([np.convolve(np.pad(cen[:, c], w, mode="edge"), ker, mode="valid") for c in range(2)], 1)
    for i, z in enumerate(rows):
        raw = [r for r in model.cut(z) if r[3] and len(r[0]) >= 3]
        if not raw:
            continue
        P = max(raw, key=lambda r: float(np.linalg.norm(np.diff(r[0][:, :2], axis=0), axis=1).sum()))[0]
        pts, _ = _resample(P, True, 0.5 * hm)
        if len(pts) < 8 or not _inside(cen[i], pts[:, :2]):
            continue
        lp = Loop(pts[:, :2], pts, np.zeros(len(pts), np.int64), np.zeros((len(pts), 3)), float(z), True)
        t = _rays(lp, cen[i], th)
        ok = np.isfinite(t)
        xy, _ = lp.at(np.where(ok, t, 0.0))
        r = np.linalg.norm(xy - cen[i], axis=1)
        hull = _hull(pts[:, :2])
        hl = Loop(hull, np.c_[hull, np.zeros(len(hull))], np.zeros(len(hull), np.int64), np.zeros((len(hull), 3)),
                  float(z), True)
        te = _rays(hl, cen[i], th)
        okh = np.isfinite(te)
        he, _ = hl.at(np.where(okh, te, 0.0))
        re = np.where(okh, np.linalg.norm(he - cen[i], axis=1), r)
        renv[i] = re
        vx = pal.voxel(np.c_[xy, np.full(len(xy), z)])
        light[i] = np.where(vx >= 0, pal.lab[np.maximum(vx, 0), 0], pal.median)
        rec[i] = np.where(ok, re - r, 0.0)
        dark[i] = ok & (re - r > depth)
    renv = np.where(renv > 0, renv, radius)
    # clean: close small gaps, drop specks, keep holes off the top and the bottom of the body
    m = _erode(_dilate(dark, 1), 1)
    m = _dilate(_erode(m, 1), 1)
    lab = np.full(m.shape, -1)
    comps = []
    for i0, j0 in zip(*np.nonzero(m)):
        if lab[i0, j0] >= 0:
            continue
        stack, cells = [(i0, j0)], []
        lab[i0, j0] = len(comps)
        while stack:
            i, j = stack.pop()
            cells.append((i, j))
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, (j + dj) % m.shape[1]
                if 0 <= a < m.shape[0] and m[a, b] and lab[a, b] < 0:
                    lab[a, b] = len(comps)
                    stack.append((a, b))
        comps.append(np.asarray(cells))
    holes = np.zeros(m.shape, bool)
    least = (st.feature * s / hm) ** 2 * 2.0
    edge_rows = int(0.1 * len(rows))
    for c in comps:
        if len(c) < least or c[:, 0].min() < edge_rows or c[:, 0].max() >= len(rows) - edge_rows:
            continue
        deep = float(rec[c[:, 0], c[:, 1]].mean()) > st.carve_sure * depth  # deep enough to be carved whatever its colour
        darker = light[c[:, 0], c[:, 1]].mean() <= pal.median - 0.5 * st.hole
        uc = np.unique(c[:, 1])  # its width round the body (it may straddle the seam of the map)
        span = m.shape[1] - int(np.max(np.diff(np.r_[uc, uc[0] + m.shape[1]])))
        wide = renv[c[:, 0], c[:, 1]].mean() >= st.carve_body * renv.max()  # on the body, not in a top dimple
        if span > 0.5 * m.shape[1] or not wide or not (deep or darker):
            continue
        holes[c[:, 0], c[:, 1]] = True
    if not holes.any():
        return None
    # outlines, on a map rolled so that no hole crosses its seam
    free = np.flatnonzero(~holes.any(axis=0))
    shift = int(free[0]) if len(free) else 0
    rolled = np.roll(holes, -shift, axis=1)
    polys = []
    tol = st.corner_tol * st.feature * s
    for loop in _outlines(rolled):
        x = (loop[:, 0] + shift) * dth + theta0
        z = model.lo + loop[:, 1] * hm
        P = np.c_[x * radius, z]
        # simplify the closed line: split at its farthest pair, Douglas-Peucker on both halves
        far = int(np.argmax(np.linalg.norm(P - P[0], axis=1)))
        a = _douglas(P[:far + 1], tol)
        b = _douglas(np.vstack([P[far:], P[:1]]), tol) + far
        idx = np.r_[a, b[1:-1]]
        Q = np.c_[P[idx, 0] / radius, P[idx, 1]]
        if len(Q) >= 3 and abs(_signed_area(np.c_[Q[:, 0] * radius, Q[:, 1]])) >= least * hm * hm * 0.5:
            polys.append(Q)
    return _Carving(theta0, dth, model.lo, hm, radius, cen, holes, renv, polys)


def _stems(model: _Model, st: Style, s: float, d0: float, shape: Optional[_Shape] = None):
    """The thin parts (a stem, the points of a crown). On a figure: a drone every stem spacing up the middle
    of each. On a round body, its stem (thin loops in its top part): a tube of three drones round, turning a
    sixth of a turn a step, and a collar ring where it leaves the fruit.
    Returns (pos, tri, bary, the width below which a loop is thin, role (0 line, 1 tube, 2 collar, 3 neck))."""
    sc = st.column * s
    thin = st.thin * sc
    step = max(st.stem * s, 1.05 * d0)
    h = max(min(sc / 8.0, 0.5 * d0), 1e-3)
    tube = shape is not None and shape.round
    found = []
    wide = False  # a wide loop at the very top: no stem there (the thin neck of a goblet is no stem)
    for z in np.r_[np.arange(model.lo + 0.5 * step, model.hi, step), model.hi - 0.15 * step]:
        wide = False
        for lp in _loops(model, z, h, st, sc):
            if lp.closed and st.speck * sc <= lp.length and lp.length / math.pi < thin:
                found.append(lp)
            wide |= lp.closed and lp.length / math.pi >= thin
    neck = tube and wide  # a wide top: the thin loops are a neck (a goblet's), a tube in the body's colour
    if tube and not neck:  # a fruit's stem is in its top part
        found = [lp for lp in found if lp.z >= model.lo + st.stem_from * (model.hi - model.lo)]
    pos, tri, bary, role = [], [], [], []

    def put(p, lp, r):
        near = int(np.argmin(np.linalg.norm(lp.xy - p[:2], axis=1)))
        pos.append(p)
        tri.append(lp.tri[near])
        bary.append(lp.bary[near])
        role.append(r)

    if not tube:
        for lp in found:
            put(np.r_[lp.centre, lp.z], lp, 0)
    elif found:
        found.sort(key=lambda l: l.z)
        rt = 1.05 * d0 / math.sqrt(3.0)  # three drones round, d0 apart
        for i, lp in enumerate(found):
            r = max(rt, 0.8 * lp.length / (2 * math.pi))
            for k in range(3):
                a = (i % 2) * math.pi / 3 + k * 2 * math.pi / 3 - 0.5 * math.pi
                put(np.r_[lp.centre + r * np.array([math.cos(a), math.sin(a)]), lp.z], lp, 3 if neck else 1)
    if found and tube and not neck:
        foot = found[0]
        # round the stem itself, not round its flared foot (a wide collar seen edge-on is a bar)
        r_stem = float(np.median([max(rt, 0.8 * lp.length / (2 * math.pi)) for lp in found]))
        rc = min(max(rt, 0.8 * foot.length / (2 * math.pi)), r_stem + d0, rt + 1.5 * d0) + st.collar * d0
        n = max(6, int(2 * math.pi * rc // (1.1 * d0)))
        for k in range(n):
            a = -0.5 * math.pi + k * 2 * math.pi / n
            put(np.r_[foot.centre + rc * np.array([math.cos(a), math.sin(a)]), foot.z - 0.5 * step], foot, 2)
    if not pos:
        return np.zeros((0, 3)), np.zeros(0, np.int64), np.zeros((0, 3)), thin, np.zeros(0, np.int64)
    return np.asarray(pos), np.asarray(tri, np.int64), np.asarray(bary), thin, np.asarray(role, np.int64)


def _twist(out: _Drones, n0: int, shape: _Shape, amount: float) -> None:
    """Rings of a round body wound into one helix: each drone slides up or down its column by up to half a
    ring, by its angle round the axis, so front and back drones of a ring never stand at one height (seen
    level, they would hide one another)."""
    pos = np.vstack(out.pos)
    ring = np.concatenate(out.ring)
    col = np.concatenate(out.column)
    key = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(ring, col)) if c >= 0}
    f = amount * (np.mod((col / n0) + 0.25 + (shape.groove + 0.5 * math.pi) / (2 * math.pi), 1.0) - 0.5)
    new = pos.copy()
    for i in np.flatnonzero(col >= 0):
        r, c = int(ring[i]), int(col[i])
        up, down = key.get((r + 1, c)), key.get((r - 1, c))
        if f[i] > 0 and up is not None:
            new[i] = pos[i] + f[i] * (pos[up] - pos[i])
        elif f[i] < 0 and down is not None:
            new[i] = pos[i] - f[i] * (pos[down] - pos[i])
        elif f[i] > 0 and down is not None:
            new[i] = pos[i] - f[i] * (pos[down] - pos[i])
        elif f[i] < 0 and up is not None:
            new[i] = pos[i] + f[i] * (pos[up] - pos[i])
    k = 0
    for j, p in enumerate(out.pos):
        out.pos[j] = new[k:k + len(p)]
        k += len(p)


# ---------------------------------------------------------------- the net

@dataclass
class _Drones:
    pos: list = field(default_factory=list)
    tri: list = field(default_factory=list)
    bary: list = field(default_factory=list)
    kind: list = field(default_factory=list)
    ring: list = field(default_factory=list)
    column: list = field(default_factory=list)

    def add(self, xyz, tri, bary, kind, ring, column=None):
        xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
        n = len(xyz)
        self.pos.append(xyz)
        self.tri.append(np.asarray(tri, np.int64).reshape(n))
        self.bary.append(np.asarray(bary, np.float64).reshape(n, 3))
        self.kind.append(np.full(n, kind, np.int8))
        self.ring.append(np.full(n, ring, np.int64))
        self.column.append(np.full(n, -1, np.int64) if column is None else np.asarray(column, np.int64).reshape(n))


def _place_loop(loop: Loop, t, ids, sc: float, st: Style, new_id):
    """Columns at arc lengths ``t`` (with ids) carried onto ``loop``: crowded ones merged, gaps filled,
    nudged towards even spacing. Returns (t, ids)."""
    L = loop.length
    if loop.closed:
        if t is not None and len(t):
            order = np.argsort(t, kind="stable")
            t, ids = t[order], ids[order]
            start = int(np.argmin(ids))
            t, ids = np.roll(t, -start), np.roll(ids, -start)
            rel = np.mod(t - t[0], L)
            keep = [0]
            for i in range(1, len(rel)):
                if rel[i] - rel[keep[-1]] >= st.merge * sc:
                    keep.append(i)
            while len(keep) > 1 and L - rel[keep[-1]] < st.merge * sc:
                keep.pop()
            t0, ids, rel = t[0], ids[keep], rel[keep]
            gaps = np.diff(np.r_[rel, L])
            nt, ni = [], []
            for i, g in enumerate(gaps):
                nt.append(rel[i])
                ni.append(ids[i])
                if g > st.split * sc:
                    q = max(1, int(round(g / sc)) - 1)
                    for k in range(1, q + 1):
                        nt.append(rel[i] + g * k / (q + 1))
                        ni.append(new_id())
            rel, ids = np.asarray(nt), np.asarray(ni, np.int64)
            n = len(rel)
            if n > 1 and st.even > 0:
                even = np.arange(n) * (L / n)
                ang = (rel - even) / L * 2 * math.pi
                phase = math.atan2(np.sin(ang).mean(), np.cos(ang).mean())
                even = even + phase / (2 * math.pi) * L
                d = np.mod(even - rel + L / 2, L) - L / 2
                rel = rel + st.even * d
            return np.mod(t0 + rel, L), ids
        n = max(int(round(L / sc)), 1)
        n = max(4, 4 * int(round(n / 4))) if n >= 6 else n
        front = float(loop.arc[int(np.argmin(loop.xy[:, 1]))])  # the first drone faces the front (-y)
        return np.mod(front + np.arange(n) * (L / n), L), np.asarray([new_id() for _ in range(n)], np.int64)
    n = max(int(round(L / sc)), 1)
    return (np.arange(n) + 0.5) * (L / n), np.asarray([new_id() for _ in range(n)], np.int64)


def _ladder(n0: int, length: float, sc: float, st: Style) -> int:
    """Columns on a ring of ``length``: the base count halved while they would crowd."""
    n = n0
    while n > 1 and length / n < st.merge * sc:
        if n % 2:
            return 0
        n //= 2
    return n if n >= 3 else 0


def _base_columns(rings, thin: float, sc: float, shape: _Shape, st: Style) -> int:
    """Columns of a round body: about its ribs times a power of two (a smooth fruit: ``st.meridians``),
    always an odd count, so no column has another straight behind it (seen face on, two would stack into
    one bright spine)."""
    mains = [_main(r, thin) for r in rings]
    mains = [m for m in mains if m is not None]
    if not shape.round or not mains:
        return 0
    want = max(m.length for m in mains) / sc
    if shape.ribs >= st.rib_min:
        n = shape.ribs * 2 ** max(0, int(round(math.log2(max(want / shape.ribs, 1.0)))))
    else:
        n = st.meridians
    if n % 2 == 0:
        n = n - 1 if abs(n - 1 - want) < abs(n + 1 - want) and n > 3 else n + 1
    return n


def _net(model: _Model, s: float, st: Style, d0: float, shape: _Shape, n0=None, feats=(), pal=None, carv=None,
         stems=None, s_line=None, edges=()):
    """Drones on the rings and columns at base spacing ``s``, the feature lines first, the edge rings of a
    turned shape (``edges``, from ``_edge_rings``) as whole rings: (pos, tri, bary, kind, ring, colour or NaN, n0)."""
    sr = max(st.ring * s, 1.08 * d0)  # rings and columns keep clear of the minimum distance ...
    sc = max(st.ring * st.column * s * s / sr, 1.08 * d0)  # ... the area per drone kept
    levels = model.levels(sr)
    h = max(min(sc / 8.0, 0.5 * d0), 1e-3)
    thin = st.thin * sc  # a loop narrower than this (its length / pi) is a stem: one drone in its middle
    rings = [_loops(model, lv, h, st, sc) for lv in levels]
    if n0 is None:
        n0 = _base_columns(rings, thin, sc, shape, st)
    if n0:
        lengths = [(_main(r, thin).length if _main(r, thin) is not None else 0.0) for r in rings]
    else:
        lengths = [sum(l.length for l in r) for r in rings]
    seed = int(np.argmax(lengths)) if len(lengths) else 0
    counter = [max(n0, 0) + 1]

    def new_id():
        counter[0] += 1
        return counter[0]

    out = _Drones()
    placed = {}

    def dot(lp: Loop, k):
        near = int(np.argmin(np.linalg.norm(lp.xy - lp.centre, axis=1)))
        out.add(np.r_[lp.centre, lp.z], lp.tri[near], lp.bary[near], 0, k)
        return lp.centre[None], np.array([new_id()])

    caps = []
    if n0 and st.caps and rings:
        lmax = max(lengths)
        share = st.cap_share_smooth if shape.ribs < st.rib_min and st.smooth_rings else st.cap_share
        big_k = [k for k, L in enumerate(lengths) if L >= share * lmax]
        if big_k:  # the base ring; a top ring only when asked (seen edge-on it is a flat lid)
            caps = sorted({big_k[0], big_k[-1]}) if st.top_cap else [big_k[0]]
        if caps and edges and min(z for z, _ in edges) <= levels[caps[0]] + 2.0 * sr:
            caps = []  # the lowest edge ring is the base ring
    smooth = bool(n0) and shape.ribs < st.rib_min and st.smooth_rings and len(caps) == 2
    a_lat = max(st.lat_dot * s, 1.1 * d0)

    def columns(loops, k):
        """A ring of a round body: one drone per column, where the ray from the axis at the column's
        angle leaves the body (the outermost loop it meets), every other column dropped while they crowd."""
        big = [lp for lp in loops if lp.closed and lp.length / math.pi >= thin]
        if not big:
            return [], []
        if smooth and caps[0] < k < caps[-1]:  # a latitude ring, drawn whole
            xs, ns = [], []
            for lp in big:
                n = int(lp.length // a_lat)
                if n < 3:
                    continue
                front = float(lp.arc[int(np.argmin(lp.xy[:, 1]))])
                xy, near = lp.at(front + np.arange(n) * (lp.length / n))
                out.add(np.c_[xy, np.full(n, lp.z)], lp.tri[near], lp.bary[near], 2, k)
                xs.append(xy)
                ns.append(np.array([new_id() for _ in range(n)]))
            return xs, ns
        ids = np.arange(n0)
        ang = shape.groove + 2 * math.pi * ids / n0
        best_r = np.full(n0, -np.inf)
        which = np.full(n0, -1)
        tt = np.zeros(n0)
        main_l = max(big, key=lambda l: l.length)
        ctr = main_l.centre if _inside(main_l.centre, main_l.xy) else shape.axis
        for li, lp in enumerate(big):
            t = _rays(lp, ctr, ang)
            ok = np.isfinite(t)
            xy, _ = lp.at(np.where(ok, t, 0.0))
            r = np.where(ok, np.linalg.norm(xy - ctr, axis=1), -np.inf)
            better = r > best_r
            best_r = np.where(better, r, best_r)
            which = np.where(better, li, which)
            tt = np.where(better, t, tt)
        hit = which >= 0
        step = 1
        while True:
            sel = hit & (np.minimum(ids, n0 - ids) % step == 0)
            idx = np.flatnonzero(sel)
            if len(idx) < 3:
                break
            pts = np.stack([np.cos(ang[idx]), np.sin(ang[idx])], 1) * best_r[idx, None]
            gap = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
            nxt = np.roll(idx, -1)
            adjacent = np.mod(nxt - idx, n0) == step
            if not adjacent.any() or float(np.median(gap[adjacent])) >= st.merge * sc:
                break
            step *= 2
        sel = hit & (np.minimum(ids, n0 - ids) % step == 0)
        if sel.sum() < 3:
            main = max(big, key=lambda l: l.length)
            xy, nid = dot(main, k)
            return [xy], [nid]
        xs, ns = [], []
        for li, lp in enumerate(big):
            m = sel & (which == li)
            if not m.any():
                continue
            xy, near = lp.at(tt[m])
            out.add(np.c_[xy, np.full(len(xy), lp.z)], lp.tri[near], lp.bary[near], 2, k, ids[m])
            xs.append(xy)
            ns.append(ids[m])
        return xs, ns

    def do(k, prev):
        loops = rings[k]
        if n0:
            xy_all, id_all = columns(loops, k)
            for lp in loops:
                if lp.closed and st.speck * sc <= lp.length and lp.length / math.pi < thin and not (stems is not None and lp.length / math.pi < stems[3]):
                    xy, nid = dot(lp, k)
                    xy_all.append(xy)
                    id_all.append(nid)
            placed[k] = (np.vstack(xy_all) if xy_all else np.zeros((0, 2)),
                         np.concatenate(id_all) if id_all else np.zeros(0, np.int64))
            return
        xy_all, id_all = [], []
        pxy, pid = placed.get(prev, (np.zeros((0, 2)), np.zeros(0, np.int64)))
        owner = np.full(len(pxy), -1)
        tpar = np.zeros(len(pxy))
        if len(pxy) and loops:
            best = np.full(len(pxy), np.inf)
            for li, lp in enumerate(loops):
                if lp.length / math.pi < thin or not lp.closed:
                    continue
                d = np.linalg.norm(pxy[:, None, :] - lp.xy[None, :, :], axis=2)
                j = np.argmin(d, axis=1)
                dj = d[np.arange(len(pxy)), j]
                better = dj < best
                best = np.where(better, dj, best)
                owner = np.where(better, li, owner)
                tpar = np.where(better, lp.arc[j], tpar)
            owner[best > st.reach * sr + sc] = -1
        for li, lp in enumerate(loops):
            if lp.length < st.speck * sc:
                continue
            if lp.closed and lp.length / math.pi < thin:
                if stems is not None and lp.length / math.pi < stems[3]:
                    continue
                xy, ids = dot(lp, k)
            else:
                if lp.closed:
                    mine = owner == li
                    t, ids = _place_loop(lp, tpar[mine], pid[mine], sc, st, new_id)
                else:
                    t, ids = _place_loop(lp, None, None, sc, st, new_id)
                xy, near = lp.at(t)
                out.add(np.c_[xy, np.full(len(xy), lp.z)], lp.tri[near], lp.bary[near], 2, k)
            xy_all.append(xy)
            id_all.append(ids)
        placed[k] = (np.vstack(xy_all) if xy_all else np.zeros((0, 2)),
                     np.concatenate(id_all) if id_all else np.zeros(0, np.int64))

    if rings:
        do(seed, None)
        for k in range(seed + 1, len(rings)):
            do(k, k - 1)
        for k in range(seed - 1, -1, -1):
            do(k, k + 1)
    if stems is not None and len(stems[0]):
        out.add(stems[0], stems[1], stems[2], 0, -10 - stems[4])  # ring -10 line, -11 tube, -12 collar
    if n0 and st.twist and out.pos:
        _twist(out, n0, shape, st.twist)
    if edges:  # whole rings: the edges of a turned shape (ring -6) and the rings between them (-5)
        a_edge = max(st.edge_dot * max(s, s_line or s), 1.1 * d0)
        for z, k in edges:
            big = [lp for lp in _loops(model, z, h, st, sc) if lp.closed]
            if not big:
                continue
            lp = max(big, key=lambda l: l.length)
            n = int(lp.length // a_edge)
            if n < 3:
                continue
            front = float(lp.arc[int(np.argmin(lp.xy[:, 1]))])
            xy, near = lp.at(front + np.arange(n) * (lp.length / n))
            out.add(np.c_[xy, np.full(n, lp.z)], lp.tri[near], lp.bary[near], 1, -6 + k)
    if caps:
        if True:
            a_cap = max(st.cap * max(s, s_line or s), 1.1 * d0)
            for k in caps:
                lp = _main(rings[k], thin)
                n = int(lp.length // a_cap)
                if n < 3:
                    continue
                front = float(lp.arc[int(np.argmin(lp.xy[:, 1]))])
                xy, near = lp.at(front + np.arange(n) * (lp.length / n))
                out.add(np.c_[xy, np.full(n, lp.z)], lp.tri[near], lp.bary[near], 1, -4)
    if not out.pos:
        e = np.zeros((0, 3))
        return e, np.zeros(0, np.int64), e, np.zeros(0, np.int8), np.zeros(0, np.int64), e, n0
    pos = np.vstack(out.pos)
    tri = np.concatenate(out.tri)
    bary = np.vstack(out.bary)
    kind = np.concatenate(out.kind)
    ring = np.concatenate(out.ring)
    if caps:  # the cap lines replace their rings' column drones; below the base ring, nothing
        drop = (kind == 2) & ((ring <= caps[0]) | (ring == caps[-1]))
        pos, tri, bary, kind, ring = pos[~drop], tri[~drop], bary[~drop], kind[~drop], ring[~drop]
    if edges:  # a flat top or base beyond the outermost edge rings is drawn by its rim alone
        ez = [z for z, k in edges if k == 0]
        flat = np.abs(model.normals[tri, 2]) > st.flat_top
        beyond = (pos[:, 2] > max(ez) + 0.1 * sr) | (pos[:, 2] < min(ez) - 0.1 * sr)
        drop = (kind == 2) & flat & beyond
        pos, tri, bary, kind, ring = pos[~drop], tri[~drop], bary[~drop], kind[~drop], ring[~drop]
    col = np.full((len(pos), 3), np.nan)
    s_line = max(s, s_line or s)  # lines are never denser than at the budget's own spacing
    a = st.feature * s_line
    fpos, fcol = _place_features(feats, a, d0)
    fkind = np.ones(len(fpos), np.int8)
    if carv is not None:
        cpos, corner = carv.place(max(st.rim * s_line, 1.1 * d0), d0)
        fpos = np.vstack([cpos, fpos])
        fcol = np.vstack([np.repeat(_color.srgb_to_linear(np.asarray(st.rim_color))[None], len(cpos), axis=0), fcol])
        fkind = np.r_[np.where(corner, 0, 1).astype(np.int8), fkind]
        hole = (kind == 2) & (carv.inside(pos, st.hole_margin * sc) | carv.behind(pos, st.hole_margin * sc))
        pos, tri, bary, kind, ring, col = pos[~hole], tri[~hole], bary[~hole], kind[~hole], ring[~hole], col[~hole]
    if stems is not None and len(stems[0]) and (kind == 2).any():
        net = np.flatnonzero(kind == 2)
        dd, _ = nearest(pos[net], stems[0], sc)
        drop = np.zeros(len(pos), bool)
        drop[net[dd < st.stem_clear * sc]] = True
        pos, tri, bary, kind, ring, col = pos[~drop], tri[~drop], bary[~drop], kind[~drop], ring[~drop], col[~drop]
    if len(fpos):
        # the net clears the feature lines, and leaves the dark holes (carved eyes) empty
        net = kind == 2
        near = np.zeros(len(pos), bool)
        if net.any():
            dd, _ = nearest(pos[net], fpos, st.clear * sc)
            near[np.flatnonzero(net)[dd < st.clear * sc]] = True
        keep = ~near
        pos, tri, bary, kind, ring, col = pos[keep], tri[keep], bary[keep], kind[keep], ring[keep], col[keep]
        ftri = np.zeros(len(fpos), np.int64)
        fbary = np.full((len(fpos), 3), 1.0 / 3.0)
        pos = np.vstack([fpos, pos])
        tri = np.r_[ftri, tri]
        bary = np.vstack([fbary, bary])
        kind = np.r_[fkind, kind]
        ring = np.r_[np.full(len(fpos), -1), ring]
        col = np.vstack([fcol, col])
    if shape.round and st.flat_top < 1.0 and len(pos):
        # the flat top of a round body: its rings lie at one height, seen from the side a bright bar
        up = model.normals[tri, 2]
        top = (kind == 2) & (ring >= 0) & (up > st.flat_top) & (pos[:, 2] > model.lo + 0.5 * (model.hi - model.lo))
        pos, tri, bary, kind, ring, col = pos[~top], tri[~top], bary[~top], kind[~top], ring[~top], col[~top]
    order = np.lexsort((ring, kind))  # the minimum distance: stems, then features, then ring by ring
    keep = order[_outline.park(pos[order], d0 * 1.0005)]
    keep.sort()
    return pos[keep], tri[keep], bary[keep], kind[keep], ring[keep], col[keep], n0


# ---------------------------------------------------------------- entry point

def around(parts, options: Optional[DistributeOptions] = None, style: Optional[Style] = None) -> DroneResult:
    """Drones all round the model of ``parts``, at most ``options.count``, at least
    ``options.target_min_distance`` apart, on rings and columns, the borders between its colours drawn."""
    opt = options or DistributeOptions()
    st = style or Style()
    budget = int(opt.count)
    if budget < 1:
        raise ValueError("count must be at least 1")
    if not parts:
        raise ValueError("no parts to lay out")
    mesh = MeshData.concatenate([p.mesh for p in parts])
    offsets = np.cumsum([0] + [p.mesh.triangle_count for p in parts])
    tris = mesh.triangles
    areas, normals = triangle_areas_normals(tris)
    up = int(opt.up_axis)
    frame = [i for i in range(3) if i != up] + [up]
    inv = np.argsort(frame)
    model = _Model(tris[..., frame], normals[:, frame], areas)
    d0 = float(opt.target_min_distance)
    area = float(areas.sum())
    s_min = st.min_spacing * d0  # never denser: a legible net, even when the budget would allow more
    s_first = max(math.sqrt(area / budget), s_min)
    shape = _shape(model, 0.5 * s_first, st)
    if style is None and st.auto and shape.round:  # a round body: meridian lines (a cage), rings close
        st = replace(st, ring=st.cage[0], column=st.cage[1])
    if style is None and st.auto and shape.round and shape.ribs < st.rib_min and st.smooth_rings:
        st = replace(st, ring=st.smooth_style[0], column=st.smooth_style[1])  # a smooth fruit: latitude rings
    if shape.round:  # the first column faces the front: in the rib groove nearest it, or straight on
        if shape.ribs >= st.rib_min:
            w = 2 * math.pi / shape.ribs
            shape.groove += round((-0.5 * math.pi - shape.groove) / w) * w
        else:
            shape.groove = -0.5 * math.pi
    rng = np.random.default_rng(int(opt.seed))
    feats, pal, carv = [], None, None
    if st.features and opt.color_mode != "none":
        pal = _palette(mesh, model.T, areas, st.voxel * s_first, st, rng)
        pal.fill_recess(model)
        feats = [] if shape.round else _features(model, pal, st.feature * s_first, st)
        carv = _carvings(model, pal, shape, st, s_first)

    stems = _stems(model, st, s_first, d0, shape)
    stem = stems[4] != 3  # the stem of a fruit (not the neck of a goblet): no edges from its foot up
    edges = _edge_rings(model, shape, st, d0, s_first, float(stems[0][stem, 2].min()) if stem.any() else None)
    s = s_first
    best = None
    lo_s, hi_s = None, None
    first_n = None
    n0 = None
    for it in range(12):
        res = _net(model, s, st, d0, shape, n0 if it >= 2 else None, feats, pal, carv, stems, s_first, edges)
        n0 = res[6]
        n = len(res[0])
        if first_n is None:
            first_n = n
        if n <= budget and (best is None or n > len(best[1][0])):
            best = (s, res)
        if n <= budget:
            hi_s = s if hi_s is None else min(hi_s, s)
            if (n >= st.fit * budget or s <= s_min * 1.0001) and it >= 2:
                break
        else:
            lo_s = s if lo_s is None else max(lo_s, s)
        guess = s * math.sqrt(n / (0.985 * budget)) if n > 0 else s * 0.7
        if lo_s is not None and hi_s is not None and it >= 2:
            guess = min(max(guess, lo_s + 0.15 * (hi_s - lo_s)), hi_s - 0.15 * (hi_s - lo_s))
            if hi_s - lo_s < 2e-3 * s:
                break
        s = max(guess, s_min)
    if best is None:
        s *= 1.5
        res = _net(model, s, st, d0, shape, None, feats, pal, carv, stems, s_first, edges)
        best = (s, tuple(x[:budget] if isinstance(x, np.ndarray) else x for x in res))
    s, (pos, tri, bary, kind, ring, fcol, _) = best
    needed = len(pos)
    if s > s_min * 1.001:  # what the shape would take at the densest net it is drawn with
        needed = len(_net(model, s_min, st, d0, shape, n0, feats, pal, carv, stems, s_min, edges)[0])

    model_pos = pos.copy()
    pos = pos[:, inv]
    n = len(pos)
    # a feature drone sits off the surface (a smoothed line): the nearest surface point gives its triangle
    feat = np.flatnonzero(~np.isnan(fcol[:, 0]))
    if len(feat):
        cp, ct, cb = surface_candidates(tris, areas, int(min(300_000, 40 * n)), rng)
        _, j = nearest(pos[feat], cp, s)
        tri[feat], bary[feat] = ct[j], cb[j]
    colors = _colors(mesh, tris, areas, pos, tri, bary, s, opt)
    if shape.round and pal is not None and opt.color_mode != "none":
        # one palette: the body colour, nudged by the colour round each drone (texture speckle averaged out)
        vx = pal.voxel(model_pos)
        local = np.where((vx >= 0)[:, None], pal.lin[np.maximum(vx, 0)], colors)
        body = np.median(local[kind == 2], axis=0) if (kind == 2).any() else np.median(local, axis=0)
        colors = 0.5 * local + 0.5 * body
        colors = _color.glow(colors, opt.min_brightness)
        stem = (ring <= -10) & (ring >= -12)
        if stem.any():
            sc_ = _stem_color(local[stem & (ring == -11)] if (ring == -11).any() else local[stem], st)
            colors[stem] = sc_
            colors[ring == -12] = _full(sc_[None])[0] * 0.95
        rib = st.edge_rib_level if edges else st.rib_level_carved if carv is not None else st.rib_level
        colors[kind == 2] = _dim(colors[kind == 2], rib)
        colors[ring == -5] = _dim(colors[ring == -5], st.edge_fill_level)
        cap = ring == -4
        colors[cap] = _dim(colors[cap], st.cap_level)
    else:
        colors = _glow(colors, opt.min_brightness, st.contrast)
        net = kind == 2
        if feats or carv is not None:  # lines at full, the net dimmer
            colors[net] = _dim(colors[net], st.net_level)
    if len(feat):
        colors[feat] = _full(fcol[feat])
    dmin, dmean = _spacing(pos, s) if n > 1 else (0.0, 0.0)
    part = np.searchsorted(offsets, tri, side="right") - 1
    lo, hi = (pos.min(axis=0), pos.max(axis=0)) if n else (np.zeros(3), np.zeros(3))
    return DroneResult(
        positions=pos,
        normals=normals[tri],
        triangle_ids=tri,
        barycentrics=bary,
        colors_linear=colors,
        min_distance=dmin,
        mean_distance=dmean,
        area=area,
        pivot=scale_pivot(lo, hi, up),
        options=opt,
        visibility=np.ones(n),
        outline=kind <= 1,
        kind=kind,
        part=part,
        budget=budget,
        needed=int(max(needed, n)),
        spread=float(s / s_first),
    )


def _oklab_to_linear(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, np.float64).reshape(-1, 3)
    m1 = np.array([[1.0, 0.3963377774, 0.2158037573], [1.0, -0.1055613458, -0.0638541728],
                   [1.0, -0.0894841775, -1.2914855480]])
    lms = (lab @ m1.T) ** 3
    m2 = np.array([[4.0767416621, -3.3077115913, 0.2309699292], [-1.2684380046, 2.6097574011, -0.3413193965],
                   [-0.0041960863, -0.7034186147, 1.7076147010]])
    return np.clip(lms @ m2.T, 0.0, 1.0)


def _stem_color(linear: np.ndarray, st: Style) -> np.ndarray:
    """A stem's own colour (brown, olive), more saturated and at a middle lightness: not cream."""
    lab = _color.linear_to_oklab(np.clip(np.asarray(linear).reshape(-1, 3).mean(axis=0, keepdims=True), 0, None))[0]
    ab = lab[1:] * st.stem_chroma
    c = float(np.hypot(*ab))
    if c < 0.06:  # a grey stem keeps a hint of olive
        ab = ab + np.array([-0.02, 0.06 - c])
    return _oklab_to_linear(np.r_[st.stem_light, ab][None])[0]


def _full(linear: np.ndarray) -> np.ndarray:
    """The colours at the full brightness of their hue."""
    s = _color.linear_to_srgb(np.clip(linear, 0.0, None))
    v = np.maximum(s.max(axis=1), 1e-6)[:, None]
    return _color.srgb_to_linear(np.clip(s / v, 0.0, 1.0))


def _dim(linear: np.ndarray, level: float) -> np.ndarray:
    s = _color.linear_to_srgb(np.clip(linear, 0.0, None))
    return _color.srgb_to_linear(np.clip(s * level, 0.0, 1.0))


def _colors(mesh, tris, areas, pos, tri, bary, s, opt):
    """Colour of each drone: the mean colour of the surface round it (the fur of a coat is noisy)."""
    n = len(pos)
    if opt.color_mode == "none" or not n:
        return np.ones((n, 3))
    if opt.color_mode == "point":
        return _color.surface_colors(mesh, tri, bary)
    rng = np.random.default_rng(int(opt.seed) + 1)
    cp, ct, cb = surface_candidates(tris, areas, int(min(400_000, 60 * n)), rng)
    col = _color.surface_colors(mesh, ct, cb)
    d, owner = nearest(cp, pos, s)
    ok = d < 0.5 * s
    out = _color.average_by_owner(col[ok], owner[ok], n)
    got = np.bincount(owner[ok], minlength=n) > 0
    if not got.all():
        out[~got] = _color.surface_colors(mesh, tri[~got], bary[~got])
    return out


def _glow(linear: np.ndarray, floor: float, contrast: float) -> np.ndarray:
    """``color.glow`` with the brightness range of the model stretched: the darkest drones (dark fur)
    at ``floor``, the brightest (a face, a crown) at full, so parts of one hue apart by light read apart."""
    c = np.asarray(linear, np.float64).reshape(-1, 3)
    if not len(c) or not floor > 0:
        return c
    if contrast <= 0:
        return _color.glow(c, floor)
    s = _color.linear_to_srgb(np.clip(c, 0.0, None))
    v = s.max(axis=1)
    lo, hi = float(np.percentile(v, 2)), float(np.percentile(v, 98))
    hi = max(hi, lo + 0.05, _color.GLOW_STRETCH)
    x = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    full = s / np.maximum(v, 1e-6)[:, None]
    hue = np.clip((v - _color.GLOW_NEAR_BLACK[0]) / (_color.GLOW_NEAR_BLACK[1] - _color.GLOW_NEAR_BLACK[0]), 0.0, 1.0)
    keep = np.clip(v / hi, 0.0, 1.0)
    level = floor + (1.0 - floor) * ((1.0 - contrast) * keep + contrast * x)
    return _color.srgb_to_linear((hue[:, None] * full + (1.0 - hue[:, None])) * level[:, None])


# ---------------------------------------------------------------- variants

def around_cage(parts, options: Optional[DistributeOptions] = None) -> DroneResult:
    """Silhouette first: rings twice as close, columns twice as far apart (meridian lines, a cage)."""
    return around(parts, options, Style(ring=0.6, column=1.7))


def around_net(parts, options: Optional[DistributeOptions] = None) -> DroneResult:
    """An even net of rings and columns on every model (no cage for round bodies)."""
    return around(parts, options, Style(auto=False))


def around_helix(parts, options: Optional[DistributeOptions] = None) -> DroneResult:
    """The rings of a round body wound into a helix (fewer drones hiding one another seen level)."""
    return around(parts, options, Style(twist=1.0))


def classify(parts, options: Optional[DistributeOptions] = None) -> str:
    """"round" when the model is a round body (one loop round a vertical axis at nearly every height: a
    fruit, a ball, a vase), else "figure"."""
    opt = options or DistributeOptions()
    mesh = MeshData.concatenate([p.mesh for p in parts])
    tris = mesh.triangles
    areas, normals = triangle_areas_normals(tris)
    up = int(opt.up_axis)
    frame = [i for i in range(3) if i != up] + [up]
    model = _Model(tris[..., frame], normals[:, frame], areas)
    s = max(math.sqrt(float(areas.sum()) / max(int(opt.count), 1)), Style().min_spacing * opt.target_min_distance)
    return "round" if _shape(model, 0.5 * s, Style()).round else "figure"


def around_round(parts, options: Optional[DistributeOptions] = None) -> DroneResult:
    """The fruit layout: a cage of meridian lines (odd count, in the rib grooves), cap rings at the base and
    the shoulder, the stem as a tube with a collar, carved holes as bright rims."""
    return around(parts, options)
