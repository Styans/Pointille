"""Outline: drones on the lines that draw the model, closer together than the fill.

Spread evenly, drones blur a model: nothing marks its silhouette, its edges,
its eyes and mouth or the picture painted on it. An outline puts some of the
drones right on those lines, denser than the fill, and the fill keeps clear of
them. The lines come from five sources, each optional:

* borders    - mesh edges with a single triangle (text, logos, planes);
* creases    - edges whose triangles meet at more than ``crease_angle``;
* silhouette - the outline of the model as the middle viewer sees it, and
  where one part of it covers another (an arm in front of the body);
* relief     - what a line drawing of the model shows inside the silhouette:
  where the surface bends by more than ``relief_angle`` within a short stretch
  (eyes, nostrils, mouth, folds);
* colors     - where the colour of the model jumps by more than
  ``color_contrast`` (a flag, text, painted eyes).

Silhouette, relief and colours are read from views of the model, like a
picture: the model is rendered (a visibility buffer, in numpy) from the middle
viewer, or for relief and colours also from all around for a show seen from
every side. The outline is that of the picture smoothed over ``VIEW_SIGMA``
pixels, so fur and noise of an AI model do not fray it; relief and colours
come from an edge detector in the spirit of Canny: curvature or colour change
per length, smoothed alike, its crests, hysteresis, thinning. Every line pixel
knows the triangle and barycentrics it shows, so the lines lie on the surface.

All lines are chained between corners: where lines end or meet, and where they
bend sharply, judged over ``CORNER_WINDOW`` so that the zigzag of a dense,
noisy mesh or of pixels is no bend. Every corner gets a drone and the drones
between are spaced evenly along the line, measured at that same scale; a short
relaxation along the lines then settles where two lines come close.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .color import linear_to_oklab, surface_colors
from .facade import normalize, rotate, view_basis
from .relax import SurfaceProjector, relax
from .sampling import poisson_radius
from .spatial import Grid, _expand_ranges, nearest

AMOUNTS = ("density", "share")
STEP = 1.0 / 16  # line points per fill spacing: fine enough for drones to slide along
PIECE = 4  # segments are cut into pieces of at most this many steps for the sight tests
MIN_LENGTH = 1.0  # lines that all fit in less than one fill spacing are noise
MAX_SHARE = 0.5  # "density" puts at most half of the drones on lines
MIN_SPACING = 0.4  # "share" never packs a line closer than this many fill spacings
GAP = 0.75  # the fill keeps this many fill spacings clear of the lines
CORNER_TURN = 45.0  # a line bending sharper than this (degrees) keeps a drone in the corner
CORNER_GAP = 0.8  # corner drones closer than this many line spacings to a stronger one are dropped
CORNER_WINDOW = 0.25  # bends and lengths are judged over this many fill spacings of line
LINE_GAP = 0.7  # where two lines run close, drones nearer than this many line spacings to another go
GRADE = 0.2  # with no count to meet, the spacing grows by this much per length of line away from a corner
VIEW_PIXEL = 1.0 / 12  # pixel of the views relief and colours are read from, in fill spacings
VIEW_SIGMA = 2.0  # the views are smoothed over this many pixels: finer detail draws no line
_INSET = 1e-3  # points on a mesh edge sit a hair inside their triangle, so sight tests do not graze it
_VIEW_NODES = 1 << 40  # node names of the view lines, apart from the segments' ones
_MAX_PIXELS = 4_000_000
_RASTER_PAIRS = 1 << 22
_CHUNK = 1 << 16


@dataclass
class _Segs:
    """Straight segments inside triangles, from barycentrics ``b0`` to ``b1``;
    ``k0`` / ``k1`` name their end nodes, so that segments chain into lines."""

    tri: np.ndarray
    b0: np.ndarray
    b1: np.ndarray
    k0: np.ndarray
    k1: np.ndarray

    def take(self, keep) -> "_Segs":
        return _Segs(self.tri[keep], self.b0[keep], self.b1[keep], self.k0[keep], self.k1[keep])

    def ends(self, tris):
        return np.einsum("ij,ijk->ik", self.b0, tris[self.tri]), np.einsum("ij,ijk->ik", self.b1, tris[self.tri])

    @staticmethod
    def concatenate(parts: List["_Segs"]) -> "_Segs":
        if not parts:
            i, b = np.zeros(0, np.int64), np.zeros((0, 3))
            return _Segs(i, b, b, i, i)
        return _Segs(*(np.concatenate([getattr(p, f) for p in parts]) for f in ("tri", "b0", "b1", "k0", "k1")))


@dataclass
class _Chain:
    """A line from corner ``start`` to corner ``end`` (or round a loop): its
    nodes ``p`` in walking order, a loop's last node back on its first.
    Segment pieces lie in triangle ``tri`` (barycentrics ``ba`` to ``bb``);
    colour pieces have no triangle, their points are put back onto the
    surface. ``node_tri`` / ``node_bary`` put each node on the surface.
    Distances along the line are chords between points ``res`` apart, so
    wiggles smaller than that do not count. A line read from a view is
    measured in the picture, ``p`` its nodes there (as the audience sees it:
    drones evenly spaced in the picture are no closer in space), and ``q``
    holds the same nodes on the model."""

    p: np.ndarray
    tri: Optional[np.ndarray]
    ba: Optional[np.ndarray]
    bb: Optional[np.ndarray]
    node_tri: np.ndarray
    node_bary: np.ndarray
    closed: bool
    start: int
    end: int
    res: float
    q: Optional[np.ndarray] = None

    def __post_init__(self):
        self.t = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.p, axis=0), axis=1))]  # arc length at the nodes
        k = max(2, int(math.ceil(self.t[-1] / self.res)) + 1)
        self.tk = np.linspace(0.0, self.t[-1], k)
        self.ck = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.point(self.tk), axis=0), axis=1))]

    @property
    def length(self) -> float:
        return float(self.ck[-1])

    def point(self, arc: np.ndarray) -> np.ndarray:
        """Positions at arc lengths ``arc`` (round and round a loop, clamped to the ends otherwise)."""
        total = self.t[-1]
        arc = np.mod(arc, total) if self.closed and total > 0 else np.clip(arc, 0.0, total)
        return np.stack([np.interp(arc, self.t, self.p[:, i]) for i in range(3)], 1)

    def at(self, x: np.ndarray, tris, surface):
        """Points at distances ``x`` along the line: (pos, tri, bary)."""
        pos, tri, bary, found = self.at_found(x, tris, surface)
        return pos[found], tri[found], bary[found]

    def at_found(self, x: np.ndarray, tris, surface):
        """Points at distances ``x`` along the line: (pos, tri, bary, found); a point of a line read
        from a view that does not land on the surface is not found."""
        arc = np.interp(x, self.ck, self.tk)
        i = np.clip(np.searchsorted(self.t, arc, side="right") - 1, 0, len(self.t) - 2)
        f = np.clip((arc - self.t[i]) / np.maximum(self.t[i + 1] - self.t[i], 1e-300), 0.0, 1.0)[:, None]
        if self.tri is not None:
            bary = self.ba[i] * (1.0 - f) + self.bb[i] * f
            tri = self.tri[i]
            return np.einsum("ij,ijk->ik", bary, tris[tri]), tri, bary, np.ones(len(x), bool)
        q = self.p if self.q is None else self.q
        return surface.project(q[i] * (1.0 - f) + q[i + 1] * f)

    def node(self, i: int) -> np.ndarray:
        """Node ``i`` on the model."""
        return (self.p if self.q is None else self.q)[i]

    def cut(self, a: int, b: int, start: int, end: int) -> "_Chain":
        """The open line from node ``a`` to node ``b``; on a loop ``b`` may run past its last node."""
        n = len(self.p) - 1
        nodes = np.arange(a, b + 1) % n if self.closed else np.arange(a, b + 1)
        pieces = nodes[:-1]

        def take(x):
            return None if x is None else x[pieces]

        return _Chain(self.p[nodes], take(self.tri), take(self.ba), take(self.bb), self.node_tri[nodes],
                      self.node_bary[nodes], False, start, end, self.res, None if self.q is None else self.q[nodes])


@dataclass
class Lines:
    pos: np.ndarray  # (K, 3) dense points along all lines
    tri: np.ndarray  # (K,)
    bary: np.ndarray  # (K, 3)
    color: np.ndarray  # (K, 3) linear colour a drone there shows; NaN: the colour under it
    step: float  # about the distance between neighbouring line points
    chains: List[_Chain]
    corners: tuple  # (node, pos, tri, bary), strongest first
    surface: Optional[SurfaceProjector] = None  # puts drones on colour borders onto the surface
    # of each corner: 2 + degree where lines end (3) or meet, up to 2 for a bend (1 - cos of its angle)
    strength: Optional[np.ndarray] = None

    @property
    def length(self) -> float:
        return float(sum(c.length for c in self.chains))

    def colors_at(self, points: np.ndarray) -> np.ndarray:
        """The colour of the line point nearest to each of ``points``."""
        _, idx = nearest(points, self.pos, 2.0 * self.step)
        return self.color[idx]


class LineProjector:
    """Snaps drones to the nearest line point: the relaxation's projector for an outline."""

    def __init__(self, pos, tri, bary, max_move: float, step: float):
        self.pos, self.tri, self.bary = pos, tri, bary
        self.max_move = max_move
        self.reach = max_move + 2.0 * step
        self.grid = Grid(pos, self.reach)

    def project(self, points: np.ndarray):
        """Returns (positions, triangle_ids, barycentrics, found)."""
        qi, pj, d = self.grid.query_radius(points, self.reach)
        order = np.lexsort((d, qi))
        first = np.ones(len(order), bool)
        first[1:] = qi[order][1:] != qi[order][:-1]
        rows, best = qi[order[first]], pj[order[first]]
        found = np.zeros(len(points), bool)
        found[rows] = True
        out_pos = points.copy()
        out_tri = np.full(len(points), -1, np.int64)
        out_bary = np.zeros((len(points), 3))
        out_pos[rows], out_tri[rows], out_bary[rows] = self.pos[best], self.tri[best], self.bary[best]
        return out_pos, out_tri, out_bary, found


def find_lines(mesh, areas, normals, weights, opt, spacing: float, rng, audience=None) -> Optional[Lines]:
    """The lines to outline; None when the model has none.

    ``spacing`` is the fill spacing: it sets how fine the line points are and
    how small a line may be. ``weights`` are the fill's triangle weights (zero
    where nobody may see the surface); in facade mode ``audience`` drops what
    nobody sees. ``rng`` is not used: the lines do not depend on the seed.
    """
    tris = mesh.triangles
    step = spacing * STEP
    window = CORNER_WINDOW * spacing
    live = areas > 0
    sight = _sight(opt)
    smooth_cos = math.cos(math.radians(opt.crease_angle))

    kinds = []  # segment lines, the most certain first
    views = opt.outline_silhouette or opt.outline_relief or opt.outline_colors
    if opt.outline_borders or opt.outline_creases or views:
        vid, verts = _weld(tris)
        edge, va, vb = _edges(vid)
        next_key = len(verts) + int(edge.max()) + 1  # vertices, then edges, then cuts
    if views:
        cn = _corner_normals(vid, normals, areas, smooth_cos)
    if opt.outline_borders or opt.outline_creases:
        border, crease = _feature_edges(tris, normals, live, edge, va, vb, smooth_cos, sight)
        for on, he in ((opt.outline_borders, border), (opt.outline_creases, crease)):
            if on and len(he):
                segs = _Segs(he // 3, _onehot(he % 3), _onehot((he + 1) % 3), va[he], vb[he])
                segs, next_key = _cut(segs, tris, PIECE * step, next_key)
                if audience is not None:
                    segs = segs.take(audience.visibility(_mid(segs, tris), segs.tri) > 0)
                kinds.append(segs)

    # a line two sources find (a coloured edge is a crease and a colour border) is drawn once
    kept, dense = [], np.zeros((0, 3))
    for segs in kinds:
        if len(dense) and len(segs.tri):
            segs = segs.take(~_near(_mid(segs, tris), dense, 2.0 * step))
        if len(segs.tri):
            kept.append(segs)
            dense = np.concatenate([dense, _points(segs, tris, step)[0]])
    segs = _Segs.concatenate(kept)
    # lines that all fit in a small spot are noise: a nick in the mesh, a speck of silhouette
    if len(segs.tri):
        p0, p1 = segs.ends(tris)
        segs = segs.take(_wide(segs.k0, segs.k1, p0, p1, MIN_LENGTH * spacing))
        p0, p1 = segs.ends(tris)
        segs = segs.take(_unspurred(segs.k0, segs.k1, p0, p1, window))

    pos, tri, bary = _points(segs, tris, step)
    pos, tri, bary, color = [pos], [tri], [bary], [np.full((len(pos), 3), np.nan)]
    chains, corners = [], []
    if len(segs.tri):
        p0, p1 = segs.ends(tris)
        prefer = np.einsum("ij,ij->i", normals[np.concatenate([segs.tri, segs.tri])], sight(np.concatenate([p0, p1])))
        walks, (node, end, strength) = _chains(segs.k0, segs.k1, p0, p1, prefer)
        for e, fw, closed, a, b in walks:
            f = fw[:, None]
            ba, bb = np.where(f, segs.b0[e], segs.b1[e]), np.where(f, segs.b1[e], segs.b0[e])
            nodes = np.concatenate([np.where(f, p0[e], p1[e]), np.where(f, p1[e], p0[e])[-1:]])
            chains.append(_Chain(nodes, segs.tri[e], ba, bb, np.r_[segs.tri[e], segs.tri[e][-1:]],
                                 np.concatenate([ba, bb[-1:]]), closed, a, b, 0.5 * window))
        piece, first = end % len(segs.tri), end < len(segs.tri)
        corners.append((node, np.where(first[:, None], p0[piece], p1[piece]), segs.tri[piece],
                        np.where(first[:, None], segs.b0[piece], segs.b1[piece]), strength))

    # silhouette, relief and colours, read from views of the model
    surface = None
    if views:
        on = np.flatnonzero(live)
        surface = SurfaceProjector(tris[on], on, 0.5 * spacing, 0.5 * spacing)
        dense = np.concatenate(pos)
        block = 0
        for index, (eye, toward) in enumerate(_views(opt, tris)):
            render = _render(tris, eye, toward, VIEW_PIXEL * spacing)
            if render is None:
                continue
            for kind in ("silhouette", "relief", "colors"):
                if not getattr(opt, f"outline_{kind}") or (kind == "silhouette" and index):
                    continue  # the outline is the middle viewer's
                found = _view_lines(mesh, tris, cn, render, eye, toward, opt, kind)
                block += 1
                if found is None:
                    continue
                rows, cols, p, t, b, c = found
                keep = ~_near(p, dense, 2.0 * step)  # drawn already, from another source or view
                rows, cols, p, t, b, c = rows[keep], cols[keep], p[keep], t[keep], b[keep], c[keep]
                i, j = _pixel_graph(rows, cols)
                if kind != "silhouette":
                    # neighbours in the picture on two parts of the model, one in front of the other, are two
                    # lines; the outline is one line as the audience sees it, wherever its points lie in depth
                    near = np.linalg.norm(p[i] - p[j], axis=1) < 8.0 * render["h"]
                    i, j = i[near], j[near]
                if not len(i):
                    continue
                m = np.stack([cols, -rows, np.zeros(len(rows))], 1) * render["h"]  # the picture
                wide = _wide(i, j, m[i], m[j], MIN_LENGTH * spacing)
                i, j = i[wide], j[wide]
                if len(i):
                    short = _unspurred(i, j, m[i], m[j], window)
                    i, j = i[short], j[short]
                if not len(i):
                    continue
                base = block * _VIEW_NODES
                walks, (node, end, strength) = _chains(i + base, j + base, m[i], m[j], np.zeros(2 * len(i)))
                for e, fw, closed, a, bb in walks:
                    k = np.r_[np.where(fw, i[e], j[e]), np.where(fw, j[e], i[e])[-1:]]
                    chains.append(_Chain(m[k], None, None, None, t[k], b[k], closed, a, bb, 0.5 * window, p[k]))
                k = np.where(end < len(i), i[end % len(i)], j[end % len(i)])
                corners.append((node, p[k], t[k], b[k], strength))
                used = np.unique(np.concatenate([i, j]))
                for out, part in zip((pos, tri, bary, color), (p[used], t[used], b[used], c[used])):
                    out.append(part)
                dense = np.concatenate([dense, p[used]])
    pos, tri, bary, color = (np.concatenate(x) for x in (pos, tri, bary, color))
    if not len(pos) or not chains:
        return None
    chains, bends = _bends(chains, window)
    corners.append(bends)
    node, c_pos, c_tri, c_bary, strength = (np.concatenate([c[i] for c in corners]) for i in range(5))
    order = np.argsort(-strength, kind="stable")
    return Lines(pos, tri, bary, color, step, chains, (node[order], c_pos[order], c_tri[order], c_bary[order]),
                 surface, strength[order])


def split(lines: Lines, cand_pos: np.ndarray, area: float, count: int, opt):
    """How many of the ``count`` drones go on the lines and how far apart, and
    which fill candidates make room for them.

    Returns (line drones, their spacing, fill candidate mask, fill area);
    ``area`` is the (density-weighted) area of all candidates.
    """
    free = np.ones(len(cand_pos), bool)
    fill_area, k = area, 0
    length = lines.length
    d_fill = 2.0 * poisson_radius(area, count)
    for _ in range(3):  # the gap depends on the fill spacing and that on the gap
        free = ~_near(cand_pos, lines.pos, GAP * d_fill)
        fill_area = area * max(float(free.mean()), 0.05)
        k = line_count(opt, count, length, fill_area)
        d_fill = 2.0 * poisson_radius(fill_area, count - k)
    if k < 3:
        return 0, 0.0, np.ones(len(cand_pos), bool), area
    if free.sum() < 2 * (count - k):  # lines everywhere: the fill runs over them
        free, fill_area = np.ones(len(cand_pos), bool), area
    return k, length / k, free, fill_area


def line_count(opt, count: int, length: float, fill_area: float) -> int:
    """Drones for lines of ``length``; the other drones fill ``fill_area``."""
    if opt.outline_amount == "share":
        k = min(float(opt.outline_share) * count, _fit(count, length, fill_area, 1.0 / MIN_SPACING))
    else:
        k = min(_fit(count, length, fill_area, max(float(opt.outline_density), 1e-3)), MAX_SHARE * count)
    return int(min(max(round(k), 0), count - 1))


def _fit(count: int, length: float, fill_area: float, ratio: float) -> float:
    """Drones k on lines spaced ``1 / ratio`` of the fill spacing ``2 r(fill_area, count - k)``:
    ``k = a * sqrt(count - k)``, solved for k."""
    a2 = (ratio * length / (2.0 * poisson_radius(fill_area, 1))) ** 2
    return 0.5 * (math.sqrt(a2 * a2 + 4.0 * a2 * count) - a2)


def place(lines: Lines, count: int, spacing: float, iterations: int, tris: np.ndarray):
    """At most ``count`` drones on the lines, ``spacing`` apart: a drone in every
    corner and evenly spaced drones along the lines between, then relaxed along
    the lines where two of them come close. Fewer when the lines do not hold
    ``count`` (a corner takes the place of the drones next to it) and where
    lines run closer than ``LINE_GAP`` spacings: one drone there stands for both.

    Returns (positions, triangle_ids, barycentrics).
    """
    c_node, c_pos, c_tri, c_bary = lines.corners
    chosen = []
    for i in range(len(c_node)):
        if len(chosen) >= count:
            break
        if not chosen or np.linalg.norm(c_pos[chosen] - c_pos[i], axis=1).min() >= CORNER_GAP * spacing:
            chosen.append(i)
    chosen = np.asarray(chosen, np.int64)
    # how far into a line its first drone stands, in drone spacings: one from a corner drone,
    # a bit more from a corner merged into a stronger one nearby, half where no drone stands
    lead = np.full(len(c_node), 0.5)
    if len(chosen):
        gap, _ = nearest(c_pos, c_pos[chosen], CORNER_GAP * spacing)
        merged = gap < CORNER_GAP * spacing
        lead[merged] = 1.0 + gap[merged] / spacing
    lead = dict(zip(c_node.tolist(), lead.tolist()))

    # drones per line: as many as fit at ``spacing``, rounded so that they add up to ``count``
    length = np.array([c.length for c in lines.chains])
    held = np.array([[0.0, 0.0] if c.closed else [lead.get(c.start, 0.5), lead.get(c.end, 0.5)]
                     for c in lines.chains])
    closed = np.array([c.closed for c in lines.chains], bool)
    want = np.where(closed, length / spacing, length / spacing - held.sum(axis=1) + 1.0)
    many = _allocate(want, count - len(chosen))

    p, t, b = [c_pos[chosen]], [c_tri[chosen]], [c_bary[chosen]]
    for chain, m, ell, (h0, h1) in zip(lines.chains, many, length, held):
        if m <= 0:
            continue
        if chain.closed:
            x = np.arange(m) * (ell / m)
        else:
            u = ell / (m - 1 + h0 + h1)
            x = h0 * u + np.arange(m) * u
        for out, part in zip((p, t, b), chain.at(x, tris, lines.surface)):
            out.append(part)
    p, t, b = np.concatenate(p), np.concatenate(t), np.concatenate(b)
    n_corners = len(chosen)
    if iterations > 0 and len(p) > n_corners:
        projector = LineProjector(lines.pos, lines.tri, lines.bary, 0.15 * spacing, lines.step)
        rp, rt, rb = relax(p[n_corners:], t[n_corners:], b[n_corners:], projector, spacing, iterations,
                           fixed=p[:n_corners] if n_corners else None)
        p = np.concatenate([p[:n_corners], rp])
        t = np.concatenate([t[:n_corners], rt])
        b = np.concatenate([b[:n_corners], rb])
    keep = _park(p, LINE_GAP * spacing)  # the corners first
    return p[keep], t[keep], b[keep]


def place_graded(lines: Lines, near: float, spacing: float, tris: np.ndarray):
    """Drones on the lines with no count to meet, densest at the corners: a
    drone in every corner, the next ones ``near`` from it, the spacing growing
    by ``GRADE`` per length of line to ``spacing``; loops with no corner get
    ``spacing`` all round. The loose end of a line keeps a drone but is no
    corner: a noisy model breaks its lines into many pieces. Where lines run
    close, one drone stands for both.

    Returns (positions, triangle_ids, barycentrics, kind, local spacing);
    ``kind`` is 0 for the corner drones (they come first) and 1 on the lines.
    """
    c_node, c_pos, c_tri, c_bary = lines.corners
    strength = lines.strength if lines.strength is not None else np.full(len(c_node), 4.0)
    chosen = []
    for i in range(len(c_node)):
        if not chosen or np.linalg.norm(c_pos[chosen] - c_pos[i], axis=1).min() >= CORNER_GAP * spacing:
            chosen.append(i)
    chosen = np.asarray(chosen, np.int64)
    loose = c_node[chosen[strength[chosen] == 3.0]].tolist()  # 2 + degree 1
    # how far into a line its first drone stands, in local spacings: one from a corner drone, half
    # from a corner merged into a stronger one nearby (that one may lie off the line: drones too close
    # to it go below); the spacing grows away from either. Elsewhere half a spacing and no growing
    lead, graded = {}, set()
    if len(chosen):
        gap, _ = nearest(c_pos, c_pos[chosen], CORNER_GAP * spacing)
        for node, g in zip(c_node.tolist(), gap.tolist()):
            if g < CORNER_GAP * spacing:
                lead[node] = 1.0 if g <= 0 else 0.5
                graded.add(node)
    for node in loose:
        lead[node] = 1.0  # the next drone a whole line spacing in
        graded.discard(node)

    corner = strength[chosen] != 3.0
    order = np.argsort(~corner, kind="stable")  # the corners first, then the loose ends
    chosen, corner = chosen[order], corner[order]
    p, t, b = [c_pos[chosen]], [c_tri[chosen]], [c_bary[chosen]]
    s = [np.where(corner, near, spacing)]
    for chain in lines.chains:
        ell = chain.length
        if ell <= 0:
            continue
        if chain.closed:
            m = max(int(round(ell / spacing)), 1)
            x = np.arange(m) * (ell / m)
            sigma = np.full(m, spacing)
        else:
            # F(x) counts local spacings from the start: drones stand at whole steps of it
            k = max(int(math.ceil(ell / (0.125 * near))), 2) + 1
            xs = np.linspace(0.0, ell, k)
            dist = np.full(k, np.inf)
            if chain.start in graded:
                dist = np.minimum(dist, xs)
            if chain.end in graded:
                dist = np.minimum(dist, ell - xs)
            local = np.minimum(spacing, near + GRADE * dist)
            F = np.r_[0.0, np.cumsum(np.diff(xs) * 0.5 * (1.0 / local[1:] + 1.0 / local[:-1]))]
            h0, h1 = lead.get(chain.start, 0.5), lead.get(chain.end, 0.5)
            room = F[-1] - h0 - h1
            if room < 0:
                continue  # the corners at both ends stand close enough
            m = int(round(room)) + 1
            at = np.array([h0 + 0.5 * room]) if m == 1 else h0 + np.arange(m) * (room / (m - 1))
            x = np.interp(at, F, xs)
            sigma = np.interp(x, xs, local)
        pos, tri, bary, found = chain.at_found(x, tris, lines.surface)
        for out, part in zip((p, t, b, s), (pos, tri, bary, sigma)):
            out.append(part[found])
    p, t, b, s = (np.concatenate(x) for x in (p, t, b, s))
    kind = np.ones(len(p), np.int8)
    kind[: int(corner.sum())] = 0
    keep = park(p, np.maximum(near, LINE_GAP * s))  # the corners first
    return p[keep], t[keep], b[keep], kind[keep], s[keep]


def park(points: np.ndarray, radius) -> np.ndarray:
    """Indices of the points (taken in order) farther than their own ``radius`` (one for all, or one
    per point) from every one kept before."""
    radius = np.broadcast_to(np.asarray(radius, dtype=np.float64), (len(points),))
    if not len(points):
        return np.zeros(0, np.int64)
    cell = float(radius.max())
    cells = {}
    kept = []
    keys = np.floor(points / cell).astype(np.int64).tolist()
    for n, (p, (x, y, z), r) in enumerate(zip(points.tolist(), keys, (radius * radius).tolist())):
        if any((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2 < r
               for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
               for q in cells.get((x + dx, y + dy, z + dz), ())):
            continue
        cells.setdefault((x, y, z), []).append(p)
        kept.append(n)
    return np.asarray(kept, np.int64)


def _allocate(want: np.ndarray, total: int) -> np.ndarray:
    """Whole numbers close to ``want`` (at most one above it rounded down, none where it is 0) that
    add up to ``total``, by largest remainders; less than ``total`` when even that is too few."""
    want = np.maximum(want, 0.0)
    m = np.floor(want).astype(np.int64)
    diff = max(int(total), 0) - int(m.sum())
    if diff > 0:
        up = np.flatnonzero(want > 0)
        m[up[np.argsort(-(want[up] - m[up]), kind="stable")][:diff]] += 1
    while diff < 0:
        have = np.flatnonzero(m > 0)
        if not len(have):
            break
        drop = have[np.argsort(want[have] - m[have], kind="stable")][:-diff]
        m[drop] -= 1
        diff += len(drop)
    return m


def _near(queries: np.ndarray, points: np.ndarray, radius: float) -> np.ndarray:
    """True for every query closer than ``radius`` to some point."""
    out = np.zeros(len(queries), bool)
    if len(queries) and len(points) and radius > 0:
        qi, _, _ = Grid(points, radius).query_radius(queries, radius)
        out[qi] = True
    return out


def _components(n: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Connected component label of each of ``n`` nodes joined by the pairs (i, j)."""
    parent = np.arange(n)
    while True:
        pi, pj = parent[i], parent[j]
        apart = pi != pj
        if not apart.any():
            return parent
        # hook every root onto the smallest root it touches, then flatten the trees
        np.minimum.at(parent, np.maximum(pi[apart], pj[apart]), np.minimum(pi[apart], pj[apart]))
        while True:
            up = parent[parent]
            if np.array_equal(up, parent):
                break
            parent = up


def _wide(k0, k1, p0, p1, least: float) -> np.ndarray:
    """True for the edges (node k0 at p0 to node k1 at p1) of the connected
    lines whose bounding box is at least ``least`` across."""
    nodes, ends = np.unique(np.concatenate([k0, k1]), return_inverse=True)
    a = ends[: len(k0)]
    label = _components(len(nodes), a, ends[len(k0):])[ends]
    at = np.concatenate([p0, p1])
    lo = np.full((len(nodes), 3), np.inf)
    hi = np.full((len(nodes), 3), -np.inf)
    np.minimum.at(lo, label, at)
    np.maximum.at(hi, label, at)
    across = np.linalg.norm(np.where(np.isfinite(hi), hi - lo, 0.0), axis=1)
    return across[label[: len(k0)]] >= least


def _unspurred(k0, k1, p0, p1, least: float) -> np.ndarray:
    """True for the edges (node k0 at p0 to node k1 at p1) that are not on a
    spur: a branch from a loose end to a fork, shorter than ``least`` (the
    hairs a noisy line or a thinned band of pixels grows)."""
    keep = np.ones(len(k0), bool)
    for _ in range(3):  # a fork left with two branches may have been a spur's root
        nodes, ends = np.unique(np.concatenate([k0[keep], k1[keep]]), return_inverse=True)
        live = np.flatnonzero(keep)
        s = len(live)
        a, b = ends[:s].tolist(), ends[s:].tolist()
        length = np.linalg.norm(p1[live] - p0[live], axis=1).tolist()
        degree = np.bincount(ends, minlength=len(nodes)).tolist()
        touch = [[] for _ in range(len(nodes))]
        for e in range(s):
            touch[a[e]].append(e)
            touch[b[e]].append(e)
        cut = []
        for leaf in [v for v in range(len(nodes)) if degree[v] == 1]:
            path, node, came, total = [], leaf, -1, 0.0
            while True:
                nxt = [e for e in touch[node] if e != came]
                if degree[node] >= 3 or not nxt or (node != leaf and degree[node] != 2):
                    break
                came = nxt[0]
                path.append(came)
                total += length[came]
                node = b[came] if a[came] == node else a[came]
                if total >= least:
                    break
            if path and degree[node] >= 3 and total < least:
                cut.extend(path)
        if not cut:
            break
        keep[live[np.unique(cut)]] = False
    return keep


def _sight(opt):
    """Unit vectors from points towards the middle viewer."""
    if opt.view_point is not None:
        eye = np.asarray(opt.view_point, dtype=np.float64).reshape(3)

        def sight(p):
            d = eye - p
            return d / np.maximum(np.linalg.norm(d, axis=1), 1e-300)[:, None]

        return sight
    v = normalize(opt.view_dir)
    return lambda p: np.broadcast_to(v, p.shape)


def _weld(tris: np.ndarray):
    """Vertex ids (T, 3) of the corners merged by position, and the vertices."""
    corners = tris.reshape(-1, 3)
    lo = corners.min(axis=0)
    extent = float(np.ptp(corners, axis=0).max()) or 1.0
    q = np.floor((corners - lo) / (extent * 1e-5) + 0.5).astype(np.int64)  # < 2**17 per axis
    key = q[:, 0] | (q[:, 1] << 17) | (q[:, 2] << 34)  # exact in a double too (the web core)
    _, first, vid = np.unique(key, return_index=True, return_inverse=True)
    return vid.reshape(-1, 3), corners[first]


def _edges(vid: np.ndarray):
    """Half-edge ``3t + k`` runs from corner k of triangle t to corner k + 1.
    Returns (mesh edge id, start vertex, end vertex) of every half-edge."""
    a = vid.ravel()
    b = vid[:, [1, 2, 0]].ravel()
    nv = int(vid.max()) + 1
    _, edge = np.unique(np.minimum(a, b) * nv + np.maximum(a, b), return_inverse=True)
    return edge.ravel(), a, b


def _feature_edges(tris, normals, live, edge, va, vb, crease_cos, sight):
    """Half-edges on the borders and on the sharp creases, one per mesh edge
    (on a crease the one whose triangle faces the viewer more)."""
    he = np.flatnonzero(live[np.arange(len(edge)) // 3] & (va != vb))
    order = he[np.argsort(edge[he], kind="stable")]
    first = np.ones(len(order), bool)
    first[1:] = edge[order][1:] != edge[order][:-1]
    starts = np.flatnonzero(first)
    sides = np.diff(np.append(starts, len(order)))
    border = order[starts[sides == 1]]
    two = sides >= 2
    a, b = order[starts[two]], order[starts[two] + 1]
    cos = np.einsum("ij,ij->i", normals[a // 3], normals[b // 3])
    cos = np.where(va[a] == va[b], -cos, cos)  # both run the same way: the faces wind oppositely
    sharp = (sides[two] > 2) | (cos < crease_cos)
    a, b = a[sharp], b[sharp]
    mid = 0.5 * (tris[a // 3, a % 3] + tris[a // 3, (a % 3 + 1) % 3])
    s = sight(mid)
    better = np.einsum("ij,ij->i", normals[b // 3], s) > np.einsum("ij,ij->i", normals[a // 3], s)
    return border, np.where(better, b, a)


def _onehot(k: np.ndarray) -> np.ndarray:
    bary = np.full((len(k), 3), _INSET / 3.0)
    bary[np.arange(len(k)), k] = 1.0 - 2.0 * _INSET / 3.0
    return bary


def _mid(segs: _Segs, tris) -> np.ndarray:
    return np.einsum("ij,ijk->ik", 0.5 * (segs.b0 + segs.b1), tris[segs.tri])


def _cut(segs: _Segs, tris, max_len: float, next_key: int):
    """Segments cut into pieces no longer than ``max_len``; the cuts get new
    node names from ``next_key`` on."""
    p0, p1 = segs.ends(tris)
    m = np.maximum(1, np.ceil(np.linalg.norm(p1 - p0, axis=1) / max_len)).astype(np.int64)
    seg = np.repeat(np.arange(len(m)), m)
    i = np.arange(int(m.sum())) - np.repeat(np.cumsum(m) - m, m)
    f0, f1 = (i / m[seg])[:, None], ((i + 1) / m[seg])[:, None]
    base = next_key + np.cumsum(m - 1) - (m - 1)  # first cut of each segment
    k0 = np.where(i == 0, segs.k0[seg], base[seg] + i - 1)
    k1 = np.where(i == m[seg] - 1, segs.k1[seg], base[seg] + i)
    b0, b1 = segs.b0[seg], segs.b1[seg]
    out = _Segs(segs.tri[seg], b0 * (1 - f0) + b1 * f0, b0 * (1 - f1) + b1 * f1, k0, k1)
    return out, next_key + int((m - 1).sum())


def _points(segs: _Segs, tris, step):
    """Points every ``step`` or closer along the segments, their ends left out:
    (pos, tri, bary)."""
    p0, p1 = segs.ends(tris)
    m = np.maximum(1, np.ceil(np.linalg.norm(p1 - p0, axis=1) / step)).astype(np.int64)
    seg = np.repeat(np.arange(len(m)), m)
    f = ((np.arange(int(m.sum())) - np.repeat(np.cumsum(m) - m, m) + 0.5) / m[seg])[:, None]
    bary = segs.b0[seg] * (1.0 - f) + segs.b1[seg] * f
    tri = segs.tri[seg]
    return np.einsum("ij,ijk->ik", bary, tris[tri]), tri, bary


def _chains(k0, k1, p0, p1, prefer):
    """Edges (node ``k0`` at ``p0`` to node ``k1`` at ``p1``) chained into lines
    that run between the nodes where lines end or meet, or round a loop.

    Returns (chains, corners). A chain is (edges, forward, closed, start node,
    end node); the corners are the nodes where lines end or meet, as (node,
    end, strength), ``end`` the index into ``concatenate([p0, p1])`` of the
    edge end ``prefer`` scores highest there.
    """
    s = len(k0)
    nodes, ends = np.unique(np.concatenate([k0, k1]), return_inverse=True)
    n0, n1 = ends[:s].tolist(), ends[s:].tolist()
    order = np.lexsort((-prefer, ends))
    starts = np.flatnonzero(np.r_[True, ends[order][1:] != ends[order][:-1]])
    degree = np.diff(np.append(starts, len(order)))
    pick = np.flatnonzero(degree != 2)
    pick = pick[np.argsort(-degree[pick], kind="stable")]

    # half h is edge h % s seen from its end h // s; walk through the nodes of degree 2
    inc, st, dg = order.tolist(), starts.tolist(), degree.tolist()
    visited = [False] * s
    chains = []

    def walk(node, h):
        edges, forward = [], []
        start = node
        while True:
            e, back = h % s, h >= s
            visited[e] = True
            edges.append(e)
            forward.append(not back)
            node = n0[e] if back else n1[e]
            if dg[node] != 2:
                return edges, forward, False, start, node
            a, b = inc[st[node]], inc[st[node] + 1]
            h = b if a % s == e else a
            if visited[h % s]:
                return edges, forward, True, start, node

    for node in pick.tolist():
        for h in inc[st[node]: st[node] + dg[node]]:
            if not visited[h % s]:
                chains.append(walk(node, h))
    for e in range(s):
        if not visited[e]:  # a loop with no end, walked from the start of edge e
            chains.append(walk(n0[e], e))
    chains = [(np.asarray(e, np.int64), np.asarray(f, bool), c, int(nodes[a]), int(nodes[b]))
              for e, f, c, a, b in chains]
    return chains, (nodes[pick], order[starts[pick]], (2.0 + degree[pick]).astype(np.float64))


def _bends(chains: List[_Chain], window: float):
    """The chains cut where they bend sharper than ``CORNER_TURN`` over
    ``window`` of line on either side, at the sharpest node of each bend.

    Returns (chains, corners); the corners are (node, pos, tri, bary,
    strength), with new node names below 0.
    """
    out, corners = [], ([], [], [], [], [])
    limit = math.cos(math.radians(CORNER_TURN))
    name = -1
    for chain in chains:
        t, total = chain.t, chain.t[-1]
        inner = np.arange(len(t) - 1) if chain.closed else np.flatnonzero((t >= window) & (t <= total - window))
        if total <= 2.0 * window or not len(inner):
            out.append(chain)
            continue
        here = chain.p[inner]
        d1 = here - chain.point(t[inner] - window)
        d2 = chain.point(t[inner] + window) - here
        bend = np.einsum("ij,ij->i", d1, d2) / np.maximum(np.linalg.norm(d1, axis=1) * np.linalg.norm(d2, axis=1),
                                                          1e-300)
        cuts, names = [], {}
        for k in np.argsort(bend, kind="stable"):  # sharpest first; one cut per bend
            if bend[k] >= limit:
                break
            node = int(inner[k])
            if cuts:
                gap = np.abs(t[cuts] - t[node])
                if chain.closed:
                    gap = np.minimum(gap, total - gap)
                if gap.min() <= window:
                    continue
            cuts.append(node)
            names[node] = name
            for box, value in zip(corners, (name, chain.node(node), chain.node_tri[node], chain.node_bary[node],
                                            1.0 - bend[k])):
                box.append(value)
            name -= 1
        if not cuts:
            out.append(chain)
            continue
        cuts.sort()
        n = len(chain.p) - 1
        if chain.closed:  # a loop opens at its first cut and ends there again
            for a, b in zip(cuts, cuts[1:] + [cuts[0] + n]):
                out.append(chain.cut(a, b, names[a], names[b % n]))
        else:
            ends = [0] + cuts + [n]
            keys = [chain.start] + [names[c] for c in cuts] + [chain.end]
            for (a, b), (ka, kb) in zip(zip(ends[:-1], ends[1:]), zip(keys[:-1], keys[1:])):
                out.append(chain.cut(a, b, ka, kb))
    node, pos, tri, bary, strength = corners
    return out, (np.asarray(node, np.int64), np.asarray(pos, float).reshape(-1, 3), np.asarray(tri, np.int64),
                 np.asarray(bary, float).reshape(-1, 3), np.asarray(strength, float))


def _corner_normals(vid, normals, areas, smooth_cos):
    """Normal at each triangle corner (T, 3, 3): the area-weighted mean of the
    triangles around the vertex that meet this one smoothly (auto smooth)."""
    flat = vid.ravel()
    order = np.argsort(flat, kind="stable")
    counts = np.bincount(flat)
    starts = np.cumsum(counts) - counts
    out = np.zeros((len(flat), 3))
    for lo in range(0, len(flat), _CHUNK):
        c = np.arange(lo, min(lo + _CHUNK, len(flat)))
        owner, at = _expand_ranges(starts[flat[c]], counts[flat[c]])
        mine, theirs = c[owner] // 3, order[at] // 3
        w = areas[theirs] * (np.einsum("ij,ij->i", normals[mine], normals[theirs]) >= smooth_cos)
        out[c] = np.stack([np.bincount(owner, weights=normals[theirs, i] * w, minlength=len(c))
                           for i in range(3)], 1)
    length = np.linalg.norm(out, axis=1)
    own = normals[np.arange(len(flat)) // 3]
    out = np.where(length[:, None] > 0, out / np.maximum(length, 1e-300)[:, None], own)
    return out.reshape(-1, 3, 3)


def _views(opt, tris):
    """(eye, toward) of the views relief and colours are read from: the middle
    viewer in facade mode; otherwise it, the three views round the vertical
    axis and the one from above, for a show seen from every side. ``eye`` is
    None for a viewer far away in the direction ``toward``."""
    corners = tris.reshape(-1, 3)
    center = (corners.min(axis=0) + corners.max(axis=0)) * 0.5
    if opt.view_point is not None:
        middle = np.asarray(opt.view_point, dtype=np.float64).reshape(3) - center
    else:
        middle = normalize(opt.view_dir)
    if opt.facade:
        out = [middle]
    else:
        up = np.zeros(3)
        up[opt.up_axis] = 1.0
        out = [rotate(middle, up, a) for a in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)]
        out.append(up * np.linalg.norm(middle))
    if opt.view_point is None:
        return [(None, v) for v in out]
    return [(center + v, v) for v in out]


def _render(tris, eye, toward, h):
    """What a view of the model shows: a visibility buffer, the triangle and
    (perspective-correct) barycentrics each pixel sees, or None when it shows
    nothing. ``eye`` None is a viewer far away in the direction ``toward``;
    otherwise ``h`` is the pixel size at the model's centre.

    Returns dict(tri (H, W), -1 where no triangle; bary (H, W, 3); h; lo, hi): pixel (row, col)
    is centred on the picture point (lo[0] + (col + 0.5) h, hi[1] - (row + 0.5) h)."""
    if eye is None:
        d, e1, e2 = view_basis(toward)
        x, y, near = tris @ e1, tris @ e2, tris @ d  # nearer is larger
        w = None
        ids = np.arange(len(tris))
    else:
        eye = np.asarray(eye, dtype=np.float64)
        d, e1, e2 = view_basis(toward)
        rel = tris - eye
        z = -(rel @ d)  # distance in front of the eye
        far = float(np.linalg.norm(toward))
        ids = np.flatnonzero((z > 0.02 * far).all(axis=1))
        z = z[ids]
        x, y = (rel[ids] @ e1) * far / z, (rel[ids] @ e2) * far / z
        near, w = 1.0 / z, z  # 1/z is affine in the picture
    if not len(ids):
        return None
    lo = np.array([x.min(), y.min()])
    hi = np.array([x.max(), y.max()])
    h = max(h, float(np.sqrt(np.prod(hi - lo + 4 * h) / _MAX_PIXELS)))
    lo, hi = lo - 2 * h, hi + 2 * h
    W, H = (np.ceil((hi - lo) / h).astype(np.int64) + 1).tolist()
    px = (x - lo[0]) / h - 0.5  # pixel centres are at whole coordinates, row 0 on top
    py = (hi[1] - y) / h - 0.5
    x0 = np.clip(np.ceil(px.min(axis=1)), 0, W - 1).astype(np.int64)
    x1 = np.clip(np.floor(px.max(axis=1)), 0, W - 1).astype(np.int64)
    y0 = np.clip(np.ceil(py.min(axis=1)), 0, H - 1).astype(np.int64)
    y1 = np.clip(np.floor(py.max(axis=1)), 0, H - 1).astype(np.int64)
    nx, ny = x1 - x0 + 1, y1 - y0 + 1
    count = np.where((px.max(axis=1) >= 0) & (py.max(axis=1) >= 0), np.maximum(nx, 0) * np.maximum(ny, 0), 0)
    best = np.full(W * H, -np.inf)
    seen = np.full(W * H, -1, np.int64)
    weights = np.zeros((W * H, 3))
    todo = np.flatnonzero(count)
    total = np.cumsum(count[todo])
    start = 0
    while start < len(todo):
        stop = max(int(np.searchsorted(total, (total[start - 1] if start else 0) + _RASTER_PAIRS, side="right")),
                   start + 1)
        t = todo[start:stop]
        c = count[t]
        owner = np.repeat(np.arange(len(t)), c)
        k = np.arange(int(c.sum())) - np.repeat(np.cumsum(c) - c, c)
        t = t[owner]
        cx = x0[t] + k % nx[t]
        cy = y0[t] + k // nx[t]
        ax, ay, bx, by, qx, qy = px[t, 0], py[t, 0], px[t, 1], py[t, 1], px[t, 2], py[t, 2]
        den = (by - qy) * (ax - qx) + (qx - bx) * (ay - qy)
        good = np.abs(den) > 1e-18
        den = np.where(good, den, 1.0)
        l0 = ((by - qy) * (cx - qx) + (qx - bx) * (cy - qy)) / den
        l1 = ((qy - ay) * (cx - qx) + (ax - qx) * (cy - qy)) / den
        l2 = 1.0 - l0 - l1
        inside = good & (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
        t, cx, cy = t[inside], cx[inside], cy[inside]
        lam = np.stack([l0[inside], l1[inside], l2[inside]], 1)
        depth = np.einsum("ij,ij->i", lam, near[t])
        pix = cy * W + cx
        order = np.lexsort((-depth, pix))
        first = np.ones(len(order), bool)
        first[1:] = pix[order][1:] != pix[order][:-1]
        sel = order[first]
        sel = sel[depth[sel] > best[pix[sel]]]
        best[pix[sel]] = depth[sel]
        seen[pix[sel]] = t[sel]
        weights[pix[sel]] = lam[sel]
        start = stop
    hit = seen >= 0
    if not hit.any():
        return None
    if w is not None:  # back from the picture to the triangle
        corrected = weights[hit] / w[seen[hit]]
        weights[hit] = corrected / corrected.sum(axis=1, keepdims=True)
    seen[hit] = ids[seen[hit]]
    return dict(tri=seen.reshape(H, W), bary=weights.reshape(H, W, 3), h=h, lo=lo, hi=hi)


def _shift(a, dy, dx, fill=0.0):
    """``a`` moved by (dy, dx) pixels: out[y, x] = a[y - dy, x - dx]."""
    out = np.full_like(a, fill)
    H, W = a.shape[:2]
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[yd, xd] = a[ys, xs]
    return out


def _blur(img, sigma):
    """Separable Gaussian blur of an (H, W[, C]) image, edges repeated."""
    r = max(1, int(math.ceil(3.0 * sigma)))
    kernel = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    out = img
    for axis in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r, r)
        padded = np.pad(out, pad, mode="edge")
        acc = np.zeros_like(out)
        for i, kw in enumerate(kernel):
            cut = [slice(None)] * out.ndim
            cut[axis] = slice(i, i + out.shape[axis])
            acc += kw * padded[tuple(cut)]
        out = acc
    return out


_NEIGHBOURS = ((0, 1), (1, 0), (1, 1), (1, -1))


def _pairs(H, W, dy, dx):
    """Index slices (a, b) of each pixel and its neighbour (dy, dx)."""
    return ((slice(0, H - dy), slice(max(-dx, 0), W + min(-dx, 0))),
            (slice(dy, None), slice(max(dx, 0), W + min(dx, 0))))


def _apart(P, facing, h, a, b, dy, dx):
    """(distance, covers) of neighbouring pixels ``a`` and ``b``: ``covers``
    where one part of the model covers another between them, a step far longer
    than the slope of the surface explains (a grazed surface has long steps too)."""
    apart = np.linalg.norm(P[a] - P[b], axis=2)
    slope = np.maximum(np.minimum(facing[a], facing[b]), 0.1)
    return apart, apart > 3.0 * h * math.hypot(dx, dy) / slope


def _change(field, P, facing, mask, h):
    """How fast ``field`` changes per length of surface: the largest difference
    to a neighbouring pixel over their distance, where both are in ``mask`` and
    no part of the model covers another between them."""
    H, W = mask.shape
    out = np.zeros((H, W))
    for dy, dx in _NEIGHBOURS:
        a, b = _pairs(H, W, dy, dx)
        apart, covers = _apart(P, facing, h, a, b, dy, dx)
        ok = mask[a] & mask[b] & ~covers
        rate = np.where(ok, np.linalg.norm(field[a] - field[b], axis=2) / np.maximum(apart, 1e-300), 0.0)
        out[a] = np.maximum(out[a], rate)
        out[b] = np.maximum(out[b], rate)
    return out


def _crest(strength, mask, lo: float, hi: float):
    """Pixels on the crests of ``strength`` (at least ``lo``, connected to a
    crest of at least ``hi``) and, for each pixel, the whole-pixel direction
    across its crest."""
    s = _blur(strength, 1.0)
    sxx = _shift(s, 0, -1) + _shift(s, 0, 1) - 2.0 * s
    syy = _shift(s, -1, 0) + _shift(s, 1, 0) - 2.0 * s
    sxy = 0.25 * (_shift(s, -1, -1) + _shift(s, 1, 1) - _shift(s, -1, 1) - _shift(s, 1, -1))
    tr, det = sxx + syy, sxx * syy - sxy * sxy
    lam = 0.5 * tr - np.sqrt(np.maximum(0.25 * tr * tr - det, 0.0))  # the most negative curvature
    vx, vy = sxy, lam - sxx
    flat = np.hypot(vx, vy) < 1e-12
    vx, vy = np.where(flat, lam - syy, vx), np.where(flat, sxy, vy)
    n = np.maximum(np.hypot(vx, vy), 1e-300)
    ox, oy = np.rint(vx / n).astype(np.int64), np.rint(vy / n).astype(np.int64)
    top = np.zeros(s.shape, bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                here = (ox == dx) & (oy == dy)
                top |= here & (s >= _shift(s, -dy, -dx, -np.inf)) & (s >= _shift(s, dy, dx, -np.inf))
    weak = top & (lam < 0) & mask & (s >= lo)
    return _hysteresis(weak, weak & (s >= hi)), ox, oy


def _hysteresis(weak, strong):
    """The weak pixels 8-connected to a strong one."""
    H, W = weak.shape
    idx = np.full((H, W), -1, np.int64)
    idx[weak] = np.arange(int(weak.sum()))
    i, j = [], []
    for dy, dx in _NEIGHBOURS:
        a, b = _pairs(H, W, dy, dx)
        both = (idx[a] >= 0) & (idx[b] >= 0)
        i.append(idx[a][both])
        j.append(idx[b][both])
    label = _components(int(weak.sum()), np.concatenate(i), np.concatenate(j))
    lit = np.zeros(len(label), bool)
    lit[label[strong[weak]]] = True
    out = np.zeros((H, W), bool)
    out[weak] = lit[label]
    return out


def _thin(img):
    """Zhang-Suen thinning: lines one pixel wide."""
    img = img.copy()
    ring = ((1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1))  # north, then clockwise
    while True:
        changed = False
        for second in (False, True):
            p = [_shift(img, dy, dx, False) for dy, dx in ring]
            count = sum(q.astype(np.int64) for q in p)
            steps = sum(((~p[i]) & p[(i + 1) % 8]).astype(np.int64) for i in range(8))
            if second:
                keep = (p[0] & p[2] & p[6]) | (p[0] & p[4] & p[6])
            else:
                keep = (p[0] & p[2] & p[4]) | (p[2] & p[4] & p[6])
            kill = img & (count >= 2) & (count <= 6) & (steps == 1) & ~keep
            if kill.any():
                img &= ~kill
                changed = True
        if not changed:
            return img


def _view_lines(mesh, tris, cn, render, eye, toward, opt, kind):
    """The silhouette, relief or colour lines of one view: (rows, cols, pos,
    tri, bary, colour) of their pixels, or None."""
    tri, bary, h = render["tri"], render["bary"], render["h"]
    fg = tri >= 0
    t = np.where(fg, tri, 0)
    P = np.einsum("hwk,hwkd->hwd", bary, tris[t])
    N = np.einsum("hwk,hwkd->hwd", bary, cn[t])
    N /= np.maximum(np.linalg.norm(N, axis=2, keepdims=True), 1e-300)
    if eye is None:
        s = normalize(toward)
        nearness = P @ s
        facing = N @ s
    else:
        to_eye = np.asarray(eye, dtype=np.float64) - P
        dist = np.linalg.norm(to_eye, axis=2)
        nearness = -dist
        facing = np.einsum("hwd,hwd->hw", N, to_eye) / np.maximum(dist, 1e-300)
    cover = _blur(fg.astype(np.float64), VIEW_SIGMA)
    solid = fg & (cover > 0.5)  # the field is measured here, lines are kept where the view is not grazing
    inner = solid & (facing > math.cos(math.radians(opt.facade_angle)))
    H, W = fg.shape
    if kind == "silhouette":
        # the edge of the smoothed picture: fur and noise smaller than sigma leave it smooth
        line = solid & ~(_shift(solid, 1, 0, False) & _shift(solid, -1, 0, False)
                         & _shift(solid, 0, 1, False) & _shift(solid, 0, -1, False))
        # where one part of the model covers another, the nearer one draws the line
        for dy, dx in _NEIGHBOURS:
            a, b = _pairs(H, W, dy, dx)
            jump = solid[a] & solid[b] & _apart(P, facing, h, a, b, dy, dx)[1]
            front = nearness[a] > nearness[b]
            line[a] |= jump & front
            line[b] |= jump & ~front
        line = _thin(line)
        colour = np.full((int(line.sum()), 3), np.nan)
    elif kind == "relief":
        smooth = _blur(np.where(fg[..., None], N, 0.0), VIEW_SIGMA) / np.maximum(cover[..., None], 1e-300)
        smooth /= np.maximum(np.linalg.norm(smooth, axis=2, keepdims=True), 1e-300)
        bend = _change(smooth, P, facing, solid, h) * (4.0 * VIEW_SIGMA * h)  # radians turned over ~2 sigma
        angle = math.radians(opt.relief_angle)
        line, _, _ = _crest(bend, solid, 0.5 * angle, angle)
        line = _thin(line & inner)
        colour = np.full((int(line.sum()), 3), np.nan)
    else:
        lin = np.ones((H, W, 3))
        lin[fg] = surface_colors(mesh, t[fg], bary[fg])
        lab = linear_to_oklab(lin)
        smooth = _blur(np.where(fg[..., None], lab, 0.0), VIEW_SIGMA) / np.maximum(cover[..., None], 1e-300)
        # a step of colour smoothed over sigma: its steepest slope times sigma * sqrt(2 pi) is the step
        step = _change(smooth, P, facing, solid, h) * (VIEW_SIGMA * h * math.sqrt(2.0 * math.pi))
        line, ox, oy = _crest(step, solid, 0.5 * opt.color_contrast, opt.color_contrast)
        line = _thin(line & inner)
        # each pixel shows the colour of the side that stands out from the model's mean colour
        y, x = np.nonzero(line)
        r = int(round(2 * VIEW_SIGMA))
        mean = lab[inner].mean(axis=0) if inner.any() else np.zeros(3)
        sides = []
        for sign in (1, -1):
            sy = np.clip(y + sign * r * oy[y, x], 0, H - 1)
            sx = np.clip(x + sign * r * ox[y, x], 0, W - 1)
            far = np.where(fg[sy, sx], np.linalg.norm(lab[sy, sx] - mean, axis=1), -1.0)
            sides.append((far, lin[sy, sx]))
        colour = np.where((sides[0][0] >= sides[1][0])[:, None], sides[0][1], sides[1][1])
    if not line.any():
        return None
    y, x = np.nonzero(line)
    return y, x, P[y, x], tri[y, x], bary[y, x], colour


def _pixel_graph(rows, cols):
    """Edges (i, j) between the given line pixels: 4-neighbours, and
    diagonal neighbours with no line pixel at the corner between them."""
    if not len(rows):
        e = np.zeros(0, np.int64)
        return e, e.copy()
    H, W = int(rows.max()) + 2, int(cols.max()) + 2
    idx = np.full((H, W), -1, np.int64)
    idx[rows, cols] = np.arange(len(rows))
    i, j = [], []
    for dy, dx in _NEIGHBOURS:
        a, b = _pairs(H, W, dy, dx)
        both = (idx[a] >= 0) & (idx[b] >= 0)
        if dy and dx:  # a diagonal step where an orthogonal path exists is no edge of its own
            side1 = _shift(idx, 0, -dx, -1)[a] >= 0  # pixel (y, x + dx)
            side2 = _shift(idx, -dy, 0, -1)[a] >= 0  # pixel (y + dy, x)
            both &= ~side1 & ~side2
        i.append(idx[a][both])
        j.append(idx[b][both])
    return np.concatenate(i), np.concatenate(j)


def _park(points: np.ndarray, radius: float) -> np.ndarray:
    """Indices of the points (taken in order) that keep ``radius`` from every one kept before."""
    cells = {}
    kept = []
    r2 = radius * radius
    keys = np.floor(points / radius).astype(np.int64).tolist()
    for n, (p, (x, y, z)) in enumerate(zip(points.tolist(), keys)):
        if any((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2 < r2
               for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
               for q in cells.get((x + dx, y + dy, z + dz), ())):
            continue
        cells.setdefault((x, y, z), []).append(p)
        kept.append(n)
    return np.asarray(kept, np.int64)
