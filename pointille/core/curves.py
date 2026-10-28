"""Curves: a drawing of the model to edit, and drones on any curves.

A drone picture reads from all over the audience only when its lines lie
near one surface facing it. On the model itself the outline runs round its
sides and a carved eye lies deep inside it, so the drawing falls apart for
anyone who is not right in the middle. ``trace`` draws the model as the
middle viewer sees it (``picture``: outline, features, the strongest inner
lines, simple shapes) and lays the curves on a smooth shell of its front:
recesses filled, depth kept (``"shape"``), flattened (``"shallow"``) or gone
(``"flat"``). Or (``"volume"``) the model is drawn whole, to be seen from
every side: horizontal slices of it, dimmer, and on its very surface the
outline, the features and the strongest lines from four views round it.
The curves are meant to be edited by hand; ``compose_curves`` then puts
drones on them, or on any other curves: corners first, evenly along the
lines, and a light fill inside the closed curves, on the shell.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import List, Optional

import numpy as np

from . import color as _color
from . import layout as _layout
from . import outline as _outline
from . import picture as _pic
from . import shapes as _shapes
from .api import DistributeOptions, DroneResult
from .facade import _ProjectedGrid, normalize, view_basis
from .mesh import MeshData
from .sampling import triangle_areas_normals
from .stats import scale_pivot, spacing

DEPTHS = ("flat", "shallow", "shape", "volume")
SHALLOW = 0.25  # "shallow" keeps this share of the depth of the shell
ENVELOPE = 1.5  # recesses narrower than about this many line spacings are filled ...
SHELL_SMOOTH = 0.5  # ... and the shell is smoothed over this many
SHELL_STEP = 0.5  # the grid of the shell, in line spacings
SIMPLIFY = 0.04  # a curve keeps this close (line spacings) to the drawing with as few points as that takes
SAMPLE = 0.1  # curves are measured every this many minimum distances
CURVE_DEPTH = 1.0  # the depth along a traced line is smoothed over this many line spacings
LINE_MIN = 4.0  # a traced inner line is at least this many line spacings long ...
LINE_KEEP = 0.2  # ... and scores at least this share of the best one of its part: no fur, no wrinkles
# how bright the drones of a drawing glow (sRGB, of their colour at full): what draws the figure most
LEVEL = {"outline": 1.0, "feature": 1.0, "dot": 1.0, "line": 0.7, "slice": 0.5}
FILL_GLOW = 0.45  # ... and the fill, only a background to the lines
VOLUME_STEP = 2.0  # the slices of a volume drawing stand this many line spacings apart up the model
SLICE_SMOOTH = 0.5  # a slice is smoothed over this many line spacings (fur, facets)
SLICE_MIN = 2.0  # pieces of slices shorter than this many line spacings are left out (a colour on one, merged)
SLICE_CLEAR = 0.8  # slices keep this many line spacings clear of the lines of the drawing
SLICE_COLORS = 4  # the colours of the model the slices take
LOOP_GAP = 1.5  # open curves whose ends meet this close (line spacings) close a loop for the fill
SLICE_SAMPLE = 0.25  # slices are sampled every this many line spacings
VOLUME_SAMPLES = 12  # slices measured to tell how many drones the slices of a volume take


@dataclass
class Curve:
    points: np.ndarray  # (N, 3); one point: a single drone (an eye too small for a ring)
    closed: bool = False
    color: Optional[np.ndarray] = None  # linear colour of its drones (None: white)
    kind: str = "line"  # "outline" | "feature" | "line" | "dot" | "slice" (of a volume: drones after the rest)


@dataclass
class Sketch:
    """The curves of one part, the shell they lie on (None: their own plane) and the direction they face;
    a ``solid`` one (a volume) is no plane: its corners are where it turns in space, it takes no fill."""

    curves: List[Curve]
    shell: Optional[MeshData] = None
    normal: Optional[np.ndarray] = None
    options: object = None  # layout.PartOptions; None: the defaults
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    solid: bool = False


# ---------------------------------------------------------------- tracing a model

def trace(parts, options: DistributeOptions = None, depth: str = "shallow") -> List[Sketch]:
    """The drawing of ``parts`` (layout.Part) as curves, one sketch per part, as the middle viewer sees
    it (``options.view_dir`` / ``view_point``) at the line spacing the budget ``options.count`` allows;
    with ``depth`` "volume" the whole model, seen from every side (see the module)."""
    opt = options or DistributeOptions()
    if depth not in DEPTHS:
        raise ValueError(f"unknown depth {depth!r}")
    if not parts:
        raise ValueError("no parts to trace")
    mesh = MeshData.concatenate([p.mesh for p in parts])
    offsets = np.cumsum([0] + [p.mesh.triangle_count for p in parts])
    tris = mesh.triangles
    areas, normals = triangle_areas_normals(tris)
    d0 = _min_distance(opt, tris.reshape(-1, 3))
    bodies = _layout._bodies(parts, tris, offsets, areas, opt.fill_ratio * d0, opt.up_axis)
    round_tri = {k: np.concatenate([b.tri for b in bs]) for k, bs in bodies.items()}
    simple = {k for k, p in enumerate(parts) if p.options.shapes}
    fopt = replace(opt, facade=True)
    if depth == "volume":
        return _trace_volume(parts, fopt, mesh, offsets, areas, normals, d0, bodies, simple)
    f = _spread_guess(parts, fopt, mesh, offsets, areas, d0, int(opt.count))
    fl = _layout.lines_factor(f)
    s, sf, near = opt.line_ratio * d0 * fl, opt.fill_ratio * d0 * f, d0 * fl * _layout.CLEAR
    figure = _layout.read_picture(parts, fopt, mesh, offsets, normals, s, sf, round_tri, simple,
                                  eyes=opt.line_ratio * d0)
    figure["inner"] = _strong(figure["inner"], s)
    taken, dots = _layout.drawing(figure, parts, s, sf, near, simple)
    pic, lin = figure["pic"], figure["lin"]
    surface = _envelope(pic, s)
    ref = float(np.median(surface[pic.fg]))
    # past the edge of the model the shell goes on as it was at the edge: an outline lies on it, not half off it
    sg = max(2.0, 0.25 * s / pic.h)
    weight = _outline._blur(pic.fg.astype(np.float64), sg)
    out_edge = _outline._blur(np.where(pic.fg, surface, 0.0), sg) / np.maximum(weight, 1e-300)
    field_ = np.where(pic.fg, surface, np.where(weight > 1e-3, out_edge, ref))
    keep = {"flat": 0.0, "shallow": SHALLOW, "shape": 1.0}[depth]

    def lift(xy, closed=False, step=None):
        """Model points under picture points ``xy`` on the shell; along a line its depth smoothed over
        ``CURVE_DEPTH`` line spacings (``step`` apart): no zigzag in depth."""
        e = _pic._bilinear(field_, *pic.to_px(xy))
        if step is not None and len(e) > 2:
            e = _pic.smooth(np.c_[e, np.zeros(len(e))], CURVE_DEPTH * s / step, closed)[:, 0]
        return _unproject(pic, xy, ref + keep * (e - ref))

    out = [Sketch([], None, None, p.options) for p in parts]
    for st in taken:
        dense, kept = _drawn(st, s, st.part in simple)
        kind = _kind(st)
        colour = vivid(_stroke_color(pic, lin, st, s), LEVEL[kind])
        pts = lift(dense, st.closed, st.step)[kept]
        out[st.part].curves.append(Curve(pts, st.closed and len(pts) > 2, colour, kind))
    H, W = pic.shape
    for xy, part in dots:
        col, row = pic.to_px(np.asarray(xy)[None])
        colour = lin[int(np.clip(np.rint(row[0]), 0, H - 1)), int(np.clip(np.rint(col[0]), 0, W - 1))]
        out[part].curves.append(Curve(lift(np.asarray(xy)[None]), False, vivid(colour, LEVEL["dot"]), "dot"))
    toward = normalize(pic.toward) if pic.eye is not None else pic.d
    for k, sk in enumerate(out):
        sk.shell = _shell(pic, k, s, lift, lin)
        sk.normal = toward
        sk.origin = _unproject(pic, pic.project(tris.reshape(-1, 3).mean(axis=0)[None]), ref)[0]
    return out


def _kind(st):
    """The kind of curve a stroke of the drawing makes: an eye drawn as a ring is a feature too."""
    return "outline" if st.kind == "silhouette" else "feature" if st.feature or st.shape is not None else "line"


def _strong(inner, s):
    """The inner lines worth a curve: long (``LINE_MIN``) and strong (``LINE_KEEP`` of the best of their
    part), and every band; the fur, the wrinkles and the folds of a model are no drawing."""
    top = {}
    for st in inner:
        top[st.part] = max(top.get(st.part, 0.0), st.score)
    return [st for st in inner if st.band or (st.length >= LINE_MIN * s and st.score >= LINE_KEEP * top[st.part])]


# ---------------------------------------------------------------- a volume

def _trace_volume(parts, opt, mesh, offsets, areas, normals, d0, bodies, simple):
    """The whole model as curves (``trace`` with depth "volume"): slices up it ``VOLUME_STEP`` line spacings
    apart, and on its surface the drawing seen from ``layout.VIEWS_360`` views round it (the outline the
    middle viewer's), each view drawing what faces it most."""
    s = _volume_spacing(parts, opt, mesh, offsets, areas, normals, d0, int(opt.count))
    near = d0 * (s / (opt.line_ratio * d0)) * _layout.CLEAR
    sf = s * opt.fill_ratio / opt.line_ratio  # inner lines keep apart as in a drawing with fill
    share = np.array([_pic.SHARE[p.options.role] if p.has_lines else 0.0 for p in parts])
    reading = _layout._views_read(parts, opt, mesh, offsets, normals, s, bodies, eyes=opt.line_ratio * d0,
                                  real=True)
    out = [Sketch([], None, None, p.options, solid=True) for p in parts]
    drawn = [[] for _ in parts]  # points of the features (the slices keep clear of them; of lines, the drones do)
    holes = [[] for _ in parts]  # closed features as their view sees them: no slice runs through an eye
    for v, view in enumerate(reading["views"]):
        pic, lin = view["pic"], view["lin"]
        allow = dict(enumerate((share / np.maximum(1.0 - share, 1e-9) * view["outline"] / s).tolist()))
        # the middle view draws its strong lines; from the others the slices show the form, they add features
        inner = [st for st in _strong(view["inner"], s) if v == 0 or st.band]
        taken, dots = _pic.choose(view["sil"], inner, view["dots"], s, sf, allow, simple, view["feats"])
        if simple:
            taken, dots = _pic.simplify(taken, dots, pic, lin, near, s, simple)
        else:
            dots = [(xy, part) for xy, part, _ in dots]
        _pic.corners(taken, s)
        for st in taken:
            dense, kept = _drawn(st, s, st.part in simple)
            pts = _on_surface(pic, dense, st.part, st.closed, s)
            if pts is None:
                continue
            kind = _kind(st)
            colour = vivid(_stroke_color(pic, lin, st, s), LEVEL[kind])
            out[st.part].curves.append(Curve(pts[kept], st.closed and len(kept) > 2, colour, kind))
            if kind == "feature":
                drawn[st.part].append(pts)
                if st.closed:
                    holes[st.part].append((pic, dense, float(_nearness(pic, pts).min()) - s))
        H, W = pic.shape
        for xy, part in dots:
            pts = _on_surface(pic, np.asarray(xy)[None], part, False, s)
            if pts is None:
                continue
            col, row = pic.to_px(np.asarray(xy)[None])
            colour = lin[int(np.clip(np.rint(row[0]), 0, H - 1)), int(np.clip(np.rint(col[0]), 0, W - 1))]
            out[part].curves.append(Curve(pts, False, vivid(colour, LEVEL["dot"]), "dot"))
            drawn[part].append(pts)
    tris = mesh.triangles
    for k, part in enumerate(parts):
        ids = np.arange(offsets[k], offsets[k + 1])
        keep_off = np.vstack(drawn[k]) if drawn[k] else np.zeros((0, 3))
        out[k].curves += _slices(part.mesh, tris, ids, opt.up_axis, VOLUME_STEP * s, s, keep_off, holes[k])
        corners = tris[ids].reshape(-1, 3)
        out[k].origin = 0.5 * (corners.min(axis=0) + corners.max(axis=0))
        out[k].normal = None
    return out


def _volume_spacing(parts, opt, mesh, offsets, areas, normals, d0, budget):
    """The line spacing a volume drawing of the model stands at to fit ``budget`` drones (a quick look):
    its slices as long as ``VOLUME_SAMPLES`` of them measure, its lines the outline and the share of inner
    lines of the middle view."""
    tris = mesh.triangles
    s1 = opt.line_ratio * d0
    up = opt.up_axis
    walls = 0.0  # length of the slices per height
    for k in range(len(parts)):
        T = tris[offsets[k]:offsets[k + 1]]
        if not len(T):
            continue
        z = T[..., up]
        zlo, zhi = float(z.min()), float(z.max())
        if zhi <= zlo:
            continue
        vid, _ = _outline._weld(T)
        nv = int(vid.max()) + 1
        total = 0.0
        for level in zlo + (np.arange(VOLUME_SAMPLES) + 0.5) * ((zhi - zlo) / VOLUME_SAMPLES):
            for dense, _, _, closed in _smooth_cuts(T, z, float(level) + 1e-9 * (zhi - zlo), vid, nv, s1):
                total += float(_pic._arc(dense, closed)[-1])
        walls += total / VOLUME_SAMPLES * (zhi - zlo)
    ext = np.sort(np.ptp(tris.reshape(-1, 3), axis=0))
    quick = _pic.Picture(mesh, offsets, opt, max(s1 / _pic.PX, math.sqrt(float(ext[1] * ext[2]) / _layout.ESTIMATE_PIXELS)))
    _, length = _pic.estimate(quick, len(parts), max(0.5 * s1, 2.0 * quick.h))
    share = np.array([_pic.SHARE[p.options.role] if p.has_lines else 0.0 for p in parts])
    lines = float((length * (1.0 + _layout.INNER_USE * share / (1.0 - share))).sum())  # inner: the middle view's

    def need(s):
        return walls / (VOLUME_STEP * s * s) + lines / s

    if need(s1) <= budget:
        return s1
    lo, hi = s1, 2.0 * s1
    while need(hi) > budget and hi < 1e6 * s1:
        lo, hi = hi, 2.0 * hi
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if need(mid) > budget else (lo, mid)
    return hi


def _on_surface(pic, xy, part, closed, s):
    """Points of the model's surface under picture points ``xy`` (a line of a view, or a dot), their depth
    smoothed along the line over ``CURVE_DEPTH`` line spacings: fur gives it no zigzag. None when the
    view finds no surface there."""
    pos, _, _, ok = pic.lift(xy, part, max(3.0, 0.6 * _pic.THIN * s / pic.h))
    if not ok.any():
        return None
    depth = _nearness(pic, pos)
    if not ok.all():
        idx = np.arange(len(xy))
        depth = np.interp(idx, idx[ok], depth[ok], period=len(xy) if closed else None)
    if len(xy) > 2:
        step = float(_pic._arc(xy, closed)[-1]) / max(len(xy) - (0 if closed else 1), 1)
        if step > 0:
            depth = _pic.smooth(np.c_[depth, np.zeros(len(depth))], CURVE_DEPTH * s / step, closed)[:, 0]
    return _unproject(pic, xy, depth)


def _nearness(pic, pts):
    """How near the viewer of ``pic`` model points are (as ``Picture.nearness``; ``_unproject`` undoes it)."""
    return pts @ pic.d if pic.eye is None else (pts - pic.eye) @ pic.d


def _slices(part_mesh, tris, ids, up, step, s, clear, holes=()):
    """Curves round the model where horizontal planes ``step`` apart cut it: smoothed, clear of the points
    ``clear`` (the drawing) and out of the ``holes`` (view, its loop, the depth it lies above), cut where
    the colour of the model changes (the colours of ``part_mesh``)."""
    if not len(ids):
        return []
    T = tris[ids]
    z = T[..., up]
    zlo, zhi = float(z.min()), float(z.max())
    if zhi - zlo <= 0:
        return []
    groups = _layout.color_groups(part_mesh, SLICE_COLORS)
    vid, _ = _outline._weld(T)
    nv = int(vid.max()) + 1
    n = max(int(round((zhi - zlo) / step)), 1)
    out = []
    for level in zlo + (np.arange(n) + 0.5) * ((zhi - zlo) / n):
        for dense, tri, bary, closed in _smooth_cuts(T, z, float(level) + 1e-9 * (zhi - zlo), vid, nv, s):
            lab = _color.linear_to_oklab(_color.surface_colors(part_mesh, tri, bary))
            label = np.argmin(((lab[:, None, :] - groups.lab[None]) ** 2).sum(axis=2), axis=1)
            label = _mode(label, len(groups.lab), max(int(round(SLICE_MIN / SLICE_SAMPLE)), 1), closed)
            free = ~_outline._near(dense, clear, SLICE_CLEAR * s) if len(clear) else np.ones(len(dense), bool)
            for pic, loop, deep in holes:
                free &= ~(_pic._inside(pic.project(dense), loop) & (_nearness(pic, dense) >= deep))
            for a, b, whole in _pieces(label, free, closed):
                piece = dense[np.arange(a, b) % len(dense)]
                if float(_pic._arc(piece, whole)[-1]) < SLICE_MIN * s:
                    continue
                keep = _rdp(piece, SIMPLIFY * s, whole)
                colour = vivid(groups.linear[label[a % len(label)]], LEVEL["slice"])
                out.append(Curve(piece[keep], whole, colour, "slice"))
    return out


def _smooth_cuts(T, z, level, vid, nv, s):
    """The cuts of the plane at height ``level`` (``_cut``) at least ``SLICE_MIN`` line spacings long,
    sampled every ``SLICE_SAMPLE`` and smoothed over ``SLICE_SMOOTH`` line spacings: (points, triangle
    and barycentrics under each, closed)."""
    out = []
    for pts, tri, bary, closed in _cut(T, z, level, vid, nv):
        arc = _pic._arc(pts, closed)
        if arc[-1] < SLICE_MIN * s:
            continue
        m = max(int(math.ceil(arc[-1] / (SLICE_SAMPLE * s))), 3)
        x = np.arange(m) * (arc[-1] / m) if closed else np.linspace(0.0, arc[-1], m + 1)
        seg = np.clip(np.searchsorted(arc, x, side="right") - 1, 0, len(pts) - 1)
        q = np.vstack([pts, pts[:1]]) if closed else pts
        dense = np.stack([np.interp(x, arc, q[:, i]) for i in range(3)], 1)
        dense = _pic.smooth(dense, SLICE_SMOOTH * s / (arc[-1] / m), closed)
        out.append((dense, tri[seg], bary[seg], closed))
    return out


def _cut(T, z, level, vid, nv):
    """The polylines where the plane at height ``level`` cuts the triangles ``T`` (heights ``z``):
    [(points, triangle (into T) of each segment, barycentrics of its start, closed)]."""
    above = z > level
    tt = np.flatnonzero(above.any(axis=1) & ~above.all(axis=1))
    if not len(tt):
        return []
    ab = above[tt]
    crossing = ab != np.roll(ab, -1, axis=1)  # edge i runs from corner i to corner i + 1
    which = np.argsort(~crossing, axis=1, kind="stable")[:, :2]
    rows = np.arange(len(tt))
    ends = []
    for q in range(2):
        i = which[:, q]
        j = (i + 1) % 3
        f = (level - z[tt, i]) / (z[tt, j] - z[tt, i])
        p = T[tt, i] + f[:, None] * (T[tt, j] - T[tt, i])
        b = np.zeros((len(tt), 3))
        b[rows, i], b[rows, j] = 1.0 - f, f
        va, vb = vid[tt, i], vid[tt, j]
        ends.append((np.minimum(va, vb) * nv + np.maximum(va, vb), p, b))
    (k0, p0, b0), (k1, p1, b1) = ends
    walks, _ = _outline._chains(k0, k1, p0, p1, np.zeros(2 * len(k0)))
    out = []
    for edges, fw, closed, _, _ in walks:
        f = fw[:, None]
        pts = np.where(f, p0[edges], p1[edges])
        bary = np.where(f, b0[edges], b1[edges])
        if not closed:
            pts = np.vstack([pts, np.where(f[-1:], p1[edges[-1:]], p0[edges[-1:]])])
            bary = np.vstack([bary, np.where(f[-1:], b1[edges[-1:]], b0[edges[-1:]])])
            tri = np.r_[tt[edges], tt[edges[-1:]]]
        else:
            tri = tt[edges]
        out.append((pts, tri, bary, bool(closed)))
    return out


def _mode(label, k, width, closed):
    """The label most of the ``width`` points round each point have: a colour shorter than about half
    that along a slice is its neighbours'."""
    if width < 3 or len(label) < 3:
        return label
    r = width // 2
    n = len(label)
    i = np.arange(-r, n + r)
    j = np.mod(i, n) if closed else np.clip(i, 0, n - 1)
    onehot = np.eye(k)[label[j]]
    c = np.cumsum(np.vstack([np.zeros((1, k)), onehot]), axis=0)
    counts = c[2 * r + 1:] - c[:-2 * r - 1]
    return np.argmax(counts, axis=1)


def _pieces(label, free, closed):
    """(start, end, whole) of the stretches of a slice that are free and of one label; on a closed slice
    a stretch may run on past its end (indices modulo its length); ``whole`` for the whole loop."""
    n = len(label)
    if closed and free.all() and (label == label[0]).all():
        return [(0, n, True)]
    out = []
    if closed:  # a stretch starts at a free point after a taken one or a change of label
        for a in np.flatnonzero(free & (~np.roll(free, 1) | (label != np.roll(label, 1)))):
            b = a + 1
            while b < a + n and free[b % n] and label[b % n] == label[a]:
                b += 1
            out.append((int(a), int(b), False))
        return out
    a = 0
    while a < n:
        if not free[a]:
            a += 1
            continue
        b = a + 1
        while b < n and free[b] and label[b] == label[a]:
            b += 1
        out.append((a, b, False))
        a = b
    return out


def _min_distance(opt, corners):
    d0 = float(opt.target_min_distance)
    if opt.scale_mode == "height":
        h = float(np.ptp(corners[:, opt.up_axis]))
        if h <= 0:
            raise ValueError("cannot scale: the model is flat along the up axis")
        d0 /= float(opt.target_height) / h
    if not d0 > 0:
        raise ValueError("the minimum distance must be positive")
    return d0


def _spread_guess(parts, opt, mesh, offsets, areas, d0, budget):
    """How much sparser than asked the drawing will stand to fit the budget (a quick look, as in
    ``layout``): the curves are drawn at that scale."""
    tris = mesh.triangles
    s1, sf1 = opt.line_ratio * d0, opt.fill_ratio * d0
    ext = np.sort(np.ptp(tris.reshape(-1, 3), axis=0))
    quick = _pic.Picture(mesh, offsets, opt, max(s1 / _pic.PX, math.sqrt(float(ext[1] * ext[2]) / _layout.ESTIMATE_PIXELS)))
    area, length = _pic.estimate(quick, len(parts), max(0.5 * s1, 2.0 * quick.h))
    share = np.array([_pic.SHARE[p.options.role] if p.has_lines else 0.0 for p in parts])
    fill_on = np.array([p.has_fill for p in parts])

    def guess(f):
        s, sf = s1 * _layout.lines_factor(f), sf1 * f
        n_sil = length / s
        n_fill = np.where(fill_on, np.maximum(area - length * _pic.FILL_GAP * sf, 0.0) / (_layout.HEX * sf * sf), 0.0)
        return float((n_sil + n_fill + _layout.INNER_USE * share / (1.0 - share) * (n_sil + n_fill)).sum())

    if guess(1.0) <= budget:
        return 1.0
    lo, hi = 1.0, 2.0
    while guess(hi) > budget and hi < 1e6:
        lo, hi = hi, 2.0 * hi
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if guess(mid) > budget else (lo, mid)
    return hi


def _envelope(pic, s):
    """The front of the model with its recesses filled and smoothed (nearness, as ``Picture``): a
    carved eye is no hole in the shell."""
    fg = pic.fg
    near = np.where(fg, pic.nearness, 0.0)
    f = fg.astype(np.float64)

    def smooth(img, sigma):
        return _blur_mean(img, f, sigma)

    filled = np.maximum(near, smooth(near, ENVELOPE * s / pic.h))
    return np.where(fg, smooth(np.where(fg, filled, 0.0), SHELL_SMOOTH * s / pic.h), near)


def _blur_mean(img, weight, sigma):
    """Mean of ``img`` over ``weight`` (a mask) blurred over ``sigma`` pixels; a big blur is taken on the
    picture shrunk to match."""
    k = max(1, int(sigma / 4.0))
    if k > 1:
        H, W = img.shape
        h2, w2 = -(-H // k), -(-W // k)
        pad = ((0, h2 * k - H), (0, w2 * k - W))
        a = np.pad(img * weight, pad).reshape(h2, k, w2, k).sum(axis=(1, 3))
        b = np.pad(weight, pad).reshape(h2, k, w2, k).sum(axis=(1, 3))
        a, b = _outline._blur(a, sigma / k), _outline._blur(b, sigma / k)
        out = a / np.maximum(b, 1e-300)
        return np.repeat(np.repeat(out, k, axis=0), k, axis=1)[:H, :W]
    if sigma < 0.3:
        return img
    return _outline._blur(img * weight, sigma) / np.maximum(_outline._blur(weight, sigma), 1e-300)


def _unproject(pic, xy, depth):
    """Model points under picture points ``xy`` at ``depth`` (nearness, as ``Picture``)."""
    xy = np.asarray(xy, np.float64)
    depth = np.broadcast_to(np.asarray(depth, np.float64), (len(xy),))
    if pic.eye is None:
        return xy[:, :1] * pic.e1 + xy[:, 1:2] * pic.e2 + depth[:, None] * pic.d
    dist = -depth  # nearness of a camera is minus the distance
    return pic.eye + (xy[:, :1] * pic.e1 + xy[:, 1:2] * pic.e2) * (dist / pic.far)[:, None] - dist[:, None] * pic.d


def _drawn(st, s, simple=False):
    """The picture points of a stroke as the drones will draw it: smoothed between its corners (a simple
    shape as it is; with ``simple`` a nearly straight stretch straight), and the indices of the fewest of
    them that keep within ``SIMPLIFY`` of it."""
    n = len(st.xy)
    if st.shape is not None or n < 3:
        pts = st.xy
        corners = []
    else:
        sig = _pic.SMOOTH * s / st.step
        cs = list(st.corners)
        if st.closed and not cs:
            pts, corners = _pic.smooth(st.xy, sig, True), []
        else:
            marks = sorted(set(cs) | ({0, n - 1} if not st.closed else set()))
            bounds = list(zip(marks, marks[1:] + ([marks[0] + n] if st.closed else [])))
            pieces = [_pic.smooth(_pic._take(st.xy, a, b + 1), sig, False) for a, b in bounds]
            if simple:
                pieces = [_shapes.straighten(p, s) for p in pieces]
            pts = np.vstack([p[:-1] for p in pieces] + ([] if st.closed else [pieces[-1][-1:]]))
            at, corners = 0, []
            for p, (a, b) in zip(pieces, bounds):
                if a in cs:
                    corners.append(at)
                at += len(p) - 1
    return pts, _rdp(pts, SIMPLIFY * s, st.closed, corners)


def _rdp(p, tol, closed, fixed=()):
    """Indices of the points of polyline ``p`` Ramer-Douglas-Peucker keeps within ``tol``, ``fixed``
    (corners) always."""
    n = len(p)
    if n < 3:
        return np.arange(n)
    keep = np.zeros(n, bool)
    keep[list(fixed)] = True
    if closed:  # split a loop at its corners, or at two far points
        anchors = sorted(set(fixed)) or [0, int(np.argmax(np.linalg.norm(p - p[0], axis=1)))]
        keep[anchors] = True
        spans = list(zip(anchors, anchors[1:] + [anchors[0] + n]))
    else:
        keep[[0, n - 1]] = True
        anchors = sorted(set(fixed) | {0, n - 1})
        spans = list(zip(anchors[:-1], anchors[1:]))
    stack = list(spans)
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        idx = np.arange(a + 1, b) % n
        pa, pb = p[a % n], p[b % n]
        d = pb - pa
        length = float(np.linalg.norm(d))
        q = p[idx] - pa
        if length > 0:
            proj = np.clip(q @ d / length ** 2, 0.0, 1.0)
            off = np.linalg.norm(q - proj[:, None] * d, axis=1)
        else:
            off = np.linalg.norm(q, axis=1)
        i = int(np.argmax(off))
        if off[i] > tol:
            m = a + 1 + i
            keep[m % n] = True
            stack += [(a, m), (m, b)]
    return np.flatnonzero(keep)


def vivid(rgb, level: float = 1.0):
    """``rgb`` (linear) at its full brightness times ``level`` (sRGB): the hue and saturation kept, the
    brightest channel at ``level``. A dark colour (a carved eye) glows as bright as the rest."""
    srgb = _color.linear_to_srgb(np.asarray(rgb, np.float64))
    top = float(srgb.max())
    if top <= 1e-6:
        return np.full(3, _color.srgb_to_linear(np.array([level]))[0])
    return _color.srgb_to_linear(srgb / top * level)


def _stroke_color(pic, lin, st, s):
    """The colour a stroke glows in: its own, or (a colour border) the side that stands out from the
    part's colour, or the colour under it."""
    if st.color is not None:
        return np.asarray(st.color, np.float64)
    pts = st.xy[:: max(1, len(st.xy) // 64)]
    stand = None
    if st.kind == "colors":
        mean = _color.linear_to_oklab(lin[pic.part == st.part]).mean(axis=0)
        stand = np.broadcast_to(mean, (len(pts), 3))
    c = _pic.colors_near(pic, lin, pts, st.part, 0.5 * s if stand is not None else 0.4 * s, stand)
    c = c[~np.isnan(c).any(axis=1)]
    return c.mean(axis=0) if len(c) else np.ones(3)


def _shell(pic, k, s, lift, lin):
    """The smooth shell of part ``k`` the curves lie on: a grid over its picture, its corners lifted
    like the curves, coloured like the model under them."""
    mask = pic.part == k
    if not mask.any():
        return None
    step = max(int(round(SHELL_STEP * s / pic.h)), 1)
    rows, cols = np.nonzero(mask)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    gr = np.arange(r0, r1 + step, step)
    gc = np.arange(c0, c1 + step, step)
    R, C = np.meshgrid(gr, gc, indexing="ij")
    H, W = pic.shape
    grown = _pic._grow(mask, max(step // 2, 1))
    inside = grown[np.clip(R, 0, H - 1), np.clip(C, 0, W - 1)] & (R < H) & (C < W)
    quad = inside[:-1, :-1] & inside[1:, :-1] & inside[:-1, 1:] & inside[1:, 1:]
    if not quad.any():
        return None
    idx = -np.ones(R.shape, np.int64)
    used = np.zeros(R.shape, bool)
    qi, qj = np.nonzero(quad)
    for di, dj in ((0, 0), (1, 0), (0, 1), (1, 1)):
        used[qi + di, qj + dj] = True
    idx[used] = np.arange(int(used.sum()))
    xy = pic.pixel_xy(R[used], C[used])
    verts = lift(xy)
    rr, cc = np.clip(R[used], 0, H - 1), np.clip(C[used], 0, W - 1)
    vcol = lin[rr, cc]
    a, b, c, d = idx[qi, qj], idx[qi + 1, qj], idx[qi, qj + 1], idx[qi + 1, qj + 1]
    faces = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])
    tris = verts[faces]
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    toward = normalize(pic.toward) if pic.eye is not None else pic.d
    flip = n @ toward < 0  # every face looks at the audience
    faces[flip] = faces[flip][:, ::-1]
    return MeshData(verts[faces], vertex_colors=vcol[faces])


# ---------------------------------------------------------------- drones on curves

def compose_curves(sketches: List[Sketch], options: DistributeOptions = None) -> DroneResult:
    """Drones on the curves of ``sketches``, at most ``options.count``: a drone at every corner (where a
    curve turns sharply), the spacing growing from the minimum distance there to the line spacing along
    the curves, a small closed curve at least a ring; then the fill inside the closed curves (on the
    shell, or in their own plane), clear of the curves. Short of drones, the fill gives way first."""
    opt = options or DistributeOptions()
    budget = int(opt.count)
    if budget < 1:
        raise ValueError("count must be at least 1")
    sets = [sk for sk in sketches if sk.curves]
    if not sets:
        raise ValueError("no curves")
    allpts = np.vstack([c.points for sk in sets for c in sk.curves])
    lo, hi = allpts.min(axis=0), allpts.max(axis=0)
    scale = 1.0
    if opt.scale_mode == "height":
        h = float(hi[opt.up_axis] - lo[opt.up_axis])
        if h <= 0:
            raise ValueError("cannot scale: the drawing is flat along the up axis")
        scale = float(opt.target_height) / h
    d0 = float(opt.target_min_distance) / scale
    frames = [_frame(sk, opt) for sk in sets]
    shells = [_ShellLookup(sk.shell, fr) if sk.shell is not None and sk.shell.triangle_count else None
              for sk, fr in zip(sets, frames)]

    def plan(f):
        fl = _layout.lines_factor(f)
        s, sf, near = opt.line_ratio * d0 * fl, opt.fill_ratio * d0 * f, d0 * fl * _layout.CLEAR
        lines = [_on_curves(sk, fr, near, s) for sk, fr in zip(sets, frames)]
        fills = [_fill(sk, fr, sh, sf, ln[0], s) if _has_fill(sk) else np.zeros((0, 3))
                 for sk, fr, sh, ln in zip(sets, frames, shells, lines)]
        n_line = sum(len(ln[0]) for ln in lines)
        n_fill = sum(len(x) for x in fills)
        return dict(f=f, lines=lines, fills=fills, n_line=n_line, n_fill=n_fill, total=n_line + n_fill)

    best, last, needed = None, None, None
    f, over, under = 1.0, None, None
    for _ in range(_layout.ROUNDS + 4):
        last = plan(f)
        if f == 1.0:
            needed = last["total"]
        if last["total"] <= budget:
            if best is None or last["total"] > best["total"]:
                best = last
            if f == 1.0 or last["total"] >= _layout.FIT * budget:
                break
            under = f
        else:
            over = f
        if over is not None and under is not None:
            if under / over < 1.01:
                break
            f = math.sqrt(over * under)
        else:
            a, b = last["n_line"] * f, last["n_fill"] * f * f
            g = (a + math.sqrt(a * a + 4.0 * b * budget)) / (2.0 * budget)
            f = max(g, f * 1.02) if last["total"] > budget else max(min(g, f / 1.02), 1.0)
    plan_ = best or last
    f = plan_["f"]
    fl = _layout.lines_factor(f)
    pos, kind, part, colors, normals, rank = [], [], [], [], [], []
    for k, ((lp, lk, lc, lr, ln), fp, sk, fr, sh) in enumerate(zip(plan_["lines"], plan_["fills"], sets, frames,
                                                                   shells)):
        pos += [lp, fp]
        kind += [lk, np.full(len(fp), 2, np.int8)]
        rank += [ln, np.full(len(fp), 4, np.int8)]
        part += [np.full(len(lp) + len(fp), k)]
        fc = sh.colors(fp) if sh is not None else _nearest_color(fp, lp, lc)
        colors += [lc, _color.srgb_to_linear(_color.linear_to_srgb(fc) * FILL_GLOW)]  # the fill a background
        normals.append(np.broadcast_to(fr[0], (len(lp) + len(fp), 3)))
    points, kind, part = np.vstack(pos), np.concatenate(kind), np.concatenate(part)
    colors, normals, rank = np.vstack(colors), np.vstack(normals), np.concatenate(rank)
    radius = np.full(len(points), d0 * fl)
    order = np.argsort(rank, kind="stable")
    points, kind, part, colors, normals = (x[order] for x in (points, kind, part, colors, normals))
    keep = _outline.park(points, radius)[:budget]  # nobody closer than the minimum distance
    points, kind, part, colors, normals = (x[keep] for x in (points, kind, part, colors, normals))
    n = len(points)
    if f == 1.0:
        needed = n
    if opt.color_mode == "none":
        colors = np.ones((n, 3))
    colors = _color.glow(np.where(np.isnan(colors), 1.0, colors), opt.min_brightness)
    dmin, dmean = spacing(points, opt.fill_ratio * d0 * f)
    plo, phi = points.min(axis=0), points.max(axis=0)
    return DroneResult(
        positions=points,
        normals=normals,
        triangle_ids=np.zeros(n, np.int64),
        barycentrics=np.tile([1.0, 0.0, 0.0], (n, 1)),
        colors_linear=colors,
        min_distance=dmin,
        mean_distance=dmean,
        area=0.0,
        scale=scale,
        pivot=scale_pivot(plo, phi, opt.up_axis),
        options=opt,
        visibility=np.ones(n),
        outline=kind < 2,
        outline_spacing=opt.line_ratio * d0 * fl if (kind < 2).any() else 0.0,
        kind=kind,
        part=part,
        budget=budget,
        needed=int(needed),
        spread=fl,
        fill_spread=f,
    )


def _has_fill(sk):
    o = sk.options
    return not sk.solid and (o is None or o.role != "outline")


def _frame(sk, opt):
    """(normal, e1, e2, depth of the plane along the normal) of a sketch: its own normal, or the plane its
    points lie closest to, facing the audience."""
    pts = np.vstack([c.points for c in sk.curves])
    if sk.normal is not None:
        n = normalize(sk.normal)
    else:
        centred = pts - pts.mean(axis=0)
        _, sv, vt = np.linalg.svd(centred, full_matrices=False) if len(pts) >= 3 else (None, None, None)
        n = vt[-1] if vt is not None and sv[1] > 1e-9 else normalize(opt.view_dir)
        if n @ normalize(opt.view_dir) < 0:
            n = -n
    _, e1, e2 = view_basis(n)
    return n, e1, e2, float(np.median(pts @ n))


def _sample(points, closed, step):
    """The polyline every ``step`` or a hair less, and the index in it of each of its points."""
    q = np.vstack([points, points[:1]]) if closed else points
    seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
    m = np.maximum(np.ceil(seg / step).astype(np.int64), 1)
    t = np.concatenate([np.arange(k) / k for k in m])
    i = np.repeat(np.arange(len(seg)), m)
    out = q[i] * (1 - t)[:, None] + q[i + 1] * t[:, None]
    where = np.r_[0, np.cumsum(m)][: len(points)]
    if not closed:
        out = np.vstack([out, q[-1:]])
    return out, where


def _corner_points(points, closed, near):
    """Indices of the points of a curve where it turns sharply (``picture.CORNER_TURN`` over about a
    minimum distance either side, and still most of it over twice that)."""
    n = len(points)
    if n < 3:
        return []
    out = []
    for i in range(n):
        if not closed and (i == 0 or i == n - 1):
            continue
        a = _walk(points, closed, i, -1, near)
        b = _walk(points, closed, i, 1, near)
        a2 = _walk(points, closed, i, -1, 2 * near)
        b2 = _walk(points, closed, i, 1, 2 * near)
        if a is None or b is None:
            continue
        t1 = _angle(points[i] - a, b - points[i])
        t2 = _angle(points[i] - a2, b2 - points[i]) if a2 is not None and b2 is not None else t1
        if t1 >= _pic.CORNER_TURN and t2 >= _pic.CORNER_KEEP * _pic.CORNER_TURN:
            out.append(i)
    return out


def _walk(points, closed, i, sign, dist):
    """The point ``dist`` along the curve from point ``i`` (backwards for ``sign`` -1), or None past an end."""
    n = len(points)
    left, j = dist, i
    while True:
        k = j + sign
        if closed:
            k %= n
            if k == i:
                return None
        elif k < 0 or k >= n:
            return None
        seg = float(np.linalg.norm(points[k] - points[j]))
        if seg >= left:
            return points[j] + (points[k] - points[j]) * (left / max(seg, 1e-300))
        left -= seg
        j = k


def _angle(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return 0.0
    return math.degrees(math.acos(float(np.clip(a @ b / (na * nb), -1.0, 1.0))))


def _on_curves(sk, frame, near, s):
    """Drones on the curves of a sketch: (positions, kind 0/1, colours, local spacing, rank: the order
    they are taken in, the slices of a volume last). Corners are where a curve turns sharply as the
    audience sees it (in the plane of the sketch; in space, for a solid one)."""
    _, e1, e2, _ = frame
    pos, kind, col, sig, last = [], [], [], [], []
    for c in sk.curves:
        colour = np.ones(3) if c.color is None else np.asarray(c.color, np.float64)
        p = np.asarray(c.points, np.float64)
        if len(p) == 1:
            pos.append(p), kind.append([1]), col.append([colour]), sig.append([s])
            last.append(np.zeros(1, bool))
            continue
        closed = c.closed and len(p) > 2
        corners = _corner_points(p if sk.solid else np.stack([p @ e1, p @ e2], 1), closed, near)
        dense, where = _sample(p, closed, SAMPLE * near)
        cs = [int(where[i]) for i in corners]
        n = len(dense)
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(np.vstack([dense, dense[:1]]) if closed else dense,
                                                         axis=0), axis=1))]
        if closed and not cs:
            total = arc[-1]
            g = max(int(round(total / s)), min(_shapes.RING_MIN, int(total // near)), 3 if total >= 3 * near else 1)
            x = np.arange(g) * (total / g)
            pts = _along(dense, arc, x, True)
            pos.append(pts), kind.append(np.ones(g, np.int8)), col.append([colour] * g), sig.append(np.full(g, s))
            last.append(np.full(g, c.kind == "slice"))
            continue
        for i in cs:
            pos.append(dense[i:i + 1]), kind.append([0]), col.append([colour]), sig.append([near])
        if closed:
            bounds = list(zip(cs, cs[1:] + [cs[0] + n]))
            ends = [(True, True)] * len(bounds)
        else:
            marks = sorted(set([0] + cs + [n - 1]))
            bounds = list(zip(marks[:-1], marks[1:]))
            ends = [(a in cs, b in cs) for a, b in bounds]
        for (a, b), (c0, c1) in zip(bounds, ends):
            piece = _pic._take(dense, a, b + 1)
            if len(piece) < 2:
                continue
            parc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(piece, axis=0), axis=1))]
            x, sigma = _pic._graded(parc[-1], c0, c1, near, s)
            if not len(x):
                continue
            pos.append(_along(piece, parc, x, False)), kind.append(np.ones(len(x), np.int8))
            col.append([colour] * len(x)), sig.append(sigma)
        while len(last) < len(pos):  # the corners and pieces of this curve
            last.append(np.full(len(pos[len(last)]), c.kind == "slice"))
    if not pos:
        e = np.zeros((0, 3))
        return e, np.zeros(0, np.int8), e, np.zeros(0), np.zeros(0, np.int8)
    p = np.vstack(pos)
    k = np.concatenate([np.asarray(x, np.int8) for x in kind])
    c = np.vstack([np.asarray(x, np.float64).reshape(-1, 3) for x in col])
    sg = np.concatenate([np.asarray(x, np.float64) for x in sig])
    r = (k + 2 * np.concatenate(last)).astype(np.int8)  # corners, lines, then the slices' corners and lines
    order = np.argsort(r, kind="stable")
    p, k, c, sg, r = p[order], k[order], c[order], sg[order], r[order]
    keep = _outline.park(p, np.where(k == 0, _pic.CORNER_GAP_FACTOR * s, np.maximum(near, _pic.LINE_GAP * sg)))
    return p[keep], k[keep], c[keep], sg[keep], r[keep]


def _along(dense, arc, x, closed):
    q = np.vstack([dense, dense[:1]]) if closed else dense
    if closed:
        x = np.mod(x, arc[-1])
    return np.stack([np.interp(x, arc, q[:, i]) for i in range(3)], 1)


def _inside(uv, loops):
    """True for the plane points ``uv`` inside the closed curves ``loops`` (even-odd: a closed curve
    inside another, an eye in a face, is a hole)."""
    out = np.zeros(len(uv), bool)
    for loop in loops:
        out ^= _pic._inside(uv, loop)
    return out


def _fill(sk, frame, shell, sf, line_pts, s):
    """Fill drones of a sketch: a lattice of spacing ``sf`` in its plane inside its closed curves (and the
    loops open curves make whose ends meet, within ``LOOP_GAP`` line spacings ``s``), the shift that keeps
    the most, clear of the curves; on the shell when it has one."""
    n, e1, e2, depth = frame
    loops = [np.stack([c.points @ e1, c.points @ e2], 1) for c in sk.curves if c.closed and len(c.points) > 2]
    loops += _joined([np.stack([c.points @ e1, c.points @ e2], 1) for c in sk.curves
                      if not c.closed and len(c.points) > 1], LOOP_GAP * s)
    if not loops:
        return np.zeros((0, 3))
    lines2 = [np.stack([c.points @ e1, c.points @ e2], 1) for c in sk.curves if len(c.points) > 1]
    dense = np.vstack([_sample(l2, False, 0.25 * sf)[0] for l2 in lines2]) if lines2 else np.zeros((0, 2))
    if len(line_pts):
        dense = np.vstack([dense, np.stack([line_pts @ e1, line_pts @ e2], 1)])
    allp = np.vstack(loops)
    lo, hi = allp.min(axis=0) - sf, allp.max(axis=0) + sf
    pattern = "square" if sk.options is not None and sk.options.pattern == "square" else "hex"
    a1 = np.array([sf, 0.0])
    a2 = np.array([0.0, sf]) if pattern == "square" else np.array([0.5 * sf, math.sqrt(3.0) / 2.0 * sf])
    nj = int(math.ceil((hi[1] - lo[1]) / a2[1])) + 2
    ni = int(math.ceil((hi[0] - lo[0]) / sf)) + nj + 2
    if ni * nj > 4_000_000:
        return np.zeros((0, 3))
    jj, ii = np.mgrid[0:nj, -nj:ni - nj]
    base = lo + ii.reshape(-1, 1) * a1 + jj.reshape(-1, 1) * a2
    base = base[(base[:, 0] >= lo[0] - sf) & (base[:, 0] <= hi[0] + sf)]
    best = np.zeros((0, 2))
    for u in range(_pic.PHASES):
        for v in range(_pic.PHASES):
            p = base + (u / _pic.PHASES) * a1 + (v / _pic.PHASES) * a2
            p = p[_inside(p, loops)]
            if len(p) and len(dense):
                p = p[~_pic._near(p, dense, _pic.FILL_GAP * sf)]
            if len(p) > len(best):
                best = p
    if not len(best):
        return np.zeros((0, 3))
    if shell is not None:
        return shell.lift(best)
    return best[:, :1] * e1 + best[:, 1:2] * e2 + depth * n


def _joined(lines, reach):
    """The loops open ``lines`` (plane polylines) make end to end, where an end meets the nearest end of
    another within ``reach`` and that end's nearest is it (an outline cut in pieces where it ran close to
    itself); the lines left over make none."""
    if len(lines) < 2:
        return []
    ends = np.array([l[e] for l in lines for e in (0, -1)])
    d = np.linalg.norm(ends[:, None] - ends[None], axis=2)
    d[np.arange(len(ends)), np.arange(len(ends))] = np.inf
    d[np.arange(0, len(ends), 2), np.arange(1, len(ends), 2)] = np.inf  # not a line to itself
    d[np.arange(1, len(ends), 2), np.arange(0, len(ends), 2)] = np.inf
    best = np.argmin(d, axis=1)
    mate = {i: int(best[i]) for i in range(len(ends)) if d[i, best[i]] <= reach and best[best[i]] == i}
    loops, seen = [], set()
    for start in range(len(lines)):
        if start in seen:
            continue
        chain, line, enter = [], start, 0
        while True:
            seen.add(line)
            chain.append(lines[line] if enter == 0 else lines[line][::-1])
            nxt = mate.get(2 * line + 1 - enter)
            if nxt is None:
                break
            line, enter = nxt // 2, nxt % 2
            if line == start:
                if enter == 0 and len(chain) > 1:
                    loops.append(np.vstack(chain))
                break
            if line in seen:
                break
    return loops


class _ShellLookup:
    """Points of the shell over plane points (along the normal of the sketch), and its colours."""

    def __init__(self, mesh, frame):
        n, e1, e2, depth = frame
        self.mesh, self.frame = mesh, frame
        t = mesh.triangles
        self.uv = np.stack([t @ e1, t @ e2], -1)  # (T, 3, 2)
        self.grid = _ProjectedGrid(self.uv[..., 0], self.uv[..., 1], np.arange(len(t)))

    def lift(self, uv):
        n, e1, e2, depth = self.frame
        q, t = self.grid.pairs(uv[:, 0], uv[:, 1])
        lam = _pic._bary2(uv[q], self.uv[t])
        hit = lam.min(axis=1) >= -1e-9
        q, t, lam = q[hit], t[hit], lam[hit]
        first = np.unique(q, return_index=True)[1]
        q, t, lam = q[first], t[first], lam[first]
        out = uv[:, :1] * e1 + uv[:, 1:2] * e2 + depth * n  # off the shell: its plane
        out[q] = np.einsum("ij,ijk->ik", lam, self.mesh.triangles[t])
        return out

    def colors(self, pts):
        if not len(pts):
            return np.zeros((0, 3))
        n, e1, e2, depth = self.frame
        uv = np.stack([pts @ e1, pts @ e2], 1)
        q, t = self.grid.pairs(uv[:, 0], uv[:, 1])
        lam = _pic._bary2(uv[q], self.uv[t])
        hit = lam.min(axis=1) >= -1e-9
        q, t, lam = q[hit], t[hit], lam[hit]
        first = np.unique(q, return_index=True)[1]
        out = np.ones((len(pts), 3))
        out[q[first]] = _color.surface_colors(self.mesh, t[first], np.clip(lam[first], 0.0, 1.0))
        return out


def _nearest_color(points, line_pts, line_colors):
    if not len(points):
        return np.zeros((0, 3))
    if not len(line_pts):
        return np.ones((len(points), 3))
    from .spatial import nearest
    _, idx = nearest(points, line_pts, 1.0)
    return line_colors[idx]
