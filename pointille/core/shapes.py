"""Simple shapes: what a designer draws where the traced line is only nearly one.

People read a drone picture by its simple shapes, so where the model has one
the drones draw it exactly:

* a closed line that is nearly an ellipse (an eye, a wheel, the rim of a
  clock) is that ellipse, the drones evenly round it;
* a stretch of line between two corners that is nearly straight is straight;
* an eye too small for the drones to draw its outline becomes the smallest
  ring they can draw, with a pupil in it when there is room;
* a round body (a ball, a pumpkin, the balls of a snowman) is filled with
  rings round its vertical axis, from the bottom up, instead of a flat
  lattice: the rings show that it is round.

The rings follow the real surface: they are where horizontal planes cut it,
as far apart along the surface as the fill spacing, so a pumpkin keeps its
ribs and a ball its poles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from . import outline as _outline

ELLIPSE_RMS = 0.06  # a closed line is an ellipse when it keeps this close to one (share of its minor radius) ...
ELLIPSE_TOLERANCE = 0.25  # ... and within this many line spacings of it
ELLIPSE_FLAT = 0.25  # an ellipse flatter than this (minor / major) is no simple shape: a line
STRAIGHT = 0.12  # a stretch between corners straying less than this many line spacings from its chord is straight
RING_MIN = 6  # the fewest drones a ring has (fewer read as a polygon)
EYE_GROW = 2.0  # an eye may be drawn up to this many times its size to make it a ring (shows enlarge eyes)
PUPIL = 2.0  # a ring this many minimum distances across (its minor radius) gets a drone in the middle
ROUND_RMS = 0.04  # a body is round when its surface keeps this close to a spheroid round the vertical axis ...
ROUND_ASPECT = (0.45, 2.2)  # ... no flatter and no taller than this (height / width) ...
ROUND_SPAN = 0.6  # ... and it spans this share of the spheroid's height and width (a painted band does not)
ROUND_SIZE = 1.5  # ... and it is at least this many fill spacings across: smaller ones take no rings
SLOPE_FLOOR = 0.2  # where the surface is nearly level (a pole), rings are this much closer in height
_LEVELS = 512


# ---------------------------------------------------------------- ellipses and straight lines

@dataclass
class Ellipse:
    center: np.ndarray  # (2,)
    axes: np.ndarray  # (2,) semi-axes, the major first
    angle: float  # direction of the major axis, radians
    rms: float  # how far the fitted points are from it, root mean square

    @property
    def minor(self) -> float:
        return float(self.axes[1])

    def scaled(self, k: float) -> "Ellipse":
        return Ellipse(self.center, self.axes * k, self.angle, self.rms)

    def perimeter(self) -> float:
        a, b = self.axes
        h = ((a - b) / (a + b)) ** 2 if a + b > 0 else 0.0
        return float(math.pi * (a + b) * (1.0 + 3.0 * h / (10.0 + math.sqrt(max(4.0 - 3.0 * h, 0.0)))))

    def points(self, count: int, start_up: bool = True) -> np.ndarray:
        """``count`` points evenly spaced along the ellipse, the first on top (the highest point)."""
        t = np.linspace(0.0, 2.0 * math.pi, 2049)
        c, s = math.cos(self.angle), math.sin(self.angle)
        ex, ey = self.axes[0] * np.cos(t), self.axes[1] * np.sin(t)
        p = np.stack([c * ex - s * ey, s * ex + c * ey], 1) + self.center
        if start_up:  # walk round from the top
            p = np.roll(p[:-1], -int(np.argmax(p[:-1, 1])), axis=0)
            p = np.vstack([p, p[:1]])
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        x = np.arange(count) * (arc[-1] / max(count, 1))
        return np.stack([np.interp(x, arc, p[:, i]) for i in range(2)], 1)


def fit_ellipse(xy: np.ndarray) -> Optional[Ellipse]:
    """The ellipse nearest the points ``xy`` (N, 2): Fitzgibbon's direct least squares, in the
    numerically stable form of Halir and Flusser. None when they make no ellipse."""
    xy = np.asarray(xy, np.float64)
    if len(xy) < 6:
        return None
    m = xy.mean(axis=0)
    scale = float(np.ptp(xy, axis=0).max()) or 1.0
    x, y = ((xy - m) / scale).T
    D1 = np.stack([x * x, x * y, y * y], 1)
    D2 = np.stack([x, y, np.ones_like(x)], 1)
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    try:
        T = -np.linalg.solve(S3, S2.T)
    except np.linalg.LinAlgError:
        return None
    M = S1 + S2 @ T
    M = np.array([M[2] / 2.0, -M[1], M[0] / 2.0])
    try:
        w, v = np.linalg.eig(M)
    except np.linalg.LinAlgError:
        return None
    v = np.real(v)
    cond = 4.0 * v[0] * v[2] - v[1] ** 2
    if not (cond > 0).any():
        return None
    a1 = v[:, int(np.argmax(cond > 0))]
    A, B, C = a1
    D, E, F = T @ a1
    den = B * B - 4.0 * A * C
    if den >= 0:
        return None
    x0 = (2.0 * C * D - B * E) / den
    y0 = (2.0 * A * E - B * D) / den
    Fc = A * x0 * x0 + B * x0 * y0 + C * y0 * y0 + D * x0 + E * y0 + F
    lam, vec = np.linalg.eigh(np.array([[A, B / 2.0], [B / 2.0, C]]))
    ax2 = -Fc / lam
    if not (ax2 > 0).all():
        return None
    axes = np.sqrt(ax2)
    major = int(np.argmax(axes))
    angle = math.atan2(vec[1, major], vec[0, major])
    e = Ellipse(np.array([x0, y0]) * scale + m, np.array([axes[major], axes[1 - major]]) * scale, angle, 0.0)
    e.rms = float(np.sqrt(np.mean(ellipse_distance(e, xy) ** 2)))
    return e


def ellipse_distance(e: Ellipse, xy: np.ndarray) -> np.ndarray:
    """About how far each point is from the ellipse (along the ray from its centre)."""
    c, s = math.cos(e.angle), math.sin(e.angle)
    d = np.asarray(xy, np.float64) - e.center
    u, v = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
    rho = np.sqrt((u / e.axes[0]) ** 2 + (v / e.axes[1]) ** 2)
    return np.abs(rho - 1.0) * np.hypot(u, v) / np.maximum(rho, 1e-12)


def as_ellipse(xy: np.ndarray, s: float) -> Optional[Ellipse]:
    """The ellipse a closed line drawn at line spacing ``s`` is, or None when it is no simple one."""
    e = fit_ellipse(xy)
    if e is None or e.axes[1] < ELLIPSE_FLAT * e.axes[0]:
        return None
    dist = ellipse_distance(e, xy)
    if e.rms > ELLIPSE_RMS * e.minor or float(dist.max()) > max(ELLIPSE_TOLERANCE * s, 2.5 * ELLIPSE_RMS * e.minor):
        return None
    # the points must go all round it, not along a piece of it
    c, s_ = math.cos(e.angle), math.sin(e.angle)
    d = xy - e.center
    ang = np.arctan2(-s_ * d[:, 0] + c * d[:, 1], c * d[:, 0] + s_ * d[:, 1])
    if len(np.unique(np.floor((ang + math.pi) / (2.0 * math.pi) * 12.0).astype(int))) < 11:
        return None
    return e


def blob_ellipse(rows, cols, h: float, lo, hi) -> Optional[Ellipse]:
    """The ellipse of a filled blob of pixels (its second moments), in picture units."""
    if len(rows) < 3:
        return None
    xy = np.stack([lo[0] + (cols + 0.5) * h, hi[1] - (rows + 0.5) * h], 1)
    m = xy.mean(axis=0)
    cov = np.cov((xy - m).T) + np.eye(2) * (h * h / 12.0)
    lam, vec = np.linalg.eigh(cov)
    axes = 2.0 * np.sqrt(np.maximum(lam[::-1], 1e-300))
    return Ellipse(m, axes, math.atan2(vec[1, 1], vec[0, 1]), 0.0)


def ring(e: Ellipse, near: float, s: float):
    """The ring a drone show draws for the ellipse ``e`` (an eye) with drones at least ``near`` apart,
    about ``s`` apart where it is big: (ellipse drawn, drones on it, pupil or not). None when the
    ellipse is too small even grown ``EYE_GROW`` times."""
    if e.axes[1] < ELLIPSE_FLAT * e.axes[0]:
        return None
    count = max(RING_MIN, int(round(e.perimeter() / s)))
    grow = max(1.0, near * count / max(e.perimeter(), 1e-12))
    while grow > 1.0 and count > RING_MIN:  # fewer drones before a bigger ring
        count -= 1
        grow = max(1.0, near * count / max(e.perimeter(), 1e-12))
    if grow > EYE_GROW:
        return None
    # the chord between neighbours is shortest at the flat ends: keep it ``near``
    drawn = e.scaled(grow)
    pts = drawn.points(count)
    while count > RING_MIN and np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1).min() < near:
        count -= 1
        pts = drawn.points(count)
    k = 1.0
    while np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1).min() < near and k < EYE_GROW:
        k *= 1.05
        pts = drawn.scaled(k).points(count)
    drawn = drawn.scaled(k)
    if grow * k > EYE_GROW * 1.05:
        return None
    return drawn, count, drawn.minor >= PUPIL * near


def straighten(p: np.ndarray, s: float) -> np.ndarray:
    """``p`` (an open polyline) as the straight line between its ends when it strays from that line
    less than ``STRAIGHT`` line spacings, else as it is."""
    if len(p) < 3:
        return p
    a, b = p[0], p[-1]
    d = b - a
    length = float(np.linalg.norm(d))
    if length < s:
        return p
    off = np.abs((p[:, 0] - a[0]) * d[1] - (p[:, 1] - a[1]) * d[0]) / length
    if float(off.max()) > STRAIGHT * s:
        return p
    return a + np.linspace(0.0, 1.0, len(p))[:, None] * d


# ---------------------------------------------------------------- round bodies and their rings

@dataclass
class Body:
    tri: np.ndarray  # the triangles of the body
    center: np.ndarray  # (3,) of the spheroid it is nearly
    radius: float  # horizontal semi-axis
    height: float  # vertical semi-axis
    rms: float  # how far its surface is from the spheroid, as a share of the radii


def pieces(tris: np.ndarray, ids: np.ndarray) -> List[np.ndarray]:
    """The connected pieces of the triangles ``ids`` (vertices merged by position)."""
    if not len(ids):
        return []
    vid, _ = _outline._weld(tris[ids])
    nv = int(vid.max()) + 1
    comp = _outline._components(nv, np.r_[vid[:, 0], vid[:, 1]], np.r_[vid[:, 1], vid[:, 2]])
    label = comp[vid[:, 0]]
    order = np.argsort(label, kind="stable")
    cuts = np.flatnonzero(np.diff(label[order])) + 1
    return [ids[g] for g in np.split(order, cuts)]


def _up_frame(up_axis: int):
    """Axis indices (horizontal, horizontal, up)."""
    return [i for i in range(3) if i != up_axis] + [up_axis]


def round_body(tris: np.ndarray, ids: np.ndarray, areas: np.ndarray, up_axis: int = 2) -> Optional[Body]:
    """The spheroid round the vertical axis the triangles ``ids`` nearly are, or None when they are no
    round body (a ball, a pumpkin, an egg are; a figure, a disc, a box are not)."""
    if len(ids) < 20:
        return None
    w = areas[ids]
    if w.sum() <= 0:
        return None
    frame = _up_frame(up_axis)
    c = tris[ids].mean(axis=1)[:, frame]
    origin = np.average(c, axis=0, weights=w)  # inside the body: the equation below cannot pass through it
    c = c - origin
    x, y, z = c[:, 0], c[:, 1], c[:, 2]
    # A (x^2 + y^2) + C z^2 + D x + E y + F z = 1, weighted by area
    M = np.stack([x * x + y * y, z * z, x, y, z], 1)
    sw = np.sqrt(w / w.sum())
    try:
        coef, *_ = np.linalg.lstsq(M * sw[:, None], sw, rcond=None)
    except np.linalg.LinAlgError:
        return None
    A, C, D, E, F = coef
    if not (A > 0 and C > 0):
        return None
    cx, cy, cz = -D / (2 * A), -E / (2 * A), -F / (2 * C)
    rhs = 1.0 + A * (cx * cx + cy * cy) + C * cz * cz
    if rhs <= 0:
        return None
    a, b = math.sqrt(rhs / A), math.sqrt(rhs / C)
    if not ROUND_ASPECT[0] <= b / a <= ROUND_ASPECT[1]:
        return None
    # the corners of the triangles, not their centres: a coarse mesh is no flatter than it is
    p = tris[ids].reshape(-1, 3)[:, frame] - origin
    rho = np.sqrt(((p[:, 0] - cx) ** 2 + (p[:, 1] - cy) ** 2) / (a * a) + ((p[:, 2] - cz) / b) ** 2)
    rms = float(np.sqrt(np.average((rho - 1.0) ** 2, weights=np.repeat(w, 3))))
    if rms > ROUND_RMS:
        return None
    span = np.ptp(p, axis=0)
    if span[2] < ROUND_SPAN * 2.0 * b or min(span[0], span[1]) < ROUND_SPAN * 2.0 * a:
        return None
    # all round the axis, not a piece of a shell
    ang = np.arctan2(c[:, 1] - cy, c[:, 0] - cx)
    sectors = np.bincount(np.floor((ang + math.pi) / (2 * math.pi) * 12).astype(int) % 12, weights=w, minlength=12)
    if (sectors > 0.2 * w.sum() / 12).sum() < 12:
        return None
    center = np.zeros(3)
    center[frame] = np.array([cx, cy, cz]) + origin
    return Body(ids, center, a, b, rms)


def round_bodies(tris: np.ndarray, ids: np.ndarray, areas: np.ndarray, least: float, up_axis: int = 2) -> List[Body]:
    """The round bodies among the pieces of the triangles ``ids`` at least ``least`` across."""
    out = []
    for piece in pieces(tris, ids):
        ext = np.ptp(tris[piece].reshape(-1, 3), axis=0)
        if float(ext.max()) < least:
            continue
        body = round_body(tris, piece, areas, up_axis)
        if body is not None:
            out.append(body)
    return out


def rings(tris: np.ndarray, ids: np.ndarray, normals: np.ndarray, spacing: float, toward, up_axis: int = 2,
          along: Optional[float] = None):
    """Drones on rings round the surface of the triangles ``ids``: where horizontal planes cut it,
    ``spacing`` apart along the surface from the bottom up, the drones ``along`` apart on each ring
    (``spacing`` by default). Every other ring is shifted half a step; a drone of each ring faces
    ``toward`` (the audience, a direction). Returns (pos, tri, bary, ring index)."""
    along = spacing if along is None else along
    e = np.zeros((0, 3))
    empty = e, np.zeros(0, np.int64), e, np.zeros(0, np.int64)
    if not len(ids):
        return empty
    T = tris[ids]
    z = T[..., up_axis]
    zlo, zhi = float(z.min()), float(z.max())
    if zhi - zlo <= 0:
        return empty
    # how far apart the rings stand in height: level ground needs them closer than a steep wall
    zc = z.mean(axis=1)
    w = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    steep = np.sqrt(np.clip(1.0 - normals[ids, up_axis] ** 2, 0.0, 1.0))  # |dz / ds| along the surface
    bins = np.clip(((zc - zlo) / (zhi - zlo) * _LEVELS).astype(int), 0, _LEVELS - 1)
    num = np.bincount(bins, weights=w * steep, minlength=_LEVELS)
    den = np.bincount(bins, weights=w, minlength=_LEVELS)
    have = np.flatnonzero(den > 0)  # a coarse mesh leaves heights between its rows of triangles empty
    slope = np.interp(np.arange(_LEVELS), have, num[have] / den[have])
    slope = np.maximum(slope, SLOPE_FLOOR)
    dz = (zhi - zlo) / _LEVELS
    s_of_z = np.r_[0.0, np.cumsum(dz / slope)]
    total = float(s_of_z[-1])
    n = max(int(round(total / spacing)), 1)
    levels = np.interp((np.arange(n) + 0.5) * (total / n), s_of_z, np.linspace(zlo, zhi, _LEVELS + 1))
    vid, _ = _outline._weld(T)
    nv = int(vid.max()) + 1
    toward = np.asarray(toward, np.float64)
    toward = toward - toward[up_axis] * np.eye(3)[up_axis]
    out_p, out_t, out_b, out_r = [], [], [], []
    for r, level in enumerate(levels):
        level = float(level) + 1e-9 * (zhi - zlo)  # off the vertices
        above = z > level
        cut = above.any(axis=1) & ~above.all(axis=1)
        tt = np.flatnonzero(cut)
        if not len(tt):
            continue
        # two edges of each cut triangle cross the plane (it passes no vertex): a segment between them
        ab = above[tt]
        crossing = ab != np.roll(ab, -1, axis=1)  # edge i runs from corner i to corner i + 1
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
        seg_t = ids[tt]
        walks, _ = _outline._chains(k0, k1, p0, p1, np.zeros(2 * len(k0)))
        for edges, fw, closed, _, _ in walks:
            f = fw[:, None]
            a, b = np.where(f, p0[edges], p1[edges]), np.where(f, p1[edges], p0[edges])
            ba, bb = np.where(f, b0[edges], b1[edges]), np.where(f, b1[edges], b0[edges])
            seg = np.linalg.norm(b - a, axis=1)
            length = float(seg.sum())
            if length <= 0:
                continue
            arc = np.r_[0.0, np.cumsum(seg)]
            m = max(int(round(length / along)), 1)
            if closed:
                # a drone faces the audience; every other ring is shifted half a step
                mid = 0.5 * (a + b)
                face = int(np.argmax(mid @ toward)) if np.linalg.norm(toward) > 0 else 0
                x0 = 0.5 * (arc[face] + arc[face + 1]) + (0.5 * length / m if r % 2 else 0.0)
                x = np.mod(x0 + np.arange(m) * (length / m), length)
            else:
                x = (np.arange(m) + 0.5) * (length / m)
            i = np.clip(np.searchsorted(arc, x, side="right") - 1, 0, len(seg) - 1)
            u = ((x - arc[i]) / np.maximum(seg[i], 1e-300))[:, None]
            out_p.append(a[i] * (1 - u) + b[i] * u)
            out_t.append(seg_t[edges[i]])
            out_b.append(ba[i] * (1 - u) + bb[i] * u)
            out_r.append(np.full(len(x), r))
    if not out_p:
        return empty
    return np.vstack(out_p), np.concatenate(out_t), np.vstack(out_b), np.concatenate(out_r)
