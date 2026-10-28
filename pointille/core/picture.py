"""Picture: the figure drawn as the audience sees it, by structure and salience.

In facade mode the audience sees one picture of the scene. A drone show draws
that picture the way a designer would, not the mesh:

1. the silhouette of every part is the frame of the drawing: closed, smooth
   lines read from the picture of the part (where two parts overlap, the one
   in front draws the edge), drones evenly spaced along them. A stretch of a
   part narrower than a line spacing (a rod, a ray, a thin leg) is one line
   along its middle;
2. inner lines - where a part covers itself, relief, colour borders - are
   candidates scored by salience: contrast (at the line's scale and at twice
   it) x length x smoothness. The best are taken first; each keeps about a
   fill spacing from the lines taken before it (crossing or meeting them is
   fine, running alongside is not) until they hold a share of the drones
   that leaves the figure a body. A small closed line (an eye) is one drone;
3. every line is smoothed along its length at drone scale before a drone is
   placed: the zigzag of pixels and of a dense, noisy mesh is gone;
4. a corner is a real one: the line turns sharply over about a line spacing
   on either side and still does over twice that, or a line ends on another;
5. the fill is a lattice the audience sees straight, a gap clear of the lines.

Everything is measured in the picture: drones evenly spaced there look even
to the audience and stand no closer in space (for a far audience).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import List, Optional

import numpy as np

from . import color as _color
from . import outline as _outline
from . import shapes as _shapes
from .facade import view_basis
from .spatial import Grid, nearest

# reading the picture (lengths in line spacings unless said otherwise)
PX = 16.0  # pixels per line spacing in the picture the lines are read from (rendered at twice that)
MASK_SIGMA = 0.1  # the silhouette is that of the picture smoothed over this much (fur, specks)
FIELD_SIGMA = 0.25  # relief and colour steps are read smoothed over this much, and over twice that
COLOR_PIXELS = 0.7  # colour regions are smoothed over this many pixels only: a thin painted line stays
REGIONS = 6  # colours the picture is cut into (fewer when they do not differ by the colour contrast)
SPOT = 0.08  # a spot (an eye) differs from its surroundings by at least this OKLab lightness
SPOT_TEXTURE = 8  # more dots than this on a part are its texture (a leopard's): only the outstanding ones count
# the shape of a line
STEP = 0.125  # lines are resampled every this much
LIGHT = 0.15  # lines are first smoothed over this much (pixel steps), for the corners
SMOOTH = 0.35  # and before the drones are placed over this much, between their corners
THIN = 1.0  # a part or a colour narrower than this ...
THIN_RUN = 1.0  # ... for at least this long is drawn by one line along its middle
ROD = 1.5  # a narrow stretch of a part whose middle line is at least this long is a rod
BLOB = 1.5  # a thick piece less than this across, among rods, is a knot of them
JUMP = 0.6  # a silhouette cut for less than this (the base of a rod) is bridged
SELF_GAP = 0.5  # a silhouette coming back this close to itself or another is drawn once
MIN_RUN = 1.0  # silhouette pieces shorter than this are dropped
SPECK = 0.5  # islands smaller than this across are dropped ...
DOT = 1.3  # ... up to this they are one drone (and so is a small, strong inner line: an eye)
ROUND_SPOT = 0.85  # a spot is round (a ring can draw it) when it overlaps its ellipse this much
# choosing the inner lines
WEIGHT = {"occlusion": 1.5, "colors": 1.2, "relief": 0.8}  # salience of the kinds of inner lines
SHARE = {"full": 0.4, "pattern": 0.5, "outline": 0.0, "fill": 0.0}  # share of a part's drones they may take
MIN_LENGTH = {"occlusion": 2.0, "colors": 2.0, "relief": 3.0}  # shorter inner lines are noise (bands: 1)
MIN_STROKE = 2.0  # and so are the pieces shorter than this that crowding leaves of them
WIGGLE = 0.8  # radians per line spacing an inner line may turn on average before it counts as noise
BRIDGE = 0.8  # pieces of a line this far apart that continue each other are one line
CROWD = 0.8  # an inner line keeps this many fill spacings from the lines taken before it ...
CROSS = 2.2  # ... unless it only crosses or meets them: near them for less than this many CROWD distances
BAND_GAP = 1.2  # a band (a painted line, a ray) keeps only this far from other lines
CRISP = 2.0  # lines this many times stronger than the threshold count half toward the share (and make bands)
DOT_STRENGTH = 1.5  # contrast (times the threshold) a small inner line or colour island needs to be a dot
MAX_DOTS = 6  # dots a part may have
DOT_CLEAR = 1.0  # inner lines keep this far from a dot
# corners and drones
CORNER_TURN = 60.0  # degrees a line turns over CORNER_WINDOW on either side to make a corner ...
INNER_TURN = 80.0  # ... on an inner line that is no band (a relief, a colour border)
CORNER_WINDOW = 0.7
CORNER_KEEP = 0.8  # ... and still turns this share of it over twice the window (noise does not)
APEX = 0.5  # a corner moves at most this far to where its sides meet
CORNER_GAP_FACTOR = 0.8  # corners closer than this to one taken before are one corner
JOIN = 0.8  # a line ending this far from another meets it there: a corner ...
JOIN_ANGLE = 35.0  # ... when it meets it at more than this many degrees
JOIN_BAND = 0.3  # ... a painted band only this far (the gap between a ray and the sun is drawn)
GRADE = 0.25  # the spacing grows by this much per length of line away from a corner
LINE_GAP = 0.7  # drones of a later line nearer than this many local spacings to one placed before go
# the fill
FILL_GAP = 0.45  # the fill keeps this many fill spacings clear of the lines
POCKET = 0.85  # where the lattice leaves a hole this many fill spacings wide, a drone fills it
PHASES = 4  # the lattice is tried at this many shifts along each of its axes
# a pair of eyes: two dots of a part about level (within this many of their radii), side by side (at most this
# many radii apart) and alike in size (the larger at most this many times the smaller)
PAIR_LEVEL = 1.5
PAIR_APART = 12.0
PAIR_SIZE = 1.6
# eyeballs the colour of the face (sculpted in their sockets): pairs of bumps standing this many line
# spacings out of the surface round them, looked for at this scale (line spacings)
BUMP = 0.02
BUMP_SIGMA = 0.15
BUMP_FALL = 0.7  # ... falling off below this share of its height two scales away all round
BUMP_MIRROR = 0.2  # the middle of a pair lies at most this share of its width off the middle of the part
BUMP_APART = 24.0  # and its bumps at most this many sizes apart (a bump is the top of an eyeball, smaller than it)
# features (feature_loops): a region inside a part ...
FEATURE_CONTRAST = 2.0  # ... differing from the ring round it by this many colour contrasts ...
FEATURE_DEPTH = 0.5  # ... or lying this many line spacings deeper than the surface round it ...
FEATURE_SURFACE = 2.0  # ... (the surface smoothed over this many line spacings) ...
FEATURE_SIZE = 1.0  # ... with the area of a disc at least this many line spacings across ...
FEATURE_RING = 0.5  # (the ring round it this wide)
FEATURE_EDGE = 0.5  # ... not touching (this near) the outline of the part ...
FEATURE_TOUCH = 0.05  # ... with more than this share of it
MAX_FEATURES = 12  # outlines of features a part may have
_NO = np.zeros(0, np.int64)


@dataclass
class Stroke:
    """A line of the drawing: picture points ``xy`` every ``step`` along it."""

    xy: np.ndarray
    closed: bool
    kind: str  # "silhouette", "occlusion", "relief", "colors"
    part: int
    step: float
    score: float = 0.0
    strength: float = 1.0
    corners: List[int] = field(default_factory=list)
    band: bool = False  # the middle of a narrow band, rod or ray (rather than an edge)
    color: Optional[np.ndarray] = None  # linear colour its drones show (None: the picture's)
    shape: Optional[_shapes.Ellipse] = None  # the simple shape it was snapped to (drawn exactly, no corners)
    feature: bool = False  # the whole outline of a feature (an eye, a mouth): drawn right after the silhouettes
    count: int = 0  # drones round a ring (0: as many as fit)

    def like(self, xy, closed=False):
        """A stroke as this one along other points."""
        return Stroke(xy, closed, self.kind, self.part, self.step, self.score, self.strength, [], self.band,
                      self.color, feature=self.feature)

    @property
    def length(self) -> float:
        return self.step * (len(self.xy) if self.closed else len(self.xy) - 1)


class Picture:
    """The scene as the middle viewer sees it: the triangle and barycentrics every pixel shows."""

    def __init__(self, mesh, offsets, opt, h: float):
        tris = mesh.triangles
        self.mesh, self.tris = mesh, tris
        (self.eye, self.toward), = _outline._views(replace(opt, facade=True), tris)
        self.d, self.e1, self.e2 = view_basis(self.toward)
        self.far = float(np.linalg.norm(self.toward))
        # rendered at half the pixel and each pixel shows the nearest of its four sub-pixels that hits
        # the model: a rod or a painted line thinner than a pixel is not broken into dashes
        r = _outline._render(tris, self.eye, self.toward, 0.5 * h)
        if r is None:
            raise ValueError("no part of the scene faces the audience")
        tri, bary, self.lo, self.hi = r["tri"], r["bary"], r["lo"], r["hi"]
        self.h = 2.0 * r["h"]
        H2, W2 = tri.shape
        tri = np.pad(tri, ((0, H2 % 2), (0, W2 % 2)), constant_values=-1)
        bary = np.pad(bary, ((0, H2 % 2), (0, W2 % 2), (0, 0)))
        hit = tri >= 0
        P = np.einsum("hwk,hwkd->hwd", bary, tris[np.where(hit, tri, 0)])
        if self.eye is None:
            near = np.where(hit, P @ self.d, -np.inf)
        else:
            near = np.where(hit, -np.linalg.norm(P - self.eye, axis=2), -np.inf)
        H, W = tri.shape[0] // 2, tri.shape[1] // 2
        sub = near.reshape(H, 2, W, 2).transpose(0, 2, 1, 3).reshape(H, W, 4)
        pick = np.argmax(sub, axis=2)
        ii, jj = np.mgrid[0:H, 0:W]
        rr, cc = 2 * ii + pick // 2, 2 * jj + pick % 2
        self.tri, self.bary, self.P, self.nearness = tri[rr, cc], bary[rr, cc], P[rr, cc], near[rr, cc]
        self.shape = self.tri.shape
        self.fg = self.tri >= 0
        self.part = np.full(self.shape, -1, np.int64)
        self.part[self.fg] = np.searchsorted(offsets, self.tri[self.fg], side="right") - 1
        self._corners = None

    def project(self, p):
        """Picture points (..., 2) of model points (..., 3)."""
        p = np.asarray(p, np.float64)
        if self.eye is None:
            return np.stack([p @ self.e1, p @ self.e2], -1)
        rel = p - self.eye
        z = np.maximum(-(rel @ self.d), 1e-12)
        return np.stack([(rel @ self.e1) * self.far / z, (rel @ self.e2) * self.far / z], -1)

    def to_px(self, xy):
        """(col, row) pixel coordinates of picture points."""
        return (xy[:, 0] - self.lo[0]) / self.h - 0.5, (self.hi[1] - xy[:, 1]) / self.h - 0.5

    def from_px(self, cr):
        """Picture points of (col, row) pixel coordinates (N, 2)."""
        return np.stack([self.lo[0] + (cr[:, 0] + 0.5) * self.h, self.hi[1] - (cr[:, 1] + 0.5) * self.h], 1)

    def pixel_xy(self, rows, cols):
        return np.stack([self.lo[0] + (cols + 0.5) * self.h, self.hi[1] - (rows + 0.5) * self.h], 1)

    def sample(self, img, xy, fill=0):
        """``img`` at the pixel nearest each picture point (``fill`` outside the picture)."""
        col, row = self.to_px(xy)
        c, r = np.rint(col).astype(np.int64), np.rint(row).astype(np.int64)
        H, W = self.shape
        ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        out = np.full(len(xy), fill, dtype=img.dtype)
        out[ok] = img[r[ok], c[ok]]
        return out

    def shows(self, points, tri, normals):
        """True for the model points (on triangles ``tri``) the middle viewer sees: they face it and
        nothing stands in front of them in the picture."""
        xy = self.project(points)
        if self.eye is None:
            near, facing = points @ self.d, normals[tri] @ self.d
        else:
            to_eye = self.eye - points
            near = -np.linalg.norm(to_eye, axis=1)
            facing = np.einsum("ij,ij->i", normals[tri], to_eye)
        front = self.sample(self.nearness, xy, -np.inf)
        return (facing > 0) & np.isfinite(front) & (near >= front - 2.0 * self.h)

    def lift(self, xy, part, reach: float, inside_only: bool = False):
        """The surface points of ``part`` (one per point) under picture points ``xy``: the triangle of a
        pixel within ``reach`` pixels whose picture holds the point (else the nearest one's, clamped).

        Returns (pos, tri, bary, found)."""
        n = len(xy)
        if self._corners is None:
            self._corners = self.project(self.tris)
        col, row = self.to_px(xy)
        ci, ri = np.rint(col).astype(np.int64), np.rint(row).astype(np.int64)
        H, W = self.shape
        best = np.full(n, -np.inf)
        bt = np.full(n, -1, np.int64)
        bl = np.zeros((n, 3))
        r = int(math.ceil(reach))
        offs = sorted(((dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)
                       if dy * dy + dx * dx <= reach * reach + 1e-9), key=lambda o: o[0] ** 2 + o[1] ** 2)
        part = np.broadcast_to(np.asarray(part, np.int64), (n,))
        for dy, dx in offs:
            rr, cc = ri + dy, ci + dx
            ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W) & (best < 50.0)
            i = np.flatnonzero(ok)
            if not len(i):
                continue
            i = i[self.part[rr[i], cc[i]] == part[i]]
            if not len(i):
                continue
            t = self.tri[rr[i], cc[i]]
            lam = _bary2(xy[i], self._corners[t])
            m = lam.min(axis=1)
            dist = np.hypot(cc[i] - col[i], rr[i] - row[i])
            score = np.where(m >= -1e-9, 100.0 - dist, -dist)
            better = score > best[i]
            i, t, lam, score = i[better], t[better], lam[better], score[better]
            best[i], bt[i], bl[i] = score, t, lam
        found = bt >= 0
        if inside_only:
            found &= best >= 50.0
        t = np.where(found, bt, 0)
        lam = np.clip(bl, 0.0, None)
        lam = lam / np.maximum(lam.sum(axis=1, keepdims=True), 1e-300)
        if self.eye is not None:  # back from the picture to the triangle
            z = -((self.tris[t] - self.eye) @ self.d)
            lam = lam / np.maximum(z, 1e-12)
            lam /= np.maximum(lam.sum(axis=1, keepdims=True), 1e-300)
        pos = np.einsum("ij,ijk->ik", lam, self.tris[t])
        return pos[found], t[found], lam[found], found


def _bary2(p, tri2):
    """Barycentrics of 2-D points ``p`` (N, 2) in triangles ``tri2`` (N, 3, 2); -1 for a degenerate one."""
    a, b, c = tri2[:, 0], tri2[:, 1], tri2[:, 2]
    v0, v1, v2 = b - a, c - a, p - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    good = np.abs(den) > 1e-18
    den = np.where(good, den, 1.0)
    l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
    l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
    lam = np.stack([1.0 - l1 - l2, l1, l2], 1)
    lam[~good] = -1.0
    return lam


# ---------------------------------------------------------------- lines as polylines

def _arc(p, closed):
    q = np.vstack([p, p[:1]]) if closed else p
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]


def resample(p, step: float, closed: bool):
    """The polyline ``p`` resampled every ``step`` (or a hair less) along its length."""
    t = _arc(p, closed)
    q = np.vstack([p, p[:1]]) if closed else p
    total = t[-1]
    if total <= 0:
        return p[:1].copy()
    n = max(int(math.ceil(total / step)), 1)
    s = np.arange(n) * (total / n) if closed else np.linspace(0.0, total, n + 1)
    return np.stack([np.interp(s, t, q[:, i]) for i in range(p.shape[1])], 1)


def smooth(p, sigma: float, closed: bool):
    """Gaussian smoothing along the polyline over ``sigma`` samples; an open one keeps its ends."""
    n = len(p)
    if sigma < 0.3 or n < 3:
        return p
    r = int(math.ceil(3.0 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    i = np.arange(-r, n + r)
    if closed:
        ext = p[np.mod(i, n)]
    else:
        low, high = i < 0, i > n - 1
        j = np.clip(np.where(low, -i, np.where(high, 2 * (n - 1) - i, i)), 0, n - 1)
        ext = p[j].copy()
        ext[low] = 2.0 * p[0] - ext[low]
        ext[high] = 2.0 * p[-1] - ext[high]
    return np.stack([np.convolve(ext[:, c], k, mode="valid") for c in range(p.shape[1])], 1)


def _normals(p, closed):
    """Unit normals (left of the walking direction) of the polyline."""
    n = len(p)
    if n < 2:
        return np.zeros((n, 2))
    if closed:
        t = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    else:
        t = np.gradient(p, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-300)
    return np.stack([-t[:, 1], t[:, 0]], 1)


def _turn(p, closed, w):
    """Angle (radians) the polyline turns at each point between the chords ``w`` samples back and ahead."""
    n = len(p)
    i = np.arange(n)
    if closed:
        a, b = p - p[(i - w) % n], p[(i + w) % n] - p
    else:
        a, b = p - p[np.maximum(i - w, 0)], p[np.minimum(i + w, n - 1)] - p
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    cos = np.einsum("ij,ij->i", a, b) / np.maximum(na * nb, 1e-300)
    out = np.arccos(np.clip(cos, -1.0, 1.0))
    out[(na <= 0) | (nb <= 0)] = 0.0
    return out


def _runs(keep, closed):
    """(start, stop) index ranges of the True runs of ``keep``; on a loop a run may wrap (stop > n)."""
    n = len(keep)
    if keep.all():
        return [(0, n)]
    if not keep.any():
        return []
    if closed:
        shift = int(np.flatnonzero(~keep)[0])
        k = np.roll(keep, -shift)
    else:
        shift, k = 0, keep
    d = np.diff(np.r_[0, k.astype(np.int8), 0])
    starts, stops = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(int(a) + shift, int(b) + shift) for a, b in zip(starts, stops)]


def _take(p, a, b):
    n = len(p)
    return p[np.arange(a, b) % n]


def _pad3(xy):
    return np.c_[xy, np.zeros(len(xy))]


def _near(q, pts, radius):
    return _outline._near(_pad3(q), _pad3(pts), radius)


# ---------------------------------------------------------------- reading the picture

def contours(F, level: float = 0.5):
    """Closed lines where ``F`` crosses ``level`` (marching squares; outside the array counts as
    below): a list of (n, 2) arrays of (col, row) pixel coordinates, the first point not repeated."""
    F = np.pad(np.asarray(F, np.float64), 1, constant_values=min(0.0, level - 1.0))
    b = F > level
    H, W = F.shape
    hz = b[:, :-1] != b[:, 1:]
    vt = b[:-1, :] != b[1:, :]
    nh = H * (W - 1)
    pts = np.zeros((nh + (H - 1) * W, 2))
    r, c = np.nonzero(hz)
    a0, a1 = F[r, c], F[r, c + 1]
    pts[r * (W - 1) + c] = np.stack([c + (level - a0) / (a1 - a0), r], 1)
    r, c = np.nonzero(vt)
    a0, a1 = F[r, c], F[r + 1, c]
    pts[nh + r * W + c] = np.stack([c, r + (level - a0) / (a1 - a0)], 1)
    code = b[:-1, :-1] * 1 + b[:-1, 1:] * 2 + b[1:, 1:] * 4 + b[1:, :-1] * 8
    rr, cc = np.nonzero((code > 0) & (code < 15))
    if not len(rr):
        return []
    code = code[rr, cc]
    E = np.stack([rr * (W - 1) + cc, nh + rr * W + cc + 1, (rr + 1) * (W - 1) + cc, nh + rr * W + cc], 1)
    cross = np.stack([hz[rr, cc], vt[rr, cc + 1], hz[rr + 1, cc], vt[rr, cc]], 1)  # top, right, bottom, left
    two = cross.sum(axis=1) == 2
    sel = E[two][cross[two]].reshape(-1, 2)
    k0, k1 = [sel[:, 0]], [sel[:, 1]]
    four = np.flatnonzero(~two)
    if len(four):
        centre = 0.25 * (F[rr, cc] + F[rr, cc + 1] + F[rr + 1, cc] + F[rr + 1, cc + 1])[four] > level
        cut_b = (code[four] == 5) == centre  # True: (top, right) and (bottom, left), else (top, left), (right, bottom)
        e = E[four]
        k0 += [e[:, 0], np.where(cut_b, e[:, 2], e[:, 1])]
        k1 += [np.where(cut_b, e[:, 1], e[:, 3]), np.where(cut_b, e[:, 3], e[:, 2])]
    k0, k1 = np.concatenate(k0), np.concatenate(k1)
    p0, p1 = _pad3(pts[k0]), _pad3(pts[k1])
    walks, _ = _outline._chains(k0, k1, p0, p1, np.zeros(2 * len(k0)))
    out = []
    for e, fw, closed, _, _ in walks:
        nodes = np.where(fw, k0[e], k1[e])
        out.append(pts[nodes] - 1.0)
    return out


def _region_lines(pic: Picture, F, s: float, owner, kind: str, part: int, strength=None, band=None,
                  collapse: bool = True, color=None, jump: bool = False):
    """Lines round the region where ``F`` (a smoothed mask) exceeds 1/2, at line spacing ``s``:
    (strokes, islands as picture points). ``owner(xy, inward normals)`` says which points this
    region draws. A long stretch narrower than ``THIN`` is drawn along its middle, a line coming
    back to itself (both sides of such a stretch) once. ``strength`` (a picture) scores the points:
    stretches below half the threshold are no line; a narrow stretch scores ``band`` (its contrast:
    blurred, a thin painted line looks weaker than it is)."""
    px = s / pic.h
    step = STEP * s
    h = pic.h
    strokes, islands = [], []
    for loop in contours(F):
        xy = pic.from_px(loop)
        across = float(np.linalg.norm(xy.max(axis=0) - xy.min(axis=0)))
        if across < SPECK * s:
            continue
        if across < DOT * s:
            if _bilinear(F, *pic.to_px(xy.mean(axis=0)[None]))[0] > 0.5:
                islands.append((xy.mean(axis=0), _shapes.fit_ellipse(xy)))
            continue
        xy = smooth(resample(xy, step, True), LIGHT / STEP, True)
        n = _normals(xy, True)
        grow = _bilinear(F, *pic.to_px(xy + n * h)) - _bilinear(F, *pic.to_px(xy - n * h))
        n_in = n * np.where(grow >= 0, 1.0, -1.0)[:, None]
        mine = owner(xy, n_in)
        power = np.full(len(xy), 10.0) if strength is None else pic.sample(strength, xy, 0.0)
        # how wide the region is behind each point; a long narrow stretch is drawn along its middle
        width = np.full(len(xy), np.inf)
        for j in range(1, int(math.ceil(2.0 * THIN * px)) + 2):
            dist = 0.5 * j * h
            out = (_bilinear(F, *pic.to_px(xy + n_in * dist)) <= 0.5) & np.isinf(width)
            width[out] = dist
        thin = width < THIN * s
        moved = np.zeros(len(xy), bool)
        for a, b in _runs(thin, True) if collapse else []:
            if (b - a) * step >= THIN_RUN * s:
                moved[np.arange(a, b) % len(xy)] = True
                i = np.arange(a, b) % len(xy)
                xy[i] += n_in[i] * 0.5 * width[i, None]
                if band is not None:
                    power[i] = np.maximum(power[i], band)
        mine &= power >= 0.5
        if moved.any():
            xy = smooth(xy, LIGHT / STEP, True)
        start = int(np.argmax(np.where(mine, width, -1.0)))  # the loop starts where it is widest
        xy, mine, power = np.roll(xy, -start, axis=0), np.roll(mine, -start), np.roll(power, -start)
        moved = np.roll(moved, -start)
        m = len(xy)
        qi, pj, _ = Grid(_pad3(xy), SELF_GAP * s).query_radius(_pad3(xy), SELF_GAP * s)
        gap = np.abs(qi - pj)
        gap = np.minimum(gap, m - gap) * step
        later = (gap > 3.0 * SELF_GAP * s) & (pj > qi) & mine[qi] & mine[pj]
        again = np.zeros(m, bool)
        again[pj[later]] = True
        keep = mine & ~again
        if jump and not keep.all():  # a short gap (the base of a rod drawn on its own) is bridged
            for a, b in _runs(~keep, True):
                p0, p1 = xy[(a - 1) % m], xy[b % m]
                if b - a < m - 2 and np.linalg.norm(p1 - p0) < JUMP * s:
                    i = np.arange(a, b) % m
                    f = ((np.arange(b - a) + 1.0) / (b - a + 1.0))[:, None]
                    xy[i] = p0 * (1.0 - f) + p1 * f
                    keep[i] = True
        whole = bool(keep.all())
        for a, b in _runs(keep, True):
            if (b - a) * step < MIN_RUN * s:
                continue
            i = np.arange(a, b) % m
            if jump:
                fine = resample(xy[i], step, whole)
                i = np.clip(np.rint(np.linspace(0, len(i) - 1, len(fine))).astype(np.int64), 0, len(i) - 1) + a
                i = i % m
                xy_run = fine
            else:
                xy_run = xy[i]
            strokes.append(Stroke(xy_run, whole, kind, part, step, strength=float(np.mean(np.minimum(power[i], 4.0))),
                                  band=bool(moved[i].mean() >= 0.5 and (band is None or band >= CRISP)),
                                  color=color))
    return strokes, islands


def _mask_field(mask, s, h, sigma=MASK_SIGMA):
    F = mask.astype(np.float64)
    if sigma * s / h >= 0.3:
        F = _outline._blur(F, sigma * s / h)
    return F


def silhouettes(pic: Picture, k: int, s: float):
    """The silhouette lines of part ``k`` at line spacing ``s``: (strokes, dots). Where the part borders
    a nearer one, that one draws the line. A long stretch narrower than ``THIN`` line spacings (a rod,
    a ray, a comb of them) is drawn by one line along its middle, its skeleton; a short one (the tip
    of a corner) keeps its outline."""
    mask = pic.part == k
    if not mask.any():
        return [], []
    h = pic.h
    mask = _shrink(_grow(mask, 1), 1) & (pic.fg | ~_grow(pic.fg, 1) | mask)  # a rod thinner than a pixel is whole
    F = _mask_field(mask, s, h)
    solid = F > 0.5
    r = max(int(round(0.5 * THIN * s / h)), 1)
    opened = _grow(_shrink(solid, r), r) & solid
    opened &= ~_small(opened, BLOB * s, h)  # a knot where rods meet is part of them
    thin = solid & ~_grow(opened, 1)
    rods = np.zeros(pic.shape, bool)
    rod_strokes = []
    if thin.any():
        rows, cols = np.nonzero(thin)
        i, j = _outline._pixel_graph(rows, cols)
        comp = _outline._components(len(rows), i, j)
        skel = _outline._thin(thin)[rows, cols]
        length = np.bincount(comp, weights=skel.astype(np.float64), minlength=len(rows)) * h
        rod = length[comp] >= ROD * s
        rods[rows[rod], cols[rod]] = True
        keep = rod & skel
        for xy, closed in _chain_pixels(pic, rows[keep], cols[keep], s):
            xy = smooth(resample(xy, STEP * s, closed), LIGHT / STEP, closed)
            st = Stroke(xy, closed, "silhouette", k, STEP * s, strength=10.0, band=True)
            if st.length >= MIN_RUN * s:
                rod_strokes.append(st)
    near_rod = _grow(rods & ~_grow(opened, 3), 2)  # not the sliver where a rod meets the body

    def owner(xy, n_in):  # a nearer part right outside draws the line, a rod its middle
        out_part = pic.sample(pic.part, xy - n_in * 1.5 * h, -1)
        out_near = pic.sample(pic.nearness, xy - n_in * 1.5 * h, -np.inf)
        in_near = pic.sample(pic.nearness, xy + n_in * 1.5 * h, -np.inf)
        return ~((out_part >= 0) & (out_part != k) & (out_near > in_near)) & \
            ~pic.sample(near_rod, xy + n_in * 1.5 * h, False)

    strokes, islands = _region_lines(pic, F, s, owner, "silhouette", k, collapse=False, jump=True)
    strokes.sort(key=lambda st: -st.length)
    rod_strokes.sort(key=lambda st: -st.length)
    return strokes + rod_strokes, [(p, math.inf, k, e) for p, e in islands]


def _small(m, size: float, h: float):
    """The connected pieces of mask ``m`` less than ``size`` across."""
    rows, cols = np.nonzero(m)
    out = np.zeros(m.shape, bool)
    if not len(rows):
        return out
    i, j = _outline._pixel_graph(rows, cols)
    comp = _outline._components(len(rows), i, j)
    n = len(rows)
    lo_r = np.full(n, np.inf)
    hi_r = np.full(n, -np.inf)
    lo_c, hi_c = lo_r.copy(), hi_r.copy()
    np.minimum.at(lo_r, comp, rows)
    np.maximum.at(hi_r, comp, rows)
    np.minimum.at(lo_c, comp, cols)
    np.maximum.at(hi_c, comp, cols)
    across = np.hypot(hi_r - lo_r, hi_c - lo_c) * h
    small = across[comp] < size
    out[rows[small], cols[small]] = True
    return out


def _shrink(m, r):
    """Binary erosion by about a disc of ``r`` pixels (crosses and squares in turn)."""
    for n in range(r):
        out = m & _outline._shift(m, 1, 0, False) & _outline._shift(m, -1, 0, False) & \
            _outline._shift(m, 0, 1, False) & _outline._shift(m, 0, -1, False)
        if n % 2:
            out &= _outline._shift(m, 1, 1, False) & _outline._shift(m, -1, -1, False) & \
                _outline._shift(m, 1, -1, False) & _outline._shift(m, -1, 1, False)
        m = out
    return m


def _grow(m, r):
    """Binary dilation by about a disc of ``r`` pixels (crosses and squares in turn)."""
    for n in range(r):
        out = m | _outline._shift(m, 1, 0, False) | _outline._shift(m, -1, 0, False) |             _outline._shift(m, 0, 1, False) | _outline._shift(m, 0, -1, False)
        if n % 2:
            out |= _outline._shift(m, 1, 1, False) | _outline._shift(m, -1, -1, False) |                 _outline._shift(m, 1, -1, False) | _outline._shift(m, -1, 1, False)
        m = out
    return m


def _chain_pixels(pic: Picture, rows, cols, s: float):
    """Line pixels chained into polylines (spurs shorter than half a line spacing gone, pieces that
    continue each other across a fork joined): [(picture points, closed)]."""
    if len(rows) < 2:
        return []
    i, j = _outline._pixel_graph(rows, cols)
    if not len(i):
        return []
    m = _pad3(pic.pixel_xy(rows, cols))
    short = _outline._unspurred(i, j, m[i], m[j], 0.5 * s)
    i, j = i[short], j[short]
    if not len(i):
        return []
    walks, _ = _outline._chains(i, j, m[i], m[j], np.zeros(2 * len(i)))
    return [(m[nodes, :2], closed) for nodes, closed in _merge(walks, i, j, m[:, :2], 0.5 * s)]


def color_regions(pic: Picture, lin, opt, s: float):
    """The picture cut into regions of one colour: k-means in OKLab (centres closer than the colour
    contrast merged, blends of two others dropped). Returns (labels (H, W), -1 off the model; centres)."""
    fg = pic.fg
    lab = _color.linear_to_oklab(lin[fg])
    rng = np.random.default_rng(0)
    sample = lab[rng.choice(len(lab), min(len(lab), 20000), replace=False)] if len(lab) else lab
    centres, count = _kmeans(sample, REGIONS, rng)
    while len(centres) > 1:  # colours that do not differ enough are one
        d = np.linalg.norm(centres[:, None] - centres[None], axis=2) + np.eye(len(centres)) * 1e9
        i, j = np.unravel_index(int(np.argmin(d)), d.shape)
        if d[i, j] >= opt.color_contrast:
            break
        w = count[i] + count[j]
        centres[i] = (centres[i] * count[i] + centres[j] * count[j]) / max(w, 1)
        count[i] = w
        centres, count = np.delete(centres, j, 0), np.delete(count, j)
    if len(centres) > 2:  # a colour between two others is their blend (pixels on the border between them)
        blend = np.zeros(len(centres), bool)
        for c in range(len(centres)):
            for a in range(len(centres)):
                for b in range(a + 1, len(centres)):
                    if c in (a, b) or blend[a] or blend[b]:
                        continue
                    ab = centres[b] - centres[a]
                    t = float(np.clip((centres[c] - centres[a]) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
                    if 0.1 < t < 0.9 and np.linalg.norm(centres[a] + t * ab - centres[c]) < 0.5 * opt.color_contrast \
                            and count[c] < min(count[a], count[b]):
                        blend[c] = True
        centres, count = centres[~blend], count[~blend]
    label = np.full(pic.shape, -1, np.int64)
    if not len(centres):
        return label, centres
    near = np.empty(len(lab), np.int64)
    for lo in range(0, len(lab), 1 << 18):  # in pieces: a big picture has a million pixels
        near[lo:lo + (1 << 18)] = np.argmin(np.linalg.norm(lab[lo:lo + (1 << 18), None] - centres[None], axis=2), axis=1)
    label[fg] = near
    if len(centres) > 1:  # specks of a colour smaller than the line scale go to their surroundings
        sig = COLOR_PIXELS
        votes = np.stack([_outline._blur((label == c).astype(np.float64), sig) for c in range(len(centres))])
        label[fg] = np.argmax(votes, axis=0)[fg]
    return label, centres


def color_lines(pic: Picture, k: int, regions, step_map, s: float, contrast: float, lin):
    """Candidate colour borders of part ``k``: the lines round its colour regions (a thin band or
    ray along its middle). Returns (strokes, islands as (point, score, part))."""
    label, centres = regions
    mine = pic.part == k
    h = pic.h
    strokes, dots = [], []
    reach = int(math.ceil(2.0 * THIN * s / h))
    apart = np.linalg.norm(centres[:, None] - centres[None], axis=2) + np.eye(len(centres)) * 1e9
    main = int(np.argmax(np.bincount(label[mine], minlength=len(centres)))) if mine.any() else 0
    for c in np.unique(label[mine]):
        region = mine & (label == c)
        if region.sum() * h * h < (SPECK * s) ** 2:
            continue

        def owner(xy, n_in, c=c):
            # the part's edge is its silhouette; a thin neighbour draws the border along its middle
            q = xy - n_in * 1.5 * h
            nb = pic.sample(label, q, -1)
            ok = (pic.sample(pic.part, q, -1) == k) & (nb != c) & (nb >= 0)
            theirs, own = np.full(len(xy), np.inf), np.full(len(xy), np.inf)
            for j in range(1, reach + 1):
                out = (pic.sample(label, q - n_in * 0.5 * j * h, -1) != nb) & np.isinf(theirs)
                theirs[out] = 0.5 * j * h
                out = (pic.sample(label, xy + n_in * 0.5 * j * h, -1) != c) & np.isinf(own)
                own[out] = 0.5 * j * h
            # both narrow (rays and the gaps between them): the colour that is not the main one draws
            defer = (theirs < THIN * s) & ((own >= THIN * s) | (c == main))
            if c == main:  # the figure's colour draws its border, not the colour around it
                defer |= nb != main
            return ok & ~defer

        band = float(apart[c].min()) / contrast if len(centres) > 1 else 0.0
        colour = lin[region].mean(axis=0)
        st, isl = _region_lines(pic, _mask_field(region, s, h, COLOR_PIXELS * h / s), s, owner, "colors", k,
                                step_map, band, color=colour)
        far = float(np.linalg.norm(centres[c] - centres[main])) / contrast
        for x in st:  # a border is drawn in the colour of the side that stands out from the main one
            x.strength *= 1.0 + 0.25 * min(far, 2.0)
        strokes += st
        for p, e in isl:
            v = max(float(pic.sample(step_map, p[None], 0.0)[0]), band)
            if v >= DOT_STRENGTH:
                dots.append((p, 2.0 * v, k, e))
    return strokes, dots


def _pieces(mask):
    """The connected pieces of ``mask``: (rows, cols) of each."""
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return []
    i, j = _outline._pixel_graph(rows, cols)
    label = _outline._components(len(rows), i, j)
    order = np.argsort(label, kind="stable")
    cuts = np.flatnonzero(np.diff(label[order])) + 1
    return [(rows[g], cols[g]) for g in np.split(order, cuts)]


def feature_loops(pic: Picture, k: int, regions, lin, s: float, contrast: float, depth: bool = True):
    """The features of part ``k`` a designer outlines first: whole regions inside it that stand out
    from what is round them, by colour (a dark carved eye on an orange pumpkin, a gold sun on a blue
    flag) or, with ``depth``, by lying deeper than the surface round them (a carved eye on a grey
    model). Each is drawn by its whole outline, a closed line (its corners are marked later). A region
    narrower than ``THIN`` is a band (drawn along its middle elsewhere), one touching the outline of
    the part is not inside it. Returns strokes, the most outstanding first."""
    mine = pic.part == k
    if not mine.any():
        return []
    h = pic.h
    edge = mine & ~_shrink(mine, max(int(round(FEATURE_EDGE * s / h)), 1))
    ring_r = max(int(round(FEATURE_RING * s / h)), 1)
    thin_r = max(int(round(0.5 * THIN * s / h)), 1)
    least = (FEATURE_SIZE * s / h) ** 2
    candidates = []  # (score, rows, cols)
    if regions is not None and len(regions[1]) > 1:
        label = regions[0]
        lab = _color.linear_to_oklab(lin)
        for c in np.unique(label[mine]):
            for rows, cols in _pieces(mine & (label == c)):
                if len(rows) < least or edge[rows, cols].mean() > FEATURE_TOUCH:
                    continue
                v = _standout(mine, rows, cols, ring_r, thin_r, lambda r, c_: lab[r, c_])
                if v is not None and v / contrast >= FEATURE_CONTRAST:
                    candidates.append((v / contrast, rows, cols))
    if depth:  # deeper than the surface round it, against the surface smoothed over a few line spacings
        near = np.where(mine, pic.nearness, 0.0)
        f = mine.astype(np.float64)
        sg = FEATURE_SURFACE * s / h
        surface = _outline._blur(near, sg) / np.maximum(_outline._blur(f, sg), 1e-300)
        deep = mine & (surface - near > FEATURE_DEPTH * s)  # nearness grows towards the audience
        for rows, cols in _pieces(_shrink(_grow(deep, 1), 1)):
            if len(rows) < least or edge[rows, cols].mean() > FEATURE_TOUCH:
                continue
            if _shrink_fits(rows, cols, thin_r, pic.shape):
                v = float((surface[rows, cols] - near[rows, cols]).mean()) / (FEATURE_DEPTH * s)
                candidates.append((FEATURE_CONTRAST * min(v, 3.0), rows, cols))
    candidates.sort(key=lambda x: -(x[0] * len(x[1])))
    taken = np.zeros(pic.shape, bool)
    power = np.zeros(pic.shape)
    for v, rows, cols in candidates[:4 * MAX_FEATURES]:
        if taken[rows, cols].mean() > 0.5:  # the same feature found by colour and by depth
            continue
        taken[rows, cols] = True
        power[rows, cols] = np.maximum(power[rows, cols], v)
    # touching regions (a cavity of two shades, an eye and its rim) are one feature: their outline is drawn
    out = []
    pieces = [(float(power[r, c].max()), r, c) for r, c in _pieces(_shrink(_grow(taken, 1), 1))]
    for v, rows, cols in sorted(pieces, key=lambda x: -(x[0] * len(x[1]))):
        if len(rows) < least:
            continue
        r0, r1, c0, c1 = rows.min() - 3, rows.max() + 4, cols.min() - 3, cols.max() + 4
        box = np.zeros((r1 - r0, c1 - c0))
        box[rows - r0, cols - c0] = 1.0
        if MASK_SIGMA * s / h >= 0.3:
            box = _outline._blur(box, MASK_SIGMA * s / h)
        loops = [pic.from_px(loop + np.array([c0, r0])) for loop in contours(box)]
        loops = [xy for xy in loops if len(xy) >= 4 and _arc(xy, True)[-1] >= FEATURE_SIZE * s]
        for i, xy in enumerate(loops):
            # a hole in the region (an eye in a face) is a feature of its own, not this one's outline
            if any(j != i and len(other) > len(xy) and _inside(xy[:1], other)[0] for j, other in enumerate(loops)):
                continue
            xy = smooth(resample(xy, STEP * s, True), LIGHT / STEP, True)
            out.append(Stroke(xy, True, "colors", k, STEP * s, score=v * len(rows), strength=float(min(v, 4.0)),
                              feature=True))
        if len(out) >= MAX_FEATURES:
            break
    return out


def _standout(mine, rows, cols, ring_r: int, thin_r: int, value):
    """How much a region (pixels ``rows``, ``cols`` of the part ``mine``) differs from the ring of the
    part round it, or None when it is too narrow to be more than a band."""
    r0, r1 = max(rows.min() - ring_r - 1, 0), min(rows.max() + ring_r + 2, mine.shape[0])
    c0, c1 = max(cols.min() - ring_r - 1, 0), min(cols.max() + ring_r + 2, mine.shape[1])
    box = np.zeros((r1 - r0, c1 - c0), bool)
    box[rows - r0, cols - c0] = True
    if not _shrink(box, thin_r).any():
        return None
    ring = _grow(box, ring_r) & ~box & mine[r0:r1, c0:c1]
    if not ring.any():
        return None
    rr, cc = np.nonzero(ring)
    inside = value(rows, cols).mean(axis=0)
    around = value(rr + r0, cc + c0).mean(axis=0)
    return float(np.linalg.norm(np.atleast_1d(inside - around)))


def _shrink_fits(rows, cols, r: int, shape) -> bool:
    """True when the region is wider than ``2 r`` pixels somewhere."""
    r0, c0 = rows.min() - 1, cols.min() - 1
    box = np.zeros((rows.max() - r0 + 2, cols.max() - c0 + 2), bool)
    box[rows - r0, cols - c0] = True
    return bool(_shrink(box, r).any())


def _inside(points, loop):
    """True for the points (N, 2) inside the closed polyline ``loop`` (even-odd rule)."""
    x, y = points[:, 0][:, None], points[:, 1][:, None]
    ax, ay = loop[:, 0][None], loop[:, 1][None]
    bx, by = np.roll(loop[:, 0], -1)[None], np.roll(loop[:, 1], -1)[None]
    cross = ((ay > y) != (by > y)) & (x < (bx - ax) * (y - ay) / np.where(by != ay, by - ay, 1e-300) + ax)
    return cross.sum(axis=1) % 2 == 1


def _kmeans(x, k, rng, iterations=25):
    """k-means (k-means++ start) of the rows of ``x``: (centres, count of each)."""
    n = len(x)
    if not n:
        return np.zeros((0, x.shape[1])), np.zeros(0)
    centres = [x[int(rng.integers(n))]]
    d2 = ((x - centres[0]) ** 2).sum(axis=1)
    for _ in range(1, min(k, n)):
        total = float(d2.sum())
        if total <= 1e-12:
            break
        centres.append(x[int(rng.choice(n, p=d2 / total))])
        d2 = np.minimum(d2, ((x - centres[-1]) ** 2).sum(axis=1))
    c = np.array(centres)
    for _ in range(iterations):
        lab = np.argmin(((x[:, None] - c[None]) ** 2).sum(axis=2), axis=1)
        new = np.array([x[lab == i].mean(axis=0) if (lab == i).any() else c[i] for i in range(len(c))])
        if np.abs(new - c).max() < 1e-6:
            break
        c = new
    lab = np.argmin(((x[:, None] - c[None]) ** 2).sum(axis=2), axis=1)
    count = np.bincount(lab, minlength=len(c)).astype(np.float64)
    return c[count > 0], count[count > 0]


def _bilinear(img, col, row):
    H, W = img.shape
    col = np.clip(col, 0.0, W - 1.0)
    row = np.clip(row, 0.0, H - 1.0)
    c0 = np.minimum(np.floor(col).astype(np.int64), W - 2) if W > 1 else np.zeros(len(col), np.int64)
    r0 = np.minimum(np.floor(row).astype(np.int64), H - 2) if H > 1 else np.zeros(len(row), np.int64)
    fc, fr = col - c0, row - r0
    c1, r1 = np.minimum(c0 + 1, W - 1), np.minimum(r0 + 1, H - 1)
    return ((1 - fr) * ((1 - fc) * img[r0, c0] + fc * img[r0, c1]) + fr * ((1 - fc) * img[r1, c0] + fc * img[r1, c1]))


def fields(pic: Picture, mesh, normals, opt, s: float, kinds, octaves: int = 1, eyes: float = 0.0):
    """Line pixels of the inner lines and their strength (contrast over its threshold, the geometric
    mean over two scales): {kind: (line mask, strength)}; "bumps" looked for at line spacing ``eyes``
    (an eye is as big as it is, however sparse the drawing)."""
    out = {}
    fg, P, h = pic.fg, pic.P, pic.h
    t = np.where(fg, pic.tri, 0)
    N = normals[t]
    if pic.eye is None:
        facing = N @ pic.d
    else:
        to_eye = pic.eye - P
        facing = np.einsum("hwd,hwd->hw", N, to_eye) / np.maximum(np.linalg.norm(to_eye, axis=2), 1e-300)
    px = s / h
    sig = max(FIELD_SIGMA * px, 1.0)
    cover = _outline._blur(fg.astype(np.float64), sig)
    solid = fg & (cover > 0.5)
    inner = solid & (facing > math.cos(math.radians(opt.facade_angle)))
    if "relief" in kinds:
        angle = math.radians(opt.relief_angle)
        maps = []
        for sg in (sig, 2.0 * sig):
            c = _outline._blur(fg.astype(np.float64), sg)
            sm = _outline._blur(np.where(fg[..., None], N, 0.0), sg) / np.maximum(c[..., None], 1e-300)
            sm /= np.maximum(np.linalg.norm(sm, axis=2, keepdims=True), 1e-300)
            maps.append(_outline._change(sm, P, facing, solid, h) * (4.0 * sg * h))
        line, _, _ = _outline._crest(maps[0], solid, 0.5 * angle, angle)
        out["relief"] = (_outline._thin(line & inner), np.sqrt(maps[0] * maps[1]) / angle)
    if "colors" in kinds:
        lin = np.ones(fg.shape + (3,))
        lin[fg] = _color.surface_colors(mesh, pic.tri[fg], pic.bary[fg])
        lab = _color.linear_to_oklab(lin)
        maps = []
        for sg in (sig, 2.0 * sig):
            c = _outline._blur(fg.astype(np.float64), sg)
            sm = _outline._blur(np.where(fg[..., None], lab, 0.0), sg) / np.maximum(c[..., None], 1e-300)
            maps.append(_outline._change(sm, P, facing, fg, h) * (sg * h * math.sqrt(2.0 * math.pi)))
        step_map = np.sqrt(maps[0] * maps[1]) / opt.color_contrast
        for _ in range(2):  # the border of a region may lie a pixel or two off the crest
            step_map = np.maximum.reduce([_outline._shift(step_map, dy, dx)
                                          for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
        out["colors"] = (None, step_map)
    if "spots" in kinds:
        out["spots"] = _spots(pic, mesh, fg, sig, octaves)
    if "bumps" in kinds:
        out["bumps"] = _bumps(pic, eyes or s)
    if "occlusion" in kinds:
        H, W = fg.shape
        line = np.zeros(fg.shape, bool)
        for dy, dx in _outline._NEIGHBOURS:
            a, b = _outline._pairs(H, W, dy, dx)
            same = solid[a] & solid[b] & (pic.part[a] == pic.part[b])
            jump = same & _outline._apart(P, facing, h, a, b, dy, dx)[1]
            front = pic.nearness[a] > pic.nearness[b]
            line[a] |= jump & front
            line[b] |= jump & ~front
        out["occlusion"] = (_outline._thin(line), np.full(fg.shape, 2.0))
    return out


def _spots(pic, mesh, fg, sig, octaves: int = 1):
    """Small dark or bright spots (eyes, a nose): (spot mask at their centres, contrast over ``SPOT``,
    (lightness, the lightness round each spot, its sign, its size in pixels))). A difference of Gaussians of
    the lightness, at its extremes, where it falls off all round (no edge); with more ``octaves`` also at
    twice and four times the size, in the picture shrunk to match (eyes bigger than a line spacing)."""
    lin = np.ones(fg.shape + (3,))
    lin[fg] = _color.surface_colors(mesh, pic.tri[fg], pic.bary[fg])
    light = _color.linear_to_oklab(lin)[..., 0]
    H, W = fg.shape
    found = []  # (strength, row, col, surroundings, sign, size)
    for o in range(octaves):
        k = 2 ** o
        h2, w2 = H // k, W // k
        if min(h2, w2) < 8:
            break
        f = fg[:h2 * k, :w2 * k].reshape(h2, k, w2, k).mean(axis=(1, 3))
        lk = np.where(fg, light, 0.0)[:h2 * k, :w2 * k].reshape(h2, k, w2, k).mean(axis=(1, 3))
        on = f > 0.5
        lk = np.where(on, lk / np.maximum(f, 1e-300), 0.0)
        fo = on.astype(np.float64)

        def mean(sg):
            return _outline._blur(np.where(on, lk, 0.0), sg) / np.maximum(_outline._blur(fo, sg), 1e-300)

        around = mean(3.0 * sig)
        dog = mean(sig) - around
        a = np.abs(dog)
        peak = on & (a >= SPOT)
        r = max(int(round(1.5 * sig)), 1)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy or dx:
                    peak &= a >= _outline._shift(a, dy, dx, 0.0)
        # a blob falls off in every direction; along an edge it does not
        far = max(int(round(2.0 * sig)), 2)
        for dy, dx in ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)):
            peak &= _outline._shift(dog, dy * far, dx * far, 0.0) * np.sign(dog) < 0.5 * a
        peak &= _outline._blur(fo, 2.0 * sig) > 0.95  # whole all round: not at the silhouette
        for rr, cc in zip(*np.nonzero(peak)):
            found.append((a[rr, cc] / SPOT, min(rr * k + k // 2, H - 1), min(cc * k + k // 2, W - 1), around[rr, cc],
                          np.sign(dog[rr, cc]), sig * k))
    found.sort(key=lambda x: -x[0])
    kept = []
    for x in found:  # one spot found at two sizes is the stronger one
        if all(math.hypot(x[1] - y[1], x[2] - y[2]) >= 1.5 * max(x[5], y[5]) for y in kept):
            kept.append(x)
    peak = np.zeros((H, W), bool)
    spot, around, sign, size = (np.zeros((H, W)) for _ in range(4))
    for v, rr, cc, sur, sg_, sz in kept:
        peak[rr, cc] = True
        spot[rr, cc], around[rr, cc], sign[rr, cc], size[rr, cc] = v, sur, sg_, sz
    return peak, spot, (light, around, sign, size)


def _bumps(pic: Picture, s: float):
    """Pairs of small bumps side by side, about level and alike: eyeballs sculpted in their sockets, which
    no colour tells from the face. [(picture point, strength over ``BUMP``, part, ellipse)]."""
    fg = pic.fg
    fo = fg.astype(np.float64)
    near = np.where(fg, pic.nearness, 0.0)
    sig = max(BUMP_SIGMA * s / pic.h, 1.0)

    def mean(sg):
        return _outline._blur(near, sg) / np.maximum(_outline._blur(fo, sg), 1e-300)

    dog = np.where(fg, mean(sig) - mean(3.0 * sig), 0.0)
    peak = fg & (dog >= BUMP * s)
    r = max(int(round(1.5 * sig)), 1)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy or dx:
                peak &= dog >= _outline._shift(dog, dy, dx, -np.inf)
    far = max(int(round(2.0 * sig)), 2)
    for dy, dx in ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)):  # an eye is wide
        peak &= _outline._shift(dog, dy * far, dx * far, 0.0) < BUMP_FALL * dog
    peak &= _outline._blur(fo, 2.0 * sig) > 0.95  # whole all round: not at the silhouette
    H, W = fg.shape
    found = []
    R = int(math.ceil(4.0 * sig))
    for row, col in zip(*np.nonzero(peak)):
        r0, r1, c0, c1 = max(row - R, 0), min(row + R + 1, H), max(col - R, 0), min(col + R + 1, W)
        inside = (dog[r0:r1, c0:c1] >= 0.5 * dog[row, col]) & (pic.part[r0:r1, c0:c1] == pic.part[row, col])
        blob = np.zeros(inside.shape, bool)
        blob[row - r0, col - c0] = True
        for _ in range(2 * R):
            grown = _grow(blob, 1) & inside
            if grown.sum() == blob.sum():
                break
            blob = grown
        if blob[0].any() or blob[-1].any() or blob[:, 0].any() or blob[:, -1].any():
            continue  # it runs on: a ridge, not a bump
        rows, cols = np.nonzero(blob)
        e = _shapes.blob_ellipse(rows + r0, cols + c0, pic.h, pic.lo, pic.hi)
        if e is not None:
            found.append((e.center, float(dog[row, col] / (BUMP * s)), int(pic.part[row, col]), e))
    # a pair alike, about level and mirrored about the middle of the part at its height (the ears are not)
    pairs = []
    for i in range(len(found)):
        for j in range(i + 1, len(found)):
            a, b = found[i][3], found[j][3]
            if found[i][2] != found[j][2]:
                continue
            size, d = max(a.axes[0], b.axes[0]), b.center - a.center
            if (max(a.axes[0] / b.axes[0], b.axes[0] / a.axes[0]) >= PAIR_SIZE or abs(d[1]) >= PAIR_LEVEL * size
                    or not 2.0 * size <= abs(d[0]) < BUMP_APART * size):
                continue
            mid = 0.5 * (a.center + b.center)
            _, row = pic.to_px(mid[None])
            band = slice(max(int(row[0] - size / pic.h), 0), min(int(row[0] + size / pic.h) + 1, H))
            cols = np.nonzero((pic.part[band] == found[i][2]).any(axis=0))[0]
            if not len(cols):
                continue
            axis = pic.pixel_xy(np.zeros(2, np.int64), cols[[0, -1]])[:, 0].mean()
            off = abs(mid[0] - axis) / abs(d[0])
            if off <= BUMP_MIRROR:
                pairs.append((off + abs(d[1]) / abs(d[0]), i, j))
    out, used = [], set()
    for _, i, j in sorted(pairs):
        if i not in used and j not in used:
            used.update((i, j))
            out += [found[i], found[j]]
    return out


def _blob(pic: Picture, light, surround, sign, row: int, col: int, s: float, size: float = 0.0):
    """The ellipse of the spot whose centre is pixel (``row``, ``col``): the pixels round it darker
    (or lighter) than halfway to its surroundings, joined to it. None when they reach far (an edge,
    not a spot)."""
    H, W = light.shape
    r = int(math.ceil(max(1.5 * DOT * s / pic.h, 3.0 * size)))
    r0, r1, c0, c1 = max(row - r, 0), min(row + r + 1, H), max(col - r, 0), min(col + r + 1, W)
    mid = 0.5 * (light[row, col] + surround[row, col])
    inside = ((light[r0:r1, c0:c1] - mid) * sign > 0) & (pic.part[r0:r1, c0:c1] == pic.part[row, col])
    seed = np.zeros(inside.shape, bool)
    seed[row - r0, col - c0] = True
    if not inside[row - r0, col - c0]:
        return None
    blob = seed
    for _ in range(2 * r):
        grown = _grow(blob, 1) & inside
        if grown.sum() == blob.sum():
            break
        blob = grown
    if blob[0].any() or blob[-1].any() or blob[:, 0].any() or blob[:, -1].any():
        return None
    rows, cols = np.nonzero(blob)
    e = _shapes.blob_ellipse(rows + r0, cols + c0, pic.h, pic.lo, pic.hi)
    # only a round spot is an eye drawn as a ring: a triangle (a carved eye) keeps its own outline
    yy, xx = np.mgrid[0:blob.shape[0], 0:blob.shape[1]]
    xy = pic.pixel_xy((yy + r0).ravel(), (xx + c0).ravel())
    c, s_ = math.cos(e.angle), math.sin(e.angle)
    d = xy - e.center
    u, v = c * d[:, 0] + s_ * d[:, 1], -s_ * d[:, 0] + c * d[:, 1]
    disc = ((u / e.axes[0]) ** 2 + (v / e.axes[1]) ** 2 <= 1.0).reshape(blob.shape)
    if (disc & blob).sum() < ROUND_SPOT * (disc | blob).sum():
        return None
    return e


def inner_lines(pic: Picture, k: int, found, s: float, regions=None, contrast: float = 0.15, lin=None):
    """Candidate inner lines of part ``k`` (strokes scored by salience, best first) and dots
    (picture point, score, part)."""
    mine = pic.part == k
    step = STEP * s
    raw, dots = [], []
    if "spots" in found:
        peak, spot, (light, surround, sign, size) = found["spots"]
        rows, cols = np.nonzero(peak & mine)
        for p, v, r, c in zip(pic.pixel_xy(rows, cols), spot[rows, cols].tolist(), rows.tolist(), cols.tolist()):
            dots.append((p, 2.0 * v, k, _blob(pic, light, surround, sign[r, c], r, c, s, size[r, c])))
    if "colors" in found and regions is not None:
        st, dt = color_lines(pic, k, regions, found["colors"][1], s, contrast, lin)
        raw += st
        dots += dt
    for kind, field_ in found.items():
        if kind in ("spots", "colors", "bumps"):
            continue
        line, strength = field_
        rows, cols = np.nonzero(line & mine)
        if len(rows) < 2:
            continue
        i, j = _outline._pixel_graph(rows, cols)
        if not len(i):
            continue
        m = _pad3(pic.pixel_xy(rows, cols))
        short = _outline._unspurred(i, j, m[i], m[j], 0.5 * s)
        i, j = i[short], j[short]
        if not len(i):
            continue
        walks, _ = _outline._chains(i, j, m[i], m[j], np.zeros(2 * len(i)))
        power = strength[rows, cols]
        for nodes, closed in _merge(walks, i, j, m[:, :2], 0.5 * s):
            xy = m[nodes, :2]
            pw = float(np.mean(np.minimum(power[nodes], 4.0)))
            across = float(np.linalg.norm(xy.max(axis=0) - xy.min(axis=0)))
            if across < DOT * s:
                if pw >= DOT_STRENGTH and (closed or across >= 0.5 * s):
                    dots.append((xy.mean(axis=0), pw * WEIGHT[kind], k, _shapes.fit_ellipse(xy) if closed else None))
                continue
            xy = smooth(resample(xy, step, closed), LIGHT / STEP, closed)
            raw.append(Stroke(xy, closed, kind, k, step, strength=pw))
    raw = _bridge(raw, s)
    strokes = []
    for st in raw:
        if st.length < (1.0 if st.band else MIN_LENGTH[st.kind]) * s:
            continue
        # how much it turns per line spacing, along chords of half a spacing
        coarse = resample(st.xy, 0.5 * s, st.closed)
        wiggle = float(_turn(coarse, st.closed, 1).sum()) / max(st.length / s, 1e-9) if len(coarse) > 2 else 0.0
        straight = 1.0 / (1.0 + max(0.0, wiggle - WIGGLE))
        st.score = st.length / s * min(st.strength, 3.0) * WEIGHT[st.kind] * straight
        strokes.append(st)
    if len(dots) > SPOT_TEXTURE:  # many alike are a texture: only the outstanding ones count
        cut = 1.5 * float(np.median([d[1] for d in dots]))
        dots = [d for d in dots if d[1] >= cut]
    dots += [(xy, 2.0 * v, k, e) for xy, v, part, e in found.get("bumps", ()) if part == k]  # a pair: no texture
    strokes.sort(key=lambda x: -x.score)
    dots.sort(key=lambda x: -x[1])
    return strokes, dots


def _bridge(strokes: List[Stroke], s: float):
    """Open strokes of one kind that continue each other across a small gap joined into one: the
    nearest ends first, each end once, never a stroke to itself."""
    strokes = list(strokes)
    limit = math.cos(math.radians(35.0))
    who, at, out_dir = [], [], []  # stroke * 2 + end, the end point, its outward direction
    for i, st in enumerate(strokes):
        if st.closed or len(st.xy) < 3:
            continue
        back = max(1, min(len(st.xy) - 1, int(round(0.5 * s / st.step))))
        for end, (p, q) in enumerate(((st.xy[0], st.xy[back]), (st.xy[-1], st.xy[-1 - back]))):
            d = p - q
            who.append(2 * i + end)
            at.append(p)
            out_dir.append(d / max(float(np.linalg.norm(d)), 1e-300))
    if len(who) < 4:
        return strokes
    who, at, out_dir = np.array(who), np.array(at), np.array(out_dir)
    a, b, dist = Grid(_pad3(at), BRIDGE * s).query_radius(_pad3(at), BRIDGE * s, exclude_self=True)
    kinds = {k: n for n, k in enumerate(sorted({st.kind for st in strokes}))}
    kind = np.array([kinds[st.kind] for st in strokes])
    sa, sb = who[a] // 2, who[b] // 2
    ok = (a < b) & (sa != sb) & (kind[sa] == kind[sb])
    a, b, dist = a[ok], b[ok], dist[ok]
    g = (at[b] - at[a]) / np.maximum(dist, 1e-300)[:, None]
    straight = (np.einsum("ij,ij->i", out_dir[a], g) >= limit) & (-np.einsum("ij,ij->i", out_dir[b], g) >= limit)
    ok = (dist <= 0.25 * s) | straight
    a, b, dist = a[ok], b[ok], dist[ok]
    if not len(a):
        return strokes
    root = list(range(len(strokes)))

    def find(x):
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    link = {}
    for k in np.argsort(dist, kind="stable"):
        ea, eb = int(who[a[k]]), int(who[b[k]])
        if ea in link or eb in link:
            continue
        ra, rb = find(ea // 2), find(eb // 2)
        if ra == rb:
            continue
        link[ea], link[eb] = eb, ea
        root[ra] = rb
    if not link:
        return strokes
    out, joined, done = [], [], set()
    for i, st in enumerate(strokes):
        if 2 * i not in link and 2 * i + 1 not in link:
            out.append(st)
        elif i not in done and (2 * i not in link or 2 * i + 1 not in link):  # a chain from its free end
            enter = 0 if 2 * i not in link else 1
            chain, j = [], i
            while True:
                chain.append((j, enter))
                done.add(j)
                nxt = link.get(2 * j + 1 - enter)
                if nxt is None:
                    break
                j, enter = nxt // 2, nxt % 2
            first = strokes[chain[0][0]]
            xy = np.vstack([strokes[j].xy if e == 0 else strokes[j].xy[::-1] for j, e in chain])
            w = sum(strokes[j].length for j, _ in chain)
            strength = sum(strokes[j].strength * strokes[j].length for j, _ in chain) / max(w, 1e-300)
            joined.append(Stroke(resample(xy, first.step, False), False, first.kind, first.part, first.step,
                                 strength=strength))
    return out + joined


def _merge(walks, i, j, xy, reach):
    """Walks of a pixel graph joined into long lines where they continue each other straight across a
    fork. Returns [(node indices, closed)]."""
    seqs = []
    for e, fw, closed, _, _ in walks:
        nodes = np.r_[np.where(fw, i[e], j[e]), np.where(fw, j[e], i[e])[-1:]]
        if closed:
            nodes = nodes[:-1]
        seqs.append((nodes, closed))
    ends = {}  # node -> [(walk, which end, direction away from the node)]
    for w, (nodes, closed) in enumerate(seqs):
        if closed or len(nodes) < 2:
            continue
        p = xy[nodes]
        t = _arc(p, False)
        for end, (a, sign) in enumerate(((0, 1), (len(nodes) - 1, -1))):
            far = int(np.searchsorted(t, reach)) if sign > 0 else int(np.searchsorted(t, t[-1] - reach)) - 1
            far = min(max(far, 0), len(nodes) - 1)
            d = p[far] - p[a]
            norm = float(np.linalg.norm(d))
            if norm > 0:
                ends.setdefault(int(nodes[a]), []).append((w, end, d / norm))
    partner = {}
    limit = -math.cos(math.radians(45.0))
    for node, here in ends.items():
        if len(here) < 2:
            continue
        pairs = sorted((float(a[2] @ b[2]), x, y) for x, a in enumerate(here) for y, b in enumerate(here)
                       if x < y and a[0] != b[0])
        used = set()
        for cos, x, y in pairs:
            if cos > limit:
                break
            if x in used or y in used:
                continue
            used.update((x, y))
            a, b = here[x], here[y]
            partner[(a[0], a[1])] = (b[0], b[1])
            partner[(b[0], b[1])] = (a[0], a[1])
    out, seen = [], set()

    def follow(w, end):
        """Walk ``w`` entered at its end ``end``, then on through the partners."""
        parts, closed = [], False
        while True:
            seen.add(w)
            nodes = seqs[w][0]
            parts.append(nodes if end == 0 else nodes[::-1])
            nxt = partner.get((w, 1 - end))
            if nxt is None:
                break
            if nxt[0] in seen:
                closed = True
                break
            w, end = nxt
        joined = np.concatenate([parts[0]] + [p[1:] for p in parts[1:]])
        if closed and len(joined) > 1 and joined[-1] == joined[0]:
            joined = joined[:-1]
        return joined, closed

    for w, (nodes, closed) in enumerate(seqs):
        if closed:
            out.append((nodes, True))
            seen.add(w)
    for w in range(len(seqs)):
        if w in seen:
            continue
        for end in (0, 1):
            if (w, end) not in partner:
                out.append(follow(w, end))
                break
    for w in range(len(seqs)):  # rings of walks joined all round
        if w not in seen:
            out.append(follow(w, 0))
    return out


# ---------------------------------------------------------------- choosing and placing

def _crowd_cut(st: Stroke, taken, radius: float, tolerate: float, least: float):
    """The pieces of ``st`` that do not run within ``radius`` of the points ``taken``: a stretch near
    them shorter than ``tolerate`` stays (a crossing, a meeting). Pieces shorter than ``least`` go."""
    whole = [st.like(st.xy.copy(), st.closed)]
    if not len(taken):
        return whole
    near = _near(st.xy, taken, radius)
    if not near.any():
        return whole
    keep = np.ones(len(near), bool)
    for a, b in _runs(near, st.closed):
        if (b - a) * st.step > tolerate:
            keep[np.arange(a, b) % len(near)] = False
    if keep.all():
        return whole
    out = []
    for a, b in _runs(keep, st.closed):
        if (b - a - 1) * st.step >= least:
            out.append(st.like(_take(st.xy, a, b)))
    return out


def _dense(strokes):
    return np.concatenate([st.xy for st in strokes]) if strokes else np.zeros((0, 2))


def choose(sil, inner, dots, s: float, s_fill: float, allow, simple=(), features=()):
    """The drawing: every silhouette; then the dots (islands, then spots such as eyes, the strongest
    first, at most ``MAX_DOTS`` a part); then the best inner lines that do not crowd what was taken
    before, while part k's inner lines and dots hold at most ``allow[k]`` drones. On the parts in
    ``simple`` a dot (an eye) may come as close to a line as two lines do. The outlines of
    ``features`` come right after the silhouettes; a dot inside one is drawn by it.

    Returns (strokes, dots as (picture point, part, ellipse or None))."""
    taken: List[Stroke] = []
    for st in sil:
        taken += _crowd_cut(st, _dense(taken), SELF_GAP * s, 0.0, MIN_RUN * s)
    loops = []
    for st in sorted(features, key=lambda x: -x.score):
        pieces = _crowd_cut(st, _dense(taken), SELF_GAP * s, 0.0, MIN_RUN * s)
        for piece in pieces:
            piece.feature = True
        taken += pieces
        loops.append(st.xy)
    used, count = {}, {}
    chosen = []
    dense = _dense(taken)
    for xy, score, part, *shape in sorted(dots, key=lambda d: -d[1]):
        free = score == math.inf
        if not free and (used.get(part, 0.0) + 1 > allow.get(part, 0.0) or count.get(part, 0) >= MAX_DOTS):
            continue
        if len(dense) and _near(xy[None], dense, (LINE_GAP if part in simple else 0.9) * s)[0]:
            continue
        if any(_inside(xy[None], loop)[0] for loop in loops):
            continue
        if chosen and np.linalg.norm(np.array([c[0] for c in chosen]) - xy, axis=1).min() < s:
            continue
        chosen.append((xy, part, shape[0] if shape else None))
        if not free:
            used[part] = used.get(part, 0.0) + 1
            count[part] = count.get(part, 0) + 1
    spots = np.array([c[0] for c in chosen]).reshape(-1, 2)
    for st in inner:
        if used.get(st.part, 0.0) + MIN_STROKE > allow.get(st.part, 0.0):
            continue
        radius = BAND_GAP * s if st.band else CROWD * s_fill
        least = (1.0 if st.band else MIN_STROKE) * s
        cheap = 0.5 if st.strength >= CRISP else 1.0
        pieces = _crowd_cut(st, spots, DOT_CLEAR * s, 0.0, least)
        pieces = [q for piece in pieces for q in _crowd_cut(piece, _dense(taken), radius, CROSS * radius, least)]
        for piece in pieces:
            c = cheap * piece.length / s
            if used.get(st.part, 0.0) + c > allow.get(st.part, 0.0):
                continue
            used[st.part] = used.get(st.part, 0.0) + c
            taken.append(piece)
    return taken, chosen


def simplify(strokes: List[Stroke], dots, pic: Picture, lin, near: float, s: float, simple):
    """Simple shapes on the parts in ``simple`` (see ``shapes``): a closed line that is nearly an
    ellipse is that ellipse; a dot with the size of an eye becomes a ring round it (a pupil in a big
    one) when the ring keeps clear of the lines. Returns (strokes, dots as (picture point, part))."""
    out = []
    for st in strokes:
        if st.part in simple and st.closed:
            e = _shapes.as_ellipse(st.xy, s)
            if e is not None:
                count = max(int(round(e.perimeter() / s)), 3)
                if count < _shapes.RING_MIN:  # a small one: the smallest ring that reads as one
                    r = _shapes.ring(e, near, s)
                    if r is not None:
                        e, count, _ = r
                st = Stroke(e.points(max(int(math.ceil(e.perimeter() / st.step)), 8)), True, st.kind, st.part,
                            st.step, st.score, st.strength, [], st.band, st.color, e, count)
        out.append(st)
    shapes = [e if part in simple else None for _, part, e in dots]
    for i, j in _pairs(dots, shapes):  # a pair of eyes is drawn alike: both rings of one size, or both dots
        a = np.sqrt(shapes[i].axes * shapes[j].axes)
        shapes[i] = _shapes.Ellipse(shapes[i].center, a, shapes[i].angle, 0.0)
        shapes[j] = _shapes.Ellipse(shapes[j].center, a, shapes[j].angle, 0.0)
    rings = [_shapes.ring(e, near, s) if e is not None else None for e in shapes]
    pts = [r[0].points(max(int(math.ceil(r[0].perimeter() / (STEP * s))), 8)) if r is not None else None
           for r in rings]
    # a ring keeps clear of the silhouettes and of the other rings; the inner lines give way to it
    frame = _dense([st for st in out if st.kind == "silhouette"])
    ok = [p is not None and not (len(frame) and _near(p, frame, LINE_GAP * s).any()) for p in pts]
    pairs = _pairs(dots, shapes)
    partner = {i: j for i, j in pairs} | {j: i for i, j in pairs}
    taken = np.zeros((0, 2))
    for i in range(len(dots)):  # a pair together: both rings (only the minimum distance between them) or none
        j = partner.get(i)
        if j is not None and j < i:
            continue
        group = [i] if j is None else [i, j]
        good = all(ok[g] for g in group)
        if good and len(taken):
            good = not any(_near(pts[g], taken, LINE_GAP * s).any() for g in group)
        if good and j is not None:
            good = not _near(pts[i], pts[j], near).any()
        for g in group:
            ok[g] = good
            if good:
                taken = np.vstack([taken, pts[g]])
    if len(taken):
        kept = []
        for st in out:
            if st.kind == "silhouette":
                kept.append(st)
                continue
            least = (1.0 if st.band else MIN_STROKE) * s
            kept += _crowd_cut(st, taken, LINE_GAP * s, 0.0, least) if st.shape is None else \
                ([] if _near(st.xy, taken, LINE_GAP * s).any() else [st])
        out = kept
    rest = []
    H, W = pic.shape
    for (xy, part, _), r, p, good in zip(dots, rings, pts, ok):
        if not good:
            rest.append((xy, part))  # no room for a ring: the eye stays a dot
            continue
        drawn, count, pupil = r
        col, row = pic.to_px(drawn.center[None])  # the ring glows in the colour of the eye
        colour = lin[int(np.clip(np.rint(row[0]), 0, H - 1)), int(np.clip(np.rint(col[0]), 0, W - 1))]
        out.append(Stroke(p, True, "colors", part, STEP * s, strength=10.0, color=colour, shape=drawn, count=count))
        if pupil:
            rest.append((drawn.center, part))
    return out, rest


def _pairs(dots, shapes):
    """(i, j) of the dots that are a pair of eyes: on one part, side by side about level, alike in size."""
    out, used = [], set()
    idx = [i for i, e in enumerate(shapes) if e is not None]
    for x in idx:
        for y in idx:
            if y <= x or x in used or y in used or dots[x][1] != dots[y][1]:
                continue
            a, b = shapes[x], shapes[y]
            size = max(a.axes[0], b.axes[0])
            d = b.center - a.center
            alike = max(a.axes[0] / b.axes[0], b.axes[0] / a.axes[0]) < PAIR_SIZE
            if alike and abs(d[1]) < PAIR_LEVEL * size and abs(d[0]) < PAIR_APART * size:
                out.append((x, y))
                used.update((x, y))
    return out


def corners(strokes: List[Stroke], s: float):
    """Marks the corners on the strokes (``Stroke.corners``, sample indices): sharp turns at drone
    scale, and where a line ends on another (it is drawn on to meet it). Returns the corner points."""
    pts = []
    for st in strokes:
        st.corners = []
        if st.shape is not None:
            continue
        w = max(int(round(CORNER_WINDOW * s / st.step)), 1)
        n = len(st.xy)
        if (n if st.closed else n - 2 * w) < 2 * w + 1:
            continue
        turn = np.degrees(_turn(st.xy, st.closed, w))
        turn2 = np.degrees(_turn(st.xy, st.closed, 2 * w))
        sharp = CORNER_TURN if st.kind == "silhouette" or st.band else INNER_TURN  # soft inner lines bend more
        ok = (turn >= sharp) & (turn2 >= CORNER_KEEP * sharp)
        if not st.closed:
            ok[:w] = False
            ok[n - w:] = False
        taken = []
        for i in np.argsort(-turn, kind="stable"):
            if not ok[i]:
                continue
            if taken:
                gap = np.abs(np.array(taken) - i)
                if st.closed:
                    gap = np.minimum(gap, n - gap)
                if gap.min() <= w:
                    continue
            taken.append(int(i))
        st.corners = sorted(taken)
        for i in st.corners:  # the corner itself, where the straight sides either side of it meet
            ix = [(i - 2 * w), (i - w), (i + w), (i + 2 * w)]
            ix = [j % n for j in ix] if st.closed else [min(max(j, 0), n - 1) for j in ix]
            a1, a2, b1, b2 = (st.xy[j] for j in ix)
            da, db = a2 - a1, b2 - b1
            den = da[0] * db[1] - da[1] * db[0]
            if abs(den) > 1e-12 * max(float(da @ da), 1e-300):
                t = ((b1[0] - a1[0]) * db[1] - (b1[1] - a1[1]) * db[0]) / den
                apex = a1 + t * da
                if np.linalg.norm(apex - st.xy[i]) < APEX * s:
                    st.xy[i] = apex
    # a line that ends next to another meets it there
    for a, st in enumerate(strokes):
        if st.closed:
            continue
        reach = (JOIN_BAND if st.band and st.kind == "colors" else JOIN) * s
        for end in (0, -1):
            e = st.xy[end]
            best = None
            for b, other in enumerate(strokes):
                if b == a:
                    continue
                d = np.linalg.norm(other.xy - e, axis=1)
                i = int(np.argmin(d))
                if d[i] <= reach and (best is None or d[i] < best[0]):
                    best = (float(d[i]), b, i)
            if best is None:
                continue
            _, b, i = best
            q = strokes[b].xy[i]
            other = strokes[b]
            back = max(1, min(len(st.xy) - 1, int(round(0.5 * s / st.step))))
            d_end = st.xy[end] - st.xy[back if end == 0 else -1 - back]
            j0, j1 = max(i - 2, 0), min(i + 2, len(other.xy) - 1)
            if other.closed:
                j0, j1 = (i - 2) % len(other.xy), (i + 2) % len(other.xy)
            tangent = other.xy[j1] - other.xy[j0]
            cos = abs(float(d_end @ tangent)) / max(float(np.linalg.norm(d_end) * np.linalg.norm(tangent)), 1e-300)
            if cos > math.cos(math.radians(JOIN_ANGLE)) and 0 < i < len(other.xy) - 1:
                continue  # it runs into the other line at a glance: no corner there
            if best[0] > 0.5 * st.step:
                bridge = resample(np.vstack([e, q]), st.step, False)[1:]
                st.xy = np.vstack([st.xy, bridge]) if end == -1 else np.vstack([bridge[::-1], st.xy])
                if end == 0:
                    st.corners = [c + len(bridge) for c in st.corners]
            st.corners = sorted(set(st.corners) | {0 if end == 0 else len(st.xy) - 1})
            other = strokes[b]
            if not other.closed and (i < 2 or i > len(other.xy) - 3):
                i = 0 if i < 2 else len(other.xy) - 1
            other.corners = sorted(set(other.corners) | {i})
    for st in strokes:
        for c in st.corners:
            pts.append(st.xy[c])
    return np.array(pts).reshape(-1, 2)


def place(strokes: List[Stroke], near: float, s: float, simple=()):
    """Drones along the strokes: one in every corner, then the line drones, the spacing growing from
    ``near`` at a corner to ``s``; a loop with no corner evenly all round, a loose end keeps a drone.
    On the parts in ``simple`` a nearly straight stretch between corners is drawn straight.

    Returns (picture points, kind 0/1, stroke index, local spacing), corners first."""
    c_xy, c_st = [], []
    l_xy, l_st, l_sig = [], [], []
    for idx, st in enumerate(strokes):
        n = len(st.xy)
        cs = list(st.corners)
        for c in cs:
            c_xy.append(st.xy[c])
            c_st.append(idx)
        sig = SMOOTH * s / st.step
        if st.closed and not cs:
            p = st.xy if st.shape is not None else smooth(st.xy, sig, True)
            total = _arc(p, True)[-1]
            g = st.count or max(int(round(total / s)), 3)
            x = np.arange(g) * (total / g)
            l_xy.append(_at(p, x, True))
            l_sig.append(np.full(g, s))
            l_st.append(np.full(g, idx))
            continue
        if st.closed:
            bounds = list(zip(cs, cs[1:] + [cs[0] + n]))
            ends = [(True, True)] * len(bounds)
        else:
            marks = sorted(set([0] + cs + [n - 1]))
            bounds = list(zip(marks[:-1], marks[1:]))
            ends = [(a in cs, b in cs) for a, b in bounds]
        for (a, b), (c0, c1) in zip(bounds, ends):
            piece = _take(st.xy, a, b + 1)
            if len(piece) < 2:
                continue
            piece = smooth(piece, sig, False)
            if st.part in simple and st.shape is None:
                piece = _shapes.straighten(piece, s)
            x, sigma = _graded(_arc(piece, False)[-1], c0, c1, near, s)
            if not len(x):
                continue
            l_xy.append(_at(piece, x, False))
            l_sig.append(sigma)
            l_st.append(np.full(len(x), idx))
    xy = np.array(c_xy).reshape(-1, 2)
    if l_xy:
        xy = np.vstack([xy] + l_xy)
    kind = np.r_[np.zeros(len(c_xy), np.int8), np.ones(len(xy) - len(c_xy), np.int8)]
    st_idx = np.r_[np.array(c_st, np.int64), np.concatenate(l_st) if l_st else _NO]
    sigma = np.r_[np.full(len(c_xy), near), np.concatenate(l_sig) if l_sig else np.zeros(0)]
    # the drones of a line keep their places in the order the lines were taken, the corners first
    order = np.lexsort((st_idx, kind))
    xy, kind, st_idx, sigma = xy[order], kind[order], st_idx[order], sigma[order]
    radius = np.where(kind == 0, CORNER_GAP_FACTOR * s, np.maximum(near, LINE_GAP * sigma))
    keep = _outline.park(_pad3(xy), radius)
    return xy[keep], kind[keep], st_idx[keep], sigma[keep]




def _graded(total, c0, c1, near, s):
    """Distances along a piece of ``total`` length where its drones stand (its corners left out):
    ``near`` from a corner end, the spacing growing by ``GRADE`` per length to ``s``; a loose end keeps
    a drone. Returns (distances, local spacing)."""
    if total <= 0:
        return np.zeros(0), np.zeros(0)
    k = max(int(math.ceil(total / (0.1 * near))), 2) + 1
    xs = np.linspace(0.0, total, k)
    dist = np.full(k, np.inf)
    if c0:
        dist = np.minimum(dist, xs)
    if c1:
        dist = np.minimum(dist, total - xs)
    local = np.minimum(s, near + GRADE * dist)
    F = np.r_[0.0, np.cumsum(np.diff(xs) * 0.5 * (1.0 / local[1:] + 1.0 / local[:-1]))]
    g = max(int(round(F[-1])), 1)
    j = np.arange(1 if c0 else 0, g if c1 else g + 1)
    if not len(j):
        return np.zeros(0), np.zeros(0)
    x = np.interp(j * (F[-1] / g), F, xs)
    return x, np.interp(x, xs, local)


def _at(p, x, closed):
    t = _arc(p, closed)
    q = np.vstack([p, p[:1]]) if closed else p
    if closed:
        x = np.mod(x, t[-1])
    return np.stack([np.interp(x, t, q[:, i]) for i in range(2)], 1)


def lattice(pic: Picture, allowed, s: float, pattern: str, avoid, gap: float, clear_pts, clear: float):
    """Points of a lattice of spacing ``s`` (rows level in the picture) where ``allowed`` (H, W) is
    true, ``gap`` clear of the points ``avoid`` and ``clear`` of ``clear_pts``; the shift of the lattice
    that keeps the most points is taken. Returns picture points."""
    rows, cols = np.nonzero(allowed)
    if not len(rows):
        return np.zeros((0, 2))
    xy = pic.pixel_xy(rows, cols)
    lo, hi = xy.min(axis=0) - s, xy.max(axis=0) + s
    if pattern == "square":
        a1, a2 = np.array([s, 0.0]), np.array([0.0, s])
    else:
        a1, a2 = np.array([s, 0.0]), np.array([0.5 * s, math.sqrt(3.0) / 2.0 * s])
    nj = int(math.ceil((hi[1] - lo[1]) / a2[1])) + 2
    ni = int(math.ceil((hi[0] - lo[0]) / s)) + nj + 2
    if ni * nj > 4_000_000:
        return np.zeros((0, 2))
    jj, ii = np.mgrid[0:nj, -nj:ni - nj]
    base = lo + ii.reshape(-1, 1) * a1 + jj.reshape(-1, 1) * a2
    base = base[(base[:, 0] >= lo[0] - s) & (base[:, 0] <= hi[0] + s)]
    best = np.zeros((0, 2))
    for u in range(PHASES):
        for v in range(PHASES):
            p = base + (u / PHASES) * a1 + (v / PHASES) * a2
            p = p[pic.sample(allowed, p, False)]
            if len(avoid) and len(p):
                p = p[~_near(p, avoid, gap)]
            if len(clear_pts) and len(p):
                p = p[~_near(p, clear_pts, clear)]
            if len(p) > len(best):
                best = p
    return best


def pockets(pic: Picture, allowed, s: float, avoid, gap: float, taken_pts):
    """Fill drones in the holes the lattice leaves next to the lines: the farthest free spot first,
    while it is at least ``POCKET`` fill spacings from every drone. Returns picture points."""
    stride = max(int(s / (4.0 * pic.h)), 1)
    sub = np.zeros_like(allowed)
    sub[::stride, ::stride] = allowed[::stride, ::stride]
    rows, cols = np.nonzero(sub)
    if not len(rows):
        return np.zeros((0, 2))
    c = pic.pixel_xy(rows, cols)
    if len(avoid):
        c = c[~_near(c, avoid, gap)]
    if len(taken_pts) and len(c):
        c = c[~_near(c, taken_pts, gap)]
    if not len(c):
        return np.zeros((0, 2))
    d, _ = nearest(_pad3(c), _pad3(taken_pts), s) if len(taken_pts) else (np.full(len(c), np.inf), None)
    out = []
    while True:
        i = int(np.argmax(d))
        if not d[i] >= POCKET * s:
            break
        out.append(c[i])
        d = np.minimum(d, np.linalg.norm(c - c[i], axis=1))
    return np.array(out).reshape(-1, 2)


def estimate(pic: Picture, nparts: int, s: float):
    """(area, silhouette length) of each part in the picture, read at line spacing ``s``."""
    area = np.bincount(pic.part[pic.fg], minlength=nparts) * pic.h * pic.h
    length = np.zeros(nparts)
    for k in range(nparts):
        mask = pic.part == k
        if mask.any():
            for loop in contours(mask.astype(np.float64)):
                xy = pic.from_px(loop)
                if len(xy) > 2:
                    length[k] += _arc(resample(xy, 0.5 * s, True), True)[-1]
    return area, length


def colors_near(pic: Picture, lin, xy, part, radius: float, stand_out=None):
    """Mean colour of the pixels of ``part`` within ``radius`` of each picture point; with
    ``stand_out`` (OKLab mean colour of each point's part) only the half that differs most from it."""
    n = len(xy)
    out = np.full((n, 3), np.nan)
    col, row = pic.to_px(xy)
    r = max(int(math.ceil(radius / pic.h)), 1)
    offs = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if dy * dy + dx * dx <= r * r]
    H, W = pic.shape
    ci, ri = np.rint(col).astype(np.int64), np.rint(row).astype(np.int64)
    acc = np.zeros((n, len(offs), 3))
    ok = np.zeros((n, len(offs)), bool)
    for m, (dy, dx) in enumerate(offs):
        rr, cc = np.clip(ri + dy, 0, H - 1), np.clip(ci + dx, 0, W - 1)
        good = (ri + dy >= 0) & (ri + dy < H) & (ci + dx >= 0) & (ci + dx < W) & (pic.part[rr, cc] == part)
        ok[:, m] = good
        acc[:, m] = lin[rr, cc]
    if stand_out is not None:
        far = np.linalg.norm(_color.linear_to_oklab(acc.reshape(-1, 3)).reshape(n, len(offs), 3)
                             - stand_out[:, None, :], axis=2)
        far = np.where(ok, far, -1.0)
        cut = np.array([np.median(f[o]) if o.any() else 0.0 for f, o in zip(far, ok)])
        ok &= far >= cut[:, None]
    count = ok.sum(axis=1)
    has = count > 0
    out[has] = (acc * ok[..., None]).sum(axis=1)[has] / count[has, None]
    return out
