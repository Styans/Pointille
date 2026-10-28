"""Figure: drones that draw a character or an animal crisply from every side the audience may stand.

By default (``CFG["rows"]``) the figure is drawn as the big shows draw one: even rows of drones (every
horizontal ring of it, the drones evenly along it, every other ring half a step on), the features over
them bright, the rows dimmer - the eyes found in the picture from the front (``find_eyes_front``) as rings
with a dark gap round them, the rims of hollows (an arm against the body) white, the crown in its colour.

Without rows: the figure is cut into horizontal rings (``Slicer``). On every ring the drones stand where the ring
meets its own outline seen from K directions round the vertical (``place_ring``): the same directions on
every ring, so the drones of the rings stack into lines running up the body - the lines the outline of
the figure follows from every side. Seen from any azimuth, the outermost line of every ring is within
``tau`` of the true outline, so the silhouette is drawn by an unbroken column of drones ``g`` apart from
every side (the multi-view outline cover). Features come first (``features``): the rims of carved or
sculpted hollows (eyes, mouth, nose) and the borders of regions of a clearly different colour (crown,
stem) as closed loops with their corners, bright; the ring drones keep clear of them and of the inside of
hollows, and glow dimmer. The ring spacing ``g`` is fitted to the budget; with drones to spare, more
directions round every ring. A final multi-view relaxation slides ring drones along their rings so that
fewer pairs coincide in projection from 72 views, keeping mirror pairs mirrored. numpy only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Loop:
    pts: np.ndarray  # (n, 3) crossing points in order
    tri: np.ndarray  # (n,) triangle of the segment from point i to point i + 1 (closed: n segments)
    b0: np.ndarray  # (n, 3) barycentrics of point i in that triangle
    b1: np.ndarray  # (n, 3) barycentrics of point i + 1 in that triangle
    closed: bool
    z: float

    @property
    def length(self) -> float:
        return float(self.seglen.sum())

    @property
    def seglen(self) -> np.ndarray:
        q = self.pts if not self.closed else np.vstack([self.pts, self.pts[:1]])
        return np.linalg.norm(np.diff(q, axis=0), axis=1)


class Slicer:
    """Horizontal cuts of a triangle mesh into ordered loops (welded by mesh edge)."""

    def __init__(self, tris: np.ndarray, up: int = 2):
        self.tris = tris
        corners = tris.reshape(-1, 3)
        lo = corners.min(axis=0)
        extent = float(np.ptp(corners, axis=0).max()) or 1.0
        q = np.floor((corners - lo) / (extent * 1e-5) + 0.5).astype(np.int64)
        key = q[:, 0] | (q[:, 1] << 17) | (q[:, 2] << 34)
        _, first, vid = np.unique(key, return_index=True, return_inverse=True)
        self.vid = vid.reshape(-1, 3)
        self.verts = corners[first]
        self.nv = int(vid.max()) + 1
        self.z = tris[:, :, up]
        zmin, zmax = self.z.min(axis=1), self.z.max(axis=1)
        self.zlo = float(zmin.min())
        self.zhi = float(zmax.max())
        # triangles bucketed by height: a cut only looks at the buckets it crosses
        self.bin = max((self.zhi - self.zlo) / 400.0, 1e-9)
        b0 = np.floor((zmin - self.zlo) / self.bin).astype(np.int64)
        b1 = np.floor((zmax - self.zlo) / self.bin).astype(np.int64)
        span = b1 - b0 + 1
        owner = np.repeat(np.arange(len(tris)), span)
        cell = np.repeat(b0, span) + (np.arange(span.sum()) - np.repeat(np.cumsum(span) - span, span))
        order = np.argsort(cell, kind="stable")
        self.cell_tri = owner[order]
        cells = cell[order]
        self.nbins = int(b1.max()) + 1
        self.cell_start = np.searchsorted(cells, np.arange(self.nbins + 1))

    def cut(self, z: float):
        k = int(np.clip(math.floor((z - self.zlo) / self.bin), 0, self.nbins - 1))
        return self.iso(self.z, z, self.cell_tri[self.cell_start[k]:self.cell_start[k + 1]])

    def iso(self, values: np.ndarray, z: float, cand=None):
        """Loops where ``values`` (T, 3), one per triangle corner, cross ``z`` (among triangles ``cand``)."""
        if cand is None:
            cand = np.arange(len(values))
        zz = values[cand]
        above = zz >= z
        na = above.sum(axis=1)
        t = cand[(na == 1) | (na == 2)]
        if not len(t):
            return []
        above = values[t] >= z
        # the corner alone on its side, and the two edges from it
        lone = np.where(above.sum(axis=1) == 1, np.argmax(above, axis=1), np.argmin(above, axis=1))
        i0 = lone
        i1 = (lone + 1) % 3
        i2 = (lone + 2) % 3
        r = np.arange(len(t))
        tri = self.tris[t]
        za, zb, zc = values[t, i0], values[t, i1], values[t, i2]
        s1 = np.clip((z - za) / (zb - za), 0.0, 1.0)
        s2 = np.clip((z - za) / (zc - za), 0.0, 1.0)
        pa = tri[r, i0] + s1[:, None] * (tri[r, i1] - tri[r, i0])
        pb = tri[r, i0] + s2[:, None] * (tri[r, i2] - tri[r, i0])
        ba = np.zeros((len(t), 3))
        bb = np.zeros((len(t), 3))
        ba[r, i0] = 1 - s1
        ba[r, i1] = s1
        bb[r, i0] = 1 - s2
        bb[r, i2] = s2
        v = self.vid[t]
        ka = np.minimum(v[r, i0], v[r, i1]) * self.nv + np.maximum(v[r, i0], v[r, i1])
        kb = np.minimum(v[r, i0], v[r, i2]) * self.nv + np.maximum(v[r, i0], v[r, i2])
        return self._chain(t, pa, pb, ba, bb, ka, kb, z)

    @staticmethod
    def _chain(t, pa, pb, ba, bb, ka, kb, z):
        n = len(t)
        keys = np.concatenate([ka, kb])
        order = np.argsort(keys, kind="stable")
        sk = keys[order]
        # partner end of every end (the other triangle on the same mesh edge), -1 at a border
        partner = np.full(2 * n, -1, np.int64)
        same = np.flatnonzero(sk[1:] == sk[:-1])
        # pair consecutive equal keys (non-manifold edges: first two only)
        used = np.zeros(2 * n, bool)
        for j in same.tolist():
            a, b = order[j], order[j + 1]
            if used[a] or used[b]:
                continue
            partner[a], partner[b] = b, a
            used[a] = used[b] = True
        partner = partner.tolist()
        seen = [False] * n
        loops = []

        def walk(s, end_out):
            """Follow from segment s leaving through end ``end_out`` (0: a, 1: b). Returns segment list
            with their direction (True: a -> b)."""
            seq = []
            cur, out = s, end_out
            while True:
                seq.append((cur, out == 1))
                seen[cur] = True
                e = partner[cur + out * n]
                if e < 0:
                    return seq, False
                nxt = e % n
                if nxt == s:
                    return seq, True
                if seen[nxt]:
                    return seq, False
                cur, out = nxt, 1 - e // n
        for s in range(n):
            if seen[s]:
                continue
            fwd, closed = walk(s, 1)
            if not closed:
                # open chain: walk back from s too and join
                seen[s] = False
                back, _ = walk(s, 0)
                back = [(c, not d) for c, d in reversed(back)]
                fwd = back[:-1] + fwd if back else fwd
            idx = np.array([c for c, _ in fwd])
            d = np.array([f for _, f in fwd])
            p0 = np.where(d[:, None], pa[idx], pb[idx])
            p1 = np.where(d[:, None], pb[idx], pa[idx])
            b0 = np.where(d[:, None], ba[idx], bb[idx])
            b1 = np.where(d[:, None], bb[idx], ba[idx])
            if closed:
                pts = p0
            else:
                pts = np.vstack([p0, p1[-1:]])
            loops.append(Loop(pts, t[idx], b0, b1, closed, z))
        return loops




# ---------------------------------------------------------------------------------------------- rings

from . import color as _color  # noqa: E402
from .api import DistributeOptions, DroneResult  # noqa: E402
from .mesh import MeshData  # noqa: E402
from .outline import park  # noqa: E402
from .sampling import surface_candidates, triangle_areas_normals  # noqa: E402
from .spatial import Grid, nearest  # noqa: E402
from .stats import spacing  # noqa: E402

CFG = dict(K=16, stagger=False, fill=None, min_arc=1.15, deep=0.5, g0=1.0, step=0.2, tau=0.5, smooth=1.0, rows=0.5)


@dataclass
class Run:
    """A loop resampled at a fine step: positions, triangles, barycentrics, arc length."""
    pos: np.ndarray
    tri: np.ndarray
    bary: np.ndarray
    arc: np.ndarray
    length: float
    closed: bool


def resample(loop: Loop, step: float) -> Run:
    seg = loop.seglen
    total = float(seg.sum())
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(math.ceil(total / step)), 1)
    s = np.arange(n) * (total / n) if loop.closed else np.linspace(0.0, total, n + 1)
    k = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
    f = np.clip((s - cum[k]) / np.maximum(seg[k], 1e-12), 0.0, 1.0)
    q = loop.pts if not loop.closed else np.vstack([loop.pts, loop.pts[:1]])
    pos = q[k] + f[:, None] * (q[k + 1] - q[k])
    bary = loop.b0[k] + f[:, None] * (loop.b1[k] - loop.b0[k])
    return Run(pos, loop.tri[k], bary, s, total, loop.closed)


def directions(K: int, shift: bool) -> np.ndarray:
    """K directions round the vertical, mirror symmetric about the x = 0 plane (one points to -y, the
    audience in front), or the K halfway between them."""
    th = -0.5 * math.pi + (2.0 * math.pi / K) * (np.arange(K) + (0.5 if shift else 0.0))
    return np.stack([np.cos(th), np.sin(th)], 1)


def smooth_xy(run: Run, width: float) -> np.ndarray:
    xy = run.pos[:, :2]
    w = int(round(width / max(run.arc[1] - run.arc[0], 1e-9))) if len(run.arc) > 1 else 0
    if w < 2 or not run.closed or len(xy) < 2 * w + 1:
        return xy
    k = np.ones(2 * w + 1) / (2 * w + 1)
    pad = np.concatenate([xy[-w:], xy, xy[:w]])
    return np.stack([np.convolve(pad[:, c], k, mode="valid") for c in range(2)], 1)


def ladder(K: int):
    """Direction counts from K down: halving powers of two, or for an odd K the odd numbers near K/2^i
    (odd counts keep the front and the back drones of a ring from stacking in the views from the front)."""
    out = [K]
    while out[-1] > 1:
        k = out[-1] // 2
        if K % 2 == 1 and k > 1 and k % 2 == 0:
            k += 1
        out.append(max(k, 1) if k < out[-1] else out[-1] - 1)
    return out


def ring_k(length: float, K: int, tau, min_arc: float) -> int:
    """Directions for a ring: enough that its outline from any direction between two of them is
    within ``tau`` of the true one, from a fixed ladder, so the rings of a body stack into whole lines."""
    r = length / (2.0 * math.pi)
    need = 1.0
    if tau is not None and tau < 2 * r:
        need = math.pi / math.acos(max(1.0 - tau / max(r, 1e-9), -1.0))
    best = 1
    for k in ladder(K):
        if length / k >= min_arc:
            best = k
            if tau is None or k <= 1.41 * need:
                break
        if tau is not None and k < need / 1.41:
            break
    return best


def place_ring(run: Run, K: int, shift: bool, amax, deep: float, min_arc: float, tau=None, smooth=0.0):
    """Indices into ``run`` of the drones of one ring: where the ring meets its outline from K directions
    round it (the same directions on every ring, so the drones of the rings stack into the lines the
    outline of the figure runs along from every side), then gaps longer than ``amax`` split evenly (not in
    pockets deeper than ``deep``). Returns (idx, kind)."""
    m = len(run.pos)
    xy = smooth_xy(run, smooth) if smooth > 0 else run.pos[:, :2]
    k = ring_k(run.length, K, tau, min_arc) if run.closed else K
    if run.length < min_arc or m < 3 or k < 2:
        u = directions(4, shift)[:1]
        place_ring.angles = [float("nan")]
        return np.array([int(np.argmax(xy @ u[0]))]), np.array([0], np.int8)
    u = directions(k, shift)
    proj = xy @ u.T
    hit = np.argmax(proj, axis=0)
    sel, first = np.unique(hit, return_index=True)
    place_ring.angles = list(np.arctan2(u[first, 1], u[first, 0]))
    out = [(int(i), 1) for i in sel]
    if amax is not None and run.closed:
        depth = (proj.max(axis=0) - proj).min(axis=1) if k >= 8 else np.zeros(m)
        arc, L = run.arc, run.length
        pts = list(sel)
        for a, b in zip(pts, pts[1:] + pts[:1]):
            gap = (arc[b] - arc[a]) % L if len(pts) > 1 else L
            n = int(math.ceil(gap / amax - 1e-9)) - 1
            for j in range(1, n + 1):
                s = (arc[a] + gap * j / (n + 1)) % L
                i = int(np.argmin(np.abs(arc - s)))
                if depth[i] <= deep:
                    out.append((i, 2))
    if not run.closed:
        out += [(0, 1), (m - 1, 1)]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out], np.int8)


def place_row(run: Run, a: float, half: bool, min_arc: float):
    """Indices into ``run`` of the drones of one row: evenly ``a`` apart along it (every other row half a
    step on, so the rows do not stack into columns), the first one facing the audience. Returns (idx, kind)."""
    m = len(run.pos)
    if run.length < min_arc or m < 3:
        return np.array([int(np.argmin(run.pos[:, 1]))]), np.array([1], np.int8)
    if run.closed:
        n = max(int(round(run.length / a)), 1)
        start = run.arc[int(np.argmin(run.pos[:, 1]))]
        s = np.mod(start + (np.arange(n) + (0.5 if half else 0.0)) * (run.length / n), run.length)
    else:
        n = max(int(round(run.length / a)), 1)
        s = np.linspace(0.0, run.length, n + 1)
    idx = np.unique(np.clip(np.searchsorted(run.arc, s), 0, m - 1))
    return idx, np.full(len(idx), 1, np.int8)


class Rings:
    def __init__(self, slicer: Slicer, step: float):
        self.slicer = slicer
        self.step = step
        self.cache = {}

    def runs(self, z: float):
        key = round(z, 4)
        if key not in self.cache:
            self.cache[key] = [resample(lp, self.step) for lp in self.slicer.cut(z) if lp.length > 1e-3]
        return self.cache[key]

    def build(self, g: float, K: int, stagger: bool, amax, deep: float, min_arc: float, tau=None, smooth=0.0,
              row=None):
        """Drones on rings ``g`` apart up the figure: at its outline from K directions round it, or with
        ``row`` set, evenly that far apart along every ring (rows, like a drawing filled with lines)."""
        s = self.slicer
        h = s.zhi - s.zlo
        n = max(int(math.floor((h - 0.3 * g) / g)) + 1, 1)
        first = s.zlo + 0.5 * (h - (n - 1) * g)
        P, T, Bc, Kd, R, S, Lv, An = [], [], [], [], [], [], [], []
        self.used = []
        for j in range(n):
            z = first + g * j
            for run in self.runs(z):
                if row is not None:
                    idx, kind = place_row(run, row, j % 2 == 1, min_arc)
                    place_ring.angles = []
                else:
                    # no direction straight at the audience: its drones would stack into a seam down the front
                    idx, kind = place_ring(run, K, j % 2 == 1 if stagger else True, amax, deep, min_arc, tau,
                                           smooth)
                P.append(run.pos[idx])
                T.append(run.tri[idx])
                Bc.append(run.bary[idx])
                Kd.append(kind)
                R.append(np.full(len(idx), len(self.used)))
                Lv.append(np.full(len(idx), j))
                ang = list(place_ring.angles)
                An.append(np.array((ang + [float("nan")] * len(idx))[:len(idx)]))
                S.append(idx)
                self.used.append(run)
        self.run_id, self.sample = np.concatenate(R), np.concatenate(S)
        self.level, self.angle = np.concatenate(Lv), np.concatenate(An)
        return np.concatenate(P), np.concatenate(T), np.concatenate(Bc), np.concatenate(Kd)


def _colors(mesh, tris, pos, tri, bary, radius, rng):
    areas, _ = triangle_areas_normals(tris)
    cp, ct, cb = surface_candidates(tris, areas, 200000, rng)
    lin = _color.surface_colors(mesh, ct, cb)
    _, near = nearest(pos, cp, radius)
    qi, pj, _ = Grid(cp, radius).query_radius(pos, radius)
    lab = _color.linear_to_oklab(lin)
    close = np.linalg.norm(lab[pj] - lab[near[qi]], axis=1) < 0.08
    qi, pj = qi[close], pj[close]
    col = np.stack([np.bincount(qi, weights=lin[pj, c], minlength=len(pos)) for c in range(3)], 1)
    cnt = np.bincount(qi, minlength=len(pos))
    col = np.where(cnt[:, None] > 0, col / np.maximum(cnt, 1)[:, None], lin[near])
    return col



# ---------------------------------------------------------------------------------------------- features
from .outline import _components  # noqa: E402
from .spatial import nn_distances  # noqa: E402

FEAT = dict(R=2.0, depth=0.33, core=0.5, spacing=1.15, share=0.4, chroma=0.035, ring_level=0.65, crease_level=0.95, face_high=0.55, face_area=18.0, eyes=True, tips=True, smooth_lines=True, clear=1.1, clear_abs=1.2, fill_min=99.0, slim=5.0,
            colors=True, cavities=True, stray=1.7, eyes_front=True, eye_clear=2.2, pupil=False,
            lift=(1.0, 0.78, 0.35), lift_hollow=(1.0, 1.0, 1.0))


class Surface:
    """Dense samples of the surface with a relief field (depth below the local patch) and smoothed colour."""

    def __init__(self, mesh, tris, areas, normals, d0, rng):
        R = FEAT["R"] * d0
        total = float(areas.sum())
        n = int(min(max(total / (0.12 * R) ** 2, 20000), 60000))
        self.p, self.t, self.b = surface_candidates(tris, areas, n, rng)
        self.cell = total / n  # area per sample
        nrm = normals[self.t]
        qi, pj, _ = Grid(self.p, R).query_radius(self.p, R)
        cnt = np.maximum(np.bincount(qi, minlength=n), 1)[:, None]
        mu = np.stack([np.bincount(qi, weights=self.p[pj, c], minlength=n) for c in range(3)], 1) / cnt
        nu = np.stack([np.bincount(qi, weights=nrm[pj, c], minlength=n) for c in range(3)], 1)
        nu /= np.linalg.norm(nu, axis=1, keepdims=True) + 1e-12
        self.relief = np.einsum("ij,ij->i", self.p - mu, nu)
        self.n = nrm
        r = 0.5 * R
        qi, pj, _ = Grid(self.p, r).query_radius(self.p, r)
        cnt = np.maximum(np.bincount(qi, minlength=n), 1)[:, None]
        mu = np.stack([np.bincount(qi, weights=self.p[pj, c], minlength=n) for c in range(3)], 1) / cnt
        nu = np.stack([np.bincount(qi, weights=nrm[pj, c], minlength=n) for c in range(3)], 1)
        nu /= np.linalg.norm(nu, axis=1, keepdims=True) + 1e-12
        self.fine = np.einsum("ij,ij->i", self.p - mu, nu)
        self.lin = _color.surface_colors(mesh, self.t, self.b)
        lab = _color.linear_to_oklab(self.lin)
        r = 0.4 * d0
        qi, pj, _ = Grid(self.p, r).query_radius(self.p, r)
        cnt = np.maximum(np.bincount(qi, minlength=n), 1)[:, None]
        self.lab = np.stack([np.bincount(qi, weights=lab[pj, c], minlength=n) for c in range(3)], 1) / cnt
        main = np.median(self.lab[:, 1:], axis=0)
        self.chroma_off = np.linalg.norm(self.lab[:, 1:] - main, axis=1)
        self.spacing = math.sqrt(self.cell)

    def corner_field(self, slicer: Slicer, value: np.ndarray, band) -> np.ndarray:
        """``value`` (per sample) at the corners of every triangle: the nearest sample's, smoothed over the
        samples around the vertices near the level ``band`` (lo, hi)."""
        V = slicer.verts
        _, idx = nearest(V, self.p, 2.0 * self.spacing)
        fv = value[idx]
        near = np.flatnonzero((fv > band[0]) & (fv < band[1]))
        if len(near):
            r = 2.5 * self.spacing
            qi, pj, d = Grid(self.p, r).query_radius(V[near], r)
            w = np.exp(-0.5 * (d / (0.8 * self.spacing)) ** 2)
            num = np.bincount(qi, weights=w * value[pj], minlength=len(near))
            den = np.bincount(qi, weights=w, minlength=len(near))
            fv[near] = np.where(den > 0, num / np.maximum(den, 1e-12), fv[near])
        return fv[slicer.vid]


def _regions(surf: Surface, mask: np.ndarray, core: np.ndarray, least: float, most: float, slim: float = 0.0):
    """Connected regions of the samples in ``mask`` that hold a ``core`` sample and an area in [least, most].
    Returns the region label of every sample (-1 outside a kept region)."""
    idx = np.flatnonzero(mask)
    lab = np.full(len(surf.p), -1, np.int64)
    if not len(idx):
        return lab
    r = 1.8 * surf.spacing
    qi, pj, _ = Grid(surf.p[idx], r).query_radius(surf.p[idx], r)
    comp = _components(len(idx), qi, pj)
    _, comp = np.unique(comp, return_inverse=True)
    area = np.bincount(comp) * surf.cell
    has_core = np.bincount(comp, weights=core[idx].astype(float)) > 0
    keep = (area >= least) & (area <= most) & has_core
    if slim:  # long thin crevices (where an arm meets the body) are no feature
        for k in np.flatnonzero(keep):
            q = surf.p[idx[comp == k]]
            ev = np.sort(np.linalg.eigvalsh(np.cov((q - q.mean(axis=0)).T)))[::-1]
            if math.sqrt(ev[0] / max(ev[1], 1e-12)) > slim:
                keep[k] = False
    remap = np.cumsum(keep) - 1
    lab[idx] = np.where(keep[comp], remap[comp], -1)
    return lab


def _loop_drones(run: Run, a: float, min_count: int = 0):
    """Drones on a closed feature loop: its corners (turning more than 60 degrees within ``a``), then even
    spacing about ``a`` between them. Returns (indices, is_corner)."""
    m = len(run.pos)
    L = run.length
    if m < 4 or L < 0.5 * a:
        return np.array([int(np.argmax(run.pos[:, 2]))]), np.array([True])
    step = L / m
    w = max(int(round(0.35 * a / step)), 1)
    p = run.pos
    fwd = np.roll(p, -w, axis=0) - p
    back = p - np.roll(p, w, axis=0)
    cosang = np.einsum("ij,ij->i", fwd, back) / (np.linalg.norm(fwd, axis=1) * np.linalg.norm(back, axis=1) + 1e-12)
    turn = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    corners = []
    for i in np.argsort(-turn):
        if turn[i] < 60:
            break
        if all(min(abs(i - c), m - abs(i - c)) * step > 0.8 * a for c in corners):
            corners.append(int(i))
    corners.sort()
    if not corners:
        n = max(int(round(L / a)), min_count, 1)
        start = int(np.argmax(p[:, 2]))
        idx = (start + np.round(np.arange(n) * m / n).astype(int)) % m
        return idx, np.zeros(n, bool)
    out, flag = [], []
    for k, c in enumerate(corners):
        nxt = corners[(k + 1) % len(corners)]
        gap = (nxt - c) % m or m
        n = max(int(round(gap * step / a)), 1)
        for j in range(n):
            out.append((c + int(round(j * gap / n))) % m)
            flag.append(j == 0)
    return np.array(out), np.array(flag)


def features(surf: Surface, slicer: Slicer, total_area: float, a: float, d0: float, most: int):
    """Feature loops: rims of carved or sculpted hollows (eyes, mouth, nose) and borders of regions of a
    clearly different colour (crown, stem), the strongest first, at most ``most`` drones. Returns (pos, tri,
    bary, kind, colour, hole label per sample)."""
    hole = np.full(len(surf.p), -1, np.int64)
    V = slicer.verts
    _, vidx = nearest(V, surf.p, 2.0 * surf.spacing)
    cands = []  # (priority, run, colour)
    sources = []
    if FEAT["cavities"]:
        lvl, core = -FEAT["depth"] * d0, -FEAT["core"] * d0
        lab = _regions(surf, surf.relief < lvl, surf.relief < core, (1.4 * a) ** 2, 0.03 * total_area, FEAT["slim"])
        hole = lab
        sources.append((surf.relief, lvl, lab, -1.0, 2.0))
    if FEAT["colors"]:
        q = np.percentile(surf.chroma_off, [50, 80, 98])
        lvl = max(FEAT["chroma"], min(2.0 * float(q[1]), 0.5 * float(q[0] + q[2])))
        lab = _regions(surf, surf.chroma_off > lvl, surf.chroma_off > 1.3 * lvl, (1.5 * a) ** 2, 0.3 * total_area)
        sources.append((surf.chroma_off, lvl, lab, 1.0, 1.0))
    L = surf.lab[:, 0]
    for value, lvl, lab, sign, weight in sources:
        nreg = int(lab.max()) + 1
        if nreg <= 0:
            continue
        # region colour: its brighter samples (the lit walls of a hollow, the gold of a crown)
        rcol = np.ones((nreg, 3))
        for r in range(nreg):
            sel = np.flatnonzero(lab == r)
            top = sel[L[sel] >= np.percentile(L[sel], 70)]
            rcol[r] = surf.lin[top].mean(axis=0)
        # a feature too close to the main colour is lifted towards a warm white, so it stands out
        main_lab = np.median(surf.lab, axis=0)
        rlab = _color.linear_to_oklab(rcol)
        weak = np.linalg.norm((rlab - main_lab)[:, 1:], axis=1) < 0.06
        warm = np.asarray(FEAT["lift"] if sign > 0 else FEAT["lift_hollow"], float)  # a crown warm, a hollow's rim
        rcol = np.where(weak[:, None], 0.35 * rcol / np.maximum(rcol.max(axis=1, keepdims=True), 1e-6) + 0.65 * warm,
                        rcol)
        strength = np.bincount(lab[lab >= 0], weights=np.abs(value[lab >= 0] - lvl), minlength=nreg) * surf.cell
        vlab = lab[vidx]
        fv = value[vidx].copy()
        # the vertices near a kept region: their value smoothed over the samples around them
        rs = np.flatnonzero(lab >= 0)
        _, vj, _ = Grid(V, 4.0 * surf.spacing).query_radius(surf.p[rs], 4.0 * surf.spacing)
        ring = np.unique(vj)
        r = 2.5 * surf.spacing
        qi, pj, d = Grid(surf.p, r).query_radius(V[ring], r)
        w = np.exp(-0.5 * (d / (0.8 * surf.spacing)) ** 2)
        num = np.bincount(qi, weights=w * value[pj], minlength=len(ring))
        den = np.bincount(qi, weights=w, minlength=len(ring))
        fv[ring] = np.where(den > 0, num / np.maximum(den, 1e-12), fv[ring])
        inset = np.zeros(len(V), bool)
        inset[ring] = True
        tri_live = np.flatnonzero(inset[slicer.vid].all(axis=1))
        for loop in slicer.iso(fv[slicer.vid], lvl, tri_live):
            if not loop.closed or loop.length < 2.5 * a:
                continue
            run = resample(loop, 0.1)
            _, si = nearest(run.pos, surf.p, 2.0 * surf.spacing)
            near = lab[si]
            if (near >= 0).mean() < 0.3:
                continue
            reg = int(np.bincount(near[near >= 0]).argmax())
            level = 1.0
            if sign < 0:  # a hollow: part of the face (small, high, facing the audience) or a crease
                rs = lab == reg
                ny = float(surf.n[rs, 1].mean())
                high = float(surf.p[rs, 2].mean()) >= slicer.zlo + FEAT["face_high"] * (slicer.zhi - slicer.zlo)
                small = rs.sum() * surf.cell <= FEAT["face_area"] * d0 * d0
                level = 0.99 if (high and small and ny < -0.3) else FEAT["crease_level"]  # 1.0: colour
            cands.append((weight * strength[reg], run, rcol[reg], (weight, reg), level))
    # every region its longest loop first (strongest region first), then the other loops
    cands.sort(key=lambda x: (-x[0] * x[4], -x[1].length))
    seen, first, rest = set(), [], []
    for cd in cands:
        (rest if cd[3] in seen else first).append(cd)
        seen.add(cd[3])
    cands = first + rest
    P, T, Bc, Kd, C, Lv = [], [], [], [], [], []
    used = 0
    for _, run, col, _, level in cands:
        idx, corner = _loop_drones(run, a, 6)
        if used + len(idx) > most:
            continue
        used += len(idx)
        P.append(run.pos[idx])
        T.append(run.tri[idx])
        Bc.append(run.bary[idx])
        Kd.append(np.where(corner, 0, 1).astype(np.int8))
        C.append(np.repeat(col[None], len(idx), axis=0))
        Lv.append(np.full(len(idx), level))
    if not P:
        z = np.zeros((0, 3))
        return z, np.zeros(0, np.int64), z, np.zeros(0, np.int8), z, hole, np.zeros(0)
    return (np.concatenate(P), np.concatenate(T), np.concatenate(Bc), np.concatenate(Kd), np.concatenate(C), hole,
            np.concatenate(Lv))


RELAX = dict(iters=24, radius=1.0, step=0.25, spring=0.08, views_az=24, views_el=(0.0, 20.0, 45.0), sym=True)


def _view_axes(n_az: int, els):
    out = []
    for e in els:
        for k in range(n_az):
            a, ee = 2.0 * math.pi * k / n_az, math.radians(e)
            d = np.array([math.sin(a) * math.cos(ee), -math.cos(a) * math.cos(ee), math.sin(ee)])
            r = np.cross(d, [0.0, 0.0, 1.0])
            r /= np.linalg.norm(r)
            out.append((r, np.cross(r, d)))
    return out


def relax_rings(runs, run_id, sample, fixed, d0: float, mirror_x=None):
    """Slide the ring drones along their rings so that, seen from many sides (azimuths x elevations), no two
    drones nearly coincide in projection (front and back drones stacking into bright clumps), each kept
    near its place in the structure by a spring and at least ``d0`` from every other drone in 3D.
    Returns the new sample index of every ring drone."""
    n = len(run_id)
    if not n or not RELAX["iters"]:
        return sample
    steps = np.array([r.length / len(r.pos) for r in runs])
    lens = np.array([len(r.pos) for r in runs])
    closed = np.array([r.closed for r in runs])
    offs = np.concatenate([[0], np.cumsum(lens)])
    allpos = np.vstack([r.pos for r in runs])
    st = steps[run_id]
    m = lens[run_id]
    s0 = sample.astype(np.float64) * st  # arc position
    s = s0.copy()
    L = m * st
    views = _view_axes(RELAX["views_az"], RELAX["views_el"])
    rho = RELAX["radius"]
    nf = len(fixed)
    cl = closed[run_id]

    def where(arc):
        k = arc / st
        k0 = np.floor(k).astype(np.int64)
        f = (k - k0)[:, None]
        a0 = np.where(cl, k0 % m, np.clip(k0, 0, m - 1))
        a1 = np.where(cl, (k0 + 1) % m, np.clip(k0 + 1, 0, m - 1))
        p0, p1 = allpos[offs[run_id] + a0], allpos[offs[run_id] + a1]
        return p0 + f * (p1 - p0), p1 - p0

    for it in range(RELAX["iters"]):
        p, tang = where(s)
        tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12
        every = np.vstack([p, fixed]) if nf else p
        F = np.zeros((n, 3))
        for r, u in views:
            q = np.stack([every @ r, every @ u, np.zeros(len(every))], 1)
            qi, pj, d = Grid(q, rho).query_radius(q[:n], rho, exclude_self=True)
            if not len(qi):
                continue
            w = 1.0 - d / rho
            e2 = (q[qi] - q[pj])[:, :2]
            tiny = d < 1e-6
            e2[tiny] = np.stack([np.where(qi[tiny] < pj[tiny], 1.0, -1.0), np.zeros(int(tiny.sum()))], 1)
            e2 /= np.linalg.norm(e2, axis=1, keepdims=True)
            f3 = e2[:, :1] * r + e2[:, 1:] * u
            np.add.at(F, qi, w[:, None] * f3)
        # never closer than d0 in 3D
        qi, pj, d = Grid(every, 1.1 * d0).query_radius(p, 1.1 * d0, exclude_self=True)
        if len(qi):
            w = 4.0 * (1.0 - d / (1.1 * d0))
            dirs = (p[qi] - every[pj]) / np.maximum(d, 1e-9)[:, None]
            np.add.at(F, qi, w[:, None] * dirs)
        ds = (F * tang).sum(axis=1) * RELAX["step"]
        if mirror_x is not None:
            partner, sign = mirror_x
            has = partner >= 0
            # a drone and its mirror image move as mirror images: the figure stays symmetric
            ds[has] = 0.5 * (ds[has] + sign[has] * ds[partner[has]])
        ds -= RELAX["spring"] * (s - s0)
        ds = np.clip(ds, -0.3, 0.3)
        s = s + ds
        s = np.where(cl, np.mod(s, L), np.clip(s, 0, L - st))
    return np.round(s / st).astype(np.int64) % m


def _mirror_pairs(pos, runs, run_id, sample, tol):
    """For every ring drone the ring drone at its mirror image in x = centre (-1: none), and the sign that
    maps an arc move of one onto the other (mirrored rings run the opposite way)."""
    c = 0.5 * (pos[:, 0].min() + pos[:, 0].max())
    q = pos.copy()
    q[:, 0] = 2 * c - q[:, 0]
    d, j = nearest(q, pos, tol)
    partner = np.where(d < tol, j, -1)
    sign = np.zeros(len(pos))
    for i in np.flatnonzero(partner >= 0):
        k = partner[i]
        ra, rb = runs[run_id[i]], runs[run_id[k]]
        na, nb = len(ra.pos), len(rb.pos)
        ta = ra.pos[(sample[i] + 1) % na] - ra.pos[(sample[i] - 1) % na]
        tb = rb.pos[(sample[k] + 1) % nb] - rb.pos[(sample[k] - 1) % nb]
        ta[0] = -ta[0]
        sign[i] = 1.0 if ta @ tb >= 0 else -1.0
    partner[partner == np.arange(len(pos))] = -1
    return partner, sign


EYE_COLOR = np.array([0.85, 0.95, 1.0])  # cool white: apart from a gold crown


def _snap(surf, pts):
    _, si = nearest(pts, surf.p, 2.0 * surf.spacing)
    return surf.p[si], surf.t[si], surf.b[si]


EYE_SCALES = (1.0, 1.33, 1.67, 2.0)  # sculpted eyeballs looked for in the front picture at these sizes (x d0)


def find_eyes_front(surf, mesh, opt, d0):
    """Sculpted eyes as the audience in front sees them: the strongest pair of eyeballs (small bumps side by
    side, mirrored, ``picture._bumps``) in the picture from the front, at a few sizes, in the upper half of
    the figure. Each is drawn as ``find_eyes`` draws it. Returns (pos, tri, bary) or None."""
    from . import picture as _pic
    tris = mesh.triangles
    zlo, zhi = float(tris[..., 2].min()), float(tris[..., 2].max())
    best = None
    for k in EYE_SCALES:
        s = k * d0
        try:
            pic = _pic.Picture(mesh, np.array([0, len(tris)]), opt, s / _pic.PX)
        except ValueError:
            return None
        found = _pic._bumps(pic, s)
        for i in range(0, len(found) - 1, 2):
            pair = found[i:i + 2]
            pos = []
            for xy, _, part, _ in pair:
                p, _, _, ok = pic.lift(np.asarray(xy)[None], part, 3.0)
                if ok[0]:
                    pos.append(p[0])
            if len(pos) < 2 or min(p[2] for p in pos) < zlo + 0.5 * (zhi - zlo):
                continue
            strength = min(f[1] for f in pair)
            if best is None or strength > best[0]:
                best = (strength, np.array(pos))
            break
    if best is None:
        return None
    out = []
    rad = 1.08 * d0  # the ring's neighbours and its pupil all a minimum distance apart
    _, si = nearest(best[1], surf.p, 2.0 * surf.spacing)
    for p0, n0 in zip(best[1], surf.n[si]):
        side = np.cross([0.0, 0.0, 1.0], n0)
        side /= np.linalg.norm(side) + 1e-12
        up = np.cross(n0, side)
        if FEAT["pupil"]:
            out.append(p0)
        for t in np.arange(6) * (math.pi / 3.0):
            out.append(p0 + rad * (math.cos(t) * side + math.sin(t) * up))
    return _snap(surf, np.array(out))


def find_eyes(surf, slicer, facing, d0, avoid):
    """Sculpted eyes: the deepest pair of small hollows (fine relief) mirrored about the middle, level,
    facing the audience, in the upper part of the figure and not on a colour border (a crown rim). Each
    eye is drawn as a ring of 6 drones round a pupil, big enough to read from far away. Returns (pos, tri,
    bary) or None."""
    h = surf.fine
    r = 0.8 * d0
    qi, pj, _ = Grid(surf.p, r).query_radius(surf.p, r)
    lower = np.zeros(len(h), bool)
    np.logical_or.at(lower, qi, h[pj] < h[qi])
    H = slicer.zhi - slicer.zlo
    face = (surf.n @ np.asarray(facing, float)) > 0.3
    pit = np.flatnonzero(~lower & (h < -0.08 * d0) & face & (surf.p[:, 2] >= slicer.zlo + 0.7 * H))
    if len(avoid) and len(pit):
        dd, _ = nearest(surf.p[pit], avoid, 3.0 * d0)
        pit = pit[dd > 2.0 * d0]
    if len(pit) < 2:
        return None
    c = 0.5 * (surf.p[:, 0].min() + surf.p[:, 0].max())
    P = surf.p[pit]
    best = None
    for i in range(len(pit)):
        d = P - P[i]
        ok = ((np.abs(0.5 * (P[:, 0] + P[i, 0]) - c) < 0.7 * d0) & (np.abs(d[:, 2]) < 0.8 * d0)
              & (np.abs(d[:, 1]) < 0.8 * d0) & (np.abs(d[:, 0]) > 2.0 * d0))
        if not ok.any():
            continue
        cut = [lp.pts for lp in slicer.cut(float(P[i, 2]))]
        width = float(np.ptp(np.vstack(cut)[:, 0])) if cut else 0.0
        ok &= np.abs(d[:, 0]) < 0.55 * width  # the eyes sit inside the head, not at its sides
        for j in np.flatnonzero(ok):
            depth = max(h[pit[i]], h[pit[j]])
            if best is None or depth < best[0]:
                best = (depth, i, j)
    if best is None or best[0] > -0.2 * d0:
        return None
    out = []
    rad = 1.08 * d0  # the ring's neighbours and its pupil all a minimum distance apart
    for k in (best[1], best[2]):
        p0, n0 = P[k], surf.n[pit[k]]
        side = np.cross([0.0, 0.0, 1.0], n0)
        side /= np.linalg.norm(side) + 1e-12
        up = np.cross(n0, side)
        if FEAT["pupil"]:
            out.append(p0)
        for t in np.arange(6) * (math.pi / 3.0):
            out.append(p0 + rad * (math.cos(t) * side + math.sin(t) * up))
    return _snap(surf, np.array(out))


def find_tips(surf, slicer, d0):
    """Tips of spikes and points (crown points, ears, wing tips): the highest point within a few metres,
    on a thin top. Each tip becomes a small ball: 3 drones round it a little below the top, and the top.
    Returns (pos, tri, bary)."""
    z = surf.p[:, 2]
    r = 2.0 * d0
    qi, pj, _ = Grid(surf.p, r).query_radius(surf.p, r)
    higher = np.zeros(len(z), bool)
    np.logical_or.at(higher, qi, z[pj] > z[qi])
    top = np.flatnonzero(~higher & (z >= slicer.zlo + 0.5 * (slicer.zhi - slicer.zlo)))
    top = top[np.argsort(-z[top])][:40]
    pts = []
    drop = 0.6 * d0
    for k in top:
        p = surf.p[k]
        loops = [lp for lp in slicer.cut(float(p[2] - drop)) if lp.closed]
        best = None
        for lp in loops:
            dd = np.linalg.norm(lp.pts[:, :2] - p[:2], axis=1).min()
            if dd < 1.5 * d0 and (best is None or dd < best[0]):
                best = (dd, lp)
        if best is None or not best[1].length <= 8.0 * d0:
            continue  # no thin top: a dome, a shoulder
        lp = best[1]
        ctr = lp.pts.mean(axis=0)
        rho = max(0.62 * d0, lp.length / (2 * math.pi))  # a ball of 3 drones round the tip and one on top
        for th in (-0.5 * math.pi, math.pi / 6, 5 * math.pi / 6):
            pts.append(ctr + rho * np.array([math.cos(th), math.sin(th), 0.0]))
        pts.append(ctr + np.array([0.0, 0.0, math.sqrt(max(d0 * d0 * 1.03 - rho * rho, 0.0)) + 0.05]))
    if not pts:
        z3 = np.zeros((0, 3))
        return z3, np.zeros(0, np.int64), z3
    pts = np.array(pts)
    _, t, b = _snap(surf, pts)
    return pts, t, b


def smooth_lines(runs, rid, smp, level, angle, g, passes=3):
    """The direction lines straightened: every drone moved along its ring towards the mean of its
    neighbours one ring up and one down on the same line (no wobble), then put back on its ring."""
    n = len(rid)
    pos = np.stack([runs[r].pos[k] for r, k in zip(rid, smp)])
    key = {}
    for i in range(n):
        if not np.isnan(angle[i]):
            key.setdefault((int(level[i]), round(float(angle[i]), 4)), []).append(i)
    up = np.full(n, -1)
    down = np.full(n, -1)
    for (lv, an), ids in key.items():
        nxt = key.get((lv + 1, an))
        if not nxt:
            continue
        for i in ids:
            d = np.linalg.norm(pos[nxt] - pos[i], axis=1)
            j = int(np.argmin(d))
            if d[j] < 2.5 * g and down[nxt[j]] < 0:
                up[i], down[nxt[j]] = nxt[j], i
    target = pos.copy()
    for _ in range(passes):
        both = (up >= 0) & (down >= 0)
        t = target.copy()
        t[both, :2] = 0.5 * target[both, :2] + 0.25 * (target[up[both], :2] + target[down[both], :2])
        target = t
    out = smp.copy()
    for i in np.flatnonzero((up >= 0) & (down >= 0)):
        run = runs[rid[i]]
        m = len(run.pos)
        w = max(int(1.5 * g / max(run.length / m, 1e-9)), 1)  # stay near: never jump to another part of a ring
        cand = (smp[i] + np.arange(-w, w + 1)) % m if run.closed else np.clip(smp[i] + np.arange(-w, w + 1), 0, m - 1)
        out[i] = cand[int(np.argmin(np.linalg.norm(run.pos[cand, :2] - target[i, :2], axis=1)))]
    return out


def _mirror_features(surf, fp, ft, fb, fk, fc, a, height):
    """On a mirror-symmetric model (x = centre), a feature found on one side only (an arm crease, an eye)
    is drawn on the other side too: its drones mirrored onto the surface."""
    if not len(fp):
        return fp, ft, fb, fk, fc
    c = 0.5 * (surf.p[:, 0].min() + surf.p[:, 0].max())
    q = surf.p.copy()
    q[:, 0] = 2 * c - q[:, 0]
    d, _ = nearest(q[::7], surf.p, 0.02 * height)
    if float(np.median(d)) > 0.01 * height:  # not symmetric
        return fp, ft, fb, fk, fc
    m = fp.copy()
    m[:, 0] = 2 * c - m[:, 0]
    dd, _ = nearest(m, fp, a)
    new = np.flatnonzero(dd > 0.8 * a)
    if not len(new):
        return fp, ft, fb, fk, fc
    ds, si = nearest(m[new], surf.p, 0.05 * height)
    new, si = new[ds < 0.5 * a], si[ds < 0.5 * a]
    return (np.vstack([fp, surf.p[si]]), np.concatenate([ft, surf.t[si]]), np.vstack([fb, surf.b[si]]),
            np.concatenate([fk, fk[new]]), np.vstack([fc, fc[new]]))


def around(parts, options=None) -> DroneResult:
    opt = options or DistributeOptions()
    budget = int(opt.count)
    mesh = MeshData.concatenate([p.mesh for p in parts])
    tris = mesh.triangles
    d0 = float(opt.target_min_distance)
    rng = np.random.default_rng(int(opt.seed))
    areas, normals = triangle_areas_normals(tris)
    total = float(areas.sum())
    slicer = Slicer(tris, opt.up_axis)
    rings = Rings(slicer, CFG["step"])
    surf = Surface(mesh, tris, areas, normals, d0, rng)
    c = CFG

    # features first, at most a share of the budget
    a = FEAT["spacing"] * d0
    fp, ft, fb, fk, fc, hole, fl = features(surf, slicer, total, a, d0, int(FEAT["share"] * budget))
    fc = np.hstack([fc, fl[:, None]])  # the brightness travels with the colour
    fp, ft, fb, fk, fc = _mirror_features(surf, fp, ft, fb, fk, fc, a, slicer.zhi - slicer.zlo)
    extra = []
    facing = np.asarray(opt.view_dir, float)  # model -> audience: the face looks this way
    eyes = None
    if FEAT["eyes"] and FEAT.get("eyes_front"):
        eyes = find_eyes_front(surf, mesh, opt, d0)
    if eyes is None and FEAT["eyes"]:
        eyes = find_eyes(surf, slicer, facing / (np.linalg.norm(facing) + 1e-12), d0,
                         fp[fc[:, 3] >= 1.0] if len(fp) else fp)  # off colour rims
    if eyes is not None:
        extra.append((eyes, np.append(EYE_COLOR, 1.0)))
    if FEAT["tips"]:
        tp = find_tips(surf, slicer, d0)
        if len(tp[0]):
            _, si = nearest(tp[0], surf.p, 2.0 * surf.spacing)
            col = surf.lin[si] / np.maximum(surf.lin[si].max(axis=1, keepdims=True), 1e-6)
            extra.append((tp, np.hstack([col, np.ones((len(col), 1))])))
    for (ep, et, eb), ec in extra:  # eyes and tips first: they keep their places
        ec = np.broadcast_to(ec, (len(ep), 4))
        fp, ft, fb = np.vstack([ep, fp]), np.concatenate([et, ft]), np.vstack([eb, fb])
        fk, fc = np.concatenate([np.zeros(len(ep), np.int8), fk]), np.vstack([ec, fc])
    keep = park(fp, d0)
    fp, ft, fb, fk, fc = fp[keep], ft[keep], fb[keep], fk[keep], fc[keep]
    fl, fc = fc[:, 3], fc[:, :3]

    def make(g, tau=None, K=None, fill=None):
        tau = c["tau"] if tau is None else tau
        fill = c["fill"] if fill is None else fill
        rp, rt, rb, rk = rings.build(g, K or c["K"], c["stagger"], None if fill is None else fill * g,
                                     c["deep"] * g, c["min_arc"] * d0, None if tau is None else tau * g,
                                     c["smooth"] * g, None if c["rows"] is None else c["rows"] * g)
        rid, smp = rings.run_id, rings.sample
        ok = np.ones(len(rp), bool)
        if (hole >= 0).any():  # no ring drone inside a hollow
            _, si = nearest(rp, surf.p, 2.0 * surf.spacing)
            ok &= hole[si] < 0
        if len(fp):  # ring drones make room for the features: a face is drawn by its features alone
            reach = max(FEAT["clear"] * min(g, 2.0 * a), FEAT["clear_abs"] * d0, d0)
            dd, _ = nearest(rp, fp, 2.0 * reach)
            ok &= dd >= np.where(rk == 2, 2.0 * reach, reach)  # the drones between the lines keep farther off
        if eyes is not None and FEAT["eye_clear"] > 1.0:  # a dark ring round each eye: it reads as an eye
            de, _ = nearest(rp, eyes[0], 2.0 * FEAT["eye_clear"] * d0)
            ok &= de >= FEAT["eye_clear"] * d0
        sel = np.flatnonzero(ok)
        if len(sel) > 2 and FEAT["stray"]:  # a lone drone of a tiny ring with no neighbour reads as noise
            nn = nn_distances(np.vstack([rp[sel], fp]), g)[: len(sel)]
            alone = np.bincount(rid[sel], minlength=int(rid.max()) + 1)[rid[sel]] <= 2
            sel = sel[~alone | (nn <= FEAT["stray"] * g)]
        sel = sel[np.argsort(rk[sel], kind="stable")]  # the outline drones before the drones between them
        kept = park(np.vstack([fp, rp[sel]]), d0)
        sel = sel[kept[kept >= len(fp)] - len(fp)]
        return (rp[sel], rt[sel], rb[sel], rk[sel], rid[sel], smp[sel], list(rings.used), rings.level[sel],
                rings.angle[sel])

    room = budget - len(fp)
    gmin = c["g0"] * d0 if c["rows"] is None else max(c["g0"] * d0, 1.05 * d0 / c["rows"])
    res = make(gmin)
    needed = len(res[0]) + len(fp)
    g = gmin
    if len(res[0]) > room:
        lo, hi = gmin, gmin * math.sqrt(len(res[0]) / max(room, 1)) * 1.3
        while len(make(hi)[0]) > room:
            lo, hi = hi, hi * 1.4
        for _ in range(8):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if len(make(mid)[0]) > room else (lo, mid)
        g = hi
        res = make(g)
    if g == gmin and len(res[0]) < 0.95 * room and c["rows"] is None:
        # room to spare at the closest spacing: more lines round every ring (a finer outline from every
        # side), then drones along the rings between them
        lo, hi = 0.02, c["tau"]
        if len(make(g, lo, 2 * c["K"])[0]) > room:
            for _ in range(7):
                mid = math.sqrt(lo * hi)
                lo, hi = (mid, hi) if len(make(g, mid, 2 * c["K"])[0]) > room else (lo, mid)
            res = make(g, hi, 2 * c["K"])
        else:
            lo, hi = FEAT["fill_min"], 12.0
            if len(make(g, 0.02, 2 * c["K"], lo)[0]) > room:
                for _ in range(7):
                    mid = math.sqrt(lo * hi)
                    lo, hi = (lo, mid) if len(make(g, 0.02, 2 * c["K"], mid)[0]) <= room else (mid, hi)
                res = make(g, 0.02, 2 * c["K"], hi)
            else:
                res = make(g, 0.02, 2 * c["K"], lo)
    rp, rt, rb, rk, rid, smp, runs, lvl, ang = res
    rk = np.where(rk == 0, 1, rk)  # a single drone of a ring is no feature corner
    if RELAX["iters"] and len(rp) and c["rows"] is None:  # rows stay even
        mir = _mirror_pairs(rp, runs, rid, smp, 0.25 * d0) if RELAX["sym"] else None
        smp = relax_rings(runs, rid, smp, fp, d0, mir)
    if FEAT["smooth_lines"] and len(rp) and c["rows"] is None:
        smp = smooth_lines(runs, rid, smp, lvl, ang, g)
    if len(rp):
        rp = np.stack([runs[r].pos[k] for r, k in zip(rid, smp)])
        rt = np.array([runs[r].tri[k] for r, k in zip(rid, smp)])
        rb = np.stack([runs[r].bary[k] for r, k in zip(rid, smp)])
    pos = np.vstack([fp, rp])
    tri = np.concatenate([ft, rt])
    bary = np.vstack([fb, rb])
    kind = np.concatenate([fk, rk]).astype(np.int8)
    feat = np.zeros(len(pos), bool)
    feat[: len(fp)] = True
    order = np.argsort(np.where(feat, kind, kind + 3), kind="stable")
    pos, tri, bary, kind, feat = pos[order], tri[order], bary[order], kind[order], feat[order]
    keep = park(pos, d0)[:budget]
    pos, tri, bary, kind, feat = pos[keep], tri[keep], bary[keep], kind[keep], feat[keep]
    _, nrm = triangle_areas_normals(tris[tri])
    col = _colors(mesh, tris, pos, tri, bary, 0.5 * g, rng)
    fcol = np.vstack([fc, np.zeros((len(rp), 3))])[order][keep]
    col = np.where(feat[:, None], fcol, col)
    col = _color.glow(col, opt.min_brightness)
    s = _color.linear_to_srgb(col)
    s = s / np.maximum(s.max(axis=1, keepdims=True), 1e-6)  # full brightness of the hue ...
    flev = np.concatenate([fl, np.full(len(rp), FEAT["ring_level"])])[order][keep]
    level = np.where(feat, flev, FEAT["ring_level"])  # ... face and crown bright, outline less, creases dim
    col = _color.srgb_to_linear(s * level[:, None])
    dmin, dmean = spacing(pos, g)
    return DroneResult(positions=pos, normals=nrm, triangle_ids=tri, barycentrics=bary, colors_linear=col,
                       min_distance=dmin, mean_distance=dmean, area=total, options=opt, outline=kind < 2, kind=kind,
                       part=np.zeros(len(pos), np.int64), budget=budget, needed=needed, spread=g / d0,
                       fill_spread=g / d0)


def around_figure(parts, options=None) -> DroneResult:
    """``around`` for figures (characters, animals): the face (eyes, muzzle), crown and spike tips as bright
    features, the outline lines from every side, creases dim."""
    return around(parts, options)
