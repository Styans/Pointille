"""Burst: a wave set off at the core of a model runs through its inside at one speed.

It goes round corners (an arm is reached through the shoulder, not through the air), so it reaches every
part of the surface. When it reaches each point tells the parts of the model apart:
- cut into bands of equal arrival time, the wave front is a set of pieces, each a slice across one part.
  A piece whose centre moves sideways from band to band (the front runs along a tail, a wing, a lying
  body) belongs to a sideways part; the growing shell round the core, a trunk, hanging arms and legs are
  upright (``Parts``);
- where the wave arrives last at the end of a part is a tip: a paw, the tip of a tail, a crown spike.

The inside is a grid of voxels (a vote of the ray parity along x, y and z, so a hole in a mesh spoils one
direction, not three); the wave is the shortest path through the 26 neighbours of each voxel. numpy only.
"""

from __future__ import annotations

import math

import numpy as np

from .outline import _components, park
from .spatial import Grid, _expand_ranges

CFG = dict(voxel=1.0 / 3.0, max_voxels=220, core_depth=0.85, band=0.5, moving=0.7, side_cos=0.64, flat=0.2,
           thin=4.0, smooth=3, tip_reach=3.0, tip_drop=1.5, tip_len=1.5, tip_cell=0.5)
_RASTER = 1 << 22
_STEPS = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) > (0, 0, 0)]
_NEAR = sorted(((dx, dy, dz) for dx in range(-3, 4) for dy in range(-3, 4) for dz in range(-3, 4)
                if dx * dx + dy * dy + dz * dz <= 9), key=lambda o: o[0] ** 2 + o[1] ** 2 + o[2] ** 2)


def _parity(tris: np.ndarray, axis: int, lo: np.ndarray, v: float, shape) -> np.ndarray:
    """Inside by the parity of the surface crossings along ``axis`` below every voxel centre."""
    a1, a2 = [k for k in range(3) if k != axis]
    n1, n2, n = shape[a1], shape[a2], shape[axis]
    e1, e2 = 0.1234e-3 * v, 0.2345e-3 * v  # rays off the vertices and edges of a mesh on the grid
    x, y, z = tris[:, :, a1], tris[:, :, a2], tris[:, :, axis]
    i0 = np.clip(np.ceil((x.min(axis=1) - lo[a1] - e1) / v), 0, n1 - 1).astype(np.int64)
    i1 = np.clip(np.floor((x.max(axis=1) - lo[a1] - e1) / v), 0, n1 - 1).astype(np.int64)
    j0 = np.clip(np.ceil((y.min(axis=1) - lo[a2] - e2) / v), 0, n2 - 1).astype(np.int64)
    j1 = np.clip(np.floor((y.max(axis=1) - lo[a2] - e2) / v), 0, n2 - 1).astype(np.int64)
    nx, ny = np.maximum(i1 - i0 + 1, 0), np.maximum(j1 - j0 + 1, 0)
    count = nx * ny
    todo = np.flatnonzero(count)
    total = np.cumsum(count[todo])
    flips = np.zeros((n1, n2, n + 1), np.int32)
    start = 0
    while start < len(todo):
        stop = max(int(np.searchsorted(total, (total[start - 1] if start else 0) + _RASTER, side="right")),
                   start + 1)
        t = todo[start:stop]
        c = count[t]
        owner = np.repeat(np.arange(len(t)), c)
        k = np.arange(int(c.sum())) - np.repeat(np.cumsum(c) - c, c)
        t = t[owner]
        ci = i0[t] + k % nx[t]
        cj = j0[t] + k // nx[t]
        px = lo[a1] + ci * v + e1
        py = lo[a2] + cj * v + e2
        ax, ay, bx, by, qx, qy = x[t, 0], y[t, 0], x[t, 1], y[t, 1], x[t, 2], y[t, 2]
        den = (by - qy) * (ax - qx) + (qx - bx) * (ay - qy)
        good = np.abs(den) > 1e-18
        den = np.where(good, den, 1.0)
        l0 = ((by - qy) * (px - qx) + (qx - bx) * (py - qy)) / den
        l1 = ((qy - ay) * (px - qx) + (ax - qx) * (py - qy)) / den
        l2 = 1.0 - l0 - l1
        hit = good & (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        zc = l0 * z[t, 0] + l1 * z[t, 1] + l2 * z[t, 2]
        kk = np.clip(np.ceil((zc - lo[axis]) / v), 0, n).astype(np.int64)
        np.add.at(flips, (ci[hit], cj[hit], kk[hit]), 1)
        start = stop
    inside = (np.cumsum(flips, axis=2)[:, :, :n] % 2).astype(bool)
    return np.moveaxis(inside, 2, axis)


def voxelize(tris: np.ndarray, v: float):
    """Voxels inside the model. Returns (inside, lo); voxel (i, j, k) is centred on lo + (i, j, k) v."""
    c = tris.reshape(-1, 3)
    lo = c.min(axis=0) - 2 * v
    shape = tuple(int(s) for s in np.ceil((c.max(axis=0) + 2 * v - lo) / v).astype(np.int64) + 1)
    votes = sum(_parity(tris, a, lo, v, shape).astype(np.uint8) for a in range(3))
    return votes >= 2, lo


def _spread(n: int, csr, sources, start=None) -> np.ndarray:
    """Shortest path lengths from ``sources`` along the weighted edges ``csr`` (ptr, to, w): the front
    moves one edge per round, a node is taken up again whenever a shorter way reaches it."""
    ptr, to, w = csr
    dist = np.full(n, np.inf) if start is None else start.copy()
    sources = np.asarray(sources, np.int64)
    dist[sources] = 0.0
    front = sources
    while len(front):
        u, pos = _expand_ranges(ptr[front], ptr[front + 1] - ptr[front])
        u = front[u]
        cand = dist[u] + w[pos]
        v = to[pos]
        better = cand < dist[v]
        if not better.any():
            break
        v, cand = v[better], cand[better]
        before = dist.copy()
        np.minimum.at(dist, v, cand)
        front = np.unique(v[dist[v] < before[v]])
    return dist


class Body:
    """The inside of a model as voxels, and the time the wave from its core takes to each of them."""

    def __init__(self, tris: np.ndarray, d0: float, voxel: float = None):
        tris = np.asarray(tris, dtype=np.float64)
        ext = float(np.ptp(tris.reshape(-1, 3), axis=0).max())
        self.v = v = voxel or max(CFG["voxel"] * d0, ext / CFG["max_voxels"])
        self.inside, self.lo = voxelize(tris, v)
        shape = self.inside.shape
        flat = np.flatnonzero(self.inside.ravel())
        self.n = n = len(flat)
        self.node = np.full(self.inside.size, -1, np.int64)
        self.node[flat] = np.arange(n)
        self.ijk = np.stack(np.unravel_index(flat, shape), axis=1)
        strides = np.array([shape[1] * shape[2], shape[2], 1])
        a, b, w = [], [], []
        for step in _STEPS:  # the inside never touches the border of the grid: no wrap round
            q = self.node[flat + int(np.dot(step, strides))]
            ok = q >= 0
            a.append(np.flatnonzero(ok))
            b.append(q[ok])
            w.append(np.full(int(ok.sum()), v * math.sqrt(sum(s * s for s in step))))
        self.a, self.b, w = np.concatenate(a), np.concatenate(b), np.concatenate(w)
        self.T = np.full(n, np.inf)
        self.core = -1
        if not n:
            return
        src, dst = np.concatenate([self.a, self.b]), np.concatenate([self.b, self.a])
        order = np.argsort(src, kind="stable")
        ptr = np.zeros(n + 1, np.int64)
        np.cumsum(np.bincount(src, minlength=n), out=ptr[1:])
        csr = (ptr, dst[order], np.concatenate([w, w])[order])
        # the core: deep inside (far from the voxels on the skin), of the deepest the nearest to the middle of
        # the body (a big head is nearly as deep as the chest)
        skin = np.flatnonzero(np.diff(ptr) < 26)
        depth = _spread(n, csr, skin)
        deep = np.flatnonzero(depth >= CFG["core_depth"] * depth.max())
        mid = self.ijk.mean(axis=0)
        self.core = int(deep[np.argmin(np.linalg.norm(self.ijk[deep] - mid, axis=1))])
        self.T = _spread(n, csr, [self.core])

    def centre(self, nodes: np.ndarray) -> np.ndarray:
        return self.lo + self.ijk[nodes] * self.v

    def node_at(self, p: np.ndarray) -> np.ndarray:
        """The nearest inside voxel of each point on or near the surface (-1 where none is within 3 voxels)."""
        p = np.asarray(p, dtype=np.float64)
        out = np.full(len(p), -1, np.int64)
        if not self.n or not len(p):
            return out
        base = np.rint((p - self.lo) / self.v).astype(np.int64)
        shape = np.array(self.inside.shape)
        todo = np.arange(len(p))
        for off in _NEAR:
            q = base[todo] + off
            ok = np.all((q >= 0) & (q < shape), axis=1)
            k = np.full(len(todo), -1, np.int64)
            k[ok] = self.node[np.ravel_multi_index(q[ok].T, self.inside.shape)]
            found = k >= 0
            out[todo[found]] = k[found]
            todo = todo[~found]
            if not len(todo):
                break
        return out

    def at(self, p: np.ndarray) -> np.ndarray:
        """When the wave reaches points on or near the surface (inf where it does not)."""
        p = np.asarray(p, dtype=np.float64)
        k = self.node_at(p)
        t = np.full(len(p), np.inf)
        ok = k >= 0
        t[ok] = self.T[k[ok]] + np.linalg.norm(p[ok] - self.centre(k[ok]), axis=1)
        return t


class Parts:
    """The pieces of the wave front, band by band, and which of them are slices of sideways parts."""

    def __init__(self, body: Body, d0: float):
        self.body, self.d0 = body, d0
        self.step = step = CFG["band"] * d0
        n = body.n
        ok = np.isfinite(body.T)
        band = np.where(ok, np.floor(np.where(ok, body.T, 0.0) / step), -1).astype(np.int64)
        a, b = body.a, body.b
        both = ok[a] & ok[b]
        same = both & (band[a] == band[b])
        root = _components(n, a[same], b[same])
        _, piece = np.unique(root, return_inverse=True)
        piece = np.where(ok, piece, -1)
        self.piece = piece
        nc = int(piece.max()) + 1 if n else 0
        live = np.flatnonzero(ok)
        cid = piece[live]
        cnt = np.bincount(cid, minlength=nc).astype(float)
        x = body.ijk[live].astype(float)
        cen = np.stack([np.bincount(cid, x[:, d], nc) for d in range(3)], axis=1) / np.maximum(cnt, 1)[:, None]
        # links between pieces of neighbouring bands: (earlier, later)
        up = both & (band[b] == band[a] + 1)
        down = both & (band[a] == band[b] + 1)
        pairs = np.vstack([np.stack([piece[a[up]], piece[b[up]]], 1), np.stack([piece[b[down]], piece[a[down]]], 1)])
        pairs = np.unique(pairs, axis=0) if len(pairs) else np.zeros((0, 2), np.int64)
        e, l = pairs[:, 0], pairs[:, 1]
        self.nch = np.bincount(e, minlength=nc)
        self.npar = np.bincount(l, minlength=nc)
        # how the centre of a piece moves: the mean of the pieces after it minus the mean of those before
        ahead = np.zeros((nc, 3))
        np.add.at(ahead, e, cen[l])
        behind = np.zeros((nc, 3))
        np.add.at(behind, l, cen[e])
        ahead = np.where(self.nch[:, None] > 0, ahead / np.maximum(self.nch, 1)[:, None], cen)
        behind = np.where(self.npar[:, None] > 0, behind / np.maximum(self.npar, 1)[:, None], cen)
        span = (self.nch > 0).astype(float) + (self.npar > 0)
        disp = (ahead - behind) / np.maximum(span, 1)[:, None]
        length = np.linalg.norm(disp, axis=1)
        speed = length / (step / body.v)
        axis = disp / np.maximum(length, 1e-9)[:, None]
        # a slice across a part is flat; the shell growing round the core inside a fat body is curved
        rel = x - cen[cid]
        cov = np.zeros((nc, 3, 3))
        for i in range(3):
            for j in range(3):
                cov[:, i, j] = np.bincount(cid, rel[:, i] * rel[:, j], nc) / np.maximum(cnt, 1)
        lam = np.linalg.eigvalsh(cov) if nc else np.zeros((0, 3))
        flat = (lam[:, 0] <= CFG["flat"] * lam[:, 1]) | (lam[:, 1] < CFG["thin"])
        side = (speed >= CFG["moving"]) & (np.abs(axis[:, 2]) < CFG["side_cos"]) & (cnt >= 4) & flat
        # a slice one or two voxels thick flickers: the pieces before and after vote too, by size
        score = side.astype(float)
        for _ in range(CFG["smooth"]):
            acc, wsum = score * cnt, cnt.copy()
            np.add.at(acc, e, score[l] * cnt[l])
            np.add.at(wsum, e, cnt[l])
            np.add.at(acc, l, score[e] * cnt[e])
            np.add.at(wsum, l, cnt[e])
            score = acc / np.maximum(wsum, 1)
        self.side = score >= 0.5
        self.parent = np.full(nc, -1, np.int64)
        one = self.npar[l] == 1
        self.parent[l[one]] = e[one]
        # the last pieces of a part: none follows, or only last pieces do (the corners of an end face are
        # reached last, apart), two bands deep
        self.end = self.nch == 0
        for _ in range(2):
            self.end = np.bincount(e, weights=~self.end[l], minlength=nc) == 0

    def piece_at(self, p: np.ndarray) -> np.ndarray:
        k = self.body.node_at(p)
        return np.where(k >= 0, self.piece[np.maximum(k, 0)], -1)

    def sideways_at(self, p: np.ndarray) -> np.ndarray:
        c = self.piece_at(p)
        return (c >= 0) & self.side[np.maximum(c, 0)] if len(self.side) else np.zeros(len(c), bool)

    def _branch(self, c: int) -> float:
        """How far back from piece ``c`` its part runs on its own (one piece per band), in metres."""
        n = 0
        while self.npar[c] == 1 and n < 1000:
            p = self.parent[c]
            if p < 0 or self.nch[p] != 1:
                break
            c, n = p, n + 1
        return (n + 1) * self.step

    def tips(self, pts: np.ndarray) -> np.ndarray:
        """Tips among surface points ``pts``: where the wave arrives last (within ``tip_reach`` d0, at least
        ``tip_drop`` d0 later than the earliest there) at the end of a part (one of its last pieces, the
        part at least ``tip_len`` d0 long on its own), ``tip_reach`` d0 apart."""
        d0 = self.d0
        pts = np.asarray(pts, dtype=np.float64)
        if not len(pts):
            return np.zeros((0, 3))
        # one point per cell of ``tip_cell`` d0 is enough
        key = np.floor((pts - pts.min(axis=0)) / (CFG["tip_cell"] * d0)).astype(np.int64)
        _, first = np.unique(key, axis=0, return_index=True)
        pts = pts[np.sort(first)]
        t = self.body.at(pts)
        ok = np.isfinite(t)
        pts, t = pts[ok], t[ok]
        if not len(pts):
            return np.zeros((0, 3))
        r = CFG["tip_reach"] * d0
        qi, pj, _ = Grid(pts, r).query_radius(pts, r)
        hi = t.copy()
        lo = t.copy()
        np.maximum.at(hi, qi, t[pj])
        np.minimum.at(lo, qi, t[pj])
        top = np.flatnonzero((t >= hi - 1e-9) & (t - lo >= CFG["tip_drop"] * d0))
        top = top[np.argsort(-t[top], kind="stable")]
        top = top[park(pts[top], r)]
        c = self.piece_at(pts[top])
        end = [i for i, k in zip(top, c) if k >= 0 and self.end[k] and self._branch(k) >= CFG["tip_len"] * d0]
        return pts[np.asarray(end, np.int64)] if end else np.zeros((0, 3))
