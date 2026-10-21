"""Uniform-grid neighbour queries in pure numpy (Blender ships without scipy)."""

from __future__ import annotations

import numpy as np

_OFFSETS = np.array(
    [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
    dtype=np.int64,
)
_CHUNK = 1 << 16


def _expand_ranges(starts: np.ndarray, counts: np.ndarray):
    """For ranges [s, s+c) return (owner, position) of every element."""
    total = int(counts.sum())
    owner = np.repeat(np.arange(len(counts)), counts)
    offs = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    return owner, np.repeat(starts, counts) + offs


class Grid:
    """Points bucketed into cubic cells of edge ``cell``."""

    def __init__(self, points: np.ndarray, cell: float):
        if not cell > 0:
            raise ValueError("cell size must be positive")
        self.points = np.asarray(points, dtype=np.float64)
        self.origin = self.points.min(axis=0) if len(self.points) else np.zeros(3)
        extent = float(np.ptp(self.points, axis=0).max()) if len(self.points) else 0.0
        # keep linear cell keys inside int64; a coarser grid is still correct
        self.cell = max(float(cell), extent / 1.0e6)
        coords = self._coords(self.points)
        self.dims = coords.max(axis=0) + 3 if len(coords) else np.ones(3, np.int64)
        keys = self._key(coords)
        self.order = np.argsort(keys, kind="stable")
        sorted_keys = keys[self.order]
        self.keys, self.starts, self.counts = np.unique(
            sorted_keys, return_index=True, return_counts=True
        )

    def _coords(self, pts):
        return np.floor((pts - self.origin) / self.cell).astype(np.int64) + 1

    def _key(self, coords):
        c = np.clip(coords, 0, self.dims - 1)
        return c[:, 0] + self.dims[0] * (c[:, 1] + self.dims[1] * c[:, 2])

    def _cell_ranges(self, coords):
        inside = np.all((coords >= 0) & (coords < self.dims), axis=1)
        keys = self._key(coords)
        pos = np.searchsorted(self.keys, keys)
        pos = np.minimum(pos, len(self.keys) - 1)
        hit = inside & (self.keys[pos] == keys) if len(self.keys) else np.zeros(len(keys), bool)
        starts = np.where(hit, self.starts[pos] if len(self.keys) else 0, 0)
        counts = np.where(hit, self.counts[pos] if len(self.keys) else 0, 0)
        return starts, counts

    def query_radius(self, queries: np.ndarray, radius: float, exclude_self: bool = False):
        """All (query_index, point_index, distance) with distance < radius.

        ``radius`` must not exceed the cell size.  With ``exclude_self`` the
        queries are assumed to be the grid points themselves.
        """
        if radius > self.cell * (1 + 1e-9):
            raise ValueError("radius larger than grid cell")
        queries = np.asarray(queries, dtype=np.float64)
        out_q, out_p, out_d = [], [], []
        r2 = radius * radius
        for lo in range(0, len(queries), _CHUNK):
            q = queries[lo : lo + _CHUNK]
            qc = self._coords(q)
            for off in _OFFSETS:
                starts, counts = self._cell_ranges(qc + off)
                if not counts.any():
                    continue
                qi, pos = _expand_ranges(starts, counts)
                pi = self.order[pos]
                diff = self.points[pi] - q[qi]
                d2 = np.einsum("ij,ij->i", diff, diff)
                keep = d2 < r2
                qi = qi + lo
                if exclude_self:
                    keep &= pi != qi
                out_q.append(qi[keep])
                out_p.append(pi[keep])
                out_d.append(np.sqrt(d2[keep]))
        if not out_q:
            e = np.zeros(0, np.int64)
            return e, e.copy(), np.zeros(0)
        return np.concatenate(out_q), np.concatenate(out_p), np.concatenate(out_d)


def nearest(queries: np.ndarray, points: np.ndarray, guess: float, exclude_self: bool = False):
    """Nearest point for every query. Returns (distance, index).

    ``guess`` is a starting search radius; it is doubled for queries that find
    nothing. With ``exclude_self`` queries must be ``points`` itself.
    """
    queries = np.asarray(queries, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    n = len(queries)
    dist = np.full(n, np.inf)
    idx = np.full(n, -1, np.int64)
    if n == 0 or len(points) == 0 or (exclude_self and len(points) < 2):
        return dist, idx
    extent = float(np.ptp(np.vstack([points, queries]), axis=0).max()) or 1.0
    radius = max(float(guess), extent * 1e-6)
    todo = np.arange(n)
    while len(todo):
        grid = Grid(points, radius)
        qi, pi, d = grid.query_radius(queries[todo], radius)
        if exclude_self:
            keep = pi != todo[qi]
            qi, pi, d = qi[keep], pi[keep], d[keep]
        if len(qi):
            order = np.lexsort((d, qi))
            qi, pi, d = qi[order], pi[order], d[order]
            first = np.ones(len(qi), bool)
            first[1:] = qi[1:] != qi[:-1]
            dist[todo[qi[first]]] = d[first]
            idx[todo[qi[first]]] = pi[first]
        todo = todo[idx[todo] < 0]
        if radius > extent * 4:
            break
        radius *= 2.0
    return dist, idx


def nn_distances(points: np.ndarray, guess: float) -> np.ndarray:
    """Distance from every point to its nearest other point."""
    d, _ = nearest(points, points, guess, exclude_self=True)
    return d
