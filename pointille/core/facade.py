"""Facade mode: spend the drones where the audience can see them.

The audience is a row of viewers across its width. Each viewer is either far
away (sight lines parallel to ``view_dir``, pointing from the model towards
the viewers) or a point, such as a camera, where the sight lines converge. A
point's visibility is the share of viewers that see it: it faces them and no
part of the model is in the way. Occlusion uses a 2D grid of the triangles
projected along the sight lines, in plain numpy.
"""

from __future__ import annotations

import math

import numpy as np

_MAX_PAIRS = 8_000_000
_CHUNK = 1 << 15
_MAX_VIEWS = 9


def normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("view direction must be non-zero")
    return v / n


def view_basis(view_dir):
    d = normalize(view_dir)
    helper = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(helper, d)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    return d, e1, e2


def rotate(v, axis, angle: float) -> np.ndarray:
    """``v`` turned by ``angle`` radians around the unit ``axis`` (Rodrigues)."""
    v, k = np.asarray(v, dtype=np.float64), np.asarray(axis, dtype=np.float64)
    c, s = math.cos(angle), math.sin(angle)
    return v * c + np.cross(k, v) * s + k * (k @ v) * (1.0 - c)


class _ProjectedGrid:
    """Triangles bucketed into square cells by their 2D projection ``(px, py)``, each (T, 3)."""

    def __init__(self, px: np.ndarray, py: np.ndarray, ids: np.ndarray):
        self.origin = np.array([px.min(), py.min()]) if len(px) else np.zeros(2)
        extent = max(np.ptp(px), np.ptp(py), 1e-12) if len(px) else 1.0
        res = int(np.clip(math.sqrt(len(px)) * 1.5, 8, 2048))
        while True:
            self.cell = extent / res
            x0 = self._cell(px.min(axis=1), 0)
            x1 = self._cell(px.max(axis=1), 0)
            y0 = self._cell(py.min(axis=1), 1)
            y1 = self._cell(py.max(axis=1), 1)
            spans = (x1 - x0 + 1) * (y1 - y0 + 1)
            if spans.sum() <= _MAX_PAIRS or res <= 8:
                break
            res //= 2
        self.res = res + 1
        owner = np.repeat(np.arange(len(px)), spans)
        k = np.arange(spans.sum()) - np.repeat(np.cumsum(spans) - spans, spans)
        w = (x1 - x0 + 1)[owner]
        keys = x0[owner] + k % w + self.res * (y0[owner] + k // w)
        order = np.argsort(keys, kind="stable")
        self.tri_sorted = ids[owner[order]]
        self.keys, self.starts, self.counts = np.unique(keys[order], return_index=True, return_counts=True)

    def _cell(self, coord, axis):
        return np.clip(((coord - self.origin[axis]) / self.cell).astype(np.int64), 0, None)

    def pairs(self, qx: np.ndarray, qy: np.ndarray):
        """(query, triangle) for every triangle whose cell holds the query."""
        if not len(self.keys):
            e = np.zeros(0, np.int64)
            return e, e.copy()
        keys = np.minimum(self._cell(qx, 0), self.res - 1) + self.res * self._cell(qy, 1)
        pos = np.minimum(np.searchsorted(self.keys, keys), len(self.keys) - 1)
        found = self.keys[pos] == keys
        counts = np.where(found, self.counts[pos], 0)
        starts = np.where(found, self.starts[pos], 0)
        total = int(counts.sum())
        query = np.repeat(np.arange(len(qx)), counts)
        k = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        return query, self.tri_sorted[np.repeat(starts, counts) + k]


class _Occluder:
    """Exact segment-triangle tests (Möller–Trumbore) against a projected grid."""

    def __init__(self, tris: np.ndarray):
        self.a = tris[:, 0]
        self.e1 = tris[:, 1] - tris[:, 0]
        self.e2 = tris[:, 2] - tris[:, 0]
        diag = float(np.linalg.norm(np.ptp(tris.reshape(-1, 3), axis=0))) if len(tris) else 0.0
        self.eps = max(diag, 1e-12) * 1e-6

    def occluded(self, points: np.ndarray, own_tri: np.ndarray = None) -> np.ndarray:
        """True where the sight line from the point to the viewer hits the model."""
        points = np.asarray(points, dtype=np.float64)
        out = np.zeros(len(points), bool)
        for lo in range(0, len(points), _CHUNK):
            p = points[lo : lo + _CHUNK]
            direction, reach, blind = self._sight(p)
            ray, tri = self._pairs(p)
            if own_tri is not None:
                keep = tri != own_tri[lo : lo + _CHUNK][ray]
                ray, tri = ray[keep], tri[keep]
            hit = self._intersect(p[ray] + direction[ray] * self.eps, direction[ray], reach[ray], tri)
            out[lo : lo + _CHUNK] = blind | (np.bincount(ray[hit], minlength=len(p)) > 0)
        return out

    def _intersect(self, origin, direction, reach, tri):
        e1, e2 = self.e1[tri], self.e2[tri]
        pvec = np.cross(direction, e2)
        det = np.einsum("ij,ij->i", e1, pvec)
        ok = np.abs(det) > 1e-18
        inv = np.divide(1.0, det, out=np.zeros_like(det), where=ok)
        tvec = origin - self.a[tri]
        u = np.einsum("ij,ij->i", tvec, pvec) * inv
        qvec = np.cross(tvec, e1)
        v = np.einsum("ij,ij->i", direction, qvec) * inv
        t = np.einsum("ij,ij->i", e2, qvec) * inv
        return ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > self.eps) & (t < reach)


class OrthoOccluder(_Occluder):
    """A far-away viewer: every sight line is parallel to ``view_dir``."""

    def __init__(self, tris: np.ndarray, view_dir):
        super().__init__(tris)
        self.dir, e1, e2 = view_basis(view_dir)
        self.basis = (e1, e2)
        self.grid = _ProjectedGrid(tris @ e1, tris @ e2, np.arange(len(tris)))

    def _sight(self, p):
        n = len(p)
        return np.broadcast_to(self.dir, (n, 3)), np.full(n, np.inf), np.zeros(n, bool)

    def _pairs(self, p):
        e1, e2 = self.basis
        return self.grid.pairs(p @ e1, p @ e2)


class PointOccluder(_Occluder):
    """A viewer at ``eye`` looking at ``target``: sight lines converge on the eye.

    Triangles are projected through the eye onto a plane facing ``target``; a
    segment from a point to the eye then projects to a single spot. Triangles
    reaching behind the eye cannot be projected and are tested against every
    point; points behind the eye are not seen at all.
    """

    def __init__(self, tris: np.ndarray, eye, target):
        super().__init__(tris)
        self.eye = np.asarray(eye, dtype=np.float64).reshape(3)
        self.axis, e1, e2 = view_basis(np.asarray(target, dtype=np.float64) - self.eye)
        self.basis = (e1, e2)
        rel = tris - self.eye
        depth = rel @ self.axis
        reach = float(np.linalg.norm(np.asarray(target, dtype=np.float64) - self.eye))
        self.near = max(reach, self.eps) * 0.02
        ahead = (depth > self.near).all(axis=1)
        self.always = np.flatnonzero(~ahead)
        idx = np.flatnonzero(ahead)
        self.grid = _ProjectedGrid((rel[idx] @ e1) / depth[idx], (rel[idx] @ e2) / depth[idx], idx)

    def _sight(self, p):
        to_eye = self.eye - p
        dist = np.linalg.norm(to_eye, axis=1)
        direction = to_eye / np.maximum(dist, 1e-300)[:, None]
        return direction, dist - self.eps, (to_eye @ -self.axis) <= self.eps

    def _pairs(self, p):
        e1, e2 = self.basis
        rel = p - self.eye
        depth = np.maximum(rel @ self.axis, self.eps)
        ray, tri = self.grid.pairs((rel @ e1) / depth, (rel @ e2) / depth)
        if len(self.always):
            ray = np.concatenate([ray, np.repeat(np.arange(len(p)), len(self.always))])
            tri = np.concatenate([tri, np.tile(self.always, len(p))])
        return ray, tri


class Audience:
    """Viewers spread over ``spread`` degrees around the vertical axis.

    ``view_point`` puts the middle viewer at a point (a camera) and turns the
    others around the model's centre; otherwise the viewers are far away in
    the direction ``view_dir``.
    """

    def __init__(self, tris: np.ndarray, normals: np.ndarray, view_dir=(0.0, -1.0, 0.0), view_point=None,
                 spread: float = 0.0, facade_angle: float = 80.0, occlusion: bool = True, up_axis: int = 2):
        self.tris, self.normals = tris, normals
        self.cos = math.cos(math.radians(facade_angle))
        corners = tris.reshape(-1, 3)
        center = (corners.min(axis=0) + corners.max(axis=0)) * 0.5
        up = np.zeros(3)
        up[up_axis] = 1.0
        spread = math.radians(min(max(float(spread), 0.0), 360.0))
        k = 1 if spread <= 0 else min(max(int(math.ceil(math.degrees(spread) / 10.0)) + 1, 3), _MAX_VIEWS)
        if k == 1:
            angles = [0.0]
        else:  # on a full circle the first and the last viewer would stand at the same spot
            angles = np.linspace(-spread / 2, spread / 2, k, endpoint=spread < 2 * math.pi - 1e-9)

        self.points = view_point is not None
        middle = np.asarray(view_point, dtype=np.float64) - center if self.points else normalize(view_dir)
        if np.linalg.norm(middle) == 0:
            raise ValueError("the camera is at the centre of the model")
        axis = up if abs(normalize(middle) @ up) < 0.9 else view_basis(middle)[1]
        self.views = [rotate(middle, axis, a) for a in angles]
        if self.points:
            self.views = [center + v for v in self.views]
            make = lambda eye: PointOccluder(tris, eye, center)  # noqa: E731
        else:
            make = lambda d: OrthoOccluder(tris, d)  # noqa: E731
        self.occluders = [make(v) for v in self.views] if occlusion else [None] * len(self.views)

    def triangle_mask(self) -> np.ndarray:
        """Triangles at least one viewer may see (the exact test is per point)."""
        mask = np.zeros(len(self.tris), bool)
        for v in self.views:
            if self.points:
                mask |= np.einsum("ij,ij->i", self.normals, v - self.tris[:, 0]) > 0
            else:
                mask |= self.normals @ v > self.cos
        return mask

    def visibility(self, points: np.ndarray, tri: np.ndarray) -> np.ndarray:
        """Share of the viewers that see each point, in [0, 1]."""
        points = np.asarray(points, dtype=np.float64)
        seen = np.zeros(len(points))
        n = self.normals[tri]
        for v, occ in zip(self.views, self.occluders):
            if self.points:
                to_eye = v - points
                facing = np.einsum("ij,ij->i", n, to_eye) > self.cos * np.linalg.norm(to_eye, axis=1)
            else:
                facing = n @ v > self.cos
            if occ is not None and facing.any():
                facing[facing] = ~occ.occluded(points[facing], tri[facing])
            seen += facing
        return seen / len(self.views)
