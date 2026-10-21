"""Blue-noise selection of exactly N points on a surface.

1. ``surface_candidates`` – dense area-uniform random points (oversampling).
2. ``eliminate``          – Weighted Sample Elimination (C. Yuksel, 2015):
                            drop the most crowded candidate until N remain.

The result is then evened out further by ``relax.relax``.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from .spatial import Grid

_CHUNK = 1 << 12


def triangle_areas_normals(tris: np.ndarray):
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    length = np.linalg.norm(cross, axis=1)
    normals = np.divide(cross, length[:, None], out=np.zeros_like(cross), where=length[:, None] > 0)
    return 0.5 * length, normals


def surface_candidates(tris: np.ndarray, weights: np.ndarray, count: int, rng: np.random.Generator):
    """``count`` points uniform over the surface, triangles picked ∝ ``weights``.

    Returns (positions, triangle_ids, barycentrics).
    """
    cdf = np.cumsum(weights, dtype=np.float64)
    if cdf[-1] <= 0:
        raise ValueError("surface has zero area")
    tri = np.searchsorted(cdf, rng.random(count) * cdf[-1], side="right")
    tri = np.minimum(tri, len(tris) - 1)
    r1 = np.sqrt(rng.random(count))
    r2 = rng.random(count)
    bary = np.stack([1.0 - r1, r1 * (1.0 - r2), r1 * r2], axis=1)
    pos = np.einsum("ij,ijk->ik", bary, tris[tri])
    return pos, tri, bary


def poisson_radius(area: float, count: int) -> float:
    """Max possible spacing of ``count`` points on ``area`` (hexagonal packing)."""
    return math.sqrt(area / (2.0 * math.sqrt(3.0) * max(count, 1)))


def spacing_scale(density: np.ndarray, max_scale: float = np.inf) -> np.ndarray:
    """How much wider than at full density the spacing is: ``1 / sqrt(density)``."""
    return np.minimum(1.0 / np.sqrt(np.maximum(density, 1e-12)), max_scale)


def _neighbours(points: np.ndarray, d_max: float, scale):
    """(i, j, distance) for every j closer to i than ``d_max``; with ``scale``
    the distance is measured in i's own spacing, so i only feels the points
    inside its spacing (a dense area is not squeezed by a sparse one nearby)."""
    if scale is None:
        return Grid(points, d_max).query_radius(points, d_max, exclude_self=True)
    reach = d_max * float(scale.max())
    grid = Grid(points, reach)
    parts = []
    for lo in range(0, len(points), _CHUNK):
        qi, pj, d = grid.query_radius(points[lo : lo + _CHUNK], reach)
        qi = qi + lo
        d = d / scale[qi]
        keep = (pj != qi) & (d < d_max)
        parts.append((qi[keep], pj[keep], d[keep]))
    return tuple(np.concatenate(column) for column in zip(*parts))


def eliminate(points: np.ndarray, count: int, area: float, alpha: float = 8.0, scale=None) -> np.ndarray:
    """Weighted Sample Elimination. Returns indices of the ``count`` kept points.

    ``scale`` (per point, see ``spacing_scale``) makes the spacing local, for a
    density that varies over the surface; ``area`` is then the area weighted
    by that density.
    """
    m = len(points)
    if count >= m:
        return np.arange(m)
    d_max = 2.0 * poisson_radius(area, count)
    # weight limiting (Yuksel 2015, eq. 4) keeps clumps from dominating
    d_min = d_max * (1.0 - (count / m) ** 1.5) * 0.65

    qi, pj, d = _neighbours(points, d_max, scale)
    d = np.maximum(d, d_min)
    w = (1.0 - d / d_max) ** alpha

    # removing a point lowers the weight of every point it crowds
    order = np.argsort(pj, kind="stable")
    nbr = qi[order]
    nbr_w = w[order]
    offsets = np.zeros(m + 1, np.int64)
    np.cumsum(np.bincount(pj, minlength=m), out=offsets[1:])

    weight = np.bincount(qi, weights=w, minlength=m)
    version = np.zeros(m, np.int64)
    removed = np.zeros(m, bool)
    heap = [(-weight[i], 0, i) for i in range(m)]
    heapq.heapify(heap)

    remaining = m
    nbr_list = nbr.tolist()
    nbr_w_list = nbr_w.tolist()
    weight_list = weight.tolist()
    version_list = version.tolist()
    removed_list = removed.tolist()
    offs = offsets.tolist()
    push, pop = heapq.heappush, heapq.heappop
    while remaining > count:
        _, ver, i = pop(heap)
        if removed_list[i] or ver != version_list[i]:
            continue
        removed_list[i] = True
        remaining -= 1
        for k in range(offs[i], offs[i + 1]):
            j = nbr_list[k]
            if removed_list[j]:
                continue
            weight_list[j] -= nbr_w_list[k]
            version_list[j] += 1
            push(heap, (-weight_list[j], version_list[j], j))
    return np.flatnonzero(~np.asarray(removed_list))
