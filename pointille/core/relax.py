"""Relaxation: drones push each other apart while staying on the surface.

Each iteration moves every drone away from neighbours closer than the ideal
hexagonal spacing, then snaps it back onto the model with an exact
closest-point projection. This evens out the blue-noise start from Weighted
Sample Elimination (min spacing ~0.73 -> ~0.83 of the hexagonal optimum).
"""

from __future__ import annotations

import numpy as np

from .sampling import spacing_scale
from .spatial import Grid


def closest_on_triangles(p, a, b, c):
    """Closest point on triangles (a, b, c) to points p, all (K, 3).

    Returns barycentric weights (K, 3). Ericson, Real-Time Collision
    Detection, 5.1.5, vectorised.
    """
    ab, ac, ap = b - a, c - a, p - a
    dot = lambda x, y: np.einsum("ij,ij->i", x, y)  # noqa: E731
    d1, d2 = dot(ab, ap), dot(ac, ap)
    bp = p - b
    d3, d4 = dot(ab, bp), dot(ac, bp)
    cp = p - c
    d5, d6 = dot(ab, cp), dot(ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    denom = va + vb + vc
    safe = np.where(np.abs(denom) > 1e-300, denom, 1.0)
    v = vb / safe
    w = vc / safe
    bary = np.stack([1 - v - w, v, w], axis=1)
    done = np.zeros(len(p), bool)

    def assign(mask, values):
        nonlocal done
        m = mask & ~done
        bary[m] = values[m] if values.ndim == 2 else values
        done |= m

    one = np.ones(len(p))
    zero = np.zeros(len(p))
    assign((d1 <= 0) & (d2 <= 0), np.array([1.0, 0.0, 0.0]))
    assign((d3 >= 0) & (d4 <= d3), np.array([0.0, 1.0, 0.0]))
    t = d1 / np.where(d1 - d3 != 0, d1 - d3, 1.0)
    assign((vc <= 0) & (d1 >= 0) & (d3 <= 0), np.stack([one - t, t, zero], 1))
    assign((d6 >= 0) & (d5 <= d6), np.array([0.0, 0.0, 1.0]))
    t = d2 / np.where(d2 - d6 != 0, d2 - d6, 1.0)
    assign((vb <= 0) & (d2 >= 0) & (d6 <= 0), np.stack([one - t, zero, t], 1))
    e = (d4 - d3) + (d5 - d6)
    t = (d4 - d3) / np.where(e != 0, e, 1.0)
    assign((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), np.stack([zero, one - t, t], 1))
    return bary


def _lattice(k: int) -> np.ndarray:
    """Barycentric corners of the k*k sub-triangles of a triangle, (k*k, 3, 3)."""
    out = []
    for i in range(k):
        for j in range(k - i):
            out.append([(k - i - j, i, j), (k - i - j - 1, i + 1, j), (k - i - j - 1, i, j + 1)])
            if i + j <= k - 2:
                out.append([(k - i - j - 1, i + 1, j), (k - i - j - 2, i + 1, j + 1), (k - i - j - 1, i, j + 1)])
    return np.array(out, dtype=np.float64) / k


class SurfaceProjector:
    """Snap points back onto the surface after a small move.

    Triangles longer than ``max_edge`` are split for indexing only, so a
    query looks at a bounded neighbourhood. ``max_move`` is the furthest a
    query point may be from the surface.
    """

    def __init__(self, tris: np.ndarray, tri_ids: np.ndarray, max_edge: float, max_move: float):
        edges = np.stack(
            [np.linalg.norm(tris[:, i] - tris[:, (i + 1) % 3], axis=1) for i in range(3)], 1
        ).max(axis=1)
        k = np.clip(np.ceil(edges / max_edge), 1, 32).astype(np.int64)
        sub, parent, sub_bary = [], [], []
        for kk in np.unique(k):
            idx = np.flatnonzero(k == kk)
            lat = _lattice(int(kk))  # (S, 3, 3)
            corners = np.einsum("sjk,tkd->tsjd", lat, tris[idx]).reshape(-1, 3, 3)
            sub.append(corners)
            parent.append(np.repeat(tri_ids[idx], len(lat)))
            sub_bary.append(np.broadcast_to(lat, (len(idx),) + lat.shape).reshape(-1, 3, 3))
        self.sub = np.concatenate(sub)
        self.parent = np.concatenate(parent)
        self.sub_bary = np.concatenate(sub_bary)
        sub_edge = max(
            float(np.linalg.norm(self.sub[:, i] - self.sub[:, (i + 1) % 3], axis=1).max())
            for i in range(3)
        )
        self.max_move = max_move
        self.reach = sub_edge + max_move
        self.grid = Grid(self.sub.mean(axis=1), self.reach)

    def project(self, points: np.ndarray):
        """Returns (positions, parent_triangle_ids, parent_barycentrics)."""
        qi, si, _ = self.grid.query_radius(points, self.reach)
        tri = self.sub[si]
        bary = closest_on_triangles(points[qi], tri[:, 0], tri[:, 1], tri[:, 2])
        pos = np.einsum("ij,ijk->ik", bary, tri)
        d2 = np.einsum("ij,ij->i", pos - points[qi], pos - points[qi])
        order = np.lexsort((d2, qi))
        first = np.ones(len(order), bool)
        first[1:] = qi[order][1:] != qi[order][:-1]
        best = order[first]
        found = np.zeros(len(points), bool)
        found[qi[best]] = True
        out_pos = points.copy()
        out_tri = np.full(len(points), -1, np.int64)
        out_bary = np.zeros((len(points), 3))
        rows = qi[best]
        out_pos[rows] = pos[best]
        out_tri[rows] = self.parent[si[best]]
        out_bary[rows] = np.einsum("ij,ijk->ik", bary[best], self.sub_bary[si[best]])
        return out_pos, out_tri, out_bary, found


def relax(
    points: np.ndarray,
    tri_ids: np.ndarray,
    bary: np.ndarray,
    projector: "SurfaceProjector",
    spacing: float,
    iterations: int,
    step: float = 0.2,
    density=None,
    max_scale: float = 1.0,
):
    """Repulsion relaxation. Returns (points, tri_ids, bary) with the best
    minimum spacing seen.

    ``density(points, tri_ids)`` in [0, 1] makes the spacing local: a drone
    keeps ``spacing / sqrt(density)`` (at most ``max_scale`` times
    ``spacing``) from its neighbours, and moves to where the density is 0 are
    vetoed (facade mode keeps drones visible that way). The projector's
    ``max_move`` is the step limit for the widest spacing."""
    p, t, b = points.copy(), tri_ids.copy(), bary.copy()
    if iterations <= 0 or len(p) < 2:
        return p, t, b
    s = np.ones(len(p)) if density is None else spacing_scale(density(p, t), max_scale)
    best_min, best = -1.0, (p, t, b)
    for it in range(iterations + 1):
        reach = spacing * float(s.max())
        grid = Grid(p, reach)
        qi, pj, dist = grid.query_radius(p, reach, exclude_self=True)
        local = spacing * s[qi]  # every drone keeps its own spacing
        near = dist < local
        qi, pj, dist, local = qi[near], pj[near], dist[near], local[near]
        cur_min = float(dist.min()) if len(dist) else spacing
        if cur_min > best_min:
            best_min, best = cur_min, (p.copy(), t.copy(), b.copy())
        if not len(dist) or it == iterations:
            break
        # soft 1/d repulsion: strong for close pairs, zero at the ideal spacing
        mag = local / np.maximum(dist, local * 1e-3) - 1.0
        force = (p[qi] - p[pj]) * (mag * local / np.maximum(dist, 1e-300))[:, None]
        disp = np.zeros_like(p)
        np.add.at(disp, qi, force)
        disp *= step
        length = np.linalg.norm(disp, axis=1)
        max_move = projector.max_move * s / max(max_scale, 1.0)
        disp *= np.minimum(1.0, max_move / np.maximum(length, 1e-300))[:, None]
        new_p, new_t, new_b, ok = projector.project(p + disp)
        if density is not None:
            rho = density(new_p[ok], new_t[ok])
            s[np.flatnonzero(ok)[rho > 0]] = spacing_scale(rho[rho > 0], max_scale)
            ok[ok] = rho > 0
        p = np.where(ok[:, None], new_p, p)
        t = np.where(ok, new_t, t)
        b = np.where(ok[:, None], new_b, b)
    return best
