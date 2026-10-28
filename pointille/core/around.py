"""Around: drones that draw a 3D model crisply from every side (the add-on's «Вокруг (3D)»).

At night nothing hides a drone: from any side the front and the back of a figure shine through each other.
Drones spread evenly over the surface then stack into clumps and noise; drones on smooth lines stack into
lines, so the figure keeps a crisp outline from every side. Each object is drawn by lines, by its kind:
- a round body (a fruit, a pumpkin, a vase: one loop round a vertical axis at nearly every height) as a cage
  of meridians in its rib grooves and rings at its base and shoulder, the stem a tube with a collar, carved
  holes (a jack-o'-lantern's face) as bright rims around dark holes (``cage``);
- any other figure (a character, an animal) as lines that keep its outline from every side, its face (eyes
  found as sculpted bumps, the muzzle), crown and spikes as bright features, creases dim (``figure``).
Several objects share the budget by their surface and keep the minimum distance between them.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional

import numpy as np

from . import cage as _cage
from . import figure as _figure
from . import outline as _outline
from .api import DistributeOptions, DroneResult
from .sampling import triangle_areas_normals
from .stats import scale_pivot, spacing as _spacing

KINDS = ("round", "figure")


def kind_of(part, options: Optional[DistributeOptions] = None) -> str:
    """"round" for a round body, else "figure" (see the module)."""
    return _cage.classify([part], options)


def around(parts, options: Optional[DistributeOptions] = None) -> DroneResult:
    """Drones all round ``parts`` (layout.Part), at most ``options.count``, at least
    ``options.target_min_distance`` apart, each object drawn by lines as its kind asks (see the module)."""
    opt = options or DistributeOptions()
    if not parts:
        raise ValueError("no parts to lay out")
    budget = int(opt.count)
    if budget < 1:
        raise ValueError("count must be at least 1")
    if len(parts) == 1:
        return _one(parts[0], opt)
    areas = np.array([float(triangle_areas_normals(p.mesh.triangles)[0].sum()) for p in parts])
    share = np.maximum(np.floor(budget * areas / max(areas.sum(), 1e-300)).astype(int), 1)
    results = [_one(p, replace(opt, count=int(n))) for p, n in zip(parts, share)]
    offsets = np.cumsum([0] + [p.mesh.triangle_count for p in parts])
    return _merged(results, offsets, opt, budget)


def _one(part, opt) -> DroneResult:
    if kind_of(part, opt) == "round":
        return _cage.around_round([part], opt)
    return _figure.around_figure([part], opt)


def _merged(results: List[DroneResult], offsets, opt, budget: int) -> DroneResult:
    """One layout of several objects' layouts (their triangles numbered from ``offsets``, as in the meshes
    concatenated): the important drones first, nobody nearer than the minimum distance to a drone of another
    object."""
    d0 = float(opt.target_min_distance)
    pos = np.vstack([r.positions for r in results])
    kind = np.concatenate([r.kind if r.kind is not None else np.full(r.count, 2, np.int8) for r in results])
    part = np.concatenate([np.full(r.count, k, np.int64) for k, r in enumerate(results)])
    order = np.argsort(kind, kind="stable")
    keep = order[_outline.park(pos[order], np.full(len(pos), d0))][:budget]
    keep.sort()

    def cat(name):
        return np.concatenate([getattr(r, name) for r in results])[keep]

    pos = pos[keep]
    dmin, dmean = _spacing(pos, 2.0 * d0)
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    return DroneResult(
        positions=pos,
        normals=cat("normals"),
        triangle_ids=np.concatenate([r.triangle_ids + offsets[k] for k, r in enumerate(results)])[keep],
        barycentrics=cat("barycentrics"),
        colors_linear=cat("colors_linear"),
        min_distance=dmin,
        mean_distance=dmean,
        area=float(sum(r.area for r in results)),
        pivot=scale_pivot(lo, hi, opt.up_axis),
        options=opt,
        visibility=np.ones(len(pos)),
        outline=kind[keep] < 2,
        kind=kind[keep].astype(np.int8),
        part=part[keep],
        budget=budget,
        needed=int(sum(r.needed for r in results)),
        spread=max(r.spread for r in results),
        fill_spread=max(r.fill_spread for r in results),
    )
