"""The drones a layout leaves over (its budget less the drones it used).

Skybrush gives every drone of the show a place in every formation; a drone a formation has no marker for
stays where it was, lit as the light effects say. So the rest of the budget is placed too:

- ``"park"``: dark (lights off), on an even grid under the figure (behind it, seen from the audience, when
  the figure is too low for that), out of the drawing and close to it;
- ``"stars"``: sparkles round the figure, as the big shows spend their spare drones: little stars of five
  (a centre and four arms, each turned its own way round the vertical, so they read from every side) on a
  shell round the figure, warm white, and single sparkles for the last few. What does not fit is parked.

Both keep the minimum distance to each other and to the figure. numpy only.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from . import color as _color
from .api import DroneResult
from .spatial import nearest

MODES = ("none", "park", "stars")
KIND = 3  # the kind of a drone placed here (0 corner, 1 line, 2 fill: the figure's)
PARK_SPACING = 1.5  # parked drones stand this many minimum distances apart ...
PARK_GAP = 3.0  # ... this many under the figure (and its stars)
GROUND = 2.0  # ... never lower than this many metres (else they go behind it)
STAR_ARM = 1.1  # a star's arms are this many minimum distances long
STAR_APART = 5.0  # star centres stand at least this many minimum distances apart ...
STAR_CLEAR = 3.0  # ... and this many off the figure
STAR_SHELL = (0.12, 0.45)  # ... at this share of the figure's size outside its box (nearest, farthest)
STAR_COLOR = (1.0, 0.85, 0.55)  # sRGB: warm white
STAR_ARM_LEVEL = 0.75  # the arms glow this bright, the centre at full


def leftover(result: DroneResult, mode: str = "park", d0: float = None, seed: int = 0) -> DroneResult:
    """``result`` with the rest of its budget placed (``mode`` "park" or "stars"); unchanged with "none" or no
    budget left over. The new drones come last, of kind ``KIND``, dark (parked) or warm white (stars)."""
    if mode not in MODES:
        raise ValueError(f"unknown leftover mode {mode!r}")
    rest = int(result.budget) - result.count
    if mode == "none" or rest <= 0 or result.count == 0:
        return result
    opt = result.options
    d0 = float(d0 if d0 is not None else (opt.target_min_distance if opt is not None else result.min_distance))
    up = int(opt.up_axis) if opt is not None else 2
    view = np.asarray(opt.view_dir if opt is not None else (0.0, -1.0, 0.0), np.float64)
    pos = np.asarray(result.positions, np.float64)
    add, col = np.zeros((0, 3)), np.zeros((0, 3))
    if mode == "stars":
        add, col = _stars(pos, rest, d0, up, np.random.default_rng(int(seed)))
    if len(add) < rest:  # parked: the rest (all of it, or what the stars left)
        p = _park(np.vstack([pos, add]), rest - len(add), d0, up, view)
        add, col = np.vstack([add, p]), np.vstack([col, np.zeros((len(p), 3))])
    return _append(result, add, col)


def _frame(up: int):
    """The two level axes and the up axis."""
    return [i for i in range(3) if i != up], up


def _park(pos, n, d0, up, view):
    """``n`` drones on an even grid under ``pos`` (behind them from the audience if too low)."""
    (a, b), u = _frame(up)
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    sp = PARK_SPACING * d0
    cols = max(int(math.ceil(math.sqrt(n * max(hi[a] - lo[a], sp) / max(hi[b] - lo[b], sp)))), 1)
    rows = int(math.ceil(n / cols))
    i = np.arange(n)
    ga, gb = (i % cols - 0.5 * (cols - 1)) * sp, (i // cols - 0.5 * (rows - 1)) * sp
    out = np.zeros((n, 3))
    height = lo[u] - PARK_GAP * d0
    if height >= GROUND:  # a level grid under the figure
        out[:, a] = 0.5 * (lo[a] + hi[a]) + ga
        out[:, b] = 0.5 * (lo[b] + hi[b]) + gb
        out[:, u] = height
        return out
    # too low: a standing grid behind it, its foot at the figure's foot
    v = -view  # view_dir points to the audience: this is from it into the scene
    v[u] = 0.0
    v = v / (np.linalg.norm(v) or 1.0)
    side = np.zeros(3)
    side[a], side[b] = -v[b], v[a]
    centre = 0.5 * (lo + hi)
    depth = max(float(np.max((pos - centre) @ v)), 0.0) + PARK_GAP * d0
    base = centre + depth * v
    base[u] = max(lo[u], GROUND)
    return base + ga[:, None] * side + (gb - gb.min())[:, None] * np.eye(3)[u]


def _stars(pos, n, d0, up, rng):
    """Up to ``n`` drones as stars of five (and single sparkles) round ``pos``: (positions, linear colours)."""
    (a, b), u = _frame(up)
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    size = float((hi - lo).max())
    centre = 0.5 * (lo + hi)
    r0 = 0.5 * float(np.hypot(hi[a] - lo[a], hi[b] - lo[b]))
    arm = STAR_ARM * d0
    want = n // 5
    singles = n - 5 * want
    placed, kinds = [], []
    tries = 0
    shell = np.array(STAR_SHELL) * size
    while len(placed) < want + singles and tries < 400 * (want + singles + 1):
        tries += 1
        th = rng.uniform(0.0, 2.0 * math.pi)
        r = r0 + rng.uniform(*shell)
        p = centre.copy()
        p[a] += r * math.cos(th)
        p[b] += r * math.sin(th)
        p[u] = rng.uniform(lo[u] + 0.15 * (hi[u] - lo[u]), hi[u] + shell[0])
        if p[u] < GROUND + arm:
            continue
        dd, _ = nearest(p[None], pos, STAR_CLEAR * d0 + arm)
        if dd[0] < STAR_CLEAR * d0 + arm:
            continue
        if placed and np.min(np.linalg.norm(np.asarray(placed) - p, axis=1)) < STAR_APART * d0:
            continue
        placed.append(p)
        kinds.append(len(placed) <= want)
    srgb = np.asarray(STAR_COLOR, np.float64)
    out, col = [], []
    for p, star in zip(placed, kinds):
        out.append(p)
        col.append(srgb)
        if star:
            t = rng.uniform(0.0, math.pi)  # the star's own turn round the vertical
            side = np.zeros(3)
            side[a], side[b] = math.cos(t), math.sin(t)
            vert = np.eye(3)[u]
            for s in (side, -side, vert, -vert):
                out.append(p + arm * s)
                col.append(srgb * STAR_ARM_LEVEL)
    if not out:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.asarray(out)[:n], _color.srgb_to_linear(np.clip(np.asarray(col), 0.0, 1.0))[:n]


def _append(result: DroneResult, add: np.ndarray, col: np.ndarray) -> DroneResult:
    m = len(add)
    if not m:
        return result
    n = result.count

    def cat(x, fill, dtype=None):
        if x is None:
            return None
        extra = np.full((m,) + np.shape(x)[1:], fill, dtype or np.asarray(x).dtype)
        return np.concatenate([np.asarray(x), extra])

    up = np.zeros((m, 3))
    up[:, int(result.options.up_axis) if result.options is not None else 2] = 1.0
    return replace(
        result,
        positions=np.vstack([result.positions, add]),
        normals=np.vstack([result.normals, up]),
        triangle_ids=cat(result.triangle_ids, 0),
        barycentrics=np.vstack([result.barycentrics, np.full((m, 3), 1.0 / 3.0)]),
        colors_linear=np.vstack([result.colors_linear, col]),
        visibility=cat(result.visibility, 0.0),
        outline=cat(result.outline, False),
        kind=cat(result.kind if result.kind is not None else np.full(n, 2, np.int8), KIND),
        part=cat(result.part, 0),
    )
