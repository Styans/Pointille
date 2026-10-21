"""Spacing statistics and safety scaling of a formation."""

from __future__ import annotations

import numpy as np

from .spatial import nn_distances

SCALE_MODES = ("none", "min_distance", "height")


def spacing(points: np.ndarray, guess: float):
    """(min, mean) distance from each drone to its nearest neighbour."""
    if len(points) < 2:
        return 0.0, 0.0
    nn = nn_distances(points, guess)
    return float(nn.min()), float(nn.mean())


def scale_factor(mode: str, min_distance: float, size: np.ndarray, up_axis: int,
                 target_min_distance: float, target_height: float) -> float:
    if mode == "none":
        return 1.0
    if mode == "min_distance":
        if min_distance <= 0:
            raise ValueError("cannot scale: some drones coincide")
        return float(target_min_distance) / min_distance
    if mode == "height":
        h = float(size[up_axis])
        if h <= 0:
            raise ValueError("cannot scale: formation is flat along the up axis")
        return float(target_height) / h
    raise ValueError(f"unknown scale mode {mode!r}, expected one of {SCALE_MODES}")


def scale_pivot(bbox_min: np.ndarray, bbox_max: np.ndarray, up_axis: int) -> np.ndarray:
    """Bottom centre of the bounding box: scaling keeps the figure on the ground."""
    pivot = (np.asarray(bbox_min) + np.asarray(bbox_max)) * 0.5
    pivot[up_axis] = bbox_min[up_axis]
    return pivot
