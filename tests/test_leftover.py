import numpy as np
import pytest

from helpers import uv_sphere
from pointille.core import DistributeOptions, Part, around, leftover
from pointille.core.mesh import MeshData
from pointille.core.spatial import nn_distances


def opt(**kw):
    kw.setdefault("target_min_distance", 1.5)
    return DistributeOptions(view_dir=(0.0, -1.0, 0.0), **kw)


def ball(z, count=1500):
    return around([Part(MeshData(uv_sphere(8.0, 48, 96).triangles + np.array([0.0, 0.0, z])))], opt(count=count))


def test_the_rest_is_parked_dark_under_the_figure():
    r = ball(40.0)
    assert r.count < r.budget
    q = leftover(r, "park")
    assert q.count == q.budget
    rest = q.kind == 3
    assert rest.sum() == r.budget - r.count
    assert (q.positions[rest, 2] < 40.0 - 8.0).all()  # under the ball
    assert (q.colors_linear[rest] == 0).all()  # lights off
    assert nn_distances(q.positions, 1.5).min() >= 1.5 - 1e-6
    assert np.allclose(q.positions[~rest], r.positions)  # the figure is untouched


def test_a_low_figure_parks_behind_it():
    q = leftover(ball(9.0), "park")
    rest = q.positions[q.kind == 3]
    assert (rest[:, 1] > 8.0).all() and (rest[:, 2] >= 2.0 - 1e-9).all()  # behind (+y: the audience is at -y)


def test_stars_round_the_figure():
    r = ball(40.0, count=300)  # a rest of about the figure's own size (a far bigger one is mostly parked)
    q = leftover(r, "stars", seed=3)
    rest = q.kind == 3
    assert q.count == q.budget and rest.sum() == r.budget - r.count
    lit = rest & (q.colors_linear.max(axis=1) > 0)
    assert lit.sum() >= 0.5 * rest.sum()  # most of the rest are stars, not parked
    d = np.linalg.norm(q.positions[lit, :2], axis=1)
    assert (d > 8.0 + 2 * 1.5).all()  # round the ball, clear of it
    assert nn_distances(q.positions, 1.5).min() >= 1.5 - 1e-6


def test_nothing_to_place():
    r = ball(40.0, count=200)
    assert leftover(r, "none") is r
    if r.count == r.budget:
        assert leftover(r, "park") is r
    with pytest.raises(ValueError):
        leftover(r, "confetti")
