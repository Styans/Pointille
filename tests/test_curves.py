import math

import numpy as np
import pytest

from helpers import uv_sphere
from pointille.core import Curve, DistributeOptions, Part, PartOptions, Sketch, compose_curves, trace
from pointille.core.mesh import MeshData
from pointille.core.spatial import nn_distances
from test_shapes import carved


def square(size, center=(0.0, 0.0), y=0.0):
    h = size / 2
    cx, cz = center
    return np.array([[cx - h, y, cz - h], [cx + h, y, cz - h], [cx + h, y, cz + h], [cx - h, y, cz + h]])


def opt(**kw):
    kw.setdefault("count", 2000)
    kw.setdefault("target_min_distance", 1.0)
    return DistributeOptions(view_dir=(0.0, -1.0, 0.0), **kw)


def test_a_circle_takes_drones_evenly_round_it():
    t = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    circle = np.stack([10 * np.cos(t), np.zeros_like(t), 10 * np.sin(t)], 1)
    r = compose_curves([Sketch([Curve(circle, True)], options=PartOptions(role="outline"))], opt())
    assert (r.kind == 1).all() and r.count == round(2 * np.pi * 10 / 1.5)
    gaps = nn_distances(r.positions, 1.0)
    assert np.ptp(gaps) < 0.05 and abs(np.hypot(r.positions[:, 0], r.positions[:, 2]) - 10).max() < 0.05


def test_corners_of_a_square_and_the_fill_inside():
    r = compose_curves([Sketch([Curve(square(20.0), True)])], opt())
    corner = r.positions[r.kind == 0]
    assert len(corner) == 4
    for c in square(20.0):
        assert np.linalg.norm(corner - c, axis=1).min() < 1e-6
    fill = r.positions[r.kind == 2]
    assert len(fill) > 10 and np.abs(fill[:, [0, 2]]).max() < 10.0 - 0.45 * 3.0 + 1e-6
    assert np.allclose(r.positions[:, 1], 0.0)
    assert r.min_distance >= 1.0 - 1e-9


def test_a_closed_curve_inside_another_is_a_hole():
    sk = Sketch([Curve(square(30.0), True), Curve(square(10.0), True)])
    r = compose_curves([sk], opt())
    fill = r.positions[r.kind == 2]
    assert len(fill) > 10 and not (np.abs(fill[:, [0, 2]]).max(axis=1) < 5.0).any()


def test_a_point_is_one_drone_and_the_budget_holds():
    sk = Sketch([Curve(square(40.0), True), Curve(np.array([[0.0, 0.0, 0.0]]))])
    r = compose_curves([sk], opt())
    assert np.linalg.norm(r.positions - 0.0, axis=1).min() < 1e-9
    small = compose_curves([sk], opt(count=60))
    assert small.count <= 60 and small.spread > 1.0 and small.fill_spread >= small.spread
    assert small.min_distance >= 1.0 * small.spread - 1e-9


@pytest.mark.parametrize("depth", ["flat", "shallow", "shape"])
def test_trace_a_carved_eye(depth):
    m, corners = carved()
    sk = trace([Part(m)], opt(facade=True), depth)
    assert len(sk) == 1 and sk[0].shell is not None
    curves = sk[0].curves
    kinds = {c.kind for c in curves}
    assert "outline" in kinds and "feature" in kinds
    pts = np.vstack([c.points for c in curves])
    assert np.ptp(pts[:, 1]) < 1e-6  # a flat model is flat in any depth
    feature = max((c for c in curves if c.kind == "feature"), key=lambda c: len(c.points))
    xz = feature.points[:, [0, 2]]
    for c in corners:  # the triangle keeps its tips
        assert np.linalg.norm(xz - c, axis=1).min() < 1.0
    r = compose_curves(sk, opt())
    assert r.count > 30 and (r.kind == 0).sum() >= 3 and r.min_distance >= 1.0 - 1e-9


def test_depth_of_a_round_model():
    ball = uv_sphere(10.0, 40, 80)
    depth = {}
    for d in ("flat", "shallow", "shape"):
        sk = trace([Part(ball)], opt(facade=True, count=3000), d)
        pts = np.vstack([c.points for c in sk[0].curves] + [sk[0].shell.triangles.reshape(-1, 3)])
        depth[d] = float(np.ptp(pts[:, 1]))
    assert depth["flat"] < 1e-6 < depth["shallow"] < depth["shape"]
    assert depth["shallow"] == pytest.approx(0.25 * depth["shape"], rel=0.1)
    assert depth["shape"] < 10.5  # the front of the ball only


def test_unknown_depth():
    m, _ = carved()
    with pytest.raises(ValueError):
        trace([Part(m)], opt(), "deep")


def ball_head():
    """A ball with two eyeballs sculpted on its front (facing -Y), the colour of the face."""
    eyes = [MeshData(uv_sphere(1.6, 16, 32).triangles + np.array([x, -8.9, 2.5])) for x in (-3.5, 3.5)]
    return MeshData.concatenate([uv_sphere(10.0, 60, 120)] + eyes)


@pytest.mark.parametrize("depth", ["flat", "volume"])
def test_sculpted_eyes_are_found(depth):
    sk = trace([Part(ball_head())], opt(facade=True, count=3000), depth)
    dots = np.array([c.points[0] for c in sk[0].curves if c.kind == "dot"])
    for x in (-3.5, 3.5):  # an eye on each side, level with the eyeballs (their tops lie a hair inward)
        near = dots[np.abs(dots[:, 0] - x) < 1.0]
        assert len(near) and np.abs(near[:, 2] - 2.5).min() < 0.5


def test_a_volume_is_rings_round_the_model():
    ball = uv_sphere(10.0, 40, 80)
    sk = trace([Part(ball)], opt(facade=True, count=3000), "volume")
    assert len(sk) == 1 and sk[0].solid and sk[0].shell is None
    slices = [c for c in sk[0].curves if c.kind == "slice"]
    assert len(slices) >= 3
    for c in slices:  # level, round the ball, a hair inside it (smoothed)
        assert c.closed and np.ptp(c.points[:, 2]) < 1e-6
        r = np.hypot(c.points[:, 0], c.points[:, 1])
        expect = math.sqrt(100.0 - c.points[0, 2] ** 2)
        assert abs(r - expect).max() < 0.1 * expect + 0.2
    r = compose_curves(sk, opt(count=3000))
    assert not (r.kind == 2).any()  # a volume takes no fill
    assert (r.positions[:, 1] > 5).any() and (r.positions[:, 1] < -5).any()  # front and back
    assert r.min_distance >= 1.0 - 1e-9
    small = compose_curves(sk, opt(count=100))
    assert small.count <= 100 and small.min_distance >= 1.0 * small.spread - 1e-9


def test_slices_of_a_volume_keep_out_of_a_carved_eye():
    m, corners = carved()
    sk = trace([Part(m)], opt(facade=True), "volume")
    feature = max((c for c in sk[0].curves if c.kind == "feature"), key=lambda c: len(c.points))
    slices = np.vstack([c.points for c in sk[0].curves if c.kind == "slice"])
    from pointille.core.picture import _inside
    assert not _inside(slices[:, [0, 2]], feature.points[:, [0, 2]]).any()
    r = compose_curves(sk, opt())
    assert r.count > 30 and r.min_distance >= 1.0 - 1e-9


def test_an_outline_in_pieces_still_takes_the_fill():
    a = np.array([[-10.0, 0.0, -10.0], [10.0, 0.0, -10.0], [10.0, 0.0, 10.0], [-9.2, 0.0, 10.0]])
    b = np.array([[-10.0, 0.0, 9.6], [-10.0, 0.0, -9.2]])  # a square in two open pieces, their ends < 1 apart
    r = compose_curves([Sketch([Curve(a, False), Curve(b, False)])], opt())
    fill = r.positions[r.kind == 2]
    assert len(fill) > 10 and np.abs(fill[:, [0, 2]]).max() < 10.0
