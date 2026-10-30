import numpy as np

from helpers import cube
from pointille.core import DistributeOptions, Part, burst
from pointille.core.color import linear_to_srgb
from pointille.core.figure import around_figure
from pointille.core.mesh import MeshData
from pointille.core.sampling import surface_candidates, triangle_areas_normals
from pointille.core.spatial import nn_distances


def box(lo, hi, res=8):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return cube(1.0, res).triangles * (hi - lo) + (lo + hi) / 2


def samples(tris, n=20000):
    areas, _ = triangle_areas_normals(tris)
    return surface_candidates(tris, areas, n, np.random.default_rng(0))[0]


def test_the_inside_of_a_box():
    inside, _ = burst.voxelize(box((0, 0, 0), (10, 6, 4)), 0.5)
    assert abs(inside.sum() * 0.5 ** 3 - 240.0) < 0.15 * 240.0


def test_the_wave_reaches_the_corners_of_a_box_last():
    body = burst.Body(box((-5, -5, -5), (5, 5, 5)), 1.5)
    corners = body.at(np.array([[5.0, 5.0, 5.0], [-5.0, 5.0, -5.0]]))
    faces = body.at(np.array([[5.0, 0.0, 0.0], [0.0, 0.0, -5.0]]))
    assert np.isfinite(corners).all() and corners.min() > faces.max()
    assert np.abs(faces - 5.0).max() < 1.0  # straight out of a convex body: the distance from its middle


def test_the_wave_goes_round_corners():
    # a U: the wave from the middle of the base reaches the top of an arm along the base and up the arm
    t = np.vstack([box((0, 0, 0), (30, 4, 4)), box((0, 0, 4), (4, 4, 20)), box((26, 0, 4), (30, 4, 20))])
    body = burst.Body(t, 1.5)
    top = np.array([[2.0, 2.0, 20.0]])
    straight = float(np.linalg.norm(top[0] - body.centre(np.array([body.core]))[0]))
    assert body.at(top)[0] > 1.25 * straight
    assert np.isfinite(body.at(samples(t))).all()


def test_a_lying_body_is_a_sideways_part_and_a_standing_one_upright():
    lying = burst.Parts(burst.Body(box((-15, -2.5, -2.5), (15, 2.5, 2.5)), 1.5), 1.5)
    assert lying.sideways_at(np.array([[13.0, 0.0, 2.5], [-13.0, 0.0, -2.5], [12.0, 2.5, 0.0]])).all()
    t = box((-2.5, -2.5, 0), (2.5, 2.5, 30))
    standing = burst.Parts(burst.Body(t, 1.5), 1.5)
    assert not standing.sideways_at(samples(t)).any()


def test_tips_at_the_ends_of_the_arms():
    t = np.vstack([box((-4, -3, 0), (4, 3, 16)), box((-14, -1.5, 11), (-4, 1.5, 14)),
                   box((4, -1.5, 11), (14, 1.5, 14))])
    tips = burst.Parts(burst.Body(t, 1.5), 1.5).tips(samples(t))
    for hand in ([-14.0, 0.0, 12.5], [14.0, 0.0, 12.5]):
        assert np.linalg.norm(tips - np.array(hand), axis=1).min() < 3.0


def animal():
    """A lying body on four legs, a raised tail."""
    t = [box((-12, -3, 8), (12, 3, 14)), box((12, -1, 12), (14, 1, 22))]
    for x in (-10.0, 8.0):
        for y in (-3.0, 1.5):
            t.append(box((x, y, 0), (x + 2, y + 1.5, 8), 4))
    return MeshData(np.vstack(t))


def test_a_figure_by_its_parts():
    opt = DistributeOptions(count=400, target_min_distance=1.5, view_dir=(0.0, -1.0, 0.0))
    m = animal()
    for parts in (True, False):
        r = around_figure([Part(m)], DistributeOptions(**{**opt.__dict__, "around_parts": parts}))
        assert 0 < r.count <= 400
        assert nn_distances(r.positions, 1.5).min() >= 1.5 - 1e-6
        assert np.isfinite(r.colors_linear).all()
    r = around_figure([Part(m)], opt)
    level = linear_to_srgb(r.colors_linear).max(axis=1)
    # the outline columns bright, the drones between them dim
    assert np.median(level[r.kind == 1]) > 0.9 and np.median(level[r.kind == 2]) < 0.6
