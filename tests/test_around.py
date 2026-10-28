import numpy as np
import pytest

from helpers import cube, uv_sphere
from pointille.core import DistributeOptions, Part, around
from pointille.core.around import kind_of
from pointille.core.mesh import MeshData
from pointille.core.spatial import nn_distances


def opt(**kw):
    kw.setdefault("count", 400)
    kw.setdefault("target_min_distance", 1.5)
    return DistributeOptions(view_dir=(0.0, -1.0, 0.0), **kw)


def ball(r=12.0, at=(0.0, 0.0, 0.0)):
    return MeshData(uv_sphere(r, 48, 96).triangles + np.asarray(at, float))


def legs():
    """A figure, not a round body: two legs under a body (two loops at every height of the legs)."""
    blocks = []
    for x in (-6.0, 6.0):
        blocks.append(MeshData(cube(1.0, 6).triangles * np.array([3.0, 3.0, 9.0]) + np.array([x, 0.0, 9.0])))
    blocks.append(MeshData(cube(1.0, 6).triangles * np.array([10.0, 4.0, 7.0]) + np.array([0.0, 0.0, 24.0])))
    return MeshData.concatenate(blocks)


def check(r, budget, d0=1.5):
    assert 0 < r.count <= budget
    assert nn_distances(r.positions, d0).min() >= d0 - 1e-6
    assert r.kind is not None and len(r.kind) == r.count
    assert r.colors_linear.shape == (r.count, 3) and np.isfinite(r.colors_linear).all()


def test_a_ball_is_a_round_body_drawn_all_round():
    m = ball()
    assert kind_of(Part(m), opt()) == "round"
    r = around([Part(m)], opt())
    check(r, 400)
    p = r.positions
    for axis in (0, 1):  # drones on every side, near the surface
        assert (p[:, axis] > 8).any() and (p[:, axis] < -8).any()
    assert np.abs(np.linalg.norm(p, axis=1) - 12.0).max() < 2.0


def test_a_figure_is_drawn_by_its_outline():
    m = legs()
    assert kind_of(Part(m), opt()) == "figure"
    r = around([Part(m)], opt(count=300))
    check(r, 300)
    t = m.triangles.reshape(-1, 3)
    assert (r.positions >= t.min(axis=0) - 1.0).all() and (r.positions <= t.max(axis=0) + 1.0).all()


def lathe(profile, n=72):
    """A turned shape: the (radius, height) ``profile`` from the base to the top, round the z axis, closed by
    flat discs at both ends."""
    prof = np.vstack([[0.0, profile[0][1]], profile, [0.0, profile[-1][1]]])
    a = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    v = np.array([[r * np.cos(x), r * np.sin(x), z] for r, z in prof for x in a])
    f = []
    for i in range(len(prof) - 1):
        for j in range(n):
            p, q = i * n + j, i * n + (j + 1) % n
            f += [[p, q, q + n], [p, q + n, p + n]]
    t = v[np.array(f)]
    return MeshData(t[np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1) > 1e-9])


def test_a_turned_shape_is_drawn_by_rings_on_its_edges():
    # the user's spool: two wide rims, a neck between two cones
    profile = [(27.6, 0.0), (19.2, 7.4), (8.9, 14.8), (10.1, 19.2), (16.2, 27.5), (27.6, 33.1)]
    m = lathe(profile)
    assert kind_of(Part(m), opt()) == "round"
    r = around([Part(m)], opt(count=500))
    check(r, 500)
    p = r.positions
    rad = np.hypot(p[:, 0], p[:, 1])
    for rr, z in profile:  # a whole ring of drones on every edge of the profile, top and base rims too
        if z == 7.4:  # a slight bend (5 degrees) is no edge: a ring between the edges stands near it
            assert (np.abs(p[:, 2] - z) < 1.0).sum() >= 2 * np.pi * rr / (3 * 1.5)
            continue
        near = np.abs(p[:, 2] - z) < 0.3
        assert near.sum() >= 2 * np.pi * rr / (3 * 1.5), z
        assert np.abs(rad[near] - rr).max() < 1.0, z
    assert not ((np.abs(p[:, 2] - 33.1) < 0.05) & (rad < 20)).any()  # the flat top is drawn by its rim alone


def test_a_fruit_has_no_edge_rings():
    from pointille.core import cage
    from pointille.core.sampling import triangle_areas_normals
    t = ball().triangles
    areas, normals = triangle_areas_normals(t)
    model = cage._Model(t, normals, areas)
    shape = cage._shape(model, 2.0, cage.Style())
    assert shape.round and cage._edge_rings(model, shape, cage.Style(), 1.5, 3.0) == []


def test_two_objects_share_the_budget():
    a, b = ball(8.0, (-20.0, 0.0, 10.0)), ball(8.0, (20.0, 0.0, 10.0))
    r = around([Part(a), Part(b)], opt(count=300))
    check(r, 300)
    assert set(np.unique(r.part)) == {0, 1}
    assert (r.positions[r.part == 0, 0] < 0).all() and (r.positions[r.part == 1, 0] > 0).all()


def test_budget_and_errors():
    with pytest.raises(ValueError):
        around([], opt())
    with pytest.raises(ValueError):
        around([Part(ball())], opt(count=0))
    small = around([Part(ball())], opt(count=60))
    check(small, 60)
