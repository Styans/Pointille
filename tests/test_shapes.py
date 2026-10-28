import math
from dataclasses import replace

import numpy as np
import pytest

from helpers import cube, grid_plane, uv_sphere
from pointille.core import DistributeOptions, Material, MeshData, Part, PartOptions, Texture, compose
from pointille.core import shapes as S
from pointille.core.sampling import triangle_areas_normals


def ellipse_points(cx, cy, a, b, angle, n=200, noise=0.0, seed=0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = math.cos(angle), math.sin(angle)
    xy = np.stack([cx + a * np.cos(t) * c - b * np.sin(t) * s, cy + a * np.cos(t) * s + b * np.sin(t) * c], 1)
    return xy + np.random.default_rng(seed).normal(0.0, noise, xy.shape)


def test_fit_ellipse():
    e = S.fit_ellipse(ellipse_points(3.0, -1.0, 5.0, 2.0, 0.4, noise=0.02))
    assert np.allclose(e.center, (3.0, -1.0), atol=0.02) and np.allclose(e.axes, (5.0, 2.0), atol=0.03)
    assert math.cos(2 * (e.angle - 0.4)) > 0.999  # the major axis either way round
    assert S.as_ellipse(ellipse_points(0, 0, 3.0, 2.0, 0.0, noise=0.03), 1.0) is not None


def test_no_ellipse_where_there_is_none():
    square = np.array([[x, 0.0] for x in np.linspace(0, 4, 20)] + [[4.0, y] for y in np.linspace(0, 4, 20)]
                      + [[x, 4.0] for x in np.linspace(4, 0, 20)] + [[0.0, y] for y in np.linspace(4, 0, 20)])
    assert S.as_ellipse(square, 1.0) is None
    arc = ellipse_points(0, 0, 3.0, 3.0, 0.0)[:60]  # a third of a circle is a curve, not a ring
    assert S.as_ellipse(arc, 1.0) is None


def test_straighten():
    x = np.linspace(0.0, 10.0, 41)
    wobbly = np.stack([x, 0.05 * np.sin(x * 3)], 1)
    line = S.straighten(wobbly, 1.0)
    d = line[-1] - line[0]
    assert np.allclose((line[:, 0] - line[0, 0]) * d[1] - (line[:, 1] - line[0, 1]) * d[0], 0.0)
    assert np.array_equal(line[[0, -1]], wobbly[[0, -1]])
    bent = np.stack([x, 0.5 * np.sin(x / 10 * np.pi)], 1)
    assert np.array_equal(S.straighten(bent, 1.0), bent)


def test_ring_for_an_eye():
    near, s = 1.0, 1.5
    assert S.ring(S.Ellipse(np.zeros(2), np.array([0.3, 0.25]), 0.0, 0.0), near, s) is None  # far too small
    drawn, count, pupil = S.ring(S.Ellipse(np.zeros(2), np.array([0.8, 0.7]), 0.0, 0.0), near, s)
    assert count >= S.RING_MIN and not pupil
    pts = drawn.points(count)
    assert np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1).min() >= near - 1e-9
    assert drawn.axes[0] <= 0.8 * S.EYE_GROW + 1e-9
    drawn, count, pupil = S.ring(S.Ellipse(np.zeros(2), np.array([3.0, 2.5]), 0.0, 0.0), near, s)
    assert pupil and count == round(drawn.perimeter() / s)


def pumpkin():
    body = uv_sphere(10.0, 30, 60).triangles * np.array([1.0, 1.0, 0.75]) + np.array([0.0, 0.0, 7.5])
    stem = cube(1.0, 2).triangles + np.array([0.0, 0.0, 15.5])
    return MeshData(np.concatenate([body, stem]))


def test_round_bodies():
    m = pumpkin()
    areas, _ = triangle_areas_normals(m.triangles)
    bodies = S.round_bodies(m.triangles, np.arange(m.triangle_count), areas, 3.0)
    assert len(bodies) == 1 and bodies[0].radius == pytest.approx(10.0, rel=0.05)
    assert bodies[0].height == pytest.approx(7.5, rel=0.05)
    for flat in (cube(10.0, 4), grid_plane(size=(20.0, 20.0))):  # a box and a plane are no round bodies
        a, _ = triangle_areas_normals(flat.triangles)
        assert not S.round_bodies(flat.triangles, np.arange(flat.triangle_count), a, 3.0)


def test_rings_on_a_sphere():
    m = uv_sphere(10.0, 40, 80)
    _, normals = triangle_areas_normals(m.triangles)
    pos, tri, bary, ring = S.rings(m.triangles, np.arange(m.triangle_count), normals, 3.0, (0, -1, 0), 2, 1.5)
    assert np.allclose(np.einsum("ij,ijk->ik", bary, m.triangles[tri]), pos)  # on the surface
    levels = np.unique(ring)
    assert abs(len(levels) - math.pi * 10.0 / 3.0) <= 1  # rows a spacing apart along the meridian
    z = np.array([pos[ring == r, 2].mean() for r in levels])
    assert np.all(np.diff(z) > 0)  # from the bottom up
    for r in levels:  # each row level, its drones evenly round it
        p = pos[ring == r]
        assert np.ptp(p[:, 2]) < 1e-6
        if len(p) > 4:
            rad = np.hypot(p[:, 0], p[:, 1]).mean()
            assert abs(len(p) - 2 * math.pi * rad / 1.5) <= 1


def front(**kw):
    kw.setdefault("count", 1000)
    kw.setdefault("target_min_distance", 1.0)
    kw.setdefault("view_dir", (0.0, -1.0, -0.3))
    return DistributeOptions(facade=True, **kw)


def test_a_round_body_is_filled_with_rings():
    m = pumpkin()
    r = compose([Part(m)], front())
    fill = r.positions[r.kind == 2]
    assert len(fill) > 20
    # the fill stands on a few levels (the rows of the rings), the lattice would not
    levels = np.unique(np.round(fill[:, 2], 3))
    assert len(levels) <= 0.25 * len(fill)
    flat = compose([Part(m, PartOptions(shapes=False))], front())
    flat_fill = flat.positions[flat.kind == 2]
    assert len(np.unique(np.round(flat_fill[:, 2], 3))) > 0.5 * len(flat_fill)
    assert r.min_distance >= 1.0 - 1e-9


def test_rings_go_round_the_back_and_all_round():
    m = pumpkin()
    r = compose([Part(m)], front(hidden_density=0.25))
    back = r.positions[(r.kind == 2) & (r.positions[:, 1] > 2.0)]
    assert len(back) > 3 and len(np.unique(np.round(back[:, 2], 3))) <= 0.5 * len(back)
    whole = compose([Part(m)], DistributeOptions(count=1000, target_min_distance=1.0))  # no facade
    fill = whole.positions[whole.kind == 2]
    assert (fill[:, 1] > 2.0).any() and (fill[:, 1] < -2.0).any()
    assert len(np.unique(np.round(fill[:, 2], 3))) <= 0.25 * len(fill)


def test_rings_on_request():
    box = cube(12.0, 6)
    r = compose([Part(box, PartOptions(pattern="rings"))], DistributeOptions(count=1000, target_min_distance=1.0))
    fill = r.positions[r.kind == 2]
    assert len(fill) > 10 and len(np.unique(np.round(fill[:, 2], 3))) <= 0.3 * len(fill)


def smiley(eyes=((-6.0, 4.0, 2.0, 2.4), (6.0, 4.0, 2.0, 2.4)), size=40.0, px=400):
    """A plane facing -Y: a yellow field with dark elliptic eyes (x, z, half width, half height)."""
    img = np.zeros((px, px, 3), np.uint8)
    img[:] = (255, 205, 30)
    yy, xx = np.mgrid[0:px, 0:px]
    X, Z = (xx + 0.5) / px * size - size / 2, size / 2 - (yy + 0.5) / px * size
    for x, z, a, b in eyes:
        img[((X - x) / a) ** 2 + ((Z - z) / b) ** 2 < 1.0] = (25, 15, 10)
    m = grid_plane(size=(size, size), res=8)
    m.materials = [Material(texture=Texture(img))]
    m.material_ids = np.zeros(m.triangle_count, np.int64)
    return m


def ring_round(r, x, z, inner=1.5, outer=3.0, least=6):
    """Line drones on a ring round the point (x, z) of the plane, all round it."""
    d = np.hypot(r.positions[:, 0] - x, r.positions[:, 2] - z)
    on = (d > inner) & (d < outer) & (r.kind < 2)
    sectors = np.floor((np.arctan2(r.positions[on, 2] - z, r.positions[on, 0] - x) + np.pi) / (np.pi / 3)) % 6
    return int(on.sum()) >= least and len(np.unique(sectors)) == 6


def test_eyes_are_rings():
    r = compose([Part(smiley())], front(view_dir=(0.0, -1.0, 0.0)))
    assert ring_round(r, -6.0, 4.0) and ring_round(r, 6.0, 4.0)
    assert r.min_distance >= 1.0 - 1e-9


def test_a_pair_of_eyes_is_drawn_alike():
    # two small eyes a little apart in size: both rings (grown alike) or both dots, never one of each
    eyes = ((-6.0, 4.0, 0.62, 0.62), (6.0, 4.0, 0.55, 0.55))
    r = compose([Part(smiley(eyes))], front(view_dir=(0.0, -1.0, 0.0)))
    near = [np.sort(np.hypot(r.positions[:, 0] - x, r.positions[:, 2] - z))[:6] for x, z, _, _ in eyes]
    assert np.allclose(near[0], near[1], atol=0.35)


def carved(size=40.0, px=400):
    """A plane facing -Y: an orange field with a dark triangle (a carved eye) and its corners."""
    img = np.zeros((px, px, 3), np.uint8)
    img[:] = (230, 120, 20)
    corners = np.array([(-10.0, -6.0), (10.0, -6.0), (0.0, 10.0)])  # (x, z)
    yy, xx = np.mgrid[0:px, 0:px]
    X, Z = (xx + 0.5) / px * size - size / 2, size / 2 - (yy + 0.5) / px * size
    inside = np.ones(X.shape, bool)
    for i in range(3):
        a, b = corners[i], corners[(i + 1) % 3]
        inside &= (b[0] - a[0]) * (Z - a[1]) - (b[1] - a[1]) * (X - a[0]) >= 0
    img[inside] = (40, 10, 5)
    m = grid_plane(size=(size, size), res=8)
    m.materials = [Material(texture=Texture(img))]
    m.material_ids = np.zeros(m.triangle_count, np.int64)
    return m, corners


@pytest.mark.parametrize("facade", [True, False])
def test_a_carved_triangle_is_outlined_with_its_corners(facade):
    m, corners = carved()
    opt = DistributeOptions(count=1000, target_min_distance=1.0, facade=facade, view_dir=(0.0, -1.0, 0.0))
    r = compose([Part(m)], opt)
    xz = r.positions[:, [0, 2]]
    for c in corners:  # a corner drone at each tip of the triangle
        d = np.linalg.norm(xz[r.kind == 0] - c, axis=1)
        assert len(d) and d.min() < 1.0
    # its edges are drawn: line drones all along them, and no ring stands in for it
    for i in range(3):
        a, b = corners[i], corners[(i + 1) % 3]
        t = np.linspace(0.15, 0.85, 8)[:, None]
        pts = a + t * (b - a)
        d = np.linalg.norm(xz[r.kind < 2][None] - pts[:, None], axis=2).min(axis=1)
        assert d.max() < 1.6
    centre = corners.mean(axis=0)
    assert not ring_round(r, *centre, inner=1.5, outer=4.0)


def test_short_of_drones_the_fill_gives_way_first():
    m, _ = carved()
    opt = DistributeOptions(count=1000, target_min_distance=1.0, facade=True, view_dir=(0.0, -1.0, 0.0))
    full = compose([Part(m)], opt)
    lines = int((full.kind < 2).sum())
    short = compose([Part(m)], replace(opt, count=lines + max((full.kind == 2).sum() // 3, 1)))
    assert short.spread == 1.0 and short.fill_spread > 1.0  # the lines as asked, the fill sparser
    assert abs(int((short.kind < 2).sum()) - lines) <= 0.05 * lines
    assert short.count <= short.budget
