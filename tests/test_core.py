import math

import numpy as np
import pytest

from helpers import QUADRANTS, cube, grid_plane, quadrant_texture, textured_plane, uv_sphere
from pointille.core import DistributeOptions, Material, MeshData, Texture, distribute
from pointille.core.color import glow, linear_to_srgb, sample_texture, srgb_to_linear
from pointille.core.relax import closest_on_triangles
from pointille.core.sampling import poisson_radius
from pointille.core.spatial import Grid, nearest


@pytest.fixture(scope="module")
def sphere():
    return uv_sphere()


def ideal_spacing(result):
    return 2.0 * poisson_radius(result.area, result.count)


@pytest.mark.parametrize("count", [1, 2, 17, 500])
@pytest.mark.parametrize("relax", [0, 10])
def test_exact_count(sphere, count, relax):
    r = distribute(sphere, DistributeOptions(count=count, relax_iterations=relax, color_mode="none"))
    assert r.count == count
    assert r.colors_linear.shape == (count, 3)
    assert len(np.unique(r.positions.round(12), axis=0)) == count


def test_points_lie_on_surface(sphere):
    r = distribute(sphere, DistributeOptions(count=400, color_mode="none"))
    rebuilt = np.einsum("ij,ijk->ik", r.barycentrics, sphere.triangles[r.triangle_ids])
    assert np.allclose(rebuilt, r.positions, atol=1e-9)
    assert np.all(r.barycentrics > -1e-9)
    assert np.allclose(r.barycentrics.sum(axis=1), 1.0)


def test_uniform_spacing(sphere):
    raw = distribute(sphere, DistributeOptions(count=800, relax_iterations=0, color_mode="none"))
    relaxed = distribute(sphere, DistributeOptions(count=800, color_mode="none"))
    d = ideal_spacing(raw)
    assert raw.min_distance > 0.65 * d
    assert relaxed.min_distance > 0.75 * d
    assert relaxed.mean_distance > 0.87 * d
    assert relaxed.min_distance >= raw.min_distance


def test_deterministic(sphere):
    a = distribute(sphere, DistributeOptions(count=200, seed=7))
    b = distribute(sphere, DistributeOptions(count=200, seed=7))
    c = distribute(sphere, DistributeOptions(count=200, seed=8))
    assert np.array_equal(a.positions, b.positions)
    assert not np.array_equal(a.positions, c.positions)


def test_mesh_is_not_modified():
    mesh = textured_plane()
    before = (mesh.triangles.copy(), mesh.uvs.copy(), mesh.materials[0].texture.pixels.copy())
    distribute(mesh, DistributeOptions(count=100, facade=True, view_dir=(0, -1, 0)))
    assert np.array_equal(before[0], mesh.triangles)
    assert np.array_equal(before[1], mesh.uvs)
    assert np.array_equal(before[2], mesh.materials[0].texture.pixels)


def test_facade_normals(sphere):
    r = distribute(sphere, DistributeOptions(count=300, facade=True, view_dir=(0, -1, 0), facade_angle=60))
    assert np.all(r.normals @ np.array([0, -1, 0]) > math.cos(math.radians(60)) - 1e-9)
    # a hemisphere-ish cap: area far below the full sphere
    assert r.area < 0.3 * 4 * math.pi


def test_facade_occlusion():
    # the front plane (y=-1) hides the left half (x<0) of the back plane (y=0)
    back = grid_plane(size=(4, 2), res=8, y=0.0)
    front = grid_plane(size=(2, 2), res=4, y=-1.0, center=(-1.0, 0.0))
    mesh = MeshData.concatenate([back, front])
    r = distribute(mesh, DistributeOptions(count=200, facade=True, view_dir=(0, -1, 0)))
    on_back = np.abs(r.positions[:, 1]) < 1e-9
    assert on_back.any()
    assert np.all(r.positions[on_back, 0] > -1e-6)
    # visible area: the front plane (4) plus the uncovered half of the back plane (4)
    assert abs(r.area - 8.0) < 0.6


def test_facade_without_visible_surface(sphere):
    plane = grid_plane(facing=1)
    with pytest.raises(ValueError):
        distribute(plane, DistributeOptions(count=10, facade=True, view_dir=(0, -1, 0)))



def test_audience_spread_reaches_the_sides():
    # straight on, the audience sees one face of a cube; spread over 60° it sees the sides too
    box = cube()
    narrow = distribute(box, DistributeOptions(count=300, facade=True, view_dir=(0, -1, 0)))
    wide = distribute(box, DistributeOptions(count=300, facade=True, view_dir=(0, -1, 0), audience_spread=60))
    assert np.all(narrow.normals[:, 1] < -0.99)
    side = np.abs(wide.normals[:, 0]) > 0.99
    assert side.sum() > 60
    assert np.all(wide.normals[:, 1] < 0.5) and np.all(np.abs(wide.normals[:, 2]) < 0.5)  # no back, top, bottom
    # the sides are seen by part of the audience only, so they get fewer drones per area
    assert np.all(wide.visibility[side] < 1) and np.all(wide.visibility[~side] == 1)
    assert side.sum() < (~side).sum()
    assert wide.min_distance > 0.75 * ideal_spacing(wide)


def test_hidden_density_keeps_the_back_sparse(sphere):
    r = distribute(sphere, DistributeOptions(count=400, facade=True, view_dir=(0, -1, 0), hidden_density=0.25))
    front, back = r.positions[:, 1] < -0.2, r.positions[:, 1] > 0.2
    assert back.sum() > 40
    assert front.sum() > 2.5 * back.sum()
    assert np.all(r.visibility[front] == 1) and np.all(r.visibility[back] == 0)
    # a quarter of the density: hidden drones stand about twice as far apart
    d, _ = nearest(r.positions, r.positions, 0.1, exclude_self=True)
    assert np.median(d[back]) > 1.6 * np.median(d[front])
    assert r.min_distance > 0.7 * ideal_spacing(r)


def test_hidden_density_one_is_the_whole_surface(sphere):
    r = distribute(sphere, DistributeOptions(count=300, facade=True, hidden_density=1.0))
    assert r.area == pytest.approx(4 * math.pi, rel=0.02)
    assert (r.positions[:, 1] > 0.2).sum() > 100


def test_camera_sees_in_perspective():
    # a plane 1 m in front of the camera hides twice its width of a plane 2 m away
    back = grid_plane(size=(6, 2), res=12, y=0.0)
    front = grid_plane(size=(1, 2), res=2, y=-1.0)
    mesh = MeshData.concatenate([back, front])
    r = distribute(mesh, DistributeOptions(count=300, facade=True, view_point=(0.0, -2.0, 0.0)))
    on_back = np.abs(r.positions[:, 1]) < 1e-9
    assert on_back.sum() > 100
    assert np.all(np.abs(r.positions[on_back, 0]) > 1.0 - 1e-6)
    assert np.all(r.visibility == 1)
    # a far-away audience sees the back plane right next to the front one
    far = distribute(mesh, DistributeOptions(count=300, facade=True, view_dir=(0, -1, 0)))
    far_back = np.abs(far.positions[:, 1]) < 1e-9
    assert np.any(np.abs(far.positions[far_back, 0]) < 0.9)


def test_visibility_without_facade(sphere):
    r = distribute(sphere, DistributeOptions(count=50))
    assert np.all(r.visibility == 1)

def test_bad_count(sphere):
    with pytest.raises(ValueError):
        distribute(sphere, DistributeOptions(count=0))


def test_sample_texture_orientation():
    tex = Texture(quadrant_texture())
    uv = np.array([[0.25, 0.75], [0.75, 0.75], [0.25, 0.25], [0.75, 0.25]])
    got = np.round(linear_to_srgb(sample_texture(tex, uv)) * 255)
    expect = [QUADRANTS[k] for k in ("top_left", "top_right", "bottom_left", "bottom_right")]
    assert np.array_equal(got, np.array(expect, float))


def quadrant_of(p):
    return ("top" if p[2] > 0 else "bottom") + "_" + ("left" if p[0] < 0 else "right")


@pytest.mark.parametrize("mode", ["point", "area"])
def test_drone_colors_follow_texture(mode):
    mesh = textured_plane(size=(2, 2), res=8)
    r = distribute(mesh, DistributeOptions(count=150, color_mode=mode))
    colors = r.colors_srgb8
    margin = 0.25 if mode == "area" else 0.05
    checked = 0
    for p, c in zip(r.positions, colors):
        if min(abs(p[0]), abs(p[2])) < margin:
            continue  # near a color border
        assert tuple(c) == QUADRANTS[quadrant_of(p)]
        checked += 1
    assert checked > 50


def test_base_color_and_vertex_colors_multiply():
    mesh = grid_plane(res=2)
    mesh.materials = [Material(base_color=(0.5, 1.0, 1.0))]
    mesh.material_ids = np.zeros(mesh.triangle_count, np.int64)
    mesh.vertex_colors = np.tile([1.0, 0.25, 1.0], (mesh.triangle_count, 3, 1))
    r = distribute(mesh, DistributeOptions(count=20, color_mode="point"))
    assert np.allclose(r.colors_linear, [0.5, 0.25, 1.0])


def test_srgb_roundtrip():
    x = np.linspace(0, 1, 101)
    assert np.allclose(linear_to_srgb(srgb_to_linear(x)), x)


# black, dark brown, brown, gold, white, near-black noise (sRGB 0..255)
GLOW_CASES = [(0, 0, 0), (60, 35, 18), (120, 70, 35), (240, 200, 60), (255, 255, 255), (8, 3, 6)]


def glow8(srgb8, floor):
    return np.round(linear_to_srgb(glow(srgb_to_linear(np.asarray(srgb8, float) / 255), floor)) * 255)


def test_drones_never_go_black():
    """An LED cannot be black: dark colours glow dim but keep their hue and their order of brightness."""
    black, dark_brown, brown, gold, white, noise = glow8(GLOW_CASES, 0.4)
    assert black.max() == round(0.4 * 255) and np.ptp(black) == 0  # a dim white, not off
    assert np.array_equal(noise, black)  # near-black has no hue left to keep, only noise
    assert dark_brown[0] > dark_brown[1] > dark_brown[2]  # still brown, only brighter
    assert black.max() < dark_brown.max() < brown.max() < gold.max() <= white.max()
    assert tuple(white) == (255, 255, 255)


def test_glow_off_dark_and_black_models():
    lin = srgb_to_linear(np.array(GLOW_CASES, float) / 255)
    assert np.array_equal(glow(lin, 0.0), lin)  # 0: the model's colours as they are
    dark_red = glow8([(40, 10, 10), (90, 20, 20)], 0.4)
    assert dark_red[1][0] == 255 and dark_red[0][0] < 255  # a dark model is stretched up to full brightness
    noisy_black = glow8(np.random.default_rng(0).integers(2, 22, (50, 3)), 0.4)
    assert np.ptp(noisy_black.max(axis=1)) <= 10  # glows evenly (within 4 %): its noise does not make drones flicker


def test_distribute_lights_a_dark_model():
    mesh = grid_plane(res=2)
    brown = srgb_to_linear(np.array([0.25, 0.15, 0.08]))
    mesh.materials = [Material(base_color=tuple(brown))]
    mesh.material_ids = np.zeros(mesh.triangle_count, np.int64)
    lit = distribute(mesh, DistributeOptions(count=20, color_mode="point", min_brightness=0.4))
    srgb = linear_to_srgb(lit.colors_linear)
    assert (srgb.max(axis=1) > 0.4).all() and (srgb[:, 0] > srgb[:, 1]).all() and (srgb[:, 1] > srgb[:, 2]).all()
    plain = distribute(mesh, DistributeOptions(count=20, color_mode="point"))
    assert np.allclose(plain.colors_linear, brown)  # the core keeps the colours unless asked


def test_scale_to_min_distance(sphere):
    r = distribute(sphere, DistributeOptions(count=300, scale_mode="min_distance", target_min_distance=2.0))
    assert r.summary()["min_distance"] == pytest.approx(2.0)
    p = r.scaled_positions()
    d, _ = nearest(p, p, 1.0, exclude_self=True)
    assert d.min() == pytest.approx(2.0)


def test_scale_to_height(sphere):
    r = distribute(sphere, DistributeOptions(count=300, scale_mode="height", target_height=30.0, up_axis=2))
    p = r.scaled_positions()
    assert np.ptp(p[:, 2]) == pytest.approx(30.0)
    # scaling keeps the bottom centre in place
    assert p[:, 2].min() == pytest.approx(r.positions[:, 2].min())


def test_grid_matches_brute_force():
    rng = np.random.default_rng(1)
    pts = rng.random((500, 3))
    q = rng.random((50, 3))
    qi, pi, d = Grid(pts, 0.2).query_radius(q, 0.2)
    got = set(zip(qi.tolist(), pi.tolist()))
    full = np.linalg.norm(q[:, None] - pts[None], axis=2)
    want = set(zip(*np.nonzero(full < 0.2)))
    assert got == {(int(a), int(b)) for a, b in want}
    dist, idx = nearest(q, pts, 0.01)
    assert np.array_equal(idx, full.argmin(axis=1))
    assert np.allclose(dist, full.min(axis=1))


def test_closest_point_on_triangle():
    rng = np.random.default_rng(3)
    tri = rng.normal(size=(3, 3))
    pts = rng.normal(size=(300, 3)) * 2
    k = len(pts)
    bary = closest_on_triangles(pts, np.tile(tri[0], (k, 1)), np.tile(tri[1], (k, 1)), np.tile(tri[2], (k, 1)))
    got = bary @ tri
    s = rng.random((20000, 2))
    s[s.sum(1) > 1] = 1 - s[s.sum(1) > 1]
    dense = tri[0] + s[:, :1] * (tri[1] - tri[0]) + s[:, 1:] * (tri[2] - tri[0])
    brute = np.linalg.norm(pts[:, None] - dense[None], axis=2).min(axis=1)
    assert np.all(np.linalg.norm(got - pts, axis=1) <= brute + 1e-9)
    assert np.all(bary > -1e-9)


def test_concatenate_keeps_materials():
    a, b = textured_plane(), grid_plane(y=1.0)
    m = MeshData.concatenate([a, b])
    assert len(m.materials) == 2
    assert set(m.material_ids[: a.triangle_count]) == {0}
    assert set(m.material_ids[a.triangle_count :]) == {1}
    assert m.materials[1].texture is None
