import math

import numpy as np
import pytest

from helpers import grid_plane, quadrant_texture, textured_plane, uv_sphere
from pointille.core import (DistributeOptions, Material, MeshData, Part, PartOptions, Texture, auto_selected,
                            color_groups, compose)
from pointille.core.spatial import nn_distances
from pointille.core.color import srgb_to_linear


def triangle(side=20.0, k=10):
    """An equilateral triangle in XZ facing -Y, its base on z = 0."""
    a, b, c = np.array([-side / 2, 0, 0]), np.array([0, 0, side * math.sqrt(3) / 2]), np.array([side / 2, 0, 0])
    out = []
    P = lambda i, j: a + (b - a) * i / k + (c - a) * j / k  # noqa: E731
    for i in range(k):
        for j in range(k - i):
            out.append([P(i, j), P(i + 1, j), P(i, j + 1)])
            if i + j <= k - 2:
                out.append([P(i + 1, j), P(i + 1, j + 1), P(i, j + 1)])
    t = np.array(out)
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    t[n[:, 1] > 0] = t[n[:, 1] > 0][:, ::-1]
    return MeshData(t), (a, b, c)


def front(**kw):
    kw.setdefault("count", 500)
    kw.setdefault("target_min_distance", 1.5)
    return DistributeOptions(facade=True, view_dir=(0.0, -1.0, 0.0), **kw)


def test_triangle_corners_then_edges_then_fill():
    mesh, corners = triangle()
    r = compose([Part(mesh)], front())
    assert r.count == r.needed < r.budget == 500  # no more than the figure needs
    assert np.all(np.diff(r.kind) >= 0)  # corners, lines, fill, in that order
    corner = r.positions[r.kind == 0]
    for c in corners:
        assert np.linalg.norm(corner - c, axis=1).min() < 0.3
    assert (r.kind == 1).sum() > 15 and (r.kind == 2).sum() >= 1
    assert r.min_distance >= 1.5 - 1e-9
    # next to a corner the drones stand closer than along the middle of an edge
    gaps = nn_distances(r.positions[r.kind < 2], 1.0)
    near = np.linalg.norm(r.positions[r.kind < 2][:, None] - np.array(corners)[None], axis=2).min(axis=1) < 2.0
    assert gaps[near].mean() < gaps[~near].mean()


def test_budget_makes_the_figure_sparser():
    mesh, _ = triangle()
    full = compose([Part(mesh)], front())
    r = compose([Part(mesh)], front(count=full.count // 2))
    assert r.count <= r.budget and r.count >= 0.8 * r.budget
    assert r.needed == full.count and r.spread > 1.2
    assert r.min_distance >= 1.5 * r.spread - 1e-9
    assert (r.kind == 0).sum() == 3  # the corners stay


def test_safety_minimum_distance_whatever_the_lines():
    sphere = uv_sphere(10.0)
    r = compose([Part(sphere)], DistributeOptions(count=2000, target_min_distance=1.0))
    assert r.min_distance >= 1.0 - 1e-9 and r.count <= 2000


@pytest.mark.parametrize("pattern,angle", [("hex", 60.0), ("square", 90.0)])
def test_fill_lattice_is_regular(pattern, angle):
    plane = grid_plane(size=(30.0, 30.0), res=6)
    r = compose([Part(plane, PartOptions(role="fill", pattern=pattern))], front(target_min_distance=1.0))
    assert (r.kind == 2).all()
    p = r.positions[:, [0, 2]]
    d = np.linalg.norm(p[:, None] - p[None], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    assert np.allclose(nn, 3.0, atol=1e-6)  # fill_ratio 3 times the minimum distance
    # the directions to the neighbours are whole multiples of the lattice angle
    i = np.argmin(np.abs(p).sum(axis=1))  # a drone near the middle
    near = np.flatnonzero(np.abs(d[i] - 3.0) < 1e-6)
    dirs = np.degrees(np.arctan2(*(p[near] - p[i])[:, ::-1].T)) % angle
    assert np.allclose(np.minimum(dirs, angle - dirs), 0.0, atol=1e-6)


def test_noise_fill_and_more_drones_without_lines():
    plane = grid_plane(size=(30.0, 30.0), res=6)
    r = compose([Part(plane, PartOptions(role="fill", pattern="noise"))], front(target_min_distance=1.0))
    assert (r.kind == 2).all() and 80 < r.count < 160  # 900 m² at 3 m spacing: about 115
    assert r.min_distance > 1.5


def test_roles():
    mesh, _ = triangle()
    only_lines = compose([Part(mesh, PartOptions(role="outline"))], front())
    assert only_lines.count and (only_lines.kind < 2).all()
    only_fill = compose([Part(mesh, PartOptions(role="fill"))], front())
    assert only_fill.count and (only_fill.kind == 2).all()


def test_parts_keep_their_roles_and_the_budget_is_shared():
    left, _ = triangle()
    right, _ = triangle(30.0)
    right.triangles = right.triangles + np.array([30.0, 0.0, 0.0])
    r = compose([Part(left), Part(right, PartOptions(role="outline"))], front())
    assert set(np.unique(r.part)) == {0, 1}
    assert not (r.kind[r.part == 1] == 2).any()
    assert (r.kind[r.part == 0] == 2).any()
    small = compose([Part(left), Part(right, PartOptions(role="outline"))], front(count=r.count // 2))
    assert small.count <= r.count // 2 and small.spread > 1


def test_part_behind_another_takes_nothing():
    near_plane = grid_plane(size=(20.0, 20.0), res=4, y=-5.0)
    hidden = grid_plane(size=(10.0, 10.0), res=4, y=0.0)
    r = compose([Part(near_plane), Part(hidden)], front(target_min_distance=1.0, hidden_density=0.0))
    assert (r.part == 0).sum() > 0 and (r.part == 1).sum() == 0


def quadrant_plane(size=30.0):
    m = textured_plane(size=(size, size), res=12)
    return m


def test_color_groups_of_a_texture():
    g = color_groups(quadrant_plane(), 4)
    assert len(g.share) == 4 and np.allclose(g.share, 0.25, atol=0.03)
    srgb = np.round(np.clip(g.linear, 0, 1) ** (1 / 2.2) * 255)
    assert {tuple(int(v > 128) for v in c) for c in srgb} == {(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)}


def _quadrant_of(points):
    # textured_plane: u along +X, v along +Z; the texture's top-left (red) is -X, +Z
    return np.where(points[:, 2] > 0, np.where(points[:, 0] < 0, "red", "green"),
                    np.where(points[:, 0] < 0, "blue", "white"))


def test_pattern_fills_the_ticked_group_only():
    plane = quadrant_plane()
    g = color_groups(plane, 4)
    red = int(np.argmax(g.linear[:, 0] - g.linear[:, 1] - g.linear[:, 2]))
    ticked = [i == red for i in range(4)]
    r = compose([Part(plane, PartOptions(role="pattern", groups=g.linear, selected=ticked))],
                front(target_min_distance=0.5))
    fill = r.positions[r.kind == 2]
    assert len(fill) > 5 and set(_quadrant_of(fill)) == {"red"}


def test_pattern_by_eyedropper():
    plane = quadrant_plane()
    blue = srgb_to_linear(np.array([0.0, 0.0, 1.0]))
    r = compose([Part(plane, PartOptions(role="pattern", pick="eyedropper", picks=[blue], tolerance=0.1))],
                front(target_min_distance=0.5))
    fill = r.positions[r.kind == 2]
    assert len(fill) > 5 and set(_quadrant_of(fill)) == {"blue"}


def test_pattern_auto_leaves_the_main_colour():
    px = np.zeros((64, 64, 3), np.uint8)
    px[:] = (230, 120, 20)  # orange coat ...
    px[:, 40:52] = (20, 15, 10)  # ... and one wide black stripe
    plane = grid_plane(size=(30.0, 30.0), res=12)
    plane.materials = [Material(texture=Texture(px))]
    plane.material_ids = np.zeros(plane.triangle_count, np.int64)
    g = color_groups(plane, 2)
    assert auto_selected(g, 0.15).tolist() == [False, True]
    r = compose([Part(plane, PartOptions(role="pattern", pick="auto", group_count=2))],
                front(target_min_distance=0.5))
    fill = r.positions[r.kind == 2]
    u = (fill[:, 0] + 15.0) / 30.0  # texture column of each fill drone
    assert len(fill) > 3 and np.all((u > 40 / 64 - 0.02) & (u < 52 / 64 + 0.02))


def test_deterministic_and_mesh_untouched():
    mesh, _ = triangle()
    before = mesh.triangles.copy()
    a = compose([Part(mesh)], front(seed=4))
    b = compose([Part(mesh)], front(seed=4))
    assert np.array_equal(a.positions, b.positions)
    assert np.array_equal(mesh.triangles, before)


def test_scale_to_height_measures_the_distance_in_the_figure():
    mesh, _ = triangle(2.0, 6)  # 1.73 tall
    r = compose([Part(mesh)], front(target_min_distance=1.5, scale_mode="height", target_height=17.32))
    assert r.scale == pytest.approx(10.0, rel=1e-3)
    assert r.min_distance * r.scale >= 1.5 - 1e-9
    assert r.count > 10


def test_bad_part_options():
    mesh, _ = triangle()
    with pytest.raises(ValueError):
        compose([Part(mesh, PartOptions(role="tiger"))], front())
    with pytest.raises(ValueError):
        compose([], front())
    with pytest.raises(ValueError):
        compose([Part(mesh)], front(count=0))


# ---------------------------------------------------------------- facade drawing (``picture``)

def disc(radius=10.0, n=96):
    """A disc in XZ facing -Y, centred at z = radius."""
    a = np.linspace(0.0, 2.0 * math.pi, n + 1)
    rim = np.stack([radius * np.cos(a), np.zeros(n + 1), radius + radius * np.sin(a)], 1)
    centre = np.array([0.0, 0.0, radius])
    t = np.stack([np.broadcast_to(centre, (n, 3)), rim[1:], rim[:-1]], 1)
    nrm = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    t[nrm[:, 1] > 0] = t[nrm[:, 1] > 0][:, ::-1]
    return MeshData(t)


def test_silhouette_is_one_even_closed_line():
    r = compose([Part(disc())], front(target_min_distance=1.5))
    line = r.positions[r.kind < 2]
    assert (r.kind == 0).sum() == 0  # a circle has no corner
    radius = np.hypot(line[:, 0], line[:, 2] - 10.0)
    rim = radius > 9.0
    assert rim.sum() >= 25 and np.abs(radius[rim] - 10.0).max() < 0.3  # on the outline, not inside it
    gaps = nn_distances(line[rim], 1.0)
    assert gaps.std() / gaps.mean() < 0.05  # evenly spaced
    assert (r.kind == 2).sum() >= 5  # and the disc has a body


def test_narrow_rod_is_drawn_along_its_middle():
    rod = grid_plane(size=(30.0, 1.2), res=10)  # narrower than a line spacing (2.25)
    r = compose([Part(rod)], front(target_min_distance=1.5))
    line = r.positions[r.kind < 2]
    assert np.abs(line[:, 2]).max() < 0.35  # one row, along the middle
    assert 11 <= len(line) <= 16  # 30 / 2.25 and a drone at each end
    assert r.min_distance >= 1.5 - 1e-9


def test_thin_painted_band_is_one_line_in_its_colour():
    px = np.zeros((128, 128, 3), np.uint8)
    px[:] = (230, 230, 230)
    px[60:66] = (220, 20, 20)  # a red band ~1.4 units wide across a 30-unit plane
    plane = grid_plane(size=(30.0, 30.0), res=12)
    plane.materials = [Material(texture=Texture(px))]
    plane.material_ids = np.zeros(plane.triangle_count, np.int64)
    r = compose([Part(plane)], front(target_min_distance=1.5))
    srgb = r.colors_srgb8.astype(int)
    red = (r.kind < 2) & (srgb[:, 0] > srgb[:, 1] + 60)
    assert red.sum() >= 10
    z = r.positions[red, 2]
    assert z.std() < 0.3  # one row, not both edges of the band
    band_z = 15.0 - 63.0 / 128.0 * 30.0  # texture row 63 from the top
    assert abs(float(np.median(z)) - band_z) < 0.6


def test_huge_need_is_estimated_before_laying_out():
    import time
    plane = grid_plane(size=(60.0, 40.0), res=8)
    t = time.perf_counter()
    r = compose([Part(plane)], front(count=300, target_min_distance=0.05))
    assert time.perf_counter() - t < 15.0
    assert r.needed > 30 * r.budget and r.spread > 5.0
    assert 0.8 * r.budget <= r.count <= r.budget
    assert r.min_distance >= 0.05 * r.spread - 1e-9


def test_hidden_surface_gets_a_sparse_fill():
    sphere = uv_sphere(10.0)
    seen = compose([Part(sphere)], front(target_min_distance=1.5, hidden_density=0.0))
    both = compose([Part(sphere)], front(target_min_distance=1.5, hidden_density=0.25))
    back = both.positions[:, 1] > 2.0  # the audience is at -Y
    assert not (seen.positions[:, 1] > 2.0).any()
    assert back.sum() >= 3 and back.sum() < (~back).sum()  # ~9 at a quarter of the fill's density
    assert both.min_distance >= 1.5 - 1e-9
