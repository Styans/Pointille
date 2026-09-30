"""Headless Blender check of the add-on (runs when the ``bpy`` module is installed:
``pip install bpy`` on the Python version that Blender release uses)."""

import csv
import hashlib
import sys
import zipfile

import numpy as np
import pytest

from helpers import QUADRANTS

bpy = pytest.importorskip("bpy")

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "tools"))
import build_blender_addon  # noqa: E402


@pytest.fixture(scope="module")
def addon(tmp_path_factory):
    out = tmp_path_factory.mktemp("addon")
    zip_path = build_blender_addon.build(out)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out / "unzipped")
    sys.path.insert(0, str(out / "unzipped"))
    import pointille_blender

    pointille_blender.register()
    yield pointille_blender
    pointille_blender.unregister()


@pytest.fixture()
def model(addon, textured_glb):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(textured_glb))
    obj = next(o for o in bpy.data.objects if o.type == "MESH")
    for o in bpy.data.objects:
        o.select_set(o == obj)
    bpy.context.view_layer.objects.active = obj
    return obj


def model_fingerprint(obj):
    me = obj.data
    h = hashlib.sha256()
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    h.update(co.tobytes())
    uv = np.empty(len(me.loops) * 2, np.float32)
    me.uv_layers[0].data.foreach_get("uv", uv)
    h.update(uv.tobytes())
    for mat in me.materials:
        h.update(str(sorted(n.bl_idname for n in mat.node_tree.nodes)).encode())
        for n in mat.node_tree.nodes:
            if n.bl_idname == "ShaderNodeTexImage":
                px = np.empty(len(n.image.pixels), np.float32)
                n.image.pixels.foreach_get(px)
                h.update(px.tobytes())
    return h.hexdigest()


def drones_of(addon, obj):
    return addon.drones.find_drones(obj)


def test_distribute_creates_colored_vertices(addon, model):
    before = model_fingerprint(model)
    bpy.context.scene.pointille.color_mode = "POINT"
    assert bpy.ops.pointille.distribute(count=150) == {"FINISHED"}
    assert model_fingerprint(model) == before

    drones = drones_of(addon, model)
    assert drones is not None and drones.parent == model
    assert drones.name == f"{model.name}_points"
    me = drones.data
    assert len(me.vertices) == 150 and len(me.edges) == 0 and len(me.polygons) == 0
    assert not drones.modifiers and not drones.children  # bare vertices, nothing placed on them
    assert drones["pointille_count"] == 150

    pts = addon.drones.world_positions(drones)
    # the importer turns glTF Y-up into Blender Z-up: sphere centre (0, 2, 0) -> (0, 0, 2)
    r = np.linalg.norm(pts - [0, 0, 2], axis=1)
    assert np.all((r > 0.99) & (r < 1.0 + 1e-4))

    colors = addon.pointille.core.color.to_srgb8(addon.drones.drone_colors_linear(drones))
    checked = 0
    for p, c in zip(pts, colors):
        # model space of the fixture: z_model = -y_blender, y_model = z_blender - 2
        mz, my = -p[1], p[2] - 2.0
        if min(abs(mz), abs(my)) < 0.2:
            continue
        quad = ("top" if mz > 0 else "bottom") + "_" + ("left" if my > 0 else "right")
        assert tuple(int(v) for v in c) == QUADRANTS[quad]
        checked += 1
    assert checked > 40


def test_object_on_vertices(addon, model):
    marker = bpy.data.objects.new("marker", bpy.data.meshes.new("marker"))
    marker.data.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
    bpy.context.scene.collection.objects.link(marker)
    marker.location = (30.0, 0.0, 0.0)  # where it stands does not matter
    marker.rotation_euler = (0.0, 0.0, 0.5)
    marker.scale = (0.25, 0.25, 0.25)
    bpy.ops.pointille.distribute(count=50)
    points = drones_of(addon, model)
    points.pointille_instance = marker

    s = bpy.context.scene.pointille
    s.scale_mode = "MIN_DISTANCE"
    s.target_min_distance = 2.0
    assert bpy.ops.pointille.apply_scale() == {"FINISHED"}  # the model grows after the object is placed
    assert model.matrix_world.to_scale()[0] > 3
    placed = [
        inst.matrix_world.copy()
        for inst in bpy.context.evaluated_depsgraph_get().object_instances
        if inst.is_instance and inst.object.original == marker
    ]
    assert len(placed) == 50
    got = np.array(sorted(tuple(m.translation) for m in placed))
    want = np.array(sorted(map(tuple, addon.drones.world_positions(points))))
    assert np.allclose(got, want, atol=1e-4)
    for m in placed:  # the object kept its own size and turn
        assert tuple(m.to_scale()) == pytest.approx((0.25, 0.25, 0.25), rel=1e-4)
        assert tuple(m.to_euler()) == pytest.approx((0.0, 0.0, 0.5), abs=1e-4)

    bpy.ops.pointille.distribute(count=40)  # a new layout keeps the object on it
    assert addon.drones.instances_modifier(points) is not None
    points.pointille_instance = None
    assert not points.modifiers


def test_rerun_removes_old_glowing_balls(addon, model):
    bpy.ops.pointille.distribute(count=30)
    points = drones_of(addon, model)
    old = bpy.data.objects.new("old_preview", bpy.data.meshes.new("old_preview"))
    bpy.context.scene.collection.objects.link(old)
    old["pointille_preview"] = 1
    old.parent = points
    bpy.ops.pointille.distribute(count=30)
    assert "old_preview" not in bpy.data.objects and not points.children


def test_rerun_reuses_object(addon, model):
    bpy.ops.pointille.distribute(count=60)
    bpy.ops.pointille.distribute(count=90)
    ours = [o for o in bpy.data.objects if o.get("pointille_drones")]
    assert len(ours) == 1 and len(ours[0].data.vertices) == 90


def group_contents(obj):
    return {
        g.name: [v.index for v in obj.data.vertices if any(x.group == g.index for x in v.groups)]
        for g in obj.vertex_groups
    }


def test_rerun_keeps_vertex_groups(addon, model):
    # Skybrush makes a formation of one group and pins each drone to a "Skybrush[Vertex i]" group
    bpy.ops.pointille.distribute(count=30)
    points = drones_of(addon, model)
    points.vertex_groups.new(name="Formation").add(list(range(30)), 1.0, "REPLACE")
    points.vertex_groups.new(name="Skybrush[Vertex 7]").add([7], 1.0, "REPLACE")
    points.vertex_groups.new(name="Skybrush[Vertex 29]").add([29], 1.0, "REPLACE")

    bpy.ops.pointille.distribute(count=30, seed=3)
    assert group_contents(points) == {
        "Formation": list(range(30)), "Skybrush[Vertex 7]": [7], "Skybrush[Vertex 29]": [29],
    }
    bpy.ops.pointille.distribute(count=20)  # fewer vertices: vertex 29 is gone
    assert group_contents(points) == {"Formation": list(range(20)), "Skybrush[Vertex 7]": [7], "Skybrush[Vertex 29]": []}


def test_scale_to_min_distance(addon, model):
    before = model_fingerprint(model)
    s = bpy.context.scene.pointille
    s.scale_mode = "MIN_DISTANCE"
    s.target_min_distance = 2.0
    bpy.ops.pointille.distribute(count=100)
    drones = drones_of(addon, model)
    assert drones["pointille_min_distance"] == pytest.approx(2.0, rel=1e-4)
    assert model.matrix_world.to_scale()[0] > 5  # the transform grew ...
    assert model_fingerprint(model) == before  # ... the mesh data did not


def test_facade_front(addon, model):
    s = bpy.context.scene.pointille
    s.area_mode = "FACADE"
    s.view = "FRONT"
    bpy.ops.pointille.distribute(count=60)
    pts = addon.drones.world_positions(drones_of(addon, model))
    assert np.all(pts[:, 1] < 0)  # front = side facing -Y


def test_export_csv(addon, model, tmp_path):
    bpy.ops.pointille.distribute(count=40)
    bpy.context.view_layer.objects.active = drones_of(addon, model)
    path = tmp_path / "drones.csv"
    assert bpy.ops.pointille.export_csv(filepath=str(path)) == {"FINISHED"}
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 40 and rows[0]["hex"].startswith("#")


@pytest.fixture()
def skybrush():
    """Skybrush Studio, when installed next to Blender (not on CI)."""
    import addon_utils

    if addon_utils.enable("ui_skybrush_studio", default_set=False) is None:
        pytest.skip("Skybrush Studio is not installed")
    yield
    addon_utils.disable("ui_skybrush_studio", default_set=False)


def test_skybrush_drones_take_vertex_colors(addon, model, skybrush):
    from sbstudio.plugin.utils.sampling import sample_colors_of_objects

    bpy.context.scene.pointille.color_mode = "POINT"
    bpy.ops.pointille.distribute(count=40)
    points = drones_of(addon, model)
    points.vertex_groups.new(name="Formation").add(list(range(40)), 1.0, "REPLACE")
    points.skybrush.formation_vertex_group = "Formation"
    formation = bpy.data.collections.new("shape")
    formation.objects.link(points)
    fleet = bpy.data.collections.new("fleet")
    bpy.context.scene.collection.children.link(fleet)
    bpy.context.scene.skybrush.settings.drone_collection = fleet
    where = addon.drones.world_positions(points)
    for i in range(40):  # drone i stands on marker 39 - i
        drone = bpy.data.objects.new(f"Drone {i + 1}", None)
        drone.location = where[39 - i]
        fleet.objects.link(drone)
    storyboard = bpy.context.scene.skybrush.storyboard
    entry = storyboard.add_new_entry(name="shape", frame_start=10, duration=20, formation=formation)
    entry.update_mapping([39 - i for i in range(40)])

    bpy.context.view_layer.objects.active = points
    assert bpy.ops.pointille.skybrush_colors() == {"FINISHED"}
    lights = sample_colors_of_objects(list(fleet.objects), [15, 40])  # in the formation and after it
    inside, after = (
        np.array([(c.r, c.g, c.b) for c in (lights[d.name].colors[k] for d in fleet.objects)], dtype=float)
        for k in (0, 1)
    )
    want = np.round(addon.drones.drone_colors_linear(points)[39 - np.arange(40)] * 255)
    assert np.abs(inside - want).max() <= 1  # each drone shows the vertex it stands on
    assert len(np.unique(inside, axis=0)) > 2 and (after == 255).all()  # outside: its own white
