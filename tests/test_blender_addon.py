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
    me = drones.data
    assert len(me.vertices) == 150 and len(me.edges) == 0 and len(me.polygons) == 0
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


def test_preview_object(addon, model):
    bpy.ops.pointille.distribute(count=80)
    drones = drones_of(addon, model)
    prev = addon.drones.preview_object(drones)
    assert prev is not None and prev.modifiers[0].type == "NODES"
    evaluated = prev.evaluated_get(bpy.context.evaluated_depsgraph_get())
    assert len(evaluated.data.vertices) >= 80 * 12  # an icosphere per drone
    # switching the toggle off removes the preview (property update callback)
    bpy.context.view_layer.objects.active = drones
    bpy.context.scene.pointille.preview = False
    assert addon.drones.preview_object(drones) is None


def test_rerun_reuses_object(addon, model):
    bpy.ops.pointille.distribute(count=60)
    bpy.ops.pointille.distribute(count=90)
    ours = [o for o in bpy.data.objects if o.get("pointille_drones")]
    assert len(ours) == 1 and len(ours[0].data.vertices) == 90


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
