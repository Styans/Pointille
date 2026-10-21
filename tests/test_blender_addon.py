"""Headless Blender check of the add-on (runs when the ``bpy`` module is installed:
``pip install bpy`` on the Python version that Blender release uses)."""

import csv
import hashlib
import sys
import types
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


def test_vertices_come_in_a_vertex_group(addon, model):
    # what Edit Mode > select all > Assign did by hand: every vertex in one group, ready for a formation
    bpy.ops.pointille.distribute(count=30)
    points = drones_of(addon, model)
    assert group_contents(points) == {addon.drones.FORMATION_GROUP: list(range(30))}
    assert points.vertex_groups.active.name == addon.drones.FORMATION_GROUP  # what "Use selected" picks
    bpy.ops.pointille.distribute(count=45)  # a new layout: the group takes all of it
    assert group_contents(points) == {addon.drones.FORMATION_GROUP: list(range(45))}


def test_rerun_keeps_vertex_groups(addon, model):
    # Skybrush makes a formation of one group and pins each drone to a "Skybrush[Vertex i]" group
    bpy.ops.pointille.distribute(count=30)
    points = drones_of(addon, model)
    points.vertex_groups.new(name="Formation").add(list(range(30)), 1.0, "REPLACE")
    points.vertex_groups.new(name="Skybrush[Vertex 7]").add([7], 1.0, "REPLACE")
    points.vertex_groups.new(name="Skybrush[Vertex 29]").add([29], 1.0, "REPLACE")
    points.vertex_groups.new(name="Wing").add([3, 4, 5], 1.0, "REPLACE")
    points.vertex_groups.active_index = points.vertex_groups["Formation"].index
    everything = addon.drones.FORMATION_GROUP

    bpy.ops.pointille.distribute(count=30, seed=3)
    assert group_contents(points) == {
        everything: list(range(30)), "Formation": list(range(30)),
        "Skybrush[Vertex 7]": [7], "Skybrush[Vertex 29]": [29], "Wing": [3, 4, 5],
    }
    assert points.vertex_groups.active.name == "Formation"
    bpy.ops.pointille.distribute(count=20)  # fewer vertices: vertex 29 is gone
    assert group_contents(points) == {
        everything: list(range(20)), "Formation": list(range(20)),
        "Skybrush[Vertex 7]": [7], "Skybrush[Vertex 29]": [], "Wing": [3, 4, 5],
    }


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
    s.audience_spread, s.hidden_density = 0.0, 0.0
    bpy.ops.pointille.distribute(count=60)
    pts = addon.drones.world_positions(drones_of(addon, model))
    assert np.all(pts[:, 1] < 0)  # front = side facing -Y


def test_facade_from_camera_keeps_hidden_parts(addon, model):
    s = bpy.context.scene.pointille
    s.area_mode, s.view, s.hidden_density = "FACADE", "CAMERA", 25.0
    with pytest.raises(RuntimeError):  # no camera yet
        bpy.ops.pointille.distribute(count=60)
    cam =bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    centre = np.mean([model.matrix_world @ v.co for v in model.data.vertices], axis=0)
    cam.location = centre + np.array([0.0, -8.0, 0.0])
    cam.rotation_euler = (np.pi / 2, 0.0, 0.0)  # looking along +Y at the model
    assert bpy.ops.pointille.distribute(count=120) == {"FINISHED"}
    drones = drones_of(addon, model)
    pts = addon.drones.world_positions(drones)
    vis = np.empty(len(pts), np.float32)
    drones.data.attributes[addon.drones.VISIBILITY_ATTR].data.foreach_get("value", vis)
    front, back = pts[:, 1] < centre[1], pts[:, 1] > centre[1]
    assert back.any() and front.sum() > 2 * back.sum()
    assert vis[front].mean() > 0.5 and vis[back].mean() < 0.25
    assert np.all(vis[pts[:, 1] > centre[1] + 0.7] == 0)  # the far side of the unit sphere


def test_export_csv(addon, model, tmp_path):
    bpy.ops.pointille.distribute(count=40)
    bpy.context.view_layer.objects.active = drones_of(addon, model)
    path = tmp_path / "drones.csv"
    assert bpy.ops.pointille.export_csv(filepath=str(path)) == {"FINISHED"}
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 40 and rows[0]["hex"].startswith("#")


@pytest.fixture()
def skybrush():
    """Skybrush Studio, when installed next to Blender (not on CI). Use it from ONE test:
    switching it on and off several times in a session crashes Blender."""
    import addon_utils

    if addon_utils.enable("ui_skybrush_studio", default_set=False) is None:
        pytest.skip("Skybrush Studio is not installed")
    yield
    addon_utils.disable("ui_skybrush_studio", default_set=False)


def test_paint_selected_vertices(addon, model):
    import bmesh

    bpy.ops.pointille.distribute(count=30)
    points = drones_of(addon, model)
    texture = addon.drones.drone_colors_linear(points).copy()
    for o in bpy.context.view_layer.objects:
        o.select_set(o == points)
    bpy.context.view_layer.objects.active = points
    bpy.ops.object.mode_set(mode="EDIT")
    for v in bmesh.from_edit_mesh(points.data).verts:
        v.select = v.index < 5
    bpy.context.scene.pointille.paint_color = (0.0, 0.0, 1.0)
    assert bpy.ops.pointille.paint_vertices() == {"FINISHED"}
    painted = addon.drones.drone_colors_linear(points)
    assert np.allclose(painted[:5], (0.0, 0.0, 1.0)) and np.allclose(painted[5:], texture[5:])
    assert bpy.ops.pointille.paint_vertices(texture=True) == {"FINISHED"}
    assert np.allclose(addon.drones.drone_colors_linear(points), texture)
    bpy.ops.object.mode_set(mode="OBJECT")


def style(**kw):
    return {**dict(style="TEXTURE", mix=1.0, brightness=1.0, saturation=1.0, speed=0.0, spread=0.5,
                   color=[1.0, 0.0, 0.0], color2=[0.0, 0.0, 1.0]), **kw}


def test_led_styles(addon):
    sb = addon.skybrush
    base = np.array([[0.6, 0.2, 0.1], [0.05, 0.05, 0.05], [0.9, 0.9, 0.9]])
    pos, still, times = np.array([0.0, 0.5, 1.0]), np.zeros(1), np.linspace(0, 4, 9)
    assert np.allclose(sb.styled(base, pos, still, style())[0], base)
    gray = sb.styled(base, pos, still, style(style="GRAYSCALE"))[0]
    assert np.allclose(gray[:, 0], gray[:, 1]) and np.allclose(gray[:, 1], gray[:, 2])
    assert gray[2, 0] > gray[0, 0] > gray[1, 0]  # the texture's light and dark stay
    half = sb.styled(base, pos, still, style(style="GRAYSCALE", mix=0.5))[0]
    assert np.allclose(half, (base + gray) / 2)  # the share of the style blends it with the texture
    solid = sb.styled(base, pos, still, style(style="SOLID", color=[0.0, 1.0, 0.0]))[0]
    assert np.allclose(solid, [0.0, 1.0, 0.0])
    grad = sb.styled(base, pos, still, style(style="GRADIENT", color=[1.0, 0.0, 0.0], color2=[0.0, 0.0, 1.0]))[0]
    assert np.allclose(grad[0], [1, 0, 0]) and np.allclose(grad[2], [0, 0, 1]) and 0 < grad[1, 0] < 1
    for name in ("SHIMMER", "RAINBOW"):
        moving = sb.styled(base, pos, times, style(style=name, speed=0.25))
        assert moving.shape == (9, 3, 3) and np.isfinite(moving).all() and moving.min() >= 0 and moving.max() <= 1
        assert np.allclose(moving[0], moving[-1] if name == "RAINBOW" else moving[0])  # 4 s = one cycle
        assert np.abs(moving[0] - moving[2]).max() > 0.1  # and it moves in between
    # lightness of the texture survives the shimmer: the darkest point stays the darkest
    shimmer = sb.styled(base, pos, times, style(style="SHIMMER", speed=0.25))
    assert (shimmer[:, 1].max(axis=1) < shimmer[:, 0].max(axis=1)).all()


@pytest.fixture()
def show(addon, model, skybrush):
    """40 Pointille vertices as formation "shape" (frames 30-50) after a formation "start" (1-6);
    drone i stands on vertex 39 - i."""
    scene = bpy.context.scene
    scene.pointille.color_mode = "POINT"
    bpy.ops.pointille.distribute(count=40)
    points = drones_of(addon, model)
    # no Edit Mode, Assign or "Use selected": the vertices are a formation as they come
    assert points.skybrush.formation_vertex_group == addon.drones.FORMATION_GROUP
    shape, start = bpy.data.collections.new("shape"), bpy.data.collections.new("start")
    shape.objects.link(points)
    fleet = bpy.data.collections.new("fleet")
    scene.collection.children.link(fleet)
    scene.skybrush.settings.drone_collection = fleet
    for i in range(40):
        start.objects.link(bpy.data.objects.new(f"start {i}", None))  # a formation before the shape
        fleet.objects.link(bpy.data.objects.new(f"Drone {i + 1}", None))
    storyboard = scene.skybrush.storyboard
    storyboard.add_new_entry(name="start", frame_start=1, duration=5, formation=start).update_mapping(list(range(40)))
    entry = storyboard.add_new_entry(name="shape", frame_start=30, duration=20, formation=shape)
    entry.update_mapping([39 - i for i in range(40)])  # drone i stands on vertex 39 - i
    bpy.context.view_layer.objects.active = points
    return types.SimpleNamespace(scene=scene, points=points, fleet=fleet, entry=entry, vertex_of_drone=39 - np.arange(40))


def led_colors(show, frame):
    """What Skybrush exports for every drone at a frame, 0-255."""
    from sbstudio.plugin.utils.sampling import sample_colors_of_objects

    lights = sample_colors_of_objects(list(show.fleet.objects), [frame])
    return np.array([(c.r, c.g, c.b) for c in (lights[d.name].colors[0] for d in show.fleet.objects)], dtype=float)


def add_layer(show, segment=""):
    show.scene.pointille.skybrush_segment = segment
    assert bpy.ops.pointille.layer_add() == {"FINISHED"}
    return show.scene.skybrush.light_effects.active_entry


def check_vertex_colors(addon, show):
    vertex = addon.drones.drone_colors_linear(show.points)
    flight = add_layer(show, "start -> shape")  # the flight into the shape, then the shape itself
    layer = add_layer(show)
    assert [e.name for e in addon.skybrush.pointille_effects(show.scene)] == [flight.name, layer.name]
    for frame in (15, 40):  # while flying in, the drone already shows the vertex it flies to
        assert np.abs(led_colors(show, frame) - np.round(vertex[show.vertex_of_drone] * 255)).max() <= 1
    assert (led_colors(show, 60) == 255).all()  # afterwards: their own white

    show.entry.update_mapping(list(range(40)))  # Skybrush recalculated the transitions ...
    bpy.context.view_layer.update()
    assert np.abs(led_colors(show, 40) - np.round(vertex * 255)).max() <= 1  # ... the colours followed
    show.scene.pointille.led_brightness = 0.5  # edits the selected layer
    assert np.abs(led_colors(show, 40) - np.round(vertex * 0.5 * 255)).max() <= 1
    show.scene.pointille.led_brightness = 1.0

    # a rainbow: one row per time step, read through Output Y = Temporal, time counted from frame 0
    show.scene.pointille.led_style = "RAINBOW"
    show.scene.pointille.led_speed = 0.5
    rows = layer.color_image.size[1]
    assert layer.output_y == "TEMPORAL" and rows > 1 and layer.color_image.size[0] == 40
    seconds = (layer.duration - 1) / show.scene.render.fps
    params = addon.skybrush.effect_params(layer)
    for frame in (32, 41, 49):
        row = min(int(rows * (frame - layer.frame_start) / (layer.duration - 1)), rows - 1)
        times = np.array([layer.frame_start / show.scene.render.fps + (row + 0.5) / rows * seconds])
        want = addon.skybrush.render(show.scene, show.entry, params, times)[0, :, :3]
        assert np.abs(led_colors(show, frame) - np.round(want * 255)).max() <= 1
    assert np.abs(led_colors(show, 32) - led_colors(show, 41)).max() > 30  # the colours really move
    # every layer keeps its own style
    effects = show.scene.skybrush.light_effects
    effects.active_entry_index = list(effects.entries).index(flight)
    assert show.scene.pointille.led_style == "TEXTURE"
    effects.active_entry_index = list(effects.entries).index(layer)
    assert show.scene.pointille.led_style == "RAINBOW" and show.scene.pointille.led_speed == pytest.approx(0.5)
    assert np.abs(led_colors(show, 15) - np.round(vertex * 255)).max() <= 1  # (the mapping is the identity by now)


def check_layers_split_off_and_gold(addon, show):
    """One formation: texture, then the LEDs go dark, then gold - each in its own frame range."""
    scene, sb = show.scene, addon.skybrush
    vertex = addon.drones.drone_colors_linear(show.points)
    first = add_layer(show)
    assert (first.frame_start, first.frame_end) == (30, 49)  # the whole formation
    scene.frame_set(1)  # outside the layer: nothing to split
    try:
        bpy.ops.pointille.layer_split()
        refused = False
    except RuntimeError:
        refused = True
    assert refused and len(sb.pointille_effects(scene)) == 1

    scene.frame_set(38)
    assert bpy.ops.pointille.layer_split() == {"FINISHED"}
    layers = sb.pointille_effects(scene)
    assert [(e.frame_start, e.frame_end) for e in layers] == [(30, 37), (38, 49)]
    second = scene.skybrush.light_effects.active_entry
    assert second == layers[1] and second.color_image != layers[0].color_image  # each part has its own image
    scene.pointille.led_style = "OFF"  # the second part goes dark ...

    scene.frame_set(44)
    assert bpy.ops.pointille.layer_split() == {"FINISHED"}
    third = scene.skybrush.light_effects.active_entry
    assert (third.frame_start, third.frame_end) == (44, 49) and layers[1].frame_end == 43
    scene.pointille.led_style = "SOLID"  # ... and the third is gold
    scene.pointille.led_color = (1.0, 0.7, 0.0)

    want = np.round(vertex[show.vertex_of_drone] * 255)
    assert np.abs(led_colors(show, 33) - want).max() <= 1  # texture
    assert (led_colors(show, 40) == 0).all()  # dark
    gold = np.round(np.array([1.0, 0.7, 0.0]) * 255)
    assert np.abs(led_colors(show, 47) - gold).max() <= 1  # gold (8-bit image: rounding)

    # move the range with the current frame: gold now starts at 46
    scene.frame_set(46)
    assert bpy.ops.pointille.layer_range(which="START") == {"FINISHED"}
    assert third.frame_start == 46
    assert (led_colors(show, 45) == 255).all()  # frames 44-45 are covered by no layer now: their own white
    assert np.abs(led_colors(show, 47) - gold).max() <= 1


def check_group_made_in_edit_mode(addon, show):
    """Part of the vertices as the formation: Edit Mode > select > Assign (new group) > Use selected."""
    import bmesh
    from sbstudio.plugin.model.formation import count_markers_in_formation

    points, formation = show.points, show.entry.formation
    vertex = addon.drones.drone_colors_linear(points)
    for o in bpy.context.view_layer.objects:
        o.select_set(o == points)
    bpy.context.view_layer.objects.active = points
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(points.data)
    for v in bm.verts:
        v.select = v.index < 10
    bm.select_flush_mode()
    bmesh.update_edit_mesh(points.data)
    assert bpy.ops.object.vertex_group_assign_new() == {"FINISHED"}
    mine = points.vertex_groups.active.name
    assert bpy.ops.skybrush.use_selected_vertex_group_for_formation() == {"FINISHED"}
    bpy.ops.object.mode_set(mode="OBJECT")

    assert points.skybrush.formation_vertex_group == mine != addon.drones.FORMATION_GROUP
    assert count_markers_in_formation(formation) == 10  # Skybrush's markers ...
    colors = addon.skybrush.marker_colors(formation)  # ... and the colours Pointille gives them
    assert np.allclose(colors[:, :3], vertex[:10]) and (colors[:, 3] == 1).all()

    bpy.ops.pointille.distribute(count=40)  # a new layout keeps the user's choice and group
    assert points.skybrush.formation_vertex_group == mine
    assert group_contents(points)[mine] == list(range(10))
    assert group_contents(points)[addon.drones.FORMATION_GROUP] == list(range(40))
    assert count_markers_in_formation(formation) == 10

    points.skybrush.formation_vertex_group = ""  # nothing named: the whole layout again
    bpy.ops.pointille.distribute(count=40)
    assert points.skybrush.formation_vertex_group == addon.drones.FORMATION_GROUP
    assert count_markers_in_formation(formation) == 40


def test_skybrush_layers(addon, model, show):
    check_vertex_colors(addon, show)
    effects = show.scene.skybrush.light_effects
    while addon.skybrush.pointille_effects(show.scene):  # start over with no layers and Skybrush's mapping
        effects.active_entry_index = list(effects.entries).index(addon.skybrush.pointille_effects(show.scene)[0])
        assert addon.skybrush.remove_layer(bpy.context)
    show.entry.update_mapping([int(m) for m in show.vertex_of_drone])
    show.scene.frame_set(1)
    check_layers_split_off_and_gold(addon, show)
    check_group_made_in_edit_mode(addon, show)
