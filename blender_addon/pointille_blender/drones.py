"""The objects Pointille creates: the drone vertex cloud and its glowing preview.

The source model is never edited. Only its object transform changes, and
only when the user asks to scale the formation.
"""

import bpy
import numpy as np
from mathutils import Matrix, Vector

from ._vendor import core, export

DRONES_KEY = "pointille_drones"
SOURCES_KEY = "pointille_sources"
PREVIEW_KEY = "pointille_preview"
COLOR_ATTR = "drone_color"
GLOW_MATERIAL = "Pointille Drone Glow"


def is_ours(obj):
    return obj is not None and (obj.get(DRONES_KEY) or obj.get(PREVIEW_KEY))


def source_objects(drones):
    names = str(drones.get(SOURCES_KEY, "")).split("\n")
    return [bpy.data.objects[n] for n in names if n in bpy.data.objects]


def find_drones(obj):
    """Drone object for a source, a drone object or a preview object."""
    if obj is None:
        return None
    if obj.get(DRONES_KEY):
        return obj
    if obj.get(PREVIEW_KEY):
        return obj.parent if obj.parent and obj.parent.get(DRONES_KEY) else None
    for o in bpy.data.objects:
        if o.get(DRONES_KEY) and obj.name in str(o.get(SOURCES_KEY, "")).split("\n"):
            return o
    return None


def resolve_sources(context):
    """Selected models; a selected drone object stands for its model."""
    picked = list(context.selected_objects)
    if context.active_object is not None and context.active_object not in picked:
        picked.append(context.active_object)
    out = []
    for obj in picked:
        if is_ours(obj):
            drones = find_drones(obj)
            candidates = source_objects(drones) if drones else []
        elif obj.type == "MESH":
            candidates = [obj]
        else:
            candidates = []
        for c in candidates:
            if c not in out:
                out.append(c)
    return out


def world_positions(drones):
    me = drones.data
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    mw = np.array(drones.matrix_world, dtype=np.float64)
    return co.reshape(-1, 3).astype(np.float64) @ mw[:3, :3].T + mw[:3, 3]


def drone_colors_linear(drones):
    me = drones.data
    attr = me.color_attributes.get(COLOR_ATTR)
    if attr is None:
        return np.ones((len(me.vertices), 3))
    buf = np.empty(len(me.vertices) * 4, np.float32)
    attr.data.foreach_get("color", buf)
    return buf.reshape(-1, 4)[:, :3].astype(np.float64)


def measure(drones):
    """Store spacing and size (world units) on the drone object."""
    pts = world_positions(drones)
    if len(pts) > 1:
        extent = float(np.ptp(pts, axis=0).max()) or 1.0
        dmin, dmean = core.stats.spacing(pts, extent / max(len(pts), 1) ** 0.5)
        size = np.ptp(pts, axis=0)
    else:
        dmin = dmean = 0.0
        size = np.zeros(3)
    drones["pointille_count"] = len(pts)
    drones["pointille_min_distance"] = dmin
    drones["pointille_mean_distance"] = dmean
    drones["pointille_size"] = [float(v) for v in size]
    return dmin, dmean, size


def write_drones(context, sources, result):
    anchor = sources[0]
    name = f"{anchor.name}_drones"
    obj = find_drones(anchor)

    mesh = bpy.data.meshes.new(name)
    n = result.count
    mesh.vertices.add(n)
    mesh.vertices.foreach_set("co", result.positions.astype(np.float32).ravel())
    attr = mesh.color_attributes.new(COLOR_ATTR, "FLOAT_COLOR", "POINT")
    rgba = np.ones((n, 4), np.float32)
    rgba[:, :3] = result.colors_linear
    attr.data.foreach_set("color", rgba.ravel())
    mesh.update()

    if obj is None:
        obj = bpy.data.objects.new(name, mesh)
        collections = anchor.users_collection or [context.scene.collection]
        collections[0].objects.link(obj)
    else:
        old = obj.data
        obj.data = mesh
        if old.users == 0:
            bpy.data.meshes.remove(old)
    obj[DRONES_KEY] = 1
    obj[SOURCES_KEY] = "\n".join(s.name for s in sources)
    # follow the model when it is moved; vertices are stored in world space
    obj.parent = anchor
    try:
        obj.matrix_parent_inverse = anchor.matrix_world.inverted()
    except ValueError:  # degenerate model transform
        obj.parent = None
    obj.matrix_basis = Matrix.Identity(4)
    context.view_layer.update()
    measure(obj)
    return obj


def apply_scale(context, drones, factor):
    """Uniformly scale the model(s) and drones about the formation's bottom centre."""
    if factor <= 0 or abs(factor - 1.0) < 1e-9:
        return
    pts = world_positions(drones)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pivot = Vector(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]))
    m = Matrix.Translation(pivot) @ Matrix.Scale(factor, 4) @ Matrix.Translation(-pivot)
    roots = []
    for obj in source_objects(drones) + [drones]:
        while obj.parent is not None:
            obj = obj.parent
        if obj not in roots:
            roots.append(obj)
    for root in roots:
        root.matrix_world = m @ root.matrix_world
    context.view_layer.update()
    measure(drones)


def glow_material():
    mat = bpy.data.materials.get(GLOW_MATERIAL)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(GLOW_MATERIAL)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    attr = nodes.new("ShaderNodeAttribute")
    attr.attribute_type = "GEOMETRY"
    attr.attribute_name = COLOR_ATTR
    emit = nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = 3.0
    out = nodes.new("ShaderNodeOutputMaterial")
    attr.location, emit.location, out.location = (-300, 0), (0, 0), (250, 0)
    links.new(attr.outputs["Color"], emit.inputs["Color"])
    links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def _preview_tree(name, drones, radius):
    tree = bpy.data.node_groups.get(name)
    if tree is None:
        tree = bpy.data.node_groups.new(name, "GeometryNodeTree")
        if hasattr(tree, "interface"):  # Blender 4.0+
            tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        else:
            tree.outputs.new("NodeSocketGeometry", "Geometry")
        nodes, links = tree.nodes, tree.links
        out = nodes.new("NodeGroupOutput")
        info = nodes.new("GeometryNodeObjectInfo")
        info.name = "Drones"
        info.transform_space = "RELATIVE"
        ico = nodes.new("GeometryNodeMeshIcoSphere")
        ico.name = "Ball"
        ico.inputs["Subdivisions"].default_value = 2
        inst = nodes.new("GeometryNodeInstanceOnPoints")
        real = nodes.new("GeometryNodeRealizeInstances")
        setm = nodes.new("GeometryNodeSetMaterial")
        setm.inputs["Material"].default_value = glow_material()
        for i, node in enumerate((info, ico, inst, real, setm, out)):
            node.location = (i * 200 - 600, 0 if node is not ico else -200)
        links.new(info.outputs["Geometry"], inst.inputs["Points"])
        links.new(ico.outputs["Mesh"], inst.inputs["Instance"])
        links.new(inst.outputs["Instances"], real.inputs["Geometry"])
        links.new(real.outputs["Geometry"], setm.inputs["Geometry"])
        links.new(setm.outputs["Geometry"], out.inputs[0])
    tree.nodes["Drones"].inputs["Object"].default_value = drones
    tree.nodes["Ball"].inputs["Radius"].default_value = radius
    return tree


def preview_object(drones):
    return next((c for c in drones.children if c.get(PREVIEW_KEY)), None)


def refresh_preview(context, drones):
    settings = context.scene.pointille
    prev = preview_object(drones)
    if not settings.preview:
        if prev is not None:
            tree = prev.modifiers[0].node_group if prev.modifiers else None
            bpy.data.objects.remove(prev)
            if tree is not None and tree.users == 0:
                bpy.data.node_groups.remove(tree)
        return None
    if prev is None:
        prev = bpy.data.objects.new(f"{drones.name}_preview", bpy.data.meshes.new(f"{drones.name}_preview"))
        for coll in drones.users_collection or [context.scene.collection]:
            coll.objects.link(prev)
        prev[PREVIEW_KEY] = 1
        prev.parent = drones
        prev.matrix_parent_inverse = Matrix.Identity(4)
        prev.hide_select = True
    # ball size in the drones' local units (world spacing / world scale)
    scale = max(drones.matrix_world.to_scale()) or 1.0
    spacing = float(drones.get("pointille_min_distance", 0.0)) or 0.1
    radius = settings.preview_size * spacing / scale
    mod = prev.modifiers.get("Pointille Preview") or prev.modifiers.new("Pointille Preview", "NODES")
    mod.node_group = _preview_tree(f"Pointille Preview {drones.name}", drones, radius)
    return prev


def export_csv(drones, path):
    pts = world_positions(drones)  # Blender is already Z-up, metres
    colors = core.color.to_srgb8(drone_colors_linear(drones))
    export.write_csv(path, pts, colors)
