"""The object Pointille creates: a mesh of bare vertices, one per point.

The vertices are in a vertex group named as the Skybrush formation group of the
object, so the object can be put into a formation as it is.

Nothing is placed on the vertices unless the user picks an object for that
(``Object.pointille_instance``); it is then instanced on every vertex with
Geometry Nodes, and the vertices themselves stay a plain mesh.

The source model is never edited. Only its object transform changes, and
only when the user asks to scale the formation.
"""

import bpy
import numpy as np
from mathutils import Matrix, Vector

from ._vendor import core, export

DRONES_KEY = "pointille_drones"
SOURCES_KEY = "pointille_sources"
COLOR_ATTR = "drone_color"
TEXTURE_ATTR = "pointille_texture_color"  # the texture colours, to undo hand-painting
VISIBILITY_ATTR = "pointille_visibility"  # share of the audience that sees the vertex (facade mode)
FORMATION_GROUP = "Pointille"  # vertex group of all vertices, the markers of a Skybrush formation
INSTANCES_KEY = "pointille_instances"
INSTANCES_NAME = "Pointille Instances"
# 0.1 showed glowing balls on the vertices as a separate child object
LEGACY_PREVIEW_KEY = "pointille_preview"
LEGACY_GLOW_MATERIAL = "Pointille Drone Glow"


def is_ours(obj):
    return obj is not None and bool(obj.get(DRONES_KEY) or obj.get(LEGACY_PREVIEW_KEY))


def source_objects(drones):
    names = str(drones.get(SOURCES_KEY, "")).split("\n")
    return [bpy.data.objects[n] for n in names if n in bpy.data.objects]


def find_drones(obj):
    """Vertex object for a source, a vertex object or an old preview object."""
    if obj is None:
        return None
    if obj.get(DRONES_KEY):
        return obj
    if obj.get(LEGACY_PREVIEW_KEY):
        return obj.parent if obj.parent and obj.parent.get(DRONES_KEY) else None
    for o in bpy.data.objects:
        if o.get(DRONES_KEY) and obj.name in str(o.get(SOURCES_KEY, "")).split("\n"):
            return o
    return None


def resolve_sources(context):
    """Selected models; a selected vertex object stands for its model."""
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
    if me.is_editmode:  # attribute data is only reachable through the edit mesh then
        import bmesh

        bm = bmesh.from_edit_mesh(me)
        layer = bm.verts.layers.float_color.get(COLOR_ATTR)
        if layer is None:
            return np.ones((len(bm.verts), 3))
        return np.array([v[layer][:3] for v in bm.verts], dtype=np.float64).reshape(-1, 3)
    attr = me.color_attributes.get(COLOR_ATTR)
    if attr is None:
        return np.ones((len(me.vertices), 3))
    buf = np.empty(len(me.vertices) * 4, np.float32)
    attr.data.foreach_get("color", buf)
    return buf.reshape(-1, 4)[:, :3].astype(np.float64)


def paint_selected(obj, rgb=None):
    """Edit mode: give the selected vertices a colour, or their texture colour back
    with ``rgb=None``. Returns how many vertices changed."""
    import bmesh

    bm = bmesh.from_edit_mesh(obj.data)
    layers = bm.verts.layers.float_color
    target = layers.get(COLOR_ATTR)
    source = layers.get(TEXTURE_ATTR)
    if target is None or (rgb is None and source is None):
        return 0
    changed = 0
    for v in bm.verts:
        if v.select:
            v[target] = (*rgb, 1.0) if rgb is not None else v[source]
            changed += 1
    bmesh.update_edit_mesh(obj.data)
    obj.update_from_editmode()  # vertices and groups of the mesh match the edit mesh again
    return changed


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


def group_members(obj):
    """Vertex groups of the current layout: name and (vertex index, weight) pairs of each."""
    groups = [(g.name, []) for g in obj.vertex_groups]
    if groups:
        for v in obj.data.vertices:
            for g in v.groups:
                if g.group < len(groups):
                    groups[g.group][1].append((v.index, g.weight))
    return groups


def restore_groups(obj, groups, old_count, active_index):
    """Give a new layout the vertex groups of the old one (they belong to the mesh).

    Skybrush builds a formation from a vertex group and pins every drone to a
    ``Skybrush[Vertex i]`` group, so a group that held the whole old layout takes
    the whole new one and any other group keeps its vertex indices.
    """
    count = len(obj.data.vertices)
    for name, pairs in groups:
        group = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
        if old_count and len(pairs) == old_count:
            group.add(list(range(count)), 1.0, "REPLACE")
            continue
        for i, weight in pairs:
            if i < count:
                group.add([i], weight, "REPLACE")
    if groups:
        obj.vertex_groups.active_index = active_index


def ensure_formation_group(obj):
    """Make the vertices a Skybrush formation as they are, without Edit Mode.

    Skybrush puts drones on the vertices of the group named in the object's
    *Drone Show* panel (``skybrush.formation_vertex_group``). A group the user
    named there stays as it is, a part of the vertices too. With no name (or no
    group of that name) the group is made of all vertices and named there, as
    if Assign and *Use selected* had been pressed in Edit Mode.
    """
    sb = getattr(obj, "skybrush", None)  # None without Skybrush Studio
    name = (sb.formation_vertex_group if sb is not None else "") or FORMATION_GROUP
    if obj.vertex_groups.get(name) is None:
        active = obj.vertex_groups.active_index if len(obj.vertex_groups) else None
        obj.vertex_groups.new(name=name).add(list(range(len(obj.data.vertices))), 1.0, "REPLACE")
        if active is not None:  # the user's active group stays active
            obj.vertex_groups.active_index = active
    if sb is not None and sb.formation_vertex_group != name:
        sb.formation_vertex_group = name
    return name


def write_drones(context, sources, result):
    anchor = sources[0]
    name = f"{anchor.name}_points"
    obj = find_drones(anchor)

    mesh = bpy.data.meshes.new(name)
    n = result.count
    mesh.vertices.add(n)
    mesh.vertices.foreach_set("co", result.positions.astype(np.float32).ravel())
    rgba = np.ones((n, 4), np.float32)
    rgba[:, :3] = result.colors_linear
    for attr_name in (COLOR_ATTR, TEXTURE_ATTR):
        mesh.color_attributes.new(attr_name, "FLOAT_COLOR", "POINT").data.foreach_set("color", rgba.ravel())
    if result.visibility is not None:
        mesh.attributes.new(VISIBILITY_ATTR, "FLOAT", "POINT").data.foreach_set(
            "value", result.visibility.astype(np.float32))
    mesh.update()

    if obj is None:
        obj = bpy.data.objects.new(name, mesh)
        collections = anchor.users_collection or [context.scene.collection]
        collections[0].objects.link(obj)
    else:
        groups, old_count, active = group_members(obj), len(obj.data.vertices), obj.vertex_groups.active_index
        old = obj.data
        obj.data = mesh
        if old.users == 0:
            bpy.data.meshes.remove(old)
        restore_groups(obj, groups, old_count, active)
    ensure_formation_group(obj)
    obj[DRONES_KEY] = 1
    obj[SOURCES_KEY] = "\n".join(s.name for s in sources)
    # follow the model when it is moved; vertices are stored in world space
    obj.parent = anchor
    try:
        obj.matrix_parent_inverse = anchor.matrix_world.inverted()
    except ValueError:  # degenerate model transform
        obj.parent = None
    obj.matrix_basis = Matrix.Identity(4)
    remove_legacy_preview(obj)
    apply_instance(obj)
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


def _instances_tree():
    """Geometry Nodes: keep the vertices and add the chosen object on each of them."""
    tree = bpy.data.node_groups.new(INSTANCES_NAME, "GeometryNodeTree")
    tree[INSTANCES_KEY] = 1
    if hasattr(tree, "interface"):  # Blender 4.0+
        tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    else:
        tree.inputs.new("NodeSocketGeometry", "Geometry")
        tree.outputs.new("NodeSocketGeometry", "Geometry")
    if hasattr(tree, "is_modifier"):
        tree.is_modifier = True
    nodes, links = tree.nodes, tree.links
    vertices = nodes.new("NodeGroupInput")
    # the object itself (any type, its own materials), placed by its origin ...
    shape = nodes.new("GeometryNodeObjectInfo")
    shape.name = "Object"
    shape.inputs["As Instance"].default_value = True
    # ... turned and sized as it is in the scene: scaling the model does not resize it
    pose = nodes.new("GeometryNodeObjectInfo")
    pose.name = "Object Pose"
    pose.transform_space = "RELATIVE"
    place = nodes.new("GeometryNodeInstanceOnPoints")
    join = nodes.new("GeometryNodeJoinGeometry")
    out = nodes.new("NodeGroupOutput")
    vertices.location, shape.location, pose.location = (-500, 0), (-500, -150), (-500, -400)
    place.location, join.location, out.location = (-200, -100), (50, 0), (250, 0)
    links.new(vertices.outputs[0], place.inputs["Points"])
    links.new(shape.outputs["Geometry"], place.inputs["Instance"])
    links.new(pose.outputs["Rotation"], place.inputs["Rotation"])
    links.new(pose.outputs["Scale"], place.inputs["Scale"])
    links.new(vertices.outputs[0], join.inputs[0])
    links.new(place.outputs["Instances"], join.inputs[0])
    links.new(join.outputs[0], out.inputs[0])
    return tree


def instances_modifier(points):
    return next(
        (m for m in points.modifiers
         if m.type == "NODES" and m.node_group is not None and m.node_group.get(INSTANCES_KEY)),
        None,
    )


def apply_instance(points):
    """Put ``points.pointille_instance`` on every vertex; with none set, leave bare vertices."""
    target = points.pointille_instance
    mod = instances_modifier(points)
    if target is None:
        if mod is not None:
            tree = mod.node_group
            points.modifiers.remove(mod)
            if tree.users == 0:
                bpy.data.node_groups.remove(tree)
        return
    if mod is None:
        mod = points.modifiers.new(INSTANCES_NAME, "NODES")
        mod.node_group = _instances_tree()
    elif mod.node_group.users > 1:  # shared with a duplicate of this object
        mod.node_group = mod.node_group.copy()
    for name in ("Object", "Object Pose"):
        node = mod.node_group.nodes.get(name)
        if node is not None:
            node.inputs["Object"].default_value = target


def remove_legacy_preview(points):
    """Drop the glowing balls that version 0.1 put on the vertices."""
    for child in [c for c in points.children if c.get(LEGACY_PREVIEW_KEY)]:
        trees = [m.node_group for m in child.modifiers if m.type == "NODES" and m.node_group is not None]
        mesh = child.data
        bpy.data.objects.remove(child)
        for tree in trees:
            if tree.users == 0:
                bpy.data.node_groups.remove(tree)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    glow = bpy.data.materials.get(LEGACY_GLOW_MATERIAL)
    if glow is not None and glow.users == 0:
        bpy.data.materials.remove(glow)


def export_csv(drones, path):
    pts = world_positions(drones)  # Blender is already Z-up, metres
    colors = core.color.to_srgb8(drone_colors_linear(drones))
    export.write_csv(path, pts, colors)
