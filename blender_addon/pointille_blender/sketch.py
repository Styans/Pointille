"""Curves a drawing is made of: built from a model (``core.trace``), edited by hand, read back for drones.

The drawing of a model is a curve object ``<model>_кривые``: one poly spline per line, its colour a
material (``Pointille кривая #rrggbb``, the viewport colour is what the drones show; ``Pointille срез
#rrggbb`` for the slices of a volume, whose drones come after the rest). A spline of one point is a single
drone (an eye too small for a ring). Its child ``<model>_оболочка`` is the smooth shell the fill stands
on; it is hidden and never rendered. A volume (``pointille_solid``) has no shell and no fill. Any other
curve object (drawn by hand, text converted to a curve) can take drones too: its own plane, no shell.
"""

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.geometry import interpolate_bezier

from ._vendor import core

SKETCH_KEY = "pointille_curves"
SHELL_KEY = "pointille_shell"
NORMAL_KEY = "pointille_normal"
SOURCE_KEY = "pointille_traced_from"
SOLID_KEY = "pointille_solid"
MATERIAL_PREFIX = "Pointille кривая "
SLICE_PREFIX = "Pointille срез "
CURVES_SUFFIX = "_кривые"
SHELL_SUFFIX = "_оболочка"


def is_curves(obj):
    return obj is not None and obj.type == "CURVE"


def find_curves(model):
    """The curve object traced from ``model``, or None."""
    for o in bpy.data.objects:
        if o.get(SKETCH_KEY) and o.get(SOURCE_KEY) == model.name:
            return o
    return None


def _material(rgb, prefix=MATERIAL_PREFIX):
    srgb = core.color.to_srgb8(np.asarray(rgb, np.float64)[None])[0]
    name = prefix + "#%02x%02x%02x" % tuple(int(v) for v in srgb)
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.diffuse_color = (*[float(min(max(v, 0.0), 1.0)) for v in rgb], 1.0)
    return mat


def _replace_object(context, name, data, anchor, existing):
    """``existing`` given ``data`` (its old data removed when nobody else uses it), or a new object."""
    if existing is None:
        obj = bpy.data.objects.new(name, data)
        collections = anchor.users_collection or [context.scene.collection]
        collections[0].objects.link(obj)
        return obj
    old = existing.data
    existing.data = data
    if old is not None and old.users == 0:
        (bpy.data.curves if isinstance(old, bpy.types.Curve) else bpy.data.meshes).remove(old)
    return existing


def build(context, model, sketch):
    """The curve object (and its shell) of ``sketch`` traced from ``model``, made or remade."""
    name = model.name + CURVES_SUFFIX
    data = bpy.data.curves.new(name, "CURVE")
    data.dimensions = "3D"
    data.use_path = False  # a child of a path is moved along it: the shell must stay where it is
    materials, used = [], []
    for c in sketch.curves:
        rgb = np.ones(3) if c.color is None else np.asarray(c.color, np.float64)
        mat = _material(rgb, SLICE_PREFIX if c.kind == "slice" else MATERIAL_PREFIX)
        if mat not in materials:
            materials.append(mat)
        used.append(materials.index(mat))
    for mat in materials:  # first: a spline's material index is cut to the materials there are
        data.materials.append(mat)
    for c, index in zip(sketch.curves, used):
        sp = data.splines.new("POLY")
        pts = np.asarray(c.points, np.float64)
        sp.points.add(len(pts) - 1)
        co = np.c_[pts, np.ones(len(pts))].astype(np.float32).ravel()
        sp.points.foreach_set("co", co)
        sp.use_cyclic_u = bool(c.closed)
        sp.material_index = index
    obj = _replace_object(context, name, data, model, find_curves(model))
    obj[SKETCH_KEY] = 1
    obj[SOURCE_KEY] = model.name
    for key in (NORMAL_KEY, SOLID_KEY):
        if key in obj:
            del obj[key]
    if sketch.normal is not None:
        obj[NORMAL_KEY] = [float(v) for v in sketch.normal]
    if sketch.solid:
        obj[SOLID_KEY] = 1
    obj.matrix_world = Matrix.Identity(4)
    _follow(obj, model)
    shell = bpy.data.objects.get(str(obj.get(SHELL_KEY, "")))
    if sketch.shell is not None:
        mesh = _shell_mesh(model.name + SHELL_SUFFIX, sketch.shell)
        shell = _replace_object(context, model.name + SHELL_SUFFIX, mesh, model, shell)
        shell.matrix_world = Matrix.Identity(4)
        _follow(shell, obj)
        shell.hide_render = True
        shell.display_type = "WIRE"
        context.view_layer.update()  # a new object is in the view layer only after an update: hide it there
        shell.hide_set(True)
        obj[SHELL_KEY] = shell.name
    elif shell is not None:
        bpy.data.objects.remove(shell)
        del obj[SHELL_KEY]
    obj.pointille_part.role = model.pointille_part.role
    obj.pointille_part.pattern = "SQUARE" if model.pointille_part.pattern == "SQUARE" else "HEX"
    return obj


def _follow(obj, parent):
    """``obj`` moves with ``parent`` from now on and stays where it is."""
    world = obj.matrix_world.copy()
    obj.parent = parent
    try:
        obj.matrix_parent_inverse = parent.matrix_world.inverted()
    except ValueError:
        obj.parent = None
    obj.matrix_world = world


def _shell_mesh(name, shell):
    corners = shell.triangles.reshape(-1, 3)
    verts, index = np.unique(np.round(corners, 9), axis=0, return_inverse=True)
    faces = index.reshape(-1, 3)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    me.loops.add(len(faces) * 3)
    me.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
    me.polygons.add(len(faces))
    me.polygons.foreach_set("loop_start", np.arange(0, len(faces) * 3, 3, dtype=np.int32))
    me.update()
    if shell.vertex_colors is not None:
        rgba = np.ones((len(faces) * 3, 4), np.float32)
        rgba[:, :3] = shell.vertex_colors.reshape(-1, 3)
        me.color_attributes.new("Col", "FLOAT_COLOR", "CORNER").data.foreach_set("color", rgba.ravel())
    return me


def read(obj, depsgraph, adapter):
    """The sketch of a curve object as the drones see it (world space): its splines (Bézier ones
    sampled at their resolution), their colours, its shell and the direction it faces."""
    mw = obj.matrix_world
    curves = []
    slots = [s.material for s in obj.material_slots]
    for sp in obj.data.splines:
        mat = slots[sp.material_index] if sp.material_index < len(slots) else None
        rgb = np.array(mat.diffuse_color[:3], np.float64) if mat is not None else None
        if sp.type == "BEZIER":
            pts = _bezier_points(sp)
        else:
            pts = [p.co.xyz.copy() for p in sp.points]
        if not pts:
            continue
        world = np.array([tuple(mw @ p) for p in pts], np.float64)
        closed = bool(sp.use_cyclic_u) and len(world) > 2
        kind = "slice" if mat is not None and mat.name.startswith(SLICE_PREFIX) else             "line" if len(world) > 1 else "dot"
        curves.append(core.Curve(world, closed, rgb, kind))
    shell = None
    holder = bpy.data.objects.get(str(obj.get(SHELL_KEY, "")))
    if holder is not None:
        shell = adapter.object_mesh(holder, depsgraph, {})
    normal = obj.get(NORMAL_KEY)
    if normal is not None:
        normal = np.array(mw.to_3x3() @ Vector(tuple(normal)), np.float64)
    return core.Sketch(curves, shell, normal, solid=bool(obj.get(SOLID_KEY)))


def _bezier_points(sp):
    bp = list(sp.bezier_points)
    if len(bp) < 2:
        return [p.co.copy() for p in bp]
    n = max(int(sp.resolution_u), 1) + 1
    pairs = list(zip(bp[:-1], bp[1:])) + ([(bp[-1], bp[0])] if sp.use_cyclic_u else [])
    out = []
    for a, b in pairs:
        seg = interpolate_bezier(a.co, a.handle_right, b.handle_left, b.co, n)
        out += list(seg[:-1])
    if not sp.use_cyclic_u:
        out.append(bp[-1].co.copy())
    return out
