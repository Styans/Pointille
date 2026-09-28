"""Read Blender objects into the core's arrays. Nothing here writes to the model."""

import numpy as np

from ._vendor import core

_COLOR_INPUTS = ("Base Color", "Color")
_SHADERS_WITH_COLOR = {
    "ShaderNodeBsdfPrincipled",
    "ShaderNodeBsdfDiffuse",
    "ShaderNodeEmission",
    "ShaderNodeBackground",
    "ShaderNodeBsdfToon",
    "ShaderNodeSubsurfaceScattering",
}
_VERTEX_COLOR_NODES = {"ShaderNodeVertexColor", "ShaderNodeAttribute"}


def image_texture(image, cache):
    """Pixels of a Blender image as a core Texture (row 0 = top)."""
    if image is None:
        return None
    key = image.name_full
    if key in cache:
        return cache[key]
    w, h = image.size[:]
    tex = None
    if w and h:
        ch = image.channels
        buf = np.empty(w * h * ch, np.float32)
        image.pixels.foreach_get(buf)
        px = buf.reshape(h, w, ch)[::-1]
        # byte images hand out their stored (sRGB) values, float images are linear
        srgb = not image.is_float and image.colorspace_settings.name not in ("Non-Color", "Linear", "Linear Rec.709")
        if not image.is_float:
            px = (np.clip(px, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        tex = core.Texture(np.ascontiguousarray(px), srgb=srgb)
    cache[key] = tex
    return tex


def _upstream(socket):
    """All nodes feeding a socket, nearest first."""
    seen, order, queue = set(), [], [link.from_node for link in socket.links]
    while queue:
        node = queue.pop(0)
        if node.as_pointer() in seen:
            continue
        seen.add(node.as_pointer())
        order.append(node)
        for inp in node.inputs:
            queue.extend(link.from_node for link in inp.links)
    return order


def _surface_shader(tree):
    outputs = [n for n in tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial"]
    outputs.sort(key=lambda n: not n.is_active_output)
    for out in outputs:
        surface = out.inputs.get("Surface")
        if surface is None or not surface.is_linked:
            continue
        for node in _upstream(surface):
            if node.bl_idname in _SHADERS_WITH_COLOR:
                return node
    return None


def material_info(mat):
    """(base RGBA linear, image or None, uv map name or None, uses vertex colors)."""
    if mat is None:
        return (1.0, 1.0, 1.0, 1.0), None, None, False
    if not mat.use_nodes or mat.node_tree is None:
        return tuple(mat.diffuse_color), None, None, False
    tree = mat.node_tree
    shader = _surface_shader(tree)
    socket = None
    if shader is not None:
        socket = next((shader.inputs[n] for n in _COLOR_INPUTS if n in shader.inputs), None)
    if socket is not None and not socket.is_linked:
        return tuple(socket.default_value), None, None, False
    nodes = _upstream(socket) if socket is not None else []
    uses_vc = any(n.bl_idname in _VERTEX_COLOR_NODES for n in nodes)
    image_node = next((n for n in nodes if n.bl_idname == "ShaderNodeTexImage" and n.image), None)
    if image_node is None and socket is None:
        # unknown shader setup: fall back to any color (sRGB) texture in the material
        image_node = next(
            (n for n in tree.nodes if n.bl_idname == "ShaderNodeTexImage" and n.image
             and n.image.colorspace_settings.name == "sRGB"),
            None,
        )
    if image_node is None:
        return (1.0, 1.0, 1.0, 1.0), None, None, uses_vc
    uv_name = None
    vec = image_node.inputs.get("Vector")
    if vec is not None and vec.is_linked:
        uv_node = next((n for n in _upstream(vec) if n.bl_idname == "ShaderNodeUVMap"), None)
        if uv_node is not None and uv_node.uv_map:
            uv_name = uv_node.uv_map
    return (1.0, 1.0, 1.0, 1.0), image_node.image, uv_name, uses_vc


def _foreach(collection, attr, count, width, dtype):
    buf = np.empty(count * width, dtype)
    collection.foreach_get(attr, buf)
    return buf.reshape(count, width) if width > 1 else buf


def object_mesh(obj, depsgraph, image_cache):
    """World-space MeshData of an evaluated object (modifiers applied)."""
    eval_obj = obj.evaluated_get(depsgraph)
    me = eval_obj.to_mesh()
    try:
        me.calc_loop_triangles()
        t = len(me.loop_triangles)
        if t == 0:
            return None
        v = len(me.vertices)
        co = _foreach(me.vertices, "co", v, 3, np.float32).astype(np.float64)
        mw = np.array(eval_obj.matrix_world, dtype=np.float64)
        co = co @ mw[:3, :3].T + mw[:3, 3]
        tri_v = _foreach(me.loop_triangles, "vertices", t, 3, np.int32)
        tri_l = _foreach(me.loop_triangles, "loops", t, 3, np.int32)
        mat_idx = _foreach(me.loop_triangles, "material_index", t, 1, np.int32).astype(np.int64)

        slots = [s.material for s in eval_obj.material_slots] or [None]
        mat_idx = np.clip(mat_idx, 0, len(slots) - 1)
        infos = [material_info(m) for m in slots]

        uv_cache = {}

        def loop_uvs(name):
            layer = me.uv_layers.get(name) if name else None
            if layer is None:
                layer = next((l for l in me.uv_layers if l.active_render), None) or me.uv_layers.active
            if layer is None:
                return None
            if layer.name not in uv_cache:
                uv_cache[layer.name] = _foreach(layer.data, "uv", len(me.loops), 2, np.float32)
            return uv_cache[layer.name]

        uvs = np.zeros((t, 3, 2))
        has_uv = False
        vcol = None
        corner_colors = None
        no_texture_anywhere = all(info[1] is None for info in infos)
        materials = []
        for i, (base, image, uv_name, uses_vc) in enumerate(infos):
            tex = image_texture(image, image_cache) if image is not None else None
            materials.append(core.Material(base_color=base, texture=tex))
            mask = mat_idx == i
            if tex is not None and mask.any():
                luv = loop_uvs(uv_name)
                if luv is not None:
                    uvs[mask] = luv[tri_l[mask]]
                    has_uv = True
            if (uses_vc or no_texture_anywhere) and mask.any():
                if corner_colors is None:
                    corner_colors = _vertex_colors(me, tri_v, tri_l)
                if corner_colors is not None:
                    if vcol is None:
                        vcol = np.ones((t, 3, 3))
                    vcol[mask] = corner_colors[mask]
        return core.MeshData(
            triangles=co[tri_v],
            uvs=uvs if has_uv else None,
            material_ids=mat_idx,
            vertex_colors=vcol,
            materials=materials,
        )
    finally:
        eval_obj.to_mesh_clear()


def _vertex_colors(me, tri_v, tri_l):
    attrs = me.color_attributes
    if not len(attrs):
        return None
    idx = attrs.render_color_index if attrs.render_color_index >= 0 else 0
    attr = attrs[idx]
    if attr.domain == "POINT":
        c = _foreach(attr.data, "color", len(me.vertices), 4, np.float32)
        return c[tri_v][:, :, :3].astype(np.float64)
    if attr.domain == "CORNER":
        c = _foreach(attr.data, "color", len(me.loops), 4, np.float32)
        return c[tri_l][:, :, :3].astype(np.float64)
    return None


def scene_mesh(objects, depsgraph):
    cache = {}
    parts = [m for m in (object_mesh(o, depsgraph, cache) for o in objects) if m is not None]
    if not parts:
        raise ValueError("у выбранных объектов нет полигонов")
    return core.MeshData.concatenate(parts)
