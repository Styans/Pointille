"""Read a model file (GLB, glTF, OBJ, PLY, STL) into a core MeshData. Needs trimesh and Pillow."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.color import srgb_to_linear
from ..core.mesh import Material, MeshData, Texture


def load_mesh(path) -> MeshData:
    """Every mesh of the file in world space, in the file's own axes (Y-up for glTF and OBJ)."""
    import trimesh

    path = Path(path)
    scene = trimesh.load(str(path), force="scene")
    # glTF stores vertex colours linear; the other formats store what the screen shows (sRGB)
    linear_colors = path.suffix.lower() in (".glb", ".gltf")
    parts, materials = [], {}
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        geom = scene.geometry[name]
        if isinstance(geom, trimesh.Trimesh) and len(geom.faces):
            parts.append(_part(geom, transform, linear_colors, materials))
    if not parts:
        raise ValueError(f"{path.name}: no triangles")
    return MeshData.concatenate(parts)


def _part(geom, transform, linear_colors, materials) -> MeshData:
    import trimesh

    faces = np.asarray(geom.faces)
    tris = trimesh.transform_points(np.asarray(geom.vertices, np.float64), transform)[faces]
    visual = geom.visual
    if isinstance(visual, trimesh.visual.TextureVisuals):
        uv = visual.uv
        uvs = np.asarray(uv, np.float64)[faces] if uv is not None and len(uv) == len(geom.vertices) else None
        return MeshData(tris, uvs=uvs, materials=[_material(visual.material, materials)])
    colors = None
    if visual.kind == "vertex":
        colors = np.asarray(visual.vertex_colors)[:, :3][faces]
    elif visual.kind == "face":
        colors = np.repeat(np.asarray(visual.face_colors)[:, None, :3], 3, axis=1)
    if colors is None:
        return MeshData(tris)
    colors = colors.astype(np.float64) / 255.0
    return MeshData(tris, vertex_colors=colors if linear_colors else srgb_to_linear(colors))


def _material(mat, cache) -> Material:
    """trimesh keeps glTF's base colour factor (linear) as bytes and the texture as a PIL image."""
    from trimesh.visual.material import PBRMaterial

    key = id(mat)
    if key not in cache:
        if not isinstance(mat, PBRMaterial) and hasattr(mat, "to_pbr"):  # OBJ/MTL
            mat = mat.to_pbr()
        factor = getattr(mat, "baseColorFactor", None)
        base = np.ones(4) if factor is None else np.asarray(factor, np.float64).ravel() / 255.0
        image = getattr(mat, "baseColorTexture", None)
        texture = None if image is None else Texture(np.asarray(image.convert("RGB")))
        cache[key] = Material(base_color=base, texture=texture)
    return cache[key]
