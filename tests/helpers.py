"""Tiny procedural meshes for tests (numpy only)."""

import numpy as np

from pointille.core import Material, MeshData, Texture


def uv_sphere(radius=1.0, rings=48, segments=96):
    th = np.linspace(0, np.pi, rings + 1)
    ph = np.linspace(0, 2 * np.pi, segments + 1)
    t, p = np.meshgrid(th, ph, indexing="ij")
    verts = np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], -1) * radius
    tris = []
    for i in range(rings):
        for j in range(segments):
            a, b = verts[i, j], verts[i + 1, j]
            c, d = verts[i + 1, j + 1], verts[i, j + 1]
            if i != 0:
                tris.append([a, b, d])
            if i != rings - 1:
                tris.append([b, c, d])
    tris = np.array(tris)
    # outward winding
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    flip = np.einsum("ij,ij->i", n, tris.mean(axis=1)) < 0
    tris[flip] = tris[flip][:, ::-1]
    return MeshData(tris)


def grid_plane(size=(2.0, 2.0), res=8, y=0.0, center=(0.0, 0.0), facing=-1):
    """Plane in XZ at height ``y``; normal is ``facing`` * Y.

    UVs: u follows +X, v follows +Z (so v-up in the texture = +Z).
    """
    sx, sz = size
    xs = np.linspace(-sx / 2, sx / 2, res + 1) + center[0]
    zs = np.linspace(-sz / 2, sz / 2, res + 1) + center[1]
    tris, uvs = [], []
    for i in range(res):
        for j in range(res):
            q = [(xs[i], zs[j]), (xs[i + 1], zs[j]), (xs[i + 1], zs[j + 1]), (xs[i], zs[j + 1])]
            p = [np.array([x, y, z]) for x, z in q]
            uv = [np.array([i / res, j / res]), np.array([(i + 1) / res, j / res]),
                  np.array([(i + 1) / res, (j + 1) / res]), np.array([i / res, (j + 1) / res])]
            for tri in ((0, 1, 2), (0, 2, 3)):
                tris.append([p[k] for k in tri])
                uvs.append([uv[k] for k in tri])
    tris, uvs = np.array(tris), np.array(uvs)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    flip = np.sign(n[:, 1]) != facing
    tris[flip] = tris[flip][:, ::-1]
    uvs[flip] = uvs[flip][:, ::-1]
    return MeshData(tris, uvs=uvs)


QUADRANTS = {  # texture quadrant -> sRGB color
    "top_left": (255, 0, 0),
    "top_right": (0, 255, 0),
    "bottom_left": (0, 0, 255),
    "bottom_right": (255, 255, 255),
}


def quadrant_texture(size=64):
    px = np.zeros((size, size, 3), np.uint8)
    h = size // 2
    px[:h, :h] = QUADRANTS["top_left"]
    px[:h, h:] = QUADRANTS["top_right"]
    px[h:, :h] = QUADRANTS["bottom_left"]
    px[h:, h:] = QUADRANTS["bottom_right"]
    return px


def textured_plane(**kw):
    m = grid_plane(**kw)
    m.materials = [Material(texture=Texture(quadrant_texture()))]
    m.material_ids = np.zeros(m.triangle_count, np.int64)
    return m
