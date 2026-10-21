"""Reading the model's surface color at arbitrary points (read-only)."""

from __future__ import annotations

import numpy as np

from .mesh import MeshData, Texture


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1.0 / 2.4) - 0.055)


def to_srgb8(linear) -> np.ndarray:
    return np.round(linear_to_srgb(linear) * 255.0).astype(np.uint8)


def _texels_rgb(tex: Texture, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    px = tex.pixels[y, x]
    if px.dtype == np.uint8:
        px = px.astype(np.float64) / 255.0
    elif px.dtype == np.uint16:
        px = px.astype(np.float64) / 65535.0
    else:
        px = px.astype(np.float64)
    ch = px.shape[-1]
    rgb = np.repeat(px[:, :1], 3, axis=1) if ch < 3 else px[:, :3]
    return srgb_to_linear(rgb) if tex.srgb else rgb


def sample_texture(tex: Texture, uv: np.ndarray) -> np.ndarray:
    """Bilinear lookup, returns linear RGB. ``uv`` uses v-up (OpenGL).

    "repeat" wraps the UV itself, but the filter footprint is clamped to the
    image edge: AI models use texture atlases, and wrapping the footprint
    would bleed colour from the opposite side of the atlas.
    """
    h, w = tex.pixels.shape[:2]
    u, v = uv[:, 0], uv[:, 1]
    if tex.wrap != "clamp":
        u = np.where((u < 0) | (u > 1), np.mod(u, 1.0), u)
        v = np.where((v < 0) | (v > 1), np.mod(v, 1.0), v)
    x = np.clip(u * w - 0.5, 0, w - 1)
    y = np.clip((1.0 - v) * h - 0.5, 0, h - 1)
    x0 = np.floor(x)
    y0 = np.floor(y)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    x0 = x0.astype(np.int64)
    y0 = y0.astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    top = _texels_rgb(tex, y0, x0) * (1 - fx) + _texels_rgb(tex, y0, x1) * fx
    bot = _texels_rgb(tex, y1, x0) * (1 - fx) + _texels_rgb(tex, y1, x1) * fx
    return top * (1 - fy) + bot * fy


def surface_colors(mesh: MeshData, tri_ids: np.ndarray, bary: np.ndarray) -> np.ndarray:
    """Linear RGB of the model surface at the given barycentric points."""
    color = np.ones((len(tri_ids), 3))
    if mesh.vertex_colors is not None:
        color *= np.einsum("ij,ijk->ik", bary, mesh.vertex_colors[tri_ids])
    if not mesh.materials:
        return color
    if mesh.material_ids is not None:
        mids = np.clip(mesh.material_ids[tri_ids], 0, len(mesh.materials) - 1)
    else:
        mids = np.zeros(len(tri_ids), np.int64)
    for mid in np.unique(mids):
        sel = np.flatnonzero(mids == mid)
        mat = mesh.materials[mid]
        color[sel] *= mat.base_color[:3]
        if mat.texture is not None and mesh.uvs is not None:
            uv = np.einsum("ij,ijk->ik", bary[sel], mesh.uvs[tri_ids[sel]])
            color[sel] *= sample_texture(mat.texture, uv)
    return color


# sRGB brightness below which a colour has no hue left to keep, only noise: it glows an even white
GLOW_NEAR_BLACK = (0.06, 0.2)
GLOW_STRETCH = 0.3  # the brightest drone counts as at least this bright: a black model is not stretched to full


def glow(linear: np.ndarray, floor: float) -> np.ndarray:
    """Colours a drone can show. An LED cannot be black: a drone that is off is not seen at night, so a
    dark model (a dark-brown eagle, black tyres) leaves holes in the figure.

    Every colour keeps its hue, its saturation and its place among the drones' brightnesses (sRGB value of
    the strongest channel): the brightest drone glows at full, the darkest at ``floor`` 0..1, near-black an
    even dim white. ``floor`` 0 keeps the colours as they are.
    """
    c = np.asarray(linear, np.float64).reshape(-1, 3)
    if not floor > 0 or not len(c):
        return c
    floor = min(float(floor), 1.0)
    s = linear_to_srgb(c)
    v = s.max(axis=1)
    top = max(float(np.percentile(v, 98)), GLOW_STRETCH)
    full = s / np.maximum(v, 1e-6)[:, None]  # the colour at full brightness: the same hue and saturation
    hue = np.clip((v - GLOW_NEAR_BLACK[0]) / (GLOW_NEAR_BLACK[1] - GLOW_NEAR_BLACK[0]), 0.0, 1.0)
    level = floor + (1.0 - floor) * np.clip(v / top, 0.0, 1.0) * hue
    return srgb_to_linear((hue[:, None] * full + (1.0 - hue[:, None])) * level[:, None])


def average_by_owner(colors: np.ndarray, owner: np.ndarray, count: int) -> np.ndarray:
    """Mean color of the candidates owned by each drone (its Voronoi cell)."""
    total = np.stack([np.bincount(owner, weights=colors[:, c], minlength=count) for c in range(3)], 1)
    n = np.bincount(owner, minlength=count)[:, None]
    return total / np.maximum(n, 1)
