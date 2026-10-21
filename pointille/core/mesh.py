"""Host-independent description of a textured triangle soup.

Every host (Blender, CLI, web) converts its scene into these arrays; the
algorithm never touches the original model data.

Conventions
-----------
* Positions are world-space, in the model's units.
* UVs follow the OpenGL convention: ``v = 0`` is the *bottom* row of the image.
* ``Texture.pixels`` is stored top row first (row 0 = top of the image), the
  way PIL and browsers hand images over.
* Colors inside the core are linear RGB; sRGB is used only for input textures
  (when ``Texture.srgb`` is set) and for the final LED colors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Texture:
    pixels: np.ndarray  # (H, W) or (H, W, C), uint8 or float in [0, 1]
    srgb: bool = True
    wrap: str = "repeat"  # "repeat" | "clamp"

    def __post_init__(self):
        px = np.asarray(self.pixels)
        if px.ndim == 2:
            px = px[:, :, None]
        if px.ndim != 3 or px.shape[0] == 0 or px.shape[1] == 0:
            raise ValueError("Texture.pixels must be an (H, W[, C]) array")
        self.pixels = px


@dataclass
class Material:
    base_color: np.ndarray = field(default_factory=lambda: np.ones(4))  # linear RGBA
    texture: Optional[Texture] = None

    def __post_init__(self):
        c = np.asarray(self.base_color, dtype=np.float64).ravel()
        if c.size == 3:
            c = np.append(c, 1.0)
        self.base_color = c[:4]


@dataclass
class MeshData:
    triangles: np.ndarray  # (T, 3, 3) world-space corners
    uvs: Optional[np.ndarray] = None  # (T, 3, 2)
    material_ids: Optional[np.ndarray] = None  # (T,) index into ``materials``
    vertex_colors: Optional[np.ndarray] = None  # (T, 3, 3|4) linear
    materials: List[Material] = field(default_factory=list)

    def __post_init__(self):
        tris = np.asarray(self.triangles, dtype=np.float64)
        if tris.ndim != 3 or tris.shape[1:] != (3, 3):
            raise ValueError("triangles must have shape (T, 3, 3)")
        self.triangles = tris
        t = len(tris)
        if self.uvs is not None:
            self.uvs = np.asarray(self.uvs, dtype=np.float64).reshape(t, 3, 2)
        if self.material_ids is not None:
            self.material_ids = np.asarray(self.material_ids, dtype=np.int64).reshape(t)
        if self.vertex_colors is not None:
            vc = np.asarray(self.vertex_colors, dtype=np.float64)
            self.vertex_colors = vc.reshape(t, 3, vc.shape[-1])[:, :, :3]

    @property
    def triangle_count(self) -> int:
        return len(self.triangles)

    @staticmethod
    def concatenate(parts: List["MeshData"]) -> "MeshData":
        """Merge several meshes (e.g. multi-part AI models) into one."""
        parts = [p for p in parts if p.triangle_count]
        if not parts:
            raise ValueError("no triangles to merge")
        tris = np.concatenate([p.triangles for p in parts])
        has_uv = any(p.uvs is not None for p in parts)
        has_vc = any(p.vertex_colors is not None for p in parts)
        uvs, mats, vcs, mids = [], [], [], []
        for p in parts:
            t = p.triangle_count
            uvs.append(p.uvs if p.uvs is not None else np.zeros((t, 3, 2)))
            vcs.append(p.vertex_colors if p.vertex_colors is not None else np.ones((t, 3, 3)))
            if p.materials:
                ids = p.material_ids if p.material_ids is not None else np.zeros(t, np.int64)
                mids.append(np.clip(ids, 0, len(p.materials) - 1) + len(mats))
                mats.extend(p.materials)
            else:
                mids.append(np.full(t, len(mats), np.int64))
                mats.append(Material())
        return MeshData(
            triangles=tris,
            uvs=np.concatenate(uvs) if has_uv else None,
            material_ids=np.concatenate(mids),
            vertex_colors=np.concatenate(vcs) if has_vc else None,
            materials=mats,
        )
