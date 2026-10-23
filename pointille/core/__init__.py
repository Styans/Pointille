"""Pointille core: even drone placement on a textured surface (numpy only)."""

from .api import COLOR_MODES, OUTLINE_AMOUNTS, DistributeOptions, DroneResult, distribute
from .mesh import Material, MeshData, Texture
from .stats import SCALE_MODES

__all__ = [
    "COLOR_MODES",
    "OUTLINE_AMOUNTS",
    "SCALE_MODES",
    "DistributeOptions",
    "DroneResult",
    "Material",
    "MeshData",
    "Texture",
    "distribute",
]
