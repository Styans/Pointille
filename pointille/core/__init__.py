"""Pointille core: even drone placement on a textured surface (numpy only)."""

from .api import COLOR_MODES, OUTLINE_AMOUNTS, DistributeOptions, DroneResult, distribute
from .around import around
from .leftover import leftover
from .curves import DEPTHS, Curve, Sketch, compose_curves, trace
from .layout import PATTERNS, PICKS, ROLES, ColorGroups, Part, PartOptions, auto_selected, color_groups, compose
from .mesh import Material, MeshData, Texture
from .stats import SCALE_MODES

__all__ = [
    "COLOR_MODES",
    "PATTERNS",
    "PICKS",
    "ROLES",
    "ColorGroups",
    "Curve",
    "DEPTHS",
    "OUTLINE_AMOUNTS",
    "Part",
    "PartOptions",
    "Sketch",
    "SCALE_MODES",
    "DistributeOptions",
    "DroneResult",
    "Material",
    "MeshData",
    "Texture",
    "around",
    "leftover",
    "auto_selected",
    "color_groups",
    "compose",
    "compose_curves",
    "distribute",
    "trace",
]
