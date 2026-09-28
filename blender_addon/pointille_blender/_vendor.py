"""The algorithm lives in the ``pointille`` package; release zips vendor it
next to this file (see tools/build_blender_addon.py)."""

try:
    from .pointille import core
    from .pointille.io import export
except ImportError:  # source checkout with the repository on sys.path
    from pointille import core
    from pointille.io import export

__all__ = ["core", "export"]
