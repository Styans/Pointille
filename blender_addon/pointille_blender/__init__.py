"""Pointille for Blender: even drone placement over a textured model.

The model stays untouched; a separate vertex-only object holds one vertex per
drone with its LED color in the ``drone_color`` attribute.
"""

bl_info = {
    "name": "Pointille",
    "author": "Tyan Stanislav",
    "version": (0, 1, 0),
    "blender": (3, 6, 0),
    "location": "3D Viewport > Sidebar (N) > Pointille",
    "description": "Равномерно распределяет дронов по поверхности 3D-модели и берёт цвета из текстуры",
    "category": "Object",
}

from . import operators, panel, props  # noqa: E402

_modules = (props, operators, panel)


def register():
    for m in _modules:
        m.register()


def unregister():
    for m in reversed(_modules):
        m.unregister()
