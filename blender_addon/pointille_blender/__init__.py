"""Pointille for Blender: even vertex placement over a textured model.

The model stays untouched; a separate vertex-only object holds one vertex per
drone position with its LED color in the ``drone_color`` attribute. Any object
(a drone model, say) can then be placed on those vertices, and Skybrush Studio
drones can take the colors (see ``skybrush.py``).
"""

bl_info = {
    "name": "Pointille",
    "author": "Tyan Stanislav",
    "version": (0, 18, 2),
    "blender": (3, 6, 0),
    "location": "3D Viewport > Sidebar (N) > Pointille",
    "description": "Равномерно распределяет вершины по поверхности 3D-модели и берёт цвета из текстуры; "
    "на вершины можно поставить любой объект",
    "category": "Object",
}

from . import operators, panel, props, skybrush  # noqa: E402

_modules = (props, operators, panel, skybrush)


def register():
    for m in _modules:
        m.register()


def unregister():
    for m in reversed(_modules):
        m.unregister()
