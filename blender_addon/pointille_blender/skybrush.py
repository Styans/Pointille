"""Skybrush Studio: every drone lights up in the colour of the vertex it stands on.

Skybrush maps each drone to a marker of the formation it flies to, and a light
effect of type *Image* with the output *Indexed by formation* reads pixel k for
the drone on marker k. Pointille writes a one-row image, pixel k = colour of
marker k, into such an effect attached to the storyboard entry. It is an
ordinary Skybrush light effect: saved in the .blend and exported with the show.

Only Skybrush's scene data (``scene.skybrush``) is used, never its modules.
"""

import bpy
import numpy as np

from . import drones

EFFECT_PREFIX = "Pointille · "
TRANSPARENT = (0.0, 0.0, 0.0, 0.0)  # markers that are not Pointille vertices keep their colour


def available(context):
    return getattr(context.scene, "skybrush", None) is not None


def entries_with(scene, points):
    """Storyboard entries whose formation contains the vertex object."""
    return [
        e for e in scene.skybrush.storyboard.entries
        if e.formation is not None and e.formation.objects.get(points.name) == points
    ]


def marker_colors(formation):
    """Linear RGBA of every marker of a formation, in Skybrush's marker order.

    A mesh in a formation gives the vertices of its formation vertex group, in
    vertex order; any other object is a single marker.
    """
    rows = []
    for obj in formation.objects:
        group_name = obj.skybrush.formation_vertex_group
        if not group_name:
            rows.append(TRANSPARENT)
            continue
        group = obj.vertex_groups.get(group_name)
        if group is None or obj.type != "MESH":
            continue
        colors = drones.drone_colors_linear(obj) if obj.get(drones.DRONES_KEY) else None
        for v in obj.data.vertices:
            if any(g.group == group.index for g in v.groups):
                rows.append(TRANSPARENT if colors is None else (*colors[v.index], 1.0))
    return np.array(rows, dtype=np.float32).reshape(-1, 4)


def drone_count(scene):
    drones_collection = scene.skybrush.settings.drone_collection or bpy.data.collections.get("Drones")
    return len(drones_collection.objects) if drones_collection else 0


def color_drones(context, points, *, create=True):
    """Colour the drones of every storyboard entry whose formation holds ``points``.

    Returns ``(entry name, markers, drones, transitions ready)`` per entry. With
    ``create`` off only the effects made earlier are refreshed (after a new layout).
    """
    scene = context.scene
    effects = scene.skybrush.light_effects
    count = drone_count(scene)
    done = []
    for entry in entries_with(scene, points):
        name = EFFECT_PREFIX + entry.name
        effect = next((e for e in effects.entries if e.name == name), None)
        if effect is None:
            if not create:
                continue
            effect = effects.append_new_entry(name=name, frame_start=entry.frame_start, duration=entry.duration)
            effect.type = "IMAGE"
            effect.output = "INDEXED_BY_FORMATION"
            effect.output_y = "FIRST_COLOR"
            effect.target = "ALL"
            effect.storyboard_entry_or_transition_selection = entry.name  # timing follows the entry
        colors = marker_colors(entry.formation)
        image = effect.color_image
        if image is None or tuple(image.size) != (max(len(colors), 1), 1):
            image = effect.create_color_image(name, max(len(colors), 1), 1)  # linear, as Skybrush reads it
        if len(colors):
            image.pixels.foreach_set(colors.ravel())
        image.pack()
        effect.invalidate_color_image()
        done.append((entry.name, len(colors), count, bool(str(getattr(entry, "mapping", "")).strip())))
    if done:
        scene.frame_set(scene.frame_current)  # let Skybrush re-evaluate its light effects
    return done
