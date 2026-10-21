"""Skybrush Studio: drones light up in the colour of their Pointille vertex, or in a style built on it.

Skybrush maps each drone to a marker of every formation in the storyboard. For
the formation holding Pointille vertices, drone i gets the colour of the vertex
it occupies there. The colours go into an image of a light effect of type
*Image*: pixel x = drone x (Output X *Indexed by drones*). A static look is one
row; an animated style is one row per time step and the effect reads them with
Output Y *Temporal*.

A **layer** is such an effect (its name starts with ``Pointille · ``). It has its
own frame range, fades, influence and style, so one formation can show the
texture for a while, go dark, then glow gold. Layers are ordinary Skybrush light
effects: saved in the .blend, exported with the show, editable in the LEDs panel
and layered with other effects as usual (a later effect covers an earlier one).

The style settings of a layer are a JSON string on its effect image. The colours
depend on Skybrush's drone-to-marker mapping, so layers are baked again when
Skybrush recalculates the transitions or the range of a layer changes
(``refresh_stale``). Time inside a layer is absolute (from frame 0), so a layer
split in two keeps one continuous animation.

Only Skybrush's scene data (``scene.skybrush``) is used, never its modules.
"""

import json
import re

import bpy
import numpy as np
from bpy.app.handlers import persistent

from . import drones

EFFECT_PREFIX = "Pointille · "
TRANSPARENT = (0.0, 0.0, 0.0, 0.0)  # drones that stand on no Pointille vertex keep their colour
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)  # Rec.709
ROWS_PER_SECOND = 12  # time resolution of animated styles
MAX_ROWS = 4096
STYLES = ("TEXTURE", "GRAYSCALE", "SHIMMER", "RAINBOW", "GRADIENT", "SOLID", "OFF")
DEFAULTS = {
    "style": "TEXTURE", "mix": 1.0, "brightness": 1.0, "saturation": 1.0,
    "speed": 0.0, "spread": 0.5, "color": [1.0, 0.0, 0.0], "color2": [0.0, 0.0, 1.0],
}
COLOR_KEYS = ("color", "color2")
STYLE_KEY = "pointille_style"


def available(context):
    return getattr(context.scene, "skybrush", None) is not None


# ------------------------------------------------------------------ storyboard

def entries_with(scene, points):
    """Storyboard entries whose formation contains the vertex object."""
    return [
        e for e in scene.skybrush.storyboard.entries
        if e.formation is not None and e.formation.objects.get(points.name) == points
    ]


def source_entry(scene, points, segment):
    """The formation entry whose colours a storyboard segment shows.

    The segment itself if it is a formation of ``points``; for a transition its
    target, else its source; otherwise the first formation of ``points``.
    """
    ours = entries_with(scene, points)
    if not ours:
        return None
    by_name = {e.name: e for e in ours}
    if segment in by_name:
        return by_name[segment]
    entries = list(scene.skybrush.storyboard.entries)
    for prev, nxt in zip(entries, entries[1:]):
        if f"{prev.name} -> {nxt.name}" == segment:
            return by_name.get(nxt.name) or by_name.get(prev.name) or ours[0]
    return ours[0]


def mapping_of(entry):
    """Skybrush's drone -> marker indices of an entry, or None before transitions are calculated."""
    try:
        mapping = json.loads(entry.mapping)
    except ValueError:
        return None
    return mapping if isinstance(mapping, list) else None


def drone_count(scene):
    drones_collection = scene.skybrush.settings.drone_collection or bpy.data.collections.get("Drones")
    return len(drones_collection.objects) if drones_collection else 0


def marker_data(formation):
    """(linear RGBA, world height) of every marker of a formation, in Skybrush's marker order.

    A mesh in a formation gives the vertices of its formation vertex group, in
    vertex order; any other object is a single marker (alpha 0: not ours).
    """
    rows, heights = [], []
    for obj in formation.objects:
        group_name = obj.skybrush.formation_vertex_group
        if not group_name:
            rows.append(TRANSPARENT)
            heights.append(0.0)
            continue
        group = obj.vertex_groups.get(group_name)
        if group is None or obj.type != "MESH":
            continue
        ours = bool(obj.get(drones.DRONES_KEY))
        colors = drones.drone_colors_linear(obj) if ours else None
        z = drones.world_positions(obj)[:, 2]
        for v in obj.data.vertices:
            if any(g.group == group.index for g in v.groups):
                rows.append(TRANSPARENT if colors is None else (*colors[v.index], 1.0))
                heights.append(float(z[v.index]))
    return np.array(rows, np.float32).reshape(-1, 4), np.array(heights, np.float32)


def marker_colors(formation):
    return marker_data(formation)[0]


# ------------------------------------------------------------------ colour styles

def _to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def _to_linear(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, np.power((c + 0.055) / 1.055, 2.4))


def _rgb_to_hsv(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = mx - mn
    safe = np.where(d == 0, 1.0, d)
    h = np.where(mx == r, ((g - b) / safe) % 6, np.where(mx == g, (b - r) / safe + 2, (r - g) / safe + 4)) / 6
    h = np.where(d == 0, 0.0, h)
    s = np.where(mx > 0, d / np.where(mx == 0, 1.0, mx), 0.0)
    return h, s, mx


def _hsv_to_rgb(h, s, v):
    h6 = (h % 1.0) * 6
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return np.stack([np.choose(i, [v, q, p, p, t, v]), np.choose(i, [t, v, v, q, p, p]),
                     np.choose(i, [p, p, t, v, v, q])], -1)


def is_animated(p):
    return p["style"] not in ("TEXTURE", "OFF") and p["speed"] > 0


def styled(base, pos, times, p):
    """Linear RGB per time step and drone, (T, N, 3).

    ``base`` (N, 3) is the texture colour, ``pos`` (N,) the height 0..1 of the
    drone in its formation, ``times`` (T,) seconds, ``p`` the style settings.
    """
    base = np.asarray(base, np.float64).reshape(-1, 3)
    n, t = len(base), len(times)
    plain = np.broadcast_to(base, (t, n, 3))
    style = p["style"]
    if style == "TEXTURE":
        return plain.copy()
    s = _to_srgb(base)
    h0, s0, v0 = _rgb_to_hsv(s)
    luma = s @ LUMA
    animated = is_animated(p)
    phase = p["speed"] * np.asarray(times, np.float64)[:, None] + p["spread"] * np.asarray(pos, np.float64)[None, :]
    wave = 0.5 + 0.5 * np.sin(2 * np.pi * phase)
    ones = np.ones((t, n))
    if style == "OFF":  # the LEDs are dark
        out = np.zeros((t, n, 3))
    elif style == "GRAYSCALE":
        gray = luma[None, :] * ((0.55 + 0.45 * wave) if animated else ones)
        out = _to_linear(np.repeat(gray[..., None], 3, -1))
    elif style == "SHIMMER":  # every point keeps its lightness, its hue drifts around the colour wheel
        out = _to_linear(_hsv_to_rgb(h0[None, :] + phase, np.maximum(s0, 0.6)[None, :] * ones, v0[None, :] * ones))
    elif style == "RAINBOW":  # the texture sets how light each point is, the rainbow sets the colour
        out = _to_linear(_hsv_to_rgb(phase, ones, (0.2 + 0.8 * v0)[None, :] * ones))
    else:
        c1 = np.asarray(p["color"], np.float64)
        if style == "GRADIENT":
            g = (0.5 - 0.5 * np.cos(2 * np.pi * phase))[..., None]
            c2 = np.asarray(p["color2"], np.float64)
            out = c1 * (1 - g) + c2 * g
        else:  # SOLID, breathing when animated
            out = c1 * ((0.6 + 0.4 * wave) if animated else ones)[..., None]
    mix = float(np.clip(p["mix"], 0.0, 1.0))
    return plain * (1 - mix) + out * mix


def render(scene, entry, p, times):
    """Linear RGBA of every drone at every time step for ``entry``, (T, drones, 4)."""
    markers, heights = marker_data(entry.formation)
    n = drone_count(scene)
    out = np.zeros((len(times), n, 4), np.float32)
    idx = np.full(n, -1)
    for i, m in enumerate((mapping_of(entry) or [])[:n]):
        if m is not None and 0 <= m < len(markers):
            idx[i] = m
    lit = idx >= 0
    lit[lit] = markers[idx[lit], 3] > 0
    if not lit.any():
        return out
    ours = markers[:, 3] > 0
    lo, hi = float(heights[ours].min()), float(heights[ours].max())
    pos = (heights[idx[lit]] - lo) / ((hi - lo) or 1.0)
    rgb = styled(markers[idx[lit], :3], pos, times, p)
    gray = rgb @ LUMA
    rgb = (gray[..., None] + (rgb - gray[..., None]) * p["saturation"]) * p["brightness"]
    out[:, lit, :3] = np.clip(rgb, 0.0, 1.0)
    out[:, lit, 3] = 1.0
    return out


# ------------------------------------------------------------------ layers

def effect_params(effect):
    """The style settings an effect was baked with (defaults for older effects)."""
    image = effect.color_image
    raw = image.get(STYLE_KEY) if image is not None else None
    try:
        stored = json.loads(raw) if raw else {}
    except ValueError:
        stored = {}
    return {**DEFAULTS, **{k: v for k, v in stored.items() if k in DEFAULTS}}


def pointille_effects(scene):
    """All layers, in Skybrush's order (a later one covers an earlier one)."""
    return [e for e in scene.skybrush.light_effects.entries if e.name.startswith(EFFECT_PREFIX)]


def active_layer(scene):
    """The layer selected in the light effect list, or None."""
    if getattr(scene, "skybrush", None) is None:
        return None
    effect = scene.skybrush.light_effects.active_entry
    return effect if effect is not None and effect.name.startswith(EFFECT_PREFIX) else None


def effect_source(scene, effect):
    """The formation entry a layer takes its colours from."""
    image = effect.color_image
    name = image.get("pointille_entry") if image is not None else None
    entry = next((e for e in scene.skybrush.storyboard.entries if e.name == name), None)
    if entry is not None:
        return entry
    for points in (o for o in bpy.data.objects if o.get(drones.DRONES_KEY)):  # made by 0.3
        entry = source_entry(scene, points, effect.storyboard_entry_or_transition_selection)
        if entry is not None:
            return entry
    return None


def bake(scene, effect, entry, p=None):
    """Write the drone colours of ``entry``, in the style ``p`` (or the layer's own), into the layer."""
    p = {**DEFAULTS, **(p or effect_params(effect))}
    n = drone_count(scene)
    if is_animated(p):
        seconds = max(effect.duration - 1, 1) / scene.render.fps
        rows = int(min(max(round(seconds * ROWS_PER_SECOND), 2), MAX_ROWS))
        times = effect.frame_start / scene.render.fps + (np.arange(rows) + 0.5) / rows * seconds
    else:
        rows, times = 1, np.zeros(1)
    colors = render(scene, entry, p, times)
    width = max(n, 1)
    for prop, value in (("type", "IMAGE"), ("output", "INDEXED_BY_DRONES"),
                        ("output_y", "TEMPORAL" if rows > 1 else "FIRST_COLOR")):
        if getattr(effect, prop) != value:
            setattr(effect, prop, value)
    image = effect.color_image
    if image is None or image.get("pointille_owner") != effect.name or tuple(image.size) != (width, rows):
        image = effect.create_color_image(effect.name, width, rows)  # its own image, linear as Skybrush reads it
    if n:
        image.pixels.foreach_set(colors.ravel())
    image["pointille_owner"] = effect.name
    image["pointille_entry"] = entry.name
    image["pointille_mapping"] = entry.mapping
    image["pointille_drones"] = n
    image["pointille_duration"] = effect.duration
    image["pointille_start"] = effect.frame_start
    image[STYLE_KEY] = json.dumps(p)
    image.pack()
    effect.invalidate_color_image()


def unique_name(effects, base):
    names = {e.name for e in effects.entries}
    name, k = base, 1
    while name in names:
        k += 1
        name = f"{base} {k}"
    return name


def add_layer(context, points, segment=""):
    """A new layer over the whole of ``segment`` (default: the formation of ``points``), textured.
    Returns ``(layer, source entry)`` or None."""
    scene = context.scene
    entry = source_entry(scene, points, segment)
    if entry is None:
        return None
    segment = segment or entry.name
    span = next((s for s in scene.skybrush.storyboard.entries_or_transitions if s.name == segment), None)
    if span is None:
        return None
    effects = scene.skybrush.light_effects
    effect = effects.append_new_entry(
        name=unique_name(effects, EFFECT_PREFIX + segment), frame_start=span.frame_start,
        duration=max(span.frame_end - span.frame_start, 1), select=True,
    )
    effect.target = "ALL"
    effect.storyboard_entry_or_transition_selection = segment  # the range follows the storyboard
    bake(scene, effect, entry, DEFAULTS)
    scene.frame_set(scene.frame_current)  # let Skybrush re-evaluate its light effects
    return effect, entry


def remove_layer(context):
    scene = context.scene
    if active_layer(scene) is None:
        return False
    scene.skybrush.light_effects.remove_active_entry()
    scene.frame_set(scene.frame_current)
    return True


def move_layer(context, up):
    effects = context.scene.skybrush.light_effects
    if active_layer(context.scene) is None:
        return
    (effects.move_active_entry_up if up else effects.move_active_entry_down)()
    context.scene.frame_set(context.scene.frame_current)


def split_layer(context, frame):
    """Cut the active layer in two at ``frame``: the first part ends before it, the second starts on it."""
    scene = context.scene
    effect = active_layer(scene)
    if effect is None or not (effect.frame_start < frame <= effect.frame_end):
        return None
    effects = scene.skybrush.light_effects
    index, end, fade_out = effects.active_entry_index, effect.frame_end, effect.fade_out_duration
    effects.duplicate_selected_entry(select=True)
    first, second = effects.entries[index], effects.entries[index + 1]
    second.frame_start = frame
    second.frame_end = end
    first.frame_end = frame - 1
    first.fade_out_duration, second.fade_in_duration, second.fade_out_duration = 0, 0, fade_out
    second.name = unique_name(effects, re.sub(r"( \d+)+$", "", first.name))  # "shape", "shape 2", "shape 3"
    for part in (first, second):  # each part gets its own image
        entry = effect_source(scene, part)
        if entry is not None:
            bake(scene, part, entry)
    effects.active_entry_index = index + 1
    scene.frame_set(scene.frame_current)
    return second


def set_range(context, end, frame):
    """Move the start (or the end) of the active layer to ``frame``."""
    effect = active_layer(context.scene)
    if effect is None:
        return False
    if end:
        effect.frame_end = frame
    else:
        last = effect.frame_end
        effect.frame_start = frame
        effect.frame_end = max(last, frame)
    return True


def refresh(scene, points=None):
    """Bake the layers again (only those showing ``points`` if given)."""
    if getattr(scene, "skybrush", None) is None:
        return 0
    done = 0
    for effect in pointille_effects(scene):
        entry = effect_source(scene, effect)
        if entry is None or (points is not None and entry.formation.objects.get(points.name) != points):
            continue
        bake(scene, effect, entry)
        done += 1
    return done


# ------------------------------------------------------------------ the panel edits the active layer

def active_param(key):
    """A style setting of the active layer (the default when no layer is selected)."""
    effect = active_layer(bpy.context.scene)
    return (effect_params(effect) if effect is not None else DEFAULTS)[key]


def set_active_param(key, value):
    scene = bpy.context.scene
    effect = active_layer(scene)
    entry = effect_source(scene, effect) if effect is not None else None
    if entry is None:
        return
    p = effect_params(effect)
    p[key] = list(value) if key in COLOR_KEYS else value
    if key == "style" and value in ("SHIMMER", "RAINBOW", "GRADIENT") and p["speed"] == 0:
        p["speed"] = 0.25  # motion is the point of these styles (stop it with Speed = 0)
    bake(scene, effect, entry, p)
    schedule_reevaluate()


def layer_name():
    effect = active_layer(bpy.context.scene)
    return effect.name[len(EFFECT_PREFIX):] if effect is not None else ""


def rename_layer(name):
    effect = active_layer(bpy.context.scene)
    if effect is None or not name.strip():
        return
    new = EFFECT_PREFIX + name.strip()
    if any(e.name == new for e in bpy.context.scene.skybrush.light_effects.entries if e != effect):
        return
    effect.name = new
    if effect.color_image is not None:
        effect.color_image["pointille_owner"] = new


def layer_source():
    effect = active_layer(bpy.context.scene)
    entry = effect_source(bpy.context.scene, effect) if effect is not None else None
    return entry.name if entry is not None else ""


def set_layer_source(name):
    scene = bpy.context.scene
    effect = active_layer(scene)
    entry = next((e for e in scene.skybrush.storyboard.entries if e.name == name), None)
    if effect is not None and entry is not None:
        bake(scene, effect, entry)  # remembers the new source and its mapping
        schedule_reevaluate()


_reevaluate_pending = False


def _reevaluate():
    global _reevaluate_pending
    _reevaluate_pending = False
    try:
        scene = bpy.context.scene
        scene.frame_set(scene.frame_current)
    except Exception:  # noqa: BLE001 - the scene may be gone by then
        pass
    return None


def schedule_reevaluate():
    """Let Skybrush show a changed effect in the viewport (a frame change does it)."""
    global _reevaluate_pending
    if not _reevaluate_pending:
        _reevaluate_pending = True
        bpy.app.timers.register(_reevaluate, first_interval=0.05)


_busy = False


@persistent
def refresh_stale(scene, depsgraph=None):
    """Re-bake layers whose mapping, drone count, range or image changed (e.g. transitions recalculated)."""
    global _busy
    if _busy or getattr(scene, "skybrush", None) is None:
        return
    _busy = True
    try:
        count = drone_count(scene)
        for effect in pointille_effects(scene):
            image = effect.color_image
            entry = effect_source(scene, effect)
            if image is None or entry is None:
                continue
            if (image.get("pointille_mapping") != entry.mapping or image.get("pointille_drones") != count
                    or image.get("pointille_duration") != effect.duration
                    or image.get("pointille_start") != effect.frame_start
                    or image.get("pointille_owner") != effect.name):
                bake(scene, effect, entry)
    finally:
        _busy = False


def register():
    bpy.app.handlers.depsgraph_update_post.append(refresh_stale)


def unregister():
    if refresh_stale in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(refresh_stale)
