import time

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty
from bpy_extras.io_utils import ExportHelper
from mathutils import Vector

from . import adapter, drones, sketch, skybrush
from ._vendor import core
from .props import VIEW_DIRS

SCALE_MODES = {"NONE": "none", "MIN_DISTANCE": "min_distance", "HEIGHT": "height"}
COLOR_MODES = {"AREA": "area", "POINT": "point", "NONE": "none"}
OUTLINE_AMOUNTS = {"DENSITY": "density", "SHARE": "share"}
ROLES = {"FULL": "full", "OUTLINE": "outline", "PATTERN": "pattern", "FILL": "fill"}
PATTERNS = {"HEX": "hex", "SQUARE": "square", "NOISE": "noise", "RINGS": "rings"}
PICKS = {"GROUPS": "groups", "EYEDROPPER": "eyedropper", "AUTO": "auto"}
DEPTHS = {"FLAT": "flat", "SHALLOW": "shallow", "SHAPE": "shape", "VOLUME": "volume"}


def _audience(context, settings):
    """(view_dir, view_point) for the core: where the audience watches from."""
    if settings.view == "CAMERA":
        cam = context.scene.camera
        if cam is None:
            raise ValueError("в сцене нет активной камеры")
        # from the model towards the camera = opposite of the camera's look direction
        view_dir = tuple(cam.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0)))
        if cam.type == "CAMERA" and cam.data.type == "ORTHO":
            return view_dir, None
        return view_dir, tuple(cam.matrix_world.translation)  # sight lines converge on the camera
    return VIEW_DIRS[settings.view], None


def active_drones(context):
    return drones.find_drones(context.active_object)


class POINTILLE_OT_distribute(bpy.types.Operator):
    """Равномерно распределить вершины по поверхности выбранной модели (модель не меняется)"""

    bl_idname = "pointille.distribute"
    bl_label = "Распределить вершины"
    bl_options = {"REGISTER", "UNDO"}

    count: IntProperty(name="Вершин", default=500, min=1, soft_max=10000)
    seed: IntProperty(name="Seed", default=0, min=0)

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT"

    def invoke(self, context, event):
        s = context.scene.pointille
        self.count, self.seed = s.count, s.seed
        return self.execute(context)

    def execute(self, context):
        s = context.scene.pointille
        s.count, s.seed = self.count, self.seed
        sources = drones.resolve_sources(context)
        if not sources:
            self.report({"ERROR"}, "Выберите меш-модель")
            return {"CANCELLED"}
        started = time.perf_counter()
        around = s.area_mode == "AROUND"
        limit = s.count_mode == "LIMIT" or around  # a figure all round always takes the count as its budget
        curves = [o for o in sources if sketch.is_curves(o)]
        if curves and len(curves) < len(sources):
            self.report({"ERROR"}, "Выберите либо модели, либо кривые")
            return {"CANCELLED"}
        try:
            depsgraph = context.evaluated_depsgraph_get()
            opts = _options(context, s, self.count, self.seed, limit or bool(curves))
            if curves:  # a drawing: drones on its curves, the fill on its shell
                limit = True
                sketches = []
                for o in curves:
                    sk = sketch.read(o, depsgraph, adapter)
                    sk.options = part_options(o)
                    sketches.append(sk)
                result = core.compose_curves(sketches, opts)
            elif limit:
                cache, parts, used = {}, [], []
                for o in sources:
                    m = adapter.object_mesh(o, depsgraph, cache)
                    if m is not None:
                        parts.append(core.Part(m, part_options(o)))
                        used.append(o)
                if not parts:
                    raise ValueError("у выбранных объектов нет полигонов")
                sources = used
                result = core.around(parts, opts) if around else core.compose(parts, opts)
            else:
                result = core.distribute(adapter.scene_mesh(sources, depsgraph), opts)
        except ValueError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        if limit and s.leftover_mode != "NONE":  # the rest of the budget: parked dark or stars round the figure
            result = core.leftover(result, s.leftover_mode.lower(), seed=self.seed)
        obj = drones.write_drones(context, sources, result)
        if limit:
            obj["pointille_leftover_mode"] = LEFTOVER_LABELS[s.leftover_mode]
        if s.scale_mode == "HEIGHT" or (s.scale_mode != "NONE" and not limit):
            drones.apply_scale(context, obj, result.scale)
        skybrush.refresh(context.scene, obj)  # new colours for the Skybrush effects made earlier
        took = f"({time.perf_counter() - started:.1f} с)"
        if limit:
            self.report({"INFO"}, budget_report(result) + f", мин. дистанция {obj['pointille_min_distance']:.3f} м "
                        + took)
            return {"FINISHED"}
        lines = ""
        if result.outline_count:
            lines = f", на обводке {result.outline_count}"
        elif s.outline:
            lines = ", линий для обводки не нашлось"
        self.report({"INFO"}, f"Вершин: {result.count}{lines}, мин. дистанция {obj['pointille_min_distance']:.3f} м "
                    + took)
        return {"FINISHED"}


LEFTOVER_LABELS = {"NONE": "", "PARK": "парковка", "STARS": "звёздочки"}


def budget_report(result):
    """What a layout with a budget used: created, reserve (or what the rest became), and how it fit."""
    kind = np.bincount(result.kind, minlength=4)
    figure = result.count - int(kind[3])
    rest = (f"остаток {kind[3]} размещён" if kind[3] else f"резерв {result.budget - result.count}")
    text = (f"Создано {figure} из {result.budget}, {rest} "
            f"(углы {kind[0]}, линии {kind[1]}, заливка {kind[2]})")
    fill = getattr(result, "fill_spread", result.spread)
    if fill > 1.0 + 1e-6:
        text += f"; фигуре нужно {result.needed} — заливка реже ×{fill:.2f}"
        if result.spread > 1.0 + 1e-6:
            text += f", линии реже ×{result.spread:.2f}"
    return text


def _options(context, s, count, seed, limit):
    view_dir, view_point = _audience(context, s)
    scale = SCALE_MODES[s.scale_mode]
    if limit and scale == "min_distance":
        scale = "none"  # the layout keeps the minimum distance by itself
    return core.DistributeOptions(
        count=count,
        seed=seed,
        oversample=s.oversample,
        relax_iterations=s.relax_iterations,
        facade=s.area_mode == "FACADE",
        view_dir=view_dir,
        view_point=view_point,
        facade_angle=s.facade_angle,
        audience_spread=s.audience_spread,
        hidden_density=s.hidden_density / 100.0,
        color_mode=COLOR_MODES[s.color_mode],
        min_brightness=s.min_brightness / 100.0,
        outline=s.outline,
        outline_silhouette="SILHOUETTE" in s.outline_lines,
        outline_creases="CREASES" in s.outline_lines,
        outline_borders="BORDERS" in s.outline_lines,
        outline_relief="RELIEF" in s.outline_lines,
        outline_colors="COLORS" in s.outline_lines,
        outline_amount=OUTLINE_AMOUNTS[s.outline_amount],
        outline_density=s.outline_density,
        outline_share=s.outline_share / 100.0,
        crease_angle=s.crease_angle,
        relief_angle=s.relief_angle,
        color_contrast=s.color_contrast / 100.0,
        line_ratio=s.line_ratio,
        fill_ratio=s.fill_ratio,
        around_parts=s.around_parts,
        scale_mode=scale,
        target_min_distance=s.target_min_distance,
        target_height=s.target_height,
        up_axis=2,
    )


def part_options(obj):
    """The core's options for one object of a layout with a budget."""
    p = obj.pointille_part
    lines = set(p.lines)
    return core.PartOptions(
        role=ROLES[p.role],
        pattern=PATTERNS[p.pattern],
        shapes=p.shapes,
        silhouette="SILHOUETTE" in lines,
        relief="RELIEF" in lines,
        creases="CREASES" in lines,
        borders="BORDERS" in lines,
        colors="COLORS" in lines,
        pick=PICKS[p.pick],
        groups=np.array([tuple(g.color) for g in p.groups], np.float64) if len(p.groups) else None,
        group_count=p.group_count,
        selected=[g.use for g in p.groups],
        picks=[tuple(c.color) for c in p.picks if c.use],
        tolerance=p.tolerance / 100.0,
    )


def _part_object(context, name):
    obj = bpy.data.objects.get(name) if name else context.active_object
    return obj if obj is not None and obj.type == "MESH" and not drones.is_ours(obj) else None


class POINTILLE_OT_build_curves(bpy.types.Operator):
    """Нарисовать выделенную модель кривыми, как её видят зрители: контур, черты (глаза, рот), сильные
    линии и простые фигуры, без глубоких углублений. Кривые можно править руками, а потом распределить
    вершины по ним"""

    bl_idname = "pointille.build_curves"
    bl_label = "Построить кривые"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT"

    def execute(self, context):
        s = context.scene.pointille
        models = [o for o in drones.resolve_sources(context) if o.type == "MESH"]
        if not models:
            self.report({"ERROR"}, "Выберите меш-модель")
            return {"CANCELLED"}
        started = time.perf_counter()
        try:
            depsgraph = context.evaluated_depsgraph_get()
            cache, parts, used = {}, [], []
            for o in models:
                m = adapter.object_mesh(o, depsgraph, cache)
                if m is not None:
                    parts.append(core.Part(m, part_options(o)))
                    used.append(o)
            if not parts:
                raise ValueError("у выбранных объектов нет полигонов")
            sketches = core.trace(parts, _options(context, s, s.count, s.seed, True), DEPTHS[s.curve_depth])
        except ValueError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        made = [sketch.build(context, o, sk) for o, sk in zip(used, sketches) if sk.curves]
        if not made:
            self.report({"WARNING"}, "Линий для кривых не нашлось")
            return {"CANCELLED"}
        for o in context.view_layer.objects:
            o.select_set(o in made)
        context.view_layer.objects.active = made[0]
        n = sum(len(sk.curves) for sk in sketches)
        self.report({"INFO"}, f"Кривых: {n}. Поправьте их при желании и распределите вершины "
                    f"({time.perf_counter() - started:.1f} с)")
        return {"FINISHED"}


class POINTILLE_OT_find_colors(bpy.types.Operator):
    """Разделить цвета объекта на группы, чтобы отметить, какие из них заполнять"""

    bl_idname = "pointille.find_colors"
    bl_label = "Найти цвета"
    bl_options = {"REGISTER", "UNDO"}

    object_name: StringProperty()

    def execute(self, context):
        obj = _part_object(context, self.object_name)
        if obj is None:
            self.report({"ERROR"}, "Выберите меш-модель")
            return {"CANCELLED"}
        mesh = adapter.object_mesh(obj, context.evaluated_depsgraph_get(), {})
        if mesh is None:
            self.report({"ERROR"}, "У объекта нет полигонов")
            return {"CANCELLED"}
        part = obj.pointille_part
        groups = core.color_groups(mesh, part.group_count)
        ticked = core.auto_selected(groups, context.scene.pointille.color_contrast / 100.0)
        part.groups.clear()
        for rgb, share, use in zip(groups.linear, groups.share, ticked):
            item = part.groups.add()
            item.color = [float(min(max(v, 0.0), 1.0)) for v in rgb]
            item.share = float(share) * 100.0
            item.use = bool(use)
        self.report({"INFO"}, f"Групп цветов: {len(groups.share)}")
        return {"FINISHED"}


class POINTILLE_OT_pick_add(bpy.types.Operator):
    """Добавить цвет для пипетки: заливка встанет на похожие цвета"""

    bl_idname = "pointille.pick_add"
    bl_label = "Добавить цвет"
    bl_options = {"REGISTER", "UNDO"}

    object_name: StringProperty()

    def execute(self, context):
        obj = _part_object(context, self.object_name)
        if obj is None:
            return {"CANCELLED"}
        obj.pointille_part.picks.add()
        return {"FINISHED"}


class POINTILLE_OT_pick_remove(bpy.types.Operator):
    """Убрать этот цвет"""

    bl_idname = "pointille.pick_remove"
    bl_label = "Убрать цвет"
    bl_options = {"REGISTER", "UNDO"}

    object_name: StringProperty()
    index: IntProperty()

    def execute(self, context):
        obj = _part_object(context, self.object_name)
        if obj is None or not 0 <= self.index < len(obj.pointille_part.picks):
            return {"CANCELLED"}
        obj.pointille_part.picks.remove(self.index)
        return {"FINISHED"}


class POINTILLE_OT_apply_scale(bpy.types.Operator):
    """Масштабировать модель вместе с вершинами под заданную мин. дистанцию или высоту"""

    bl_idname = "pointille.apply_scale"
    bl_label = "Масштабировать"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and active_drones(context) is not None

    def execute(self, context):
        s = context.scene.pointille
        obj = active_drones(context)
        dmin, _, size = drones.measure(obj)
        mode = "min_distance" if s.scale_mode != "HEIGHT" else "height"
        try:
            factor = core.stats.scale_factor(mode, dmin, size, 2, s.target_min_distance, s.target_height)
        except ValueError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        drones.apply_scale(context, obj, factor)
        self.report({"INFO"}, f"Масштаб ×{factor:.4g}")
        return {"FINISHED"}


class _LayerOperator(bpy.types.Operator):
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return skybrush.available(context) and skybrush.active_layer(context.scene) is not None


class POINTILLE_OT_layer_add(bpy.types.Operator):
    """Новый слой цветов на весь выбранный сегмент раскадровки: у слоя свой диапазон кадров и свой стиль"""

    bl_idname = "pointille.layer_add"
    bl_label = "Новый слой"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and skybrush.available(context) and active_drones(context) is not None

    def execute(self, context):
        segment = context.scene.pointille.skybrush_segment
        made = skybrush.add_layer(context, active_drones(context), segment)
        if made is None:
            self.report(
                {"ERROR"},
                f"Нет такой записи или перехода: «{segment}»" if segment
                else "Эти вершины не входят ни в одну формацию раскадровки Skybrush",
            )
            return {"CANCELLED"}
        effect, entry = made
        if skybrush.mapping_of(entry) is None:
            self.report({"WARNING"}, f"«{entry.name}»: пересчитайте переходы (Recalculate transitions) — цвета появятся сами")
        else:
            self.report({"INFO"}, f"{effect.name}: цвета вершин формации «{entry.name}»")
        return {"FINISHED"}


class POINTILLE_OT_layer_remove(_LayerOperator):
    """Удалить выбранный слой"""

    bl_idname = "pointille.layer_remove"
    bl_label = "Удалить слой"

    def execute(self, context):
        skybrush.remove_layer(context)
        return {"FINISHED"}


class POINTILLE_OT_layer_move(_LayerOperator):
    """Поднять или опустить слой: нижний в списке перекрывает верхний там, где диапазоны пересекаются"""

    bl_idname = "pointille.layer_move"
    bl_label = "Переместить слой"

    direction: EnumProperty(items=[("UP", "Вверх", ""), ("DOWN", "Вниз", "")])

    def execute(self, context):
        skybrush.move_layer(context, self.direction == "UP")
        return {"FINISHED"}


class POINTILLE_OT_layer_split(_LayerOperator):
    """Разрезать слой по текущему кадру на два: у каждой части свой стиль и диапазон"""

    bl_idname = "pointille.layer_split"
    bl_label = "Разрезать по кадру"

    def execute(self, context):
        if skybrush.split_layer(context, context.scene.frame_current) is None:
            self.report({"ERROR"}, "Текущий кадр должен быть внутри слоя (не на его первом кадре)")
            return {"CANCELLED"}
        return {"FINISHED"}


class POINTILLE_OT_layer_range(_LayerOperator):
    """Начало или конец слоя = текущий кадр"""

    bl_idname = "pointille.layer_range"
    bl_label = "Диапазон слоя"

    which: EnumProperty(items=[("START", "Начало", ""), ("END", "Конец", "")])

    def execute(self, context):
        skybrush.set_range(context, self.which == "END", context.scene.frame_current)
        return {"FINISHED"}


class POINTILLE_OT_paint_vertices(bpy.types.Operator):
    """Выделенные вершины (и дроны на них) горят выбранным цветом"""

    bl_idname = "pointille.paint_vertices"
    bl_label = "Покрасить выделенные"
    bl_options = {"REGISTER", "UNDO"}

    texture: BoolProperty(name="Цвета текстуры", description="Вернуть выделенным вершинам цвет текстуры")

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return context.mode == "EDIT_MESH" and obj is not None and obj.get(drones.DRONES_KEY)

    def execute(self, context):
        obj = context.active_object
        rgb = None if self.texture else tuple(context.scene.pointille.paint_color)
        changed = drones.paint_selected(obj, rgb)
        if not changed:
            no_texture = self.texture and obj.data.color_attributes.get(drones.TEXTURE_ATTR) is None
            self.report({"ERROR"}, "Нет сохранённых цветов текстуры — распределите вершины заново" if no_texture
                        else "Выделите вершины")
            return {"CANCELLED"}
        skybrush.refresh(context.scene, obj)
        self.report({"INFO"}, f"Вершин: {changed}")
        return {"FINISHED"}


class POINTILLE_OT_export_csv(bpy.types.Operator, ExportHelper):
    """Сохранить позиции (м, Z вверх) и цвета вершин в CSV"""

    bl_idname = "pointille.export_csv"
    bl_label = "Экспорт CSV"
    filename_ext = ".csv"
    filter_glob: StringProperty(default="*.csv", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return active_drones(context) is not None

    def invoke(self, context, event):
        self.filepath = active_drones(context).name + ".csv"
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        drones.export_csv(active_drones(context), self.filepath)
        self.report({"INFO"}, f"Сохранено: {self.filepath}")
        return {"FINISHED"}


classes = (
    POINTILLE_OT_distribute,
    POINTILLE_OT_build_curves,
    POINTILLE_OT_find_colors,
    POINTILLE_OT_pick_add,
    POINTILLE_OT_pick_remove,
    POINTILLE_OT_apply_scale,
    POINTILLE_OT_layer_add,
    POINTILLE_OT_layer_remove,
    POINTILLE_OT_layer_move,
    POINTILLE_OT_layer_split,
    POINTILLE_OT_layer_range,
    POINTILLE_OT_paint_vertices,
    POINTILLE_OT_export_csv,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
