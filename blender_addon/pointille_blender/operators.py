import time

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty
from bpy_extras.io_utils import ExportHelper
from mathutils import Vector

from . import adapter, drones, skybrush
from ._vendor import core
from .props import VIEW_DIRS

SCALE_MODES = {"NONE": "none", "MIN_DISTANCE": "min_distance", "HEIGHT": "height"}
COLOR_MODES = {"AREA": "area", "POINT": "point", "NONE": "none"}


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
        try:
            mesh = adapter.scene_mesh(sources, context.evaluated_depsgraph_get())
            view_dir, view_point = _audience(context, s)
            opts = core.DistributeOptions(
                count=self.count,
                seed=self.seed,
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
                scale_mode=SCALE_MODES[s.scale_mode],
                target_min_distance=s.target_min_distance,
                target_height=s.target_height,
                up_axis=2,
            )
            result = core.distribute(mesh, opts)
        except ValueError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        obj = drones.write_drones(context, sources, result)
        if s.scale_mode != "NONE":
            drones.apply_scale(context, obj, result.scale)
        skybrush.refresh(context.scene, obj)  # new colours for the Skybrush effects made earlier
        self.report(
            {"INFO"},
            f"Вершин: {result.count}, мин. дистанция {obj['pointille_min_distance']:.3f} м "
            f"({time.perf_counter() - started:.1f} с)",
        )
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
