import time

import bpy
from bpy.props import IntProperty, StringProperty
from bpy_extras.io_utils import ExportHelper
from mathutils import Vector

from . import adapter, drones
from ._vendor import core
from .props import VIEW_DIRS

SCALE_MODES = {"NONE": "none", "MIN_DISTANCE": "min_distance", "HEIGHT": "height"}
COLOR_MODES = {"AREA": "area", "POINT": "point", "NONE": "none"}


def _view_dir(context, settings):
    if settings.view == "CAMERA":
        cam = context.scene.camera
        if cam is None:
            raise ValueError("в сцене нет активной камеры")
        # from the model towards the camera = opposite of the camera's look direction
        return tuple(cam.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0)))
    return VIEW_DIRS[settings.view]


def active_drones(context):
    return drones.find_drones(context.active_object)


class POINTILLE_OT_distribute(bpy.types.Operator):
    """Равномерно распределить дронов по поверхности выбранной модели (модель не меняется)"""

    bl_idname = "pointille.distribute"
    bl_label = "Распределить дронов"
    bl_options = {"REGISTER", "UNDO"}

    count: IntProperty(name="Дронов", default=500, min=1, soft_max=10000)
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
            opts = core.DistributeOptions(
                count=self.count,
                seed=self.seed,
                oversample=s.oversample,
                relax_iterations=s.relax_iterations,
                facade=s.area_mode == "FACADE",
                view_dir=_view_dir(context, s),
                facade_angle=s.facade_angle,
                color_mode=COLOR_MODES[s.color_mode],
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
        drones.refresh_preview(context, obj)
        self.report(
            {"INFO"},
            f"{result.count} дронов, мин. дистанция {obj['pointille_min_distance']:.3f} м "
            f"({time.perf_counter() - started:.1f} с)",
        )
        return {"FINISHED"}


class POINTILLE_OT_apply_scale(bpy.types.Operator):
    """Масштабировать модель вместе с дронами под заданную мин. дистанцию или высоту"""

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
        drones.refresh_preview(context, obj)
        self.report({"INFO"}, f"Масштаб ×{factor:.4g}")
        return {"FINISHED"}


class POINTILLE_OT_refresh_preview(bpy.types.Operator):
    """Обновить светящиеся шары"""

    bl_idname = "pointille.refresh_preview"
    bl_label = "Обновить превью"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return active_drones(context) is not None

    def execute(self, context):
        drones.refresh_preview(context, active_drones(context))
        return {"FINISHED"}


class POINTILLE_OT_export_csv(bpy.types.Operator, ExportHelper):
    """Сохранить позиции (м, Z вверх) и цвета дронов в CSV"""

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
    POINTILLE_OT_refresh_preview,
    POINTILLE_OT_export_csv,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
