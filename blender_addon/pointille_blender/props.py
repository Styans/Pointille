import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty

VIEW_ITEMS = [
    ("FRONT", "Спереди (−Y)", "Зрители смотрят вдоль +Y, как в виде Front"),
    ("BACK", "Сзади (+Y)", ""),
    ("LEFT", "Слева (−X)", ""),
    ("RIGHT", "Справа (+X)", ""),
    ("TOP", "Сверху (+Z)", ""),
    ("CAMERA", "От активной камеры", "Направление взгляда активной камеры сцены"),
]
VIEW_DIRS = {
    "FRONT": (0.0, -1.0, 0.0),
    "BACK": (0.0, 1.0, 0.0),
    "LEFT": (-1.0, 0.0, 0.0),
    "RIGHT": (1.0, 0.0, 0.0),
    "TOP": (0.0, 0.0, 1.0),
}


def _update_preview(self, context):
    from . import drones

    obj = drones.find_drones(context.active_object)
    if obj is not None:
        drones.refresh_preview(context, obj)


class PointilleSettings(bpy.types.PropertyGroup):
    count: IntProperty(name="Дронов", description="Сколько точек-дронов распределить", default=500, min=1, soft_max=10000)
    seed: IntProperty(name="Seed", description="Другое число — другая, но такая же ровная расстановка", default=0, min=0)
    area_mode: EnumProperty(
        name="Область",
        items=[
            ("SURFACE", "Вся поверхность", "Равномерно по всей модели"),
            ("FACADE", "Фасад", "Только сторона, которую видят зрители"),
        ],
        default="SURFACE",
    )
    view: EnumProperty(name="Зрители", items=VIEW_ITEMS, default="FRONT")
    facade_angle: FloatProperty(
        name="Угол", description="Поверхности, повёрнутые к зрителю сильнее этого угла, пропускаются",
        default=80.0, min=1.0, max=90.0, subtype="NONE",
    )
    color_mode: EnumProperty(
        name="Цвет",
        items=[
            ("AREA", "Средний по участку", "Средний цвет текстуры на участке вокруг дрона — меньше шума"),
            ("POINT", "Точно под дроном", "Цвет текстуры ровно в точке дрона"),
            ("NONE", "Не считывать", "Все дроны белые"),
        ],
        default="AREA",
    )
    scale_mode: EnumProperty(
        name="Масштаб",
        items=[
            ("NONE", "Не менять", ""),
            ("MIN_DISTANCE", "Под мин. дистанцию", "Масштабировать модель так, чтобы дроны были не ближе заданного"),
            ("HEIGHT", "Под высоту", "Масштабировать модель под заданную высоту фигуры"),
        ],
        default="NONE",
    )
    target_min_distance: FloatProperty(name="Мин. дистанция", default=1.5, min=0.01, unit="LENGTH")
    target_height: FloatProperty(name="Высота фигуры", default=20.0, min=0.01, unit="LENGTH")
    relax_iterations: IntProperty(
        name="Выравнивание", description="Итерации выравнивания (0 — выкл.)", default=20, min=0, max=200,
    )
    oversample: FloatProperty(name="Плотность кандидатов", default=8.0, min=2.0, max=32.0)
    preview: BoolProperty(
        name="Светящиеся шары", description="Показать дронов шарами их цвета", default=True, update=_update_preview,
    )
    preview_size: FloatProperty(
        name="Размер шара", description="Радиус шара в долях мин. дистанции", default=0.2, min=0.01, max=0.5,
        update=_update_preview,
    )


def register():
    bpy.utils.register_class(PointilleSettings)
    bpy.types.Scene.pointille = PointerProperty(type=PointilleSettings)


def unregister():
    del bpy.types.Scene.pointille
    bpy.utils.unregister_class(PointilleSettings)
