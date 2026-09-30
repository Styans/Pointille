import bpy
from bpy.props import EnumProperty, FloatProperty, IntProperty, PointerProperty

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


def _can_instance(self, obj):
    from . import drones

    return not drones.is_ours(obj)


def _update_instance(self, context):
    from . import drones

    if self.get(drones.DRONES_KEY):
        drones.apply_instance(self)


class PointilleSettings(bpy.types.PropertyGroup):
    count: IntProperty(name="Вершин", description="Сколько вершин распределить по поверхности", default=500, min=1, soft_max=10000)
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
            ("AREA", "Средний по участку", "Средний цвет текстуры на участке вокруг вершины — меньше шума"),
            ("POINT", "Точно в вершине", "Цвет текстуры ровно в точке вершины"),
            ("NONE", "Не считывать", "Все вершины белые"),
        ],
        default="AREA",
    )
    scale_mode: EnumProperty(
        name="Масштаб",
        items=[
            ("NONE", "Не менять", ""),
            ("MIN_DISTANCE", "Под мин. дистанцию", "Масштабировать модель так, чтобы вершины были не ближе заданного"),
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


def register():
    bpy.utils.register_class(PointilleSettings)
    bpy.types.Scene.pointille = PointerProperty(type=PointilleSettings)
    # per vertex object: what stands on its vertices
    bpy.types.Object.pointille_instance = PointerProperty(
        name="Объект на вершинах",
        description="Любой объект (например, модель дрона) встаёт в каждую вершину в своём размере "
        "и повороте, точкой начала координат. Цвет вершины — в его материале: узел Attribute, "
        "тип Instancer, имя drone_color. Пусто — только вершины",
        type=bpy.types.Object,
        poll=_can_instance,
        update=_update_instance,
    )


def unregister():
    del bpy.types.Object.pointille_instance
    del bpy.types.Scene.pointille
    bpy.utils.unregister_class(PointilleSettings)
