import bpy
from bpy.props import EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, PointerProperty, StringProperty

VIEW_ITEMS = [
    ("FRONT", "Спереди (−Y)", "Зрители смотрят вдоль +Y, как в виде Front"),
    ("BACK", "Сзади (+Y)", ""),
    ("LEFT", "Слева (−X)", ""),
    ("RIGHT", "Справа (+X)", ""),
    ("TOP", "Сверху (+Z)", ""),
    ("CAMERA", "От активной камеры", "Зрители там, где стоит активная камера сцены: с перспективой, как в её кадре"),
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


LED_STYLE_ITEMS = [
    ("TEXTURE", "Цвета текстуры", "Дрон горит цветом своей вершины"),
    ("GRAYSCALE", "Чёрно-белое", "Оттенки серого по яркости текстуры; со скоростью — волны глубины"),
    ("SHIMMER", "Переливы текстуры", "Яркость точек как в текстуре, а оттенок плывёт по цветовому кругу"),
    ("RAINBOW", "Радуга", "Радужные цвета; светлые и тёмные места текстуры сохраняют глубину"),
    ("GRADIENT", "Градиент двух цветов", "Фоновые цвета снизу вверх (например, вода); со скоростью — течёт"),
    ("SOLID", "Один цвет", "Все дроны одним цветом (с дыханием, если задана скорость)"),
    ("OFF", "Погашены", "Дроны не светятся (с долей стиля — приглушены)"),
]
STYLE_IDS = [item[0] for item in LED_STYLE_ITEMS]

# The settings below are proxies: they read and write the style of the layer selected in the
# Skybrush light effect list (see skybrush.py), so each layer keeps its own.


def _skybrush():
    from . import skybrush

    return skybrush


def _led_float(key, **kw):
    def get(self):
        return float(_skybrush().active_param(key))

    def set(self, value):  # noqa: A001
        _skybrush().set_active_param(key, float(value))

    return FloatProperty(get=get, set=set, **kw)


def _led_color(key, **kw):
    def get(self):
        return tuple(_skybrush().active_param(key))

    def set(self, value):  # noqa: A001
        _skybrush().set_active_param(key, [float(v) for v in value])

    return FloatVectorProperty(get=get, set=set, subtype="COLOR", size=3, min=0.0, max=1.0, **kw)


def _led_style_get(self):
    return STYLE_IDS.index(_skybrush().active_param("style"))


def _led_style_set(self, value):
    _skybrush().set_active_param("style", STYLE_IDS[value])


def _led_name_get(self):
    return _skybrush().layer_name()


def _led_name_set(self, value):
    _skybrush().rename_layer(value)


def _led_source_get(self):
    return _skybrush().layer_source()


def _led_source_set(self, value):
    _skybrush().set_layer_source(value)


class PointilleSettings(bpy.types.PropertyGroup):
    count: IntProperty(name="Вершин", description="Сколько вершин распределить по поверхности", default=500, min=1, soft_max=10000)
    seed: IntProperty(name="Seed", description="Другое число — другая, но такая же ровная расстановка", default=0, min=0)
    area_mode: EnumProperty(
        name="Область",
        items=[
            ("SURFACE", "Вся поверхность", "Равномерно по всей модели"),
            ("FACADE", "Фасад", "Гуще там, где модель видят зрители; скрытые части — реже или без вершин"),
        ],
        default="SURFACE",
    )
    view: EnumProperty(name="Зрители", items=VIEW_ITEMS, default="FRONT")
    audience_spread: FloatProperty(
        name="Ширина зрителей",
        description="На сколько градусов зрители растянуты вокруг модели по горизонтали. Углы и бока, "
        "которые видит только часть зрителей, получают вершины реже фасада. 0 — все смотрят из одной точки",
        default=60.0, min=0.0, max=180.0, subtype="NONE",
    )
    facade_angle: FloatProperty(
        name="Угол", description="Поверхности, повёрнутые к зрителю сильнее этого угла, он не видит",
        default=80.0, min=1.0, max=90.0, subtype="NONE",
    )
    hidden_density: FloatProperty(
        name="Скрытые части",
        description="Плотность вершин там, где модель не видит никто из зрителей, в процентах от фасада. "
        "Фигура остаётся целой при повороте и анимации. 0 — там вершин нет",
        default=25.0, min=0.0, max=100.0, subtype="PERCENTAGE",
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
    min_brightness: FloatProperty(
        name="Мин. яркость",
        description="Дрон не светит чёрным: выключенный дрон ночью не виден. Самая тёмная вершина горит с этой "
        "яркостью, самая яркая — в полную силу; оттенок и порядок яркостей сохраняются. 0 — цвета как у модели",
        default=40.0, min=0.0, max=100.0, subtype="PERCENTAGE",
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
    # Skybrush drones: when they show the vertex colours and how
    skybrush_segment: StringProperty(
        name="Сегмент",
        description="Новый слой (кнопка +) встаёт на всю эту запись раскадровки или переход. "
        "Пусто — формация из этих вершин",
    )
    led_name: StringProperty(name="Слой", description="Название слоя", get=_led_name_get, set=_led_name_set)
    led_source: StringProperty(
        name="Цвета из", description="Формация, по вершинам которой слой берёт цвета дронов",
        get=_led_source_get, set=_led_source_set,
    )
    led_style: EnumProperty(
        name="Стиль", description="Как дроны используют цвета текстуры", items=LED_STYLE_ITEMS,
        get=_led_style_get, set=_led_style_set,
    )
    led_mix: _led_float("mix", name="Доля стиля", description="0 — только цвета текстуры, 1 — только стиль",
                        min=0.0, max=1.0)
    led_speed: _led_float("speed", name="Скорость", description="Циклов в секунду; 0 — без движения",
                          min=0.0, soft_max=2.0, precision=2)
    led_spread: _led_float("spread", name="Разброс", description="Сколько циклов укладывается по высоте фигуры",
                           min=-4.0, max=4.0, precision=2)
    led_color: _led_color("color", name="Цвет", description="Первый цвет стиля")
    led_color2: _led_color("color2", name="Цвет 2", description="Второй цвет градиента")
    led_brightness: _led_float("brightness", name="Яркость", description="Множитель яркости цветов дронов",
                               min=0.0, soft_max=3.0)
    led_saturation: _led_float("saturation", name="Насыщенность",
                               description="0 — серые, 1 — как есть, больше — ярче цвета", min=0.0, soft_max=2.0)
    paint_color: FloatVectorProperty(
        name="Цвет", description="Цвет для выделенных вершин", subtype="COLOR", size=3, min=0.0, max=1.0,
        default=(1.0, 0.0, 0.0),
    )


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
