import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty,
                       PointerProperty, StringProperty)

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


OUTLINE_LINE_ITEMS = [
    ("SILHOUETTE", "Силуэт", "Контур фигуры, каким его видят зрители (настройка «Зрители»), и края, где одна "
     "часть закрывает другую (рука перед телом)"),
    ("RELIEF", "Рельеф", "Что нарисовал бы художник внутри контура: глаза, ноздри, рот, уши, складки — где "
     "поверхность резко изгибается. Ищется по картинке модели с точки зрения зрителей"),
    ("CREASES", "Рёбра", "Острые изломы поверхности: грани, углы, стыки частей"),
    ("BORDERS", "Края", "Края открытой поверхности: текст, логотипы, плоскости"),
    ("COLORS", "Цвета", "Границы цветов текстуры: глаза, рот, полосы, картинка на флаге"),
]

ROLE_ITEMS = [
    ("FULL", "Контур + заливка", "Сначала углы и линии объекта, потом редкая заливка остальной поверхности"),
    ("OUTLINE", "Только контур", "Только углы и линии, без заливки (например, скала под фигурой)"),
    ("PATTERN", "Рисунок по цветам", "Линии и заливка только выбранных цветов (чёрные полосы тигра без "
     "оранжевой шерсти)"),
    ("FILL", "Только заливка", "Без линий: поверхность ровной сеткой или равномерно"),
]
PATTERN_ITEMS = [
    ("HEX", "Треугольная сетка", "Ряды со сдвигом (соты): самое плотное и ровное заполнение при той же дистанции"),
    ("SQUARE", "Квадратная сетка", "Ровные ряды и столбцы"),
    ("NOISE", "Равномерно", "Без сетки: как прежде, равномерный шум"),
    ("RINGS", "Кольца", "Кольцами вокруг вертикальной оси снизу вверх, по самой поверхности: шар, тыква, ваза "
     "сразу читаются круглыми"),
]
PICK_ITEMS = [
    ("GROUPS", "Группы цветов", "Плагин делит цвета модели на группы; отметьте, какие заполнять"),
    ("EYEDROPPER", "Пипетка", "Заполняются цвета, похожие на выбранные пипеткой, в пределах допуска"),
    ("AUTO", "Авто", "Заполняются все группы, заметно отличающиеся от основного цвета модели"),
]

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


class PointilleColor(bpy.types.PropertyGroup):
    color: FloatVectorProperty(name="Цвет", subtype="COLOR", size=3, min=0.0, max=1.0, default=(0.0, 0.0, 0.0))
    use: BoolProperty(name="Заполнять", description="Заливка встаёт на этот цвет", default=True)
    share: FloatProperty(name="Доля", description="Какая часть поверхности этого цвета", subtype="PERCENTAGE")


class PointillePart(bpy.types.PropertyGroup):
    """How one of the selected objects takes its drones (layout with a budget)."""

    role: EnumProperty(name="Режим", items=ROLE_ITEMS, default="FULL")
    pattern: EnumProperty(name="Заливка", items=PATTERN_ITEMS, default="HEX")
    shapes: BoolProperty(
        name="Простые фигуры",
        description="Черты объекта рисуются простыми фигурами, как у художника шоу: глаз — кольцом (с "
        "зрачком, если есть место), почти круглая или овальная линия — ровным кругом или эллипсом, почти прямой "
        "отрезок — прямым, а круглое тело (шар, тыква) заполняется кольцами снизу вверх",
        default=True,
    )
    lines: EnumProperty(
        name="Линии", description="Какие линии объекта обводить", items=OUTLINE_LINE_ITEMS, options={"ENUM_FLAG"},
        default={"SILHOUETTE", "RELIEF", "CREASES", "BORDERS", "COLORS"},
    )
    pick: EnumProperty(name="Какие цвета", items=PICK_ITEMS, default="GROUPS")
    group_count: IntProperty(name="Групп", description="На сколько групп делить цвета модели", default=4, min=2,
                             max=8)
    groups: CollectionProperty(type=PointilleColor)
    picks: CollectionProperty(type=PointilleColor)
    tolerance: FloatProperty(
        name="Допуск", description="Насколько цвет может отличаться от выбранного пипеткой (100 % — как чёрный "
        "и белый)", default=15.0, min=1.0, max=100.0, subtype="PERCENTAGE",
    )


class PointilleSettings(bpy.types.PropertyGroup):
    count_mode: EnumProperty(
        name="Количество",
        items=[
            ("LIMIT", "Не больше", "Сначала углы, потом линии, потом редкая заливка — сколько нужно фигуре, но не "
             "больше лимита; остаток остаётся в резерве"),
            ("EXACT", "Ровно", "Ровно столько вершин, равномерно по всей выделенной модели (как прежде)"),
        ],
        default="LIMIT",
    )
    leftover_mode: EnumProperty(
        name="Остаток",
        description="Что делать с дронами, которые фигуре не понадобились (лимит минус созданные). Skybrush "
        "оставляет дрона без точки в формации там, где он был, и он светится",
        items=[
            ("NONE", "Не трогать", "Остаток остаётся в резерве: вершин меньше лимита"),
            ("PARK", "Парковка", "Остаток — тёмные дроны (свет выключен) ровной сеткой под фигурой; если "
             "фигура низко — за ней. Вершин ровно по лимиту"),
            ("STARS", "Звёздочки", "Остаток — звёздочки из 5 дронов и искры вокруг фигуры, тёплый белый, как "
             "в больших шоу; что не поместилось — на парковку. Вершин ровно по лимиту"),
        ],
        default="NONE",
    )
    around_parts: BoolProperty(
        name="По частям тела",
        description="Персонажи и животные: волна из середины модели находит части тела. Туловище и висящие "
        "лапы — горизонтальными рядами, хвост, крылья, лежащее тело — кольцами поперёк, яркие столбики "
        "рисуют силуэт с любой стороны, кончики лап и хвоста — яркими точками. Выключено — горизонтальные "
        "ряды везде (как в 0.18)",
        default=True,
    )
    count: IntProperty(name="Вершин", description="Сколько вершин распределить по поверхности; в режиме «Не "
                       "больше» — сколько можно использовать самое большее", default=500, min=1, soft_max=10000)
    line_ratio: FloatProperty(
        name="Шаг линий", description="Во сколько раз вершины на линиях стоят дальше друг от друга, чем в углах "
        "(там — мин. дистанция)", default=1.5, min=1.0, max=5.0,
    )
    curve_depth: EnumProperty(
        name="Глубина кривых",
        description="Как глубоко лежат кривые, построенные из модели: углубления (вырезанные глаза, рот) "
        "всегда убраны, а глубина фигуры",
        items=[
            ("FLAT", "Плоско", "Все кривые в одной плоскости лицом к зрителям: рисунок читается с любого места"),
            ("SHALLOW", "Неглубоко", "Четверть глубины фигуры: немного объёма, рисунок читается почти отовсюду"),
            ("SHAPE", "По форме", "На сглаженной поверхности модели: объём как у модели"),
            ("VOLUME", "Объём", "Вся модель, видна со всех сторон: горизонтальные срезы снизу доверху и прямо "
             "на поверхности контур, глаза, рот и сильные линии с четырёх сторон"),
        ],
        default="SHALLOW",
    )
    fill_ratio: FloatProperty(
        name="Шаг заливки", description="Во сколько раз вершины заливки стоят дальше друг от друга, чем в углах",
        default=3.0, min=1.0, max=10.0,
    )
    seed: IntProperty(name="Seed", description="Другое число — другая, но такая же ровная расстановка", default=0, min=0)
    area_mode: EnumProperty(
        name="Область",
        items=[
            ("SURFACE", "Вся поверхность", "Равномерно по всей модели"),
            ("FACADE", "Фасад", "Гуще там, где модель видят зрители; скрытые части — реже или без вершин"),
            ("AROUND", "Вокруг (3D)", "Фигура видна со всех сторон: чёткий силуэт и детали линиями. Круглые "
             "тела (фрукты, тыквы) — каркас из рёбер и колец, персонажи — контур, черты лица и корона"),
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
    outline: BoolProperty(
        name="Обводка",
        description="Часть вершин встаёт точно на линии модели гуще заливки, а заливка оставляет им место: "
        "контур и черты фигуры читаются чётче. В углах линий вершина стоит всегда",
        default=False,
    )
    outline_lines: EnumProperty(
        name="Линии", description="Какие линии модели обводить", items=OUTLINE_LINE_ITEMS, options={"ENUM_FLAG"},
        default={"SILHOUETTE", "RELIEF", "CREASES", "BORDERS", "COLORS"},
    )
    outline_amount: EnumProperty(
        name="Сколько",
        items=[
            ("DENSITY", "Плотность", "Вершины на линиях стоят во столько раз гуще заливки; сколько их — "
             "по длине линий, но не больше половины всех"),
            ("SHARE", "Доля", "Столько процентов вершин встаёт на линии; шаг на них — из длины линий"),
        ],
        default="DENSITY",
    )
    outline_density: FloatProperty(
        name="Гуще заливки", description="Во сколько раз вершины на линиях стоят ближе друг к другу, чем в заливке",
        default=1.5, min=1.0, max=3.0,
    )
    outline_share: FloatProperty(
        name="Доля вершин", description="Сколько вершин из всех встаёт на линии",
        default=30.0, min=5.0, max=60.0, subtype="PERCENTAGE",
    )
    crease_angle: FloatProperty(
        name="Угол рёбер", description="Грани, которые сходятся под углом больше этого, дают линию ребра",
        default=40.0, min=5.0, max=170.0,
    )
    relief_angle: FloatProperty(
        name="Изгиб рельефа",
        description="На сколько градусов поверхность должна изогнуться на коротком отрезке (около трети шага "
        "дронов), чтобы там прошла линия рельефа. Меньше — больше деталей, но и шума (мех, вмятины)",
        default=40.0, min=10.0, max=120.0,
    )
    color_contrast: FloatProperty(
        name="Контраст цветов",
        description="Насколько должны различаться соседние цвета текстуры, чтобы между ними прошла линия "
        "(100 % — как чёрный и белый). Меньше — больше линий, но и шума",
        default=15.0, min=1.0, max=100.0, subtype="PERCENTAGE",
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
    target_min_distance: FloatProperty(
        name="Мин. дистанция", description="Ближе этого вершины не встанут; в режиме «Не больше» так близко стоят "
        "вершины в углах фигуры", default=1.5, min=0.01, unit="LENGTH",
    )
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
    bpy.utils.register_class(PointilleColor)
    bpy.utils.register_class(PointillePart)
    bpy.utils.register_class(PointilleSettings)
    bpy.types.Scene.pointille = PointerProperty(type=PointilleSettings)
    # per model object: its role in a layout with a budget
    bpy.types.Object.pointille_part = PointerProperty(type=PointillePart)
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
    del bpy.types.Object.pointille_part
    del bpy.types.Scene.pointille
    bpy.utils.unregister_class(PointilleSettings)
    bpy.utils.unregister_class(PointillePart)
    bpy.utils.unregister_class(PointilleColor)
