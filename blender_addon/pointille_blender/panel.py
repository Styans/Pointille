import bpy

from . import drones, skybrush
from .operators import active_drones
from .props import LED_STYLE_ITEMS


class POINTILLE_PT_main(bpy.types.Panel):
    bl_label = "Pointille — вершины"
    bl_idname = "POINTILLE_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Pointille"

    def draw(self, context):
        s = context.scene.pointille
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        sources = drones.resolve_sources(context)
        box = layout.box()
        if sources:
            names = ", ".join(o.name for o in sources[:3]) + (" …" if len(sources) > 3 else "")
            box.label(text=f"Модель: {names}", icon="MESH_DATA")
        else:
            box.label(text="Выберите меш-модель", icon="ERROR")

        layout.prop(s, "count")
        layout.prop(s, "area_mode")
        if s.area_mode == "FACADE":
            col = layout.column(align=True)
            col.prop(s, "view")
            col.prop(s, "audience_spread")
            col.prop(s, "facade_angle")
            col.prop(s, "hidden_density", slider=True)
        layout.prop(s, "color_mode")
        if s.color_mode != "NONE":
            layout.prop(s, "min_brightness", slider=True)

        layout.prop(s, "outline")
        if s.outline:
            col = layout.column(align=True)
            col.prop(s, "outline_lines")
            col.prop(s, "outline_amount", expand=True)
            if s.outline_amount == "DENSITY":
                col.prop(s, "outline_density")
            else:
                col.prop(s, "outline_share", slider=True)
            if "RELIEF" in s.outline_lines:
                col.prop(s, "relief_angle")
            if "CREASES" in s.outline_lines:
                col.prop(s, "crease_angle")
            if "COLORS" in s.outline_lines:
                col.prop(s, "color_contrast", slider=True)
            if {"SILHOUETTE", "RELIEF"} & set(s.outline_lines) and s.area_mode != "FACADE":
                col.prop(s, "view")

        col = layout.column(heading="Безопасность")
        col.prop(s, "scale_mode")
        if s.scale_mode == "MIN_DISTANCE":
            col.prop(s, "target_min_distance")
        elif s.scale_mode == "HEIGHT":
            col.prop(s, "target_height")

        row = layout.row()
        row.scale_y = 1.6
        row.operator("pointille.distribute", icon="VERTEXSEL")


class POINTILLE_PT_result(bpy.types.Panel):
    bl_label = "Результат"
    bl_idname = "POINTILLE_PT_result"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Pointille"
    bl_parent_id = "POINTILLE_PT_main"

    @classmethod
    def poll(cls, context):
        return active_drones(context) is not None

    def draw(self, context):
        s = context.scene.pointille
        obj = active_drones(context)
        layout = self.layout
        col = layout.column(align=True)
        col.label(text=f"Объект: {obj.name}", icon="OUTLINER_OB_POINTCLOUD")
        col.label(text=f"Вершин: {obj.get('pointille_count', 0)}")
        col.label(text=f"Мин. дистанция: {obj.get('pointille_min_distance', 0.0):.3f} м")
        col.label(text=f"Средняя дистанция: {obj.get('pointille_mean_distance', 0.0):.3f} м")
        if obj.get("pointille_outline_count"):
            col.label(text=f"На обводке: {obj['pointille_outline_count']}, шаг "
                      f"{obj.get('pointille_outline_spacing', 0.0):.3f} м")
        size = obj.get("pointille_size", (0.0, 0.0, 0.0))
        col.label(text="Габариты: " + " × ".join(f"{v:.2f}" for v in size) + " м")

        col = layout.column(align=True)
        col.label(text="Объект на вершинах:")
        col.prop(obj, "pointille_instance", text="")
        if obj.pointille_instance is not None:
            col.label(text=f"Цвет: Instancer › {drones.COLOR_ATTR}", icon="INFO")

        if s.scale_mode != "NONE":
            layout.operator("pointille.apply_scale", icon="FULLSCREEN_ENTER")
        layout.operator("pointille.export_csv", icon="EXPORT")

        if context.mode == "EDIT_MESH" and obj is context.active_object:
            box = layout.box()
            box.label(text="Цвет выделенных вершин", icon="VERTEXSEL")
            row = box.row(align=True)
            row.prop(s, "paint_color", text="")
            row.operator("pointille.paint_vertices", text="Покрасить")
            box.operator("pointille.paint_vertices", text="Вернуть цвета текстуры", icon="LOOP_BACK").texture = True

        if skybrush.available(context):
            self.draw_skybrush(context, layout, s, obj)


    def draw_skybrush(self, context, layout, s, obj):
        sb = context.scene.skybrush
        box = layout.box()
        entries = skybrush.entries_with(context.scene, obj)
        box.label(text="Skybrush: " + (", ".join(e.name for e in entries) if entries else "нет в раскадровке"),
                  icon="LIGHT")
        # the same field as "Use selected" in Object Data > Drone Show
        group = obj.skybrush.formation_vertex_group
        col = box.column(align=True)
        col.label(text="Группа вершин формации:")
        col.prop_search(obj.skybrush, "formation_vertex_group", obj, "vertex_groups", text="")
        if not group:
            col.label(text="Не задана: весь объект = 1 дрон", icon="ERROR")
        elif obj.vertex_groups.get(group) is None:
            col.label(text="Нет такой группы: 0 дронов", icon="ERROR")
        if not entries:
            return
        row = box.row()
        row.template_list("POINTILLE_UL_layers", "", sb.light_effects, "entries", sb.light_effects,
                          "active_entry_index", rows=4)
        col = row.column(align=True)
        col.operator("pointille.layer_add", icon="ADD", text="")
        col.operator("pointille.layer_remove", icon="REMOVE", text="")
        col.separator()
        col.operator("pointille.layer_move", icon="TRIA_UP", text="").direction = "UP"
        col.operator("pointille.layer_move", icon="TRIA_DOWN", text="").direction = "DOWN"
        box.prop_search(s, "skybrush_segment", sb.storyboard, "entries_or_transitions")

        effect = skybrush.active_layer(context.scene)
        if effect is None:
            box.label(text="Слой = свой диапазон кадров и стиль. Нажмите +", icon="INFO")
            return
        col = box.column(align=True)
        col.prop(s, "led_name")
        col.prop_search(s, "led_source", sb.storyboard, "entries")
        col.prop_search(effect, "storyboard_entry_or_transition_selection", sb.storyboard, "entries_or_transitions",
                        text="К записи")

        col = box.column(align=True)
        col.label(text="Диапазон (кадры)")
        for text, prop, which in (("Начало", "frame_start", "START"), ("Конец", "frame_end", "END")):
            row = col.row(align=True)
            row.prop(effect, prop, text=text)
            row.operator("pointille.layer_range", text="", icon="TIME").which = which
        col.operator("pointille.layer_split", icon="AREA_SWAP")
        col.prop(effect, "fade_in_duration", text="Плавно зажечь")
        col.prop(effect, "fade_out_duration", text="Плавно погасить")
        col.prop(effect, "influence", text="Влияние", slider=True)

        col = box.column(align=True)
        col.prop(s, "led_style")
        if s.led_style not in ("TEXTURE", "OFF"):
            col.prop(s, "led_mix", slider=True)
            if s.led_style in ("SOLID", "GRADIENT"):
                col.prop(s, "led_color")
            if s.led_style == "GRADIENT":
                col.prop(s, "led_color2")
            col.prop(s, "led_speed")
            if s.led_style != "SOLID":
                col.prop(s, "led_spread")
        elif s.led_style == "OFF":
            col.prop(s, "led_mix", text="Приглушить", slider=True)
        col.prop(s, "led_brightness", slider=True)
        col.prop(s, "led_saturation", slider=True)


class POINTILLE_UL_layers(bpy.types.UIList):
    """Слои Pointille среди световых эффектов Skybrush."""

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        row.label(text=f"{item.frame_start}–{item.frame_end}")
        style = skybrush.effect_params(item)["style"]
        row.label(text=next((name for key, name, _ in LED_STYLE_ITEMS if key == style), style))

    def filter_items(self, context, data, propname):
        items = getattr(data, propname)
        return [self.bitflag_filter_item if e.name.startswith(skybrush.EFFECT_PREFIX) else 0 for e in items], []


class POINTILLE_PT_advanced(bpy.types.Panel):
    bl_label = "Дополнительно"
    bl_idname = "POINTILLE_PT_advanced"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Pointille"
    bl_parent_id = "POINTILLE_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        s = context.scene.pointille
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.prop(s, "seed")
        layout.prop(s, "relax_iterations")
        layout.prop(s, "oversample")


classes = (POINTILLE_UL_layers, POINTILLE_PT_main, POINTILLE_PT_result, POINTILLE_PT_advanced)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
