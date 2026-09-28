import bpy

from . import drones
from .operators import active_drones


class POINTILLE_PT_main(bpy.types.Panel):
    bl_label = "Pointille — дроны"
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
            col.prop(s, "facade_angle")
        layout.prop(s, "color_mode")

        col = layout.column(heading="Безопасность")
        col.prop(s, "scale_mode")
        if s.scale_mode == "MIN_DISTANCE":
            col.prop(s, "target_min_distance")
        elif s.scale_mode == "HEIGHT":
            col.prop(s, "target_height")

        row = layout.row()
        row.scale_y = 1.6
        row.operator("pointille.distribute", icon="PARTICLES")


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
        col.label(text=f"Дронов: {obj.get('pointille_count', 0)}")
        col.label(text=f"Мин. дистанция: {obj.get('pointille_min_distance', 0.0):.3f} м")
        col.label(text=f"Средняя дистанция: {obj.get('pointille_mean_distance', 0.0):.3f} м")
        size = obj.get("pointille_size", (0.0, 0.0, 0.0))
        col.label(text="Габариты: " + " × ".join(f"{v:.2f}" for v in size) + " м")
        if s.scale_mode != "NONE":
            layout.operator("pointille.apply_scale", icon="FULLSCREEN_ENTER")
        row = layout.row(align=True)
        row.prop(s, "preview", toggle=True, icon="LIGHT_POINT")
        sub = row.row(align=True)
        sub.enabled = s.preview
        sub.prop(s, "preview_size", text="")
        layout.operator("pointille.export_csv", icon="EXPORT")


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


classes = (POINTILLE_PT_main, POINTILLE_PT_result, POINTILLE_PT_advanced)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
