"""
Blender script to import OpenStreetMap data and export to Sionna-compatible Mitsuba format.

This script is designed to be run inside Blender in headless mode:
    blender --background --python blender_script.py -- lat_min lat_max lon_min lon_max output_dir scene_name

Prerequisites:
- Blender 3.6+ or 4.2 LTS
- Blosm addon installed (https://prochitecture.gumroad.com/l/blender-osm)
- Mitsuba-Blender addon installed (https://github.com/mitsuba-renderer/mitsuba-blender)
"""

import bpy
import bmesh
import sys
import os
import math
import time
from mathutils import Vector


MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5


def parse_arguments() -> tuple[float, float, float, float, str, str]:
    """Parse command line arguments passed after '--'."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        raise ValueError("No arguments found. Use: blender --background --python script.py -- args...")

    if len(argv) < 6:
        raise ValueError(
            "Expected 6 arguments: lat_min lat_max lon_min lon_max output_dir scene_name"
        )

    lat_min = float(argv[0])
    lat_max = float(argv[1])
    lon_min = float(argv[2])
    lon_max = float(argv[3])
    output_dir = argv[4]
    scene_name = argv[5]

    return lat_min, lat_max, lon_min, lon_max, output_dir, scene_name


def clear_scene() -> None:
    """Remove all objects from the default scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)


def enable_addons() -> None:
    """Enable required Blender addons."""
    try:
        bpy.ops.preferences.addon_enable(module='blosm')
        print("Blosm addon enabled successfully")
    except Exception as e:
        print(f"Warning: Could not enable blosm addon: {e}")
        print("Make sure blosm is installed in Blender's addons directory")

    try:
        bpy.ops.preferences.addon_enable(module='mitsuba-blender')
        print("Mitsuba-Blender addon enabled successfully")
    except Exception as e:
        print(f"Warning: Could not enable mitsuba-blender addon: {e}")
        print("Make sure mitsuba-blender is installed in Blender's addons directory")


def import_osm_data(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> None:
    """Import OpenStreetMap data using the blosm addon."""
    scene = bpy.context.scene

    scene.blosm.minLat = lat_min
    scene.blosm.maxLat = lat_max
    scene.blosm.minLon = lon_min
    scene.blosm.maxLon = lon_max
    scene.blosm.dataType = "osm"

    available_modes = scene.blosm.bl_rna.properties["mode"].enum_items.keys()
    if "3Drealistic" in available_modes:
        scene.blosm.mode = "3Drealistic"
    elif "3Dsimple" in available_modes:
        scene.blosm.mode = "3Dsimple"
    else:
        scene.blosm.mode = list(available_modes)[0]
    print(f"Using blosm mode: {scene.blosm.mode}")

    # Enable road/path import
    scene.blosm.highways = True
    print("Highway/road import enabled")

    print(f"Importing OSM data for coordinates:")
    print(f"  Latitude: {lat_min} to {lat_max}")
    print(f"  Longitude: {lon_min} to {lon_max}")

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            bpy.ops.blosm.import_data()
            print(f"Imported {len(bpy.data.objects)} objects")
            return
        except RuntimeError as e:
            last_error = e
            error_str = str(e)
            if "HTTP Error" in error_str or "Timeout" in error_str or "Connection" in error_str:
                if attempt < MAX_RETRIES:
                    print(f"Network error (attempt {attempt}/{MAX_RETRIES}): {error_str}")
                    print(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    print(f"Network error (attempt {attempt}/{MAX_RETRIES}): {error_str}")
                    raise
            else:
                raise

    raise last_error if last_error else RuntimeError("OSM import failed")


def create_itu_materials() -> dict[str, bpy.types.Material]:
    """
    Create Sionna-compatible ITU materials.

    Returns dict mapping material names to Blender materials.
    Material names must match Sionna's ITU material naming convention.
    """
    materials = {}

    # itu_concrete - for ground/roads (gray)
    mat_concrete = bpy.data.materials.new(name="itu_concrete")
    mat_concrete.use_nodes = True
    bsdf = mat_concrete.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.393, 0.393, 0.393, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.8
    materials["itu_concrete"] = mat_concrete

    # itu_marble - for building walls (light gray/white)
    mat_marble = bpy.data.materials.new(name="itu_marble")
    mat_marble.use_nodes = True
    bsdf = mat_marble.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (1.0, 0.989, 0.957, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.4
    materials["itu_marble"] = mat_marble

    # itu_metal - for roofs (orange/brown metallic)
    mat_metal = bpy.data.materials.new(name="itu_metal")
    mat_metal.use_nodes = True
    bsdf = mat_metal.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.662, 0.228, 0.042, 1.0)
        bsdf.inputs["Metallic"].default_value = 0.8
        bsdf.inputs["Roughness"].default_value = 0.3
    materials["itu_metal"] = mat_metal

    print("Created ITU materials: itu_concrete, itu_marble, itu_metal")
    return materials


def get_face_orientation(face_normal: Vector) -> str:
    """
    Determine if a face is a wall (vertical) or roof (horizontal).

    Returns: 'wall', 'roof', or 'floor'
    """
    up = Vector((0, 0, 1))
    down = Vector((0, 0, -1))

    dot_up = face_normal.dot(up)
    dot_down = face_normal.dot(down)

    threshold = 0.7

    if dot_up > threshold:
        return 'roof'
    elif dot_down > threshold:
        return 'floor'
    else:
        return 'wall'


def assign_materials_to_object(obj: bpy.types.Object, materials: dict[str, bpy.types.Material]) -> None:
    """
    Assign ITU materials to an object based on face orientations.

    Walls get itu_marble, roofs get itu_metal, floors get itu_concrete.
    """
    if obj.type != 'MESH':
        return

    mesh = obj.data

    mesh.materials.clear()
    mesh.materials.append(materials["itu_marble"])
    mesh.materials.append(materials["itu_metal"])
    mesh.materials.append(materials["itu_concrete"])

    mat_indices = {
        'wall': 0,
        'roof': 1,
        'floor': 2
    }

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.transform(obj.matrix_world)

    for face in bm.faces:
        orientation = get_face_orientation(face.normal)
        face.material_index = mat_indices.get(orientation, 0)

    bm.to_mesh(mesh)
    bm.free()


def assign_materials(materials: dict[str, bpy.types.Material]) -> None:
    """Assign ITU materials to all mesh objects based on face orientations."""
    building_count = 0

    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue

        name_lower = obj.name.lower()
        if 'building' in name_lower or 'element' in name_lower:
            assign_materials_to_object(obj, materials)
            building_count += 1

    print(f"Assigned materials to {building_count} building objects")


def is_road_object(obj: bpy.types.Object) -> bool:
    """Check if an object is a road/path based on name patterns."""
    if obj.type not in ('MESH', 'CURVE'):
        return False

    name_lower = obj.name.lower()

    # Exclude profile objects (bevel profiles used by blosm for road curves)
    if name_lower.startswith('profile_'):
        return False

    road_keywords = ['highway', 'roads_', 'paths_', 'way_', 'street', 'lane', 'track', 'footway', 'cycleway']

    for keyword in road_keywords:
        if keyword in name_lower:
            return True
    return False


def convert_road_curves_to_meshes() -> int:
    """
    Convert road/path curve objects to meshes for Sionna compatibility.

    Returns the number of curves converted.
    """
    converted_count = 0

    # Store curves to convert (can't modify collection while iterating)
    curves_to_convert = []
    for obj in bpy.data.objects:
        if obj.type == 'CURVE' and is_road_object(obj):
            curves_to_convert.append(obj)

    for obj in curves_to_convert:
        # Select only this object
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        # Convert curve to mesh using context override (needed for background mode)
        try:
            with bpy.context.temp_override(object=obj, active_object=obj):
                bpy.ops.object.convert(target='MESH')
            converted_count += 1
        except RuntimeError as e:
            print(f"  Warning: Could not convert {obj.name}: {e}")

    return converted_count


def process_road_meshes(materials: dict[str, bpy.types.Material]) -> int:
    """
    Process road mesh objects: assign material and elevate slightly.

    Returns the number of road objects processed.
    """
    processed_count = 0
    concrete = materials.get("itu_concrete")

    if not concrete:
        print("Warning: itu_concrete material not found for roads")
        return 0

    for obj in bpy.data.objects:
        if obj.type == 'MESH' and is_road_object(obj):
            # Assign itu_concrete material
            obj.data.materials.clear()
            obj.data.materials.append(concrete)

            # Elevate slightly above ground to prevent z-fighting
            obj.location.z += 0.01

            processed_count += 1

    return processed_count


def calculate_scene_bounds() -> tuple[float, float, float, float]:
    """Calculate the bounding box of all mesh objects in the scene."""
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')

    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue

        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            min_x = min(min_x, world_corner.x)
            max_x = max(max_x, world_corner.x)
            min_y = min(min_y, world_corner.y)
            max_y = max(max_y, world_corner.y)

    return min_x, max_x, min_y, max_y


def create_ground_plane(materials: dict[str, bpy.types.Material]) -> None:
    """Create a ground plane that covers all imported buildings."""
    min_x, max_x, min_y, max_y = calculate_scene_bounds()

    if min_x == float('inf'):
        print("Warning: No mesh objects found, creating default ground plane")
        min_x, max_x, min_y, max_y = -100, 100, -100, 100

    padding = 50
    min_x -= padding
    max_x += padding
    min_y -= padding
    max_y += padding

    width = max_x - min_x
    height = max_y - min_y
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    bpy.ops.mesh.primitive_plane_add(
        size=1,
        location=(center_x, center_y, 0)
    )

    plane = bpy.context.object
    plane.name = "Ground"
    plane.scale = (width, height, 1)

    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    if "itu_concrete" in materials:
        plane.data.materials.clear()
        plane.data.materials.append(materials["itu_concrete"])

    print(f"Created ground plane: {width:.1f}m x {height:.1f}m at ({center_x:.1f}, {center_y:.1f})")


def remove_non_mesh_objects() -> int:
    """
    Remove all non-mesh objects from the scene.

    Sionna only supports triangle meshes, so curves, empties, etc. must be removed.
    Returns the number of objects removed.
    """
    objects_to_remove = []

    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            objects_to_remove.append(obj)

    for obj in objects_to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)

    return len(objects_to_remove)


VALID_ITU_MATERIALS = {
    "itu_concrete", "itu_brick", "itu_plasterboard", "itu_wood",
    "itu_glass", "itu_ceiling_board", "itu_chipboard", "itu_plywood",
    "itu_marble", "itu_floorboard", "itu_metal", "itu_very_dry_ground",
    "itu_medium_dry_ground", "itu_wet_ground", "vacuum"
}


def replace_non_itu_materials(itu_materials: dict[str, bpy.types.Material]) -> int:
    """
    Replace all non-ITU materials with itu_concrete.

    Sionna only recognizes specific ITU material names. Any other material
    will cause loading errors. This function replaces unknown materials
    with itu_concrete as a default.

    Returns the number of materials replaced.
    """
    replaced_count = 0
    concrete = itu_materials.get("itu_concrete")

    if not concrete:
        print("Warning: itu_concrete material not found")
        return 0

    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue

        mesh = obj.data
        materials_to_replace = []

        for i, mat_slot in enumerate(mesh.materials):
            if mat_slot is None:
                materials_to_replace.append(i)
                continue

            mat_name = mat_slot.name
            if mat_name not in VALID_ITU_MATERIALS:
                materials_to_replace.append(i)

        for i in materials_to_replace:
            old_name = mesh.materials[i].name if mesh.materials[i] else "None"
            mesh.materials[i] = concrete
            replaced_count += 1
            print(f"  Replaced material '{old_name}' with 'itu_concrete' on {obj.name}")

    return replaced_count


def create_readme(output_dir: str, scene_name: str, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> None:
    """Create a README file with the coordinates used for the scene."""
    readme_path = os.path.join(output_dir, "README.md")

    content = f"""# {scene_name}

## Coordinates

| Parameter | Value |
|-----------|-------|
| Latitude Min | {lat_min} |
| Latitude Max | {lat_max} |
| Longitude Min | {lon_min} |
| Longitude Max | {lon_max} |

## Blosm Format

```
{lon_min},{lat_min},{lon_max},{lat_max}
```

## OpenStreetMap Link

[View on OpenStreetMap](https://www.openstreetmap.org/?mlat={((lat_min + lat_max) / 2):.6f}&mlon={((lon_min + lon_max) / 2):.6f}#map=17/{((lat_min + lat_max) / 2):.6f}/{((lon_min + lon_max) / 2):.6f})

## Files

- `{scene_name}.xml` - Mitsuba scene file for Sionna
- `meshes/` - PLY mesh files (buildings and roads)
"""

    with open(readme_path, 'w') as f:
        f.write(content)

    print(f"Created README: {readme_path}")


def export_to_mitsuba(output_dir: str, scene_name: str) -> str:
    """
    Export the scene to Mitsuba XML format.

    Returns the path to the generated XML file.
    """
    os.makedirs(output_dir, exist_ok=True)

    meshes_dir = os.path.join(output_dir, "meshes")
    os.makedirs(meshes_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{scene_name}.xml")

    print(f"Exporting to: {output_path}")

    bpy.ops.export_scene.mitsuba(
        filepath=output_path,
        axis_forward='Y',
        axis_up='Z',
        ignore_background=True
    )

    print(f"Export complete: {output_path}")
    return output_path


def main() -> None:
    """Main entry point for the script."""
    print("=" * 60)
    print("OSM to Sionna Scene Converter")
    print("=" * 60)

    lat_min, lat_max, lon_min, lon_max, output_dir, scene_name = parse_arguments()

    print("\nStep 1: Clearing scene...")
    clear_scene()

    print("\nStep 2: Enabling addons...")
    enable_addons()

    print("\nStep 3: Creating ITU materials...")
    materials = create_itu_materials()

    print("\nStep 4: Importing OSM data...")
    import_osm_data(lat_min, lat_max, lon_min, lon_max)

    print("\nStep 5: Assigning materials to buildings...")
    assign_materials(materials)

    print("\nStep 6: Converting road curves to meshes...")
    curves_converted = convert_road_curves_to_meshes()
    print(f"Converted {curves_converted} road curves to meshes")

    print("\nStep 7: Processing road meshes...")
    roads_processed = process_road_meshes(materials)
    print(f"Processed {roads_processed} road meshes")

    print("\nStep 8: Creating ground plane...")
    create_ground_plane(materials)

    print("\nStep 9: Removing non-mesh objects (Sionna only supports triangle meshes)...")
    removed_count = remove_non_mesh_objects()
    print(f"Removed {removed_count} non-mesh objects (curves, empties, etc.)")

    print("\nStep 10: Replacing non-ITU materials (Sionna only recognizes ITU materials)...")
    replaced_count = replace_non_itu_materials(materials)
    print(f"Replaced {replaced_count} non-ITU material assignments")

    print("\nStep 11: Creating README with coordinates...")
    create_readme(output_dir, scene_name, lat_min, lat_max, lon_min, lon_max)

    print("\nStep 12: Exporting to Mitsuba format...")
    output_path = export_to_mitsuba(output_dir, scene_name)

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
