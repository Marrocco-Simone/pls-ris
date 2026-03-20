import os
from heatmap_situations import situations, Building
from typing import List

def generate_xml_header() -> str:
    """Generate the XML header with integrator, materials, and emitters."""
    return '''<scene version="2.1.0">

	<!-- Defaults, these can be set via the command line: -Darg=value -->


	<!-- Camera and Rendering Parameters -->

	<integrator type="path" id="elm__0" name="elm__0">
		<integer name="max_depth" value="12" />
	</integrator>

	<!-- Materials -->

	<bsdf type="diffuse" id="mat-itu_concrete" name="mat-itu_concrete">
		<rgb value="1.000000 0.000000 0.300000" name="reflectance" />
	</bsdf>
	<bsdf type="diffuse" id="mat-itu_marble" name="mat-itu_marble">
		<rgb value="0.284233 0.146437 0.061892" name="reflectance" />
	</bsdf>
	<bsdf type="diffuse" id="mat-itu_metal" name="mat-itu_metal">
		<rgb value="0.290000 0.250000 0.210000" name="reflectance" />
	</bsdf>

	<!-- Emitters -->

	<emitter type="constant" id="World" name="World">
		<rgb value="1.000000 1.000000 1.000000" name="radiance" />
	</emitter>

	<!-- Shapes -->

'''

def generate_floor(width: int, height: int) -> str:
    """Generate the floor mesh scaled to scenario dimensions."""
    return f'''\t<!-- Floor: {width}x{height} meters -->
	<shape type="ply" id="mesh-Piano" name="mesh-Piano">
		<string name="filename" value="meshes/Piano.ply" />
		<boolean name="face_normals" value="true" />
		<ref id="mat-itu_concrete" name="bsdf" />
		<transform name="to_world">
			<scale x="{width/20}" y="{height/20}" z="1.0" />
		</transform>
	</shape>

'''

def generate_building(building_idx: int, building: Building) -> str:
    """Generate XML for a single building using Cube.ply."""
    x = building['x']
    y = building['y']
    w = building['width']
    h = building['height']

    # Calculate scale factors for 2×2×2 Blender cube
    x_scale = w / 2
    y_scale = h / 2
    z_scale = 1.5  # Fixed 3m height
    z_translate = 1.5  # Half of building height to place on floor

    return f'''\t<!-- Building {building_idx + 1}: Target dimensions {w}m (width) × {h}m (length) × 3m (height) at position ({x}, {y}, 0)
	     Blender default cube is 2×2×2 units, so we scale by ({x_scale}, {y_scale}, {z_scale})
	     - Width: 2 × {x_scale} = {w}m
	     - Length: 2 × {y_scale} = {h}m
	     - Height: 2 × {z_scale} = 3m
	     Transform order: scale first, then translate
	     Z-translation = {z_translate}m to place bottom on floor (half of 3m height) -->
	<shape type="ply" id="mesh-building{building_idx + 1}" name="mesh-building{building_idx + 1}">
		<string name="filename" value="custom/meshes/Cube.ply" />
		<boolean name="face_normals" value="true" />
		<ref id="mat-itu_marble" name="bsdf" />
		<transform name="to_world">
			<scale x="{x_scale}" y="{y_scale}" z="{z_scale}" />
			<translate x="{x}" y="{y}" z="{z_translate}" />
		</transform>
	</shape>

'''

def generate_xml_footer() -> str:
    """Generate the XML footer."""
    return '''\t<!-- Volumes -->

</scene>'''

def generate_scene_xml(simulation_name: str, width: int, height: int, buildings: List[Building]) -> str:
    """Generate complete XML scene for a scenario."""
    xml = generate_xml_header()
    xml += generate_floor(width, height)

    for idx, building in enumerate(buildings):
        xml += generate_building(idx, building)

    xml += generate_xml_footer()
    return xml

def main():
    """Generate XML scene files for all scenarios."""
    output_dir = "mesh_scene"
    os.makedirs(output_dir, exist_ok=True)

    print("Generating XML scene files...")
    print("=" * 60)

    for situation in situations:
        simulation_name = situation['simulation_name']

        # Generate XML (force overwrite)
        output_path = os.path.join(output_dir, f"{simulation_name}.xml")

        # Generate XML
        xml_content = generate_scene_xml(
            simulation_name,
            situation['width'],
            situation['height'],
            situation['buildings']
        )

        # Write to file
        with open(output_path, 'w') as f:
            f.write(xml_content)

        print(f"✓ Generated {simulation_name}.xml")
        print(f"  - Dimensions: {situation['width']}m × {situation['height']}m")
        print(f"  - Buildings: {len(situation['buildings'])}")

    print("=" * 60)
    print("Scene generation complete!")

if __name__ == "__main__":
    main()
