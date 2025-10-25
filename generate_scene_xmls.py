import os
from heatmap_situations import situations, Building
from typing import List

def generate_xml_header() -> str:
    """Generate the XML header with integrator, materials, and emitters."""
    return '''<scene version="2.1.0">

<!-- Defaults, these can be set via the command line: -Darg=value -->


<!-- Camera and Rendering Parameters -->

	<integrator type="path" id="elm__0" name="elm__0">
		<integer name="max_depth" value="12"/>
	</integrator>

<!-- Materials -->

	<bsdf type="diffuse" id="mat-itu_concrete" name="mat-itu_concrete">
		<rgb value="1.000000 0.000000 0.300000" name="reflectance"/>
	</bsdf>
	<bsdf type="diffuse" id="mat-itu_marble" name="mat-itu_marble">
		<rgb value="0.284233 0.146437 0.061892" name="reflectance"/>
	</bsdf>
	<bsdf type="diffuse" id="mat-itu_metal" name="mat-itu_metal">
		<rgb value="0.290000 0.250000 0.210000" name="reflectance"/>
	</bsdf>

<!-- Emitters -->

	<emitter type="constant" id="World" name="World">
		<rgb value="1.000000 1.000000 1.000000" name="radiance"/>
	</emitter>

<!-- Shapes -->

'''

def generate_floor(width: int, height: int) -> str:
    """Generate the floor mesh scaled to scenario dimensions."""
    return f'''\t<!-- Floor: {width}x{height} meters -->
	<shape type="ply" id="mesh-Piano" name="mesh-Piano">
		<string name="filename" value="meshes/Piano.ply"/>
		<boolean name="face_normals" value="true"/>
		<ref id="mat-itu_concrete" name="bsdf"/>
		<transform name="to_world">
			<scale x="{width/2}" y="{height/2}" z="1"/>
		</transform>
	</shape>

'''

def generate_building(building_idx: int, building: Building) -> str:
    """Generate XML for a single building (both marble and metal layers)."""
    x = building['x']
    y = building['y']
    w = building['width']
    h = building['height']

    return f'''\t<!-- Building {building_idx + 1}: x={x}, y={y}, width={w}, height={h} -->
	<!-- Using elements to create a {w}x{h} meter building -->
	<shape type="ply" id="mesh-building{building_idx + 1}-1" name="mesh-building{building_idx + 1}-1">
		<string name="filename" value="meshes/element-itu_marble.ply"/>
		<boolean name="face_normals" value="true"/>
		<ref id="mat-itu_marble" name="bsdf"/>
		<transform name="to_world">
			<translate x="{x}" y="{y}" z="0"/>
			<scale x="{w}" y="{h}" z="3"/>
		</transform>
	</shape>
	<shape type="ply" id="mesh-building{building_idx + 1}-2" name="mesh-building{building_idx + 1}-2">
		<string name="filename" value="meshes/element-itu_metal.ply"/>
		<boolean name="face_normals" value="true"/>
		<ref id="mat-itu_metal" name="bsdf"/>
		<transform name="to_world">
			<translate x="{x}" y="{y}" z="0"/>
			<scale x="{w}" y="{h}" z="3"/>
		</transform>
	</shape>

'''

def generate_xml_footer() -> str:
    """Generate the XML footer."""
    return '''<!-- Volumes -->

</scene>
'''

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

        # Skip if XML already exists
        output_path = os.path.join(output_dir, f"{simulation_name}.xml")
        if os.path.exists(output_path):
            print(f"✓ {simulation_name}.xml (already exists)")
            continue

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
