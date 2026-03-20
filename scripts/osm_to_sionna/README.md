# OSM to Sionna Scene Converter

Convert OpenStreetMap coordinates directly to Sionna-compatible Mitsuba XML scene files using Blender's headless mode.

## Overview

This tool automates the workflow of:

1. Importing buildings from OpenStreetMap via the blosm addon
2. Assigning Sionna-compatible ITU materials (walls, roofs, ground)
3. Creating a ground plane covering the scene
4. Exporting to Mitsuba XML format with PLY meshes

## Prerequisites

### 1. Blender (3.6 LTS or 4.2 LTS recommended)

**macOS:**

```bash
brew install --cask blender
```

**Ubuntu/Debian:**

```bash
sudo apt install blender
```

**Windows:**
Download from [blender.org](https://www.blender.org/download/)

### 2. Blosm Addon (formerly blender-osm)

1. Purchase from [Gumroad](https://prochitecture.gumroad.com/l/blender-osm) (GPL license after purchase)
2. In Blender: Edit → Preferences → Add-ons → Install
3. Select the downloaded `.zip` file
4. Enable "Blosm" in the add-ons list

### 3. Mitsuba-Blender Addon

1. Download from [GitHub](https://github.com/mitsuba-renderer/mitsuba-blender/releases)
2. In Blender: Edit → Preferences → Add-ons → Install
3. Select the downloaded `.zip` file
4. Enable "Mitsuba" in the add-ons list

### 4. pyproj (for UTM coordinate conversion)

pyproj must be installed in **Blender's Python**, not your system Python:

**macOS (Blender 3.6):**

```bash
/Applications/Blender.app/Contents/Resources/3.6/python/bin/python3.10 -m pip install pyproj
```

**macOS (Blender 4.2):**

```bash
/Applications/Blender.app/Contents/Resources/4.2/python/bin/python3.11 -m pip install pyproj
```

**Linux:**

```bash
/usr/share/blender/3.6/python/bin/python3.10 -m pip install pyproj
```

**Windows:**

```bash
"C:\Program Files\Blender Foundation\Blender 3.6\3.6\python\bin\python.exe" -m pip install pyproj
```

If pip is not available, run `ensurepip` first:

```bash
<blender-python-path> -m ensurepip
<blender-python-path> -m pip install pyproj
```

## Usage

### Command Line

```bash
# Interactive mode: opens browser to select area on map
python scripts/osm_to_sionna/osm_to_sionna.py --name "my_scene"

# Use coordinates copied from blosm website directly
python scripts/osm_to_sionna/osm_to_sionna.py \
    --coords "11.12006,46.06603,11.12419,46.06868" \
    --name "trento_test"

# Explicit coordinate arguments
python scripts/osm_to_sionna/osm_to_sionna.py \
    --lat-min 41.8900 --lat-max 41.8950 \
    --lon-min 12.4900 --lon-max 12.4950 \
    --name "my_scene"

# With custom output directory
python scripts/osm_to_sionna/osm_to_sionna.py \
    --coords "11.12006,46.06603,11.12419,46.06868" \
    --name "rome_test" \
    --output-dir mesh_scene/custom/rome

# Specify Blender path (if not in PATH)
python scripts/osm_to_sionna/osm_to_sionna.py \
    --coords "11.12006,46.06603,11.12419,46.06868" \
    --name "rome_test" \
    --blender /Applications/Blender.app/Contents/MacOS/Blender
```

### Python API

```python
from scripts.osm_to_sionna import osm_to_sionna

# Convert coordinates to Sionna scene
xml_path = osm_to_sionna(
    lat_min=41.8900,
    lat_max=41.8950,
    lon_min=12.4900,
    lon_max=12.4950,
    output_dir="mesh_scene/custom/my_scene",
    scene_name="my_scene"
)

print(f"Generated: {xml_path}")
```

### Command Line Arguments

| Argument       | Required | Description                                                                |
| -------------- | -------- | -------------------------------------------------------------------------- |
| `--coords`     | No\*     | Coordinates from blosm website (format: `lon_min,lat_min,lon_max,lat_max`) |
| `--lat-min`    | No\*     | Minimum latitude of bounding box                                           |
| `--lat-max`    | No\*     | Maximum latitude of bounding box                                           |
| `--lon-min`    | No\*     | Minimum longitude of bounding box                                          |
| `--lon-max`    | No\*     | Maximum longitude of bounding box                                          |
| `--name`       | Yes      | Scene name (used for XML filename)                                         |
| `--output-dir` | No       | Output directory (default: `mesh_scene/custom`)                            |
| `--blender`    | No       | Path to Blender executable (auto-detected)                                 |
| `--quiet`      | No       | Suppress progress output                                                   |

\*Coordinates can be provided in three ways:

1. **Interactive mode** (default): If no coordinates given, opens the blosm extent selector in your browser
2. **`--coords`**: Paste coordinates directly from blosm website (format: `lon_min,lat_min,lon_max,lat_max`)
3. **Explicit arguments**: Use `--lat-min`, `--lat-max`, `--lon-min`, `--lon-max`

## How to Get Coordinates

### Option 1: Interactive Mode (Recommended)

Simply run the script with just `--name`:

```bash
python scripts/osm_to_sionna/osm_to_sionna.py --name "my_scene"
```

This will:

1. Open the blosm extent selector in your browser
2. Let you draw a rectangle on the map
3. Prompt you to paste the coordinates in the terminal

### Option 2: Blosm Website Manually

1. Go to the [blosm extent selector](https://prochitecture.com/blender-osm/extent/)
2. Draw a rectangle on the map
3. Copy the coordinates shown (format: `lon_min,lat_min,lon_max,lat_max`)
4. Use with `--coords`:
   ```bash
   python osm_to_sionna.py --coords "11.12006,46.06603,11.12419,46.06868" --name "my_scene"
   ```

### Option 3: OpenStreetMap Export

1. Go to [openstreetmap.org](https://www.openstreetmap.org)
2. Navigate to your area of interest
3. Click "Export" in the top menu
4. The bounding box coordinates are shown:
   - Top = lat_max
   - Bottom = lat_min
   - Left = lon_min
   - Right = lon_max
5. Use the explicit argument format

## Output Structure

```
output_dir/
├── scene_name.xml          # Mitsuba scene file
└── meshes/
    ├── Ground.ply          # Ground plane mesh
    ├── building_001.ply    # Building meshes
    ├── building_002.ply
    └── ...
```

## Material Mapping

The script automatically assigns ITU materials based on surface orientation:

| Surface Type          | Material       | Description           |
| --------------------- | -------------- | --------------------- |
| Vertical faces        | `itu_marble`   | Building walls        |
| Upward-facing faces   | `itu_metal`    | Roofs                 |
| Downward-facing faces | `itu_concrete` | Floors (rarely used)  |
| Ground plane          | `itu_concrete` | Street/ground surface |

These material names are recognized by Sionna RT and will use the appropriate electromagnetic properties for ray tracing.

## Integration with Sionna

After generating the scene, load it in Sionna:

```python
import sionna
from sionna.rt import load_scene

# Load the generated scene
scene = load_scene("mesh_scene/custom/my_scene/my_scene.xml")

# Preview the scene
scene.preview()

# Add transmitters, receivers, and compute channels...
```

## Troubleshooting

### "Blender executable not found"

Specify the Blender path explicitly:

```bash
python osm_to_sionna.py ... --blender /path/to/blender
```

Common paths:

- macOS: `/Applications/Blender.app/Contents/MacOS/Blender`
- Linux: `/usr/bin/blender`
- Windows: `C:\Program Files\Blender Foundation\Blender 4.2\blender.exe`

### "Could not enable blosm addon"

1. Ensure blosm is installed in Blender
2. Try running Blender GUI once and enabling the addon manually
3. Save user preferences in Blender

### "Could not enable mitsuba-blender addon"

1. Install mitsuba-blender from the official GitHub releases
2. For Blender < 3.5, run with `--python-use-system-env` flag
3. Install mitsuba Python package: `pip install mitsuba`

### No buildings imported

- Check that the coordinates cover an area with buildings in OpenStreetMap
- Verify the bounding box is not too large (keep under ~1km² for best results)
- Ensure you have internet connectivity (blosm downloads data from OSM servers)

### Export fails

- Mitsuba export may fail with very large scenes
- Try reducing the bounding box size
- Ensure output directory is writable

## Technical Details

### How It Works

1. **Blender Headless Mode**: The script runs Blender without GUI using `--background` flag
2. **Blosm Import**: Uses blosm's Python API to download and import OSM building data
3. **Material Assignment**: Analyzes face normals to classify surfaces as walls, roofs, or floors
4. **Ground Plane**: Calculates scene bounds and creates a plane with 50m padding
5. **Mitsuba Export**: Uses mitsuba-blender addon to export XML + PLY files

### Face Classification Algorithm

```
dot_product = face_normal · (0, 0, 1)

if dot_product > 0.7:
    surface = "roof"      → itu_metal
elif dot_product < -0.7:
    surface = "floor"     → itu_concrete
else:
    surface = "wall"      → itu_marble
```

## References

- [Sionna RT Introduction](https://nvlabs.github.io/sionna/rt/tutorials/Introduction.html)
- [Mitsuba Scene Format](https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html)
- [Blosm Documentation](https://github.com/vvoovv/blosm/wiki/Documentation)
- [Mitsuba-Blender GitHub](https://github.com/mitsuba-renderer/mitsuba-blender)
