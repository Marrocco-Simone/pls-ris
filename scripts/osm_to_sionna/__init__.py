"""
OSM to Sionna Scene Converter

This module provides tools to convert OpenStreetMap data to Sionna-compatible
Mitsuba XML scene files using Blender's headless mode.

Usage:
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

Prerequisites:
    - Blender 3.6+ or 4.2 LTS
    - Blosm addon installed in Blender
    - Mitsuba-Blender addon installed in Blender
"""

from .osm_to_sionna import (
    osm_to_sionna,
    find_blender,
    parse_blosm_coordinates,
    get_coordinates_interactively,
    BLOSM_EXTENT_URL,
)

__all__ = [
    "osm_to_sionna",
    "find_blender",
    "parse_blosm_coordinates",
    "get_coordinates_interactively",
    "BLOSM_EXTENT_URL",
]
__version__ = "1.0.0"
