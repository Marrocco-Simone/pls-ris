#!/usr/bin/env python3
"""
OSM to Sionna Scene Converter

This script converts OpenStreetMap coordinates to Sionna-compatible Mitsuba XML scenes
by running Blender in headless mode with the blosm and mitsuba-blender addons.

Prerequisites:
- Blender 3.6+ or 4.2 LTS installed and accessible via command line
- Blosm addon installed in Blender (https://prochitecture.gumroad.com/l/blender-osm)
- Mitsuba-Blender addon installed in Blender (https://github.com/mitsuba-renderer/mitsuba-blender)

Usage:
    python osm_to_sionna.py --lat-min 41.89 --lat-max 41.895 --lon-min 12.49 --lon-max 12.495 --name "my_scene"

    Or from Python:
        from osm_to_sionna import osm_to_sionna
        result = osm_to_sionna(41.89, 41.895, 12.49, 12.495, "mesh_scene/custom", "my_scene")
"""

import subprocess
import os
import shutil
import webbrowser
from pathlib import Path
from typing import Optional, Tuple


BLOSM_EXTENT_URL = "https://prochitecture.com/blender-osm/extent/?blender_version=3.6&addon=blosm&addon_version=2.7.20"


def parse_blosm_coordinates(coord_string: str) -> Tuple[float, float, float, float]:
    """
    Parse coordinates from blosm format (lon_min,lat_min,lon_max,lat_max).

    Args:
        coord_string: Comma-separated coordinates from blosm website
                     Format: "lon_min,lat_min,lon_max,lat_max"
                     Example: "11.12006,46.06603,11.12419,46.06868"

    Returns:
        Tuple of (lat_min, lat_max, lon_min, lon_max)

    Raises:
        ValueError: If the format is invalid
    """
    parts = coord_string.strip().split(",")
    if len(parts) != 4:
        raise ValueError(
            f"Expected 4 comma-separated values, got {len(parts)}. "
            f"Format should be: lon_min,lat_min,lon_max,lat_max"
        )

    try:
        lon_min = float(parts[0])
        lat_min = float(parts[1])
        lon_max = float(parts[2])
        lat_max = float(parts[3])
    except ValueError as e:
        raise ValueError(f"Could not parse coordinates as numbers: {e}")

    return lat_min, lat_max, lon_min, lon_max


def get_coordinates_interactively() -> Tuple[float, float, float, float]:
    """
    Open the blosm extent website and prompt user to paste coordinates.

    Returns:
        Tuple of (lat_min, lat_max, lon_min, lon_max)
    """
    print("\n" + "=" * 60)
    print("No coordinates provided. Opening blosm extent selector...")
    print("=" * 60)
    print(f"\nOpening: {BLOSM_EXTENT_URL}")
    print("\nInstructions:")
    print("  1. Use the map to select your desired area")
    print("  2. Copy the coordinates shown (format: lon_min,lat_min,lon_max,lat_max)")
    print("  3. Paste them below")
    print()

    webbrowser.open(BLOSM_EXTENT_URL)

    while True:
        coord_input = input("Paste coordinates here: ").strip()

        if not coord_input:
            print("No input provided. Please paste the coordinates.")
            continue

        try:
            return parse_blosm_coordinates(coord_input)
        except ValueError as e:
            print(f"Error: {e}")
            print("Please try again.")
            continue


def find_blender() -> Optional[str]:
    """
    Attempt to find the Blender executable.

    Returns the path to Blender if found, None otherwise.
    """
    blender_names = ["blender", "Blender", "blender.exe"]

    for name in blender_names:
        path = shutil.which(name)
        if path:
            return path

    common_paths = [
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        "C:\\Program Files\\Blender Foundation\\Blender\\blender.exe",
        "C:\\Program Files\\Blender Foundation\\Blender 3.6\\blender.exe",
        "C:\\Program Files\\Blender Foundation\\Blender 4.2\\blender.exe",
    ]

    for path in common_paths:
        if os.path.exists(path):
            return path

    return None


def osm_to_sionna(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    output_dir: str,
    scene_name: str,
    blender_path: Optional[str] = None,
    verbose: bool = True
) -> str:
    """
    Convert OpenStreetMap coordinates to a Sionna-compatible Mitsuba scene.

    Args:
        lat_min: Minimum latitude of the bounding box
        lat_max: Maximum latitude of the bounding box
        lon_min: Minimum longitude of the bounding box
        lon_max: Maximum longitude of the bounding box
        output_dir: Directory to save the output files
        scene_name: Name for the scene (used for XML filename)
        blender_path: Path to Blender executable (auto-detected if None)
        verbose: Print progress messages

    Returns:
        Path to the generated XML file

    Raises:
        FileNotFoundError: If Blender executable not found
        RuntimeError: If Blender script execution fails
        ValueError: If coordinates are invalid
    """
    if lat_min >= lat_max:
        raise ValueError(f"lat_min ({lat_min}) must be less than lat_max ({lat_max})")
    if lon_min >= lon_max:
        raise ValueError(f"lon_min ({lon_min}) must be less than lon_max ({lon_max})")

    if blender_path is None:
        blender_path = find_blender()
        if blender_path is None:
            raise FileNotFoundError(
                "Blender executable not found. Please specify the path using --blender argument "
                "or ensure Blender is in your PATH."
            )

    if not os.path.exists(blender_path):
        raise FileNotFoundError(f"Blender not found at: {blender_path}")

    script_path = Path(__file__).parent / "blender_script.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Blender script not found at: {script_path}")

    scene_dir = os.path.abspath(os.path.join(output_dir, scene_name))
    os.makedirs(scene_dir, exist_ok=True)

    cmd = [
        blender_path,
        "--background",
        "--python", str(script_path),
        "--",
        str(lat_min),
        str(lat_max),
        str(lon_min),
        str(lon_max),
        scene_dir,
        scene_name
    ]

    if verbose:
        print(f"Running Blender: {' '.join(cmd)}")
        print("-" * 60)

    result = subprocess.run(
        cmd,
        capture_output=not verbose,
        text=True
    )

    if result.returncode != 0:
        error_msg = result.stderr if result.stderr else "Unknown error"
        raise RuntimeError(f"Blender execution failed:\n{error_msg}")

    output_path = os.path.join(scene_dir, f"{scene_name}.xml")

    if not os.path.exists(output_path):
        raise RuntimeError(
            f"Expected output file not found: {output_path}\n"
            "The Blender script may have encountered an error."
        )

    return output_path


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert OpenStreetMap data to Sionna-compatible Mitsuba scene",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode: opens browser to select area
  python osm_to_sionna.py --name "my_scene"

  # Use coordinates from blosm directly
  python osm_to_sionna.py --coords "11.12006,46.06603,11.12419,46.06868" --name "trento_test"

  # Convert a small area in Rome (explicit coordinates)
  python osm_to_sionna.py --lat-min 41.8900 --lat-max 41.8950 \\
      --lon-min 12.4900 --lon-max 12.4950 --name "rome_test"

  # Specify custom output directory and Blender path
  python osm_to_sionna.py --lat-min 41.8900 --lat-max 41.8950 \\
      --lon-min 12.4900 --lon-max 12.4950 --name "rome_test" \\
      --output-dir mesh_scene/custom/rome \\
      --blender /Applications/Blender.app/Contents/MacOS/Blender
        """
    )

    parser.add_argument(
        "--coords",
        type=str,
        default=None,
        help="Coordinates from blosm website (format: lon_min,lat_min,lon_max,lat_max)"
    )
    parser.add_argument(
        "--lat-min",
        type=float,
        default=None,
        help="Minimum latitude of the bounding box"
    )
    parser.add_argument(
        "--lat-max",
        type=float,
        default=None,
        help="Maximum latitude of the bounding box"
    )
    parser.add_argument(
        "--lon-min",
        type=float,
        default=None,
        help="Minimum longitude of the bounding box"
    )
    parser.add_argument(
        "--lon-max",
        type=float,
        default=None,
        help="Maximum longitude of the bounding box"
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name for the scene (used for XML filename)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="mesh_scene/custom",
        help="Directory to save output files (default: mesh_scene/custom)"
    )
    parser.add_argument(
        "--blender",
        type=str,
        default=None,
        help="Path to Blender executable (auto-detected if not specified)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    try:
        if args.coords:
            lat_min, lat_max, lon_min, lon_max = parse_blosm_coordinates(args.coords)
        elif all(v is not None for v in [args.lat_min, args.lat_max, args.lon_min, args.lon_max]):
            lat_min = args.lat_min
            lat_max = args.lat_max
            lon_min = args.lon_min
            lon_max = args.lon_max
        elif any(v is not None for v in [args.lat_min, args.lat_max, args.lon_min, args.lon_max]):
            missing = []
            if args.lat_min is None:
                missing.append("--lat-min")
            if args.lat_max is None:
                missing.append("--lat-max")
            if args.lon_min is None:
                missing.append("--lon-min")
            if args.lon_max is None:
                missing.append("--lon-max")
            print(f"Error: Missing coordinate arguments: {', '.join(missing)}")
            print("Provide all four coordinates or use --coords or interactive mode.")
            exit(1)
        else:
            lat_min, lat_max, lon_min, lon_max = get_coordinates_interactively()

        output_path = osm_to_sionna(
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            output_dir=args.output_dir,
            scene_name=args.name,
            blender_path=args.blender,
            verbose=not args.quiet
        )

        print("\n" + "=" * 60)
        print("SUCCESS!")
        print(f"Generated scene: {output_path}")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except ValueError as e:
        print(f"Invalid input: {e}")
        exit(1)
    except RuntimeError as e:
        print(f"Execution failed: {e}")
        exit(1)
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        exit(1)


if __name__ == "__main__":
    main()
