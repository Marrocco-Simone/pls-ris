#!/usr/bin/env python3
"""
Manual heatmap patching tool for TWC2025.
Usage:
  python heatmap_patch.py inspect <type> <scenario> <path_loss> <x> <y>
  python heatmap_patch.py patch <type> <scenario> <path_loss> <x> <y> <source_x> <source_y>
  python heatmap_patch.py set <type> <scenario> <path_loss> <x> <y> <value>
  python heatmap_patch.py regenerate <type> <scenario> <path_loss>
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
from heatmap import HeatmapGenerator, Grid
import __main__
__main__.HeatmapGenerator = HeatmapGenerator
__main__.Grid = Grid
from heatmap_utils import line_intersects_building

# File mappings - updated for new heatmap folder structure
# Using: K4_N36_symbols10000_eta90_Pt0dBm (n=10000, Pt=0dBm)
FIXED_SNR_FILES = {
    'single': 'Single Reflection (K = 4, N = 36).npz',
    'series': 'RISs in series (K = 4, N = 36).npz',
    'series_final': 'RISs in series, only final (K = 4, N = 36).npz',
    'parallel': 'RISs in parallel (K = 4, N = 36).npz',
}

NOISE_FLOOR_FILES = {
    'single': 'Single Reflection (K = 4, N = 36).npz',
    'series': 'RISs in series (K = 4, N = 36).npz',
    'series_final': 'RISs in series, only final (K = 4, N = 36).npz',
    'parallel': 'RISs in parallel (K = 4, N = 36).npz',
}

PATH_LOSS_MAP = {
    'product': 'product',
    'active': 'active_ris',
}


def get_file_path(type_name: str, scenario: str, path_loss: str) -> str:
    """Get the full path to the npz file."""
    if type_name == 'fixed_snr':
        if scenario not in FIXED_SNR_FILES:
            raise ValueError(f"Unknown fixed_snr scenario: {scenario}")
        filename = FIXED_SNR_FILES[scenario]
        return f"../heatmap/K4_N36_symbols10000_eta90_Pt0dBm_fixed_snr10dB/data/{filename}"
    elif type_name == 'noise_floor':
        if scenario not in NOISE_FLOOR_FILES:
            raise ValueError(f"Unknown noise_floor scenario: {scenario}")
        filename = NOISE_FLOOR_FILES[scenario]
        return f"../heatmap/K4_N36_symbols10000_eta90_Pt0dBm_noise_floor/data/{filename}"
    else:
        raise ValueError(f"Unknown type: {type_name}")


def get_neighbors(grid: np.ndarray, x: int, y: int) -> dict:
    """Get values of 4-connected neighbors."""
    height, width = grid.shape
    neighbors = {}

    if y > 0:
        neighbors['above'] = grid[y+1, x]
    if y < height - 1:
        neighbors['below'] = grid[y-1, x]
    if x > 0:
        neighbors['left'] = grid[y, x-1]
    if x < width - 1:
        neighbors['right'] = grid[y, x+1]

    return neighbors


def inspect_fixed_snr(filepath: str, path_loss: str, x: int, y: int):
    """Inspect a point in a fixed SNR file (now same format as noise_floor)."""
    data = np.load(filepath, allow_pickle=True)
    obj = data['ber_heatmap'].item()

    ber_key = f"BER path loss {path_loss}"
    snr_key = f"SNR path loss {path_loss}"

    print(f"\n=== BER ===")
    if ber_key in obj.stat_grids:
        grid = obj.stat_grids[ber_key].grid
        value = grid[y, x]
        neighbors = get_neighbors(grid, x, y)
        print(f"Value at ({x}, {y}): {value:.6f}")
        print("Neighbors:")
        for direction, val in neighbors.items():
            print(f"  {direction}: {val:.6f}")
    else:
        print(f"Grid '{ber_key}' not found")

    print(f"\n=== SNR ===")
    if snr_key in obj.stat_grids:
        grid = obj.stat_grids[snr_key].grid
        value = grid[y, x]
        neighbors = get_neighbors(grid, x, y)
        print(f"Value at ({x}, {y}): {value:.6f}")
        print("Neighbors:")
        for direction, val in neighbors.items():
            print(f"  {direction}: {val:.6f}")
    else:
        print(f"Grid '{snr_key}' not found")


def inspect_noise_floor(filepath: str, path_loss: str, x: int, y: int):
    """Inspect a point in a noise floor file."""
    data = np.load(filepath, allow_pickle=True)
    obj = data['ber_heatmap'].item()

    ber_key = f"BER path loss {path_loss}"
    snr_key = f"SNR path loss {path_loss}"

    print(f"\n=== BER ===")
    if ber_key in obj.stat_grids:
        grid = obj.stat_grids[ber_key].grid
        value = grid[y, x]
        neighbors = get_neighbors(grid, x, y)
        print(f"Value at ({x}, {y}): {value:.6f}")
        print("Neighbors:")
        for direction, val in neighbors.items():
            print(f"  {direction}: {val:.6f}")
    else:
        print(f"Grid '{ber_key}' not found")

    print(f"\n=== SNR ===")
    if snr_key in obj.stat_grids:
        grid = obj.stat_grids[snr_key].grid
        value = grid[y, x]
        neighbors = get_neighbors(grid, x, y)
        print(f"Value at ({x}, {y}): {value:.6f}")
        print("Neighbors:")
        for direction, val in neighbors.items():
            print(f"  {direction}: {val:.6f}")
    else:
        print(f"Grid '{snr_key}' not found")


def patch_fixed_snr(filepath: str, path_loss: str, x: int, y: int, source_x: int, source_y: int):
    """Patch a point in a fixed SNR file (now same format as noise_floor)."""
    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = filepath.replace('.npz', f'.{timestamp}.backup.npz')
    os.system(f'cp "{filepath}" "{backup_path}"')
    print(f"Backup created: {backup_path}")

    # Load and patch
    data = np.load(filepath, allow_pickle=True)
    obj = data['ber_heatmap'].item()

    ber_key = f"BER path loss {path_loss}"
    snr_key = f"SNR path loss {path_loss}"

    # Patch BER
    if ber_key in obj.stat_grids:
        grid = obj.stat_grids[ber_key].grid
        old_value = grid[y, x]
        new_value = grid[source_y, source_x]
        grid[y, x] = new_value
        print(f"BER patched ({x}, {y}): {old_value:.6f} → {new_value:.6f}")

    # Patch SNR
    if snr_key in obj.stat_grids:
        grid = obj.stat_grids[snr_key].grid
        old_value = grid[y, x]
        new_value = grid[source_y, source_x]
        grid[y, x] = new_value
        print(f"SNR patched ({x}, {y}): {old_value:.6f} → {new_value:.6f}")

    # Save
    np.savez(filepath, ber_heatmap=obj, globals=data['globals'])


def patch_noise_floor(filepath: str, path_loss: str, x: int, y: int, source_x: int, source_y: int):
    """Patch a point in a noise floor file."""
    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = filepath.replace('.npz', f'.{timestamp}.backup.npz')
    os.system(f'cp "{filepath}" "{backup_path}"')
    print(f"Backup created: {backup_path}")

    # Load and patch
    data = np.load(filepath, allow_pickle=True)
    obj = data['ber_heatmap'].item()

    ber_key = f"BER path loss {path_loss}"
    snr_key = f"SNR path loss {path_loss}"

    # Patch BER
    if ber_key in obj.stat_grids:
        grid = obj.stat_grids[ber_key].grid
        old_value = grid[y, x]
        new_value = grid[source_y, source_x]
        grid[y, x] = new_value
        print(f"BER patched ({x}, {y}): {old_value:.6f} → {new_value:.6f}")

    # Patch SNR
    if snr_key in obj.stat_grids:
        grid = obj.stat_grids[snr_key].grid
        old_value = grid[y, x]
        new_value = grid[source_y, source_x]
        grid[y, x] = new_value
        print(f"SNR patched ({x}, {y}): {old_value:.6f} → {new_value:.6f}")

    # Save
    np.savez(filepath, ber_heatmap=obj, globals=data['globals'])


def set_fixed_snr(filepath: str, path_loss: str, x: int, y: int, value: float):
    """Set a custom value for a point in a fixed SNR file (now same format as noise_floor)."""
    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = filepath.replace('.npz', f'.{timestamp}.backup.npz')
    os.system(f'cp "{filepath}" "{backup_path}"')
    print(f"Backup created: {backup_path}")

    # Load and set
    data = np.load(filepath, allow_pickle=True)
    obj = data['ber_heatmap'].item()

    ber_key = f"BER path loss {path_loss}"
    snr_key = f"SNR path loss {path_loss}"

    # Set BER
    if ber_key in obj.stat_grids:
        grid = obj.stat_grids[ber_key].grid
        old_value = grid[y, x]
        grid[y, x] = value
        print(f"BER set ({x}, {y}): {old_value:.6f} → {value:.6f}")

    # Set SNR
    if snr_key in obj.stat_grids:
        grid = obj.stat_grids[snr_key].grid
        old_value = grid[y, x]
        grid[y, x] = value
        print(f"SNR set ({x}, {y}): {old_value:.6f} → {value:.6f}")

    # Save
    np.savez(filepath, ber_heatmap=obj, globals=data['globals'])


def set_noise_floor(filepath: str, path_loss: str, x: int, y: int, value: float):
    """Set a custom value for a point in a noise floor file."""
    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = filepath.replace('.npz', f'.{timestamp}.backup.npz')
    os.system(f'cp "{filepath}" "{backup_path}"')
    print(f"Backup created: {backup_path}")

    # Load and set
    data = np.load(filepath, allow_pickle=True)
    obj = data['ber_heatmap'].item()

    ber_key = f"BER path loss {path_loss}"
    snr_key = f"SNR path loss {path_loss}"

    # Set BER
    if ber_key in obj.stat_grids:
        grid = obj.stat_grids[ber_key].grid
        old_value = grid[y, x]
        grid[y, x] = value
        print(f"BER set ({x}, {y}): {old_value:.6f} → {value:.6f}")

    # Set SNR
    if snr_key in obj.stat_grids:
        grid = obj.stat_grids[snr_key].grid
        old_value = grid[y, x]
        grid[y, x] = value
        print(f"SNR set ({x}, {y}): {old_value:.6f} → {value:.6f}")

    # Save
    np.savez(filepath, ber_heatmap=obj, globals=data['globals'])


def configure_latex():
    try:
        plt.rcParams['text.usetex'] = True
        plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
        plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    except Exception:
        plt.rcParams['text.usetex'] = False

def line_intersects_building_old(buildings, x1: float, y1: float, x2: float, y2: float) -> bool:
        """
        Check if line between two points intersects any building.
        Uses line segment intersection algorithm.
        """
        def ccw(A: tuple, B: tuple, C: tuple) -> bool:
            """Returns True if points are counter-clockwise oriented"""
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        def intersect(A: tuple, B: tuple, C: tuple, D: tuple) -> bool:
            """Returns True if line segments AB and CD intersect"""
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        for bx, by, bw, bh in buildings:
            building_corners = [
                (bx, by), (bx + bw, by),
                (bx + bw, by + bh), (bx, by + bh)
            ]

            for i in range(4):
                if intersect(
                    (x1, y1), (x2, y2),
                    building_corners[i], building_corners[(i + 1) % 4]
                ):
                    return True
        return False


def plot_heatmap(grid, width, height, buildings, points, output_path,
                 vmin=0.0, vmax=0.5, cmap='viridis', label='BER'):
    """Plot a heatmap (for fixed SNR)."""
    configure_latex()

    resolution = width / grid.shape[1]
    fig = plt.figure(figsize=(10, 8))
    masked_grid = np.ma.masked_invalid(grid)
    extent = (0, width, 0, height)

    plt.imshow(masked_grid, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, extent=extent)

    # Buildings
    if buildings is not None and len(buildings) > 0:
        for b in buildings:
            x, y, w, h = b[0], b[1], b[2], b[3]
            x_arr = [x, x+w, x+w, x, x]
            y_arr = [y, y, y+h, y+h, y]
            plt.plot(x_arr, y_arr, 'r-', linewidth=2)

    # Points
    if points is not None:
        c = 0.5 * resolution
        point_color = 'white'
        sorted_points = sorted(points.items(), key=lambda x: x[0])
        for plabel, coords in sorted_points:
            x = coords[0]
            y = coords[1]
            display_label = plabel
            if plabel[0] == 'R':
                gx = int(x / resolution)
                gy = int(y / resolution)
                if 0 <= gy < grid.shape[0] and 0 <= gx < grid.shape[1]:
                    val = grid[gy, gx]
                    if not np.isnan(val):
                        display_label += f" ({val:.2f})"
            plt.plot(x + c, y + c, 'o', color=point_color, markersize=6)
            plt.text(x + 2*c, y + 2*c, display_label, color=point_color,
                     fontweight=1000, fontsize=20,
                     bbox=dict(pad=0.2, boxstyle='round', lw=0, ec=None, fc='black', alpha=0.3))

        # Connection lines
        for l1, p1 in sorted_points:
            for l2, p2 in sorted_points:
                if l1 >= l2:
                    continue
                if l1[0] == 'R' and l2[0] == 'R':
                    continue
                x1, y1 = p1[0], p1[1]
                x2, y2 = p2[0], p2[1]
                if line_intersects_building_old(buildings, x1, y1, x2, y2):
                    continue
                plt.plot([x1 + c, x2 + c], [y1 + c, y2 + c], '--', alpha=0.5, color=point_color)

    plt.rc('font', **{'size': 22})
    fontsize = 35
    plt.grid(True)
    plt.xlabel('$x$ [m]', fontsize=fontsize)
    plt.ylabel('$y$ [m]', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax = plt.gca()
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    yticks = ax.get_yticks()
    ax.set_yticklabels(['' if t == 0 else f'{int(t)}' for t in yticks])

    try:
        plt.savefig(output_path, dpi=300, format='pdf', bbox_inches='tight')
        print(f"Saved {output_path}")
    except RuntimeError as e:
        if "latex could not be found" in str(e):
            plt.close(fig)
            plt.rcParams['text.usetex'] = False
            plot_heatmap(grid, width, height, buildings, points, output_path, vmin, vmax, cmap, label)
            return
        raise
    plt.close(fig)


def plot_heatmap_noise_floor(grid_array, width, height, resolution, buildings, points,
                              output_path, vmin=0.0, vmax=0.5, cmap='viridis', label='BER'):
    """Plot a heatmap (for noise floor)."""
    configure_latex()

    fig = plt.figure(figsize=(10, 8))
    masked_grid = np.ma.masked_invalid(grid_array)
    extent = (0, width, 0, height)

    plt.imshow(masked_grid, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, extent=extent)

    # Buildings
    if buildings:
        for b in buildings:
            x, y, w, h = b['x'], b['y'], b['width'], b['height']
            x_arr = [x, x+w, x+w, x, x]
            y_arr = [y, y, y+h, y+h, y]
            plt.plot(x_arr, y_arr, 'r-', linewidth=2)

    # Points
    if points:
        c = 0.5 * resolution
        point_color = 'white'
        sorted_points = sorted(points.items(), key=lambda x: x[0])
        for plabel, pdata in sorted_points:
            x = pdata['x']
            y = pdata['y']
            display_label = plabel
            if plabel[0] == 'R':
                gx = int(x / resolution)
                gy = int(y / resolution)
                if 0 <= gy < grid_array.shape[0] and 0 <= gx < grid_array.shape[1]:
                    val = grid_array[gy, gx]
                    if not np.isnan(val):
                        display_label += f" ({val:.2f})"
            plt.plot(x + c, y + c, 'o', color=point_color, markersize=6)
            plt.text(x + 2*c, y + 2*c, display_label, color=point_color,
                     fontweight=1000, fontsize=20,
                     bbox=dict(pad=0.2, boxstyle='round', lw=0, ec=None, fc='black', alpha=0.3))

        for l1, p1 in sorted_points:
            for l2, p2 in sorted_points:
                if l1 >= l2:
                    continue
                if l1[0] == 'R' and l2[0] == 'R':
                    continue
                x1, y1 = p1['x'], p1['y']
                x2, y2 = p2['x'], p2['y']
                if line_intersects_building(buildings, p1, p2): 
                    continue
                plt.plot([x1 + c, x2 + c], [y1 + c, y2 + c], '--', alpha=0.5, color=point_color)

    plt.rc('font', **{'size': 22})
    fontsize = 35
    plt.grid(True)
    plt.xlabel('$x$ [m]', fontsize=fontsize)
    plt.ylabel('$y$ [m]', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax = plt.gca()
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    yticks = ax.get_yticks()
    ax.set_yticklabels(['' if t == 0 else f'{int(t)}' for t in yticks])

    try:
        plt.savefig(output_path, dpi=300, format='pdf', bbox_inches='tight')
        print(f"Saved {output_path}")
    except RuntimeError as e:
        if "latex could not be found" in str(e):
            plt.close(fig)
            plt.rcParams['text.usetex'] = False
            plot_heatmap_noise_floor(grid_array, width, height, resolution, buildings, points,
                                    output_path, vmin, vmax, cmap, label)
            return
        raise
    plt.close(fig)


def regenerate_fixed_snr(filepath: str, scenario: str, path_loss: str):
    """Regenerate PDF for a fixed SNR file (now same format as noise_floor)."""
    data = np.load(filepath, allow_pickle=True)
    obj = data['ber_heatmap'].item()

    width = obj.width
    height = obj.height
    resolution = obj.resolution
    buildings = obj.buildings
    points_meters = {plabel: {'x': pdata['x'], 'y': pdata['y']}
                     for plabel, pdata in obj.points.items()}

    scenario_name = FIXED_SNR_FILES[scenario].replace('.npz', '')
    out_base = f"../heatmap/K4_N36_symbols10000_eta90_Pt0dBm_fixed_snr10dB/pdf"

    # BER
    ber_key = f"BER path loss {path_loss}"
    if ber_key in obj.stat_grids:
        grid_array = np.array(obj.stat_grids[ber_key].grid)
        out_name = f"{scenario_name} - BER path loss {path_loss}.pdf"
        out_path = f"{out_base}/{out_name}"
        os.makedirs(out_base, exist_ok=True)
        plot_heatmap_noise_floor(grid_array, width, height, resolution, buildings, points_meters,
                                 out_path, vmin=0.0, vmax=0.5, label='BER')

    # SNR
    snr_key = f"SNR path loss {path_loss}"
    if snr_key in obj.stat_grids:
        grid_array = np.array(obj.stat_grids[snr_key].grid)
        out_name = f"{scenario_name} - SNR path loss {path_loss}.pdf"
        out_path = f"{out_base}/{out_name}"
        os.makedirs(out_base, exist_ok=True)
        plot_heatmap_noise_floor(grid_array, width, height, resolution, buildings, points_meters,
                                 out_path, vmin=-200.0, vmax=100.0, label='SNR')


def regenerate_noise_floor(filepath: str, scenario: str, path_loss: str):
    """Regenerate PDF for a noise floor file."""
    data = np.load(filepath, allow_pickle=True)
    obj = data['ber_heatmap'].item()

    width = obj.width
    height = obj.height
    resolution = obj.resolution
    buildings = obj.buildings
    points_meters = {plabel: {'x': pdata['x'], 'y': pdata['y']}
                     for plabel, pdata in obj.points.items()}

    scenario_name = NOISE_FLOOR_FILES[scenario].replace('.npz', '')
    out_base = f"../heatmap/K4_N36_symbols10000_eta90_Pt0dBm_noise_floor/pdf"

    # BER
    ber_key = f"BER path loss {path_loss}"
    if ber_key in obj.stat_grids:
        grid_array = np.array(obj.stat_grids[ber_key].grid)
        out_name = f"{scenario_name} - BER path loss {path_loss}.pdf"
        out_path = f"{out_base}/{out_name}"
        os.makedirs(out_base, exist_ok=True)
        plot_heatmap_noise_floor(grid_array, width, height, resolution, buildings, points_meters,
                                 out_path, vmin=0.0, vmax=0.5, label='BER')

    # SNR
    snr_key = f"SNR path loss {path_loss}"
    if snr_key in obj.stat_grids:
        grid_array = np.array(obj.stat_grids[snr_key].grid)
        out_name = f"{scenario_name} - SNR path loss {path_loss}.pdf"
        out_path = f"{out_base}/{out_name}"
        os.makedirs(out_base, exist_ok=True)
        plot_heatmap_noise_floor(grid_array, width, height, resolution, buildings, points_meters,
                                 out_path, vmin=-200.0, vmax=100.0, label='SNR')


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == 'inspect':
        if len(sys.argv) != 7:
            print("Usage: python heatmap_patch.py inspect <type> <scenario> <path_loss> <x> <y>")
            sys.exit(1)

        type_name = sys.argv[2]
        scenario = sys.argv[3]
        path_loss = sys.argv[4]
        x = int(sys.argv[5])
        y = int(sys.argv[6])

        filepath = get_file_path(type_name, scenario, path_loss)
        print(f"Inspecting: {filepath}")

        if type_name == 'fixed_snr':
            inspect_fixed_snr(filepath, path_loss, x, y)
        else:
            inspect_noise_floor(filepath, path_loss, x, y)

    elif command == 'patch':
        if len(sys.argv) != 9:
            print("Usage: python heatmap_patch.py patch <type> <scenario> <path_loss> <x> <y> <source_x> <source_y>")
            sys.exit(1)

        type_name = sys.argv[2]
        scenario = sys.argv[3]
        path_loss = sys.argv[4]
        x = int(sys.argv[5])
        y = int(sys.argv[6])
        source_x = int(sys.argv[7])
        source_y = int(sys.argv[8])

        filepath = get_file_path(type_name, scenario, path_loss)
        print(f"Patching: {filepath}")

        if type_name == 'fixed_snr':
            patch_fixed_snr(filepath, path_loss, x, y, source_x, source_y)
        else:
            patch_noise_floor(filepath, path_loss, x, y, source_x, source_y)

    elif command == 'set':
        if len(sys.argv) != 8:
            print("Usage: python heatmap_patch.py set <type> <scenario> <path_loss> <x> <y> <value>")
            print("  Use 'nan' for NaN values")
            sys.exit(1)

        type_name = sys.argv[2]
        scenario = sys.argv[3]
        path_loss = sys.argv[4]
        x = int(sys.argv[5])
        y = int(sys.argv[6])
        value_str = sys.argv[7]

        # Parse value (handle 'nan', 'inf', '-inf')
        if value_str.lower() == 'nan':
            value = np.nan
        elif value_str.lower() == 'inf':
            value = np.inf
        elif value_str.lower() == '-inf':
            value = -np.inf
        else:
            value = float(value_str)

        filepath = get_file_path(type_name, scenario, path_loss)
        print(f"Setting value in: {filepath}")

        if type_name == 'fixed_snr':
            set_fixed_snr(filepath, path_loss, x, y, value)
        else:
            set_noise_floor(filepath, path_loss, x, y, value)

    elif command == 'regenerate':
        if len(sys.argv) != 5:
            print("Usage: python heatmap_patch.py regenerate <type> <scenario> <path_loss>")
            sys.exit(1)

        type_name = sys.argv[2]
        scenario = sys.argv[3]
        path_loss = sys.argv[4]

        filepath = get_file_path(type_name, scenario, path_loss)
        print(f"Regenerating PDF from: {filepath}")

        if type_name == 'fixed_snr':
            regenerate_fixed_snr(filepath, scenario, path_loss)
        else:
            regenerate_noise_floor(filepath, scenario, path_loss)

    elif command == 'help' or command == '--help' or command == '-h':
        print(__doc__)
        print("\nCommands:")
        print("  inspect      - View value at coordinates and its neighbors")
        print("  patch        - Copy value from source to target coordinates")
        print("  set          - Set a custom value at coordinates")
        print("  regenerate   - Regenerate PDF for a scenario")
        print("  help         - Show this help message")
        print("\nTypes:")
        print("  fixed_snr    - Fixed SNR scenarios (single, series, series_final, parallel)")
        print("  noise_floor  - Noise floor scenarios (single, series, series_final, parallel)")
        print("\nPath loss:")
        print("  product      - Product path loss model")
        print("  active       - Active path loss model")
        print("\nExamples:")
        print("  python heatmap_patch.py inspect fixed_snr series_final active 18 8")
        print("  python heatmap_patch.py patch noise_floor series_final active 18 8 19 8")
        print("  python heatmap_patch.py set noise_floor single active 14 18 nan")
        print("  python heatmap_patch.py regenerate fixed_snr series_final active")
        print("\nNotes:")
        print("  • NaN values appear as WHITE in heatmaps (masked_invalid)")
        print("  • Colored areas = valid BER/SNR values")
        print("  • Use 'set <...> nan' to make points white (e.g., inside buildings)")


    else:
        print(f"Unknown command: {command}")
        print("Run 'python heatmap_patch.py help' for usage information")
        sys.exit(1)


if __name__ == '__main__':
    main()
