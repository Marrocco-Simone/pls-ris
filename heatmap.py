import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
# * pip install PyQt6
import PyQt6
from typing import List, Tuple, Callable, Literal
import os
import json
from tqdm import tqdm
import time
from diagonalization import (
  calculate_ris_reflection_matrice,
  unify_ris_reflection_matrices,
  random_reflection_vector,
  verify_matrix_is_diagonal
)
from secrecy import (
    snr_db_to_sigma_sq,
    create_random_noise_vector
)
from ber import (
    simulate_ssk_transmission_reflection,
    simulate_ssk_transmission_direct
)

num_symbols=1000

class HeatmapGenerator:
    def __init__(self, width: int, height: int, resolution: float = 0.5):
        """
        Initialize the heatmap generator with given dimensions.
        Args:
            width: Width of the area in meters
            height: Height of the area in meters
            resolution: Resolution in meters per grid cell (e.g., 0.5 for half-meter resolution)
        """
        self.width = width
        self.height = height
        self.resolution = resolution

        # * Calculate grid dimensions based on resolution
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        self.grid = np.zeros((self.grid_height, self.grid_width))
        self.buildings = []
        # * Dictionary to store points with their labels and coordinates
        self.points = {}

    @staticmethod
    def copy_from(other: 'HeatmapGenerator') -> 'HeatmapGenerator':
        """
        Copy the grid and points from another HeatmapGenerator object.

        Args:
            other: Another HeatmapGenerator object
        """
        new_heatmap = HeatmapGenerator(other.width, other.height, other.resolution)
        new_heatmap.grid = np.copy(other.grid)
        new_heatmap.buildings = other.buildings.copy()
        new_heatmap.points = other.points.copy()
        return new_heatmap

    def _meters_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert meter coordinates to grid coordinates"""
        return (
            int(x / self.resolution),
            int(y / self.resolution)
        )

    def _grid_to_meters(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to meter coordinates"""
        return (
            grid_x * self.resolution,
            grid_y * self.resolution
        )

    def add_building(self, x: int, y: int, width: int, height: int):
        """
        Add a building to the map. Buildings are excluded from the heatmap calculation.

        Args:
            x: X coordinate of building's lower-left corner
            y: Y coordinate of building's lower-left corner
            width: Width of the building in meters
            height: Height of the building in meters
        """
        self.buildings.append((x, y, width, height))
        grid_x, grid_y = self._meters_to_grid(x, y)
        grid_width = int(width / self.resolution)
        grid_height = int(height / self.resolution)

        # * Mark building area as NaN to exclude from heatmap
        self.grid[grid_y:grid_y+grid_height, grid_x:grid_x+grid_width] = np.nan

    def add_point(self, label: str, x: float, y: float):
        """
        Add a point of interest to the map with a specific label.

        Args:
            label: Label for the point (e.g., 'A', 'B', 'Source 1')
            x: X coordinate of the point
            y: Y coordinate of the point
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"Point {label} coordinates ({x}, {y}) are outside the map boundaries")
        self.points[label] = (x, y)

    def get_point_coordinates(self, label: str) -> Tuple[float, float]:
        """
        Get the coordinates of a specific point by its label.

        Args:
            label: The label of the point
        Returns:
            Tuple of (x, y) coordinates
        """
        if label not in self.points:
            raise KeyError(f"Point {label} not found")
        return self.points[label]

    def apply_function(self, func: Callable[[int, int], float]):
        """
        Apply a custom function to calculate values for each point in the grid.
        The function should take x and y coordinates as input and return a float value.

        Args:
            func: Function that takes (x, y) coordinates and returns a value
        """
        for grid_y in tqdm(range(self.grid_height), desc="Processing rows   "):
            for grid_x in tqdm(range(self.grid_width), desc="Processing columns", leave=False):
                if not np.isnan(self.grid[grid_y, grid_x]):
                    # * Convert grid coordinates to meters for the function
                    x, y = self._grid_to_meters(grid_x, grid_y)
                    self.grid[grid_y, grid_x] = func(x, y)

    def _save_colorbar_legend(self, title: str, cmap='viridis', vmin=None, vmax=None, label='BER', orientation='horizontal'):
        """
        Save a standalone colorbar legend as a separate file.

        Args:
            title: Title for the file
            cmap: Matplotlib colormap name
            vmin: Minimum value for the color scale
            vmax: Maximum value for the color scale
            label: Label for the colorbar
            orientation: 'horizontal' or 'vertical'
        """
        legend_filename = f"./results_pdf/BER heatmap legend_{orientation}.pdf"
        if os.path.exists(legend_filename):
            # print(f"Legend file {legend_filename} already exists. Skipping creation.")
            return
        fig = plt.figure(figsize=(6, 1) if orientation == 'horizontal' else (1.5, 6))
        ax = fig.add_axes([0.1, 0.4, 0.8, 0.3] if orientation == 'horizontal' else [0.3, 0.1, 0.3, 0.8])

        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cb = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax,
            orientation=orientation,
        )

        plt.rcParams['text.usetex'] = True
        plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
        plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

        plt.savefig(legend_filename, dpi=300, format='pdf', bbox_inches='tight')
        print(f"Saved {legend_filename}")
        plt.close(fig)

    def visualize(self, title: str, cmap='viridis', show_buildings=True, show_points=True, point_color='white', vmin=None, vmax=None, log_scale=False, label='BER', show_receivers_values=False, show_heatmap=True):
        """
        Visualize the ber_heatmap with optional building outlines and points of interest.

        Args:
            cmap: Matplotlib colormap name
            show_buildings: Whether to show building outlines
            show_points: Whether to show points of interest
            point_color: Color for the points
            vmin: Minimum value for the color scale
            vmax: Maximum value for the color scale
            log_scale: Whether to use log scale for color scale
            label: Label for the color
            show_receivers_values: Whether to show values at receiver points
            show_heatmap: Whether to show the heatmap values and legend
            """
        os.makedirs("./results_pdf", exist_ok=True)
        os.makedirs("./results_data", exist_ok=True)

        data_filename = f"./results_data/{title}.npz"

        np.savez(
            data_filename,
            grid=self.grid,
            width=self.width,
            height=self.height,
            resolution=self.resolution,
            buildings=np.array(self.buildings),
            points={k: np.array(v) for k, v in self.points.items()},
            vmin=vmin,
            vmax=vmax,
            log_scale=log_scale
        )
        print(f"Saved data to {data_filename}")

        figure = plt.figure(figsize=(10, 8))

        if log_scale and show_heatmap:
            # * Add small offset to zero values before taking log
            grid_for_log = np.copy(self.grid)
            positive_values = grid_for_log[grid_for_log > 0]
    
            if len(positive_values) > 0:
                min_nonzero = np.min(positive_values)
                grid_for_log[grid_for_log == 0] = min_nonzero / num_symbols
            else:
                # If no positive values exist, set a small default value
                grid_for_log[grid_for_log == 0] = 1e-10

            masked_grid = np.ma.masked_invalid(np.log10(grid_for_log))
            title += ' (log scale)'
        else:
            masked_grid = np.ma.masked_invalid(self.grid)

        extent = [0, self.width, 0, self.height]

        if show_heatmap:
            plt.imshow(masked_grid, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, extent=extent)
            if not log_scale:
                self._save_colorbar_legend(title, cmap=cmap, vmin=vmin, vmax=vmax, label=label, orientation='horizontal')
                self._save_colorbar_legend(title, cmap=cmap, vmin=vmin, vmax=vmax, label=label, orientation='vertical')
        else:
            plt.imshow(np.ones_like(masked_grid), cmap='Greys', origin='lower', vmin=0, vmax=1, extent=extent, alpha=0.1)

        if show_buildings:
            for building in self.buildings:
                x, y, w, h = building
                x_array = [x, x+w, x+w, x, x]
                y_array = [y, y, y+h, y+h, y]
                plt.plot(x_array, y_array,'r-', linewidth=2)

        if show_points and self.points:
            c = 0.5 * self.resolution
            for label, (x, y) in self.points.items():
                if label[0] == 'R' and show_receivers_values and show_heatmap:
                    grid_x, grid_y = self._meters_to_grid(x, y)
                    value = self.grid[grid_y, grid_x]
                    label += f" ({value:.2f})"
                plt.plot(x + c, y + c, 'o', color=point_color, markersize=6)
                plt.text(x + 2 * c, y + 2 * c, label, color=point_color, fontweight=1000, fontsize=20, bbox=dict(pad=0.2, boxstyle='round',  lw=0, ec=None, fc='black', alpha=0.3))
            # * plot a line between all points if the lines is not crossing a building
            for label1, (x1, y1) in self.points.items():
                for label2, (x2, y2) in self.points.items():
                    if label1 == label2: continue
                    if label1[0] == 'R' and label2[0] == 'R': continue
                    if self._line_intersects_building(x1, y1, x2, y2): continue
                    plt.plot([x1 + c, x2 + c], [y1 + c, y2 + c], '--', alpha=0.5, color=point_color)

        plt.rcParams['text.usetex'] = True
        plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
        plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 18})
        plt.grid(True)
        # plt.title(title)
        plt.xlabel('$x$ [m]', fontsize=26)
        plt.ylabel('$y$ [m]', fontsize=26)
        plt.xticks(fontsize = 26)
        plt.yticks(fontsize = 26)
        plt.savefig(f"./results_pdf/{title}.pdf", dpi=300, format='pdf', bbox_inches='tight')
        print(f"Saved {title}.pdf")
        plt.close(figure)

    @classmethod
    def from_saved_data(cls, filename: str):
        """
        Load a heatmap from a saved data file

        Args:
            filename: Path to the saved data file

        Returns:
            HeatmapGenerator: A new HeatmapGenerator with the loaded data
        """
        data = np.load(filename, allow_pickle=True)
        width = data['width'].item()
        height = data['height'].item()
        resolution = data['resolution'].item()

        heatmap = cls(width, height, resolution)
        heatmap.grid = data['grid']
        heatmap.buildings = data['buildings'].tolist()

        # Convert points back from numpy arrays to tuples
        points_dict = data['points'].item()
        heatmap.points = {k: tuple(v) for k, v in points_dict.items()}

        return heatmap

    def _line_intersects_building(self, x1: float, y1: float, x2: float, y2: float) -> bool:
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

        for bx, by, bw, bh in self.buildings:
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

    def calculate_distance_from_point(self, point: str) -> np.ndarray:
        """
        Calculate the minimum distance from each grid cell to the specified point.

        Args:
            point: point labels to consider
        Returns:
            Grid of distances
        """
        distances = np.full_like(self.grid, np.inf)
        px, py = self.points[point]

        for grid_y in range(self.grid_height):
            for grid_x in range(self.grid_width):
                if np.isnan(self.grid[grid_y, grid_x]):
                    continue

                x, y = self._grid_to_meters(grid_x, grid_y)
                if self._line_intersects_building(x, y, px, py):
                    continue

                distance = np.sqrt((x - px)**2 + (y - py)**2)
                distances[grid_y, grid_x] = distance

        return distances

    @staticmethod
    def visualize_distance_matrix(title: str, distances: np.ndarray, cmap='viridis'):
        height, width = distances.shape
        heatmap = HeatmapGenerator(width, height)
        heatmap.grid = distances
        heatmap.visualize(title, cmap=cmap, show_buildings=False, show_points=True)

def calculate_free_space_path_loss(d: float, lam = 0.08, k = 2) -> float:
    """
    Calculate free space path loss between transmitter and receiver

    Parameters:
    -----------
    d : Distance between transmitter and receiver in meters
    lam : Wavelength of the signal (default 5G = 80mm)
    k : Exponent of path loss model (default 2)

    Returns:
    --------
    Free space path loss in dB
    """
    if d == 0: d = 0.01
    return 1 / np.sqrt((4 * np.pi / lam) ** 2 * d ** k)

def calculate_unit_spatial_signature(incidence: float, K: int, delta: float):
    """
    Calculate the unit spatial signature vector for a given angle of incidence

    Parameters:
    -----------
    incidence : Angle of incidence in radians
    K : Number of antennas
    delta : Distance between antennas in meters

    Returns:
    --------
    Unit spatial signature vector of shape (K, 1)
    """
    directional_cosine = np.cos(incidence)
    e = np.array([(1 / np.sqrt(K)) * np.exp(-1j * 2 * np.pi * (k - 1) * delta * directional_cosine) for k in range(K)])
    return e.reshape(-1, 1)

def generate_rice_matrix(L: int, K: int, nu: float, sigma = 1.0) -> np.ndarray:
    """
    Generate a Ricean channel matrix for a given number of transmit and receive antennas

    Parameters:
    -----------
    L : Number of transmit antennas
    K : Number of receive antennas
    nu : Mean magnitude of each matrix element
    sigma : Standard deviation of the complex Gaussian noise

    Returns:
    --------
    Ricean channel matrix of shape (K, L)
    """
    return np.random.normal(nu/np.sqrt(2), sigma, (K, L)) + 1j * np.random.normal(nu/np.sqrt(2), sigma, (K, L))

def generate_rice_faiding_channel(L: int, K: int, ratio: float, total_power = 1.0) -> np.ndarray:
    """
    Generate a Ricean fading channel matrix for a given number of transmit and receive antennas

    Parameters:
    -----------
    L : Number of transmit antennas
    K : Number of receive antennas
    ratio : Ratio of directed path compared to the other paths
    total_power : Total power from all paths
    """
    nu = np.sqrt(ratio * total_power / (1 + ratio))
    sigma = np.sqrt(total_power / (2 * (1 + ratio)))
    return generate_rice_matrix(L, K, nu, sigma)

def calculate_mimo_channel_gain(d: float, L: int, K: int, lam = 0.08, k = 2) -> tuple[np.ndarray, float]:
    """
    Calculate MIMO channel gains between transmitter and receiver

    Parameters:
    -----------
    d : Distance between transmitter and receiver in meters
    L : Number of transmit antennas
    K : Number of receive antennas
    lam : Wavelength of the signal (default 5G = 80mm)
    k : Exponent of path loss model (default 2)

    Returns:
    --------
    H : Complex channel gain matrix of shape (K, L)
    """
    # return generate_random_channel_matrix(K, L)
    if d == np.inf:
        return np.zeros((K, L), dtype=complex)
    if d == 0:
        d = 0.5

    delta = lam / 2
    # a = calculate_free_space_path_loss(d, lam, k)
    c = np.sqrt(L * K) * np.exp(-1j * 2 * np.pi * d / lam)
    # c = a * c
    e_r = calculate_unit_spatial_signature(0, K, delta)
    e_t = calculate_unit_spatial_signature(0, L, delta)
    H = c * (e_r @ e_t.T.conj())

    ratio = 0.6
    total_power = 1.0
    H = H * generate_rice_faiding_channel(L, K, ratio, total_power)
    return H

def calculate_channel_power(H: np.ndarray) -> float:
    '''
    Calculate the channel power of a given channel matrix H

    Parameters:
    -----------
    H : Complex channel matrix of shape (K, L)

    Returns:
    --------
    Channel power
    '''
    # return np.linalg.norm(H) ** 2
    columns, rows = H.shape
    power = 0
    for i in range(columns):
        h_i = H[i, :]
        power += np.linalg.norm(h_i) ** 2
    return power / columns

def print_low_array(v: np.ndarray) -> str:
    return print(np.array2string(np.abs(v), formatter={'float_kind':lambda x: '{:.1e}'.format(x)}))

PATH_LOSS_TYPES = (
    # 'sum', 
    'product', 
    'active_ris'
)
def ber_heatmap_reflection_simulation(
    width: int,
    height: int,
    buildings: List[Tuple[int, int, int, int]],
    transmitter: Tuple[int, int],
    ris_points: List[Tuple[int, int]],
    receivers: List[Tuple[int, int]],
    num_symbols: int,
    N: int = 16,
    K: int = 2,
    eta: float = 0.9,
    snr_db: int = 10,
    path_loss_calculation_type: Literal['sum', 'product', 'active_ris'] = 'sum',
    force_recompute: bool = False,
    n_colors: int = 256
):
    """
    Run RIS reflection simulation with given parameters

    Args:
        width: Width of the simulation area
        height: Height of the simulation area
        buildings: List of (x, y, w, h) tuples for buildings
        transmitter: (x, y) coordinates of transmitter
        ris_points: List of (x, y) coordinates for RIS points
        receivers: List of (x, y) coordinates for receivers
        N: Number of reflecting elements per RIS
        K: Number of antennas
        eta: Reflection efficiency
        snr_db: Signal-to-noise ratio in dB
        num_symbols: Number of symbols to simulate
        path_loss_calculation_type: Type of path loss calculation.
        - 'sum' means summing all distances to calculate one single path loss;
        - 'product' means multiplying all path losses of each distances;
        - 'active_ris' means only the last RIS distance is considered
        force_recompute: If True, recompute even if data exists
        n_colors: number of colors to be used for the heatmap. By default it is set to 255, so continuous. Set it to 5
        to discretize the BER into 5 stops (0, 0.1, 0.2, 0.3, 0.4, 0.5)
    """
    print(f"Called function with num_symbols = {num_symbols}")
    M = len(ris_points)
    title = f'{M} RIS(s) (K = {K}, SNR = {snr_db}) [Path Loss: {path_loss_calculation_type}]'
    data_filename = f"./results_data/{title} BER Heatmap.npz"
    print(f"filename {data_filename} exist: {os.path.exists(data_filename)}")

    # Check if data already exists
    if not force_recompute and os.path.exists(data_filename):
        print(f"Data file {data_filename} already exists. Loading...")
        ber_heatmap = HeatmapGenerator.from_saved_data(data_filename)
        viridis = matplotlib.colormaps['viridis']
        cmap = matplotlib.colors.ListedColormap([viridis(x) for x in np.linspace(0, 1, n_colors)])
        ber_heatmap.visualize(title + ' BER Heatmap', cmap=cmap, vmin=0.0, vmax=0.5, label='BER', show_receivers_values=True)
        # ber_heatmap.visualize(title + ' BER Heatmap', log_scale=True, vmin=-10.0, vmax=0.0, label='BER', show_receivers_values=True)
        return

    os.makedirs("./results_data", exist_ok=True)

    ber_heatmap = HeatmapGenerator(width, height)

    for building in buildings:
        ber_heatmap.add_building(*building)

    tx, ty = transmitter
    ber_heatmap.add_point('T', tx, ty)

    M = len(ris_points)
    for i, (px, py) in enumerate(ris_points):
        ber_heatmap.add_point(f'P{i+1}', px, py)

    J = len(receivers)
    for i, (rx, ry) in enumerate(receivers):
        ber_heatmap.add_point(f'R{i+1}', rx, ry)

    distances_from_T = ber_heatmap.calculate_distance_from_point('T')
    # HeatmapGenerator.visualize_distance_matrix('Distance from Transmitter', distances_from_T)
    distances_from_Ps = [ber_heatmap.calculate_distance_from_point(f'P{i+1}') for i in range(M)]
    # for m in range(M):
    #     HeatmapGenerator.visualize_distance_matrix(f'Distance from RIS {m+1}', distances_from_Ps[m])

    power_heatmap_from_T = HeatmapGenerator.copy_from(ber_heatmap)
    power_heatmap_from_Ps = [HeatmapGenerator.copy_from(ber_heatmap) for _ in range(M)]

    ber_heatmap.visualize(f'{M} RIS(s) (K = {K}, SNR = {snr_db})', label='', show_receivers_values=False, vmax=0.0, vmin=0.0, show_heatmap=False)

    tx_grid_y, tx_grid_x = ber_heatmap._meters_to_grid(tx, ty)
    # todo H could be multiple ones and not just transmitter to first RIS. For the multi path scenario, change this
    H = calculate_mimo_channel_gain(distances_from_Ps[0][tx_grid_y, tx_grid_x], K, N)

    if M > 1:
        receiver_grid_coords = [(ber_heatmap._meters_to_grid(rx, ry)) for rx, ry in receivers]
        Gs_per_ris = [[calculate_mimo_channel_gain(distances_from_Ps[i][ry, rx], N, K)
              for rx, ry in receiver_grid_coords] for i in range(M)]

        ris_grid_coords = [ber_heatmap._meters_to_grid(px, py) for px, py in ris_points]
        Cs = [calculate_mimo_channel_gain(
            distances_from_Ps[i+1][ris_grid_coords[i][1], ris_grid_coords[i][0]],
            N, N
        ) for i in range(M-1)]
    else:
        receiver_grid_coords = [(ber_heatmap._meters_to_grid(rx, ry)) for rx, ry in receivers]
        Gs_per_ris = [[calculate_mimo_channel_gain(distances_from_Ps[0][ry, rx], N, K)
              for rx, ry in receiver_grid_coords]]
        Cs = []

    # * Calculate cumulative path distances
    ris_path_distances = []
    for i in range(M):
        if i == 0:
            # * Distance from T to first RIS
            ris_path_distances.append(distances_from_Ps[0][ty, tx])
        else:
            # * Distance between consecutive RIS points
            ris_path_distances.append(
                distances_from_Ps[i][ris_points[i-1][1], ris_points[i-1][0]]
            )

    mean_power_per_receiver = np.zeros(J, dtype=float)
    def calculate_ber_per_point(x: int, y: int) -> float:
        grid_x, grid_y = ber_heatmap._meters_to_grid(x, y)
        distance_from_T = distances_from_T[grid_y, grid_x]
        B = calculate_mimo_channel_gain(distance_from_T, K, K) * calculate_free_space_path_loss(distance_from_T)
        B_power = calculate_channel_power(B)
        power_heatmap_from_T.grid[grid_y, grid_x] = B_power

        distances_from_Ps_current = [distances_from_Ps[i][grid_y, grid_x] for i in range(M)]
        if distance_from_T == np.inf and all(d == np.inf for d in distances_from_Ps_current):
            return np.nan

        Fs = [calculate_mimo_channel_gain(d, N, K) for d in distances_from_Ps_current]

        # * Override channel matrices for receiver positions, since fading is randomly calculated
        for i in range(M):
            for j in range(J):
                if x == receivers[j][0] and y == receivers[j][1]:
                    Fs[i] = Gs_per_ris[i][j]

        show_receivers_calculations_and_exit = False
        mean_power = 0.0
        errors = 0
        for _ in range(num_symbols):
            Ps: List[np.ndarray] = []
            for i in range(M):
                distance_pi_to_receivers = [
                    distances_from_Ps[i][ber_heatmap._meters_to_grid(receivers[j][1], receivers[j][0])] 
                    for j in range(J)
                ]
                if show_receivers_calculations_and_exit:
                    for j in range(J):
                        print(f"Distance from RIS {i+1} to receiver {j+1}: {distance_pi_to_receivers[j]:.2f} m")
                G_receivers_connected_to_ris_i = [
                    Gs_per_ris[i][j] for j in range(J)
                    if distance_pi_to_receivers[j] != np.inf
                ]
                if show_receivers_calculations_and_exit:
                    indexes_of_receivers_connected_to_ris_i = [
                        j for j in range(J)
                        if distance_pi_to_receivers[j] != np.inf
                    ]    
                    print(f"Using {len(G_receivers_connected_to_ris_i)} out of {len(Gs_per_ris[i])} receivers connected to RIS {i+1}")
                J_prime = len(G_receivers_connected_to_ris_i)
                if J_prime == 0:
                    P = np.diag(random_reflection_vector(N, eta))
                    Ps.append(P)
                    if show_receivers_calculations_and_exit: print(f"RIS {i+1} is not connected to any receiver, using random matrix")
                elif i == 0:
                    P, _ = calculate_ris_reflection_matrice(K, N, J_prime, G_receivers_connected_to_ris_i, H, eta)
                    Ps.append(P)
                    if show_receivers_calculations_and_exit: print(f"Set up RIS {i+1} reflection matrix using the receivers with index: {indexes_of_receivers_connected_to_ris_i}")
                else:
                    P_prev = unify_ris_reflection_matrices(Ps, Cs)
                    # you may also modify H in modified_H, if needed
                    modified_Gs = []
                    for G in G_receivers_connected_to_ris_i:
                        modified_Gs.append(G @ P_prev @ Cs[i-1])
                    P, _ = calculate_ris_reflection_matrice(K, N, J_prime, modified_Gs, H, eta)
                    Ps.append(P)
                    if show_receivers_calculations_and_exit: 
                        print(f"Set up RIS {i+1} reflection matrix using the receivers with index: {indexes_of_receivers_connected_to_ris_i}")
                        print()

            for j in range(J):
                effective_channel = np.zeros((K, K), dtype=complex)
                for i in range(M):
                    # verify if the RIS reflection matrices are diagonalizable for receivers
                    distance_ris_receiver = distances_from_Ps[i][ber_heatmap._meters_to_grid(receivers[j][1], receivers[j][0])]
                    if distance_ris_receiver == np.inf: 
                        if show_receivers_calculations_and_exit:
                            print(f"Receiver {j+1} is not connected to RIS {i+1}, skipping.")
                            # print_low_array(Gs_per_ris[i][j])
                        continue
                    if show_receivers_calculations_and_exit:
                        print(f"Receiver {j+1} connected to RIS {i+1} with distance {distance_ris_receiver:.2f} m")
                        print_low_array(Gs_per_ris[i][j])
                    if i == 0:
                        P_to_i = Ps[0]
                    else:
                        P_to_i = unify_ris_reflection_matrices(Ps[:i+1], Cs[:i])
                    effective_channel += Gs_per_ris[i][j] @ P_to_i @ H
                if show_receivers_calculations_and_exit:
                    is_diagonal = verify_matrix_is_diagonal(effective_channel)
                    print(f"Effective channel matrix for receiver {j+1} is diagonal: {is_diagonal}")
                    print_low_array(effective_channel)
                    print()
            if show_receivers_calculations_and_exit:
                exit()

            effective_channel = np.zeros((K, K), dtype=complex)

            for i in range(M):
                if i == 0:
                    P_to_i = Ps[0]
                else:
                    P_to_i = unify_ris_reflection_matrices(Ps[:i+1], Cs[:i])

                if path_loss_calculation_type == 'sum':
                    total_distance = sum(ris_path_distances[:i+1]) + distances_from_Ps_current[i]
                    total_path_loss = calculate_free_space_path_loss(total_distance)
                elif path_loss_calculation_type == 'product':
                    total_path_loss = 1
                    for j in range(i+1):
                        if ris_path_distances[j] == np.inf: continue
                        total_path_loss *= calculate_free_space_path_loss(ris_path_distances[j])
                    total_path_loss *= calculate_free_space_path_loss(distances_from_Ps_current[i])
                elif path_loss_calculation_type == 'active_ris':
                    total_path_loss = calculate_free_space_path_loss(distances_from_Ps_current[i])
                else:
                    raise ValueError(f"Invalid path loss calculation type: {path_loss_calculation_type}")
                new_effective_channel = Fs[i] @ P_to_i @ H * total_path_loss

                new_effective_channel_power = calculate_channel_power(new_effective_channel)
                power_heatmap_from_Ps[i].grid[grid_y, grid_x] += new_effective_channel_power / num_symbols # * Take the mean power

                effective_channel += new_effective_channel
            power = B_power if distance_from_T != np.inf else calculate_channel_power(effective_channel)
            mean_power += power / num_symbols
            sigma_sq = snr_db_to_sigma_sq(snr_db, power)

            if distance_from_T == np.inf:
                errors += simulate_ssk_transmission_reflection(K, effective_channel, sigma_sq)
            else:
                errors += simulate_ssk_transmission_direct(K, B, effective_channel, sigma_sq)

        for i in range(M):
            if distances_from_Ps_current[i] == np.inf:
                continue    
            for j in range(J):
                if x == receivers[j][0] and y == receivers[j][1]:
                    mean_power_per_receiver[j] += power_heatmap_from_Ps[i].grid[grid_y, grid_x]

        if mean_power == 0:
            return np.nan

        return errors / num_symbols

    ber_heatmap.apply_function(calculate_ber_per_point)
    print(f"Mean power per receiver: {[f'{power:.2e}' for power in mean_power_per_receiver]}")
    print("------")
    for j in range (J):
        print(f"\tReceiver {j+1} mean power: {mean_power_per_receiver[j]:.2e}, BER: {(ber_heatmap.grid[ber_heatmap._meters_to_grid(receivers[j][1], receivers[j][0])]*100):2f}%")
    print("------")
    title = f'{M} RIS(s) (K = {K}, SNR = {snr_db}) [Path Loss: {path_loss_calculation_type}]'
    ber_heatmap.visualize(title + ' BER Heatmap', vmin=0.0, vmax=0.5, label='BER', show_receivers_values=True)
    # ber_heatmap.visualize(title + ' BER Heatmap', log_scale=True, vmin=-10.0, vmax=0.0, label='BER', show_receivers_values=True)

    # power_heatmap_from_T.visualize(title + ' Mean Antenna Strenght from Transmitter', log_scale=True, vmin=-10.0, vmax=0.0, label='log MAS')
    # for i in range(M):
    #     power_heatmap_from_Ps[i].visualize(title + f' Mean Antenna Strenght {i+1}', log_scale=True, vmin=-10.0, vmax=0.0, label='log MAS')

def main():
    calculate_single_reflection = True
    calculate_multiple_reflection = True
    calculate_multiple_complex_reflection = True
    K=4
    N=36

    begin_time = time.perf_counter()
    
    # * One reflection simulation
    if calculate_single_reflection:
        buildings_single = [
            (0, 10, 7, 10),
            (8, 0, 12, 8)
        ]
        transmitter_single = (3, 3)
        ris_points_single = [(7, 9)]
        receivers_single = [(16, 11), (10, 18)]

        for path_loss_calculation_type in PATH_LOSS_TYPES:
            start_time = time.perf_counter()
            ber_heatmap_reflection_simulation(
                width=20,
                height=20,
                buildings=buildings_single,
                transmitter=transmitter_single,
                ris_points=ris_points_single,
                receivers=receivers_single,
                N=N,
                K=K,
                path_loss_calculation_type=path_loss_calculation_type,
                num_symbols=num_symbols
            )
            end_time = time.perf_counter()
            print(f"Single reflection simulation took {end_time - start_time:.2f} seconds for {num_symbols} symbols with K={K}, N={N}, path loss type: {path_loss_calculation_type}\n\n")

    # * Multiple reflection simulation
    if calculate_multiple_reflection:
        buildings_multiple = [
            (0, 10, 10, 10),
            (2, 4, 7, 1)
        ]
        transmitter_multiple = (1, 1)
        ris_points_multiple = [(0, 9), (10, 9)]
        receivers_multiple = [(16, 14), (12, 18)]

        for path_loss_calculation_type in PATH_LOSS_TYPES:
            start_time = time.perf_counter()
            ber_heatmap_reflection_simulation(
                width=20,
                height=20,
                buildings=buildings_multiple,
                transmitter=transmitter_multiple,
                ris_points=ris_points_multiple,
                receivers=receivers_multiple,
                N=N,
                K=K,
                path_loss_calculation_type=path_loss_calculation_type,
                num_symbols=num_symbols
            )
            end_time = time.perf_counter()
            print(f"Multiple reflection simulation took {end_time - start_time:.2f} seconds for {num_symbols} symbols with K={K}, N={N}, path loss type: {path_loss_calculation_type}\n\n")

    # * Multiple complex reflection simulation - one receiver gets from the middle RIS, another from the last RIS
    if calculate_multiple_complex_reflection:
        buildings_multiple = [
            (0, 10, 10, 10),
            (3, 4, 7, 1),
            (15, 10, 5, 1),
            (9, 0, 1, 8),
            (5, 7, 7, 1),
        ]
        transmitter_multiple = (1, 1)
        ris_points_multiple = [
            (0, 9), 
            (10, 9), 
            (18, 6),
        ]
        receivers_multiple = [
            (4, 5),
            (14, 16),
            (12, 18),
            (11, 3), 
            (15, 1), 
        ]

        for path_loss_calculation_type in PATH_LOSS_TYPES:
            start_time = time.perf_counter()
            ber_heatmap_reflection_simulation(
                width=20,
                height=20,
                buildings=buildings_multiple,
                transmitter=transmitter_multiple,
                ris_points=ris_points_multiple,
                receivers=receivers_multiple,
                N=N,
                K=K,
                path_loss_calculation_type=path_loss_calculation_type,
                num_symbols=num_symbols,
            )
            end_time = time.perf_counter()
            print(f"Multiple complex reflection simulation took {end_time - start_time:.2f} seconds for {num_symbols} symbols with K={K}, N={N}, path loss type: {path_loss_calculation_type}\n\n")

    end_time = time.perf_counter()
    print(f"Total time taken: {end_time - begin_time:.2f} seconds for {num_symbols} symbols with K={K}, N={N}")

if __name__ == "__main__":
    main()