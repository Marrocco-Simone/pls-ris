import numpy as np
import matplotlib.pyplot as plt
# * pip install PyQt6
import PyQt6
from typing import List, Tuple, Callable, Literal
from diagonalization import (
  generate_random_channel_matrix, 
  calculate_multi_ris_reflection_matrices, 
  unify_ris_reflection_matrices,
  verify_multi_ris_diagonalization
)
from secrecy import (
    snr_db_to_sigma_sq,
    create_random_noise_vector
)
from ber import (
    simulate_ssk_transmission_reflection,
    simulate_ssk_transmission_direct
)

num_symbols=10000

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
        for grid_y in range(self.grid_height):
            print(f"Processing row {grid_y * self.resolution}/{self.grid_height * self.resolution}")
            for grid_x in range(self.grid_width):
                if not np.isnan(self.grid[grid_y, grid_x]):
                    # * Convert grid coordinates to meters for the function
                    x, y = self._grid_to_meters(grid_x, grid_y)
                    self.grid[grid_y, grid_x] = func(x, y)

    def visualize(self, title: str, cmap='viridis', show_buildings=True, show_points=True, point_color='red', vmin=None, vmax=None, log_scale=False, label='BER', show_receivers_values=False):
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
            """
        figure = plt.figure(figsize=(10, 8))
        
        if log_scale:
            # * Add small offset to zero values before taking log
            grid_for_log = np.copy(self.grid)
            min_nonzero = np.min(grid_for_log[grid_for_log > 0])
            grid_for_log[grid_for_log == 0] = min_nonzero / num_symbols
            masked_grid = np.ma.masked_invalid(np.log10(grid_for_log))
            title += ' (log scale)'
        else:
            masked_grid = np.ma.masked_invalid(self.grid)
        
        extent = [0, self.width, 0, self.height]
        plt.imshow(masked_grid, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, extent=extent)
        plt.colorbar(label=label)
        
        if show_buildings:
            for building in self.buildings:
                x, y, w, h = building
                x_array = [x, x+w, x+w, x, x]
                y_array = [y, y, y+h, y+h, y]
                plt.plot(x_array, y_array,'r-', linewidth=2)
        
        if show_points and self.points:
            c = 0.5 * self.resolution
            for label, (x, y) in self.points.items():
                if label[0] == 'R' and show_receivers_values:
                    grid_x, grid_y = self._meters_to_grid(x, y)
                    value = self.grid[grid_y, grid_x]
                    label += f" ({value:.2f})"
                plt.plot(x + c, y + c, 'o', color=point_color, markersize=6)
                plt.text(x + 2 * c, y + 2 * c, label, color=point_color, fontweight='bold')
            # * plot a line between all points if the lines is not crossing a building
            for label1, (x1, y1) in self.points.items():
                for label2, (x2, y2) in self.points.items():
                    if label1 == label2: continue
                    if label1[0] == 'R' and label2[0] == 'R': continue
                    if self._line_intersects_building(x1, y1, x2, y2): continue
                    plt.plot([x1 + c, x2 + c], [y1 + c, y2 + c], 'k--', alpha=0.5)
        
        plt.grid(True)
        plt.title(title)
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        # plt.show()
        plt.savefig(f"./simulations/results/{title}.png", dpi=300, format='png')
        print(f"Saved {title}.png")
        plt.close(figure)

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

PATH_LOSS_TYPES = ('sum', 'product', 'active_ris')
def ber_heatmap_reflection_simulation(
    width: int,
    height: int,
    buildings: List[Tuple[int, int, int, int]],
    transmitter: Tuple[int, int],
    ris_points: List[Tuple[int, int]],
    receivers: List[Tuple[int, int]],
    N: int = 16,
    K: int = 2,
    eta: float = 0.9,
    snr_db: int = 10,
    num_symbols: int = 100,
    path_loss_calculation_type: Literal['sum', 'product', 'active_ris'] = 'sum'
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
    """
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

    tx_grid_y, tx_grid_x = ber_heatmap._meters_to_grid(tx, ty)
    H = calculate_mimo_channel_gain(distances_from_Ps[0][tx_grid_y, tx_grid_x], K, N)

    if M > 1:
        receiver_grid_coords = [(ber_heatmap._meters_to_grid(rx, ry)) for rx, ry in receivers]
        Gs = [calculate_mimo_channel_gain(distances_from_Ps[-1][ry, rx], N, K) 
              for ry, rx in receiver_grid_coords]

        ris_grid_coords = [ber_heatmap._meters_to_grid(px, py) for px, py in ris_points]
        Cs = [calculate_mimo_channel_gain(
            distances_from_Ps[i+1][ris_grid_coords[i][1], ris_grid_coords[i][0]], 
            N, N
        ) for i in range(M-1)]
    else:
        receiver_grid_coords = [(ber_heatmap._meters_to_grid(rx, ry)) for rx, ry in receivers]
        Gs = [calculate_mimo_channel_gain(distances_from_Ps[0][ry, rx], N, K) 
              for ry, rx in receiver_grid_coords]
        Cs = []

    print(f"Channel matrix from transmitter to RIS: Power {calculate_channel_power(H):.1e}")
    print(f"Channel matrix from RIS to receiver: Power {calculate_channel_power(Gs[0]):.1e}")
    
    Ps, _ = calculate_multi_ris_reflection_matrices(K, N, J, M, Gs, H, eta, Cs)
    P = unify_ris_reflection_matrices(Ps, Cs)
    print(f"Reflection matrix: Power {calculate_channel_power(P):.1e}")
    print(f"Effective channel matrix: Power {calculate_channel_power(Gs[0] @ P @ H):.1e}")
    print()

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

    def calculate_ber_per_point(x: int, y: int) -> float:
        grid_x, grid_y = ber_heatmap._meters_to_grid(x, y)
        distance_from_T = distances_from_T[grid_y, grid_x]
        B = calculate_mimo_channel_gain(distance_from_T, K, K) * calculate_free_space_path_loss(distance_from_T)
        B_power = calculate_channel_power(B)
        power_heatmap_from_T.grid[grid_y, grid_x] = B_power

        distances_from_Ps_current = [distances_from_Ps[i][grid_y, grid_x] for i in range(M)]
        
        Fs = [calculate_mimo_channel_gain(d, N, K) for d in distances_from_Ps_current]
        
        # * Override channel matrices for receiver positions
        for j in range(J):
            if x == receivers[j][0] and y == receivers[j][1]:
                Fs[-1] = Gs[j]

        errors = 0
        for _ in range(num_symbols):
            Ps, _ = calculate_multi_ris_reflection_matrices(K, N, J, M, Gs, H, eta, Cs)
            P = unify_ris_reflection_matrices(Ps, Cs)

            effective_channel = np.zeros((K, K), dtype=complex)
            
            for i in range(M):
                if i == 0:
                    P_to_i = Ps[0]
                else:
                    P_to_i = unify_ris_reflection_matrices(Ps[:i+1], Cs[:i])
                
                if path_loss_calculation_type == 'sum':
                    total_distance = sum(ris_path_distances[:i+1]) + distances_from_Ps_current[i]
                    total_path_loss = calculate_free_space_path_loss(total_distance)
                    new_effective_channel = Fs[i] @ P_to_i @ H * total_path_loss
                elif path_loss_calculation_type == 'product':
                    total_path_loss = 1
                    for j in range(i+1):
                        total_path_loss *= calculate_free_space_path_loss(ris_path_distances[j])
                    total_path_loss *= calculate_free_space_path_loss(distances_from_Ps_current[i])
                    new_effective_channel = Fs[i] @ P_to_i @ H * total_path_loss
                elif path_loss_calculation_type == 'active_ris':
                    total_path_loss = calculate_free_space_path_loss(distances_from_Ps_current[i])
                    new_effective_channel = Fs[i] @ P_to_i @ H * total_path_loss 
                else: 
                    raise ValueError(f"Invalid path loss calculation type: {path_loss_calculation_type}")   

                new_effective_channel_power = calculate_channel_power(new_effective_channel)
                power_heatmap_from_Ps[i].grid[grid_y, grid_x] += new_effective_channel_power / num_symbols # * Take the mean power

                effective_channel += new_effective_channel
            power = B_power if distance_from_T != np.inf else calculate_channel_power(effective_channel) 
            sigma_sq = snr_db_to_sigma_sq(snr_db, power)
            
            if distance_from_T == np.inf:
                errors += simulate_ssk_transmission_reflection(K, effective_channel, sigma_sq)
            else:
                errors += simulate_ssk_transmission_direct(K, B, effective_channel, sigma_sq)
                
        return errors / num_symbols

    ber_heatmap.apply_function(calculate_ber_per_point)
    title = f'BER Heatmap with {M} RIS(s) (K = {K}, SNR = {snr_db} dB) [Path Loss: {path_loss_calculation_type}]'
    ber_heatmap.visualize(title, vmin=0.0, vmax=1.0, label='BER', show_receivers_values=True)
    ber_heatmap.visualize(title, log_scale=True, vmin=-10.0, vmax=0.0, label='BER', show_receivers_values=True)

    power_heatmap_from_T.visualize(title + ' Channel Power from Transmitter', log_scale=True, vmin=-10.0, vmax=0.0, label='Power (dB)')
    for i in range(M):
        power_heatmap_from_Ps[i].visualize(title + f' Channel Power from RIS {i+1}', log_scale=True, vmin=-10.0, vmax=0.0, label='Power (dB)')
    print('\n')

def main():
    # * One reflection simulation
    buildings_single = [
        (0, 10, 7, 10),
        (8, 0, 12, 8)
    ]
    transmitter_single = (3, 3)
    ris_points_single = [(7, 9)]
    receivers_single = [(16, 11), (10, 18)]
    
    for path_loss_calculation_type in PATH_LOSS_TYPES:
        ber_heatmap_reflection_simulation(
            width=20,
            height=20,
            buildings=buildings_single,
            transmitter=transmitter_single,
            ris_points=ris_points_single,
            receivers=receivers_single,
            N=25,
            K=4,
            path_loss_calculation_type=path_loss_calculation_type
        )

    # * Multiple reflection simulation
    buildings_multiple = [
        (0, 10, 10, 10),
        (2, 4, 7, 1)
    ]
    transmitter_multiple = (1, 1)
    ris_points_multiple = [(0, 9), (10, 9)]
    receivers_multiple = [(16, 14), (12, 18)]
    
    for path_loss_calculation_type in PATH_LOSS_TYPES:
        ber_heatmap_reflection_simulation(
            width=20,
            height=20,
            buildings=buildings_multiple,
            transmitter=transmitter_multiple,
            ris_points=ris_points_multiple,
            receivers=receivers_multiple,
            N=16,
            K=2,
            path_loss_calculation_type=path_loss_calculation_type
        )

if __name__ == "__main__":
    main()