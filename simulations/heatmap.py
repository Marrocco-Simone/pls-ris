import numpy as np
import matplotlib.pyplot as plt
# * pip install PyQt6
import PyQt6
from typing import List, Tuple, Callable
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

class HeatmapGenerator:
    def __init__(self, width: int, height: int):
        """
        Initialize the heatmap generator with given dimensions.
        
        Args:
            width: Width of the area in meters
            height: Height of the area in meters
        """
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))
        self.buildings = []
        # * Dictionary to store points with their labels and coordinates
        self.points = {}

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
        # * Mark building area as NaN to exclude from heatmap
        self.grid[y:y+height, x:x+width] = np.nan

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
        for y in range(self.height):
            print(f"Processing row {y+1}/{self.height}")
            for x in range(self.width):
                if not np.isnan(self.grid[y, x]):
                    self.grid[y, x] = func(x, y)

    def visualize(self, cmap='viridis', show_buildings=True, show_points=True, point_color='red', label_offset=(0.3, 0.3)):
        """
        Visualize the heatmap with optional building outlines and points of interest.
        
        Args:
            cmap: Matplotlib colormap name
            show_buildings: Whether to show building outlines
            show_points: Whether to show points of interest
            point_color: Color for the points
            label_offset: Offset for point labels (x, y)
        """
        plt.figure(figsize=(10, 8))
        
        masked_grid = np.ma.masked_invalid(self.grid)
        
        plt.imshow(masked_grid, cmap=cmap, origin='lower', vmin=0.0, vmax=1.0)
        plt.colorbar(label='BER')
        
        if show_buildings:
            for building in self.buildings:
                x, y, w, h = building
                plt.plot([x-0.5, x+w-0.5, x+w-0.5, x-0.5, x-0.5],
                        [y-0.5, y-0.5, y+h-0.5, y+h-0.5, y-0.5],
                        'r-', linewidth=2)
        
        if show_points and self.points:
            for label, (x, y) in self.points.items():
                plt.plot(x, y, 'o', color=point_color, markersize=8)
                plt.text(x + label_offset[0], y + label_offset[1], label,
                        color=point_color, fontweight='bold')
        
        plt.grid(True)
        plt.title('Heatmap of BER of the signal from T reflected by RIS')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.show()

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
        
        for y in range(self.height):
            for x in range(self.width):
                if np.isnan(self.grid[y, x]):
                    continue

                if self._line_intersects_building(x, y, px, py):
                    continue
                    
                distance = np.sqrt((x - px)**2 + (y - py)**2)                
                distances[y, x] = distance
                
        return distances
    
    # static method that given a distance matrix, visualize a heatmap grid of the distances by creating a HeatmapGenerator object
    @staticmethod
    def visualize_distance_matrix(distances: np.ndarray, cmap='viridis'):
        height, width = distances.shape
        heatmap = HeatmapGenerator(width, height)
        heatmap.grid = distances
        heatmap.visualize(cmap=cmap, show_buildings=False, show_points=False)
    

def calculate_free_space_path_loss(d: float, lam: float, k = 2) -> float:
    """
    Calculate free space path loss between transmitter and receiver
    
    Parameters:
    -----------
    d : Distance between transmitter and receiver in meters
    lam : Wavelength of the signal
    k : Exponent of path loss model (default 2)
        
    Returns:
    --------
    Free space path loss in dB
    """
    return (4 * np.pi / lam) ** 2 * d ** k

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

def calculate_mimo_channel_gain(d: float, L: int, K: int, lam = 0.07 , k = 2) -> tuple[np.ndarray, float]:
    """
    Calculate MIMO channel gains between transmitter and receiver
    
    Parameters:
    -----------
    d : Distance between transmitter and receiver in meters
    L : Number of transmit antennas
    K : Number of receive antennas
    lam : Wavelength of the signal (default 5G = 70mm)
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
    a = 1 / np.sqrt(calculate_free_space_path_loss(d, lam))
    c = a * np.sqrt(L * K) * np.exp(-1j * 2 * np.pi * d / lam)
    e_r = calculate_unit_spatial_signature(0, K, delta)
    e_t = calculate_unit_spatial_signature(0, L, delta)
    H = c * (e_r @ e_t.T.conj())

    return H

if __name__ == "__main__":
    N = 16    # * Number of reflecting elements
    K = 2     # * Number of antennas
    J = 1     # * Number of receivers
    M = 1     # * Number of RIS surfaces
    eta = 0.9 # * Reflection efficiency

    heatmap = HeatmapGenerator(20, 20)
    
    heatmap.add_building(0, 12, 8, 8)  
    heatmap.add_building(12, 0, 8, 8)

    tx, ty = 10 , 3
    rx, ry = 15, 10
    px, py = 8, 11
    heatmap.add_point('T', tx, ty)
    heatmap.add_point('R', rx, ry)
    heatmap.add_point('P', px, py)

    distances_from_T = heatmap.calculate_distance_from_point('T')
    distances_from_P = heatmap.calculate_distance_from_point('P')

    H = calculate_mimo_channel_gain(distances_from_P[ty, tx], K, N)
    G = calculate_mimo_channel_gain(distances_from_P[ry, rx], N, K)
    print("Channel matrix from transmitter to RIS")
    print(np.round(np.abs(H), 2))
    print("Channel matrix from RIS to receiver")
    print(np.round(np.abs(G), 2))
    Ps, _ = calculate_multi_ris_reflection_matrices(
            K, N, J, M, [G], H, eta, []
        )
    P = unify_ris_reflection_matrices(Ps, [])

    diagonalization_results = verify_multi_ris_diagonalization([P], [G], H, [])
    assert all(diagonalization_results)

    snr_db = 10
    sigma_sq = snr_db_to_sigma_sq(snr_db)
    num_symbols=1000
    def calculate_ber_per_point(x: int, y: int) -> float:
        distance_from_T = distances_from_T[y, x]
        B = calculate_mimo_channel_gain(distance_from_T, K, K)

        distance_from_P = distances_from_P[y, x]
        F = G if x == rx and y == ry else calculate_mimo_channel_gain(distance_from_P, N, K)
        effective_channel = F @ P @ H
        errors = 0
        if x == rx and y == ry:
            print()
            print("----- Point R")
            assert distance_from_T == np.inf
            print("Distance from R to T is infinity")

            print(f"Distance from P: {distance_from_P}")
            print(f"Distance from T: {distance_from_T}")
            print("BER calculation for point R")
            print("Effective channel matrix GPH")
            print(np.round(np.abs(effective_channel), 10))
            print("Example of message transmission")
            print("Signal sent from T")
            signal = np.zeros(K)
            signal[np.random.randint(K)] = 1
            print(np.round(np.abs(signal), 2))
            signal = effective_channel @ signal
            print("Signal received at P without noise")
            print(np.round(np.abs(signal), 2))
            noise = create_random_noise_vector(K, sigma_sq)
            print("Noise added at T")
            print(np.round(np.abs(noise), 2))
            print("Signal received at R")
            signal = signal + noise
            print(np.round(np.abs(signal), 2))
            print("----- End Point R")
            print()
            
        for _ in range(num_symbols):
            signal = np.zeros(K)
            signal[np.random.randint(K)] = 1

            if distance_from_T == np.inf:
                if not simulate_ssk_transmission_reflection(signal, effective_channel, sigma_sq):
                    errors += 1
            else:
                if not simulate_ssk_transmission_direct(signal, B, effective_channel, sigma_sq):
                    errors += 1
        ber = errors / num_symbols
        return ber

    heatmap.apply_function(calculate_ber_per_point)
    # HeatmapGenerator.visualize_distance_matrix(distances_from_T)
    # HeatmapGenerator.visualize_distance_matrix(distances_from_P)
    heatmap.visualize()