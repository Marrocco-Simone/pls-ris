import numpy as np
import matplotlib.pyplot as plt
# * pip install PyQt6
import PyQt6
from typing import List, Tuple, Callable
from diagonalization import (
  generate_random_channel_matrix, 
  calculate_multi_ris_reflection_matrices, 
  unify_ris_reflection_matrices
)
from secrecy import (
    snr_db_to_sigma_sq,
)
from ber import (
    simulate_ssk_transmission_reflection
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
        
        # Create masked array to handle NaN values
        masked_grid = np.ma.masked_invalid(self.grid)
        
        # Plot heatmap
        plt.imshow(masked_grid, cmap=cmap, origin='lower')
        plt.colorbar(label='Value')
        
        # Draw building outlines
        if show_buildings:
            for building in self.buildings:
                x, y, w, h = building
                plt.plot([x-0.5, x+w-0.5, x+w-0.5, x-0.5, x-0.5],
                        [y-0.5, y-0.5, y+h-0.5, y+h-0.5, y-0.5],
                        'r-', linewidth=2)
        
        # Plot points of interest
        if show_points and self.points:
            for label, (x, y) in self.points.items():
                plt.plot(x, y, 'o', color=point_color, markersize=8)
                plt.text(x + label_offset[0], y + label_offset[1], label,
                        color=point_color, fontweight='bold')
        
        plt.grid(True)
        plt.title('Heatmap with Buildings and Points')
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
        
        for y in range(self.height):
            for x in range(self.width):
                if np.isnan(self.grid[y, x]):
                    continue

                px, py = self.points[point]
                if self._line_intersects_building(x, y, px, py):
                    continue
                    
                distance = np.sqrt((x - px)**2 + (y - py)**2)                
                distances[y, x] = distance
                
        return distances

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
    # free_space_path_loss = (4 * np.pi / lam) ** 2 * d ** k
    # magnitude = np.sqrt(1 / free_space_path_loss)
    # phase_shift = d * 2 * np.pi / lam
    # H = np.array([[magnitude * np.exp(1j * phase_shift) for _ in range(L)] for _ in range(K)])

    H = generate_random_channel_matrix(K, L)
    
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
    px, py = 8, 12
    heatmap.add_point('T', tx, ty)
    heatmap.add_point('R', rx, ry)
    heatmap.add_point('P', px, py)

    distances_from_T = heatmap.calculate_distance_from_point('T')
    distances_from_P = heatmap.calculate_distance_from_point('P')

    H = calculate_mimo_channel_gain(distances_from_P[tx, ty], K, N)
    G = calculate_mimo_channel_gain(distances_from_P[rx, ry], N, K)
    Ps, _ = calculate_multi_ris_reflection_matrices(
            K, N, J, M, [G], H, eta, []
        )
    P = unify_ris_reflection_matrices(Ps, [])

    snr_db = 10
    sigma_sq = snr_db_to_sigma_sq(snr_db)
    num_symbols=1000
    def calculate_ber_per_point(x: int, y: int) -> float:
        F = calculate_mimo_channel_gain(distances_from_P[x, y], N, K)
        if x == rx and y == ry:
            F = G
        effective_channel = F @ P @ H
        errors = 0
        for _ in range(num_symbols):
            x = np.zeros(K)
            x[np.random.randint(K)] = 1
            if not simulate_ssk_transmission_reflection(x, effective_channel, sigma_sq):
                errors += 1
        ber = errors / num_symbols
        return ber

    heatmap.apply_function(calculate_ber_per_point)
    heatmap.visualize()