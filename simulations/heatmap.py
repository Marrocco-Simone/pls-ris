import numpy as np
import matplotlib.pyplot as plt
# * pip install PyQt6
import PyQt6
from typing import List, Tuple, Callable

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
            for x in range(self.width):
                if not np.isnan(self.grid[y, x]):  # Skip buildings
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

        # Check each building
        for bx, by, bw, bh in self.buildings:
            # Check intersection with each edge of building
            building_corners = [
                (bx, by), (bx + bw, by),
                (bx + bw, by + bh), (bx, by + bh)
            ]
            
            # Check each edge of the building
            for i in range(4):
                if intersect(
                    (x1, y1), (x2, y2),
                    building_corners[i], building_corners[(i + 1) % 4]
                ):
                    return True
        return False

    def calculate_distance_from_points(self, points: List[str] = None) -> np.ndarray:
        """
        Calculate the minimum distance from each grid cell to the specified points.
        
        Args:
            points: List of point labels to consider. If None, use all points.
        Returns:
            Grid of distances
        """
        if points is None:
            points = list(self.points.keys())
        
        distances = np.full_like(self.grid, np.inf)
        
        for y in range(self.height):
            for x in range(self.width):
                if np.isnan(self.grid[y, x]):  # Skip buildings
                    continue
                    
                min_distance = float('inf')
                for label in points:
                    px, py = self.points[label]
                    if self._line_intersects_building(x, y, px, py):
                        continue

                    distance = np.sqrt((x - px)**2 + (y - py)**2)
                    min_distance = min(min_distance, distance)
                
                distances[y, x] = min_distance
                
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
    free_space_path_loss = (4 * np.pi / lam) ** 2 * d ** k

    magnitude = np.sqrt(1 / free_space_path_loss)
    phase_shift = d * 2 * np.pi / lam

    H = np.array([[magnitude * np.exp(1j * phase_shift) for _ in range(L)] for _ in range(K)])
    
    return H

# Example usage:
if __name__ == "__main__":
    # Parameters
    d = 100  # meters
    L = 4  # number of transmit antennas
    K = 4  # number of receive antennas
    
    # Calculate channel gains
    H = calculate_mimo_channel_gain(d, L, K)
    
    print("\nChannel Gain Matrix:")
    print(np.abs(H))  

    heatmap = HeatmapGenerator(20, 20)
    
    heatmap.add_building(0, 12, 8, 8)  
    heatmap.add_building(12, 0, 8, 8)

    # Add points of interest
    heatmap.add_point('T', 10, 3)
    heatmap.add_point('R', 15, 10)
    heatmap.add_point('P', 7.5, 11.5)
    
    # Define a sample function (distance from center)
    def distance_from_center(x: int, y: int) -> float:
        center_x, center_y = 10, 10
        return np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    def distance_from_a(x, y):
        ax, ay = heatmap.get_point_coordinates('A')
        return np.sqrt((x - ax)**2 + (y - ay)**2)
    
    # Apply the function and visualize
    # heatmap.apply_function(distance_from_center)
    # heatmap.visualize()

    distances = heatmap.calculate_distance_from_points(['T'])
    heatmap.grid = distances
    heatmap.visualize(cmap='coolwarm')