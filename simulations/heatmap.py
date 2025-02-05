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

    def apply_function(self, func: Callable[[int, int], float]):
        """
        Apply a custom function to calculate values for each point in the grid.
        The function should take x and y coordinates as input and return a float value.
        
        Args:
            func: Function that takes (x, y) coordinates and returns a value
        """
        for y in range(self.height):
            for x in range(self.width):
                if not np.isnan(self.grid[y, x]):  # * Skip buildings
                    self.grid[y, x] = func(x, y)

    def visualize(self, cmap='viridis', show_buildings=True):
        """
        Visualize the heatmap with optional building outlines.
        
        Args:
            cmap: Matplotlib colormap name
            show_buildings: Whether to show building outlines
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
        
        plt.grid(True)
        plt.title('Heatmap with Buildings')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.show()

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
    print(np.abs(H))  # Print magnitude of complex channel gains

    # Create a 20x20 meter area
    heatmap = HeatmapGenerator(20, 20)
    
    # Add some example buildings
    heatmap.add_building(5, 5, 3, 4)   # 3x4 building at (5,5)
    heatmap.add_building(12, 8, 4, 6)  # 4x6 building at (12,8)
    
    # Define a sample function (distance from center)
    def sample_function(x: int, y: int) -> float:
        center_x, center_y = 10, 10
        return np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Apply the function and visualize
    heatmap.apply_function(sample_function)
    heatmap.visualize()