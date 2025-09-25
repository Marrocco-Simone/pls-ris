import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
# * pip install PyQt6
import PyQt6
from typing import List, Tuple, Callable, Literal, Dict, Any, TypedDict, List, TypeVar
from numpy import ndarray, float64, dtype
import os
import json
from tqdm import tqdm
import time
from multiprocess import Pool, cpu_count # pyright: ignore[reportAttributeAccessIssue]
from functools import partial
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from diagonalization import (
  calculate_ris_reflection_matrice,
  unify_ris_reflection_matrices,
  random_reflection_vector,
)
from ber import (
    simulate_ssk_transmission_reflection,
    simulate_ssk_transmission_direct
)
from noise_power_utils import (
    calculate_channel_power,
    calculate_signal_power,
    create_random_noise_vector_from_snr,
    create_random_noise_vector_from_noise_floor
)
from heatmap_utils import (
    calculate_signal_power_from_channel_using_ssk,
    calculate_free_space_path_loss,
    calculate_mimo_channel_gain
)

max_cpu_count = 64
results_folder = './results_data_v2'
results_folder_pdf = results_folder + '/pdf'
results_folder_data = results_folder + '/data'

class Globals(TypedDict):
    K: int
    N: int
    num_symbols: int
    use_noise_floor: bool
    Pt_dbm: float
    eta: float
    snr_db: int

globals: Globals = {
    'K': 4,
    'N': 36,
    'eta': 0.9,
    'num_symbols': 1000,
    'use_noise_floor': True,
    'Pt_dbm': 0.0,
    'snr_db': 10,
}

PathLoss = Literal['sum', 'product', 'active']
class PathLossType(TypedDict):
    name: PathLoss
    calculate: bool

path_loss_types: List[PathLossType] = [
    {
        'name': 'sum',
        'calculate': True
    },
    {
        'name': 'product',
        'calculate': True
    },
    {
        'name': 'active',
        'calculate': True
    }
]

# Area = Tuple[int, int, int, int]
# Point = Tuple[int, int]
# GridPoint = Tuple[float, float]
class Building(TypedDict):
    x: int
    y: int
    width: int
    height: int
class Point(TypedDict):
    x: int
    y: int

def calculate_distance(point_1: Point, point_2: Point) -> float:
    return np.sqrt((point_1['x'] - point_2['x'])**2 + (point_1['y'] - point_2['y'])**2)

class Situation(TypedDict):
    simulation_name: str
    calculate: bool
    force_recompute: bool
    width: int
    height: int
    resolution: float
    buildings: List[Building]
    transmitter: Point
    ris_points: List[Point]
    receivers: List[Point]

situations: List[Situation] = [
    {
        "simulation_name": "Single Reflection",
        "calculate": True,
        "force_recompute": False,
        "width": 20,
        "height": 20,
        "resolution": 0.5,
        "buildings": [
            {'x': 0, 'y': 10, 'width': 7, 'height': 10},
            {'x': 8, 'y': 0, 'width': 12, 'height': 8},
        ],
        "transmitter": {'x': 3,'y': 3},
        "ris_points": [
            {'x': 7,'y': 9}
        ],
        "receivers": [
            {'x': 16,'y': 11}, 
            {'x': 10,'y': 18}
        ]
    },
    {
        "simulation_name": "RISs in series, only final",
        "calculate": True,
        "force_recompute": False,
        "width": 20,
        "height": 20,
        "resolution": 0.5,
        "buildings": [
            {'x': 0, 'y': 10, 'width': 10, 'height': 10},
            {'x': 2, 'y': 4, 'width': 7, 'height': 1},
        ],
        "transmitter": {'x': 1, 'y': 1},
        "ris_points": [
            {'x': 0, 'y': 9}, 
            {'x': 10, 'y': 9}
        ],
        "receivers": [
            {'x': 16, 'y': 14}, 
            {'x': 12, 'y': 18}
        ]
    },
    {
        "simulation_name": "RISs in series",
        "calculate": True,
        "force_recompute": False,
        "width": 20,
        "height": 20,
        "resolution": 0.5,
        "buildings": [
            {'x': 0, 'y': 10, 'width': 10, 'height': 10},
            {'x': 3, 'y': 4, 'width': 7, 'height': 1},
            {'x': 15, 'y': 10, 'width': 5, 'height': 1},
            {'x': 9, 'y': 0, 'width': 1, 'height': 8},
            {'x': 5, 'y': 7, 'width': 7, 'height': 1},
        ],
        "transmitter": {'x': 1, 'y': 1},
        "ris_points": [
            {'x': 0, 'y': 9},
            {'x': 10, 'y': 9},
            {'x': 18, 'y': 6},
        ],
        "receivers": [
            {'x': 4, 'y': 5},
            {'x': 14, 'y': 16},
            {'x': 12, 'y': 18},
            {'x': 11, 'y': 3},
            {'x': 15, 'y': 1},
        ]
    }
]

class Grid:
    def __init__(self, grid_width: int, grid_height: int):
        self.grid = np.zeros((grid_height, grid_width))
    
    def __setitem__(self, point: Point, value: float):
        self.grid[point['y'], point['x']] = value
    
    def __getitem__(self, point: Point) -> float:
        return self.grid[point['y'], point['x']]
    
    def set_building(self, building: Building, value: float):
        """Set a rectangular region defined by a building to a specific value."""
        y_start = building['y']
        y_end = y_start + building['height']
        x_start = building['x']
        x_end = x_start + building['width']
        
        self.grid[y_start:y_end, x_start:x_end] = value

    def __array__(self):
        """Return the grid as a NumPy array for compatibility with np.savez"""
        return self.grid

T = TypeVar('T')
Graph = Dict[str, Dict[str, T]]
class Heatmap:
    width: int
    height: int
    resolution: float
    buildings: List[Building]
    # todo add the type of the points instead of relying on the name
    points: Dict[str, Point]
    grid_width: int
    grid_height: int
    grid: Grid
    stat_grids: Dict[str, Grid]
    distance_graph: Graph[float]
    channel_graph: Graph[np.ndarray]
    parent_tree: Dict[str, str]

class HeatmapGenerator(Heatmap):
    def __init__(self, situation: Situation):
        # TODO precheck that the situation data is ok (points outside the grid, ecc)

        self.width = situation['width']
        self.height = situation['height']
        self.resolution = situation['resolution']
        
        # * Calculate grid dimensions based on resolution
        self.grid_width = int(self.width / self.resolution)
        self.grid_height = int(self.height / self.resolution)
        self.grid = Grid(self.grid_width, self.grid_height)

        self.buildings = situation['buildings']
        for building in self.buildings:
            grid_building: Building = {
                'x': self._meters_to_grid(building['x']),
                'y': self._meters_to_grid(building['y']),
                'width': self._meters_to_grid(building['width']),
                'height': self._meters_to_grid(building['height']),
            }

            # * Mark building area as NaN to exclude from heatmap
            self.grid.set_building(grid_building, np.nan)

        self.points = { 'T': situation['transmitter'] }
        for i, ris_point in enumerate(situation['ris_points']):
            self.points[f'P{i+1}'] = ris_point
        for i, receiver in enumerate(situation['receivers']):
            self.points[f'R{i+1}'] = receiver

        self.distance_graph = {}
        self.channel_graph = {}
        for label in self.points.keys():
            self.distance_graph[label] = {}
            self.channel_graph[label] = {}
        for label, point in self.points.items():
            for other_label, other_point in self.points.items():
                if other_label in self.distance_graph[label]: continue
                elif other_label == label: 
                    distance = 0
                elif self._line_intersects_building(point, other_point):
                    distance = np.inf
                else:
                    distance = calculate_distance(point, other_point)
                self.distance_graph[label][other_label] = distance
                self.distance_graph[other_label][label] = distance

        result = self.check_ris_cycles()
        if result:
            raise ValueError("The RIS configuration contains cycles, which is not allowed.")
        
        already_checked = set()
        self.parent_tree = { 'T': 'T' }
        def calculate_next_node_dfs(label: str):
            if label in already_checked: return True
            already_checked.add(label)
            dim_label = globals['N'] if label[0] == 'P' else globals['K']
            for p_label, _ in self.points.items():
                if p_label == label: continue
                distance = self.distance_graph[label][p_label]
                if distance != np.inf and p_label not in self.parent_tree: 
                    self.parent_tree[p_label] = label
            for p_label, _ in self.points.items():
                distance = self.distance_graph[label][p_label]
                dim_p_label = globals['N'] if p_label[0] == 'P' else globals['K']
                self.channel_graph[label][p_label] = calculate_mimo_channel_gain(distance, dim_label, dim_p_label)
                calculate_next_node_dfs(p_label)
        calculate_next_node_dfs("T")   

        # todo remove
        # self.log_graphs()   
        
        M = len(situation['ris_points'])
       
        self.stat_grids = {
            'BER path loss sum': Grid(self.grid_width, self.grid_height),
            'BER path loss product': Grid(self.grid_width, self.grid_height),
            'BER path loss active': Grid(self.grid_width, self.grid_height),
            'SNR path loss sum': Grid(self.grid_width, self.grid_height),
            'SNR path loss product': Grid(self.grid_width, self.grid_height),
            'SNR path loss active': Grid(self.grid_width, self.grid_height),
            'SNR from T': Grid(self.grid_width, self.grid_height),
        }
        for i in range(M):
            self.stat_grids[f'SNR from P{i+1}'] = Grid(self.grid_width, self.grid_height)


    def _meters_to_grid(self, f: int) -> int:
        """Convert meter coordinates to grid coordinates"""
        return int(f / self.resolution)
        

    # def _grid_to_meters(self, i: int) -> int:
    #     """Convert grid coordinates to meter coordinates"""
    #     return int(i * self.resolution)
    
    def _point_meters_to_grid(self, point: Point) -> Point:
        return {
            'x': self._meters_to_grid(point['x']),
            'y': self._meters_to_grid(point['y'])
        }
    
    # def _point_grid_to_meters(self, point: Point) -> Point:
    #     return {
    #         'x': self._grid_to_meters(point['x']),
    #         'y': self._grid_to_meters(point['y'])
    #     }

    def log_graphs(self):
        for s_label, children in self.distance_graph.items():
            parent_info = f"\t(parent: {self.parent_tree[s_label]})" if s_label in self.parent_tree else ""
            print(f"{s_label}{parent_info}:")
            for e_label, distance in children.items():
                if s_label in self.channel_graph:
                    if e_label in self.channel_graph[s_label]:
                        channel_power = calculate_channel_power(self.channel_graph[s_label][e_label])
                        channel_shape = self.channel_graph[s_label][e_label].shape
                        channel_info = f"channel of shape {channel_shape} and power {channel_power}"
                    else: channel_info = f"self.channel_graph[{s_label}] does not have a link to [{e_label}]"
                else: channel_info = f"self.channel_graph does not have a link to [{s_label}]"
                print(f"\t{e_label}:\tdistance: {distance:.2f}\t{channel_info}")

    
    def _line_intersects_building(self, point_1: Point, point_2: Point) -> bool:
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

        for building in self.buildings:
            bx = building['x']
            by = building['y']
            bw = building['width']
            bh = building['height']
            building_corners = [
                (bx, by), (bx + bw, by),
                (bx + bw, by + bh), (bx, by + bh)
            ]

            for i in range(4):
                if intersect(
                    (point_1['x'], point_1['y']), (point_2['x'], point_2['y']),
                    building_corners[i], building_corners[(i + 1) % 4]
                ):
                    return True
        return False
    
    def check_ris_cycles(self) -> bool:
        already_checked = set()
        def check_next_nodes_dfs(label: str, from_label: str) -> bool:
            if label in already_checked: return True
            already_checked.add(label)
            for p_label, _ in self.points.items():
                if p_label == label: continue
                if p_label == from_label: continue
                if not p_label.startswith('P'): continue
                if self.distance_graph[label][p_label] == np.inf: continue
                if check_next_nodes_dfs(p_label, label): return True
            return False
        return check_next_nodes_dfs("T", "T")
    
    def get_new_RIS_configurations(self):
        Ps: Dict[str, ndarray] = {}

        def calculate_P(p_label):
            if p_label[0] != 'P': return
            if p_label in Ps: return

            connected_receivers: List[str] = []
            connected_receivers_channel_gain: List[ndarray] = []
            for r_label, r_point in self.points.items():
                if r_label[0] != 'R': continue
                distance = self.distance_graph[p_label][r_label]
                if distance == np.inf: continue
                connected_receivers.append(r_label)
                connected_receivers_channel_gain.append(self.channel_graph[p_label][r_label])

            if len(connected_receivers) == 0:
                P = np.diag(random_reflection_vector(globals['N'], globals['eta']))
                Ps[p_label] = P
            elif self.parent_tree[p_label] == 'T':
                P, _ = calculate_ris_reflection_matrice(globals['K'], globals['N'], len(connected_receivers), connected_receivers_channel_gain, self.channel_graph['T'][p_label], globals['eta'])
                Ps[p_label] = P
            else:
                P_chain: List[ndarray] = []
                C_chain: List[ndarray] = []
                prev_label = p_label
                curr_label = self.parent_tree[p_label]
                while curr_label != 'T':
                    calculate_P(curr_label)
                    P = Ps[curr_label]
                    C = self.channel_graph[prev_label][curr_label]
                    P_chain.insert(0, P)
                    C_chain.insert(0, C)
                    prev_label = curr_label
                    curr_label = self.parent_tree[curr_label]
                P_prev = unify_ris_reflection_matrices(P_chain, C_chain)
                modified_Gs = [G @ P_prev @ C_chain[-1] for G in connected_receivers_channel_gain]
                P, _ = calculate_ris_reflection_matrice(globals['K'], globals['N'], len(connected_receivers), modified_Gs, self.channel_graph['T'][p_label], globals['eta'])
                Ps[p_label] = P

        for p_label, p_point in self.points.items():
            calculate_P(p_label)

        return Ps


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
        legend_filename = f"{results_folder_pdf}/{label} legend_{orientation}.pdf"
        if os.path.exists(legend_filename):
            return
        fig = plt.figure(figsize=(6, 1) if orientation == 'horizontal' else (1.5, 6))
        rect = (0.1, 0.4, 0.8, 0.3) if orientation == 'horizontal' else (0.3, 0.1, 0.3, 0.8)
        ax = fig.add_axes(rect=rect)

        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
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
    
    def visualize(self, title: str, cmap='viridis', show_buildings=True, show_points=True, point_color='white', vmin: float | None = None, vmax: float | None = None, log_scale=False, label='BER', show_receivers_values=False, show_heatmap=True, show_legend=False):
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
        os.makedirs(results_folder_pdf, exist_ok=True)
        os.makedirs(results_folder_data, exist_ok=True)

        # TODO save npx file
        # data_filename = f"{results_folder_data}/{title}.npz"

        # np.savez(
        #     data_filename,
        #     grid=self.grid,
        #     width=self.width,
        #     height=self.height,
        #     resolution=self.resolution,
        #     buildings=np.array(self.buildings),
        #     points=self.points,
        #     log_scale=log_scale
        # )
        # print(f"Saved data to {data_filename}")

        figure = plt.figure(figsize=(10, 8))

        if log_scale and show_heatmap:
            # * Add small offset to zero values before taking log
            grid_for_log = np.copy(self.grid)
            grid_for_log[grid_for_log == 0] = 1e-10
            masked_grid = np.ma.masked_invalid(np.log10(grid_for_log))
            title += ' (log scale)'
        else:
            masked_grid = np.ma.masked_invalid(self.grid)

        # ! changed from list to tuple, since otherwise error
        extent = (0, self.width, 0, self.height)

        if show_heatmap:
            plt.imshow(masked_grid, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, extent=extent)
            if not log_scale:
                self._save_colorbar_legend(title, cmap=cmap, vmin=vmin, vmax=vmax, label=label, orientation='horizontal')
                self._save_colorbar_legend(title, cmap=cmap, vmin=vmin, vmax=vmax, label=label, orientation='vertical')
            if show_legend:
                plt.colorbar(label=label, orientation='vertical')
        else:
            plt.imshow(np.ones_like(masked_grid), cmap='Greys', origin='lower', vmin=0, vmax=1, extent=extent, alpha=0.1)

        if show_buildings:
            for building in self.buildings:
                # x, y, w, h = building
                x = building['x']
                y = building['y']
                w = building['width']
                h = building['height']
                x_array = [x, x+w, x+w, x, x]
                y_array = [y, y, y+h, y+h, y]
                plt.plot(x_array, y_array,'r-', linewidth=2)

        if show_points and self.points:
            c = 0.5 * self.resolution
            for label, point in self.points.items():
                x = point['x']
                y = point['y']
                if label[0] == 'R' and show_receivers_values and show_heatmap:
                    grid_point = self._point_meters_to_grid(point)
                    value = self.grid[grid_point]
                    if log_scale and value > 0:
                        value = np.log10(value)
                    label += f" ({value:.2f})"
                plt.plot(x + c, y + c, 'o', color=point_color, markersize=6)
                plt.text(x + 2 * c, y + 2 * c, label, color=point_color, fontweight=1000, fontsize=20, bbox=dict(pad=0.2, boxstyle='round',  lw=0, ec=None, fc='black', alpha=0.3))
            # * plot a line between all points if the lines is not crossing a building
            for label1, point_1 in self.points.items():
                x1 = point_1['x']
                y1 = point_1['y']
                for label2, point_2 in self.points.items():
                    x2 = point_2['x']
                    y2 = point_2['y']
                    if label1 == label2: continue
                    if label1[0] == 'R' and label2[0] == 'R': continue
                    if self._line_intersects_building(point_1, point_2): continue
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
        plt.savefig(f"{results_folder_pdf}/{title}.pdf", dpi=300, format='pdf', bbox_inches='tight')
        print(f"Saved {title}.pdf")
        plt.close(figure)


def process_point(ber_heatmap: HeatmapGenerator, point: Point):
    is_building = np.isnan(ber_heatmap.grid[point])
    if is_building: return
    
    unique_seed = (int(time.time() * 1000000) % (2**32) + os.getpid() * 1000 + int(point['y'] * 100 + point['x'])) % (2**32)
    np.random.seed(unique_seed)

    distance_from: Dict[str, float] = {}
    channel_gain_from: Dict[str, np.ndarray] = {}
    point_label: str | None = None
    for label, heatmap_point in ber_heatmap.points.items():
        if heatmap_point['x'] == point['x'] and heatmap_point['y'] == point['y'] and label[0] != 'P':
            point_label = label
            break
    if point_label: 
        distance_from = ber_heatmap.distance_graph[point_label]
        channel_gain_from = ber_heatmap.channel_graph[point_label]
    else: 
        can_receive_signal = False
        for label, heatmap_point in ber_heatmap.points.items():
            distance_from[label] = calculate_distance(point, heatmap_point)
            if label[0] == 'R': continue
            dim_label = globals['N'] if label[0] == 'P' else globals['K']
            channel_gain_from[label] = calculate_mimo_channel_gain(distance_from[label], dim_label, globals['K'])
            if distance_from[label] == np.inf: continue
            can_receive_signal = True
        if not can_receive_signal: return

    # ! mean_power is used only to set the BER as np.nan
    # ! power_from_Ps is not useful

    if distance_from['T'] != np.inf:
        B = channel_gain_from['T'] * calculate_free_space_path_loss(distance_from['T'])
        power_from_T = calculate_channel_power(B)
        signal_power_from_T = calculate_signal_power_from_channel_using_ssk(globals['K'], B, globals['Pt_dbm'])

    ris_paths_to_T: List[List[str]] = []
    for label, p_point in ber_heatmap.points.items():
        if label[0] != 'P': continue
        if distance_from[label] == np.inf: continue
        curr_label = label
        ris_paths_to_T.append([])
        while curr_label != 'T':
            ris_paths_to_T[-1].append(curr_label)
            curr_label = ber_heatmap.parent_tree[curr_label]

    # for _ in range(globals['num_symbols']):
    Ps = ber_heatmap.get_new_RIS_configurations()
    

def ber_heatmap_reflection_simulation(
        situation: Situation
):
    simulation_name = situation['simulation_name']
    # force_recompute = situation['force_recompute']
    width = situation['width']
    height = situation['height']
    buildings = situation['buildings']
    transmitter = situation['transmitter']
    ris_points = situation['ris_points']
    receivers = situation['receivers']

    M = len(ris_points)
    J = len(receivers)

    n_processes = min(cpu_count(), max_cpu_count)
    print(f"Using {n_processes} CPU cores for parallel processing.")

    os.makedirs(results_folder, exist_ok=True)
    # TODO check if data is already calculated present 

    ber_heatmap = HeatmapGenerator(situation)
    ber_heatmap.visualize(f'{simulation_name}', label='', show_receivers_values=False, vmax=0.0, vmin=0.0, show_heatmap=False) 

    for y in range(ber_heatmap.height):
        for x in range(ber_heatmap.width):
            point: Point = {
                'y': y, 'x': x
            }
            process_point(ber_heatmap, point)

def main():
    begin_time = time.perf_counter()
    for situation in situations:
        if not situation['calculate']: continue
        start_time = time.perf_counter()

        ber_heatmap_reflection_simulation(situation)

        end_time = time.perf_counter()
        print(f"{situation['simulation_name']} simulation took {end_time - start_time:.2f} seconds for {globals['num_symbols']} symbols with K={globals['K']}, N={globals['N']}\n\n")
    end_time = time.perf_counter()
    print(f"Total time taken: {end_time - begin_time:.2f} seconds for {globals['num_symbols']} symbols with K={globals['K']}, N={globals['N']}")

if __name__ == "__main__":
    main()
