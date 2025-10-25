import numpy as np
import tensorflow as tf
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera, PathSolver, mi
import time
import os
import gc
import drjit as dr
from typing import Tuple, Dict, List
from tqdm import tqdm
from heatmap_situations import situations, Situation, Point, Building
from multiprocess import Pool, cpu_count

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_visible_devices([], 'GPU')

dr.set_expand_threshold(1024 * 1024 * 1024)

try:
    dr.set_backend('cpu')
    print("Dr.Jit backend set to CPU")
except:
    print("Could not set Dr.Jit backend to CPU, continuing...")

class Actor:
    def __init__(self, name: str, position: Tuple[float, float, float],
                 orientation: Tuple[float, float, float], rows: int, cols: int):
        self.name = name
        self.position = position
        self.orientation = orientation
        self.rows = rows
        self.cols = cols

def calculate_distance(point_1: Point, point_2: Point) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt((point_1['x'] - point_2['x'])**2 + (point_1['y'] - point_2['y'])**2)

def line_intersects_building(point_1: Point, point_2: Point, buildings: List[Building]) -> bool:
    """Check if line between two points intersects any building."""
    def ccw(A: tuple, B: tuple, C: tuple) -> bool:
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def intersect(A: tuple, B: tuple, C: tuple, D: tuple) -> bool:
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    for building in buildings:
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

def is_point_inside_building(point: Point, buildings: List[Building]) -> bool:
    """Check if a point is inside any building."""
    for building in buildings:
        if (building['x'] <= point['x'] < building['x'] + building['width'] and
            building['y'] <= point['y'] < building['y'] + building['height']):
            return True
    return False

def calculate_orientation(from_point: Point, to_point: Point) -> Tuple[float, float, float]:
    """Calculate orientation (roll, yaw, pitch) from one point toward another."""
    vec = (to_point['x'] - from_point['x'], to_point['y'] - from_point['y'])
    angle = float(np.degrees(np.arctan2(vec[1], vec[0])))
    return (0.0, angle, 0.0)

def calculate_ris_orientation(ris_point: Point, incident_point: Point,
                              reflected_points: List[Point]) -> Tuple[float, float, float]:
    """Calculate RIS orientation to reflect from incident to reflected points."""
    vec_incident = (incident_point['x'] - ris_point['x'], incident_point['y'] - ris_point['y'])
    incident_angle = float(np.degrees(np.arctan2(vec_incident[1], vec_incident[0])))

    if len(reflected_points) == 0:
        return (0.0, incident_angle + 90, 0.0)

    reflected_angles = []
    for rp in reflected_points:
        vec = (rp['x'] - ris_point['x'], rp['y'] - ris_point['y'])
        reflected_angles.append(float(np.degrees(np.arctan2(vec[1], vec[0]))))

    avg_reflected = sum(reflected_angles) / len(reflected_angles)
    incident_from_opposite = incident_angle + 180
    bisector = (incident_from_opposite + avg_reflected) / 2

    return (0.0, bisector - 90, 0.0)

def compute_channel_matrix(scene, my_cam, tx: Actor, rx: Actor) -> np.ndarray:
    """Compute channel matrix between transmitter and receiver."""
    scene.tx_array = PlanarArray(
        num_rows=tx.rows,
        num_cols=tx.cols,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V"
    )

    scene.rx_array = PlanarArray(
        num_rows=rx.rows,
        num_cols=rx.cols,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V"
    )

    tx_pos = [float(x) for x in tx.position]
    tx_orient = [float(x) for x in tx.orientation]
    rx_pos = [float(x) for x in rx.position]
    rx_orient = [float(x) for x in rx.orientation]

    tx_obj = Transmitter(tx.name, position=mi.Point3f(tx_pos), orientation=mi.Point3f(tx_orient))
    rx_obj = Receiver(rx.name, position=mi.Point3f(rx_pos), orientation=mi.Point3f(rx_orient))
    scene.add(tx_obj)
    scene.add(rx_obj)

    p_solver = PathSolver()

    paths = p_solver(
        scene=scene,
        max_depth=3,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=True,
        synthetic_array=False,
        seed=42
    )

    a, tau = paths.cir(normalize_delays=True, out_type="numpy")
    h = tf.reduce_sum(a, axis=-2)
    h = tf.squeeze(h)
    h_numpy = h.numpy()

    scene.remove(tx.name)
    scene.remove(rx.name)

    del paths, a, tau, h, tx_obj, rx_obj, p_solver
    dr.sync_thread()
    tf.keras.backend.clear_session()
    dr.flush_malloc_cache()
    gc.collect()

    return h_numpy

def compute_channels_for_scenario(situation: Situation, K: int, N: int) -> Dict:
    """Compute all channel matrices for a single scenario."""
    simulation_name = situation['simulation_name']
    print(f"\n{'='*60}")
    print(f"Processing scenario: {simulation_name}")
    print(f"{'='*60}")

    scene_path = f"mesh_scene/{simulation_name}.xml"
    if not os.path.exists(scene_path):
        print(f"Error: Scene file not found: {scene_path}")
        return None

    print(f"Loading scene: {scene_path}")
    scene = load_scene(scene_path)
    scene.frequency = 3.5e9
    my_cam = Camera(position=mi.Point3f([situation['width']/2, situation['height']/2, 30]),
                   look_at=[situation['width']/2, situation['height']/2, 0])

    grid_width = int(situation['width'] / situation['resolution'])
    grid_height = int(situation['height'] / situation['resolution'])

    result = {
        'metadata': {
            'K': K,
            'N': N,
            'width': situation['width'],
            'height': situation['height'],
            'resolution': situation['resolution'],
            'grid_width': grid_width,
            'grid_height': grid_height
        },
        'point_to_point': {},
        'source_to_grid': {}
    }

    transmitter = situation['transmitter']
    ris_points = situation['ris_points']
    receivers = situation['receivers']
    buildings = situation['buildings']

    M = len(ris_points)
    J = len(receivers)

    print(f"Scenario parameters:")
    print(f"  Grid: {grid_width} × {grid_height} = {grid_width * grid_height} points")
    print(f"  Transmitter: {transmitter}")
    print(f"  RIS points: {M}")
    print(f"  Receivers: {J}")
    print(f"  Buildings: {len(buildings)}")

    print(f"\n{'='*60}")
    print("Computing point-to-point channels...")
    print(f"{'='*60}")

    actor_T = Actor(
        'T',
        (transmitter['x'], transmitter['y'], 1.5),
        calculate_orientation(transmitter, ris_points[0] if M > 0 else receivers[0]),
        rows=int(np.sqrt(K)),
        cols=int(np.sqrt(K))
    )

    actors_P = []
    for i, ris_point in enumerate(ris_points):
        visible_receivers = [r for r in receivers if not line_intersects_building(ris_point, r, buildings)]
        orientation = calculate_ris_orientation(ris_point, transmitter, visible_receivers)
        actors_P.append(Actor(
            f'P{i+1}',
            (ris_point['x'], ris_point['y'], 1.5),
            orientation,
            rows=int(np.sqrt(N)),
            cols=int(np.sqrt(N))
        ))

    actors_R = []
    for i, receiver in enumerate(receivers):
        orientation = calculate_orientation(receiver, ris_points[0] if M > 0 else transmitter)
        actors_R.append(Actor(
            f'R{i+1}',
            (receiver['x'], receiver['y'], 1.5),
            orientation,
            rows=int(np.sqrt(K)),
            cols=int(np.sqrt(K))
        ))

    print("Computing T→P channels...")
    for i, actor_P in enumerate(actors_P):
        print(f"  T → P{i+1}...", end=' ', flush=True)
        try:
            channel = compute_channel_matrix(scene, my_cam, actor_T, actor_P)
            result['point_to_point'][('T', f'P{i+1}')] = channel
            print(f"✓ (shape: {channel.shape})")

            del scene
            gc.collect()
            scene = load_scene(scene_path)
            scene.frequency = 3.5e9
        except Exception as e:
            print(f"✗ Error: {e}")

    print("\nComputing P→R channels...")
    for i, actor_P in enumerate(actors_P):
        for j, actor_R in enumerate(actors_R):
            if line_intersects_building(ris_points[i], receivers[j], buildings):
                continue
            print(f"  P{i+1} → R{j+1}...", end=' ', flush=True)
            try:
                channel = compute_channel_matrix(scene, my_cam, actor_P, actor_R)
                result['point_to_point'][(f'P{i+1}', f'R{j+1}')] = channel
                print(f"✓ (shape: {channel.shape})")

                del scene
                gc.collect()
                scene = load_scene(scene_path)
                scene.frequency = 3.5e9
            except Exception as e:
                print(f"✗ Error: {e}")

    print("\nComputing P→P channels (cascaded RIS)...")
    for i in range(M):
        for j in range(i+1, M):
            if line_intersects_building(ris_points[i], ris_points[j], buildings):
                continue
            print(f"  P{i+1} → P{j+1}...", end=' ', flush=True)
            try:
                channel = compute_channel_matrix(scene, my_cam, actors_P[i], actors_P[j])
                result['point_to_point'][(f'P{i+1}', f'P{j+1}')] = channel
                result['point_to_point'][(f'P{j+1}', f'P{i+1}')] = channel.T
                print(f"✓ (shape: {channel.shape})")

                del scene
                gc.collect()
                scene = load_scene(scene_path)
                scene.frequency = 3.5e9
            except Exception as e:
                print(f"✗ Error: {e}")

    print(f"\n{'='*60}")
    print("Computing grid point channels...")
    print(f"{'='*60}")

    result['source_to_grid']['T'] = {}
    for i in range(M):
        result['source_to_grid'][f'P{i+1}'] = {}

    grid_points = []
    for y in range(grid_height):
        for x in range(grid_width):
            point: Point = {
                'x': x * situation['resolution'],
                'y': y * situation['resolution']
            }
            if not is_point_inside_building(point, buildings):
                grid_points.append((x, y, point))

    print(f"Processing {len(grid_points)} grid points (skipped {grid_width * grid_height - len(grid_points)} inside buildings)")

    for grid_x, grid_y, point in tqdm(grid_points, desc="Grid points"):
        grid_actor = Actor(
            f'Grid_{grid_x}_{grid_y}',
            (point['x'], point['y'], 1.5),
            (0.0, 0.0, 0.0),
            rows=int(np.sqrt(K)),
            cols=int(np.sqrt(K))
        )

        if not line_intersects_building(transmitter, point, buildings):
            try:
                channel = compute_channel_matrix(scene, my_cam, actor_T, grid_actor)
                result['source_to_grid']['T'][(grid_x, grid_y)] = channel

                del scene
                gc.collect()
                scene = load_scene(scene_path)
                scene.frequency = 3.5e9
            except:
                pass

        for i, ris_point in enumerate(ris_points):
            if not line_intersects_building(ris_point, point, buildings):
                try:
                    channel = compute_channel_matrix(scene, my_cam, actors_P[i], grid_actor)
                    result['source_to_grid'][f'P{i+1}'][(grid_x, grid_y)] = channel

                    del scene
                    gc.collect()
                    scene = load_scene(scene_path)
                    scene.frequency = 3.5e9
                except:
                    pass

    del scene
    gc.collect()

    print(f"\n{'='*60}")
    print(f"Completed scenario: {simulation_name}")
    print(f"  Point-to-point channels: {len(result['point_to_point'])}")
    print(f"  Grid channels from T: {len(result['source_to_grid']['T'])}")
    for i in range(M):
        print(f"  Grid channels from P{i+1}: {len(result['source_to_grid'][f'P{i+1}'])}")
    print(f"{'='*60}")

    return result

def main():
    """Main function to compute all channel matrices."""
    K = 4
    N = 36

    output_dir = "heatmap/channel_matrices"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "sionna_channels.npz")

    print("="*60)
    print("SIONNA CHANNEL MATRIX COMPUTATION")
    print("="*60)
    print(f"Parameters:")
    print(f"  K (antennas): {K}")
    print(f"  N (RIS elements): {N}")
    print(f"  Output: {output_file}")
    print("="*60)

    all_results = {}

    for situation in situations:
        if not situation['calculate']:
            print(f"\nSkipping {situation['simulation_name']} (calculate=False)")
            continue

        result = compute_channels_for_scenario(situation, K, N)
        if result is not None:
            all_results[situation['simulation_name']] = result

    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}")

    np.savez(
        output_file,
        channels=np.array([all_results], dtype=object)
    )

    print(f"✓ Saved to {output_file}")
    print(f"  Scenarios: {len(all_results)}")

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")
    print("="*60)
    print("Complete!")

if __name__ == "__main__":
    main()
