import argparse
import numpy as np
import tensorflow as tf # pyright: ignore[reportMissingImports]
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera, PathSolver, mi # pyright: ignore[reportMissingImports]
import time
import os
import gc
import drjit as dr # pyright: ignore[reportMissingImports]
from typing import Tuple, Dict, List, TypedDict
from tqdm import tqdm
from heatmap_situations import situations, Situation, Point, Building, ChannelMatrix
from heatmap_utils import line_intersects_building, is_point_inside_building
from multiprocess import Pool, cpu_count # pyright: ignore[reportAttributeAccessIssue]
from sionna_utils import Actor, create_tx_actor, create_ris_actor, create_rx_actor, calculate_orientation

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_visible_devices([], 'GPU')

dr.set_expand_threshold(1024 * 1024 * 1024)

try:
    dr.set_backend('cpu')
    print("Dr.Jit backend set to CPU")
except:
    print("Could not set Dr.Jit backend to CPU, continuing...")


class SionnaGlobals(TypedDict):
    K: int
    N: int
    max_depth: int
    los: bool
    specular_reflection: bool
    diffuse_reflection: bool
    refraction: bool
    seed: int


sionna_globals: SionnaGlobals = {
    'K': 4,
    'N': 36,
    'max_depth': 3,
    'los': True,
    'specular_reflection': True,
    'diffuse_reflection': False,
    'refraction': True,
    'seed': 42,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments to override sionna_globals."""
    parser = argparse.ArgumentParser(
        description='Compute Sionna ray-traced channel matrices'
    )
    
    parser.add_argument('--K', type=int, help='Number of transmit antennas')
    parser.add_argument('--N', type=int, help='Number of RIS elements')
    parser.add_argument('--max_depth', type=int, help='Maximum depth for path tracing')
    parser.add_argument('--los', dest='los', action='store_true', help='Enable line-of-sight')
    parser.add_argument('--no_los', dest='los', action='store_false', help='Disable line-of-sight')
    parser.set_defaults(los=None)
    parser.add_argument('--specular_reflection', dest='specular_reflection', action='store_true',
                        help='Enable specular reflection')
    parser.add_argument('--no_specular_reflection', dest='specular_reflection', action='store_false',
                        help='Disable specular reflection')
    parser.set_defaults(specular_reflection=None)
    parser.add_argument('--diffuse_reflection', dest='diffuse_reflection', action='store_true',
                        help='Enable diffuse reflection')
    parser.add_argument('--no_diffuse_reflection', dest='diffuse_reflection', action='store_false',
                        help='Disable diffuse reflection')
    parser.set_defaults(diffuse_reflection=None)
    parser.add_argument('--refraction', dest='refraction', action='store_true', help='Enable refraction')
    parser.add_argument('--no_refraction', dest='refraction', action='store_false', help='Disable refraction')
    parser.set_defaults(refraction=None)
    parser.add_argument('--seed', type=int, help='Random seed for path solver')
    
    return parser.parse_args()


def update_globals_from_args(args: argparse.Namespace) -> None:
    """Override sionna_globals values with command-line arguments."""
    global sionna_globals
    
    if args.K is not None:
        sionna_globals['K'] = args.K
    if args.N is not None:
        sionna_globals['N'] = args.N
    if args.max_depth is not None:
        sionna_globals['max_depth'] = args.max_depth
    if args.los is not None:
        sionna_globals['los'] = args.los
    if args.specular_reflection is not None:
        sionna_globals['specular_reflection'] = args.specular_reflection
    if args.diffuse_reflection is not None:
        sionna_globals['diffuse_reflection'] = args.diffuse_reflection
    if args.refraction is not None:
        sionna_globals['refraction'] = args.refraction
    if args.seed is not None:
        sionna_globals['seed'] = args.seed


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
        max_depth=sionna_globals['max_depth'],
        los=sionna_globals['los'],
        specular_reflection=sionna_globals['specular_reflection'],
        diffuse_reflection=sionna_globals['diffuse_reflection'],
        refraction=sionna_globals['refraction'],
        synthetic_array=False,
        seed=sionna_globals['seed']
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

def compute_channels_for_scenario(situation: Situation) -> ChannelMatrix | None:
    """Compute all channel matrices for a single scenario."""
    global sionna_globals
    K = sionna_globals['K']
    N = sionna_globals['N']
    
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

    channel_matrix = ChannelMatrix()
    channel_matrix.situation_name = simulation_name
    channel_matrix.metadata = {
        'K': K,
        'N': N,
        'max_depth': sionna_globals['max_depth'],
        'los': sionna_globals['los'],
        'specular_reflection': sionna_globals['specular_reflection'],
        'diffuse_reflection': sionna_globals['diffuse_reflection'],
        'refraction': sionna_globals['refraction'],
        'seed': sionna_globals['seed'],
        'width': situation['width'],
        'height': situation['height'],
        'resolution': situation['resolution'],
        'grid_width': grid_width,
        'grid_height': grid_height
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

    actor_T = create_tx_actor(transmitter, ris_points, receivers, K)

    actors_P = []
    for i, ris_point in enumerate(ris_points):
        visible_receivers = [r for r in receivers if not line_intersects_building(buildings, ris_point, r)]
        actors_P.append(create_ris_actor(f'P{i+1}', ris_point, transmitter, visible_receivers, N))

    actors_R = []
    for i, receiver in enumerate(receivers):
        actors_R.append(create_rx_actor(f'R{i+1}', receiver, ris_points, transmitter, K))

    print(f"\n{'='*60}")
    print("Computing channels from sources to all grid points...")
    print(f"{'='*60}")

    grid_points = []
    # First, add RIS and receiver points as destinations
    for ris_point in ris_points:
        grid_points.append((int(ris_point['x'] / situation['resolution']), 
                          int(ris_point['y'] / situation['resolution']), 
                          ris_point))
    for receiver in receivers:
        grid_points.append((int(receiver['x'] / situation['resolution']), 
                          int(receiver['y'] / situation['resolution']), 
                          receiver))
    
    # Then add all grid points
    for y in range(grid_height):
        for x in range(grid_width):
            point: Point = {
                'x': x * situation['resolution'],
                'y': y * situation['resolution']
            }
            if not is_point_inside_building(point, buildings):
                # Skip if this point is already a RIS or receiver
                is_special_point = any(
                    point['x'] == p['x'] and point['y'] == p['y'] 
                    for p in ris_points + receivers
                )
                if not is_special_point:
                    grid_points.append((x, y, point))

    source_actors: List[Actor] = [actor_T] + actors_P
    source_points: List[Point] = [transmitter] + ris_points

    print(f"Processing {len(source_actors)} sources × {len(grid_points)} grid points = {len(source_actors) * len(grid_points)} channels")
    print(f"(Skipped {grid_width * grid_height - len(grid_points)} points inside buildings)")

    # Initialize logging structures
    channel_stats = {
        'total_attempted': 0,
        'total_succeeded': 0,
        'total_failed': 0,
        'total_blocked': 0,
        'by_source': {},
        'channel_powers': []
    }

    for source_actor, source_point in zip(source_actors, source_points):
        source_name = source_actor.name
        channel_stats['by_source'][source_name] = {
            'attempted': 0,
            'succeeded': 0,
            'failed': 0,
            'blocked': 0,
            'powers': []
        }

    for grid_x, grid_y, point in tqdm(grid_points, desc="Grid points"):
        is_ris_dest = point in ris_points
        dim_dest = N if is_ris_dest else K

        # Orient grid point toward transmitter (not fixed orientation)
        grid_orientation = calculate_orientation(point, transmitter)

        grid_actor = Actor(
            f'Grid_{grid_x}_{grid_y}',
            (point['x'], point['y'], 1.5),
            grid_orientation,
            rows=1,
            cols=dim_dest
        )

        for source_actor, source_point in zip(source_actors, source_points):
            source_name = source_actor.name
            channel_stats['total_attempted'] += 1
            channel_stats['by_source'][source_name]['attempted'] += 1

            if line_intersects_building(buildings, source_point, point):
                channel_stats['total_blocked'] += 1
                channel_stats['by_source'][source_name]['blocked'] += 1
                continue

            try:
                channel = compute_channel_matrix(scene, my_cam, source_actor, grid_actor)
                channel_matrix.set(source_point, point, channel)

                # Log channel power
                channel_power = np.linalg.norm(channel, 'fro')**2
                channel_stats['total_succeeded'] += 1
                channel_stats['by_source'][source_name]['succeeded'] += 1
                channel_stats['by_source'][source_name]['powers'].append(channel_power)
                channel_stats['channel_powers'].append({
                    'source': source_name,
                    'dest': (grid_x, grid_y),
                    'power': channel_power
                })

                del scene
                gc.collect()
                scene = load_scene(scene_path)
                scene.frequency = 3.5e9
            except Exception as e:
                channel_stats['total_failed'] += 1
                channel_stats['by_source'][source_name]['failed'] += 1

    del scene
    gc.collect()

    # Print comprehensive channel generation summary
    print(f"\n{'='*60}")
    print(f"CHANNEL GENERATION STATISTICS")
    print(f"{'='*60}")
    print(f"\nOverall Statistics:")
    print(f"  Total attempted: {channel_stats['total_attempted']}")
    print(f"  Total blocked by buildings: {channel_stats['total_blocked']}")
    print(f"  Total succeeded: {channel_stats['total_succeeded']}")
    print(f"  Total failed: {channel_stats['total_failed']}")
    print(f"  Success rate: {100*channel_stats['total_succeeded']/max(1, channel_stats['total_attempted']):.1f}%")

    print(f"\nPer-Source Statistics:")
    for source_name, stats in channel_stats['by_source'].items():
        print(f"  {source_name}:")
        print(f"    Attempted: {stats['attempted']}")
        print(f"    Blocked: {stats['blocked']}")
        print(f"    Succeeded: {stats['succeeded']}")
        print(f"    Failed: {stats['failed']}")
        if stats['succeeded'] > 0:
            success_rate = 100 * stats['succeeded'] / stats['attempted']
            print(f"    Success rate: {success_rate:.1f}%")

            powers = np.array(stats['powers'])
            print(f"    Channel power stats:")
            print(f"      Mean: {np.mean(powers):.3e}")
            print(f"      Median: {np.median(powers):.3e}")
            print(f"      Min: {np.min(powers):.3e}")
            print(f"      Max: {np.max(powers):.3e}")
            print(f"      Std: {np.std(powers):.3e}")

    # Analyze P→R line patterns if receivers exist
    if len(receivers) > 0 and len(ris_points) > 0:
        print(f"\nP→R Line Analysis:")
        ris = ris_points[0]

        for i, receiver in enumerate(receivers):
            print(f"  P1→R{i+1} line analysis:")

            # Collect channels along and off the line
            on_line_powers = []
            off_line_powers = []

            for ch_data in channel_stats['channel_powers']:
                if ch_data['source'] == 'P1':
                    gx, gy = ch_data['dest']
                    point = {'x': gx * situation['resolution'], 'y': gy * situation['resolution']}

                    # Check if point is on P→R line
                    vec_pr = np.array([receiver['x'] - ris['x'], receiver['y'] - ris['y']])
                    vec_pp = np.array([point['x'] - ris['x'], point['y'] - ris['y']])

                    # Compute perpendicular distance to line
                    if np.linalg.norm(vec_pr) > 0:
                        proj = np.dot(vec_pp, vec_pr) / np.linalg.norm(vec_pr)**2
                        if 0 <= proj <= 1:
                            closest_point = vec_pr * proj
                            perp_dist = np.linalg.norm(vec_pp - closest_point)

                            if perp_dist < situation['resolution'] * 0.5:
                                on_line_powers.append(ch_data['power'])
                            elif perp_dist < situation['resolution'] * 3:
                                off_line_powers.append(ch_data['power'])

            if len(on_line_powers) > 0 and len(off_line_powers) > 0:
                on_line_powers = np.array(on_line_powers)
                off_line_powers = np.array(off_line_powers)

                print(f"    On-line channels: {len(on_line_powers)}")
                print(f"      Mean power: {np.mean(on_line_powers):.3e}")
                print(f"      Median power: {np.median(on_line_powers):.3e}")

                print(f"    Off-line nearby channels: {len(off_line_powers)}")
                print(f"      Mean power: {np.mean(off_line_powers):.3e}")
                print(f"      Median power: {np.median(off_line_powers):.3e}")

                ratio = np.mean(off_line_powers) / np.mean(on_line_powers)
                print(f"    Power ratio (off/on): {ratio:.2f}x")

    transmitter_key = channel_matrix._point_to_key(transmitter)
    ris_keys = [channel_matrix._point_to_key(rp) for rp in ris_points]

    total_from_T = len(channel_matrix.data.get(transmitter_key, {}))

    print(f"\n{'='*60}")
    print(f"Completed scenario: {simulation_name}")
    print(f"  Channels from T: {total_from_T}")
    for i, ris_key in enumerate(ris_keys):
        total_from_P = len(channel_matrix.data.get(ris_key, {}))
        print(f"  Channels from P{i+1}: {total_from_P}")
    total_channels = sum(len(dests) for dests in channel_matrix.data.values())
    print(f"  Total channels: {total_channels}")
    print(f"{'='*60}")

    return channel_matrix

def main():
    """Main function to compute all channel matrices."""
    # Parse CLI arguments and update sionna_globals BEFORE any usage
    args = parse_args()
    update_globals_from_args(args)
    
    # Compute output directory with parameters in name
    output_dir = (
        f"heatmap/channel_matrices_K{sionna_globals['K']}_N{sionna_globals['N']}"
        f"_depth{sionna_globals['max_depth']}_los{sionna_globals['los']}"
    )
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("SIONNA CHANNEL MATRIX COMPUTATION")
    print("="*60)
    print(f"Parameters:")
    print(f"  K (antennas): {sionna_globals['K']}")
    print(f"  N (RIS elements): {sionna_globals['N']}")
    print(f"  max_depth: {sionna_globals['max_depth']}")
    print(f"  los: {sionna_globals['los']}")
    print(f"  specular_reflection: {sionna_globals['specular_reflection']}")
    print(f"  diffuse_reflection: {sionna_globals['diffuse_reflection']}")
    print(f"  refraction: {sionna_globals['refraction']}")
    print(f"  seed: {sionna_globals['seed']}")
    print(f"  Output directory: {output_dir}")
    print(f"  File pattern: sionna_channels_<situation>_K{sionna_globals['K']}_N{sionna_globals['N']}_...npz")
    print("="*60)

    processed_scenarios = []

    for situation in situations:
        simulation_name = situation['simulation_name']
        if not situation['calculate']:
            print(f"\nSkipping {simulation_name} (calculate=False)")
            continue

        channel_matrix = compute_channels_for_scenario(situation)
        if channel_matrix is not None:
            # Save individual file for this situation with parameters in filename
            individual_file = os.path.join(
                output_dir,
                f"sionna_channels_{simulation_name}_K{sionna_globals['K']}_N{sionna_globals['N']}"
                f"_depth{sionna_globals['max_depth']}_los{sionna_globals['los']}.npz"
            )

            print(f"\n{'='*60}")
            print(f"Saving results for {simulation_name}...")
            print(f"{'='*60}")

            np.savez(
                individual_file,
                channel_matrix=np.array([channel_matrix], dtype=object)
            )

            file_size_mb = os.path.getsize(individual_file) / (1024 * 1024)
            print(f"✓ Saved to {individual_file}")
            print(f"  File size: {file_size_mb:.2f} MB")
            processed_scenarios.append(simulation_name)

    print(f"\n{'='*60}")
    if processed_scenarios:
        print(f"✓ Successfully processed {len(processed_scenarios)} scenario(s):")
        for name in processed_scenarios:
            print(f"  - {name}")
        total_size_mb = sum(
            os.path.getsize(
                os.path.join(
                    output_dir,
                    f"sionna_channels_{name}_K{sionna_globals['K']}_N{sionna_globals['N']}"
                    f"_depth{sionna_globals['max_depth']}_los{sionna_globals['los']}.npz"
                )
            ) / (1024 * 1024)
            for name in processed_scenarios
        )
        print(f"  Total file size: {total_size_mb:.2f} MB")
    else:
        print("⚠ No scenarios were processed")
    print("="*60)
    print("Complete!")

if __name__ == "__main__":
    main()
