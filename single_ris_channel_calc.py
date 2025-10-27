import numpy as np
import tensorflow as tf
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera, PathSolver, mi
import time
from typing import Tuple, TypedDict
from diagonalization import calculate_ris_reflection_matrice, verify_matrix_is_diagonal, print_effective_channel
import gc
import drjit as dr
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # proviamo ad usare direttamente la cpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # per evitare i messaggi di warning continui  
tf.config.set_visible_devices([], 'GPU')  

dr.set_expand_threshold(1024 * 1024 * 1024)  # setta la memoria massima che Dr.Jit può usare, in questo caso 1GB

# proviamo così
try:
    dr.set_backend('cpu')
    print("Dr.Jit backend set to CPU")
except:
    print("Could not set Dr.Jit backend to CPU, continuing...")

# no_preview = True

class Actor(TypedDict):
    name: str
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float]
    rows: int
    cols: int

def compute_channel_matrix(
    scene,
    my_cam,
    tx: Actor,
    rx: Actor,
    fallback_mode: bool = False
) -> np.ndarray:
    """
    Compute channel matrix between transmitter and receiver.

    Args:
        scene: Loaded Sionna scene object
        my_cam: Camera instance
        tx: Transmitter actor with name, position, orientation, rows, cols
        rx: Receiver actor with name, position, orientation, rows, cols

    Returns:
        Channel matrix H as numpy array
    """
    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"Computing channel matrix: {tx['name']} → {rx['name']}")
    print(f"Transmitter: {tx['name']} at {tx['position'][:2]} with {tx['rows']}x{tx['cols']} antennas")
    print(f"Receiver: {rx['name']} at {rx['position'][:2]} with {rx['rows']}x{rx['cols']} antennas")

    # fallback per semplificare l'array antenna
    if fallback_mode:
        tx_rows = min(tx['rows'], 1)
        tx_cols = min(tx['cols'], 1)
        rx_rows = min(rx['rows'], 1)
        rx_cols = min(rx['cols'], 1)
        print(f"FALLBACK MODE: Using reduced antenna arrays ({tx_rows}x{tx_cols} -> {rx_rows}x{rx_cols})")
    else:
        tx_rows, tx_cols = tx['rows'], tx['cols']
        rx_rows, rx_cols = rx['rows'], rx['cols']

    scene.tx_array = PlanarArray(
        num_rows=tx_rows,
        num_cols=tx_cols,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V"
    )

    scene.rx_array = PlanarArray(
        num_rows=rx_rows,
        num_cols=rx_cols,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V"
    )

    # Add transmitter and receiver with explicit orientations
    tx_pos = [float(x) for x in tx['position']]
    tx_orient = [float(x) for x in tx['orientation']]
    rx_pos = [float(x) for x in rx['position']]
    rx_orient = [float(x) for x in rx['orientation']]

    tx_obj = Transmitter(tx['name'], position=mi.Point3f(tx_pos), orientation=mi.Point3f(tx_orient))
    rx_obj = Receiver(rx['name'], position=mi.Point3f(rx_pos), orientation=mi.Point3f(rx_orient))
    scene.add(tx_obj)
    scene.add(rx_obj)

    # Create fresh PathSolver for each computation to avoid memory accumulation
    p_solver = PathSolver()

    # Compute paths
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

    # Skip rendering to save memory
    # if no_preview:
    #     scene.render(camera=my_cam, paths=paths, clip_at=20)
    # else:
    #     scene.preview(paths=paths, clip_at=20)

    # Get channel impulse response
    a, tau = paths.cir(normalize_delays=True, out_type="numpy")

    # Sum over paths to get channel matrix H
    h = tf.reduce_sum(a, axis=-2)
    h = tf.squeeze(h)
    h_numpy = h.numpy()

    elapsed_time = time.time() - start_time

    print(f"Channel matrix shape: {h_numpy.shape}")
    print(f"Channel power: {np.sum(np.abs(h_numpy)**2):.6e}")
    print("Channel Matrix:")
    print(h_numpy)
    h_amplified = h_numpy / np.linalg.norm(h_numpy)
    print(f"Channel power: {np.sum(np.abs(h_amplified)**2):.6e}")
    print("Channel Matrix:")
    print(h_amplified)
    print(f"Computation time: {elapsed_time:.2f} seconds")
    print(f"{'='*60}")

    # Remove actors from scene before cleanup
    scene.remove(tx['name'])
    scene.remove(rx['name'])

    # Aggressive memory cleanup (delete PathSolver to force Dr.Jit to free GPU memory)
    del paths, a, tau, h, tx_obj, rx_obj, p_solver

    # Force synchronization - wait for GPU operations to complete
    dr.sync_thread()

    # cancella la sessione di tensorflow?? può liberare spazio
    tf.keras.backend.clear_session()
    dr.flush_malloc_cache()
    gc.collect()

    # cancella il grafo di tensorflow?? può liberare spazio
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()
    for _ in range(3):
        gc.collect()
    
    # Additional wait to ensure async cleanup completes
    import time as time_module
    time_module.sleep(0.5) 

    return h_amplified


def main():
    scene_path = "mesh_scene/Single Reflection.xml"

    # Load scene once (reused for all channel computations)
    print("\n" + "="*60)
    print("Loading scene and initializing ray tracing...")
    print("="*60)
    scene = load_scene(scene_path)
    scene.frequency = 3.5e9

    # Create Camera once (reused for all computations)
    # PathSolver will be created fresh for each computation to avoid memory accumulation
    my_cam = Camera(position=mi.Point3f([10, 10, 30]), look_at=[10, 10, 0])
    print("Scene loaded successfully\n")

    # Configurable test point distances from receivers
    distance_U1_from_R1 = 0.05 # meters
    distance_U2_from_R2 = 5.00 # meters

    # Define positions from heatmap_v2.py "Single Reflection" scenario
    pos_T = (0, 0, 1.5)
    pos_P = (7, 9, 1.5)
    pos_R1 = (20, 11, 1.5)
    pos_R2 = (20, 9, 1.5)

    # Calculate test point positions based on distance from receivers toward P
    vec_R1_to_P = (pos_P[0] - pos_R1[0], pos_P[1] - pos_R1[1])
    dist_R1_to_P = np.sqrt(vec_R1_to_P[0]**2 + vec_R1_to_P[1]**2)
    unit_R1_to_P = (vec_R1_to_P[0] / dist_R1_to_P, vec_R1_to_P[1] / dist_R1_to_P)
    pos_U1 = (
        pos_R1[0] + unit_R1_to_P[0] * distance_U1_from_R1,
        pos_R1[1] + unit_R1_to_P[1] * distance_U1_from_R1,
        pos_R1[2]
    )

    vec_R2_to_P = (pos_P[0] - pos_R2[0], pos_P[1] - pos_R2[1])
    dist_R2_to_P = np.sqrt(vec_R2_to_P[0]**2 + vec_R2_to_P[1]**2)
    unit_R2_to_P = (vec_R2_to_P[0] / dist_R2_to_P, vec_R2_to_P[1] / dist_R2_to_P)
    pos_U2 = (
        pos_R2[0] + unit_R2_to_P[0] * distance_U2_from_R2,
        pos_R2[1] + unit_R2_to_P[1] * distance_U2_from_R2,
        pos_R2[2]
    )

    # Calculate orientations based on geometry
    # Transmitter T -> RIS P
    vec_T_to_P = (pos_P[0] - pos_T[0], pos_P[1] - pos_T[1])
    angle_T = float(np.degrees(np.arctan2(vec_T_to_P[1], vec_T_to_P[0])))
    orient_T = (0.0, angle_T, 0.0)

    # RIS P orientation for reflection
    # Incident ray from T: angle_T
    # Reflected rays to R1 and R2
    vec_P_to_R1 = (pos_R1[0] - pos_P[0], pos_R1[1] - pos_P[1])
    vec_P_to_R2 = (pos_R2[0] - pos_P[0], pos_R2[1] - pos_P[1])
    angle_P_to_R1 = float(np.degrees(np.arctan2(vec_P_to_R1[1], vec_P_to_R1[0])))
    angle_P_to_R2 = float(np.degrees(np.arctan2(vec_P_to_R2[1], vec_P_to_R2[0])))

    # Average outgoing angle
    avg_outgoing = (angle_P_to_R1 + angle_P_to_R2) / 2
    # Incident angle from opposite direction
    incident_angle = angle_T + 180
    # Surface normal bisects incident and reflected
    bisector = (incident_angle + avg_outgoing) / 2
    # Surface orientation is perpendicular to normal
    orient_P = (0.0, bisector - 90, 0.0)

    # Receivers face RIS
    vec_R1_to_P = (pos_P[0] - pos_R1[0], pos_P[1] - pos_R1[1])
    angle_R1 = float(np.degrees(np.arctan2(vec_R1_to_P[1], vec_R1_to_P[0])))
    orient_R1 = (0.0, angle_R1, 0.0)

    vec_R2_to_P = (pos_P[0] - pos_R2[0], pos_P[1] - pos_R2[1])
    angle_R2 = float(np.degrees(np.arctan2(vec_R2_to_P[1], vec_R2_to_P[0])))
    orient_R2 = (0.0, angle_R2, 0.0)

    # U1 and U2 have approximately same orientations as R1 and R2
    orient_U1 = orient_R1
    orient_U2 = orient_R2

    # Antenna configurations
    K = 2  # Transmitter/Receiver antennas (2x1)
    N = 16  # RIS elements (4x4)
    eta = 0.9  # Reflection efficiency

    print("\n" + "="*60)
    print("SINGLE RIS REFLECTION - CHANNEL MATRIX COMPUTATION")
    print("="*60)
    print(f"Parameters:")
    print(f"  K (antennas): {K} (configured as 2x1)")
    print(f"  N (RIS elements): {N} (configured as 4x4)")
    print(f"  eta (reflection efficiency): {eta}")
    print(f"\nPositions:")
    print(f"  T (Transmitter): {pos_T[:2]}")
    print(f"  P (RIS): {pos_P[:2]}")
    print(f"  R1 (Receiver 1): {pos_R1[:2]}")
    print(f"  R2 (Receiver 2): {pos_R2[:2]}")
    print(f"  U1 (Test point, {distance_U1_from_R1}m from R1): {pos_U1[:2]}")
    print(f"  U2 (Test point, {distance_U2_from_R2}m from R2): {pos_U2[:2]}")
    print(f"\nOrientations (roll, yaw, pitch in degrees):")
    print(f"  T: {orient_T}")
    print(f"  P: {orient_P}")
    print(f"  R1: {orient_R1}")
    print(f"  R2: {orient_R2}")
    print(f"  U1: {orient_U1}")
    print(f"  U2: {orient_U2}")

    # Create Actor objects
    actor_T: Actor = {
        'name': 'T',
        'position': pos_T,
        'orientation': orient_T,
        'rows': 2,
        'cols': 1
    }

    actor_P: Actor = {
        'name': 'P',
        'position': pos_P,
        'orientation': orient_P,
        'rows': 4,
        'cols': 4
    }

    actor_R1: Actor = {
        'name': 'R1',
        'position': pos_R1,
        'orientation': orient_R1,
        'rows': 2,
        'cols': 1
    }

    actor_R2: Actor = {
        'name': 'R2',
        'position': pos_R2,
        'orientation': orient_R2,
        'rows': 2,
        'cols': 1
    }

    actor_U1: Actor = {
        'name': 'U1',
        'position': pos_U1,
        'orientation': orient_U1,
        'rows': 2,
        'cols': 1
    }

    actor_U2: Actor = {
        'name': 'U2',
        'position': pos_U2,
        'orientation': orient_U2,
        'rows': 2,
        'cols': 1
    }

    # Dictionary to store channel matrices
    channel_matrices = {}

    #  resettiamo la scena per ogni calcolo
    print("Computing T→P channel...")
    channel_matrices["T-P"] = compute_channel_matrix(
        scene, my_cam, actor_T, actor_P
    )
    print("Resetting scene for next computation...")
    del scene
    gc.collect()
    scene = load_scene(scene_path)
    scene.frequency = 3.5e9
    print("Computing P→R1 channel...")
    channel_matrices["P-R1"] = compute_channel_matrix(
        scene, my_cam, actor_P, actor_R1
    )
    print("Resetting scene for next computation...")
    del scene
    gc.collect()
    scene = load_scene(scene_path)
    scene.frequency = 3.5e9
    print("Computing P→R2 channel...")
    channel_matrices["P-R2"] = compute_channel_matrix(
        scene, my_cam, actor_P, actor_R2
    )
    print("Resetting scene for next computation...")
    del scene
    gc.collect()
    scene = load_scene(scene_path)
    scene.frequency = 3.5e9
    print("Computing P→U1 channel...")
    channel_matrices["P-U1"] = compute_channel_matrix(
        scene, my_cam, actor_P, actor_U1
    )
    print("Resetting scene for next computation...")
    del scene
    gc.collect()
    scene = load_scene(scene_path)
    scene.frequency = 3.5e9
    print("Computing P→U2 channel...")
    channel_matrices["P-U2"] = compute_channel_matrix(
        scene, my_cam, actor_P, actor_U2
    )

    # Save all channel matrices to NPZ format
    output_file = "single_ris_channel_matrices.npz"
    np.savez(
        output_file,
        **{key: value for key, value in channel_matrices.items()}
    )

    print("\n" + "="*60)
    print(f"Channel matrices saved to {output_file}")
    print(f"Keys: {list(channel_matrices.keys())}")
    print(f"Load with: data = np.load('{output_file}')")
    print("="*60)

    # Now calculate RIS configuration using only R1 and R2
    print("\n" + "="*60)
    print("CALCULATING RIS CONFIGURATION (using R1 and R2 only)")
    print("="*60)

    H = channel_matrices["T-P"]
    G_R1 = channel_matrices["P-R1"]
    G_R2 = channel_matrices["P-R2"]

    Gs = [G_R1, G_R2]
    J = 2  # Only R1 and R2 used for RIS configuration

    P, dor = calculate_ris_reflection_matrice(K, N, J, Gs, H, eta)

    print(f"RIS configuration calculated successfully")
    print(f"Degree of Randomness (DoR): {dor}")

    # Verify RIS configuration for all 4 receivers
    print("\n" + "="*60)
    print("VERIFICATION: Testing RIS configuration on all receivers")
    print("="*60)

    receivers = [
        ("R1", G_R1, "Receiver 1 (used for RIS config)"),
        ("R2", G_R2, "Receiver 2 (used for RIS config)"),
        ("U1", channel_matrices["P-U1"], f"Test point U1 ({distance_U1_from_R1}m from R1)"),
        ("U2", channel_matrices["P-U2"], f"Test point U2 ({distance_U2_from_R2}m from R2)")
    ]

    results = []
    for name, G, description in receivers:
        effective_channel = G @ P @ H
        is_diagonal = verify_matrix_is_diagonal(effective_channel)
        results.append((name, is_diagonal, effective_channel))

        print(f"\n{description}:")
        print(f"  Is diagonal: {is_diagonal}")
        print(f"  Effective channel |GPH|:")
        print(G @ P @ H)

        # Calculate off-diagonal to diagonal ratio
        diag_power = np.sum(np.abs(np.diag(effective_channel))**2)
        total_power = np.sum(np.abs(effective_channel)**2)
        off_diag_power = total_power - diag_power
        if diag_power > 0:
            ratio = off_diag_power / diag_power
            print(f"  Off-diagonal/Diagonal power ratio: {ratio:.6f}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    diagonal_count = sum(1 for _, is_diag, _ in results if is_diag)
    print(f"Diagonalization successful for {diagonal_count}/4 receivers")
    print(f"  R1 (intended): {'✓' if results[0][1] else '✗'}")
    print(f"  R2 (intended): {'✓' if results[1][1] else '✗'}")
    print(f"  U1 ({distance_U1_from_R1}m away): {'✓' if results[2][1] else '✗'}")
    print(f"  U2 ({distance_U2_from_R2}m away): {'✓' if results[3][1] else '✗'}")

    print("\nThis demonstrates the 'area of understanding' around intended receivers")
    print("mentioned in the paper regarding mobility challenges.")
    print("="*60)

if __name__ == "__main__":
    main()