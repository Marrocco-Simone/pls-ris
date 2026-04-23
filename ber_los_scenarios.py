"""
BER vs Transmitter Power for LOS/NLOS Scenarios

This script simulates BER vs transmitter power for a single scenario with 4 actors:
- B1: NLOS receiver (RIS-only)
- B2: LOS receiver (direct + RIS) - tested with BOTH detection methods
- E1: NLOS eavesdropper (RIS-only)
- E2: LOS eavesdropper (direct + RIS)

The RIS reflection matrix P is computed via diagonalization for both B1 and B2.

Research question: Which detection method works for LOS receiver B2?
- B2-Direct: Uses direct detection (reference: B_b)
- B2-Reflection: Uses reflection detection on (B_b + G2PH)

Generates two graphs:
- BER_vs_Pt_Passive.pdf (5 curves for passive RIS)
- BER_vs_Pt_Active.pdf (5 curves for active RIS)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from tqdm import tqdm

from diagonalization import (
    generate_random_channel_matrix,
    calculate_multi_ris_reflection_matrices,
    unify_ris_reflection_matrices,
)
from noise_power_utils import (
    calculate_noise_floor_in_mw,
    create_random_noise_vector_from_noise_floor,
)
from ber import (
    simulate_ssk_transmission_reflection,
    simulate_ssk_transmission_direct,
    calculate_confidence_interval,
    configure_latex,
)
from heatmap_utils import calculate_free_space_path_loss


###############################################################################
# Constants
###############################################################################

N_ELEMENTS = 36      # RIS elements (matched to heatmap.py)
K_ANTENNAS = 4     # TX/RX antennas (matched to heatmap.py)
J_RECEIVERS = 2      # Two Bobs (B1 and B2) for diagonalization
M_SURFACES = 1       # One RIS
ETA = 0.9            # Reflection efficiency
DISTANCE_M = 10      # 10 meter paths
N_SIMULATIONS = 10000  # Per power point
PT_RANGE_DBM = np.arange(-120, 82, 5)  # Transmitter power range

PATH_LOSS = calculate_free_space_path_loss(DISTANCE_M)

# Output directories
OUTPUT_DIR = "./ber_los"
PDF_DIR = os.path.join(OUTPUT_DIR, "pdf")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")


###############################################################################
# Core Simulation Functions
###############################################################################

def run_single_ber_measurement(
    K: int,
    effective_channel: np.ndarray,
    B: np.ndarray,
    noise: np.ndarray,
    Pt_dbm: float,
    use_reflection: bool
) -> float:
    """
    Run a single BER measurement.
    
    Args:
        K: Number of antennas
        effective_channel: Effective channel matrix
        B: Direct link matrix (for direct detection)
        noise: Noise vector
        Pt_dbm: Transmit power in dBm
        use_reflection: If True, use reflection detection; else direct detection
    
    Returns:
        Bit error rate for this transmission
    """
    if use_reflection:
        return simulate_ssk_transmission_reflection(K, effective_channel, noise, Pt_dbm)
    else:
        return simulate_ssk_transmission_direct(K, B, effective_channel, noise, Pt_dbm)


def run_single_simulation_pass(
    Pt_dbm: float,
    N0_mw: float,
    seed: int
) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    """
    Run N_SIMULATIONS iterations for a single power point.
    Computes BER with 95% confidence intervals for all 5 lines for both passive and active RIS.
    
    Args:
        Pt_dbm: Transmitter power in dBm
        N0_mw: Noise floor in mW
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with BER results and confidence intervals:
        {'passive': {'B1': (mean, lower, upper), 'B2_Direct': (mean, lower, upper), ...},
         'active': {...}}
    """
    np.random.seed(seed)
    
    # Initialize lists to store individual BER measurements
    # Two B2 curves: SSK (reflection) detection vs Direct ML detection
    ber_measurements = {
        'passive': {'B1': [], 'B2_SSK': [], 'B2_Direct': [], 'B3': [], 'E1': [], 'E2': []},
        'active': {'B1': [], 'B2_SSK': [], 'B2_Direct': [], 'B3': [], 'E1': [], 'E2': []}
    }
    
    for _ in range(N_SIMULATIONS):
        # Generate fresh random channels
        H = PATH_LOSS * generate_random_channel_matrix(N_ELEMENTS, K_ANTENNAS)  # Tx→RIS
        G1 = PATH_LOSS * generate_random_channel_matrix(K_ANTENNAS, N_ELEMENTS)  # RIS→B1
        G2 = PATH_LOSS * generate_random_channel_matrix(K_ANTENNAS, N_ELEMENTS)  # RIS→B2
        F1 = PATH_LOSS * generate_random_channel_matrix(K_ANTENNAS, N_ELEMENTS)  # RIS→E1
        F2 = PATH_LOSS * generate_random_channel_matrix(K_ANTENNAS, N_ELEMENTS)  # RIS→E2
        Bb = PATH_LOSS * generate_random_channel_matrix(K_ANTENNAS, K_ANTENNAS)  # Tx→B2 direct
        Bb3 = PATH_LOSS * generate_random_channel_matrix(K_ANTENNAS, K_ANTENNAS)  # Tx→B3 direct (no RIS)
        Be = PATH_LOSS * generate_random_channel_matrix(K_ANTENNAS, K_ANTENNAS)  # Tx→E2 direct
        
        # Compute RIS reflection matrix P using both B1 and B2
        Gs = [G1, G2]
        Cs = []  # No inter-RIS channels for single RIS
        Ps, _ = calculate_multi_ris_reflection_matrices(
            K_ANTENNAS, N_ELEMENTS, J_RECEIVERS, M_SURFACES, Gs, H, ETA, Cs
        )
        P = unify_ris_reflection_matrices(Ps, Cs)

        # Channels are NOT scaled here - power scaling is handled inside
        # simulate_ssk_transmission() which scales x by sqrt(Pt)

        # For passive RIS: use channels as-is (with path loss already applied)
        eff_B1_passive = G1 @ P @ H
        eff_B2_passive = G2 @ P @ H
        eff_E1_passive = F1 @ P @ H
        eff_E2_passive = F2 @ P @ H

        # For active RIS: normalize H to remove path loss effect on RIS input
        H_active = H / np.max(np.abs(H))
        eff_B1_active = G1 @ P @ H_active
        eff_B2_active = G2 @ P @ H_active
        eff_E1_active = F1 @ P @ H_active
        eff_E2_active = F2 @ P @ H_active

        # Generate noise (same noise for all to ensure fair comparison)
        # FIX P0: Use correct noise floor API (no second argument for power)
        noise_B1 = create_random_noise_vector_from_noise_floor(K_ANTENNAS)
        noise_B2 = create_random_noise_vector_from_noise_floor(K_ANTENNAS)
        noise_B3 = create_random_noise_vector_from_noise_floor(K_ANTENNAS)
        noise_E1 = create_random_noise_vector_from_noise_floor(K_ANTENNAS)
        noise_E2 = create_random_noise_vector_from_noise_floor(K_ANTENNAS)

        # PASSIVE RIS measurements
        # L1: B1 (NLOS, reflection)
        ber_measurements['passive']['B1'].append(run_single_ber_measurement(
            K_ANTENNAS, eff_B1_passive, np.zeros((K_ANTENNAS, K_ANTENNAS)),
            noise_B1, Pt_dbm, use_reflection=True
        ))

        # L2: B2 with SSK/reflection detection (argmax|y|^2 on total channel B+GPH)
        # This will fail because (B+GPH) is not diagonal
        total_B2_passive = Bb + eff_B2_passive
        ber_measurements['passive']['B2_SSK'].append(
            simulate_ssk_transmission_reflection(K_ANTENNAS, total_B2_passive, noise_B2, Pt_dbm)
        )

        # L3: B2 with Direct ML detection (minimum distance on total channel B+GPH)
        # This works because receiver knows both B and GPH
        ber_measurements['passive']['B2_Direct'].append(
            simulate_ssk_transmission_direct(
                K_ANTENNAS, Bb, eff_B2_passive, noise_B2, Pt_dbm
            )
        )

        # L4: E1 (NLOS, reflection)
        ber_measurements['passive']['E1'].append(run_single_ber_measurement(
            K_ANTENNAS, eff_E1_passive, np.zeros((K_ANTENNAS, K_ANTENNAS)),
            noise_E1, Pt_dbm, use_reflection=True
        ))

        # L5: E2 (LOS, direct detection)
        ber_measurements['passive']['E2'].append(run_single_ber_measurement(
            K_ANTENNAS, eff_E2_passive, Be,
            noise_E2, Pt_dbm, use_reflection=False
        ))

        # L6: B3 (Direct-only, no RIS path) - baseline comparison
        # B3 has direct link only, no G3PH contribution (G3 = 0)
        ber_measurements['passive']['B3'].append(run_single_ber_measurement(
            K_ANTENNAS, np.zeros((K_ANTENNAS, K_ANTENNAS)), Bb3,
            noise_B3, Pt_dbm, use_reflection=False
        ))

        # ACTIVE RIS measurements (same structure)
        # L1: B1 (NLOS, reflection)
        ber_measurements['active']['B1'].append(run_single_ber_measurement(
            K_ANTENNAS, eff_B1_active, np.zeros((K_ANTENNAS, K_ANTENNAS)),
            noise_B1, Pt_dbm, use_reflection=True
        ))

        # L2: B2 with SSK/reflection detection (argmax|y|^2 on total channel)
        total_B2_active = Bb + eff_B2_active
        ber_measurements['active']['B2_SSK'].append(
            simulate_ssk_transmission_reflection(K_ANTENNAS, total_B2_active, noise_B2, Pt_dbm)
        )

        # L3: B2 with Direct ML detection (minimum distance on total channel)
        ber_measurements['active']['B2_Direct'].append(
            simulate_ssk_transmission_direct(
                K_ANTENNAS, Bb, eff_B2_active, noise_B2, Pt_dbm
            )
        )

        # L4: E1 (NLOS, reflection)
        ber_measurements['active']['E1'].append(run_single_ber_measurement(
            K_ANTENNAS, eff_E1_active, np.zeros((K_ANTENNAS, K_ANTENNAS)),
            noise_E1, Pt_dbm, use_reflection=True
        ))

        # L5: E2 (LOS, direct detection)
        ber_measurements['active']['E2'].append(run_single_ber_measurement(
            K_ANTENNAS, eff_E2_active, Be,
            noise_E2, Pt_dbm, use_reflection=False
        ))

        # L6: B3 (Direct-only, no RIS path) - baseline comparison
        # B3 has direct link only, no G3PH contribution (G3 = 0)
        # Active RIS doesn't affect B3 since G3 = 0
        ber_measurements['active']['B3'].append(run_single_ber_measurement(
            K_ANTENNAS, np.zeros((K_ANTENNAS, K_ANTENNAS)), Bb3,
            noise_B3, Pt_dbm, use_reflection=False
        ))
    
    # Compute BER with confidence intervals
    ber_results = {
        'passive': {},
        'active': {}
    }
    
    for ris_mode in ['passive', 'active']:
        for key in ber_measurements[ris_mode]:
            mean, lower, upper = calculate_confidence_interval(ber_measurements[ris_mode][key])
            ber_results[ris_mode][key] = (mean, lower, upper)
    
    return ber_results


def run_all_simulations() -> Dict:
    """
    Run simulations across all power points.
    
    Returns:
        Dictionary with BER results for all power points
    """
    # Precompute noise floor
    N0_mw = calculate_noise_floor_in_mw()
    N0_dbm = 10 * np.log10(N0_mw)
    
    print(f"Parameters: N={N_ELEMENTS}, K={K_ANTENNAS}, J={J_RECEIVERS}, M={M_SURFACES}")
    print(f"Distance: {DISTANCE_M}m, Path Loss: {PATH_LOSS:.4f}")
    print(f"Noise floor: {N0_dbm:.1f} dBm = {N0_mw:.2e} mW")
    print(f"Simulations per power point: {N_SIMULATIONS}")
    print(f"Power range: {PT_RANGE_DBM[0]} to {PT_RANGE_DBM[-1]} dBm")
    print()
    
    # Initialize results storage - now with separate arrays for mean, lower, upper
    # Two B2 curves: SSK detection vs Direct ML detection
    results = {
        'Pt_range_dbm': PT_RANGE_DBM,
        'passive': {
            key: {'mean': [], 'lower': [], 'upper': []} 
            for key in ['B1', 'B2_SSK', 'B2_Direct', 'B3', 'E1', 'E2']
        },
        'active': {
            key: {'mean': [], 'lower': [], 'upper': []} 
            for key in ['B1', 'B2_SSK', 'B2_Direct', 'B3', 'E1', 'E2']
        }
    }
    
    # Run simulations for each power point
    for idx, Pt_dbm in enumerate(tqdm(PT_RANGE_DBM, desc="Processing power points")):
        # Use deterministic seed based on power index for reproducibility
        seed = idx * 1000
        
        ber_results = run_single_simulation_pass(Pt_dbm, N0_mw, seed)
        
        # Store results - ber_results[ris_mode][key] is now a tuple (mean, lower, upper)
        for ris_mode in ['passive', 'active']:
            for key in ber_results[ris_mode]:
                mean, lower, upper = ber_results[ris_mode][key]
                results[ris_mode][key]['mean'].append(mean)
                results[ris_mode][key]['lower'].append(lower)
                results[ris_mode][key]['upper'].append(upper)
    
    return results


def save_results(results: Dict, filename: str):
    """Save simulation results with confidence intervals to numpy file"""
    data_to_save = {'Pt_range_dbm': results['Pt_range_dbm']}
    
    # Two B2 curves: SSK detection vs Direct ML detection
    for mode in ['passive', 'active']:
        for key in ['B1', 'B2_SSK', 'B2_Direct', 'B3', 'E1', 'E2']:
            data_to_save[f'{mode}_{key}_mean'] = np.array(results[mode][key]['mean'])
            data_to_save[f'{mode}_{key}_lower'] = np.array(results[mode][key]['lower'])
            data_to_save[f'{mode}_{key}_upper'] = np.array(results[mode][key]['upper'])
    
    np.savez(filename, **data_to_save)
    print(f"Results saved to {filename}")


def load_results(filename: str) -> Dict:
    """Load simulation results with confidence intervals from numpy file"""
    try:
        data = np.load(filename)
        results = {
            'Pt_range_dbm': data['Pt_range_dbm'],
            'passive': {},
            'active': {}
        }
        # Two B2 curves: SSK detection vs Direct ML detection
        for mode in ['passive', 'active']:
            for key in ['B1', 'B2_SSK', 'B2_Direct', 'B3', 'E1', 'E2']:
                results[mode][key] = {
                    'mean': data[f'{mode}_{key}_mean'],
                    'lower': data[f'{mode}_{key}_lower'],
                    'upper': data[f'{mode}_{key}_upper']
                }
        return results
    except (FileNotFoundError, IOError, KeyError):
        # Return None on any load error (including missing keys for old format)
        return None


def compute_antenna_power_profile(
    K: int,
    B: np.ndarray,
    effective_channel: np.ndarray,
    Pt_linear: float
) -> np.ndarray:
    """
    Compute received power at each antenna for each transmitted symbol.

    For SSK with K antennas, we transmit one antenna at a time.
    The transmitted signal is scaled by sqrt(Pt), so received power is Pt × |channel|².
    Returns a K×K matrix where element (i,j) is the power |y_i|² when antenna j is transmitted.

    Args:
        K: Number of antennas
        B: Direct link matrix (K×K) - unscaled channel gains
        effective_channel: RIS channel GPH (K×K) - unscaled channel gains
        Pt_linear: Transmit power in linear units (mW)

    Returns:
        K×K matrix of received powers in linear units (mW)
    """
    total_channel = B + effective_channel  # (B + GPH)
    sqrt_Pt = np.sqrt(Pt_linear)

    power_matrix = np.zeros((K, K))

    for tx_antenna in range(K):
        # Create transmitted signal (only one antenna active, scaled by sqrt(Pt))
        x = np.zeros(K)
        x[tx_antenna] = sqrt_Pt

        # Received signal (no noise for power analysis)
        y = total_channel @ x

        # Store power at each receive antenna
        for rx_antenna in range(K):
            power_matrix[rx_antenna, tx_antenna] = np.abs(y[rx_antenna])**2

    return power_matrix


def run_antenna_power_simulations() -> Tuple[Dict, Dict]:
    """
    Run simulations to compute antenna power profiles for B1 and B2 across all power points.
    
    Returns:
        Tuple of (results_B1, results_B2), each containing power profiles for each antenna
    """
    print(f"Running antenna power profile simulations...")
    print(f"Parameters: N={N_ELEMENTS}, K={K_ANTENNAS}, J={J_RECEIVERS}, M={M_SURFACES}")
    print(f"Distance: {DISTANCE_M}m, Path Loss: {PATH_LOSS:.4f}")
    print()
    
    # Initialize results storage for B1, B2, and B3
    results_B1 = {
        'Pt_range_dbm': PT_RANGE_DBM,
        'passive': [],  # List of K×K matrices, one per power point
        'active': []    # List of K×K matrices, one per power point
    }
    
    results_B2 = {
        'Pt_range_dbm': PT_RANGE_DBM,
        'passive': [],  # List of K×K matrices, one per power point (total power)
        'active': [],    # List of K×K matrices, one per power point (total power)
        'passive_B_only': [],  # Direct link only (B)
        'active_B_only': [],   # Direct link only (B)
        'passive_GPH_only': [], # RIS link only (GPH)
        'active_GPH_only': []  # RIS link only (GPH)
    }
    
    results_B3 = {
        'Pt_range_dbm': PT_RANGE_DBM,
        'passive': [],  # List of K×K matrices, one per power point
        'active': []    # List of K×K matrices, one per power point
    }
    
    # Run simulations for each power point (single iteration since we want channel realization, not noise)
    for Pt_dbm in tqdm(PT_RANGE_DBM, desc="Processing power points"):
        # Generate fresh random channels (unscaled - path loss only)
        H = PATH_LOSS * generate_random_channel_matrix(N_ELEMENTS, K_ANTENNAS)
        G1 = PATH_LOSS * generate_random_channel_matrix(K_ANTENNAS, N_ELEMENTS)
        G2 = PATH_LOSS * generate_random_channel_matrix(K_ANTENNAS, N_ELEMENTS)
        Bb = PATH_LOSS * generate_random_channel_matrix(K_ANTENNAS, K_ANTENNAS)
        Bb3 = PATH_LOSS * generate_random_channel_matrix(K_ANTENNAS, K_ANTENNAS)  # B3 direct link

        # Compute RIS reflection matrix P using both B1 and B2
        Gs = [G1, G2]
        Cs = []
        Ps, _ = calculate_multi_ris_reflection_matrices(
            K_ANTENNAS, N_ELEMENTS, J_RECEIVERS, M_SURFACES, Gs, H, ETA, Cs
        )
        P = unify_ris_reflection_matrices(Ps, Cs)

        # Transmit power in linear units (mW)
        Pt_linear = 10 ** (Pt_dbm / 10)

        # Passive RIS: channels are unscaled, power scaling happens in compute_antenna_power_profile
        eff_B1_passive = G1 @ P @ H
        power_profile_B1_passive = compute_antenna_power_profile(
            K_ANTENNAS, np.zeros((K_ANTENNAS, K_ANTENNAS)), eff_B1_passive, Pt_linear
        )
        results_B1['passive'].append(power_profile_B1_passive)

        # B2: Direct + RIS (total power)
        eff_B2_passive = G2 @ P @ H
        power_profile_B2_passive = compute_antenna_power_profile(
            K_ANTENNAS, Bb, eff_B2_passive, Pt_linear
        )
        results_B2['passive'].append(power_profile_B2_passive)
        
        # B2: Direct link only (B only)
        power_profile_B2_B_only_passive = compute_antenna_power_profile(
            K_ANTENNAS, Bb, np.zeros((K_ANTENNAS, K_ANTENNAS)), Pt_linear
        )
        results_B2['passive_B_only'].append(power_profile_B2_B_only_passive)
        
        # B2: RIS link only (GPH only)
        power_profile_B2_GPH_only_passive = compute_antenna_power_profile(
            K_ANTENNAS, np.zeros((K_ANTENNAS, K_ANTENNAS)), eff_B2_passive, Pt_linear
        )
        results_B2['passive_GPH_only'].append(power_profile_B2_GPH_only_passive)

        # B3: Direct only, no RIS path (G3 = 0)
        power_profile_B3_passive = compute_antenna_power_profile(
            K_ANTENNAS, Bb3, np.zeros((K_ANTENNAS, K_ANTENNAS)), Pt_linear
        )
        results_B3['passive'].append(power_profile_B3_passive)

        # Active RIS: normalize H to remove path loss effect on RIS input
        H_active = H / np.max(np.abs(H))
        eff_B1_active = G1 @ P @ H_active
        power_profile_B1_active = compute_antenna_power_profile(
            K_ANTENNAS, np.zeros((K_ANTENNAS, K_ANTENNAS)), eff_B1_active, Pt_linear
        )
        results_B1['active'].append(power_profile_B1_active)

        # B2: Direct + RIS (active) (total power)
        eff_B2_active = G2 @ P @ H_active
        power_profile_B2_active = compute_antenna_power_profile(
            K_ANTENNAS, Bb, eff_B2_active, Pt_linear
        )
        results_B2['active'].append(power_profile_B2_active)
        
        # B2: Direct link only (B only) - same as passive since B is not affected by RIS active mode
        power_profile_B2_B_only_active = compute_antenna_power_profile(
            K_ANTENNAS, Bb, np.zeros((K_ANTENNAS, K_ANTENNAS)), Pt_linear
        )
        results_B2['active_B_only'].append(power_profile_B2_B_only_active)
        
        # B2: RIS link only (GPH only) (active)
        power_profile_B2_GPH_only_active = compute_antenna_power_profile(
            K_ANTENNAS, np.zeros((K_ANTENNAS, K_ANTENNAS)), eff_B2_active, Pt_linear
        )
        results_B2['active_GPH_only'].append(power_profile_B2_GPH_only_active)

        # B3: Direct only, no RIS path (active doesn't affect B3 since G3 = 0)
        power_profile_B3_active = compute_antenna_power_profile(
            K_ANTENNAS, Bb3, np.zeros((K_ANTENNAS, K_ANTENNAS)), Pt_linear
        )
        results_B3['active'].append(power_profile_B3_active)
    
    return results_B1, results_B2, results_B3


def plot_antenna_power_profiles(power_results_B1: Dict, power_results_B2: Dict, power_results_B3: Dict):
    """
    Generate antenna power profile graphs for B1, B2, and B3 with consistent dBm scales.

    For each of K transmit antennas, we plot K lines showing the received power
    at each receive antenna vs transmit power. All subplots use the same y-axis scale.
    Includes a horizontal reference line at the noise floor.
    """
    configure_latex()
    fontsize = 14
    plt.rc('font', **{'size': fontsize})

    Pt_range = power_results_B2['Pt_range_dbm']
    K = K_ANTENNAS

    # Calculate noise floor in dBm for reference line
    N0_mw = calculate_noise_floor_in_mw()
    N0_dbm = 10 * np.log10(N0_mw)

    colors = plt.cm.tab10(np.linspace(0, 1, K))
    
    for ris_mode in ['passive', 'active']:
        # Convert power to dBm and find global min/max for consistent scaling
        power_key = f'{ris_mode}_power_profiles'
        
        # Get data for B1, B2, and B3
        data_B1 = np.array(power_results_B1[ris_mode])  # Shape: (num_Pt, K, K)
        data_B2 = np.array(power_results_B2[ris_mode])  # Shape: (num_Pt, K, K)
        data_B3 = np.array(power_results_B3[ris_mode])  # Shape: (num_Pt, K, K)
        
        # Convert to dBm (avoid log(0) by adding small epsilon)
        epsilon = 1e-20
        data_B1_dbm = 10 * np.log10(data_B1 + epsilon)
        data_B2_dbm = 10 * np.log10(data_B2 + epsilon)
        data_B3_dbm = 10 * np.log10(data_B3 + epsilon)
        
        # Find global min/max for consistent y-axis across all plots
        y_min = min(data_B1_dbm.min(), data_B2_dbm.min(), data_B3_dbm.min())
        y_max = max(data_B1_dbm.max(), data_B2_dbm.max(), data_B3_dbm.max())
        # Add some padding
        y_range = y_max - y_min
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range
        
        # ==================== B1 Plots ====================
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for tx_antenna in range(K):
            ax = axes[tx_antenna]
            
            # For each receive antenna, plot its power vs Pt
            for rx_antenna in range(K):
                powers_dbm = data_B1_dbm[:, rx_antenna, tx_antenna]
                
                # Line style: solid for diagonal (correct antenna), dashed for others
                linestyle = '-' if rx_antenna == tx_antenna else '--'
                linewidth = 2.5 if rx_antenna == tx_antenna else 1.5
                alpha = 1.0 if rx_antenna == tx_antenna else 0.6
                
                ax.plot(Pt_range, powers_dbm,
                       color=colors[rx_antenna],
                       linestyle=linestyle,
                       linewidth=linewidth,
                       alpha=alpha,
                       label=f'Antenna {rx_antenna}')
            
            # Add noise floor reference line
            ax.axhline(y=N0_dbm, color='gray', linestyle=':', linewidth=1.5,
                      label=f'Noise floor ({N0_dbm:.0f} dBm)')

            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Transmit Power (dBm)', fontsize=fontsize)
            ax.set_ylabel('Received Power (dBm)', fontsize=fontsize)
            ax.set_title(f'Transmitting from Antenna {tx_antenna}', fontsize=fontsize)
            ax.legend(fontsize=fontsize-2, loc='best')
            ax.tick_params(axis='both', labelsize=fontsize-2)
            # Set consistent y-axis limits
            ax.set_ylim(y_min, y_max)

        plt.suptitle(f'B$_1$ Received Power Profile - {ris_mode.capitalize()} RIS\n'
                    f'(N={N_ELEMENTS}, K={K_ANTENNAS}, J={J_RECEIVERS})',
                    fontsize=fontsize+2, y=1.02)
        plt.tight_layout()
        
        # Save figure
        filename = f'B1_Antenna_Power_Profile_{ris_mode.capitalize()}.pdf'
        filepath = os.path.join(PDF_DIR, filename)
        
        try:
            plt.savefig(filepath, dpi=300, format='pdf', bbox_inches='tight')
            print(f"Saved {filepath}")
        except RuntimeError as e:
            if "latex could not be found" in str(e):
                print(f"Warning: LaTeX not available, using default font for {filename}")
                plt.rcParams['text.usetex'] = False
                plt.savefig(filepath, dpi=300, format='pdf', bbox_inches='tight')
                print(f"Saved {filepath}")
            else:
                raise
        
        plt.close()
        
        # ==================== B2 Plots ====================
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for tx_antenna in range(K):
            ax = axes[tx_antenna]
            
            # For each receive antenna, plot its power vs Pt
            for rx_antenna in range(K):
                powers_dbm = data_B2_dbm[:, rx_antenna, tx_antenna]
                
                # Line style: solid for diagonal (correct antenna), dashed for others
                linestyle = '-' if rx_antenna == tx_antenna else '--'
                linewidth = 2.5 if rx_antenna == tx_antenna else 1.5
                alpha = 1.0 if rx_antenna == tx_antenna else 0.6
                
                ax.plot(Pt_range, powers_dbm,
                       color=colors[rx_antenna],
                       linestyle=linestyle,
                       linewidth=linewidth,
                       alpha=alpha,
                       label=f'Antenna {rx_antenna}')
            
            # Add noise floor reference line
            ax.axhline(y=N0_dbm, color='gray', linestyle=':', linewidth=1.5,
                      label=f'Noise floor ({N0_dbm:.0f} dBm)')

            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Transmit Power (dBm)', fontsize=fontsize)
            ax.set_ylabel('Received Power (dBm)', fontsize=fontsize)
            ax.set_title(f'Transmitting from Antenna {tx_antenna}', fontsize=fontsize)
            ax.legend(fontsize=fontsize-2, loc='best')
            ax.tick_params(axis='both', labelsize=fontsize-2)
            # Set consistent y-axis limits (same as B1)
            ax.set_ylim(y_min, y_max)

        plt.suptitle(f'B$_2$ Received Power Profile - {ris_mode.capitalize()} RIS\n'
                    f'(N={N_ELEMENTS}, K={K_ANTENNAS}, J={J_RECEIVERS})',
                    fontsize=fontsize+2, y=1.02)
        plt.tight_layout()
        
        # Save figure
        filename = f'B2_Antenna_Power_Profile_{ris_mode.capitalize()}.pdf'
        filepath = os.path.join(PDF_DIR, filename)
        
        try:
            plt.savefig(filepath, dpi=300, format='pdf', bbox_inches='tight')
            print(f"Saved {filepath}")
        except RuntimeError as e:
            if "latex could not be found" in str(e):
                print(f"Warning: LaTeX not available, using default font for {filename}")
                plt.rcParams['text.usetex'] = False
                plt.savefig(filepath, dpi=300, format='pdf', bbox_inches='tight')
                print(f"Saved {filepath}")
            else:
                raise
        
        plt.close()
        
        # ==================== B3 Plots ====================
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for tx_antenna in range(K):
            ax = axes[tx_antenna]
            
            # For each receive antenna, plot its power vs Pt
            for rx_antenna in range(K):
                powers_dbm = data_B3_dbm[:, rx_antenna, tx_antenna]
                
                # Line style: solid for diagonal (correct antenna), dashed for others
                linestyle = '-' if rx_antenna == tx_antenna else '--'
                linewidth = 2.5 if rx_antenna == tx_antenna else 1.5
                alpha = 1.0 if rx_antenna == tx_antenna else 0.6
                
                ax.plot(Pt_range, powers_dbm,
                       color=colors[rx_antenna],
                       linestyle=linestyle,
                       linewidth=linewidth,
                       alpha=alpha,
                       label=f'Antenna {rx_antenna}')
            
            # Add noise floor reference line
            ax.axhline(y=N0_dbm, color='gray', linestyle=':', linewidth=1.5,
                      label=f'Noise floor ({N0_dbm:.0f} dBm)')

            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Transmit Power (dBm)', fontsize=fontsize)
            ax.set_ylabel('Received Power (dBm)', fontsize=fontsize)
            ax.set_title(f'Transmitting from Antenna {tx_antenna}', fontsize=fontsize)
            ax.legend(fontsize=fontsize-2, loc='best')
            ax.tick_params(axis='both', labelsize=fontsize-2)
            # Set consistent y-axis limits (same as B1, B2)
            ax.set_ylim(y_min, y_max)

        plt.suptitle(f'B$_3$ Received Power Profile - {ris_mode.capitalize()} RIS (Direct Only)\n'
                    f'(N={N_ELEMENTS}, K={K_ANTENNAS}, J={J_RECEIVERS})',
                    fontsize=fontsize+2, y=1.02)
        plt.tight_layout()
        
        # Save figure
        filename = f'B3_Antenna_Power_Profile_{ris_mode.capitalize()}.pdf'
        filepath = os.path.join(PDF_DIR, filename)
        
        try:
            plt.savefig(filepath, dpi=300, format='pdf', bbox_inches='tight')
            print(f"Saved {filepath}")
        except RuntimeError as e:
            if "latex could not be found" in str(e):
                print(f"Warning: LaTeX not available, using default font for {filename}")
                plt.rcParams['text.usetex'] = False
                plt.savefig(filepath, dpi=300, format='pdf', bbox_inches='tight')
                print(f"Saved {filepath}")
            else:
                raise
        
        plt.close()


def plot_b2_separate_power_profiles(power_results_B2: Dict):
    """
    Generate separate antenna power profile graphs for B2 showing:
    - Direct link only (B)
    - RIS link only (GPH)
    
    This helps debug why B2 might perform differently than expected.
    """
    configure_latex()
    fontsize = 14
    plt.rc('font', **{'size': fontsize})
    
    Pt_range = power_results_B2['Pt_range_dbm']
    K = K_ANTENNAS
    
    # Calculate noise floor in dBm for reference line
    N0_mw = calculate_noise_floor_in_mw()
    N0_dbm = 10 * np.log10(N0_mw)
    
    colors = plt.cm.tab10(np.linspace(0, 1, K))
    
    for ris_mode in ['passive', 'active']:
        # Get data for B-only and GPH-only
        data_B_only = np.array(power_results_B2[f'{ris_mode}_B_only'])
        data_GPH_only = np.array(power_results_B2[f'{ris_mode}_GPH_only'])
        
        # Convert to dBm
        epsilon = 1e-20
        data_B_only_dbm = 10 * np.log10(data_B_only + epsilon)
        data_GPH_only_dbm = 10 * np.log10(data_GPH_only + epsilon)
        
        # Find global min/max for consistent y-axis
        y_min = min(data_B_only_dbm.min(), data_GPH_only_dbm.min())
        y_max = max(data_B_only_dbm.max(), data_GPH_only_dbm.max())
        y_range = y_max - y_min
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range
        
        # ==================== B2 Direct Link Only (B) ====================
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for tx_antenna in range(K):
            ax = axes[tx_antenna]
            
            for rx_antenna in range(K):
                powers_dbm = data_B_only_dbm[:, rx_antenna, tx_antenna]
                
                linestyle = '-' if rx_antenna == tx_antenna else '--'
                linewidth = 2.5 if rx_antenna == tx_antenna else 1.5
                alpha = 1.0 if rx_antenna == tx_antenna else 0.6
                
                ax.plot(Pt_range, powers_dbm,
                       color=colors[rx_antenna],
                       linestyle=linestyle,
                       linewidth=linewidth,
                       alpha=alpha,
                       label=f'Antenna {rx_antenna}')
            
            ax.axhline(y=N0_dbm, color='gray', linestyle=':', linewidth=1.5,
                      label=f'Noise floor ({N0_dbm:.0f} dBm)')
            
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Transmit Power (dBm)', fontsize=fontsize)
            ax.set_ylabel('Received Power (dBm)', fontsize=fontsize)
            ax.set_title(f'Transmitting from Antenna {tx_antenna}', fontsize=fontsize)
            ax.legend(fontsize=fontsize-2, loc='best')
            ax.tick_params(axis='both', labelsize=fontsize-2)
            ax.set_ylim(y_min, y_max)
        
        plt.suptitle(f'B$_2$ Direct Link Only (B) - {ris_mode.capitalize()} RIS\n'
                    f'(N={N_ELEMENTS}, K={K_ANTENNAS}, J={J_RECEIVERS})',
                    fontsize=fontsize+2, y=1.02)
        plt.tight_layout()
        
        filename = f'B2_Direct_Only_Power_Profile_{ris_mode.capitalize()}.pdf'
        filepath = os.path.join(PDF_DIR, filename)
        
        try:
            plt.savefig(filepath, dpi=300, format='pdf', bbox_inches='tight')
            print(f"Saved {filepath}")
        except RuntimeError as e:
            if "latex could not be found" in str(e):
                print(f"Warning: LaTeX not available, using default font for {filename}")
                plt.rcParams['text.usetex'] = False
                plt.savefig(filepath, dpi=300, format='pdf', bbox_inches='tight')
                print(f"Saved {filepath}")
            else:
                raise
        
        plt.close()
        
        # ==================== B2 RIS Link Only (GPH) ====================
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for tx_antenna in range(K):
            ax = axes[tx_antenna]
            
            for rx_antenna in range(K):
                powers_dbm = data_GPH_only_dbm[:, rx_antenna, tx_antenna]
                
                linestyle = '-' if rx_antenna == tx_antenna else '--'
                linewidth = 2.5 if rx_antenna == tx_antenna else 1.5
                alpha = 1.0 if rx_antenna == tx_antenna else 0.6
                
                ax.plot(Pt_range, powers_dbm,
                       color=colors[rx_antenna],
                       linestyle=linestyle,
                       linewidth=linewidth,
                       alpha=alpha,
                       label=f'Antenna {rx_antenna}')
            
            ax.axhline(y=N0_dbm, color='gray', linestyle=':', linewidth=1.5,
                      label=f'Noise floor ({N0_dbm:.0f} dBm)')
            
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Transmit Power (dBm)', fontsize=fontsize)
            ax.set_ylabel('Received Power (dBm)', fontsize=fontsize)
            ax.set_title(f'Transmitting from Antenna {tx_antenna}', fontsize=fontsize)
            ax.legend(fontsize=fontsize-2, loc='best')
            ax.tick_params(axis='both', labelsize=fontsize-2)
            ax.set_ylim(y_min, y_max)
        
        plt.suptitle(f'B$_2$ RIS Link Only (GPH) - {ris_mode.capitalize()} RIS\n'
                    f'(N={N_ELEMENTS}, K={K_ANTENNAS}, J={J_RECEIVERS})',
                    fontsize=fontsize+2, y=1.02)
        plt.tight_layout()
        
        filename = f'B2_RIS_Only_Power_Profile_{ris_mode.capitalize()}.pdf'
        filepath = os.path.join(PDF_DIR, filename)
        
        try:
            plt.savefig(filepath, dpi=300, format='pdf', bbox_inches='tight')
            print(f"Saved {filepath}")
        except RuntimeError as e:
            if "latex could not be found" in str(e):
                print(f"Warning: LaTeX not available, using default font for {filename}")
                plt.rcParams['text.usetex'] = False
                plt.savefig(filepath, dpi=300, format='pdf', bbox_inches='tight')
                print(f"Saved {filepath}")
            else:
                raise
        
        plt.close()


def save_antenna_power_results(results_B1: Dict, results_B2: Dict, results_B3: Dict, filename: str):
    """Save antenna power simulation results to numpy file"""
    # Convert list of matrices to a 3D array: (num_Pt_points, K, K)
    passive_array_B1 = np.array(results_B1['passive'])
    active_array_B1 = np.array(results_B1['active'])
    passive_array_B2 = np.array(results_B2['passive'])
    active_array_B2 = np.array(results_B2['active'])
    passive_array_B2_B_only = np.array(results_B2['passive_B_only'])
    active_array_B2_B_only = np.array(results_B2['active_B_only'])
    passive_array_B2_GPH_only = np.array(results_B2['passive_GPH_only'])
    active_array_B2_GPH_only = np.array(results_B2['active_GPH_only'])
    passive_array_B3 = np.array(results_B3['passive'])
    active_array_B3 = np.array(results_B3['active'])
    
    np.savez(
        filename,
        Pt_range_dbm=results_B1['Pt_range_dbm'],
        B1_passive_power_profiles=passive_array_B1,
        B1_active_power_profiles=active_array_B1,
        B2_passive_power_profiles=passive_array_B2,
        B2_active_power_profiles=active_array_B2,
        B2_passive_B_only=passive_array_B2_B_only,
        B2_active_B_only=active_array_B2_B_only,
        B2_passive_GPH_only=passive_array_B2_GPH_only,
        B2_active_GPH_only=active_array_B2_GPH_only,
        B3_passive_power_profiles=passive_array_B3,
        B3_active_power_profiles=active_array_B3
    )
    print(f"Antenna power results saved to {filename}")


def load_antenna_power_results(filename: str) -> Tuple[Dict, Dict, Dict]:
    """Load antenna power simulation results from numpy file"""
    try:
        data = np.load(filename)
        results_B1 = {
            'Pt_range_dbm': data['Pt_range_dbm'],
            'passive': list(data['B1_passive_power_profiles']),
            'active': list(data['B1_active_power_profiles'])
        }
        results_B2 = {
            'Pt_range_dbm': data['Pt_range_dbm'],
            'passive': list(data['B2_passive_power_profiles']),
            'active': list(data['B2_active_power_profiles']),
            'passive_B_only': list(data['B2_passive_B_only']),
            'active_B_only': list(data['B2_active_B_only']),
            'passive_GPH_only': list(data['B2_passive_GPH_only']),
            'active_GPH_only': list(data['B2_active_GPH_only'])
        }
        results_B3 = {
            'Pt_range_dbm': data['Pt_range_dbm'],
            'passive': list(data['B3_passive_power_profiles']),
            'active': list(data['B3_active_power_profiles'])
        }
        return results_B1, results_B2, results_B3
    except (FileNotFoundError, IOError):
        return None, None, None


def print_debug_tables(results: Dict):
    """
    Print debug tables with BER summary and key metrics for analysis.
    Includes power ratios and diagonality metrics.
    """
    Pt_range = results['Pt_range_dbm']
    
    print("\n" + "="*120)
    print("DEBUG TABLES - BER Analysis")
    print("="*120)
    
    # Table 1: BER Summary at key power points
    print("\nTable 1: BER Summary at Key Power Points")
    print("-" * 130)
    key_powers = [-100, -75, -50, -25, 0, 25, 50, 75]
    
    # Header with two B2 curves
    print(f"{'Power (dBm)':<12} | {'B1':<16} | {'B2-SSK':<16} | {'B2-Direct':<16} | {'B3':<16} | {'E1':<16} | {'E2':<16}")
    print(f"{'':12} | {'(NLOS)':<16} | {'(LOS,SSK)':<16} | {'(LOS,ML)':<16} | {'(Direct)':<16} | {'(NLOS)':<16} | {'(LOS)':<16}")
    print("-" * 130)
    
    for ris_mode in ['passive', 'active']:
        print(f"\n{ris_mode.upper()} RIS:")
        for target_pt in key_powers:
            idx = np.argmin(np.abs(Pt_range - target_pt))
            actual_pt = Pt_range[idx]
            
            row = f"{actual_pt:<12.0f} |"
            for key in ['B1', 'B2_SSK', 'B2_Direct', 'B3', 'E1', 'E2']:
                mean = results[ris_mode][key]['mean'][idx]
                lower = results[ris_mode][key]['lower'][idx]
                upper = results[ris_mode][key]['upper'][idx]
                ci_width = (upper - lower) / 2
                row += f" {mean:.3f} (±{ci_width:.3f}) |"
            print(row)
    
    # Table 2: Key Performance Metrics
    print("\n\nTable 2: Key Performance Metrics")
    print("-" * 90)
    
    for ris_mode in ['passive', 'active']:
        print(f"\n{ris_mode.upper()} RIS:")
        print(f"{'Receiver':<18} | {'BER<0.5 at':<12} | {'BER<0.1 at':<12} | {'Min BER':<10} | {'At Power':<10}")
        print("-" * 90)
        
        # Key list with two B2 curves
        perf_keys = ['B1', 'B2_SSK', 'B2_Direct', 'B3', 'E1', 'E2']
        
        for key in perf_keys:
            mean_values = np.array(results[ris_mode][key]['mean'])
            
            # Find where BER drops below 0.5
            below_0_5_idx = np.where(mean_values < 0.5)[0]
            ber_0_5_power = Pt_range[below_0_5_idx[0]] if len(below_0_5_idx) > 0 else 'N/A'
            
            # Find where BER drops below 0.1
            below_0_1_idx = np.where(mean_values < 0.1)[0]
            ber_0_1_power = Pt_range[below_0_1_idx[0]] if len(below_0_1_idx) > 0 else 'N/A'
            
            # Find minimum BER
            min_ber_idx = np.argmin(mean_values)
            min_ber = mean_values[min_ber_idx]
            min_ber_power = Pt_range[min_ber_idx]
            
            b05_str = f"{ber_0_5_power:<12}" if isinstance(ber_0_5_power, str) else f"{ber_0_5_power:<12.0f}"
            b01_str = f"{ber_0_1_power:<12}" if isinstance(ber_0_1_power, str) else f"{ber_0_1_power:<12.0f}"
            
            print(f"{key:<18} | {b05_str} | {b01_str} | {min_ber:<10.4f} | {min_ber_power:<10.0f}")
    
    # Table 3: Channel Characteristics Diagnostics
    print("\n\nTable 3: Channel Characteristics Diagnostics (Sample at Pt = -20 dBm)")
    print("-" * 90)
    print("Computing sample channel statistics...")
    
    # Sample at middle of range (-20 dBm)
    sample_pt_dbm = -20.0
    sample_Pt = 10**(sample_pt_dbm/10)
    
    # Run single realization to get diagnostics
    np.random.seed(42)  # Fixed seed for reproducibility
    H = PATH_LOSS * generate_random_channel_matrix(N_ELEMENTS, K_ANTENNAS)
    G1 = PATH_LOSS * generate_random_channel_matrix(K_ANTENNAS, N_ELEMENTS)
    G2 = PATH_LOSS * generate_random_channel_matrix(K_ANTENNAS, N_ELEMENTS)
    Bb = PATH_LOSS * generate_random_channel_matrix(K_ANTENNAS, K_ANTENNAS)
    
    Gs = [G1, G2]
    Cs = []
    Ps, _ = calculate_multi_ris_reflection_matrices(
        K_ANTENNAS, N_ELEMENTS, J_RECEIVERS, M_SURFACES, Gs, H, ETA, Cs
    )
    P = unify_ris_reflection_matrices(Ps, Cs)
    
    def offdiag_ratio(A):
        """Compute ||off-diagonal|| / ||diagonal|| ratio"""
        d = np.diag(np.diag(A))
        off = A - d
        dn = np.linalg.norm(d)
        on = np.linalg.norm(off)
        return on/(dn + 1e-20)
    
    # Passive channels
    eff_B1_passive = G1 @ P @ H
    eff_B2_passive = G2 @ P @ H
    
    # Active channels
    H_active = H / np.max(np.abs(H))
    eff_B1_active = G1 @ P @ H_active
    eff_B2_active = G2 @ P @ H_active
    
    # Power calculations (at sample Pt)
    def channel_power(C):
        return np.linalg.norm(C, 'fro')**2
    
    print("\n--- Diagonality Ratios (off-diagonal / diagonal, lower is better) ---")
    print(f"{'Channel':<30} | {'Passive':>12} | {'Active':>12}")
    print("-" * 60)
    print(f"{'B1 (G1PH) - should be ~1e-15':<30} | {offdiag_ratio(eff_B1_passive):>12.2e} | {offdiag_ratio(eff_B1_active):>12.2e}")
    print(f"{'B2 (G2PH) - should be ~1e-15':<30} | {offdiag_ratio(eff_B2_passive):>12.2e} | {offdiag_ratio(eff_B2_active):>12.2e}")
    print(f"{'B2 (B+G2PH) - should be large':<30} | {offdiag_ratio(Bb + eff_B2_passive):>12.2e} | {offdiag_ratio(Bb + eff_B2_active):>12.2e}")
    
    print("\n--- Power Levels (dB relative to noise floor) ---")
    N0_mw = calculate_noise_floor_in_mw()
    N0_db = 10*np.log10(N0_mw)
    Pt_db = 10*np.log10(sample_Pt)
    
    def rel_db(power_linear):
        return 10*np.log10(power_linear + 1e-30) - N0_db
    
    print(f"Transmit power: {Pt_db:.1f} dBm, Noise floor: {N0_db:.1f} dBm")
    print(f"{'Path':<30} | {'Passive (dB)':>15} | {'Active (dB)':>15}")
    print("-" * 70)
    
    p_b1 = channel_power(eff_B1_passive) * sample_Pt
    a_b1 = channel_power(eff_B1_active) * sample_Pt
    print(f"{'B1: RIS path ||GPH||^2*Pt':<30} | {rel_db(p_b1):>15.1f} | {rel_db(a_b1):>15.1f}")
    
    p_b2_direct = channel_power(Bb) * sample_Pt
    p_b2_ris = channel_power(eff_B2_passive) * sample_Pt
    a_b2_ris = channel_power(eff_B2_active) * sample_Pt
    print(f"{'B2: Direct ||B||^2*Pt':<30} | {rel_db(p_b2_direct):>15.1f} | {rel_db(p_b2_direct):>15.1f} (same)")
    print(f"{'B2: RIS path ||GPH||^2*Pt':<30} | {rel_db(p_b2_ris):>15.1f} | {rel_db(a_b2_ris):>15.1f}")
    
    # Power ratios
    ratio_db_passive = 10*np.log10((channel_power(Bb)+1e-30)/(channel_power(eff_B2_passive)+1e-30))
    ratio_db_active = 10*np.log10((channel_power(Bb)+1e-30)/(channel_power(eff_B2_active)+1e-30))
    print(f"\n--- B2 Direct-to-RIS Power Ratio ||B||^2/||GPH||^2 ---")
    print(f"  Passive: {ratio_db_passive:.1f} dB (positive means direct stronger)")
    print(f"  Active:  {ratio_db_active:.1f} dB")
    
    print("\n" + "="*100)
    print("End of Debug Tables")
    print("="*100 + "\n")


def plot_results(results: Dict):
    """Generate BER vs Pt graphs with 95% confidence intervals for passive and active RIS"""
    configure_latex()
    fontsize = 16
    plt.rc('font', **{'size': fontsize})
    
    Pt_range = results['Pt_range_dbm']
    
    # Plot settings - Two B2 curves: SSK detection vs Direct ML detection
    colors = {
        'B1': 'tab:blue',
        'B2_SSK': 'tab:orange',
        'B2_Direct': 'tab:green',
        'B3': 'tab:cyan',
        'E1': 'tab:red',
        'E2': 'tab:purple'
    }
    
    labels = {
        'B1': r'B$_1$ (NLOS, reflection)',
        'B2_SSK': r'B$_2$ (LOS, SSK detection)',
        'B2_Direct': r'B$_2$ (LOS, Direct ML)',
        'B3': r'B$_3$ (Direct only, no RIS)',
        'E1': r'E$_1$ (NLOS, reflection)',
        'E2': r'E$_2$ (LOS, direct + RIS)'
    }
    
    linestyles = {
        'B1': '-',
        'B2_SSK': ':',
        'B2_Direct': '-',
        'B3': '--',
        'E1': ':',
        'E2': ':'
    }
    
    markers = {
        'B1': 'o',
        'B2_SSK': '^',
        'B2_Direct': 's',
        'B3': 'D',
        'E1': 'x',
        'E2': 'd'
    }
    
    # Generate both graphs with confidence intervals
    plot_keys = ['B1', 'B2_SSK', 'B2_Direct', 'B3', 'E1', 'E2']
    
    for ris_mode in ['passive', 'active']:
        plt.figure(figsize=(12, 7))
        
        for key in plot_keys:
            # Extract mean, lower, and upper bounds
            mean_values = np.array(results[ris_mode][key]['mean'])
            lower_values = np.array(results[ris_mode][key]['lower'])
            upper_values = np.array(results[ris_mode][key]['upper'])
            
            # Clip values to avoid log(0) issues
            epsilon = 1e-10
            mean_values = np.maximum(mean_values, epsilon)
            lower_values = np.maximum(lower_values, epsilon)
            upper_values = np.maximum(upper_values, epsilon)
            
            # Plot mean line
            plt.semilogy(
                Pt_range, mean_values,
                color=colors[key],
                linestyle=linestyles[key],
                marker=markers[key],
                markersize=6,
                label=labels[key]
            )
            
            # Plot 95% confidence interval as shaded region
            plt.fill_between(
                Pt_range, lower_values, upper_values,
                color=colors[key],
                alpha=0.2
            )
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('Transmit Power (dBm)', fontsize=fontsize)
        plt.ylabel('Bit Error Rate (BER)', fontsize=fontsize)
        plt.tick_params(axis='both', labelsize=fontsize)
        plt.legend(fontsize=fontsize-2, loc='best')
        # Clip y-axis at minimum meaningful BER (1/N_SIMULATIONS)
        min_ber = 1.0 / N_SIMULATIONS
        plt.ylim(min_ber, 1.0)
        plt.tight_layout()
        
        # Save figure
        filename = f'BER_vs_Pt_{ris_mode.capitalize()}.pdf'
        filepath = os.path.join(PDF_DIR, filename)
        
        try:
            plt.savefig(filepath, dpi=300, format='pdf', bbox_inches='tight')
            print(f"Saved {filepath}")
        except RuntimeError as e:
            if "latex could not be found" in str(e):
                print(f"Warning: LaTeX not available, using default font for {filename}")
                plt.rcParams['text.usetex'] = False
                plt.savefig(filepath, dpi=300, format='pdf', bbox_inches='tight')
                print(f"Saved {filepath}")
            else:
                raise
        
        plt.close()


def main():
    """Main execution function"""
    # Create output directories
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # ==================== BER Simulations ====================
    ber_data_filename = os.path.join(DATA_DIR, 'ber_los_results.npz')
    
    # Check for existing BER results
    ber_results = load_results(ber_data_filename)
    
    if ber_results is None:
    
        print("="*60)
        print("BER SIMULATIONS (ALWAYS REGENERATE MODE)")
        print("="*60)
        print("Running simulations...")
        ber_results = run_all_simulations()
        save_results(ber_results, ber_data_filename)
    else:
        print(f"Loaded existing BER results from {ber_data_filename}")
    
    # Generate BER plots
    print("\nGenerating BER plots...")
    plot_results(ber_results)
    
    # Print debug tables for analysis
    print_debug_tables(ber_results)
    
    # ==================== Antenna Power Profile Simulations ====================
    power_data_filename = os.path.join(DATA_DIR, 'antenna_power_results.npz')
    
    # FIX P0: Commented out existing data loading to always regenerate
    # Check for existing antenna power results
    # power_results_B1, power_results_B2, power_results_B3 = load_antenna_power_results(power_data_filename)
    # 
    # if power_results_B1 is None or power_results_B2 is None or power_results_B3 is None:
    
    print("\n" + "="*60)
    print("ANTENNA POWER PROFILE SIMULATIONS (ALWAYS REGENERATE MODE)")
    print("="*60)
    print("Running simulations...")
    power_results_B1, power_results_B2, power_results_B3 = run_antenna_power_simulations()
    save_antenna_power_results(power_results_B1, power_results_B2, power_results_B3, power_data_filename)
    # else:
    #     print(f"\nLoaded existing power profile results from {power_data_filename}")
    
    # Generate antenna power profile plots
    print("\nGenerating antenna power profile plots...")
    plot_antenna_power_profiles(power_results_B1, power_results_B2, power_results_B3)
    
    # Removed: B2 separate power profiles (Direct only and RIS only)
    # These were redundant since B1 shows pure RIS path and B3 shows pure direct path
    
    print("\n" + "="*60)
    print("All simulations and plots completed!")
    print("="*60)
    print(f"\nResults saved in: {OUTPUT_DIR}")
    print(f"  - BER curves: BER_vs_Pt_Passive.pdf, BER_vs_Pt_Active.pdf")
    print(f"  - Power profiles: B1/B2/B3_Antenna_Power_Profile_Passive/Active.pdf")


if __name__ == "__main__":
    main()
