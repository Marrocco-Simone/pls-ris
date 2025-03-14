import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Callable
from diagonalization import (
    calculate_multi_ris_reflection_matrices,
    generate_random_channel_matrix
)
from secrecy import (
    create_random_noise_vector,
    snr_db_to_sigma_sq,
    unify_ris_reflection_matrices
)

def simulate_ssk_transmission(K: int, sigma_sq: float, calculate_detected_id: Callable[[np.ndarray, np.ndarray], float]):
    n_bits = int(np.log2(K))
    if 2**n_bits != K:
        raise ValueError(f"K must be a power of 2, got {K}")
    bit_mappings = np.array([format(i, f'0{n_bits}b') for i in range(K)])    
    true_bits = np.random.randint(0, 2, n_bits)
    true_bits_str = ''.join(map(str, true_bits))
    true_idx = np.where(bit_mappings == true_bits_str)[0][0]

    x = np.zeros(K)
    x[true_idx] = 1

    noise = create_random_noise_vector(K, sigma_sq)
    detected_idx = calculate_detected_id(x, noise)
    
    detected_bits = np.array(list(bit_mappings[detected_idx])).astype(int)
    errors = np.sum(detected_bits != true_bits)
    return errors / n_bits

def calculate_confidence_interval(error_rates, confidence=0.95):
    """
    Calculate the confidence interval for a list of error rates.
    
    Args:
        error_rates: List of error rates from individual trials
        confidence: Confidence level (default: 0.95 for 95% confidence interval)
        
    Returns:
        tuple: (mean, lower_bound, upper_bound)
    """
    n = len(error_rates)
    mean = np.mean(error_rates)
    if n <= 1:
        return mean, mean, mean
    
    z = 1.96 if confidence == 0.95 else -1 * np.sqrt(2) * np.log((1 - confidence) / 2)
    
    std_error = np.std(error_rates, ddof=1) / np.sqrt(n)
    margin = z * std_error
    
    return mean, mean - margin, mean + margin

def simulate_ssk_transmission_reflection(K: int, effective_channel: np.ndarray, sigma_sq: float):
    if effective_channel.shape != (K, K):
        raise ValueError(f"Reflection: Effective channel shape must be ({K}, {K}), but got {effective_channel.shape}")

    def calculate_detected_id(x: np.ndarray, noise: np.ndarray):
        y = effective_channel @ x + noise
        return np.argmax(np.abs(y)**2)
    
    return simulate_ssk_transmission(K, sigma_sq, calculate_detected_id)

def simulate_ssk_transmission_direct(K: int, B: np.ndarray, effective_channel: np.ndarray, sigma_sq: float):
    if B.shape != (K, K):
        raise ValueError(f"Direct: B shape must be ({K}, {K}), but got {B.shape}")
    
    if effective_channel.shape != (K, K):
        raise ValueError(f"Direct: Effective channel shape must be ({K}, {K}), but got {effective_channel.shape}")

    def calculate_detected_id(x: np.ndarray, noise: np.ndarray):
        y = (B + effective_channel) @ x + noise
        distances = np.array([np.linalg.norm(y - B[:, i]) for i in range(B.shape[1])])
        return np.argmin(distances)
    
    return simulate_ssk_transmission(K, sigma_sq, calculate_detected_id)

def calculate_ber_simulation(snr_db, K, N, J, M, eta=0.9, num_symbols=100000):
    sigma_sq = snr_db_to_sigma_sq(snr_db)
    results_receiver = []
    results_eavesdropper = []
    results_direct = []
    results_receiver_double = []
    results_eavesdropper_double = []
    
    for _ in range(num_symbols):
        H = generate_random_channel_matrix(N, K)
        Gs = [generate_random_channel_matrix(K, N) for _ in range(J)]
        G = random.choice(Gs)
        Fs = [generate_random_channel_matrix(K, N) for _ in range(M)]
        B = generate_random_channel_matrix(K, K)
        Cs = [generate_random_channel_matrix(N, N) for _ in range(M-1)]
        Ps, _ = calculate_multi_ris_reflection_matrices(
            K, N, J, M, Gs, H, eta, Cs
        )
        P = unify_ris_reflection_matrices(Ps, Cs)

        effective_channel_receiver = G @ P @ H
        effective_channel_eavesdropper = np.zeros((K, K), dtype=np.complex128) # F @ P @ H
        for i in range(M):
            P_to_i = unify_ris_reflection_matrices(Ps[:i+1], Cs[:i])
            effective_channel_eavesdropper += Fs[i] @ P_to_i @ H
        effective_channel_direct = np.zeros((K, K))

        results_receiver.append(simulate_ssk_transmission_reflection(K, effective_channel_receiver, sigma_sq))
        results_eavesdropper.append(simulate_ssk_transmission_direct(K, B, effective_channel_eavesdropper, sigma_sq))
        results_direct.append(simulate_ssk_transmission_direct(K, B, effective_channel_direct, sigma_sq))

        H2 = generate_random_channel_matrix(N, K)
        Gs2 = [generate_random_channel_matrix(K, N) for _ in range(J)]
        G2 = random.choice(Gs2)
        Fs2 = [generate_random_channel_matrix(K, N) for _ in range(M)]
        Cs2 = [generate_random_channel_matrix(N, N) for _ in range(M-1)]
        Ps2, _ = calculate_multi_ris_reflection_matrices(
            K, N, J, M, Gs2, H2, eta, Cs2
        )
        P2 = unify_ris_reflection_matrices(Ps2, Cs2)

        effective_channel_receiver_2 = G2 @ P2 @ H2
        effective_channel_eavesdropper_2 = np.zeros((K, K), dtype=np.complex128) # F @ P @ H
        for i in range(M):
            P_to_i = unify_ris_reflection_matrices(Ps2[:i+1], Cs2[:i])
            effective_channel_eavesdropper_2 += Fs2[i] @ P_to_i @ H2

        effective_channel_receiver_double = effective_channel_receiver + effective_channel_receiver_2
        effective_channel_eavesdropper_double = effective_channel_eavesdropper + effective_channel_eavesdropper_2

        results_receiver_double.append(simulate_ssk_transmission_reflection(K, effective_channel_receiver_double, sigma_sq))
        results_eavesdropper_double.append(simulate_ssk_transmission_direct(K, B, effective_channel_eavesdropper_double, sigma_sq))
    
    result_receiver, lower_receiver, upper_receiver = calculate_confidence_interval(results_receiver)
    result_eavesdropper, lower_eavesdropper, upper_eavesdropper = calculate_confidence_interval(results_eavesdropper)
    result_direct, lower_direct, upper_direct = calculate_confidence_interval(results_direct)
    result_receiver_double, lower_receiver_double, upper_receiver_double = calculate_confidence_interval(results_receiver_double)
    result_eavesdropper_double, lower_eavesdropper_double, upper_eavesdropper_double = calculate_confidence_interval(results_eavesdropper_double)

    return {
        'receiver': (result_receiver, lower_receiver, upper_receiver),
        'eavesdropper': (result_eavesdropper, lower_eavesdropper, upper_eavesdropper),
        'direct': (result_direct, lower_direct, upper_direct),
        'receiver_double': (result_receiver_double, lower_receiver_double, upper_receiver_double),
        'eavesdropper_double': (result_eavesdropper_double, lower_eavesdropper_double, upper_eavesdropper_double)
    }

def save_ber_data(filename, data):
    """Save BER data to a numpy file"""
    np.savez(
        filename,
        snr_range_db=data['snr_range_db'],
        ber_receiver_mean=data['ber_receiver']['mean'],
        ber_receiver_lower=data['ber_receiver']['lower'],
        ber_receiver_upper=data['ber_receiver']['upper'],
        ber_eavesdropper_mean=data['ber_eavesdropper']['mean'],
        ber_eavesdropper_lower=data['ber_eavesdropper']['lower'],
        ber_eavesdropper_upper=data['ber_eavesdropper']['upper'],
        ber_direct_mean=data['ber_direct']['mean'],
        ber_direct_lower=data['ber_direct']['lower'],
        ber_direct_upper=data['ber_direct']['upper'],
        ber_receiver_double_mean=data['ber_receiver_double']['mean'],
        ber_receiver_double_lower=data['ber_receiver_double']['lower'],
        ber_receiver_double_upper=data['ber_receiver_double']['upper'],
        ber_eavesdropper_double_mean=data['ber_eavesdropper_double']['mean'],
        ber_eavesdropper_double_lower=data['ber_eavesdropper_double']['lower'],
        ber_eavesdropper_double_upper=data['ber_eavesdropper_double']['upper']
    )
    print(f"Data saved to {filename}")

def load_ber_data(filename):
    """Load BER data from a numpy file"""
    try:
        data = np.load(filename)
        return {
            'snr_range_db': data['snr_range_db'],
            'ber_receiver': {
                'mean': data['ber_receiver_mean'],
                'lower': data['ber_receiver_lower'],
                'upper': data['ber_receiver_upper']
            },
            'ber_eavesdropper': {
                'mean': data['ber_eavesdropper_mean'],
                'lower': data['ber_eavesdropper_lower'],
                'upper': data['ber_eavesdropper_upper']
            },
            'ber_direct': {
                'mean': data['ber_direct_mean'],
                'lower': data['ber_direct_lower'],
                'upper': data['ber_direct_upper']
            },
            'ber_receiver_double': {
                'mean': data['ber_receiver_double_mean'],
                'lower': data['ber_receiver_double_lower'],
                'upper': data['ber_receiver_double_upper']
            },
            'ber_eavesdropper_double': {
                'mean': data['ber_eavesdropper_double_mean'],
                'lower': data['ber_eavesdropper_double_lower'],
                'upper': data['ber_eavesdropper_double_upper']
            }
        }
    except (FileNotFoundError, IOError):
        return None

def plot_ber_curves():
    N = 16    # * Number of reflecting elements
    K = 2     # * Number of antennas 
    eta = 0.9 # * Reflection efficiency
    
    for J in range(1, 3):  # * Number of receivers
        for M in range(1, 3):  # * Number of RIS surfaces
            print(f"Processing J={J}, M={M}")
            
            # Create data directory if it doesn't exist
            import os
            data_dir = "./simulations/data"
            os.makedirs(data_dir, exist_ok=True)
            
            data_filename = f"{data_dir}/ber_data_K{K}_N{N}_J{J}_M{M}.npz"
            plot_data = load_ber_data(data_filename)
            
            if plot_data is None:
                print(f"No existing data found. Running simulations...")
                snr_range_db = np.arange(-10, 31, 2)
                
                ber_receiver = {'mean': [], 'lower': [], 'upper': []}
                ber_eavesdropper = {'mean': [], 'lower': [], 'upper': []}
                ber_direct = {'mean': [], 'lower': [], 'upper': []}
                ber_receiver_double = {'mean': [], 'lower': [], 'upper': []}
                ber_eavesdropper_double = {'mean': [], 'lower': [], 'upper': []}
                
                for snr_db in snr_range_db:
                    results = calculate_ber_simulation(snr_db, K, N, J, M, eta)
                    
                    ber_receiver['mean'].append(results['receiver'][0])
                    ber_receiver['lower'].append(results['receiver'][1])
                    ber_receiver['upper'].append(results['receiver'][2])
                    
                    ber_eavesdropper['mean'].append(results['eavesdropper'][0])
                    ber_eavesdropper['lower'].append(results['eavesdropper'][1])
                    ber_eavesdropper['upper'].append(results['eavesdropper'][2])
                    
                    ber_direct['mean'].append(results['direct'][0])
                    ber_direct['lower'].append(results['direct'][1])
                    ber_direct['upper'].append(results['direct'][2])
                    
                    ber_receiver_double['mean'].append(results['receiver_double'][0])
                    ber_receiver_double['lower'].append(results['receiver_double'][1])
                    ber_receiver_double['upper'].append(results['receiver_double'][2])
                    
                    ber_eavesdropper_double['mean'].append(results['eavesdropper_double'][0])
                    ber_eavesdropper_double['lower'].append(results['eavesdropper_double'][1])
                    ber_eavesdropper_double['upper'].append(results['eavesdropper_double'][2])
                    
                    print(f"Processed SNR = {snr_db} dB:\t{results['receiver'][0]:.2f}\t{results['eavesdropper'][0]:.2f}\t{results['direct'][0]:.2f}")
                
                # Save the data
                plot_data = {
                    'snr_range_db': snr_range_db,
                    'ber_receiver': ber_receiver,
                    'ber_eavesdropper': ber_eavesdropper,
                    'ber_direct': ber_direct,
                    'ber_receiver_double': ber_receiver_double,
                    'ber_eavesdropper_double': ber_eavesdropper_double
                }
                save_ber_data(data_filename, plot_data)
            else:
                print(f"Loading existing data from {data_filename}")

            plt_name = f'SSK BER Performance with RIS (K={K}, N={N}, J={J}, M={M})'
            plt.figure(figsize=(10, 6))
            
            plt.semilogy(plot_data['snr_range_db'], plot_data['ber_direct']['mean'], 'o-', label=f'Simulation Direct')
            plt.fill_between(plot_data['snr_range_db'], plot_data['ber_direct']['lower'], plot_data['ber_direct']['upper'], alpha=0.2)
            
            plt.semilogy(plot_data['snr_range_db'], plot_data['ber_receiver']['mean'], 's-', label='Simulation Receiver')
            plt.fill_between(plot_data['snr_range_db'], plot_data['ber_receiver']['lower'], plot_data['ber_receiver']['upper'], alpha=0.2)
            
            plt.semilogy(plot_data['snr_range_db'], plot_data['ber_receiver_double']['mean'], '^-', label='Simulation Receiver Double RIS Source')
            plt.fill_between(plot_data['snr_range_db'], plot_data['ber_receiver_double']['lower'], plot_data['ber_receiver_double']['upper'], alpha=0.2)
            
            plt.semilogy(plot_data['snr_range_db'], plot_data['ber_eavesdropper']['mean'], 'x-', label=f'Simulation Eavesdropper')
            plt.fill_between(plot_data['snr_range_db'], plot_data['ber_eavesdropper']['lower'], plot_data['ber_eavesdropper']['upper'], alpha=0.2)
            
            plt.semilogy(plot_data['snr_range_db'], plot_data['ber_eavesdropper_double']['mean'], 'd-', label=f'Simulation Eavesdropper Double RIS Source')
            plt.fill_between(plot_data['snr_range_db'], plot_data['ber_eavesdropper_double']['lower'], plot_data['ber_eavesdropper_double']['upper'], alpha=0.2)
            
            plt.grid(True)
            plt.xlabel('SNR (dB)')
            plt.ylabel('Bit Error Rate (BER)')
            plt.title(plt_name)
            plt.legend()
            
            # Create results directory if it doesn't exist
            results_dir = "./simulations/results_pdf_ci"
            os.makedirs(results_dir, exist_ok=True)
            
            plt.savefig(f"{results_dir}/{plt_name}.pdf", dpi=300, format='pdf', bbox_inches='tight')
            print(f"Saved {plt_name}.pdf\n\n")

if __name__ == "__main__":
    plot_ber_curves()