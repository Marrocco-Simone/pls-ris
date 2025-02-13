import numpy as np
import matplotlib.pyplot as plt
import random
from diagonalization import (
    calculate_multi_ris_reflection_matrices,
    generate_random_channel_matrix
)
from secrecy import (
    create_random_noise_vector,
    snr_db_to_sigma_sq,
    unify_ris_reflection_matrices
)

def simulate_ssk_transmission_reflection(K, effective_channel, sigma_sq):
    if effective_channel.shape != (K, K):
        raise ValueError(f"Effective channel shape must be ({K}, {K}), but got {effective_channel.shape}")

    x = np.zeros(K)
    x[np.random.randint(K)] = 1

    noise = create_random_noise_vector(len(x), sigma_sq)
    y = effective_channel @ x + noise 
    detected_idx = np.argmax(np.abs(y)**2)
    true_idx = np.argmax(x)
    
    return detected_idx == true_idx

def simulate_ssk_transmission_direct(K, B, effective_channel, sigma_sq_B, sigma_sq_effective_channel):
    if B.shape != (K, K):
        raise ValueError(f"B shape must be ({K}, {K}), but got {B.shape}")
    
    if effective_channel.shape != (K, K):
        raise ValueError(f"Effective channel shape must be ({K}, {K}), but got {effective_channel.shape}")

    x = np.zeros(K)
    x[np.random.randint(K)] = 1

    if sigma_sq_effective_channel is None:
        sigma_sq_effective_channel = sigma_sq_B

    noise_B = create_random_noise_vector(len(x), sigma_sq_B)
    noise_effective_channel = create_random_noise_vector(len(x), sigma_sq_effective_channel)
    
    y = (B + effective_channel) @ x + noise_B + noise_effective_channel
    distances = np.array([np.linalg.norm(y - B[:, i]) for i in range(B.shape[1])])
    detected_idx = np.argmin(distances)
    true_idx = np.argmax(x)
    
    return detected_idx == true_idx

def calculate_ber_simulation(snr_db, K, N, J, M, eta=0.9, num_symbols=10000):
    sigma_sq = snr_db_to_sigma_sq(snr_db)
    errors_receiver = 0
    errors_eavesdropper = 0
    errors_direct = 0

    errors_receiver_double = 0
    errors_eavesdropper_double = 0
    
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

        if not simulate_ssk_transmission_reflection(K, effective_channel_receiver, sigma_sq):
            errors_receiver += 1
        
        if not simulate_ssk_transmission_direct(K, B, effective_channel_eavesdropper, sigma_sq):
            errors_eavesdropper += 1

        if not simulate_ssk_transmission_direct(K, B, effective_channel_direct, sigma_sq):
            errors_direct += 1

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

        if not simulate_ssk_transmission_reflection(K, effective_channel_receiver_double, sigma_sq):
            errors_receiver_double += 1
        if not simulate_ssk_transmission_direct(K, B, effective_channel_eavesdropper_double, sigma_sq):
            errors_eavesdropper_double += 1
    
    result_receiver = errors_receiver / num_symbols
    result_eavesdropper = errors_eavesdropper / num_symbols
    result_direct = errors_direct / num_symbols
    result_receiver_double = errors_receiver_double / num_symbols
    result_eavesdropper_double = errors_eavesdropper_double / num_symbols

    return result_receiver, result_eavesdropper, result_direct, result_receiver_double, result_eavesdropper_double

def plot_ber_curves():
    N = 16    # * Number of reflecting elements
    K = 2     # * Number of antennas 
    eta = 0.9 # * Reflection efficiency
    
    for J in range(1, 3):  # * Number of receivers
        for M in range(1, 3):  # * Number of RIS surfaces
            print(f"Processing J={J}, M={M}")
            snr_range_db = np.arange(-10, 31, 2)
            ber_simulated_receiver = []
            ber_simulated_eavesdropper = []
            ber_simulated_direct = []
            ber_simulated_receiver_double = []
            ber_simulated_eavesdropper_double = []
            
            for snr_db in snr_range_db:
                result_receiver, result_eavesdropper, result_direct,result_receiver_double, result_eavesdropper_double = calculate_ber_simulation(snr_db, K, N, J, M, eta)
                ber_simulated_receiver.append(result_receiver)
                ber_simulated_eavesdropper.append(result_eavesdropper)
                ber_simulated_direct.append(result_direct)
                ber_simulated_receiver_double.append(result_receiver_double)
                ber_simulated_eavesdropper_double.append(result_eavesdropper_double)
                print(f"Processed SNR = {snr_db} dB:\t{result_receiver:.2f}\t{result_eavesdropper:.2f}\t{result_direct:.2f}")

            plt_name = f'SSK BER Performance with RIS (K={K}, N={N}, J={J}, M={M})'
            plt.figure(figsize=(10, 6))
            plt.semilogy(snr_range_db, ber_simulated_direct, label=f'Simulation Direct')
            plt.semilogy(snr_range_db, ber_simulated_receiver, label='Simulation Receiver')
            plt.semilogy(snr_range_db, ber_simulated_receiver_double, label='Simulation Receiver Double RIS Source')
            plt.semilogy(snr_range_db, ber_simulated_eavesdropper, label=f'Simulation Eavesdropper')
            plt.semilogy(snr_range_db, ber_simulated_eavesdropper_double, label=f'Simulation Eavesdropper Double RIS Source')
            plt.grid(True)
            plt.xlabel('SNR (dB)')
            plt.ylabel('Bit Error Rate (BER)')
            plt.title(plt_name)
            plt.legend()
            plt.savefig(f"./simulations/results/{plt_name}.png", dpi=300, format='png')
            print(f"Saved {plt_name}.png\n\n")

if __name__ == "__main__":
    plot_ber_curves()