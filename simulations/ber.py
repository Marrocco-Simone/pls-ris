import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ncx2, chi2
from diagonalization import (
    calculate_multi_ris_reflection_matrices,
    generate_random_channel_matrix
)
from secrecy import (
    create_random_noise_vector,
    snr_db_to_sigma_sq,
    unify_ris_reflection_matrices
)

def calculate_scaling_factor(K):
    if K < 2 or (K & (K - 1)) != 0:
        raise ValueError("K must be power of 2")
        
    kappa = int(np.log2(K))
    delta = np.zeros(kappa + 1)
    
    for k in range(1, kappa + 1):
        delta[k] = delta[k-1] + (2**(k-1) - delta[k-1])/(2**k - 1)
    
    return delta[kappa]

def calculate_ber_theoretical(snr_db, K, N, num_monte_carlo=1000):
    snr_linear = 10**(snr_db/10)
    delta_k = calculate_scaling_factor(K)
    
    total_error_prob = 0
    
    # Monte Carlo integration over channel realizations
    for _ in range(num_monte_carlo):
        H = generate_random_channel_matrix(N, K)
        G = generate_random_channel_matrix(K, N)
        Ps, _ = calculate_multi_ris_reflection_matrices(
            K, N, 1, 1, [G], H, eta=0.9
        )
        P = unify_ris_reflection_matrices(Ps)
        
        effective_channel = G @ P @ H
        diag_elements = np.diag(effective_channel)
        channel_gains = np.abs(diag_elements)**2
        
        lambda_i = 2 * channel_gains * snr_linear
        lambda_avg = np.mean(lambda_i)
        
        # Inner integral over t
        def integrand(t):
            # CDF of chi-square with 2 DoF
            F_chi2 = chi2.cdf(t, df=2)**(K-1)
            # PDF of non-central chi-square with 2 DoF
            f_chi2_nc = ncx2.pdf(t, df=2, nc=lambda_avg)
            return F_chi2 * f_chi2_nc
            
        # Numerical integration using Simpson's rule
        t = np.linspace(0, 100, 1000)
        integral = np.trapezoid(integrand(t), t)
        
        total_error_prob += (1 - integral)
    
    ber = delta_k * (total_error_prob / num_monte_carlo)
    return ber

def simulate_ssk_transmission_reflection(x, effective_channel, sigma_sq):
    noise = create_random_noise_vector(len(x), sigma_sq)
    y = effective_channel @ x + noise 
    detected_idx = np.argmax(np.abs(y)**2)
    true_idx = np.argmax(x)
    
    return detected_idx == true_idx

def simulate_ssk_transmission_direct(x, B, effective_channel, sigma_sq):
    noise = create_random_noise_vector(len(x), sigma_sq)
    y = (B + effective_channel) @ x + noise 
    distances = np.array([np.linalg.norm(y - B[:, i]) for i in range(B.shape[1])])
    detected_idx = np.argmin(distances)
    true_idx = np.argmax(x)
    
    return detected_idx == true_idx

def calculate_ber_simulation(snr_db, K, N, J, M, eta=0.9, num_symbols=10000):
    sigma_sq = snr_db_to_sigma_sq(snr_db)
    errors_receiver = 0
    errors_eavesdropper = 0
    errors_direct = 0
    
    for _ in range(num_symbols):
        H = generate_random_channel_matrix(N, K)
        Gs = [generate_random_channel_matrix(K, N) for _ in range(J)]
        G = Gs[0]
        F = generate_random_channel_matrix(K, N)
        B = generate_random_channel_matrix(K, K)
        Ps, _ = calculate_multi_ris_reflection_matrices(
            K, N, J, M, Gs, H, eta
        )
        P = unify_ris_reflection_matrices(Ps)

        x = np.zeros(K)
        x[np.random.randint(K)] = 1

        effective_channel_receiver = G @ P @ H
        effective_channel_eavesdropper = F @ P @ H
        effective_channel_direct = np.zeros((K, K))

        if not simulate_ssk_transmission_reflection(x, effective_channel_receiver, sigma_sq):
            errors_receiver += 1
        
        if not simulate_ssk_transmission_direct(x, B, effective_channel_eavesdropper, sigma_sq):
            errors_eavesdropper += 1

        if not simulate_ssk_transmission_direct(x, B, effective_channel_direct, sigma_sq):
            errors_direct += 1
    
    result_receiver = errors_receiver / num_symbols
    result_eavesdropper = errors_eavesdropper / num_symbols
    result_direct = errors_direct / num_symbols

    return result_receiver, result_eavesdropper, result_direct

def plot_ber_curves():
    N = 16    # * Number of reflecting elements
    K = 2     # * Number of antennas 
    eta = 0.9 # * Reflection efficiency
    
    # for all combinations of J and M between 1 and 2
    for J in range(1, 3):  # * Number of receivers
        for M in range(1, 3):  # * Number of RIS surfaces
            print(f"Processing J={J}, M={M}")
            snr_range_db = np.arange(-10, 31, 2)
            ber_theoretical = []
            ber_simulated_receiver = []
            ber_simulated_eavesdropper = []
            ber_simulated_direct = []
            
            for snr_db in snr_range_db:
                ber_theoretical.append(calculate_ber_theoretical(snr_db, K, N))

                result_receiver, result_eavesdropper, result_direct = calculate_ber_simulation(snr_db, K, N, J, M, eta)
                ber_simulated_receiver.append(result_receiver)
                ber_simulated_eavesdropper.append(result_eavesdropper)
                ber_simulated_direct.append(result_direct)
                print(f"Processed SNR = {snr_db} dB:\t{result_receiver:.2f}/\t{result_eavesdropper:.2f}/\t{result_direct:.2f}")

            plt_name = f'SSK BER Performance with RIS (K={K}, N={N}, J={J}, M={M})'
            plt.figure(figsize=(10, 6))
            # plt.semilogy(snr_range_db, ber_theoretical, label='Theoretical Receiver')
            plt.semilogy(snr_range_db, ber_simulated_receiver, label='Simulation Receiver')
            plt.semilogy(snr_range_db, ber_simulated_eavesdropper, label=f'Simulation Eavesdropper')
            plt.semilogy(snr_range_db, ber_simulated_direct, label=f'Simulation Direct')
            plt.grid(True)
            plt.xlabel('SNR (dB)')
            plt.ylabel('Bit Error Rate (BER)')
            plt.title(plt_name)
            plt.legend()
            plt.savefig(f"./simulations/results/{plt_name}.png")
            print(f"Saved {plt_name}.png\n\n")

if __name__ == "__main__":
    plot_ber_curves()