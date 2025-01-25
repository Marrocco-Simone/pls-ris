import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ncx2, chi2
from diagonalization import (
    calculate_W_single,
    calculate_multi_ris_reflection_matrices,
    generate_random_channel_matrix
)

def calculate_scaling_factor(K):
    """
    Calculate the scaling factor δκ for converting symbol error rate to bit error rate.
    This implements equation (69) from the paper.
    
    Args:
        K: Number of transmit/receive antennas (must be power of 2)
    
    Returns:
        Scaling factor for BER calculation
    """
    if K < 2 or (K & (K - 1)) != 0:  # Check if K is power of 2
        raise ValueError("K must be power of 2")
        
    kappa = int(np.log2(K))  # κ = log2(K)
    delta = np.zeros(kappa + 1)  # δ0 through δκ
    
    # Calculate scaling factors iteratively using equation (69)
    for k in range(1, kappa + 1):
        delta[k] = delta[k-1] + (2**(k-1) - delta[k-1])/(2**k - 1)
    
    return delta[kappa]

def calculate_ber_theoretical(snr_db, K, N, num_monte_carlo=1000):
    """
    Calculate theoretical BER using equation (65) from the paper.
    Uses Monte Carlo integration to average over random channel realizations.
    
    Args:
        snr_db: Signal-to-noise ratio in dB
        K: Number of transmit/receive antennas
        N: Number of reflecting elements
        num_monte_carlo: Number of Monte Carlo trials
    
    Returns:
        Theoretical BER value
    """
    snr_linear = 10**(snr_db/10)
    delta_k = calculate_scaling_factor(K)
    
    total_error_prob = 0
    
    # Monte Carlo integration over channel realizations
    for _ in range(num_monte_carlo):
        # Generate random channels
        H = generate_random_channel_matrix(N, K)
        G = generate_random_channel_matrix(K, N)
        
        # Calculate reflection matrix P
        Ps, _ = calculate_multi_ris_reflection_matrices(
            K, N, 1, 1, [G], H, eta=0.9
        )
        P = Ps[0]
        
        # Calculate effective channel gains
        effective_channel = G @ P @ H
        channel_gains = np.abs(np.diag(effective_channel))**2
        
        # Calculate non-centrality parameter
        lambda_i = 2 * channel_gains * snr_linear
        lambda_avg = np.mean(lambda_i)
        
        # Inner integral over t
        def integrand(t):
            # CDF of chi-square with 2 DoF
            F_chi2 = chi2.cdf(t, df=2)
            # PDF of non-central chi-square with 2 DoF
            f_chi2_nc = ncx2.pdf(t, df=2, nc=lambda_avg)
            return F_chi2**(K-1) * f_chi2_nc
            
        # Numerical integration using Simpson's rule
        t = np.linspace(0, 100, 1000)
        integral = np.trapz(integrand(t), t)
        
        total_error_prob += (1 - integral)
    
    ber = delta_k * (total_error_prob / num_monte_carlo)
    return ber

def simulate_ssk_transmission(x, effective_channel, noise_var):
    """
    Simulate SSK transmission and detection for one symbol.
    
    Args:
        x: Transmitted SSK symbol (one-hot vector)
        effective_channel: Diagonal channel matrix
        noise_var: Noise variance (1/SNR)
    
    Returns:
        Boolean indicating if detection was correct
    """
    # Generate complex Gaussian noise
    noise = np.sqrt(noise_var/2) * (
        np.random.randn(len(x)) + 1j*np.random.randn(len(x))
    )
    
    # Received signal: y = Dx + n
    y = effective_channel @ x + noise
    
    # Non-coherent detection: find max |y_i|^2
    detected_idx = np.argmax(np.abs(y)**2)
    true_idx = np.argmax(x)
    
    return detected_idx == true_idx

def calculate_ber_simulation(snr_db, K, N, num_symbols=10000):
    """
    Calculate BER through Monte Carlo simulation of SSK transmission.
    
    Args:
        snr_db: Signal-to-noise ratio in dB
        K: Number of transmit/receive antennas
        N: Number of reflecting elements
        num_symbols: Number of symbols to simulate
    
    Returns:
        Simulated BER value
    """
    snr_linear = 10**(snr_db/10)
    noise_var = 1/snr_linear
    errors_receiver = 0
    errors_eavesdroopper = 0
    errors_eavesdroopper_completed = 0
    
    for _ in range(num_symbols):
        # Generate random channels
        H = generate_random_channel_matrix(N, K)
        G = generate_random_channel_matrix(K, N)
        F = generate_random_channel_matrix(K, N)
        B = generate_random_channel_matrix(K, K)

        # Calculate reflection matrix P
        Ps, _ = calculate_multi_ris_reflection_matrices(
            K, N, 1, 1, [G], H, eta=0.9
        )
        P = Ps[0]

        # Calculate effective channel
        effective_channel_receiver = G @ P @ H
        effective_channel_eavesdropper = F @ P @ H
        effective_channel_eavesdropper_completed = effective_channel_eavesdropper + B

        # Generate random SSK symbol
        x = np.zeros(K)
        x[np.random.randint(K)] = 1
        
        # Simulate transmission and detection
        success_receiver = simulate_ssk_transmission(x, effective_channel_receiver, noise_var)
        if not success_receiver:
            errors_receiver += 1
        success_eavesdropper = simulate_ssk_transmission(x, effective_channel_eavesdropper, noise_var)
        if not success_eavesdropper:
            errors_eavesdroopper += 1
        success_eavesdropper_completed = simulate_ssk_transmission(x, effective_channel_eavesdropper_completed, noise_var)
        if not success_eavesdropper_completed:
            errors_eavesdroopper_completed += 1
    
    receivers_errors = errors_receiver / num_symbols
    eavesdroppers_errors = errors_eavesdroopper / num_symbols
    eavesdroppers_errors_completed = errors_eavesdroopper_completed / num_symbols
    
    return receivers_errors, eavesdroppers_errors, eavesdroppers_errors_completed

def plot_ber_curves():
    """
    Generate BER vs SNR plot similar to Figure 5(a) in the paper.
    """
    K = 2  # Number of antennas
    N = 16  # Number of reflecting elements
    
    snr_range_db = np.arange(0, 31, 2)
    ber_theoretical = []
    ber_simulated_receiver = []
    ber_simulated_eavesdropper = []
    ber_simulated_eavesdropper_completed = []
    
    for snr_db in snr_range_db:
        print(f"Processing SNR = {snr_db} dB...")
        ber_theoretical.append(calculate_ber_theoretical(snr_db, K, N))
        receivers_errors, eavesdroppers_errors, eavesdroppers_errors_completed = calculate_ber_simulation(snr_db, K, N)
        ber_simulated_receiver.append(receivers_errors)
        ber_simulated_eavesdropper.append(eavesdroppers_errors)
        ber_simulated_eavesdropper_completed.append(eavesdroppers_errors_completed)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range_db, ber_theoretical, 'b-', label='Theoretical Receiver')
    plt.semilogy(snr_range_db, ber_simulated_receiver, 'r-', label='Simulation Receiver')
    plt.semilogy(snr_range_db, ber_simulated_eavesdropper, 'g-', label='Simulation Eavesdropper')
    plt.semilogy(snr_range_db, ber_simulated_eavesdropper_completed, 'm-', label='Simulation Eavesdropper (Completed)')
    plt.grid(True)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('SSK BER Performance with RIS')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_ber_curves()