import numpy as np
from scipy.linalg import sqrtm
from scipy import stats, integrate
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable
from diagonalization import generate_random_channel_matrix, calculate_multi_ris_reflection_matrices, unify_ris_reflection_matrices

def snr_db_to_sigma_sq(snr_db, path_gain = 1):
    '''
    Convert SNR in dB to noise variance (sigma squared).

    Args:
        snr_db: Signal-to-noise ratio in dB
        path_gain: Path gain (default is 1)

    Returns:
        sigma_sq: Noise variance
    '''
    snr_linear = 10**(snr_db/10)
    sigma_sq = path_gain / snr_linear
    return sigma_sq

def create_random_noise_vector_from_snr(K: int, snr_db: int, path_gain = 1) -> np.ndarray:
    """
    Generate a random noise vector with complex Gaussian entries, from SNR in dB.
    
    Args:
        K: Number of elements in the noise vector
        sigma_sq: Noise variance
    
    Returns:
        Random noise vector
    """
    sigma_sq = snr_db_to_sigma_sq(snr_db, path_gain)
    mu = np.sqrt(sigma_sq/2) * (
        np.random.randn(K) + 1j*np.random.randn(K)
    )
    
    return mu

def create_random_noise_vector_from_noise_floor(K: int, temp_kelvin = 290, f = 400) -> np.ndarray:
    """
    Generate a random noise vector with complex Gaussian entries, from noise variance.
    
    Args:
        K: Number of elements in the noise vector
        temp_kelvin: Temperature in Kelvin (default is 290)
        f: Frequency in MHz (default is 400 MHz)
    
    Returns:
        Random noise vector
    """
    botzmann_constant = 1.380649e-23
    noise_figure = 6 

    # P_dbm = -80
    # P_mw = 10**(P_dbm/10) 
    
    P_mw = botzmann_constant * temp_kelvin * (f * 1000000) * 1000 * noise_figure
    # P_dbm = 10 * np.log10(P_mw)

    mu = np.random.randn(K) + 1j*np.random.randn(K)
    mu = mu * np.sqrt(P_mw)
    
    return mu


######## Secrecy Rate Calculation ########

def calculate_receiver_term(G: np.ndarray, P: np.ndarray, H: np.ndarray, xi: np.ndarray, xj: np.ndarray, mu: np.ndarray, sigma_sq: float): 
    try: 
        a = G @ P @ H @ (xi - xj) + mu
        a = np.linalg.norm(a) ** 2
        b = np.linalg.norm(mu) ** 2
        return (-a + b) / sigma_sq
    except ValueError as e:
        print(f"Error at calculate_receiver_term: {e}")
        raise e

def calculate_eavesdropper_Sigma_inv_sqrt(K: int, N: int, eta: float, G: np.ndarray, H: np.ndarray, F: np.ndarray, xi: np.ndarray, xj: np.ndarray, sigma_sq: float, num_samples = 100):
    try:
        expected = 0
        for _ in range(num_samples):
            Ps, _ = calculate_multi_ris_reflection_matrices(K, N, 1, 1, [G], H, eta, [])
            P = unify_ris_reflection_matrices(Ps, [])
            a = F @ P @ H @ (xi - xj) @ (xi - xj).T.conj() @ H.T.conj() @ P.T.conj() @ F.T.conj()
            expected += a
        expected /= num_samples
        Sigma = expected + sigma_sq * np.eye(K)

        # Sigma_sqrt = Sigma ** -0.5
        # Sigma_inv_sqrt = np.linalg.inv(Sigma_sqrt)

        # Calculate Σ^(-1/2) using eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(Sigma)
        # Ensure eigenvalues are positive
        eigenvals = np.maximum(eigenvals, 1e-12)
        # Calculate inverse square root
        Sigma_inv_sqrt = eigenvecs @ np.diag(1/np.sqrt(eigenvals)) @ eigenvecs.conj().T

        return Sigma_inv_sqrt
    except ValueError as e:
        print(f"Error at calculate_eavesdropper_Sigma_inv_sqrt: {e}")
        raise e

def calculate_eavesdropper_noise(Sigma_inv_sqrt: np.ndarray, P: np.ndarray, H: np.ndarray, F: np.ndarray, xi: np.ndarray, xj: np.ndarray, mu: np.ndarray):
    try:
        a = F @ P @ H @ (xi - xj) + mu
        return Sigma_inv_sqrt @ a
    except ValueError as e:
        print(f"Error at calculate_eavesdropper_noise: {e}")
        raise e
    
def calculate_eavesdropper_term(K: int, N: int, eta: float, G: np.ndarray, P: np.ndarray, H: np.ndarray, F: np.ndarray, B: np.ndarray, xi: np.ndarray, xj: np.ndarray, mu: np.ndarray, sigma_sq: float, num_samples = 100):
    try:
        Sigma_inv_sqrt = calculate_eavesdropper_Sigma_inv_sqrt(K, N, eta, G, H, F, xi, xj, sigma_sq, num_samples)
        mu_prime = calculate_eavesdropper_noise(Sigma_inv_sqrt, P, H, F, xi, xj, mu)
        a = Sigma_inv_sqrt @ B @ (xi - xj) + mu_prime
        a = np.linalg.norm(a) ** 2
        b = np.linalg.norm(mu_prime) ** 2
        return -a + b
    except ValueError as e:
        print(f"Error at calculate_eavesdropper_term: {e}")
        raise e
    
def calculate_general_secrecy_rate(K: int, calculate_term: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray], snr_db: int, num_samples = 100):
    try:
        sum_i = 0
        for i in range(K):
            xi = np.zeros((K, 1))
            xi[i] = 1
            expected = 0
            for _ in range(num_samples):
                mu = create_random_noise_vector_from_snr(K, snr_db)
                sum_j = 0
                for j in range(K):
                    xj = np.zeros((K, 1))
                    xj[j] = 1
                    term = calculate_term(xi, xj, mu)
                    sum_j += np.exp(term)
                expected += np.log2(sum_j)
            expected /= num_samples
            sum_i += expected
        return np.log2(K) - sum_i / K
    except ValueError as e:
        print(f"Error at calculate_general_secrecy_rate: {e}")
        raise e
    
def calculate_secrecy_rate(N: int, K: int, G: np.ndarray, P: np.ndarray, H: np.ndarray, B: np.ndarray, F: np.ndarray, snr_db: int, eta: float, num_samples = 100):
    try:
        sigma_sq = snr_db_to_sigma_sq(snr_db)
        def calculate_receiver_term_wrapper(xi: np.ndarray, xj: np.ndarray, mu: np.ndarray):
            return calculate_receiver_term(G, P, H, xi, xj, mu, sigma_sq)
        def calculate_eavesdropper_term_wrapper(xi: np.ndarray, xj: np.ndarray, mu: np.ndarray):
            return calculate_eavesdropper_term(K, N, eta, G, P, H, F, B, xi, xj, mu, sigma_sq, num_samples)
        receiver_secrecy_rate = calculate_general_secrecy_rate(K, calculate_receiver_term_wrapper, snr_db, num_samples)
        eavesdropper_secrecy_rate = calculate_general_secrecy_rate(K, calculate_eavesdropper_term_wrapper, snr_db, num_samples)
        secrecy_rate = receiver_secrecy_rate - eavesdropper_secrecy_rate
        return secrecy_rate, receiver_secrecy_rate, eavesdropper_secrecy_rate
    except ValueError as e:
        print(f"Error at calculate_secrecy_rate: {e}")
        raise e
    
######## MAIN ###################

def main():
    N = 16    # * Number of reflecting elements
    K = 2     # * Number of antennas
    J = 2     # * Number of receivers
    M = 1     # * Number of RIS surfaces
    E = 10    # * Number of eavesdroppers
    eta = 0.9 # * Reflection efficiency
    snr_range_db = np.arange(-20, 21, 2)  # * SNR range

    print(f"Parameters: \n- RIS surfaces = {M}\n- Elements per RIS = {N}\n- Reflection efficiency = {eta}\n- Receiver Antennas = {K}")

    H = generate_random_channel_matrix(N, K)
    B = generate_random_channel_matrix(K, K)
    Gs = [generate_random_channel_matrix(K, N) for _ in range(J)]
    Es = [generate_random_channel_matrix(K, N) for _ in range(E)]
    Cs = [generate_random_channel_matrix(N, N) for _ in range(M-1)]
    G = Gs[0]
    F = Es[0]

    try:
        Ps, _ = calculate_multi_ris_reflection_matrices(K, N, J, M, Gs, H, eta, Cs)
        P = unify_ris_reflection_matrices(Ps, Cs)
        secrecy_rates = []
        receiver_secrecy_rates = []
        eavesdropper_secrecy_rates = []
        for snr_db in snr_range_db:
            secrecy_rate, receiver_secrecy_rate, eavesdropper_secrecy_rate = calculate_secrecy_rate(N, K, G, P, H, B, F, snr_db, eta)
            print(f"SNR: {snr_db}, Secrecy Rate: {secrecy_rate:.2f} ({receiver_secrecy_rate:.2f} - {eavesdropper_secrecy_rate:.2f}) bits/s/Hz")
            secrecy_rates.append(secrecy_rate)
            receiver_secrecy_rates.append(receiver_secrecy_rate)
            eavesdropper_secrecy_rates.append(eavesdropper_secrecy_rate)

        plt_name = "Secrecy Rate vs SNR"
        plt.plot(snr_range_db, secrecy_rates, label="Secrecy Rate")
        plt.plot(snr_range_db, receiver_secrecy_rates, label="Receiver Secrecy Rate")
        plt.plot(snr_range_db, eavesdropper_secrecy_rates, label="Eavesdropper Secrecy Rate")
        plt.legend()
        plt.xlabel("SNR (dB)")
        plt.ylabel("Secrecy Rate (bits/s/Hz)")
        plt.title(plt_name)
        plt.grid()
        plt.savefig(f"./simulations/results/{plt_name}.png", dpi=300, format='png')
        print(f"Saved {plt_name}.png\n\n")
        
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()