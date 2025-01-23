import numpy as np
from scipy.linalg import sqrtm
from scipy import stats, integrate
import matplotlib.pyplot as plt
from typing import List, Tuple
from diagonalization import generate_random_channel_matrix, calculate_multi_ris_reflection_matrices, unify_ris_reflection_matrices

def create_random_noise_vector(K: int, sigma_sq: float) -> np.ndarray:
    """
    Generate a random noise vector with complex Gaussian entries.
    
    Args:
        K: Number of elements in the noise vector
        sigma_sq: Noise variance
    
    Returns:
        Random noise vector
    """
    mu_real = np.random.normal(0, np.sqrt(sigma_sq/2), K)
    mu_imag = np.random.normal(0, np.sqrt(sigma_sq/2), K)
    mu = mu_real + 1j*mu_imag
    
    return mu

######## Secrecy Rate Calculation ########

def calculate_receiver_term(G: np.ndarray, P: np.ndarray, H: np.ndarray, xi: np.ndarray, xj: np.ndarray, mu: np.ndarray, sigma_sq: float): 
    try: 
        a = G @ P @ H @ (xi - xj) + mu
        a = np.linalg.norm(a, ord='fro') ** 2
        b = np.linalg.norm(mu, ord='fro') ** 2
        return (-a + b) / sigma_sq
    except ValueError as e:
        print(f"Error at calculate_receiver_term: {e}")
        raise e

def calculate_eavesdropper_Sigma_inv_sqrt(K: int, N: int, eta: float, G: np.ndarray, H: np.ndarray, F: np.ndarray, xi: np.ndarray, xj: np.ndarray, sigma_sq: float, num_samples=1000):
    try:
        expected = 0
        for _ in range(num_samples):
            Ps, _ = calculate_multi_ris_reflection_matrices(K, N, 1, 1, [G], H, eta)
            P = unify_ris_reflection_matrices(Ps)
            a = F @ P @ H @ (xi - xj) @ (xi - xj).T.conj() @ H.T.conj() @ P.T.conj() @ F.T.conj()
            expected += a
        expected /= num_samples
        Sigma = expected + sigma_sq * np.eye(K)
        Sigma_inv_sqrt = np.linalg.inv(sqrtm(Sigma))
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
    
def calculate_eavesdropper_term(K: int, N: int, eta: float, G: np.ndarray, P: np.ndarray, H: np.ndarray, F: np.ndarray, B: np.ndarray, xi: np.ndarray, xj: np.ndarray, mu: np.ndarray, sigma_sq: float, num_samples=1000):
    try:
        mu = create_random_noise_vector(K, sigma_sq)
        Sigma_inv_sqrt = calculate_eavesdropper_Sigma_inv_sqrt(K, N, eta, G, H, F, xi, xj, sigma_sq, num_samples)
        mu_prime = calculate_eavesdropper_noise(Sigma_inv_sqrt, P, H, F, xi, xj, mu)
        a = Sigma_inv_sqrt @ B @ (xi - xj) + mu_prime
        a = np.linalg.norm(a, ord='fro') ** 2
        b = np.linalg.norm(mu_prime, ord='fro') ** 2
        return -a + b
    except ValueError as e:
        print(f"Error at calculate_eavesdropper_term: {e}")
        raise e
    
######## MAIN ###################

def main():
    N = 16    # * Number of reflecting elements
    K = 2     # * Number of antennas
    J = 2     # * Number of receivers
    M = 1     # * Number of RIS surfaces
    E = 10    # * Number of eavesdroppers
    eta = 0.9 # * Reflection efficiency
    snr_range_db = np.arange(-20, 21)  # * SNR range from 0 to 30 dB

    print(f"Parameters: \n- RIS surfaces = {M}\n- Elements per RIS = {N}\n- Reflection efficiency = {eta}\n- Receiver Antennas = {K}")

    H = generate_random_channel_matrix(N, K)
    B = generate_random_channel_matrix(K, K)
    Gs = [generate_random_channel_matrix(K, N) for _ in range(J)]
    Es = [generate_random_channel_matrix(K, N) for _ in range(E)]
    G = Gs[0]
    F = Es[0]

    try:
        Ps, dor = calculate_multi_ris_reflection_matrices(K, N, J, M, Gs, H, eta)
        P = unify_ris_reflection_matrices(Ps)
        secrecy_rates = []
        for snr_db in snr_range_db:
            sigma_sq = 10**(-snr_db/10)
            secrecy_rate = calculate_secrecy_rate(N, K, G, P, H, B, F, sigma_sq)
            print(f"SNR: {snr_db}, Secrecy Rate: {secrecy_rate:.2f} bits/s/Hz")
            secrecy_rates.append(secrecy_rate)
        
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()