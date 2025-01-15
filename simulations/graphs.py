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

def calculate_achievable_rate_receiever_term(
    G: np.ndarray, 
    P: np.ndarray, 
    H: np.ndarray, 
    x_i: np.ndarray, 
    x_l: np.ndarray, 
    sigma_sq: float
):
    mu = create_random_noise_vector(K, sigma_sq)
    received = G @ P @ H @ (x_i - x_l) + mu
    term = -np.linalg.norm(received)**2 + np.linalg.norm(mu)**2
    term = term/sigma_sq
    return term
    

def calculate_achievable_rate_receiever(
    K: int,
    G: np.ndarray, 
    P: np.ndarray, 
    H: np.ndarray, 
    sigma_sq: float, 
    num_samples=10000
):
    sum_i = 0
    for i in range(K):
        x_i = np.zeros(K)
        x_i[i] = 1
        sum_exp = 0    
        for _ in range(num_samples):
            sum_l = 0
            for l in range(K):
                x_l = np.zeros(K)
                x_l[l] = 1
                term = calculate_achievable_rate_receiever_term(
                    G, P, H, x_i, x_l, sigma_sq
                )
                sum_l += np.exp(term)
            sum_l = np.log2(sum_l)
            sum_exp += sum_l
        expected = sum_exp/num_samples
        sum_i += expected
    return np.log2(K) - sum_i / K

def calculate_achievable_rate_eavesdropper_Sigma(
    N: int,
    K: int,
    F: np.ndarray,  
    H: np.ndarray, 
    diff: np.ndarray, 
    sigma_sq: float,
    num_samples=10000 
):
    sum_result = np.zeros((M, M), dtype=complex)
    for _ in range(num_samples):
        P = generate_random_channel_matrix(N, N)
        term = F @ P @ H @ diff @ diff.conj().T @ H.conj().T @ P.conj().T @ F.conj().T
        sum_result += term
    expected= sum_result / num_samples
    return expected + sigma_sq * np.eye(K)

def calculate_achievable_rate_eavesdropper_term(
    N: int,
    K: int,
    B: np.ndarray, 
    F: np.ndarray,  
    P: np.ndarray, 
    H: np.ndarray, 
    x_i: np.ndarray, 
    x_l: np.ndarray, 
    sigma_sq: float,
    num_samples=10000 
):
    mu = create_random_noise_vector(K, sigma_sq)
    diff = x_i - x_l
    Sigma = calculate_achievable_rate_eavesdropper_Sigma(
        N, K, F, H, diff, sigma_sq, num_samples
    )
    Sigma_inv_sqrt = np.linalg.inv(sqrtm(Sigma))
    interference = F @ P @ H @ (x_i - x_l)
    v_prime = Sigma_inv_sqrt @ (interference + mu)
    received = Sigma_inv_sqrt @ B @ diff + v_prime
    term = -np.linalg.norm(received)**2 + np.linalg.norm(mu)**2
    return term

def calculate_achievable_rate_eavesdropper(
    N: int,
    K: int,
    B: np.ndarray, 
    F: np.ndarray, 
    P: np.ndarray, 
    H: np.ndarray, 
    sigma_sq: float, 
    num_samples=10000 
):
    sum_i = 0
    for i in range(K):
        x_i = np.zeros(K)
        x_i[i] = 1
        sum_exp = 0    
        for _ in range(num_samples):
            sum_l = 0
            for l in range(K):
                x_l = np.zeros(K)
                x_l[l] = 1
                term = calculate_achievable_rate_eavesdropper_term(
                    N, K, B, F, P, H, x_i, x_l, sigma_sq, num_samples
                )
                sum_l += np.exp(term)
            sum_l = np.log2(sum_l)
            sum_exp += sum_l
        expected = sum_exp/num_samples
        sum_i += expected
    return np.log2(K) - sum_i / K

def calculate_secrecy_rate(
    N: int,
    K: int,
    G: np.ndarray,
    P: np.ndarray,
    H: np.ndarray,
    B: np.ndarray,
    F: np.ndarray,
    sigma_sq: float,
    num_samples=10000  
):
    R_receiver = calculate_achievable_rate_receiever(
        K, G, P, H, sigma_sq, num_samples
    )
    R_eavesdropper = calculate_achievable_rate_eavesdropper(
        N, K, B, F, P, H, sigma_sq, num_samples
    )
    secrecy_rate = R_receiver - R_eavesdropper
    if secrecy_rate < 0:
        return 0
    return secrecy_rate

######## BER Calculation ########
    
######## MAIN ###################

if __name__ == "__main__":
    N = 16    # * Number of reflecting elements
    K = 2     # * Number of antennas
    J = 2     # * Number of receivers
    M = 1     # * Number of RIS surfaces
    E = 10    # * Number of eavesdroppers
    eta = 0.9 # * Reflection efficiency
    sigma_sq = 1.0 # * Noise variance
    snr_range_db = np.linspace(0, 30, 16)  # SNR range from 0 to 30 dB

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
        secrecy_rate = calculate_secrecy_rate(N, K, G, P, H, B, F, sigma_sq)
        print(f"Secrecy Rate: {secrecy_rate:.2f} bits/s/Hz")
        
    except ValueError as e:
        print(f"Error: {e}")