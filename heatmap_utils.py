import numpy as np
from noise_power_utils import (
  calculate_signal_power
)

def calculate_signal_power_from_channel_using_ssk(K: int, H: np.ndarray, Pt_dbm = 0.0):
    signal_power = 0.0
    # just to make comparison more equal. But dont put this too high, otherwise it will take too long
    n_sim = 10
    for _ in range(n_sim):
        index = np.random.randint(0, K)
        x = np.zeros(K)
        x[index] = 1
        Pt_mw = 10**(Pt_dbm/10)
        x = x * np.sqrt(Pt_mw)
        signal_power += calculate_signal_power(H @ x) / n_sim

    return signal_power

def calculate_free_space_path_loss(d: float, lam = 0.08, k = 2) -> float:
    """
    Calculate free space path loss between transmitter and receiver

    Parameters:
    -----------
    d : Distance between transmitter and receiver in meters
    lam : Wavelength of the signal (default 5G = 80mm)
    k : Exponent of path loss model (default 2)

    Returns:
    --------
    Free space path loss in dB
    """
    if d == 0: d = 0.01
    return 1 / np.sqrt((4 * np.pi / lam) ** 2 * d ** k)

def calculate_unit_spatial_signature(incidence: float, K: int, delta: float):
    """
    Calculate the unit spatial signature vector for a given angle of incidence

    Parameters:
    -----------
    incidence : Angle of incidence in radians
    K : Number of antennas
    delta : Distance between antennas in meters

    Returns:
    --------
    Unit spatial signature vector of shape (K, 1)
    """
    directional_cosine = np.cos(incidence)
    e = np.array([(1 / np.sqrt(K)) * np.exp(-1j * 2 * np.pi * (k - 1) * delta * directional_cosine) for k in range(K)])
    return e.reshape(-1, 1)

def generate_rice_matrix(L: int, K: int, nu: float, sigma = 1.0) -> np.ndarray:
    """
    Generate a Ricean channel matrix for a given number of transmit and receive antennas

    Parameters:
    -----------
    L : Number of transmit antennas
    K : Number of receive antennas
    nu : Mean magnitude of each matrix element
    sigma : Standard deviation of the complex Gaussian noise

    Returns:
    --------
    Ricean channel matrix of shape (K, L)
    """
    return np.random.normal(nu/np.sqrt(2), sigma, (K, L)) + 1j * np.random.normal(nu/np.sqrt(2), sigma, (K, L))

def generate_rice_faiding_channel(L: int, K: int, ratio: float, total_power = 1.0) -> np.ndarray:
    """
    Generate a Ricean fading channel matrix for a given number of transmit and receive antennas

    Parameters:
    -----------
    L : Number of transmit antennas
    K : Number of receive antennas
    ratio : Ratio of directed path compared to the other paths
    total_power : Total power from all paths
    """
    nu = np.sqrt(ratio * total_power / (1 + ratio))
    sigma = np.sqrt(total_power / (2 * (1 + ratio)))
    return generate_rice_matrix(L, K, nu, sigma)

def calculate_mimo_channel_gain(d: float, L: int, K: int, lam = 0.08, k = 2) -> np.ndarray:
    """
    Calculate MIMO channel gains between transmitter and receiver

    Parameters:
    -----------
    d : Distance between transmitter and receiver in meters
    L : Number of transmit antennas
    K : Number of receive antennas
    lam : Wavelength of the signal (default 5G = 80mm)
    k : Exponent of path loss model (default 2)

    Returns:
    --------
    H : Complex channel gain matrix of shape (K, L)
    """
    # return generate_random_channel_matrix(K, L)
    if d == np.inf:
        return np.zeros((K, L), dtype=complex)
    if d == 0:
        d = 0.5

    delta = lam / 2
    # a = calculate_free_space_path_loss(d, lam, k)
    c = np.sqrt(L * K) * np.exp(-1j * 2 * np.pi * d / lam)
    # c = a * c
    e_r = calculate_unit_spatial_signature(0, K, delta)
    e_t = calculate_unit_spatial_signature(0, L, delta)
    H = c * (e_r @ e_t.T.conj())

    ratio = 0.6
    total_power = 1.0
    H = H * generate_rice_faiding_channel(L, K, ratio, total_power)
    return H