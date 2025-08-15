import numpy as np

def calculate_channel_power(H: np.ndarray) -> float:
    '''
    Calculate the channel power of a given channel matrix H

    Parameters:
    -----------
    H : Complex channel matrix of shape (K, L)

    Returns:
    --------
    Channel power
    '''
    # return np.linalg.norm(H) ** 2
    if H.ndim == 1:
        return np.linalg.norm(H) ** 2

    columns, rows = H.shape
    power = 0
    for i in range(columns):
        h_i = H[i, :]
        power += np.linalg.norm(h_i) ** 2
    return power / columns

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