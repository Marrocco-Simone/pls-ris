import numpy as np

def print_low_array(v: np.ndarray) -> str:
    return print(np.array2string(np.abs(v), formatter={'float_kind':lambda x: '{:.1e}'.format(x)}))

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
    columns, rows = H.shape
    power = 0
    for i in range(columns):
        h_i = H[i, :]
        power += np.linalg.norm(h_i) ** 2
    return power / columns

def calculate_signal_power(signal: np.ndarray) -> float:
    '''
    Calculate the signal power of a given signal vector

    Parameters:
    -----------
    signal : Complex signal vector

    Returns:
    --------
    Signal power
    '''
    return np.linalg.norm(signal) ** 2

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

def calculate_noise_floor_in_mw(temp_kelvin = 290, f = 400):
    botzmann_constant = 1.380649e-23
    noise_figure = 6

    # P_dbm = -80
    # P_mw = 10**(P_dbm/10)

    P_mw = botzmann_constant * temp_kelvin * (f * 1000000) * 1000 * noise_figure
    # P_dbm = 10 * np.log10(P_mw)

    return P_mw

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
    P_mw = calculate_noise_floor_in_mw(temp_kelvin, f)

    # TODO GENERATE THESE WITH A RICE DISTRIBUTION
    mu = np.random.randn(K) + 1j*np.random.randn(K)
    mu = mu * np.sqrt(P_mw)

    return mu

def main():
    # test noise floor
    Pn_dbm_1 = -80
    Pn_mw_1 = 10**(Pn_dbm_1/10)

    Pn_mw_2 = calculate_noise_floor_in_mw()
    Pn_dbm_2 = 10 * np.log10(Pn_mw_2)

    diff_dbm = abs(Pn_dbm_1 - Pn_dbm_2)
    diff_mw = abs(Pn_mw_1 - Pn_mw_2)

    print(f"Noise floor power comparisons:")
    print(f"\t{Pn_dbm_1:.2f} dbm == {Pn_dbm_2:.2f} dbm ({abs(diff_dbm / Pn_dbm_1 * 100):.2f} %)")
    print(f"\t{Pn_mw_1:.2e} mw == {Pn_mw_2:.2e} mw ({abs(diff_mw / Pn_mw_1 * 100):.2f} %)")

    print("\n--------------------\n")

    # test signal power calculation
    Pt_dbm = 12
    Pt_mw = 10**(Pt_dbm/10)
    Pn_mw = Pn_mw_2
    Pn_dbm = Pn_dbm_2

    K = 4
    x = np.zeros(K)
    x[0] = 1
    x = x * np.sqrt(Pt_mw)
    noise = create_random_noise_vector_from_noise_floor(K)

    x_power_mw = calculate_signal_power(x)
    x_power_dbm = 10 * np.log10(x_power_mw)
    noise_power_mw = calculate_signal_power(noise)
    noise_power_dmb = 10 * np.log10(noise_power_mw)

    print(f"Signal power comparisons:")
    print_low_array(x)
    print(f"\t{x_power_dbm:.2f} / {x_power_mw:.2e} == {Pt_dbm:.2f} / {Pt_mw:.2e}")
    print_low_array(noise)
    print(f"\t{noise_power_dmb:.2f} / {noise_power_mw:.2e} == {Pn_dbm:.2f} / {Pn_mw:.2e}")

if __name__ == "__main__":
    main()