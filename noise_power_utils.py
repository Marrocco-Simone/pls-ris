import numpy as np
from tqdm import tqdm

def ndarray_to_string(v: np.ndarray) -> str:
    return np.array2string(np.abs(v), formatter={'float_kind':lambda x: '{:.1e}'.format(x)})

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

def create_rice_vector(K: int) -> np.ndarray:
    """
    Generate a random Rice vector with complex Gaussian entries.

    Args:
        K: Number of elements in the Rice vector

    Returns:
        Random Rice vector
    """
    mu = np.random.randn(K) + 1j*np.random.randn(K)
    # mu = mu / np.linalg.norm(mu)

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
    P_mw = calculate_noise_floor_in_mw(temp_kelvin, f)
    mu = create_rice_vector(K)
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
    Pt_dbm = 20
    Pt_mw = 10**(Pt_dbm/10)
    Pn_mw = Pn_mw_2
    Pn_dbm = Pn_dbm_2

    K = 4
    x_full = np.zeros(K)
    x_full[0] = 1
    # noise = create_random_noise_vector_from_noise_floor(K)
    noise_full = create_rice_vector(K)

    x_full_power_mw = calculate_signal_power(x_full)
    x_full_power_dbm = 10 * np.log10(x_full_power_mw)
    noise_full_power_mw = calculate_signal_power(noise_full)
    noise_full_power_dmb = 10 * np.log10(noise_full_power_mw)

    print(f"Full signal power comparisons:")
    print(f"x:\t{ndarray_to_string(x_full)}")
    print(f"Power:\t{x_full_power_dbm:.2f} dbm == {x_full_power_mw:.2e} mw")
    print()
    print(f"noise:\t{ndarray_to_string(noise_full)}")
    print(f"Power:\t{noise_full_power_dmb:.2f} dbm == {noise_full_power_mw:.2e} mw")
    print()
    print(f"SNR: {x_full_power_mw / noise_full_power_mw:.2e} mw == {x_full_power_dbm - noise_full_power_dmb:.2f} dbm")

    print("\n--------------------\n")

    x = x_full * np.sqrt(Pt_mw)
    noise = noise_full * np.sqrt(Pn_mw)

    x_power_mw = calculate_signal_power(x)
    x_power_dbm = 10 * np.log10(x_power_mw)
    noise_power_mw = calculate_signal_power(noise)
    noise_power_dbm = 10 * np.log10(noise_power_mw)

    print(f"Signal power comparisons:")
    print(f"x:\t{ndarray_to_string(x)}")
    print(f"Power:\t{x_power_dbm:.2f} dbm == {x_power_mw:.2e} mw calculated\n\t{Pt_dbm:.2f} dbm == {Pt_mw:.2e} mw given")
    print()
    print(f"noise:\t{ndarray_to_string(noise)}")
    print(f"Power:\t{noise_power_dbm:.2f} dbm == {noise_power_mw:.2e} mw calculated\n\t{Pn_dbm:.2f} dbm == {Pn_mw:.2e} mw given")
    print()
    print(f"SNR:\n\t{x_power_dbm - noise_power_dbm:.2f} dbm == {x_power_mw / noise_power_mw:.2e} mw calculated\n\t{Pt_dbm - Pn_dbm:.2f} dbm == {Pt_mw / Pn_mw:.2e} mw given")

    print("\n--------------------\n")

    n = 500000
    # simualate n noise vectors, and calculate the average noise power
    noise_power_mw_mean = 0
    for _ in tqdm(range(n)):
        noise = create_random_noise_vector_from_noise_floor(K)
        noise_power_mw_mean += calculate_signal_power(noise) / n

    noise_power_dbm_mean = 10 * np.log10(noise_power_mw_mean)
    print(f"Average noise power over {n:.0e} samples:\n\t{noise_power_dbm_mean:.2f} dbm == {noise_power_mw_mean:.2e} mw calculated\n\t{Pn_dbm:.2f} dbm == {Pn_mw:.2e} mw given")

if __name__ == "__main__":
    main()