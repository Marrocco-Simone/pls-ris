import numpy as np
from typing import List, Tuple

tolerance = 1e-10

def calculate_W_single(K: int, N: int, G: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Calculate W matrix for a single receiver.
    
    Args:
        K: Number of antennas
        N: Number of reflecting elements
        G: Channel matrix from RIS to receiver (KxN)
        H: Channel matrix from transmitter to RIS (NxK)
    
    Returns:
        W: The W matrix as defined in equation (8) of the paper
    """
    W = np.zeros((N, N), dtype=complex)
    
    for i in range(K):
        for j in range(K):
            if i != j:
                # * g_j ⊙ h_i^T
                temp = np.multiply(G[j, :], H[:, i].T)
                # * (g_j ⊙ h_i^T)^H (g_j ⊙ h_i^T)
                W += np.outer(temp.conj(), temp)
    
    return W

def calculate_W_multiple(K: int, N: int, J: int, Gs: List[np.ndarray], H: np.ndarray) -> np.ndarray:
    """
    Calculate combined W matrix for multiple receivers.
    
    Args:
        K: Number of antennas
        N: Number of reflecting elements
        J: Number of receivers
        Gs: List of channel matrices from RIS to each receiver [G_1, G_2, ..., G_J]
        H: Channel matrix from transmitter to RIS
    
    Returns:
        W: The stacked W matrix for all receivers
    """
    W_combined = np.zeros((J * N, N), dtype=complex)
    
    for j in range(J):
        W_j = calculate_W_single(K, N, Gs[j], H)
        W_combined[j*N:(j+1)*N, :] = W_j
    
    return W_combined

def generate_random_channel_matrix(rows: int, cols: int) -> np.ndarray:
    """
    Generate a random complex channel matrix.
    
    Args:
        rows: Number of rows
        cols: Number of columns
    
    Returns:
        Random complex channel matrix
    """
    return (np.random.normal(0, 1, (rows, cols)) + 1j * np.random.normal(0, 1, (rows, cols))) / np.sqrt(2)

def calculate_ris_reflection_matrice(
    K: int, 
    N: int, 
    J: int, 
    Gs: List[np.ndarray], 
    H: np.ndarray, 
    eta: float,
) -> Tuple[np.ndarray, float]:
    """
    Calculate reflection matrices for M RIS surfaces.
    
    Args:
        K: Number of antennas
        N: Number of reflecting elements per RIS
        J: Number of receivers
        Gs: List of channel matrices from last RIS to receivers [G_1, G_2, ..., G_J]
        H: Channel matrix from transmitter to first RIS
        eta: Reflection efficiency
    
    Returns:
        P: diagonal reflection matrice
        dor: Degree of randomness achieved
    """
    W = calculate_W_multiple(K, N, J, Gs, H)
    R, sigma, Vh = np.linalg.svd(W)
    
    # * last N-JK²+JK columns of R and rows of Vh
    null_space_dim = N - J*K**2 + J*K
    if null_space_dim <= 0:
        raise ValueError(f"No solution exists. Need more reflecting elements. Current: {N}, Required: >{J*K**2 - J*K}")
    
    first_singular_values = sigma[:N - null_space_dim]
    last_singular_values = sigma[-null_space_dim:]
    first_all_are_not_zero = np.all(first_singular_values >= tolerance)
    last_all_are_zero = np.all(last_singular_values < tolerance)

    # if not first_all_are_not_zero or not last_all_are_zero:
    #     raise ValueError(f"Invalid singular values. First: {np.round(first_singular_values, 2)}, Last: {np.round(last_singular_values, 2)}")

    null_space_basis_old = R[:, -null_space_dim:] # * paper method
    null_space_basis = Vh[-null_space_dim:, :].T.conj()

    # print(f"N: {N}, J: {J}, K: {K}, null_space_dim: {null_space_dim}")
    # print(f"W shape: {W.shape}, null_space_basis_old shape: {null_space_basis_old.shape}, null_space_basis shape: {null_space_basis.shape}")
    # print(f"R shape: {R.shape}, sigma shape: {sigma.shape}, Vh shape: {Vh.shape}")

    if null_space_basis.shape != (N, null_space_dim):
        raise ValueError(f"Invalid null space basis shape: {null_space_basis.shape}, should be ({N}, {null_space_dim})")

    a = np.random.normal(0, 1, (null_space_dim,)) + 1j * np.random.normal(0, 1, (null_space_dim,))
    
    p_unnormalized = null_space_basis @ a
    p = eta * p_unnormalized / np.max(np.abs(p_unnormalized))
    P = np.diag(p)
    dor = 2 * null_space_dim
    return P, dor

def random_reflection_vector(N: int, eta: float) -> np.ndarray:
    """
    Generate a random reflection vector.
    
    Args:
        N: Number of reflecting elements
        eta: Reflection efficiency
    
    Returns:
        Random reflection vector of length N
    """
    absorptions = np.random.uniform(0, 1, N)
    phases = np.random.uniform(0, 2 * np.pi, N)
    # # * p_m[i] = eta * r_i * exp(j*theta_i)
    return eta * absorptions * np.exp(1j * phases)

def calculate_multi_ris_reflection_matrices(
    K: int, 
    N: int, 
    J: int, 
    M: int,
    Gs: List[np.ndarray], 
    H: np.ndarray, 
    eta: float,
    Cs: List[np.ndarray]
) -> Tuple[List[np.ndarray], float]:
    """
    Calculate reflection matrices for M RIS surfaces.
    
    Args:
        K: Number of antennas
        N: Number of reflecting elements per RIS
        J: Number of receivers
        M: Number of RIS surfaces
        Gs: List of channel matrices from last RIS to receivers [G_1, G_2, ..., G_J]
        H: Channel matrix from transmitter to first RIS
        eta: Reflection efficiency
        Cs: List of M-1 inter-RIS channel matrices [C_1, C_2, ..., C_(M-1)], where C_i is the channel between RIS i and i+1
    
    Returns:
        Ps: List of M diagonal reflection matrices [P_1, P_2, ..., P_M]
        dor: Degree of randomness achieved
    """
    if len(Cs) != M-1:
        raise ValueError(f"Expected {M-1} inter-RIS channel matrices, got {len(Cs)}")

    ps = []
    S = np.eye(N)

    # * Generate M-1 random reflection vectors
    for i in range(M-1):
        p_m = random_reflection_vector(N, eta)
        ps.append(p_m)
        S = S @ np.diag(p_m) @ Cs[i]

    Gs_prime = [G @ S for G in Gs]
    
    # * Calculate the last reflection matrix
    P_final, dor = calculate_ris_reflection_matrice(K, N, J, Gs_prime, H, eta)
    p_final = np.diag(P_final)
    ps.append(p_final)

    Ps = []
    for pm in ps:
        Ps.append(np.diag(pm))
    
    return Ps, dor

def unify_ris_reflection_matrices(
    Ps: List[np.ndarray],
    Cs: List[np.ndarray]
) -> np.ndarray:
    """
    Unify reflection matrices into a single matrix.
    
    Args:
        Ps: List of reflection matrices [P_1, P_2, ..., P_M]
        Cs: List of M-1 inter-RIS channel matrices [C_1, C_2, ..., C_(M-1)], where C_i is the channel between RIS i and i+1
    
    Returns:
        P: Combined reflection matrix
    """
    P = Ps[0]
    for i in range(len(Ps)-1):
        P = P @ Cs[i] @ Ps[i+1]
    return P

def verify_matrix_is_diagonal(effective_channel: np.ndarray) -> bool:
    off_diag_sum = np.sum(np.abs(effective_channel - np.diag(np.diag(effective_channel))))
    return off_diag_sum < tolerance

def verify_multi_ris_diagonalization(
    Ps: List[np.ndarray],
    Gs: List[np.ndarray],
    H: np.ndarray,
    Cs: List[np.ndarray]
) -> List[bool]:
    """
    Verify that G(P_1C_1P_2C_2...P_M)H is diagonal for all receivers.
    
    Args:
        Ps: List of reflection matrices [P_1, P_2, ..., P_M]
        Gs: List of channel matrices from RIS to receivers
        H: Channel matrix from transmitter to RIS
        Cs: List of M-1 inter-RIS channel matrices [C_1, C_2, ..., C_(M-1)], where C_i is the channel between RIS i and i+1
    
    Returns:
        List of boolean values indicating if effective channel is diagonal for each receiver
    """
    results = []
    
    P = unify_ris_reflection_matrices(Ps, Cs)
    for G in Gs:
        effective_channel = G @ P @ H
        results.append(verify_matrix_is_diagonal(effective_channel))
    return results

def print_effective_channel(G: np.ndarray, H: np.ndarray, P: np.ndarray):
    """
    Print the effective channel matrix GPH.

    Args:
        G: Channel matrix from RIS to receiver
        H: Channel matrix from transmitter to RIS
        P: Reflection matrix
    """
    effective_channel = G @ P @ H
    rounded_matrix = np.round(np.abs(effective_channel), 2)
    print(rounded_matrix)

def verify_results(
    Ps: List[np.ndarray],
    Gs: List[np.ndarray],
    H: np.ndarray,
    Cs: List[np.ndarray]
):
    """
    Verify diagonalization results and print effective channel matrix for each receiver.

    Args:
        Ps: List of reflection matrices
        Gs: List of channel matrices from RIS to receivers
        H: Channel matrix from transmitter to RIS
        Cs: List of M-1 inter-RIS channel matrices [C_1, C_2, ..., C_(M-1)], where C_i is the channel between RIS i and i+1
    """
    diagonalization_results = verify_multi_ris_diagonalization(Ps, Gs, H, Cs)
    
    if all(diagonalization_results):
        print("Diagonalization successful for ALL receivers")
    elif any(diagonalization_results):
        print("Diagonalization successful for SOME receivers")
    else:
        print("Diagonalization successful for NO receivers")
        
    print("Individual results:", [bool(x) for x in diagonalization_results])
    
    print("\nEffective channel matrix for first receiver:")
    P = unify_ris_reflection_matrices(Ps, Cs)
    print_effective_channel(Gs[0], H, P)

def main():
    N = 16    # * Number of reflecting elements
    K = 2     # * Number of antennas
    J = 2     # * Number of receivers
    M = 2     # * Number of RIS surfaces
    E = 10    # * Number of eavesdroppers
    eta = 0.9 # * Reflection efficiency

    print(f"Parameters: \n- RIS surfaces = {M}\n- Elements per RIS = {N}\n- Reflection efficiency = {eta}\n- Receiver Antennas = {K}")

    H = generate_random_channel_matrix(N, K)
    Gs = [generate_random_channel_matrix(K, N) for _ in range(J)]
    Es = [generate_random_channel_matrix(K, N) for _ in range(E)]
    Cs = [generate_random_channel_matrix(N, N) for _ in range(M-1)]
    
    try:
        Ps, dor = calculate_multi_ris_reflection_matrices(K, N, J, M, Gs, H, eta, Cs)
        print(f"Degree of Randomness (DoR): {dor}")
        
        print(f"\n{J} Legitimate Receivers:")
        verify_results(Ps, Gs, H, Cs)

        print(f"\n{E} End Eavesdroppers:")
        verify_results(Ps, Es, H, Cs)

        if M > 1:
            print(f"\n{E} Intermediate Eavesdroppers:")
            verify_results(Ps[1:], Es, H, Cs[1:])
        
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()