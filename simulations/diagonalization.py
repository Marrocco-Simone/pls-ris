import numpy as np
from typing import List, Tuple, Optional

def calculate_W_single(G: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Calculate W matrix for a single receiver.
    
    Args:
        G: Channel matrix from RIS to receiver (KxN)
        H: Channel matrix from transmitter to RIS (NxK)
    
    Returns:
        W: The W matrix as defined in equation (8) of the paper
    """
    K, N = G.shape  # K: number of antennas, N: number of reflecting elements
    W = np.zeros((N, N), dtype=complex)
    
    for i in range(K):
        for j in range(K):
            if i != j:
                # g_j ⊙ h_i^T
                temp = np.multiply(G[j, :], H[:, i].T)
                # (g_j ⊙ h_i^T)^† (g_j ⊙ h_i^T)
                W += np.outer(temp.conj(), temp)
    
    return W

def calculate_W_multiple(Gs: List[np.ndarray], H: np.ndarray) -> np.ndarray:
    """
    Calculate combined W matrix for multiple receivers.
    
    Args:
        Gs: List of channel matrices from RIS to each receiver [G_1, G_2, ..., G_J]
        H: Channel matrix from transmitter to RIS
    
    Returns:
        W: The stacked W matrix for all receivers
    """
    J = len(Gs)  # Number of receivers
    K, N = Gs[0].shape  # Assuming all receivers have same number of antennas
    
    # Initialize the stacked W matrix
    W_combined = np.zeros((J * N, N), dtype=complex)
    
    # Calculate W for each receiver and stack them
    for j in range(J):
        W_j = calculate_W_single(Gs[j], H)
        W_combined[j*N:(j+1)*N, :] = W_j
    
    return W_combined

def calculate_reflection_matrix(Gs: List[np.ndarray], 
                             H: np.ndarray, 
                             eta: float = 1.0,
                             random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate the reflection matrix P given channel gains.
    
    Args:
        Gs: List of channel matrices from RIS to receivers [G_1, G_2, ..., G_J]
        H: Channel matrix from transmitter to RIS
        eta: Reflection efficiency (default: 1.0)
        random_seed: Random seed for reproducibility
    
    Returns:
        P: Diagonal reflection matrix
        P_paper: Diagonal reflection matrix, made with the paper method
        dor: Degree of randomness achieved
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Get dimensions
    J = len(Gs)  # Number of receivers
    K, N = Gs[0].shape
    
    # Calculate combined W matrix
    W = calculate_W_multiple(Gs, H)
    
    # Perform SVD
    U, _, Vh = np.linalg.svd(W)
    
    # Get the null space (last N-JK²+JK columns of V^† and U)
    null_space_dim = N - J*K**2 + J*K
    if null_space_dim <= 0:
        raise ValueError(f"No solution exists. Need more reflecting elements. "
                       f"Current: {N}, Required: >{J*K**2 - J*K}")
    
    # Get the null space basis from U
    null_space_basis = Vh[-null_space_dim:, :].T.conj()
    null_space_basis_paper = U[:, -null_space_dim:]

    # Generate random vector a
    a = np.random.normal(0, 1, (null_space_dim,)) + 1j * np.random.normal(0, 1, (null_space_dim,))
    
    # Calculate reflection vector p
    p_unnormalized = null_space_basis @ a
    p = eta * p_unnormalized / np.max(np.abs(p_unnormalized))

    p_unnormalized_paper = null_space_basis_paper @ a
    p_paper = eta * p_unnormalized_paper / np.max(np.abs(p_unnormalized_paper))
    
    # Create diagonal reflection matrix P
    P = np.diag(p)
    P_paper = np.diag(p_paper)
    
    # Calculate DoR
    dor = 2 * (N - J*K**2 + J*K)
    
    return P, P_paper, dor

def verify_diagonalization(P: np.ndarray, Gs: List[np.ndarray], H: np.ndarray, 
                         tolerance: float = 1e-10) -> List[bool]:
    """
    Verify that GPH is diagonal for all receivers.
    
    Args:
        P: Calculated reflection matrix
        Gs: List of channel matrices from RIS to receivers
        H: Channel matrix from transmitter to RIS
        tolerance: Tolerance for considering an element zero
    
    Returns:
        List of boolean values indicating if GPH is diagonal for each receiver
    """
    results = []
    for G in Gs:
        effective_channel = G @ P @ H
        # Get the sum of absolute values of off-diagonal elements
        off_diag_sum = np.sum(np.abs(effective_channel - np.diag(np.diag(effective_channel))))
        results.append(off_diag_sum < tolerance)
    return results

# Example usage
if __name__ == "__main__":
    # Example parameters
    N = 16  # Number of reflecting elements
    K = 2   # Number of antennas
    J = 4   # Number of receivers
    
    # Generate random channel matrices for demonstration
    np.random.seed(42)
    H = (np.random.normal(0, 1, (N, K)) + 1j * np.random.normal(0, 1, (N, K))) / np.sqrt(2)
    Gs = [(np.random.normal(0, 1, (K, N)) + 1j * np.random.normal(0, 1, (K, N))) / np.sqrt(2) 
          for _ in range(J)]
    
    try:
        # Calculate reflection matrix
        P, P_paper, dor = calculate_reflection_matrix(Gs, H, eta=0.9, random_seed=42)
        
        # Verify the result
        print(f"Shape of P: {P.shape}")
        diagonalization_results = verify_diagonalization(P, Gs, H)
        
        print(f"Degree of Randomness (DoR): {dor}")
        print("Diagonalization successful for all receivers:", all(diagonalization_results))
        print("Individual results:", diagonalization_results)
        
        # Print example effective channel matrix for first receiver
        print("\nEffective channel matrix for first receiver:")
        effective_channel = Gs[0] @ P @ H
        rounded_matrix = np.round(np.abs(effective_channel), 2)
        print(rounded_matrix)

        # ! Paper method
        print("\nPaper method:")
        # Verify the result
        print(f"Shape of P: {P_paper.shape}")
        diagonalization_results = verify_diagonalization(P_paper, Gs, H)
        
        print(f"Degree of Randomness (DoR): {dor}")
        print("Diagonalization successful for all receivers:", all(diagonalization_results))
        print("Individual results:", diagonalization_results)
        
        # Print example effective channel matrix for first receiver
        print("\nEffective channel matrix for first receiver:")
        effective_channel = Gs[0] @ P_paper @ H
        rounded_matrix = np.round(np.abs(effective_channel), 2)
        print(rounded_matrix)
        
    except ValueError as e:
        print(f"Error: {e}")