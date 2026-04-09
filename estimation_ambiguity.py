import numpy as np
from diagonalization import calculate_ris_reflection_matrice, verify_matrix_is_diagonal, print_effective_channel, generate_random_channel_matrix

def main():
    # --- 1. SETUP PARAMETRI ---
    # Semplificazione: L = M = K (stesso numero di antenne a TX e RX)
    N = 36    # Numero di elementi RIS
    K = 4     # Numero di antenne (null_space_dim = N-J*K²+J*K = 36-32+8 = 12)
    eta = 1.0 # Reflection efficiency per i test
    J = 2     # Numero di ricevitori

    print(f"Test Allineamento Ambiguità Tensor: {K} Antenne, {N} Elementi RIS")
    print("Modello ambiguità PARAFAC (He & Yuan eq.3): G' = G @ Phi, H' = inv(Phi) @ H")
    print("Phi è una matrice diagonale N×N (una scala per elemento RIS)")

    # --- 2. GENERAZIONE CANALI FISICI (VERI) ---
    # H: N×K (BS -> RIS), G_j: K×N (RIS -> Rx_j)
    H_true = generate_random_channel_matrix(N, K)
    G1_true = generate_random_channel_matrix(K, N)
    G2_true = generate_random_channel_matrix(K, N)

    # --- 3. SIMULAZIONE OUTPUT PARAFAC (AMBIGUITÀ N×N) ---
    # Ogni ricevitore stima G_j e H con la propria ambiguità Phi_j (N×N diagonale)
    # Phi moltiplica G a DESTRA (scala le N colonne) e H^{-1} a SINISTRA (scala le N righe)
    phi1_vec = np.random.randn(N) + 1j * np.random.randn(N)
    phi2_vec = np.random.randn(N) + 1j * np.random.randn(N)
    Phi1 = np.diag(phi1_vec)
    Phi2 = np.diag(phi2_vec)

    G1_prime = G1_true @ Phi1                    # (K×N) @ (N×N) = K×N
    H1_prime = np.linalg.inv(Phi1) @ H_true      # (N×N) @ (N×K) = N×K

    G2_prime = G2_true @ Phi2
    H2_prime = np.linalg.inv(Phi2) @ H_true

    # Sanity check: per un singolo ricevitore l'ambiguità è invisibile
    # G_j' @ diag(p) @ H_j' = G_j @ Phi @ diag(p) @ Phi^-1 @ H = G_j @ diag(p) @ H
    # (perché Phi, diag(p), Phi^-1 sono tutte diagonali N×N e commutano)
    test_p = np.random.randn(N) + 1j * np.random.randn(N)
    prod_est = G1_prime @ np.diag(test_p) @ H1_prime
    prod_true = G1_true @ np.diag(test_p) @ H_true
    print(f"\nSanity check prodotto singolo ricevitore preservato: {np.allclose(prod_est, prod_true)}")

    # --- 4. ALGORITMO DI ALLINEAMENTO ---
    # Obiettivo: ottenere G2_hat con la stessa ambiguità Phi1 di G1_prime e H1_prime.
    # G2_hat = G2 @ Phi1 = G2_prime @ (inv(Phi2) @ Phi1) = G2_prime @ diag(phi1_n/phi2_n)
    #
    # Come stimare Lambda_n = phi1_n/phi2_n dai dati?
    # H1_prime[n,k] = H[n,k] / phi1_n
    # H2_prime[n,k] = H[n,k] / phi2_n
    # H2_prime[n,k] / H1_prime[n,k] = phi1_n / phi2_n = Lambda_n
    # Media sulle K colonne (axis=1) per robustezza al rumore.
    print("\n[Esecuzione Algoritmo di Allineamento...]")
    lambda_vec_est = np.mean(H2_prime / H1_prime, axis=1)  # lunghezza N
    Lambda_est = np.diag(lambda_vec_est)                    # N×N

    G2_hat = G2_prime @ Lambda_est                          # (K×N) @ (N×N) = K×N

    # Verifica: G2_hat dovrebbe essere G2_true @ Phi1
    G2_target = G2_true @ Phi1
    error_G2 = np.linalg.norm(G2_hat - G2_target)
    print(f"Errore di ricostruzione di G2_hat: {error_G2:.2e}")
    assert error_G2 < 1e-10, "L'allineamento è fallito!"
    print("=> Allineamento G2 completato con successo!")

    SEP = "=" * 60

    # --- 5. TEST 1: CANALI VERI ---
    print(f"\n{SEP}")
    print("[Test 1: Diagonalizzazione con Canali VERI (G1_true, G2_true, H_true)]")
    P1, dor1 = calculate_ris_reflection_matrice(K, N, J, [G1_true, G2_true], H_true, eta)
    print(f"DoR: {dor1}")
    for G, name in [(G1_true, "G1_true"), (G2_true, "G2_true")]:
        C = G @ P1 @ H_true
        print(f"  {name} — diagonale: {verify_matrix_is_diagonal(C)}")
        print_effective_channel(C)

    # --- 6. TEST 2: STIME GREZZE (G1', G2', H1') ---
    # Atteso: funziona sulle stime di Rx1 (stessa Phi1), ma FALLISCE fisicamente su Rx2
    # perché G2_prime ha Phi2 mentre H1_prime ha inv(Phi1).
    # Il prodotto G2' @ diag(p) @ H1' = G2 @ diag(phi2_n/phi1_n * p_n) @ H,
    # quindi p nel null-space delle stime corrisponde a un p' DIVERSO nel null-space vero.
    print(f"\n{SEP}")
    print("[Test 2: Diagonalizzazione con Stime GREZZE (G1', G2', H1')]")
    print("  Ambiguità miste: G2' ha Phi2, H1' ha inv(Phi1) -> null-space distorto su Rx2")
    P2, dor2 = calculate_ris_reflection_matrice(K, N, J, [G1_prime, G2_prime], H1_prime, eta)
    print(f"DoR: {dor2}")
    print("  -> Come le 'vede' l'algoritmo (stimate):")
    for G, name in [(G1_prime, "G1'"), (G2_prime, "G2'")]:
        C = G @ P2 @ H1_prime
        print(f"     {name} — diagonale: {verify_matrix_is_diagonal(C)}")
        print_effective_channel(C)
    print("  -> Verifica sui canali VERI (effetto fisico reale del P applicato alla RIS):")
    for G, name in [(G1_true, "G1_true"), (G2_true, "G2_true")]:
        C = G @ P2 @ H_true
        print(f"     {name} — diagonale: {verify_matrix_is_diagonal(C)}")
        print_effective_channel(C)

    # --- 7. TEST 3: STIME ALLINEATE (G1', G2_hat, H1') ---
    # Atteso: tutti hanno la stessa Phi1 -> il null-space coincide con quello vero
    print(f"\n{SEP}")
    print("[Test 3: Diagonalizzazione con Stime ALLINEATE (G1', G2_hat, H1')]")
    print("  G2_hat = G2 @ Phi1 -> stessa ambiguità di G1' e H1'")
    P3, dor3 = calculate_ris_reflection_matrice(K, N, J, [G1_prime, G2_hat], H1_prime, eta)
    print(f"DoR: {dor3}")
    print("  -> Come le 'vede' l'algoritmo (stimate):")
    for G, name in [(G1_prime, "G1'"), (G2_hat, "G2_hat")]:
        C = G @ P3 @ H1_prime
        print(f"     {name} — diagonale: {verify_matrix_is_diagonal(C)}")
        print_effective_channel(C)
    print("  -> Verifica sui canali VERI (effetto fisico reale del P applicato alla RIS):")
    for G, name in [(G1_true, "G1_true"), (G2_true, "G2_true")]:
        C = G @ P3 @ H_true
        print(f"     {name} — diagonale: {verify_matrix_is_diagonal(C)}")
        print_effective_channel(C)

if __name__ == "__main__":
    main()
