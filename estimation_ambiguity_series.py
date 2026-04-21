import numpy as np
from diagonalization import (generate_random_channel_matrix,
                             calculate_ris_reflection_matrice,
                             verify_matrix_is_diagonal)

np.random.seed(0)


# ============================================================
#  SIMULATORE JBF-MC  (conosce i canali veri, restituisce G', H' con Phi)
# ============================================================
def simulate_jbfmc(G_true, Heff_true):
    N = Heff_true.shape[0]
    phi = np.random.randn(N) + 1j * np.random.randn(N)
    return G_true @ np.diag(phi), np.diag(1 / phi) @ Heff_true


# ============================================================
#  CROSS-ROUND ALIGNMENT
# ============================================================
def align_rounds(G_prime_list, Heff_prime_list):
    G_ref = G_prime_list[0]
    aligned = [Heff_prime_list[0]]
    for r in range(1, len(G_prime_list)):
        # ratio deve essere stimato sulle colonne di G (indice N)
        # G' e' K x N, quindi media sulle righe (axis=0) per ogni colonna n
        ratio = np.mean(G_prime_list[r] / G_ref, axis=0)   # length N
        aligned.append(np.diag(ratio) @ Heff_prime_list[r])
    return aligned, G_ref


# ============================================================
#  CALIBRAZIONE INDUTTIVA
# ============================================================
def calibrate_chain(M, N, K, R, H_true, Cs_true, G_intermediate_list, G_M_true):
    """
    G_intermediate_list: lista di M-1 matrici G_m per m=1..M-1 (receiver intermedi
    di calibrazione, fissi per tutto l'hop). G_intermediate_list[m-1] e' G_m.
    Per m=M si usa G_M_true.
    """
    # ---------- HOP 1 ----------
    if M == 1:
        G1p, H1p = simulate_jbfmc(G_M_true, H_true)
        return {"H1_prime": H1p, "Cs_prime": [], "GM_prime": G1p}

    # Per la stima iniziale di H1' usiamo il receiver dell'hop 1
    G1_true = G_intermediate_list[0] if M > 1 else G_M_true
    _, H1_prime = simulate_jbfmc(G1_true, H_true)

    Cs_prime = []
    P_fix_chain = []   # P_j^fix usati per la calibrazione degli hop >= 2

    def build_controller_Heff(m_target):
        """Heff al livello m_target, con ambiguita' Phi_{m_target}^{-1}."""
        Heff = H1_prime
        for j in range(m_target - 1):
            Heff = Cs_prime[j] @ P_fix_chain[j] @ Heff
        return Heff

    def build_true_Heff(m_target, extra_P=None):
        """Heff fisico al livello m_target. Se extra_P e' dato, lo appende."""
        Heff = H_true
        P_list = P_fix_chain + ([extra_P] if extra_P is not None else [])
        for j in range(m_target - 1):
            Heff = Cs_true[j] @ P_list[j] @ Heff
        return Heff

    # ---------- HOP 2 ... M ----------
    for m in range(2, M + 1):
        # Se mancano P_fix per gli hop precedenti, li aggiungiamo ora
        # (verranno usati dal controller-Heff e dal true-Heff durante tutto l'hop m)
        while len(P_fix_chain) < m - 2:
            p_fix = np.exp(1j * 2*np.pi * np.random.rand(N))
            P_fix_chain.append(np.diag(p_fix))

        # G_m fissa per tutto l'hop m (FIX DEL BUG)
        if m < M:
            G_m_true = G_intermediate_list[m - 1]
        else:
            G_m_true = G_M_true

        # R round all'hop m, variando P_{m-1}^fix
        G_m_prime_list = []
        Heff_m_prime_list = []
        P_m_minus_1_fix_list = []

        for r in range(R):
            p_r = np.exp(1j * 2*np.pi * np.random.rand(N))
            P_m_minus_1_fix_r = np.diag(p_r)

            Heff_m_true = build_true_Heff(m, extra_P=P_m_minus_1_fix_r)
            G_m_p, Heff_m_p = simulate_jbfmc(G_m_true, Heff_m_true)

            G_m_prime_list.append(G_m_p)
            Heff_m_prime_list.append(Heff_m_p)
            P_m_minus_1_fix_list.append(P_m_minus_1_fix_r)

        # Cross-round alignment (ora funziona perche' G_m e' fissa)
        Heff_aligned, G_m_ref = align_rounds(G_m_prime_list, Heff_m_prime_list)

        # Estrazione di C_{m-1}'
        Heff_m_minus_1_ctrl = build_controller_Heff(m - 1)
        Y = np.hstack(Heff_aligned)
        E = np.hstack([P_r @ Heff_m_minus_1_ctrl for P_r in P_m_minus_1_fix_list])
        C_m_minus_1_prime = Y @ np.linalg.pinv(E)

        Cs_prime.append(C_m_minus_1_prime)

        if m == M:
            GM_prime = G_m_ref

    return {
        "H1_prime": H1_prime,
        "Cs_prime": Cs_prime,
        "GM_prime": GM_prime,
    }


# ============================================================
#  FASE OPERATIVA
# ============================================================
def operational_step(store, K, N, M, eta, H_true, Cs_true, G_M_true):
    # Nuovi P_1, ..., P_{M-1} random (a runtime)
    Ps_first = [np.diag(eta * np.exp(1j * 2*np.pi * np.random.rand(N)))
                for _ in range(M - 1)]

    # Controller costruisce Heff stimato fino alla RIS M usando i suoi oggetti
    Heff_stored = store["H1_prime"]
    for j in range(M - 1):
        Heff_stored = store["Cs_prime"][j] @ Ps_first[j] @ Heff_stored

    # Risolve per P_M
    P_M, _ = calculate_ris_reflection_matrice(
        K, N, 1, [store["GM_prime"]], Heff_stored, eta)

    Ps_all = Ps_first + [P_M]

    # Applica al canale fisico
    cascade = G_M_true
    for j in range(M - 1, -1, -1):
        cascade = cascade @ Ps_all[j]
        if j > 0:
            cascade = cascade @ Cs_true[j - 1]
    cascade = cascade @ H_true

    return cascade


# ============================================================
#  TEST
# ============================================================
def run_test(M, N=36, K=4, n_symbols=3, eta=1.0):
    R = int(np.ceil(N / K))
    print(f"\n{'='*70}")
    print(f"TEST M={M} RIS in series  (N={N}, K={K}, R={R})")
    print(f"{'='*70}")

    H_true = generate_random_channel_matrix(N, K)
    Cs_true = [generate_random_channel_matrix(N, N) for _ in range(M - 1)]
    G_M_true = generate_random_channel_matrix(K, N)

    # Receiver intermedi fissi: uno per ogni RIS < M (quello per RIS M e' G_M_true)
    G_intermediate_list = [generate_random_channel_matrix(K, N) for _ in range(M - 1)]

    print("=== CALIBRAZIONE ===")
    store = calibrate_chain(M, N, K, R, H_true, Cs_true, G_intermediate_list, G_M_true)
    print(f"Stored: H1' {store['H1_prime'].shape}, "
          f"{len(store['Cs_prime'])} C', GM' {store['GM_prime'].shape}")

    print("=== FASE OPERATIVA ===")
    for t in range(n_symbols):
        cascade = operational_step(store, K, N, M, eta, H_true, Cs_true, G_M_true)
        ok = verify_matrix_is_diagonal(cascade)
        off = np.max(np.abs(cascade - np.diag(np.diag(cascade))))
        diag = np.abs(np.diag(cascade))
        print(f"  Simbolo {t+1}: diag={ok} | off-max={off:.1e} | "
              f"diag_vals={diag.round(2)}")


if __name__ == "__main__":
    run_test(M=2)
    run_test(M=3)
    run_test(M=4)