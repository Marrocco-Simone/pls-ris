import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from multiprocess import Pool, cpu_count  # pyright: ignore[reportAttributeAccessIssue]
from tqdm import tqdm
from heatmap import configure_latex
from diagonalization import (
    generate_random_channel_matrix,
    calculate_multi_ris_reflection_matrices,
    unify_ris_reflection_matrices,
)
from noise_power_utils import snr_db_to_sigma_sq, calculate_noise_floor_in_mw
from heatmap_utils import calculate_free_space_path_loss

MAX_CPU_COUNT = 64
N_PROCESSES = min(cpu_count(), MAX_CPU_COUNT)


###############################################################################
# Core secrecy rate functions — faithful to paper eqs. (62)-(64)
###############################################################################

def _draw_fresh_P(K: int, N: int, J: int, M: int, Gs: List[np.ndarray],
                  H: np.ndarray, eta: float, Cs: List[np.ndarray]) -> np.ndarray:
    Ps, _ = calculate_multi_ris_reflection_matrices(K, N, J, M, Gs, H, eta, Cs)
    return unify_ris_reflection_matrices(Ps, Cs)


def _make_noise(K: int, sigma_sq: float) -> np.ndarray:
    return np.sqrt(sigma_sq / 2) * (
        np.random.randn(K, 1) + 1j * np.random.randn(K, 1)
    )


def compute_sigma_inv_sqrt(
    K: int, N: int, J: int, M: int,
    Gs: List[np.ndarray], H: np.ndarray, F: np.ndarray,
    eta: float, Cs: List[np.ndarray],
    delta: np.ndarray, sigma_sq: float,
    num_P_samples: int = 500,
) -> np.ndarray:
    accumulator = np.zeros((K, K), dtype=complex)
    for _ in range(num_P_samples):
        P_sample = _draw_fresh_P(K, N, J, M, Gs, H, eta, Cs)
        v = F @ P_sample @ H @ delta
        accumulator += v @ v.conj().T
    accumulator /= num_P_samples

    Sigma = accumulator + sigma_sq * np.eye(K)

    eigenvals, eigenvecs = np.linalg.eigh(Sigma)
    eigenvals = np.maximum(eigenvals, 1e-12)
    Sigma_inv_sqrt = eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals)) @ eigenvecs.conj().T
    return Sigma_inv_sqrt


def compute_bob_rate(
    K: int, N: int, J: int, M: int,
    Gs: List[np.ndarray], H: np.ndarray,
    eta: float, Cs: List[np.ndarray],
    sigma_sq: float, num_mc: int = 1000,
) -> float:
    G = Gs[0]
    sum_over_i = 0.0

    for i in range(K):
        xi = np.zeros((K, 1))
        xi[i] = 1.0

        mc_accumulator = 0.0
        for _ in range(num_mc):
            P = _draw_fresh_P(K, N, J, M, Gs, H, eta, Cs)
            mu = _make_noise(K, sigma_sq)

            log_sum = 0.0
            for l in range(K):
                xl = np.zeros((K, 1))
                xl[l] = 1.0
                delta = xi - xl

                a = G @ P @ H @ delta + mu
                exponent = (-np.linalg.norm(a) ** 2 + np.linalg.norm(mu) ** 2) / sigma_sq
                log_sum += np.exp(exponent)

            mc_accumulator += np.log2(np.maximum(log_sum, 1e-300))

        sum_over_i += mc_accumulator / num_mc

    R_bob = np.log2(K) - sum_over_i / K
    return float(np.real(R_bob))


def compute_eve_rate(
    K: int, N: int, J: int, M: int,
    Gs: List[np.ndarray], H: np.ndarray,
    F: np.ndarray, B: np.ndarray,
    eta: float, Cs: List[np.ndarray],
    sigma_sq: float,
    num_mc: int = 1000, num_P_sigma: int = 500,
) -> float:
    Sigma_inv_sqrts = {}
    for i in range(K):
        xi = np.zeros((K, 1))
        xi[i] = 1.0
        for l in range(K):
            xl = np.zeros((K, 1))
            xl[l] = 1.0
            delta = xi - xl
            if i == l:
                Sigma_inv_sqrts[(i, l)] = np.eye(K)
            else:
                Sigma_inv_sqrts[(i, l)] = compute_sigma_inv_sqrt(
                    K, N, J, M, Gs, H, F, eta, Cs, delta, sigma_sq, num_P_sigma
                )

    sum_over_i = 0.0

    for i in range(K):
        xi = np.zeros((K, 1))
        xi[i] = 1.0

        mc_accumulator = 0.0
        for _ in range(num_mc):
            P = _draw_fresh_P(K, N, J, M, Gs, H, eta, Cs)
            nu = _make_noise(K, sigma_sq)

            log_sum = 0.0
            for l in range(K):
                xl = np.zeros((K, 1))
                xl[l] = 1.0
                delta = xi - xl

                S_inv_sqrt = Sigma_inv_sqrts[(i, l)]
                nu_prime = S_inv_sqrt @ (F @ P @ H @ delta + nu)
                a = S_inv_sqrt @ B @ delta + nu_prime

                exponent = -np.linalg.norm(a) ** 2 + np.linalg.norm(nu_prime) ** 2
                log_sum += np.exp(exponent)

            mc_accumulator += np.log2(np.maximum(log_sum, 1e-300))

        sum_over_i += mc_accumulator / num_mc

    R_eve = np.log2(K) - sum_over_i / K
    return float(np.real(R_eve))


def compute_secrecy_rate(
    K: int, N: int, J: int, M: int,
    Gs: List[np.ndarray], H: np.ndarray,
    F: np.ndarray, B: np.ndarray,
    eta: float, Cs: List[np.ndarray],
    sigma_sq: float,
    num_mc: int = 1000, num_P_sigma: int = 500,
) -> Tuple[float, float, float]:
    R_bob = compute_bob_rate(K, N, J, M, Gs, H, eta, Cs, sigma_sq, num_mc)
    R_eve = compute_eve_rate(K, N, J, M, Gs, H, F, B, eta, Cs, sigma_sq, num_mc, num_P_sigma)
    secrecy = max(0.0, R_bob - R_eve)
    return secrecy, R_bob, R_eve


###############################################################################
# Constants and scenario definitions
###############################################################################

OUTPUT_DIR = "./secrecy_results"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "results.npz")

N_ELEMENTS = 16
K_ANTENNAS = 2
J_RECEIVERS = 2
M_SURFACES = 1
ETA = 0.9
DISTANCE_M = 10
NUM_MC = 20000
NUM_P_SIGMA = 20000
SNR_RANGE_DB = np.arange(-20, 22, 4)
PT_RANGE_DBM = np.arange(-120, 82, 5)
XI_RANGE_DB = np.arange(-20, 22, 4)
XI_SNR_VALUES_DB = [-10, 0, 20]

PATH_LOSS = calculate_free_space_path_loss(DISTANCE_M)

SCENARIOS = [
    {"label": "Passive, NLOS Eve", "active": False, "eve_los": False},
    {"label": "Passive, LOS Eve", "active": False, "eve_los": True},
    {"label": "Active, NLOS Eve", "active": True, "eve_los": False},
    {"label": "Active, LOS Eve", "active": True, "eve_los": True},
]

PLOT_COLORS = ['black', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red']
PLOT_LINESTYLES = ['-', '-', '-', '--', '--']
PLOT_MARKERS = ['o', 's', '^', 'D', 'v']


###############################################################################
# Parallel worker
###############################################################################

def _worker(args: dict) -> dict:
    """Top-level worker for multiprocessing. Computes one secrecy rate point."""
    sr, rb, re = compute_secrecy_rate(
        args['K'], args['N'], args['J'], args['M'],
        args['Gs'], args['H'], args['F'], args['B'],
        args['eta'], args['Cs'], args['sigma_sq'],
        num_mc=args['num_mc'], num_P_sigma=args['num_P_sigma'],
    )
    return {**args['meta'], 'sr': sr, 'rb': rb, 're': re}


###############################################################################
# Computation (pure data, no plotting)
###############################################################################

def run_all_computations() -> dict:
    N0_mw = calculate_noise_floor_in_mw()
    N0_dbm = 10 * np.log10(N0_mw)
    Cs: list = []

    print(f"Parameters: N={N_ELEMENTS}, K={K_ANTENNAS}, J={J_RECEIVERS}, M={M_SURFACES}")
    print(f"Noise floor: {N0_dbm:.1f} dBm = {N0_mw:.2e} mW")
    print(f"MC samples: {NUM_MC}, P samples for Sigma: {NUM_P_SIGMA}")
    print(f"Workers: {N_PROCESSES}\n")

    H_paper = generate_random_channel_matrix(N_ELEMENTS, K_ANTENNAS)
    G_paper = generate_random_channel_matrix(K_ANTENNAS, N_ELEMENTS)
    F_paper = generate_random_channel_matrix(K_ANTENNAS, N_ELEMENTS)
    B_paper = generate_random_channel_matrix(K_ANTENNAS, K_ANTENNAS)
    G2_paper = generate_random_channel_matrix(K_ANTENNAS, N_ELEMENTS)
    Gs_paper = [G_paper, G2_paper]

    base_args = {
        'K': K_ANTENNAS, 'N': N_ELEMENTS, 'J': J_RECEIVERS, 'M': M_SURFACES,
        'eta': ETA, 'Cs': Cs,
        'num_mc': NUM_MC, 'num_P_sigma': NUM_P_SIGMA,
    }

    # --- Build all jobs ---
    jobs = []

    # Paper replication jobs (SNR axis)
    for idx, snr_db in enumerate(SNR_RANGE_DB):
        jobs.append({
            **base_args,
            'Gs': Gs_paper, 'H': H_paper, 'F': F_paper, 'B': B_paper,
            'sigma_sq': snr_db_to_sigma_sq(snr_db),
            'meta': {'group': 'paper', 'idx': idx, 'snr_db': int(snr_db)},
        })

    # Realistic scenario jobs: random matrices scaled by path loss
    scenario_channels = []
    for scenario in SCENARIOS:
        H_sc = PATH_LOSS * generate_random_channel_matrix(N_ELEMENTS, K_ANTENNAS)
        G_sc = PATH_LOSS * generate_random_channel_matrix(K_ANTENNAS, N_ELEMENTS)
        G2_sc = PATH_LOSS * generate_random_channel_matrix(K_ANTENNAS, N_ELEMENTS)
        F_sc = PATH_LOSS * generate_random_channel_matrix(K_ANTENNAS, N_ELEMENTS)
        if scenario["eve_los"]:
            B_sc = PATH_LOSS * generate_random_channel_matrix(K_ANTENNAS, K_ANTENNAS)
        else:
            B_sc = np.zeros((K_ANTENNAS, K_ANTENNAS), dtype=complex)
        if scenario["active"]:
            H_sc = H_sc / np.max(np.abs(H_sc))
        scenario_channels.append(([G_sc, G2_sc], H_sc, F_sc, B_sc))

    # Direct link gain sweep jobs (xi axis, unit channels, multiple thermal SNRs)
    for snr_idx, snr_db in enumerate(XI_SNR_VALUES_DB):
        sigma_sq_xi = snr_db_to_sigma_sq(snr_db)
        for xi_idx, xi_db in enumerate(XI_RANGE_DB):
            xi_lin = 10 ** (xi_db / 20)
            jobs.append({
                **base_args,
                'Gs': Gs_paper, 'H': H_paper, 'F': F_paper,
                'B': xi_lin * B_paper,
                'sigma_sq': sigma_sq_xi,
                'meta': {'group': 'xi_sweep', 'snr_idx': snr_idx,
                         'xi_idx': xi_idx, 'xi_db': int(xi_db),
                         'snr_db': int(snr_db)},
            })

    # Paper baseline on Pt axis (no path loss)
    for pt_idx, Pt_dbm in enumerate(PT_RANGE_DBM):
        sqrt_Pt = np.sqrt(10 ** (Pt_dbm / 10))
        jobs.append({
            **base_args,
            'Gs': Gs_paper, 'H': sqrt_Pt * H_paper,
            'F': F_paper, 'B': sqrt_Pt * B_paper,
            'sigma_sq': N0_mw,
            'meta': {'group': 'realistic', 'sc_idx': 0,
                     'pt_idx': pt_idx, 'label': 'No path loss (paper)'},
        })

    # Each scenario on Pt axis (path loss already baked into channels)
    for sc_idx, scenario in enumerate(SCENARIOS):
        Gs_sc, H_sc, F_sc, B_sc = scenario_channels[sc_idx]
        for pt_idx, Pt_dbm in enumerate(PT_RANGE_DBM):
            sqrt_Pt = np.sqrt(10 ** (Pt_dbm / 10))
            jobs.append({
                **base_args,
                'Gs': Gs_sc, 'H': sqrt_Pt * H_sc,
                'F': F_sc, 'B': sqrt_Pt * B_sc,
                'sigma_sq': N0_mw,
                'meta': {'group': 'realistic', 'sc_idx': sc_idx + 1,
                         'pt_idx': pt_idx, 'label': scenario["label"]},
            })

    print(f"Total jobs: {len(jobs)}\n")

    # --- Run all jobs in parallel ---
    with Pool(processes=N_PROCESSES) as pool:
        results = list(tqdm(
            pool.imap(_worker, jobs),
            total=len(jobs),
            desc="Computing secrecy rates",
        ))

    # --- Collect results ---
    paper_secrecy = np.zeros(len(SNR_RANGE_DB))
    paper_rbob = np.zeros(len(SNR_RANGE_DB))
    paper_reve = np.zeros(len(SNR_RANGE_DB))

    num_scenarios = 1 + len(SCENARIOS)
    scenario_labels = ["No path loss (paper)"] + [s["label"] for s in SCENARIOS]

    pt_secrecy = np.zeros((num_scenarios, len(PT_RANGE_DBM)))
    pt_rbob = np.zeros((num_scenarios, len(PT_RANGE_DBM)))
    pt_reve = np.zeros((num_scenarios, len(PT_RANGE_DBM)))

    num_xi_snr = len(XI_SNR_VALUES_DB)
    num_xi = len(XI_RANGE_DB)
    xi_secrecy = np.zeros((num_xi_snr, num_xi))
    xi_rbob = np.zeros((num_xi_snr, num_xi))
    xi_reve = np.zeros((num_xi_snr, num_xi))

    for r in results:
        meta = r
        group = meta['group']
        if group == 'paper':
            idx = meta['idx']
            paper_secrecy[idx] = meta['sr']
            paper_rbob[idx] = meta['rb']
            paper_reve[idx] = meta['re']
            print(f"  Paper SNR={meta['snr_db']:+3d} dB: R_Bob={meta['rb']:.4f}, R_Eve={meta['re']:.4f}, Secrecy={meta['sr']:.4f}")
        elif group == 'xi_sweep':
            snr_idx, xi_idx = meta['snr_idx'], meta['xi_idx']
            xi_secrecy[snr_idx, xi_idx] = meta['sr']
            xi_rbob[snr_idx, xi_idx] = meta['rb']
            xi_reve[snr_idx, xi_idx] = meta['re']
            print(f"  Xi sweep SNR={meta['snr_db']:+3d}dB xi={meta['xi_db']:+3d}dB: R_Bob={meta['rb']:.4f}, R_Eve={meta['re']:.4f}, Secrecy={meta['sr']:.4f}")
        elif group == 'realistic':
            sc_idx, pt_idx = meta['sc_idx'], meta['pt_idx']
            pt_secrecy[sc_idx, pt_idx] = meta['sr']
            pt_rbob[sc_idx, pt_idx] = meta['rb']
            pt_reve[sc_idx, pt_idx] = meta['re']

    return dict(
        snr_range_db=SNR_RANGE_DB,
        pt_range_dbm=PT_RANGE_DBM,
        xi_range_db=XI_RANGE_DB,
        xi_snr_values_db=np.array(XI_SNR_VALUES_DB),
        xi_secrecy=xi_secrecy,
        xi_rbob=xi_rbob,
        xi_reve=xi_reve,
        paper_secrecy=paper_secrecy,
        paper_rbob=paper_rbob,
        paper_reve=paper_reve,
        scenario_labels=np.array(scenario_labels),
        pt_secrecy=pt_secrecy,
        pt_rbob=pt_rbob,
        pt_reve=pt_reve,
    )


def save_results(data: dict, path: str):
    np.savez(path, **data)
    print(f"Results saved to {path}")


def load_results(path: str) -> dict:
    loaded = np.load(path, allow_pickle=True)
    return {k: loaded[k] for k in loaded.files}


###############################################################################
# Plotting (pure visualization, no computation)
###############################################################################

def plot_paper_replication(data: dict, output_path: str):
    secrecy = data['paper_secrecy']
    rbob = data['paper_rbob']
    reve = data['paper_reve']
    snr = data['snr_range_db']

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(snr, secrecy, 'k-o', label="Secrecy Rate", markersize=5)
    ax.plot(snr, rbob, 'b--s', label=r"$R_{\mathrm{Bob}}$", markersize=4, alpha=0.7)
    ax.plot(snr, reve, 'r--^', label=r"$R_{\mathrm{Eve}}$", markersize=4, alpha=0.7)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Rate (bits/s/Hz)")
    ax.set_title(f"Paper Replication - SSK (N={N_ELEMENTS}, K={K_ANTENNAS}, $\\eta$={ETA}, unit channels)")
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, format='pdf')
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_single_rate(
    labels: np.ndarray,
    secrecy_2d: np.ndarray,
    rbob_2d: np.ndarray,
    reve_2d: np.ndarray,
    pt_range_dbm: np.ndarray,
    rate_key: str,
    title: str,
    ylabel: str,
    output_path: str,
):
    data_2d = {'secrecy': secrecy_2d, 'bob': rbob_2d, 'eve': reve_2d}[rate_key]
    num_scenarios = data_2d.shape[0]
    pt_step = pt_range_dbm[1] - pt_range_dbm[0] if len(pt_range_dbm) > 1 else 1
    marker_offsets = np.linspace(-0.15, 0.15, num_scenarios) * pt_step

    fig, ax = plt.subplots(figsize=(16, 6))
    for idx in range(num_scenarios):
        label = str(labels[idx])
        values = data_2d[idx]
        ax.plot(
            pt_range_dbm + marker_offsets[idx], values,
            label=label,
            color=PLOT_COLORS[idx % len(PLOT_COLORS)],
            linestyle=PLOT_LINESTYLES[idx % len(PLOT_LINESTYLES)],
            marker=PLOT_MARKERS[idx % len(PLOT_MARKERS)],
            markersize=5,
        )
    ax.set_xlabel("Transmit Power (dBm)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, format='pdf')
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_xi_sweep(data: dict, snr_idx: int, snr_db: int, output_path: str):
    """Plot R_Bob, R_Eve, Secrecy vs B/FPH power ratio for a given SNR."""
    xi_range = data['xi_range_db']

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(xi_range, data['xi_secrecy'][snr_idx], 'k-o', label="Secrecy Rate", markersize=5)
    ax.plot(xi_range, data['xi_rbob'][snr_idx], 'b--s',
            label=r"$R_{\mathrm{Bob}}$", markersize=4, alpha=0.7)
    ax.plot(xi_range, data['xi_reve'][snr_idx], 'r--^',
            label=r"$R_{\mathrm{Eve}}$", markersize=4, alpha=0.7)
    ax.set_xlabel(r"$\|Bx\|^2 / \|FPHx\|^2$ (dB)")
    ax.set_ylabel("Rate (bits/s/Hz)")
    ax.set_title(
        f"Secrecy vs Direct/Reflected Power Ratio "
        f"(N={N_ELEMENTS}, K={K_ANTENNAS}, $\\eta$={ETA}, SNR={snr_db} dB)"
    )
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, format='pdf')
    plt.close(fig)
    print(f"Saved {output_path}")


def generate_all_plots(data: dict):
    configure_latex()

    plot_paper_replication(data, f"{OUTPUT_DIR}/Secrecy_Rate_Paper.pdf")

    xi_snr_values = data['xi_snr_values_db']
    for snr_idx, snr_db in enumerate(xi_snr_values):
        snr_int = int(snr_db)
        plot_xi_sweep(data, snr_idx, snr_int,
                      f"{OUTPUT_DIR}/Secrecy_vs_Direct_Link_SNR{snr_int:+d}dB.pdf")

    pt_range = data['pt_range_dbm']
    labels = data['scenario_labels']
    secrecy = data['pt_secrecy']
    rbob = data['pt_rbob']
    reve = data['pt_reve']

    title_suffix = f"(N={N_ELEMENTS}, K={K_ANTENNAS}, d={DISTANCE_M}m)"
    plot_single_rate(labels, secrecy, rbob, reve, pt_range, 'secrecy',
                     f"Secrecy Rate {title_suffix}",
                     "Secrecy Rate (bits/s/Hz)",
                     f"{OUTPUT_DIR}/Secrecy_Rate.pdf")
    plot_single_rate(labels, secrecy, rbob, reve, pt_range, 'bob',
                     rf"$R_{{\mathrm{{Bob}}}}$ {title_suffix}",
                     r"$R_{\mathrm{Bob}}$ (bits/s/Hz)",
                     f"{OUTPUT_DIR}/Rate_Bob.pdf")
    plot_single_rate(labels, secrecy, rbob, reve, pt_range, 'eve',
                     rf"$R_{{\mathrm{{Eve}}}}$ {title_suffix}",
                     r"$R_{\mathrm{Eve}}$ (bits/s/Hz)",
                     f"{OUTPUT_DIR}/Rate_Eve.pdf")


###############################################################################
# Main
###############################################################################

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(RESULTS_FILE):
        data = load_results(RESULTS_FILE)
        print(f"Found cached results at {RESULTS_FILE}, skipping computation.")
    else:
        data = run_all_computations()
        save_results(data, RESULTS_FILE)

    generate_all_plots(data)


if __name__ == "__main__":
    main()
