"""
Monte Carlo BER Simulation for SEFDM + SBL-IEG IRS

Runs the full simulation over 100 independent channel realizations,
averages the BER across all runs, identifies best/worst cases,
and saves all results to a single JSON file.
"""

import sys
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

# -- ensure project root is on the path --
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.channel import generate_channels
from src.grouping import sbl_grouping, grouped_channel
from src.sefdm import generate_sefdm_matrix, mmse_detector


# =============================================
# System Parameters
# =============================================

N          = 64        # Number of SEFDM sub-carriers
M          = 1000      # Number of RIS elements
Q          = 250       # Number of groups for SBL
alpha      = 0.85      # SEFDM bandwidth compression factor
num_blocks = 1000      # Data blocks per SNR point (for BER averaging)
num_runs   = 100       # Number of independent Monte Carlo runs

SNR_dB = np.arange(0, 22, 2)   # SNR range: 0 to 20 dB (step 2)


def run_single_simulation(run_id, N, M, Q, alpha, num_blocks, SNR_dB):
    """
    Execute one full simulation with a fresh channel realization.

    Returns
    -------
    ber_list : list[float]
        BER value for each SNR point.
    channel_gain : float
        |h_eff|^2 for this realization.
    """
    # -- Generate random channels --
    h_BI, h_IU = generate_channels(M)
    cascaded = h_BI * h_IU

    # -- SBL Grouping --
    groups = sbl_grouping(cascaded, Q=Q)

    # -- Effective channel after IRS phase alignment --
    h_eff = grouped_channel(h_BI, h_IU, groups)
    h_eff = h_eff / (M * (np.pi / 4))

    channel_gain = float(np.abs(h_eff) ** 2)

    # -- SEFDM modulation matrix (fixed for all blocks) --
    F_SEFDM = generate_sefdm_matrix(N, alpha)

    ber_list = []

    for snr_db in SNR_dB:
        noise_var = 10 ** (-snr_db / 10)
        total_errors = 0
        total_bits   = 0

        for _ in range(num_blocks):
            # Random BPSK bits
            bits = np.random.randint(0, 2, N)
            s    = 1 - 2 * bits            # BPSK mapping: 0 -> +1, 1 -> -1

            # SEFDM modulation
            x = F_SEFDM @ s

            # Channel + AWGN noise
            noise = np.sqrt(noise_var / 2) * (
                np.random.randn(N) + 1j * np.random.randn(N)
            )
            y = h_eff * x + noise

            # MMSE detection
            s_est    = mmse_detector(y, F_SEFDM, h_eff, noise_var)
            bits_est = (np.real(s_est) < 0).astype(int)

            total_errors += np.sum(bits != bits_est)
            total_bits   += N

        ber = total_errors / total_bits
        ber_list.append(float(ber))

    return ber_list, channel_gain


# =============================================
# Main Monte Carlo Loop
# =============================================

def main():
    print("=" * 60)
    print("  SEFDM + SBL-IEG IRS -- Monte Carlo BER Simulation")
    print(f"  Runs = {num_runs}  |  Blocks/SNR = {num_blocks}")
    print(f"  M = {M}  |  Q = {Q}  |  N = {N}  |  alpha = {alpha}")
    print("=" * 60)

    all_ber   = np.zeros((num_runs, len(SNR_dB)))
    all_gains = np.zeros(num_runs)

    start_time = time.time()

    for run in range(num_runs):
        t0 = time.time()
        ber_list, gain = run_single_simulation(
            run, N, M, Q, alpha, num_blocks, SNR_dB
        )
        all_ber[run, :]  = ber_list
        all_gains[run]   = gain
        elapsed = time.time() - t0
        print(f"  Run {run + 1:3d}/{num_runs}  |  "
              f"Channel Gain = {gain:.6f}  |  "
              f"Time = {elapsed:.1f}s")

    total_time = time.time() - start_time

    # -- Average BER across all runs --
    avg_ber = np.mean(all_ber, axis=0)

    # -- Identify best / worst runs (by mean BER across all SNR points) --
    mean_ber_per_run = np.mean(all_ber, axis=1)
    best_idx  = int(np.argmin(mean_ber_per_run))
    worst_idx = int(np.argmax(mean_ber_per_run))

    # =============================================
    # Print Final Results
    # =============================================

    print("\n" + "=" * 60)
    print(f"  AVERAGED BER  (over {num_runs} channel realizations)")
    print("=" * 60)
    for i, snr in enumerate(SNR_dB):
        print(f"  SNR = {snr:2d} dB  |  Avg BER = {avg_ber[i]:.6e}")

    print("\n" + "-" * 60)
    print(f"  BEST CASE -- Run #{best_idx + 1}  "
          f"(Channel Gain = {all_gains[best_idx]:.6f})")
    print("-" * 60)
    for i, snr in enumerate(SNR_dB):
        print(f"  SNR = {snr:2d} dB  |  BER = {all_ber[best_idx, i]:.6e}")

    print("\n" + "-" * 60)
    print(f"  WORST CASE -- Run #{worst_idx + 1}  "
          f"(Channel Gain = {all_gains[worst_idx]:.6f})")
    print("-" * 60)
    for i, snr in enumerate(SNR_dB):
        print(f"  SNR = {snr:2d} dB  |  BER = {all_ber[worst_idx, i]:.6e}")

    print(f"\n  Total simulation time: {total_time:.1f}s")
    print()

    # =============================================
    # Save all results to a single JSON file
    # =============================================

    results = {
        "parameters": {
            "N": N,
            "M": M,
            "Q": Q,
            "alpha": alpha,
            "num_blocks": num_blocks,
            "num_runs": num_runs,
            "SNR_dB": SNR_dB.tolist(),
        },
        "averaged_BER": avg_ber.tolist(),
        "all_runs": [],
        "best_run": {
            "run_index": best_idx,
            "channel_gain": all_gains[best_idx],
            "BER": all_ber[best_idx].tolist(),
        },
        "worst_run": {
            "run_index": worst_idx,
            "channel_gain": all_gains[worst_idx],
            "BER": all_ber[worst_idx].tolist(),
        },
        "total_time_seconds": round(total_time, 2),
    }

    # Store every individual run
    for r in range(num_runs):
        results["all_runs"].append({
            "run_index": r,
            "channel_gain": float(all_gains[r]),
            "BER": all_ber[r].tolist(),
        })

    output_path = os.path.join(project_root, "simulation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [OK] All results saved to: {output_path}")

    # =============================================
    # Plot: Avg + Best + Worst
    # =============================================

    plt.figure(figsize=(10, 6))

    plt.semilogy(SNR_dB, avg_ber,
                 'ro-', linewidth=2, markersize=7,
                 label='Average (100 runs)')
    plt.semilogy(SNR_dB, all_ber[best_idx],
                 'gs--', linewidth=1.5, markersize=6,
                 label=f'Best  (Run #{best_idx + 1}, '
                       f'Gain={all_gains[best_idx]:.4f})')
    plt.semilogy(SNR_dB, all_ber[worst_idx],
                 'b^--', linewidth=1.5, markersize=6,
                 label=f'Worst (Run #{worst_idx + 1}, '
                       f'Gain={all_gains[worst_idx]:.4f})')

    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.xlabel("SNR (dB)", fontsize=13)
    plt.ylabel("BER", fontsize=13)
    plt.title("BER Performance - SEFDM + SBL-IEG IRS\n"
              f"(M={M}, Q={Q}, N={N}, alpha={alpha}, "
              f"{num_runs} Monte Carlo runs)",
              fontsize=13)
    plt.legend(fontsize=11)
    plt.tight_layout()

    plot_path = os.path.join(project_root, "ber_monte_carlo.png")
    plt.savefig(plot_path, dpi=200)
    print(f"  [OK] Plot saved to:       {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
