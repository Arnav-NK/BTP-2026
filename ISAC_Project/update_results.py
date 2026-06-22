import json
import numpy as np
import matplotlib.pyplot as plt
import os

project_root = os.path.abspath(os.path.dirname(__file__))
json_path = os.path.join(project_root, "simulation_results.json")

# Load existing data
with open(json_path, "r") as f:
    data = json.load(f)

all_runs = data["all_runs"]
SNR_dB = data["parameters"]["SNR_dB"]
avg_ber = data["averaged_BER"]
M = data["parameters"]["M"]
Q = data["parameters"]["Q"]
N = data["parameters"]["N"]
alpha = data["parameters"]["alpha"]
num_runs = data["parameters"]["num_runs"]

# Extract all BERs to a 2D array
all_ber = np.array([run["BER"] for run in all_runs])
all_gains = np.array([run["channel_gain"] for run in all_runs])

# Find best and worst runs by overall mean BER across all SNRs
mean_ber_per_run = np.mean(all_ber, axis=1)
best_idx  = int(np.argmin(mean_ber_per_run))
worst_idx = int(np.argmax(mean_ber_per_run))

# Update JSON structure
if "best_BER" in data:
    del data["best_BER"]
if "worst_BER" in data:
    del data["worst_BER"]

data["best_run"] = {
    "run_index": best_idx,
    "channel_gain": all_gains[best_idx],
    "BER": all_ber[best_idx].tolist(),
}

data["worst_run"] = {
    "run_index": worst_idx,
    "channel_gain": all_gains[worst_idx],
    "BER": all_ber[worst_idx].tolist(),
}

# Save updated JSON
with open(json_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"Updated JSON saved with Best Run #{best_idx+1} and Worst Run #{worst_idx+1}.")

# Regenerate plot
plt.figure(figsize=(10, 6))

plt.semilogy(SNR_dB, avg_ber,
             'ro-', linewidth=2, markersize=7,
             label='Average (100 runs)')
plt.semilogy(SNR_dB, all_ber[best_idx],
             'gs--', linewidth=1.5, markersize=6,
             label=f'Best  (Run #{best_idx + 1}, Gain={all_gains[best_idx]:.4f})')
plt.semilogy(SNR_dB, all_ber[worst_idx],
             'b^--', linewidth=1.5, markersize=6,
             label=f'Worst (Run #{worst_idx + 1}, Gain={all_gains[worst_idx]:.4f})')

plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.xlabel("SNR (dB)", fontsize=13)
plt.ylabel("BER", fontsize=13)
plt.title("BER Performance - SEFDM + SBL-IEG IRS\n"
          f"(M={M}, Q={Q}, N={N}, alpha={alpha}, {num_runs} Monte Carlo runs)",
          fontsize=13)
plt.legend(fontsize=11)
plt.tight_layout()

plot_path = os.path.join(project_root, "ber_monte_carlo.png")
plt.savefig(plot_path, dpi=200)
print(f"Plot saved to: {plot_path}")
