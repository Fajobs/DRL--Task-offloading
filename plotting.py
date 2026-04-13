"""
plotting.py — Generate all result figures (Figs. 3-7 from the paper).

Each function takes pre-computed data and saves a PNG to the `results/`
directory.  Matplotlib is configured to use the non-interactive "Agg"
backend so the script can run headlessly on servers or CI.

Figures produced
----------------
fig3_loss_convergence.png    — DQN training loss over epochs
fig4_reward_convergence.png  — DQN cumulative reward over epochs
fig5_scheme_comparison.png   — bar chart comparing all four schemes
fig6_scenario_comparison.png — grouped bars: latency vs. task-size scenario
fig7_density_comparison.png  — line plot: latency vs. task density
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no GUI needed — render straight to file
import matplotlib.pyplot as plt

# All plots are saved here
RESULTS_DIR = "results"


def _ensure_dir():
    """Create the results directory if it doesn't exist yet."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# Fig. 3 — Loss convergence during DQN training
# ═══════════════════════════════════════════════════════════════════════

def plot_loss_convergence(loss_history: list):
    """
    Plot the training loss (MSE between predicted and target Q-values)
    across epochs.  A decreasing trend indicates the network is learning.

    Parameters
    ----------
    loss_history : list of float — one average-loss value per epoch.
    """
    _ensure_dir()
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, linewidth=1.2, color="tab:blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Fig. 3: Convergence of Loss Function During DQN Training")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig3_loss_convergence.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════
# Fig. 4 — Reward convergence of DQN
# ═══════════════════════════════════════════════════════════════════════

def plot_reward_convergence(reward_history: list):
    """
    Plot the average episode reward (= negative total latency) per epoch.
    An increasing (less negative) trend means the agent is finding
    lower-latency offloading strategies.

    Parameters
    ----------
    reward_history : list of float — one average-reward value per epoch.
    """
    _ensure_dir()
    plt.figure(figsize=(8, 5))
    plt.plot(reward_history, linewidth=1.2, color="tab:orange")
    plt.xlabel("Epoch")
    plt.ylabel("Reward (negative latency)")
    plt.title("Fig. 4: Reward Convergence Analysis of DQN")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig4_reward_convergence.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════
# Fig. 5 — Average latency under different schemes (bar chart)
# ═══════════════════════════════════════════════════════════════════════

def plot_scheme_comparison(results: dict):
    """
    Bar chart comparing the average latency of Random, Q-Learning, GA,
    and DQN under the default scenario.

    Parameters
    ----------
    results : dict  — {"Random": float, "Q-Learning": float, …}
    """
    _ensure_dir()
    schemes = list(results.keys())
    values  = [results[s] for s in schemes]
    colors  = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(schemes, values, color=colors[:len(schemes)], width=0.5)

    # Annotate each bar with its numeric value
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.4f}", ha="center", va="bottom", fontsize=9,
        )

    plt.ylabel("Average Latency (s)")
    plt.title("Fig. 5: Average Latency Under Different Schemes")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig5_scheme_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════
# Fig. 6 — Average latency vs. scenario (grouped bar chart)
# ═══════════════════════════════════════════════════════════════════════

def plot_scenario_comparison(scenario_results: dict):
    """
    Grouped bar chart showing how latency changes as task sizes grow
    from Scenario I (smallest) to Scenario IV (largest).

    Parameters
    ----------
    scenario_results : dict of dict
        Outer key = scenario label ("I", "II", …)
        Inner key = scheme name → average latency float.
    """
    _ensure_dir()
    scenarios = list(scenario_results.keys())
    schemes   = list(scenario_results[scenarios[0]].keys())
    x         = np.arange(len(scenarios))
    width     = 0.18
    colors    = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    plt.figure(figsize=(10, 6))
    for i, scheme in enumerate(schemes):
        vals = [scenario_results[sc][scheme] for sc in scenarios]
        plt.bar(x + i * width, vals, width, label=scheme, color=colors[i])

    plt.xticks(x + width * 1.5, [f"Scenario {s}" for s in scenarios])
    plt.ylabel("Average Latency (s)")
    plt.title("Fig. 6: Average Latency Versus Scenario")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig6_scenario_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════
# Fig. 7 — Average latency vs. task density (line plot)
# ═══════════════════════════════════════════════════════════════════════

def plot_density_comparison(density_results: dict):
    """
    Line plot showing how latency scales with the number of tasks
    generated per 100 s (task density).

    Parameters
    ----------
    density_results : dict of dict
        Outer key = density (int), inner key = scheme name → latency.
    """
    _ensure_dir()
    densities = sorted(density_results.keys())
    schemes   = list(density_results[densities[0]].keys())
    markers   = ["o", "s", "^", "D"]
    colors    = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    plt.figure(figsize=(10, 6))
    for i, scheme in enumerate(schemes):
        vals = [density_results[d][scheme] for d in densities]
        plt.plot(
            densities, vals,
            marker=markers[i], label=scheme,
            color=colors[i], linewidth=1.5, markersize=6,
        )

    plt.xlabel("Task Density (tasks per 100 s)")
    plt.ylabel("Average Latency (s)")
    plt.title("Fig. 7: Average Latency Versus Task Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig7_density_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")
