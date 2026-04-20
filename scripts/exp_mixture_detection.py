"""Experiment D — Mixture Detection.

Generates bimodal salary data (junior + senior clusters) and demonstrates
that GMM correctly identifies the 2-component structure.
Produces the publication figure `mixture_detection.png`.

Usage:
    python scripts/exp_mixture_detection.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.data_gen import generate_mixed_data
from src.distributions import NormalDist
from src.mixture import detect_multimodality, fit_gmm

# --- Configuration ---
SEED = 42
N_SAMPLES = 1000
FIGURE_DPI = 300
OUTPUT_PATH = "figures/mixture_detection.png"

# --- Setup ---
sns.set_theme(style="whitegrid", font_scale=1.1)


def main() -> None:
    """Run mixture detection experiment."""
    # Generate bimodal salary data
    junior = NormalDist(mu=8000, sigma=1500)
    senior = NormalDist(mu=18000, sigma=2500)
    data = generate_mixed_data(N_SAMPLES, [junior, senior], [0.6, 0.4], seed=SEED)

    # Detect multimodality
    detection = detect_multimodality(data, seed=SEED)

    # Fit the optimal K model
    gmm_result = fit_gmm(data, K=detection.optimal_K, seed=SEED)

    # Also fit K=1 for comparison
    gmm_1 = fit_gmm(data, K=1, seed=SEED)

    print("=" * 60)
    print("MIXTURE DETECTION — Bimodal Salary Data")
    print(f"Ground truth: 60% N(8000, 1500^2) + 40% N(18000, 2500^2)")
    print("=" * 60)
    print(f"\nMultimodal: {detection.is_multimodal}")
    print(f"Optimal K: {detection.optimal_K}")
    print(f"Evidence: {detection.evidence}")
    print(f"\nFitted parameters (K={detection.optimal_K}):")
    order = np.argsort(gmm_result.means)
    for idx, k in enumerate(order):
        print(f"  Component {idx+1}: weight={gmm_result.weights[k]:.3f}, "
              f"mean={gmm_result.means[k]:.0f}, "
              f"std={np.sqrt(gmm_result.variances[k]):.0f}")

    # --- Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Data histogram with single Normal fit
    ax = axes[0, 0]
    ax.hist(data, bins=50, density=True, alpha=0.4, color="gray", label="Data")
    x = np.linspace(data.min(), data.max(), 500)
    mu1, var1 = gmm_1.means[0], gmm_1.variances[0]
    ax.plot(x, stats.norm.pdf(x, mu1, np.sqrt(var1)),
            "r--", linewidth=2, label=f"Single Normal (mean={mu1:.0f})")
    ax.set_title("Single Normal Fit (WRONG)")
    ax.set_xlabel("Salary (R$)")
    ax.set_ylabel("Density")
    ax.legend()

    # Panel 2: Data with GMM components
    ax = axes[0, 1]
    ax.hist(data, bins=50, density=True, alpha=0.4, color="gray", label="Data")
    colors = ["#2A9D8F", "#E63946", "#E9C46A"]
    for k in range(gmm_result.K):
        w = gmm_result.weights[k]
        mu = gmm_result.means[k]
        sigma = np.sqrt(gmm_result.variances[k])
        component_pdf = w * stats.norm.pdf(x, mu, sigma)
        ax.plot(x, component_pdf, linewidth=2.5, color=colors[k],
                label=f"Component {k+1}: w={w:.2f}, mu={mu:.0f}")
    # Total mixture PDF
    total_pdf = sum(
        gmm_result.weights[k] * stats.norm.pdf(x, gmm_result.means[k], np.sqrt(gmm_result.variances[k]))
        for k in range(gmm_result.K)
    )
    ax.plot(x, total_pdf, "k-", linewidth=2, label="Mixture PDF")
    ax.set_title(f"GMM Fit (K={gmm_result.K}) — CORRECT")
    ax.set_xlabel("Salary (R$)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

    # Panel 3: BIC vs K
    ax = axes[1, 0]
    bics = [r.bic for r in detection.gmm_results]
    Ks = list(range(1, len(bics) + 1))
    ax.plot(Ks, bics, "o-", linewidth=2, markersize=8, color="#457B9D")
    ax.axvline(detection.optimal_K, color="red", linestyle="--", alpha=0.7,
               label=f"Optimal K={detection.optimal_K}")
    ax.set_xlabel("Number of Components (K)")
    ax.set_ylabel("BIC (lower is better)")
    ax.set_title("BIC-based Model Selection")
    ax.set_xticks(Ks)
    ax.legend()

    # Panel 4: Responsibilities (coloring data by assignment)
    ax = axes[1, 1]
    assignments = np.argmax(gmm_result.responsibilities, axis=1)
    for k in range(gmm_result.K):
        mask = assignments == k
        ax.hist(data[mask], bins=30, alpha=0.6, color=colors[k],
                label=f"Assigned to Component {k+1} (n={mask.sum()})")
    ax.set_title("Data Colored by Component Assignment")
    ax.set_xlabel("Salary (R$)")
    ax.set_ylabel("Count")
    ax.legend()

    plt.suptitle(
        "Mixture Detection: Bimodal Salary Data",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()

    print(f"\nFigure saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
