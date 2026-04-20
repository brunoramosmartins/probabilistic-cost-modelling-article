"""Experiment B — MLE Convergence.

Demonstrates that MLE estimates converge to true parameters as sample size
grows, and that standard errors shrink proportionally to 1/sqrt(n).
Produces the publication figure `mle_convergence.png`.

Usage:
    python scripts/exp_mle_convergence.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.fitting import fit_lognormal

# --- Configuration ---
SEED = 42
SAMPLE_SIZES = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
N_REPLICATIONS = 200
TRUE_MU = 9.1
TRUE_SIGMA = 0.4
FIGURE_DPI = 300
OUTPUT_PATH = "figures/mle_convergence.png"

# --- Setup ---
sns.set_theme(style="whitegrid", font_scale=1.1)


def main() -> None:
    """Run MLE convergence experiment."""
    rng = np.random.default_rng(SEED)

    # Storage for results
    mu_estimates = {n: [] for n in SAMPLE_SIZES}
    sigma_estimates = {n: [] for n in SAMPLE_SIZES}
    mu_se = {n: [] for n in SAMPLE_SIZES}

    # Run replications
    for n in SAMPLE_SIZES:
        for rep in range(N_REPLICATIONS):
            data = rng.lognormal(TRUE_MU, TRUE_SIGMA, size=n)
            result = fit_lognormal(data)
            mu_estimates[n].append(result.params["mu"])
            sigma_estimates[n].append(result.params["sigma"])
            mu_se[n].append(result.se["mu"])

    # Compute summary statistics
    mu_means = [np.mean(mu_estimates[n]) for n in SAMPLE_SIZES]
    mu_stds = [np.std(mu_estimates[n]) for n in SAMPLE_SIZES]
    sigma_means = [np.mean(sigma_estimates[n]) for n in SAMPLE_SIZES]
    sigma_stds = [np.std(sigma_estimates[n]) for n in SAMPLE_SIZES]
    theoretical_se = [TRUE_SIGMA / np.sqrt(n) for n in SAMPLE_SIZES]

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: mu_hat convergence
    ax = axes[0, 0]
    ax.axhline(TRUE_MU, color="red", linestyle="--", linewidth=1.5, label=f"True mu = {TRUE_MU}")
    ax.errorbar(SAMPLE_SIZES, mu_means, yerr=mu_stds, fmt="o-", capsize=4, color="#457B9D")
    ax.set_xscale("log")
    ax.set_xlabel("Sample size (n)")
    ax.set_ylabel("mu_hat")
    ax.set_title("MLE Convergence: mu")
    ax.legend()

    # Panel 2: sigma_hat convergence
    ax = axes[0, 1]
    ax.axhline(TRUE_SIGMA, color="red", linestyle="--", linewidth=1.5, label=f"True sigma = {TRUE_SIGMA}")
    ax.errorbar(SAMPLE_SIZES, sigma_means, yerr=sigma_stds, fmt="o-", capsize=4, color="#2A9D8F")
    ax.set_xscale("log")
    ax.set_xlabel("Sample size (n)")
    ax.set_ylabel("sigma_hat")
    ax.set_title("MLE Convergence: sigma")
    ax.legend()

    # Panel 3: SE shrinks as 1/sqrt(n)
    ax = axes[1, 0]
    ax.plot(SAMPLE_SIZES, mu_stds, "o-", label="Empirical SD of mu_hat", color="#457B9D")
    ax.plot(SAMPLE_SIZES, theoretical_se, "s--", label="Theoretical SE = sigma/sqrt(n)", color="#E63946")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sample size (n)")
    ax.set_ylabel("Standard Error")
    ax.set_title("SE Shrinks as 1/sqrt(n)")
    ax.legend()

    # Panel 4: Distribution of mu_hat for different n
    ax = axes[1, 1]
    for n, color in zip([50, 500, 5000], ["#E9C46A", "#2A9D8F", "#264653"]):
        ax.hist(mu_estimates[n], bins=30, density=True, alpha=0.5, label=f"n={n}", color=color)
    ax.axvline(TRUE_MU, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("mu_hat")
    ax.set_ylabel("Density")
    ax.set_title("Sampling Distribution of mu_hat")
    ax.legend()

    plt.suptitle(
        "MLE Convergence: LogNormal(mu=9.1, sigma=0.4)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()

    print(f"Figure saved: {OUTPUT_PATH}")
    print(f"True parameters: mu={TRUE_MU}, sigma={TRUE_SIGMA}")
    print(f"Replications per sample size: {N_REPLICATIONS}")
    print(f"\nResults (n -> mu_hat mean +/- std):")
    for n, mu_m, mu_s in zip(SAMPLE_SIZES, mu_means, mu_stds):
        print(f"  n={n:>5}: mu_hat = {mu_m:.4f} +/- {mu_s:.4f}")


if __name__ == "__main__":
    main()
