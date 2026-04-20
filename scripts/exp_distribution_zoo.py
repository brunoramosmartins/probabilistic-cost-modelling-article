"""Experiment A — Distribution Zoo.

Visualizes all five candidate distribution families fitted to the same
synthetic salary data. Produces the publication figure `distribution_zoo.png`.

Usage:
    python scripts/exp_distribution_zoo.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.distributions import (
    GammaDist,
    LogNormalDist,
    NormalDist,
    ParetoDist,
    WeibullDist,
)
from src.data_gen import generate_salary_data

# --- Configuration ---
SEED = 42
N_SAMPLES = 2000
FIGURE_DPI = 300
OUTPUT_PATH = "figures/distribution_zoo.png"

# --- Setup ---
sns.set_theme(style="whitegrid", font_scale=1.1)
np.random.seed(SEED)


def main() -> None:
    """Generate distribution zoo figure."""
    # Generate synthetic salary data from LogNormal (ground truth)
    data = generate_salary_data(N_SAMPLES, seed=SEED)

    # Define candidate distributions with parameters roughly matching the data
    sample_mean = data.mean()
    sample_var = data.var()

    candidates = {
        "Normal": NormalDist(mu=sample_mean, sigma=np.sqrt(sample_var)),
        "LogNormal": LogNormalDist(mu=9.1, sigma=0.4),
        "Gamma": GammaDist(
            alpha=sample_mean**2 / sample_var,
            beta=sample_mean / sample_var,
        ),
        "Weibull": WeibullDist(k=2.5, lam=sample_mean / 0.887),  # approx match
    }

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Histogram of data
    ax.hist(
        data,
        bins=60,
        density=True,
        alpha=0.3,
        color="gray",
        label="Synthetic data (LogNormal ground truth)",
        edgecolor="white",
        linewidth=0.5,
    )

    # PDF overlay for each candidate
    x = np.linspace(data.min() * 0.5, np.percentile(data, 99.5), 500)
    colors = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#264653"]

    for (name, dist), color in zip(candidates.items(), colors):
        ax.plot(
            x,
            dist.pdf(x),
            label=f"{name} (mean={dist.mean():,.0f})",
            linewidth=2.5,
            color=color,
        )

    ax.set_xlabel("Salary (R$)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "Distribution Zoo: Candidate Families for Salary Data",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(0, np.percentile(data, 99.5) * 1.1)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()

    print(f"Figure saved: {OUTPUT_PATH}")
    print(f"Data: n={N_SAMPLES}, mean={data.mean():,.0f}, std={data.std():,.0f}")
    print(f"Ground truth: LogNormal(mu=9.1, sigma=0.4)")


if __name__ == "__main__":
    main()
