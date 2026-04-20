"""Experiment F — Model Comparison.

Fits all candidate distributions to synthetic salary data and produces a
comprehensive AIC/BIC/KS comparison table. Demonstrates that the true
generating distribution is recovered by information criteria.

Produces the publication figure `model_comparison.png`.

Usage:
    python scripts/exp_model_comparison.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_gen import generate_salary_data
from src.distributions import LogNormalDist
from src.fitting import fit_all
from src.model_selection import compare_models, ks_test

# --- Configuration ---
SEED = 42
N_SAMPLES = 500
FIGURE_DPI = 300
OUTPUT_PATH = "figures/model_comparison.png"

# --- Setup ---
sns.set_theme(style="whitegrid", font_scale=1.0)


def main() -> None:
    """Run model comparison experiment."""
    # Generate data from known LogNormal distribution
    data = generate_salary_data(N_SAMPLES, seed=SEED)

    # Fit all candidates
    results = fit_all(data)

    # Create comparison table
    df = compare_models(results)

    print("=" * 70)
    print("MODEL COMPARISON — Synthetic Salary Data")
    print(f"Ground truth: LogNormal(mu=9.1, sigma=0.4), n={N_SAMPLES}")
    print("=" * 70)
    print()
    print(df[["Distribution", "k", "LogLik", "AIC", "BIC", "AIC_weight", "delta_AIC"]].to_string(index=False))
    print()

    # Run KS test for each fitted model
    print("Goodness-of-Fit (KS test):")
    print("-" * 50)
    for r in results:
        if r.distribution == "LogNormal":
            from scipy.stats import lognorm
            cdf = lambda x, r=r: lognorm.cdf(x, s=r.params["sigma"], scale=np.exp(r.params["mu"]))
        elif r.distribution == "Normal":
            from scipy.stats import norm
            cdf = lambda x, r=r: norm.cdf(x, loc=r.params["mu"], scale=r.params["sigma"])
        elif r.distribution == "Gamma":
            from scipy.stats import gamma as gamma_dist
            cdf = lambda x, r=r: gamma_dist.cdf(x, a=r.params["alpha"], scale=1.0/r.params["beta"])
        elif r.distribution == "Weibull":
            from scipy.stats import weibull_min
            cdf = lambda x, r=r: weibull_min.cdf(x, c=r.params["k"], scale=r.params["lam"])
        elif r.distribution == "Pareto":
            from scipy.stats import pareto
            cdf = lambda x, r=r: pareto.cdf(x, b=r.params["alpha"], scale=r.params["x_m"])
        else:
            continue

        ks_result = ks_test(data, cdf)
        status = "REJECT" if ks_result.reject else "accept"
        print(f"  {r.distribution:12s}: D={ks_result.statistic:.4f}, p={ks_result.p_value:.4f} [{status}]")

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: AIC/BIC comparison bar chart
    ax = axes[0]
    x_pos = np.arange(len(df))
    width = 0.35
    ax.barh(x_pos - width / 2, df["delta_AIC"], width, label="delta AIC", color="#457B9D")
    ax.barh(x_pos + width / 2, df["delta_BIC"], width, label="delta BIC", color="#E63946")
    ax.set_yticks(x_pos)
    ax.set_yticklabels(df["Distribution"])
    ax.set_xlabel("Delta from best model")
    ax.set_title("Model Comparison: AIC and BIC")
    ax.legend()
    ax.invert_yaxis()

    # Panel 2: Akaike weights
    ax = axes[1]
    colors = ["#2A9D8F" if w > 0.5 else "#E9C46A" if w > 0.1 else "#264653"
              for w in df["AIC_weight"]]
    ax.barh(x_pos, df["AIC_weight"], color=colors)
    ax.set_yticks(x_pos)
    ax.set_yticklabels(df["Distribution"])
    ax.set_xlabel("Akaike Weight (probability of being best)")
    ax.set_title("Akaike Weights")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()

    plt.suptitle(
        f"Model Selection: LogNormal data (n={N_SAMPLES})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()

    print(f"\nFigure saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
