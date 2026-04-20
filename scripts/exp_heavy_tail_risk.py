"""Experiment E — Heavy Tail Risk.

Compares P(cost > threshold) under Normal vs Pareto vs LogNormal assumptions,
demonstrating how the Normal systematically underestimates tail risk.
Produces the publication figure `tail_risk_comparison.png`.

Usage:
    python scripts/exp_heavy_tail_risk.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.heavy_tails import tail_probability, compare_tail_risk
from src.budget_impact import (
    analytical_var_normal,
    analytical_var_pareto,
    analytical_es_normal,
    analytical_es_pareto,
)

# --- Configuration ---
SEED = 42
FIGURE_DPI = 300
OUTPUT_PATH = "figures/tail_risk_comparison.png"

# --- Setup ---
sns.set_theme(style="whitegrid", font_scale=1.1)


def main() -> None:
    """Run heavy tail risk experiment."""
    # Pareto parameters (ground truth for severance costs)
    alpha_pareto = 2.5
    x_m = 10000.0

    # Moment-matched Normal
    mu_pareto = alpha_pareto * x_m / (alpha_pareto - 1)  # 16667
    var_pareto = alpha_pareto * x_m**2 / ((alpha_pareto - 1)**2 * (alpha_pareto - 2))
    sigma_pareto = np.sqrt(var_pareto)  # 14907

    print("=" * 60)
    print("HEAVY TAIL RISK COMPARISON")
    print(f"Pareto(alpha={alpha_pareto}, x_m={x_m:.0f})")
    print(f"Normal(mu={mu_pareto:.0f}, sigma={sigma_pareto:.0f}) [moment-matched]")
    print("=" * 60)

    # Compare tail probabilities
    thresholds = [30000, 50000, 75000, 100000, 150000, 200000]
    print(f"\n{'Threshold':>12} {'P_Pareto':>12} {'P_Normal':>12} {'Ratio':>8}")
    print("-" * 50)
    for t in thresholds:
        p_pareto = tail_probability("pareto", {"alpha": alpha_pareto, "x_m": x_m}, t)
        p_normal = tail_probability("normal", {"mu": mu_pareto, "sigma": sigma_pareto}, t)
        ratio = p_pareto / max(p_normal, 1e-20)
        print(f"R${t:>10,} {p_pareto:>12.6f} {p_normal:>12.6f} {ratio:>8.0f}x")

    # VaR and ES comparison
    print(f"\n{'Metric':<15} {'Pareto':>12} {'Normal':>12} {'Ratio':>8}")
    print("-" * 50)
    for cl in [0.90, 0.95, 0.99]:
        var_p = analytical_var_pareto(alpha_pareto, x_m, cl)
        var_n = analytical_var_normal(mu_pareto, sigma_pareto, cl)
        print(f"VaR_{int(cl*100)}%{'':>8} R${var_p:>9,.0f} R${var_n:>9,.0f} {var_p/var_n:>7.2f}x")

    for cl in [0.90, 0.95, 0.99]:
        es_p = analytical_es_pareto(alpha_pareto, x_m, cl)
        es_n = analytical_es_normal(mu_pareto, sigma_pareto, cl)
        print(f"ES_{int(cl*100)}%{'':>9} R${es_p:>9,.0f} R${es_n:>9,.0f} {es_p/es_n:>7.2f}x")

    # --- Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Survival functions (log scale)
    ax = axes[0, 0]
    x = np.linspace(x_m, 200000, 1000)
    surv_pareto = [(x_m / xi) ** alpha_pareto for xi in x]
    from scipy.stats import norm
    surv_normal = [1 - norm.cdf(xi, loc=mu_pareto, scale=sigma_pareto) for xi in x]

    ax.semilogy(x, surv_pareto, linewidth=2.5, color="#E63946", label="Pareto(2.5, 10000)")
    ax.semilogy(x, surv_normal, linewidth=2.5, color="#457B9D", label="Normal (moment-matched)")
    ax.set_xlabel("Cost threshold (R$)")
    ax.set_ylabel("P(X > threshold) [log scale]")
    ax.set_title("Survival Functions: Pareto vs Normal")
    ax.legend()
    ax.set_ylim(1e-8, 1)

    # Panel 2: Ratio of tail probabilities
    ax = axes[0, 1]
    thresholds_plot = np.linspace(20000, 150000, 100)
    ratios = []
    for t in thresholds_plot:
        p_p = tail_probability("pareto", {"alpha": alpha_pareto, "x_m": x_m}, t)
        p_n = tail_probability("normal", {"mu": mu_pareto, "sigma": sigma_pareto}, t)
        ratios.append(p_p / max(p_n, 1e-30))

    ax.semilogy(thresholds_plot, ratios, linewidth=2.5, color="#264653")
    ax.axhline(10, color="red", linestyle="--", alpha=0.5, label="10x underestimation")
    ax.axhline(100, color="red", linestyle=":", alpha=0.5, label="100x underestimation")
    ax.set_xlabel("Cost threshold (R$)")
    ax.set_ylabel("P_Pareto / P_Normal [log scale]")
    ax.set_title("How Much Normal Underestimates Tail Risk")
    ax.legend()

    # Panel 3: VaR comparison
    ax = axes[1, 0]
    confidence_levels = np.linspace(0.80, 0.995, 50)
    vars_pareto = [analytical_var_pareto(alpha_pareto, x_m, cl) for cl in confidence_levels]
    vars_normal = [analytical_var_normal(mu_pareto, sigma_pareto, cl) for cl in confidence_levels]

    ax.plot(confidence_levels * 100, vars_pareto, linewidth=2.5, color="#E63946", label="Pareto VaR")
    ax.plot(confidence_levels * 100, vars_normal, linewidth=2.5, color="#457B9D", label="Normal VaR")
    ax.set_xlabel("Confidence Level (%)")
    ax.set_ylabel("Value at Risk (R$)")
    ax.set_title("VaR: Pareto vs Normal")
    ax.legend()

    # Panel 4: ES comparison
    ax = axes[1, 1]
    es_pareto = [analytical_es_pareto(alpha_pareto, x_m, cl) for cl in confidence_levels]
    es_normal = [analytical_es_normal(mu_pareto, sigma_pareto, cl) for cl in confidence_levels]

    ax.plot(confidence_levels * 100, es_pareto, linewidth=2.5, color="#E63946", label="Pareto ES")
    ax.plot(confidence_levels * 100, es_normal, linewidth=2.5, color="#457B9D", label="Normal ES")
    ax.set_xlabel("Confidence Level (%)")
    ax.set_ylabel("Expected Shortfall (R$)")
    ax.set_title("Expected Shortfall: Pareto vs Normal")
    ax.legend()

    plt.suptitle(
        "The Cost of Assuming Normality: Pareto vs Normal Tail Risk",
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
