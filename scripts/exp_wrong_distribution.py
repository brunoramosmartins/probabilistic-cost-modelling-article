"""Experiment C — Wrong Distribution Impact.

Demonstrates the budget error from fitting a Normal distribution to data
that is actually LogNormal. Compares P(overbudget) predictions, VaR, and
budget reserves under correct vs incorrect distributional assumptions.

Produces the publication figure `wrong_distribution_impact.png`.

Usage:
    python scripts/exp_wrong_distribution.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.data_gen import generate_salary_data
from src.fitting import fit_lognormal, fit_normal
from src.budget_impact import var_at_level, expected_shortfall, budget_reserve

# --- Configuration ---
SEED = 42
N_SAMPLES = 1000
N_SIMULATIONS = 200_000
FIGURE_DPI = 300
OUTPUT_PATH = "figures/wrong_distribution_impact.png"

# --- Setup ---
sns.set_theme(style="whitegrid", font_scale=1.1)


def main() -> None:
    """Run wrong distribution experiment."""
    # Generate salary data from LogNormal (ground truth)
    data = generate_salary_data(N_SAMPLES, seed=SEED)

    # Fit both models
    ln_fit = fit_lognormal(data)
    n_fit = fit_normal(data)

    # Simulate from each fitted model
    rng = np.random.default_rng(SEED)
    sim_lognormal = rng.lognormal(ln_fit.params["mu"], ln_fit.params["sigma"], size=N_SIMULATIONS)
    sim_normal = rng.normal(n_fit.params["mu"], n_fit.params["sigma"], size=N_SIMULATIONS)

    # Budget thresholds (multiples of mean)
    mean_salary = data.mean()
    budget_ceilings = [mean_salary * m for m in [1.2, 1.5, 2.0, 2.5, 3.0]]

    print("=" * 65)
    print("WRONG DISTRIBUTION IMPACT — Normal vs LogNormal for Salary")
    print(f"Data: n={N_SAMPLES}, true model=LogNormal(9.1, 0.4)")
    print(f"Sample mean: R${mean_salary:,.0f}")
    print("=" * 65)

    print(f"\n{'Budget Ceiling':>15} {'P(over) LN':>12} {'P(over) Normal':>15} {'Ratio':>8}")
    print("-" * 55)
    for ceiling in budget_ceilings:
        p_ln = (sim_lognormal > ceiling).mean()
        p_n = (sim_normal > ceiling).mean()
        ratio = p_ln / max(p_n, 1e-10)
        print(f"R${ceiling:>12,.0f} {p_ln:>12.4f} {p_n:>15.4f} {ratio:>7.1f}x")

    # VaR/ES comparison
    print(f"\n{'Metric':<15} {'LogNormal':>12} {'Normal':>12} {'Difference':>12}")
    print("-" * 55)
    for cl in [0.90, 0.95, 0.99]:
        var_ln = var_at_level(sim_lognormal, cl)
        var_n = var_at_level(sim_normal, cl)
        print(f"VaR {int(cl*100)}%{'':>8} R${var_ln:>9,.0f} R${var_n:>9,.0f} R${var_ln-var_n:>+9,.0f}")

    for cl in [0.90, 0.95, 0.99]:
        es_ln = expected_shortfall(sim_lognormal, cl)
        es_n = expected_shortfall(sim_normal, cl)
        print(f"ES {int(cl*100)}%{'':>9} R${es_ln:>9,.0f} R${es_n:>9,.0f} R${es_ln-es_n:>+9,.0f}")

    # --- Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: PDFs overlaid on data
    ax = axes[0, 0]
    ax.hist(data, bins=50, density=True, alpha=0.3, color="gray", label="Data (LogNormal truth)")
    x = np.linspace(data.min() * 0.5, np.percentile(data, 99.5), 500)
    pdf_ln = stats.lognorm.pdf(x, s=ln_fit.params["sigma"], scale=np.exp(ln_fit.params["mu"]))
    pdf_n = stats.norm.pdf(x, loc=n_fit.params["mu"], scale=n_fit.params["sigma"])
    ax.plot(x, pdf_ln, linewidth=2.5, color="#2A9D8F", label="LogNormal fit (correct)")
    ax.plot(x, pdf_n, linewidth=2.5, color="#E63946", linestyle="--", label="Normal fit (WRONG)")
    ax.set_xlabel("Salary (R$)")
    ax.set_ylabel("Density")
    ax.set_title("Fitted Distributions")
    ax.legend()

    # Panel 2: P(overbudget) at different ceilings
    ax = axes[0, 1]
    multiples = [1.2, 1.5, 2.0, 2.5, 3.0]
    p_over_ln = [(sim_lognormal > mean_salary * m).mean() for m in multiples]
    p_over_n = [(sim_normal > mean_salary * m).mean() for m in multiples]
    ax.semilogy(multiples, p_over_ln, "o-", linewidth=2, color="#2A9D8F", label="LogNormal (correct)")
    ax.semilogy(multiples, p_over_n, "s--", linewidth=2, color="#E63946", label="Normal (wrong)")
    ax.set_xlabel("Budget ceiling (× mean)")
    ax.set_ylabel("P(cost > ceiling) [log scale]")
    ax.set_title("Overbudget Probability")
    ax.legend()

    # Panel 3: VaR comparison
    ax = axes[1, 0]
    cls = np.linspace(0.80, 0.995, 50)
    var_ln_curve = [var_at_level(sim_lognormal, cl) for cl in cls]
    var_n_curve = [var_at_level(sim_normal, cl) for cl in cls]
    ax.plot(cls * 100, var_ln_curve, linewidth=2.5, color="#2A9D8F", label="LogNormal VaR")
    ax.plot(cls * 100, var_n_curve, linewidth=2.5, color="#E63946", linestyle="--", label="Normal VaR")
    ax.fill_between(cls * 100, var_n_curve, var_ln_curve, alpha=0.2, color="#E9C46A",
                    label="Underestimation gap")
    ax.set_xlabel("Confidence Level (%)")
    ax.set_ylabel("Value at Risk (R$)")
    ax.set_title("VaR: Correct vs Wrong Distribution")
    ax.legend()

    # Panel 4: Reserve needed
    ax = axes[1, 1]
    reserve_ln = [budget_reserve(sim_lognormal, cl) for cl in cls]
    reserve_n = [budget_reserve(sim_normal, cl) for cl in cls]
    ax.plot(cls * 100, reserve_ln, linewidth=2.5, color="#2A9D8F", label="LogNormal reserve")
    ax.plot(cls * 100, reserve_n, linewidth=2.5, color="#E63946", linestyle="--", label="Normal reserve")
    ax.set_xlabel("Confidence Level (%)")
    ax.set_ylabel("Reserve above mean (R$)")
    ax.set_title("Budget Reserve Needed")
    ax.legend()

    plt.suptitle(
        "The Cost of the Wrong Distribution: Normal vs LogNormal for Salary",
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
