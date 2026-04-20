"""Experiment G — End-to-End Pipeline.

Demonstrates the full workflow: generate synthetic team data → fit all
candidate distributions → select best model → compute budget impact.
Produces the publication figure `full_pipeline.png`.

Usage:
    python scripts/exp_full_pipeline.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.data_gen import generate_team_data
from src.fitting import fit_all
from src.model_selection import compare_models, ks_test
from src.budget_impact import (
    compare_distributions_impact,
    var_at_level,
    expected_shortfall,
)

# --- Configuration ---
SEED = 42
TEAM_SIZE = 50
FIGURE_DPI = 300
OUTPUT_PATH = "figures/full_pipeline.png"

# --- Setup ---
sns.set_theme(style="whitegrid", font_scale=1.0)


def main() -> None:
    """Run full end-to-end pipeline experiment."""
    print("=" * 70)
    print("END-TO-END PIPELINE — Synthetic 50-Person Team")
    print("=" * 70)

    # Step 1: Generate data
    team_data = generate_team_data(team_size=TEAM_SIZE, seed=SEED)

    print(f"\n--- Step 1: Data Generation ---")
    for component, values in team_data.items():
        print(f"  {component:>12}: n={len(values)}, mean=R${values.mean():,.0f}, "
              f"std=R${values.std():,.0f}, max=R${values.max():,.0f}")

    # Step 2: Fit all candidates to salary data
    print(f"\n--- Step 2: Fit Distributions to Salary (n={TEAM_SIZE}) ---")
    salary_fits = fit_all(team_data["salary"])
    salary_comparison = compare_models(salary_fits)
    print(salary_comparison[["Distribution", "LogLik", "AIC", "BIC", "AIC_weight"]].to_string(index=False))

    # Step 3: Fit severance data
    print(f"\n--- Step 3: Fit Distributions to Severance (n={len(team_data['severance'])}) ---")
    sev_fits = fit_all(team_data["severance"])
    sev_comparison = compare_models(sev_fits)
    print(sev_comparison[["Distribution", "LogLik", "AIC", "BIC", "AIC_weight"]].to_string(index=False))

    # Step 4: Budget impact
    print(f"\n--- Step 4: Budget Impact Analysis ---")
    salary_impact = compare_distributions_impact(salary_fits, seed=SEED)
    print("\nSalary budget impact:")
    print(salary_impact[["Distribution", "Mean", "VaR_95%", "ES_95%", "Reserve_95%"]].to_string(index=False))

    # Step 5: Best model selection
    best_salary = salary_comparison.iloc[0]["Distribution"]
    best_sev = sev_comparison.iloc[0]["Distribution"]
    print(f"\n--- Step 5: Model Selection ---")
    print(f"  Best model for salary: {best_salary}")
    print(f"  Best model for severance: {best_sev}")

    # --- Figure ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Panel 1: Salary histogram with best fit
    ax = axes[0, 0]
    salary = team_data["salary"]
    ax.hist(salary, bins=20, density=True, alpha=0.4, color="gray", edgecolor="white")
    best_salary_fit = salary_fits[0]
    x = np.linspace(salary.min() * 0.7, salary.max() * 1.1, 200)
    if best_salary_fit.distribution == "LogNormal":
        pdf = stats.lognorm.pdf(x, s=best_salary_fit.params["sigma"],
                                scale=np.exp(best_salary_fit.params["mu"]))
    elif best_salary_fit.distribution == "Normal":
        pdf = stats.norm.pdf(x, loc=best_salary_fit.params["mu"],
                            scale=best_salary_fit.params["sigma"])
    elif best_salary_fit.distribution == "Gamma":
        pdf = stats.gamma.pdf(x, a=best_salary_fit.params["alpha"],
                             scale=1.0 / best_salary_fit.params["beta"])
    else:
        pdf = np.zeros_like(x)
    ax.plot(x, pdf, linewidth=2.5, color="#2A9D8F")
    ax.set_title(f"Salary: Best = {best_salary_fit.distribution}")
    ax.set_xlabel("R$")
    ax.set_ylabel("Density")

    # Panel 2: Severance histogram with best fit
    ax = axes[0, 1]
    sev = team_data["severance"]
    ax.hist(sev, bins=15, density=True, alpha=0.4, color="gray", edgecolor="white")
    best_sev_fit = sev_fits[0]
    x_sev = np.linspace(sev.min() * 0.9, sev.max() * 1.1, 200)
    if best_sev_fit.distribution == "Pareto":
        pdf_sev = stats.pareto.pdf(x_sev, b=best_sev_fit.params["alpha"],
                                   scale=best_sev_fit.params["x_m"])
    elif best_sev_fit.distribution == "LogNormal":
        pdf_sev = stats.lognorm.pdf(x_sev, s=best_sev_fit.params["sigma"],
                                    scale=np.exp(best_sev_fit.params["mu"]))
    else:
        pdf_sev = np.zeros_like(x_sev)
    ax.plot(x_sev, pdf_sev, linewidth=2.5, color="#E63946")
    ax.set_title(f"Severance: Best = {best_sev_fit.distribution}")
    ax.set_xlabel("R$")
    ax.set_ylabel("Density")

    # Panel 3: AIC weights for salary
    ax = axes[0, 2]
    ax.barh(salary_comparison["Distribution"], salary_comparison["AIC_weight"], color="#457B9D")
    ax.set_xlabel("Akaike Weight")
    ax.set_title("Salary: Model Probabilities")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()

    # Panel 4: Budget reserve comparison (salary)
    ax = axes[1, 0]
    dists = salary_impact["Distribution"]
    reserves = salary_impact["Reserve_95%"]
    colors = ["#2A9D8F" if d == best_salary else "#E9C46A" for d in dists]
    ax.bar(dists, reserves, color=colors)
    ax.set_ylabel("Reserve above mean (R$)")
    ax.set_title("Salary: Budget Reserve at 95%")
    ax.tick_params(axis="x", rotation=30)

    # Panel 5: VaR comparison (salary)
    ax = axes[1, 1]
    vars_95 = salary_impact["VaR_95%"]
    ax.bar(dists, vars_95, color=colors)
    ax.set_ylabel("VaR 95% (R$)")
    ax.set_title("Salary: Value at Risk at 95%")
    ax.tick_params(axis="x", rotation=30)

    # Panel 6: Pipeline summary
    ax = axes[1, 2]
    ax.axis("off")
    summary_text = (
        f"PIPELINE SUMMARY\n"
        f"{'─' * 30}\n"
        f"Team size: {TEAM_SIZE}\n"
        f"Salary model: {best_salary}\n"
        f"Severance model: {best_sev}\n"
        f"{'─' * 30}\n"
        f"Salary mean: R${salary.mean():,.0f}\n"
        f"Salary VaR 95%: R${salary_impact[salary_impact['Distribution']==best_salary]['VaR_95%'].iloc[0]:,.0f}\n"
        f"{'─' * 30}\n"
        f"Workflow:\n"
        f"  Data → Fit → Compare → Select → Budget"
    )
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment="center", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

    plt.suptitle(
        "End-to-End Pipeline: Data → Fit → Select → Budget Impact",
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
