"""Budget impact analysis tools.

Quantifies the downstream impact of distribution choice on budget decisions:
Value at Risk (VaR), Expected Shortfall (ES/CVaR), budget reserves, and
distribution comparison tables.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from src.fitting import FitResult


def var_at_level(samples: NDArray[np.float64], alpha: float = 0.95) -> float:
    """Compute Value at Risk at a given confidence level.

    Parameters
    ----------
    samples : NDArray[np.float64]
        Sample data or simulated values.
    alpha : float
        Confidence level (e.g., 0.95 for 95% VaR).

    Returns
    -------
    float
        The alpha-quantile of the distribution.

    Notes
    -----
    VaR_alpha = F^{-1}(alpha) = the value such that P(X <= VaR) = alpha.
    In budget context: "We are alpha% confident the cost won't exceed VaR."
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    return float(np.quantile(samples, alpha))


def expected_shortfall(samples: NDArray[np.float64], alpha: float = 0.95) -> float:
    """Compute Expected Shortfall (Conditional VaR) at a given level.

    Parameters
    ----------
    samples : NDArray[np.float64]
        Sample data or simulated values.
    alpha : float
        Confidence level.

    Returns
    -------
    float
        E[X | X > VaR_alpha] — the average cost in the worst (1-alpha) cases.

    Notes
    -----
    ES is always >= VaR. It answers: "If costs DO exceed VaR, how bad is it
    on average?" This is more informative for budget planning than VaR alone.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    var = np.quantile(samples, alpha)
    tail = samples[samples > var]
    if len(tail) == 0:
        return float(var)
    return float(tail.mean())


def budget_reserve(
    samples: NDArray[np.float64],
    confidence: float = 0.95,
) -> float:
    """Compute budget reserve needed above the mean at a given confidence.

    Parameters
    ----------
    samples : NDArray[np.float64]
        Sample data or simulated values.
    confidence : float
        Confidence level.

    Returns
    -------
    float
        VaR(confidence) - mean(samples). The extra budget beyond the expected
        cost needed to cover costs at the given confidence level.
    """
    var = var_at_level(samples, confidence)
    return float(var - samples.mean())


def analytical_var_normal(mu: float, sigma: float, alpha: float = 0.95) -> float:
    """Analytical VaR for Normal distribution.

    Parameters
    ----------
    mu : float
        Mean.
    sigma : float
        Standard deviation.
    alpha : float
        Confidence level.

    Returns
    -------
    float
        VaR = mu + sigma * z_alpha.
    """
    return float(mu + sigma * stats.norm.ppf(alpha))


def analytical_var_pareto(
    alpha_param: float,
    x_m: float,
    confidence: float = 0.95,
) -> float:
    """Analytical VaR for Pareto distribution.

    Parameters
    ----------
    alpha_param : float
        Tail index (shape parameter).
    x_m : float
        Scale (minimum value).
    confidence : float
        Confidence level.

    Returns
    -------
    float
        VaR = x_m * (1 - confidence)^{-1/alpha}.
    """
    return float(x_m * (1 - confidence) ** (-1 / alpha_param))


def analytical_es_pareto(
    alpha_param: float,
    x_m: float,
    confidence: float = 0.95,
) -> float:
    """Analytical Expected Shortfall for Pareto distribution.

    Parameters
    ----------
    alpha_param : float
        Tail index (must be > 1).
    x_m : float
        Scale parameter.
    confidence : float
        Confidence level.

    Returns
    -------
    float
        ES = (alpha / (alpha - 1)) * VaR.
    """
    if alpha_param <= 1:
        return float("inf")
    var = analytical_var_pareto(alpha_param, x_m, confidence)
    return float(alpha_param / (alpha_param - 1) * var)


def analytical_es_normal(mu: float, sigma: float, alpha: float = 0.95) -> float:
    """Analytical Expected Shortfall for Normal distribution.

    Parameters
    ----------
    mu : float
        Mean.
    sigma : float
        Standard deviation.
    alpha : float
        Confidence level.

    Returns
    -------
    float
        ES = mu + sigma * phi(z_alpha) / (1 - alpha).
    """
    z = stats.norm.ppf(alpha)
    return float(mu + sigma * stats.norm.pdf(z) / (1 - alpha))


def compare_distributions_impact(
    fit_results: list[FitResult],
    n_simulations: int = 100_000,
    confidence_levels: list[float] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Compare budget impact across fitted distributions.

    Parameters
    ----------
    fit_results : list[FitResult]
        Fitted distribution results.
    n_simulations : int
        Number of Monte Carlo samples for empirical VaR/ES.
    confidence_levels : list[float], optional
        Confidence levels to evaluate. Default [0.90, 0.95, 0.99].
    seed : int, optional
        Random seed.

    Returns
    -------
    pd.DataFrame
        Comparison table with VaR, ES, and reserve for each distribution.
    """
    if confidence_levels is None:
        confidence_levels = [0.90, 0.95, 0.99]

    rng = np.random.default_rng(seed)
    rows = []

    for r in fit_results:
        # Simulate from fitted distribution
        if r.distribution == "Normal":
            samples = rng.normal(r.params["mu"], r.params["sigma"], size=n_simulations)
        elif r.distribution == "LogNormal":
            samples = rng.lognormal(r.params["mu"], r.params["sigma"], size=n_simulations)
        elif r.distribution == "Gamma":
            samples = rng.gamma(r.params["alpha"], 1.0 / r.params["beta"], size=n_simulations)
        elif r.distribution == "Pareto":
            u = rng.uniform(0, 1, size=n_simulations)
            samples = r.params["x_m"] / u ** (1.0 / r.params["alpha"])
        elif r.distribution == "Weibull":
            u = rng.uniform(0, 1, size=n_simulations)
            samples = r.params["lam"] * (-np.log(u)) ** (1.0 / r.params["k"])
        else:
            continue

        row = {
            "Distribution": r.distribution,
            "Mean": float(samples.mean()),
            "Std": float(samples.std()),
        }

        for cl in confidence_levels:
            pct = int(cl * 100)
            row[f"VaR_{pct}%"] = var_at_level(samples, cl)
            row[f"ES_{pct}%"] = expected_shortfall(samples, cl)
            row[f"Reserve_{pct}%"] = budget_reserve(samples, cl)

        rows.append(row)

    return pd.DataFrame(rows)
