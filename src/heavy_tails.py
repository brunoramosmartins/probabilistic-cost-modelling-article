"""Tail analysis tools for heavy-tailed distributions.

Implements the Hill estimator for tail index estimation, tail probability
comparison, Q-Q plot data generation, and tail risk comparison tables.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats


def hill_estimator(data: NDArray[np.float64], k: int) -> float:
    """Estimate the tail index using the Hill estimator.

    Parameters
    ----------
    data : NDArray[np.float64]
        Observed data (positive values).
    k : int
        Number of upper order statistics to use. Must be 1 <= k < n.

    Returns
    -------
    float
        Estimated tail index alpha_hat.

    Notes
    -----
    The Hill estimator assumes the tail above the (n-k)-th order statistic
    follows a Pareto distribution:

        alpha_hat = [1/k * sum_{i=1}^{k} log(X_{(n-i+1)} / X_{(n-k)})]^{-1}

    Choose k by plotting alpha_hat vs k and looking for a stable region.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    n = len(data)
    if k >= n:
        raise ValueError(f"k must be < n (got k={k}, n={n})")

    sorted_data = np.sort(data)
    threshold = sorted_data[n - k - 1]  # X_{(n-k)}

    if threshold <= 0:
        raise ValueError("Threshold must be positive for Hill estimator")

    # k largest values: X_{(n-k+1)}, ..., X_{(n)}
    exceedances = sorted_data[n - k:]
    log_ratios = np.log(exceedances / threshold)
    mean_log_ratio = log_ratios.mean()

    if mean_log_ratio <= 0:
        return float("inf")

    return 1.0 / mean_log_ratio


def hill_plot_data(
    data: NDArray[np.float64],
    k_range: range | list[int] | None = None,
) -> tuple[list[int], list[float]]:
    """Compute Hill estimates for a range of k values (for plotting).

    Parameters
    ----------
    data : NDArray[np.float64]
        Observed data.
    k_range : range or list[int], optional
        Range of k values. Defaults to [5, 10, 15, ..., n//4].

    Returns
    -------
    tuple[list[int], list[float]]
        (k_values, alpha_estimates) for plotting.
    """
    n = len(data)
    if k_range is None:
        k_range = list(range(5, max(6, n // 4), 5))

    k_values = []
    alpha_values = []
    for k in k_range:
        if k >= n:
            break
        try:
            alpha = hill_estimator(data, k)
            if np.isfinite(alpha) and alpha > 0:
                k_values.append(k)
                alpha_values.append(alpha)
        except ValueError:
            continue

    return k_values, alpha_values


def tail_probability(
    distribution: str,
    params: dict[str, float],
    threshold: float,
) -> float:
    """Compute P(X > threshold) for a given distribution.

    Parameters
    ----------
    distribution : str
        Distribution name: 'normal', 'lognormal', 'pareto', 'gamma', 'weibull'.
    params : dict[str, float]
        Distribution parameters.
    threshold : float
        Value to compute exceedance probability for.

    Returns
    -------
    float
        P(X > threshold), i.e., the survival probability.
    """
    name = distribution.lower()

    if name == "normal":
        return float(1 - stats.norm.cdf(threshold, loc=params["mu"], scale=params["sigma"]))
    elif name == "lognormal":
        return float(1 - stats.lognorm.cdf(threshold, s=params["sigma"], scale=np.exp(params["mu"])))
    elif name == "pareto":
        return float(1 - stats.pareto.cdf(threshold, b=params["alpha"], scale=params["x_m"]))
    elif name == "gamma":
        return float(1 - stats.gamma.cdf(threshold, a=params["alpha"], scale=1.0 / params["beta"]))
    elif name == "weibull":
        return float(1 - stats.weibull_min.cdf(threshold, c=params["k"], scale=params["lam"]))
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def qq_plot_data(
    data: NDArray[np.float64],
    distribution: str,
    params: dict[str, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate Q-Q plot data (theoretical vs empirical quantiles).

    Parameters
    ----------
    data : NDArray[np.float64]
        Observed data.
    distribution : str
        Distribution name.
    params : dict[str, float]
        Distribution parameters.

    Returns
    -------
    tuple[NDArray, NDArray]
        (theoretical_quantiles, empirical_quantiles) sorted ascending.
    """
    n = len(data)
    empirical = np.sort(data)
    probabilities = (np.arange(1, n + 1) - 0.5) / n

    name = distribution.lower()
    if name == "normal":
        theoretical = stats.norm.ppf(probabilities, loc=params["mu"], scale=params["sigma"])
    elif name == "lognormal":
        theoretical = stats.lognorm.ppf(probabilities, s=params["sigma"], scale=np.exp(params["mu"]))
    elif name == "pareto":
        theoretical = stats.pareto.ppf(probabilities, b=params["alpha"], scale=params["x_m"])
    elif name == "gamma":
        theoretical = stats.gamma.ppf(probabilities, a=params["alpha"], scale=1.0 / params["beta"])
    elif name == "weibull":
        theoretical = stats.weibull_min.ppf(probabilities, c=params["k"], scale=params["lam"])
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return theoretical, empirical


def compare_tail_risk(
    thresholds: list[float],
    distributions: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Compare tail probabilities across distributions at multiple thresholds.

    Parameters
    ----------
    thresholds : list[float]
        Threshold values to evaluate P(X > threshold).
    distributions : dict[str, dict[str, float]]
        Dict mapping "dist_name" -> {"param1": val, ...}.
        The key format is "family:label", e.g., "pareto:Pareto(2.5)".

    Returns
    -------
    pd.DataFrame
        Comparison table with columns for each distribution and rows for thresholds.
    """
    rows = []
    for t in thresholds:
        row = {"Threshold": f"R${t:,.0f}"}
        for key, params in distributions.items():
            family = key.split(":")[0]
            label = key.split(":")[1] if ":" in key else key
            prob = tail_probability(family, params, t)
            row[label] = prob
        rows.append(row)

    return pd.DataFrame(rows)
