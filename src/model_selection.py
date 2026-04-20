"""Model comparison tools for probabilistic cost modelling.

Provides AIC, BIC, Akaike weights, likelihood ratio tests, and
goodness-of-fit tests (KS, Anderson-Darling) for ranking candidate
distribution models.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from src.fitting import FitResult


def compute_aic(loglik: float, k: int) -> float:
    """Compute Akaike Information Criterion.

    Parameters
    ----------
    loglik : float
        Maximized log-likelihood.
    k : int
        Number of estimated parameters.

    Returns
    -------
    float
        AIC = 2k - 2*loglik. Lower is better.

    Notes
    -----
    AIC estimates the expected KL divergence (up to a constant) between the
    true distribution and the fitted model. The 2k term corrects for the
    optimistic bias of the training log-likelihood.
    """
    return 2 * k - 2 * loglik


def compute_aicc(loglik: float, k: int, n: int) -> float:
    """Compute corrected AIC for small samples.

    Parameters
    ----------
    loglik : float
        Maximized log-likelihood.
    k : int
        Number of estimated parameters.
    n : int
        Sample size.

    Returns
    -------
    float
        AICc = AIC + 2k(k+1)/(n-k-1). Use when n/k < 40.
    """
    aic = compute_aic(loglik, k)
    if n - k - 1 <= 0:
        return float("inf")
    correction = 2 * k * (k + 1) / (n - k - 1)
    return aic + correction


def compute_bic(loglik: float, k: int, n: int) -> float:
    """Compute Bayesian Information Criterion.

    Parameters
    ----------
    loglik : float
        Maximized log-likelihood.
    k : int
        Number of estimated parameters.
    n : int
        Sample size.

    Returns
    -------
    float
        BIC = k*log(n) - 2*loglik. Lower is better.

    Notes
    -----
    BIC approximates -2*log(marginal likelihood) via Laplace approximation.
    Penalizes complexity more than AIC for n > 7.
    """
    return k * np.log(n) - 2 * loglik


def aic_weights(aic_values: list[float]) -> NDArray[np.float64]:
    """Compute Akaike weights from a set of AIC values.

    Parameters
    ----------
    aic_values : list[float]
        AIC values for each candidate model.

    Returns
    -------
    NDArray[np.float64]
        Akaike weights (sum to 1). w_i is the approximate probability that
        model i is the best model among the candidates.

    Notes
    -----
    w_i = exp(-delta_i / 2) / sum(exp(-delta_j / 2))
    where delta_i = AIC_i - min(AIC)
    """
    aic_arr = np.array(aic_values, dtype=np.float64)
    deltas = aic_arr - aic_arr.min()
    raw_weights = np.exp(-deltas / 2)
    return raw_weights / raw_weights.sum()


@dataclass
class LRTResult:
    """Result of a likelihood ratio test.

    Attributes
    ----------
    statistic : float
        Test statistic Lambda = -2*(loglik_0 - loglik_1).
    df : int
        Degrees of freedom (difference in number of parameters).
    p_value : float
        P-value under chi-squared(df) distribution.
    reject : bool
        Whether to reject H0 at alpha=0.05.
    """

    statistic: float
    df: int
    p_value: float
    reject: bool


def likelihood_ratio_test(
    loglik_restricted: float,
    loglik_full: float,
    df: int,
    alpha: float = 0.05,
) -> LRTResult:
    """Perform a likelihood ratio test for nested models.

    Parameters
    ----------
    loglik_restricted : float
        Log-likelihood of the restricted (null) model.
    loglik_full : float
        Log-likelihood of the full (alternative) model.
    df : int
        Degrees of freedom = k_full - k_restricted.
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    LRTResult
        Test statistic, df, p-value, and decision.

    Notes
    -----
    Wilks' theorem: Lambda = -2*(l_0 - l_1) ~ chi2(df) under H0.
    Only valid for nested models.
    """
    if df <= 0:
        raise ValueError("df must be positive (full model must have more parameters)")

    statistic = -2 * (loglik_restricted - loglik_full)
    # Clamp to 0 in case of numerical issues
    statistic = max(0.0, statistic)

    p_value = float(1 - stats.chi2.cdf(statistic, df))
    reject = p_value < alpha

    return LRTResult(
        statistic=statistic,
        df=df,
        p_value=p_value,
        reject=reject,
    )


@dataclass
class GoFResult:
    """Result of a goodness-of-fit test.

    Attributes
    ----------
    test_name : str
        Name of the test ('KS' or 'AD').
    statistic : float
        Test statistic value.
    p_value : float
        P-value (approximate for AD with estimated parameters).
    reject : bool
        Whether to reject H0 at alpha=0.05.
    """

    test_name: str
    statistic: float
    p_value: float
    reject: bool


def ks_test(
    data: NDArray[np.float64],
    cdf_func,
    alpha: float = 0.05,
) -> GoFResult:
    """Perform Kolmogorov-Smirnov goodness-of-fit test.

    Parameters
    ----------
    data : NDArray[np.float64]
        Observed data.
    cdf_func : callable
        CDF function: cdf_func(x) -> array of probabilities.
        Should be the fitted CDF with parameters already set.
    alpha : float
        Significance level.

    Returns
    -------
    GoFResult
        Test statistic, p-value, and decision.

    Notes
    -----
    When parameters are estimated from data, standard KS p-values are
    conservative (actual p-value is smaller). Use with caution.
    """
    n = len(data)
    sorted_data = np.sort(data)

    # Empirical CDF values at data points
    ecdf = np.arange(1, n + 1) / n
    ecdf_minus = np.arange(0, n) / n

    # Theoretical CDF values
    tcdf = cdf_func(sorted_data)

    # KS statistic: max|F_n(x) - F_0(x)|
    d_plus = np.max(ecdf - tcdf)
    d_minus = np.max(tcdf - ecdf_minus)
    statistic = float(max(d_plus, d_minus))

    # Use scipy for p-value (Kolmogorov distribution)
    # Note: this assumes parameters are known (conservative when estimated)
    p_value = float(stats.kstwobign.sf(statistic * np.sqrt(n)))
    p_value = min(max(p_value, 0.0), 1.0)

    return GoFResult(
        test_name="KS",
        statistic=statistic,
        p_value=p_value,
        reject=p_value < alpha,
    )


def ad_test(
    data: NDArray[np.float64],
    cdf_func,
    alpha: float = 0.05,
) -> GoFResult:
    """Perform Anderson-Darling goodness-of-fit test.

    Parameters
    ----------
    data : NDArray[np.float64]
        Observed data.
    cdf_func : callable
        CDF function: cdf_func(x) -> array of probabilities.
    alpha : float
        Significance level.

    Returns
    -------
    GoFResult
        Test statistic, approximate p-value, and decision.

    Notes
    -----
    The AD test is more sensitive to tail deviations than KS.
    P-value is approximate (uses asymptotic distribution).
    """
    n = len(data)
    sorted_data = np.sort(data)

    # CDF values
    z = cdf_func(sorted_data)
    # Clamp to avoid log(0)
    z = np.clip(z, 1e-15, 1 - 1e-15)

    # AD statistic
    i = np.arange(1, n + 1)
    statistic = float(
        -n - np.sum((2 * i - 1) / n * (np.log(z) + np.log(1 - z[::-1])))
    )

    # Approximate p-value using the modified AD statistic
    # (Lewis, 1961 approximation for general distributions)
    ad_star = statistic * (1 + 0.75 / n + 2.25 / n**2)

    if ad_star >= 0.6:
        p_value = float(np.exp(1.2937 - 5.709 * ad_star + 0.0186 * ad_star**2))
    elif ad_star >= 0.34:
        p_value = float(np.exp(0.9177 - 4.279 * ad_star - 1.38 * ad_star**2))
    elif ad_star >= 0.2:
        p_value = float(1 - np.exp(-8.318 + 42.796 * ad_star - 59.938 * ad_star**2))
    else:
        p_value = float(1 - np.exp(-13.436 + 101.14 * ad_star - 223.73 * ad_star**2))

    p_value = min(max(p_value, 0.0), 1.0)

    return GoFResult(
        test_name="AD",
        statistic=statistic,
        p_value=p_value,
        reject=p_value < alpha,
    )


def compare_models(fit_results: list[FitResult]) -> pd.DataFrame:
    """Create a comprehensive model comparison table.

    Parameters
    ----------
    fit_results : list[FitResult]
        List of FitResult objects from fitting multiple distributions.

    Returns
    -------
    pd.DataFrame
        Comparison table with AIC, AICc, BIC, Akaike weights, and ranking.
    """
    rows = []
    aic_values = []
    bic_values = []

    for r in fit_results:
        aic = compute_aic(r.loglik, r.n_params)
        aicc = compute_aicc(r.loglik, r.n_params, r.n_obs)
        bic = compute_bic(r.loglik, r.n_params, r.n_obs)
        aic_values.append(aic)
        bic_values.append(bic)
        rows.append({
            "Distribution": r.distribution,
            "k": r.n_params,
            "LogLik": r.loglik,
            "AIC": aic,
            "AICc": aicc,
            "BIC": bic,
        })

    # Compute Akaike weights
    weights = aic_weights(aic_values)

    df = pd.DataFrame(rows)
    df["AIC_weight"] = weights
    df["delta_AIC"] = df["AIC"] - df["AIC"].min()
    df["delta_BIC"] = df["BIC"] - df["BIC"].min()

    # Sort by AIC
    df = df.sort_values("AIC").reset_index(drop=True)
    df["Rank_AIC"] = range(1, len(df) + 1)

    return df
