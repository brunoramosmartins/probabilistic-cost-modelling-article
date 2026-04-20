"""MLE fitting pipeline for probabilistic cost modelling.

Provides maximum likelihood estimation for all candidate distribution families.
Uses analytical solutions where available (Normal, LogNormal, Pareto) and
numerical optimization (scipy.optimize) for Gamma and Weibull. Reports
Fisher-information-based standard errors and confidence intervals.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, special, stats


@dataclass
class FitResult:
    """Result of fitting a distribution via MLE.

    Attributes
    ----------
    distribution : str
        Name of the fitted distribution family.
    params : dict[str, float]
        MLE parameter estimates.
    loglik : float
        Maximized log-likelihood value.
    n_params : int
        Number of estimated parameters.
    n_obs : int
        Number of observations.
    se : dict[str, float]
        Standard errors from observed Fisher information.
    ci_lower : dict[str, float]
        Lower bounds of 95% confidence intervals.
    ci_upper : dict[str, float]
        Upper bounds of 95% confidence intervals.
    converged : bool
        Whether the optimization converged (always True for closed-form).
    """

    distribution: str
    params: dict[str, float]
    loglik: float
    n_params: int
    n_obs: int
    se: dict[str, float]
    ci_lower: dict[str, float]
    ci_upper: dict[str, float]
    converged: bool = True

    @property
    def aic(self) -> float:
        """Akaike Information Criterion: 2k - 2*loglik."""
        return 2 * self.n_params - 2 * self.loglik

    @property
    def bic(self) -> float:
        """Bayesian Information Criterion: k*log(n) - 2*loglik."""
        return self.n_params * np.log(self.n_obs) - 2 * self.loglik


def fit_normal(data: NDArray[np.float64]) -> FitResult:
    """Fit Normal distribution via MLE (closed-form).

    Parameters
    ----------
    data : NDArray[np.float64]
        Observed data.

    Returns
    -------
    FitResult
        MLE estimates with standard errors and confidence intervals.

    Notes
    -----
    MLE: mu_hat = mean(x), sigma2_hat = (1/n) * sum((x - mean)^2)
    Fisher information: I(mu) = n/sigma^2, I(sigma^2) = n/(2*sigma^4)
    """
    n = len(data)
    mu_hat = float(data.mean())
    sigma2_hat = float(data.var(ddof=0))  # MLE (biased)
    sigma_hat = np.sqrt(sigma2_hat)

    # Log-likelihood
    loglik = float(np.sum(stats.norm.logpdf(data, loc=mu_hat, scale=sigma_hat)))

    # Standard errors from Fisher information
    se_mu = sigma_hat / np.sqrt(n)
    se_sigma = sigma_hat / np.sqrt(2 * n)  # SE for sigma (not sigma^2)

    z = 1.96  # 95% CI

    return FitResult(
        distribution="Normal",
        params={"mu": mu_hat, "sigma": sigma_hat},
        loglik=loglik,
        n_params=2,
        n_obs=n,
        se={"mu": se_mu, "sigma": se_sigma},
        ci_lower={"mu": mu_hat - z * se_mu, "sigma": sigma_hat - z * se_sigma},
        ci_upper={"mu": mu_hat + z * se_mu, "sigma": sigma_hat + z * se_sigma},
    )


def fit_lognormal(data: NDArray[np.float64]) -> FitResult:
    """Fit LogNormal distribution via MLE (closed-form).

    Parameters
    ----------
    data : NDArray[np.float64]
        Observed data (must be all positive).

    Returns
    -------
    FitResult
        MLE estimates for mu and sigma (log-scale parameters).

    Notes
    -----
    If X ~ LogNormal(mu, sigma^2), then log(X) ~ Normal(mu, sigma^2).
    MLE: mu_hat = mean(log(x)), sigma_hat = std(log(x), ddof=0)
    """
    if np.any(data <= 0):
        raise ValueError("LogNormal requires all positive data")

    n = len(data)
    log_data = np.log(data)

    mu_hat = float(log_data.mean())
    sigma2_hat = float(log_data.var(ddof=0))
    sigma_hat = np.sqrt(sigma2_hat)

    # Log-likelihood (computed on original scale)
    loglik = float(np.sum(stats.lognorm.logpdf(data, s=sigma_hat, scale=np.exp(mu_hat))))

    # Standard errors (same as Normal on log-scale)
    se_mu = sigma_hat / np.sqrt(n)
    se_sigma = sigma_hat / np.sqrt(2 * n)

    z = 1.96

    return FitResult(
        distribution="LogNormal",
        params={"mu": mu_hat, "sigma": sigma_hat},
        loglik=loglik,
        n_params=2,
        n_obs=n,
        se={"mu": se_mu, "sigma": se_sigma},
        ci_lower={"mu": mu_hat - z * se_mu, "sigma": sigma_hat - z * se_sigma},
        ci_upper={"mu": mu_hat + z * se_mu, "sigma": sigma_hat + z * se_sigma},
    )


def fit_gamma(data: NDArray[np.float64]) -> FitResult:
    """Fit Gamma distribution via MLE (numerical for alpha).

    Parameters
    ----------
    data : NDArray[np.float64]
        Observed data (must be all positive).

    Returns
    -------
    FitResult
        MLE estimates for alpha (shape) and beta (rate).

    Notes
    -----
    Uses scipy.optimize to solve the score equation for alpha.
    beta_hat = alpha_hat / mean(x) given alpha.
    Fisher information computed via trigamma function.
    """
    if np.any(data <= 0):
        raise ValueError("Gamma requires all positive data")

    n = len(data)
    x_bar = float(data.mean())
    log_x_bar = float(np.log(data).mean())

    # Score equation for alpha: log(alpha) - psi(alpha) = log(x_bar) - mean(log(x))
    s = np.log(x_bar) - log_x_bar  # Always positive for non-degenerate data

    # Method of moments initial estimate
    alpha_init = 0.5 / s  # Approximate starting point

    # Solve: log(alpha) - psi(alpha) - s = 0
    def score_alpha(log_alpha: float) -> float:
        alpha = np.exp(log_alpha)
        return float(np.log(alpha) - special.digamma(alpha) - s)

    result = optimize.root_scalar(
        score_alpha,
        x0=np.log(alpha_init),
        method="brentq",
        bracket=[np.log(1e-6), np.log(1e6)],
    )

    alpha_hat = np.exp(result.root)
    beta_hat = alpha_hat / x_bar

    # Log-likelihood
    loglik = float(np.sum(stats.gamma.logpdf(data, a=alpha_hat, scale=1.0 / beta_hat)))

    # Fisher information matrix (per observation)
    # I_11 = psi'(alpha), I_12 = -1/beta, I_22 = alpha/beta^2
    trigamma_alpha = float(special.polygamma(1, alpha_hat))
    fisher_matrix = n * np.array([
        [trigamma_alpha, -1.0 / beta_hat],
        [-1.0 / beta_hat, alpha_hat / beta_hat**2],
    ])

    # Covariance = inverse of Fisher information
    cov_matrix = np.linalg.inv(fisher_matrix)
    se_alpha = float(np.sqrt(cov_matrix[0, 0]))
    se_beta = float(np.sqrt(cov_matrix[1, 1]))

    z = 1.96

    return FitResult(
        distribution="Gamma",
        params={"alpha": float(alpha_hat), "beta": float(beta_hat)},
        loglik=loglik,
        n_params=2,
        n_obs=n,
        se={"alpha": se_alpha, "beta": se_beta},
        ci_lower={
            "alpha": float(alpha_hat) - z * se_alpha,
            "beta": float(beta_hat) - z * se_beta,
        },
        ci_upper={
            "alpha": float(alpha_hat) + z * se_alpha,
            "beta": float(beta_hat) + z * se_beta,
        },
        converged=result.converged,
    )


def fit_pareto(data: NDArray[np.float64], x_m: float | None = None) -> FitResult:
    """Fit Pareto distribution via MLE (closed-form with known x_m).

    Parameters
    ----------
    data : NDArray[np.float64]
        Observed data.
    x_m : float, optional
        Known minimum value. If None, estimated as min(data).

    Returns
    -------
    FitResult
        MLE estimate for alpha (tail index).

    Notes
    -----
    MLE: alpha_hat = n / sum(log(x_i / x_m))
    Fisher information: I(alpha) = 1/alpha^2 per observation.
    """
    if x_m is None:
        x_m = float(data.min())

    if np.any(data < x_m):
        raise ValueError(f"All data must be >= x_m={x_m}")

    n = len(data)

    # MLE for alpha
    alpha_hat = float(n / np.sum(np.log(data / x_m)))

    # Log-likelihood
    loglik = float(np.sum(stats.pareto.logpdf(data, b=alpha_hat, scale=x_m)))

    # Standard error from Fisher information: I_1(alpha) = 1/alpha^2
    se_alpha = alpha_hat / np.sqrt(n)

    z = 1.96

    return FitResult(
        distribution="Pareto",
        params={"alpha": alpha_hat, "x_m": x_m},
        loglik=loglik,
        n_params=1,  # x_m is fixed/known
        n_obs=n,
        se={"alpha": se_alpha},
        ci_lower={"alpha": alpha_hat - z * se_alpha},
        ci_upper={"alpha": alpha_hat + z * se_alpha},
    )


def fit_weibull(data: NDArray[np.float64]) -> FitResult:
    """Fit Weibull distribution via MLE (numerical optimization).

    Parameters
    ----------
    data : NDArray[np.float64]
        Observed data (must be all positive).

    Returns
    -------
    FitResult
        MLE estimates for k (shape) and lam (scale).

    Notes
    -----
    Uses scipy.optimize.minimize on the negative log-likelihood.
    Standard errors from the observed Fisher information (Hessian inverse).
    """
    if np.any(data <= 0):
        raise ValueError("Weibull requires all positive data")

    n = len(data)

    # Negative log-likelihood
    def neg_loglik(params: NDArray[np.float64]) -> float:
        k, lam = params
        if k <= 0 or lam <= 0:
            return 1e20
        return float(-np.sum(stats.weibull_min.logpdf(data, c=k, scale=lam)))

    # Initial estimates via method of moments
    mean_x = data.mean()
    var_x = data.var()
    # Approximate k from CV: CV = sqrt(Gamma(1+2/k)/Gamma(1+1/k)^2 - 1)
    cv = np.sqrt(var_x) / mean_x
    k_init = max(1.0 / cv, 0.5)  # Rough approximation
    lam_init = mean_x / special.gamma(1 + 1 / k_init)

    result = optimize.minimize(
        neg_loglik,
        x0=np.array([k_init, lam_init]),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 10000},
    )

    k_hat, lam_hat = result.x

    # Log-likelihood
    loglik = float(-result.fun)

    # Observed Fisher information via numerical Hessian
    eps = 1e-5
    hessian = np.zeros((2, 2))
    f0 = result.fun
    for i in range(2):
        for j in range(2):
            params_pp = result.x.copy()
            params_pm = result.x.copy()
            params_mp = result.x.copy()
            params_mm = result.x.copy()
            params_pp[i] += eps
            params_pp[j] += eps
            params_pm[i] += eps
            params_pm[j] -= eps
            params_mp[i] -= eps
            params_mp[j] += eps
            params_mm[i] -= eps
            params_mm[j] -= eps
            hessian[i, j] = (
                neg_loglik(params_pp) - neg_loglik(params_pm)
                - neg_loglik(params_mp) + neg_loglik(params_mm)
            ) / (4 * eps**2)

    # Covariance = inverse of observed Fisher information (= Hessian of neg loglik)
    try:
        cov_matrix = np.linalg.inv(hessian)
        se_k = float(np.sqrt(max(cov_matrix[0, 0], 0)))
        se_lam = float(np.sqrt(max(cov_matrix[1, 1], 0)))
    except np.linalg.LinAlgError:
        se_k = float("nan")
        se_lam = float("nan")

    z = 1.96

    return FitResult(
        distribution="Weibull",
        params={"k": float(k_hat), "lam": float(lam_hat)},
        loglik=loglik,
        n_params=2,
        n_obs=n,
        se={"k": se_k, "lam": se_lam},
        ci_lower={
            "k": float(k_hat) - z * se_k,
            "lam": float(lam_hat) - z * se_lam,
        },
        ci_upper={
            "k": float(k_hat) + z * se_k,
            "lam": float(lam_hat) + z * se_lam,
        },
        converged=result.success,
    )


# --- Dispatch ---

_FITTERS = {
    "normal": fit_normal,
    "lognormal": fit_lognormal,
    "gamma": fit_gamma,
    "pareto": fit_pareto,
    "weibull": fit_weibull,
}


def fit_mle(data: NDArray[np.float64], distribution: str, **kwargs) -> FitResult:
    """Fit a distribution to data via MLE.

    Parameters
    ----------
    data : NDArray[np.float64]
        Observed data.
    distribution : str
        Distribution name: 'normal', 'lognormal', 'gamma', 'pareto', 'weibull'.
    **kwargs
        Additional arguments passed to the specific fitter (e.g., x_m for Pareto).

    Returns
    -------
    FitResult
        MLE estimates with standard errors and confidence intervals.
    """
    key = distribution.lower().replace("-", "").replace("_", "")
    if key not in _FITTERS:
        valid = ", ".join(_FITTERS.keys())
        raise ValueError(f"Unknown distribution '{distribution}'. Valid: {valid}")
    return _FITTERS[key](data, **kwargs)


def fit_all(
    data: NDArray[np.float64],
    candidates: list[str] | None = None,
) -> list[FitResult]:
    """Fit all candidate distributions and return results sorted by log-likelihood.

    Parameters
    ----------
    data : NDArray[np.float64]
        Observed data (must be all positive for most distributions).
    candidates : list[str], optional
        List of distribution names to fit. Defaults to all five.

    Returns
    -------
    list[FitResult]
        Results sorted by log-likelihood (best first).
    """
    if candidates is None:
        candidates = ["normal", "lognormal", "gamma", "pareto", "weibull"]

    results = []
    for name in candidates:
        try:
            result = fit_mle(data, name)
            results.append(result)
        except (ValueError, RuntimeError) as e:
            # Skip distributions that can't fit this data
            print(f"Warning: could not fit {name}: {e}")
            continue

    results.sort(key=lambda r: r.loglik, reverse=True)
    return results
