"""Distribution catalogue for probabilistic cost modelling.

Provides a unified interface for the five candidate distribution families:
Normal, LogNormal, Gamma, Pareto, and Weibull. Each distribution exposes
PDF, CDF, sampling, and analytical moments through a common API.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy import stats


class Distribution(Protocol):
    """Protocol defining the interface for all distribution wrappers."""

    @property
    def name(self) -> str:
        """Human-readable distribution name."""
        ...

    @property
    def n_params(self) -> int:
        """Number of parameters."""
        ...

    def pdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the probability density function."""
        ...

    def cdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the cumulative distribution function."""
        ...

    def sample(self, n: int, seed: int | None = None) -> NDArray[np.float64]:
        """Draw n random samples."""
        ...

    def mean(self) -> float:
        """Analytical mean."""
        ...

    def var(self) -> float:
        """Analytical variance."""
        ...

    def skewness(self) -> float:
        """Analytical skewness."""
        ...

    def kurtosis_excess(self) -> float:
        """Analytical excess kurtosis (kurtosis - 3)."""
        ...


@dataclass(frozen=True)
class NormalDist:
    """Normal (Gaussian) distribution.

    Parameters
    ----------
    mu : float
        Mean (location parameter).
    sigma : float
        Standard deviation (scale parameter), must be > 0.

    Notes
    -----
    PDF: f(x) = (1 / (sigma * sqrt(2*pi))) * exp(-(x - mu)^2 / (2*sigma^2))
    Support: (-inf, inf)
    """

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

    @property
    def name(self) -> str:
        return f"Normal(mu={self.mu:.2f}, sigma={self.sigma:.2f})"

    @property
    def n_params(self) -> int:
        return 2

    def pdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return stats.norm.pdf(x, loc=self.mu, scale=self.sigma)

    def cdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return stats.norm.cdf(x, loc=self.mu, scale=self.sigma)

    def sample(self, n: int, seed: int | None = None) -> NDArray[np.float64]:
        rng = np.random.default_rng(seed)
        return rng.normal(self.mu, self.sigma, size=n)

    def mean(self) -> float:
        return self.mu

    def var(self) -> float:
        return self.sigma**2

    def skewness(self) -> float:
        return 0.0

    def kurtosis_excess(self) -> float:
        return 0.0


@dataclass(frozen=True)
class LogNormalDist:
    """LogNormal distribution.

    Parameters
    ----------
    mu : float
        Mean of the underlying Normal (log-scale location).
    sigma : float
        Std dev of the underlying Normal (log-scale), must be > 0.

    Notes
    -----
    If Y ~ N(mu, sigma^2), then X = exp(Y) ~ LogNormal(mu, sigma^2).
    PDF: f(x) = (1 / (x * sigma * sqrt(2*pi))) * exp(-(ln(x) - mu)^2 / (2*sigma^2))
    Support: (0, inf)
    """

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

    @property
    def name(self) -> str:
        return f"LogNormal(mu={self.mu:.2f}, sigma={self.sigma:.2f})"

    @property
    def n_params(self) -> int:
        return 2

    def pdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return stats.lognorm.pdf(x, s=self.sigma, scale=np.exp(self.mu))

    def cdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return stats.lognorm.cdf(x, s=self.sigma, scale=np.exp(self.mu))

    def sample(self, n: int, seed: int | None = None) -> NDArray[np.float64]:
        rng = np.random.default_rng(seed)
        return rng.lognormal(self.mu, self.sigma, size=n)

    def mean(self) -> float:
        return float(np.exp(self.mu + self.sigma**2 / 2))

    def var(self) -> float:
        return float(
            np.exp(2 * self.mu + self.sigma**2) * (np.exp(self.sigma**2) - 1)
        )

    def skewness(self) -> float:
        es2 = np.exp(self.sigma**2)
        return float((es2 + 2) * np.sqrt(es2 - 1))

    def kurtosis_excess(self) -> float:
        es2 = np.exp(self.sigma**2)
        return float(es2**4 + 2 * es2**3 + 3 * es2**2 - 6)


@dataclass(frozen=True)
class GammaDist:
    """Gamma distribution (shape-rate parametrization).

    Parameters
    ----------
    alpha : float
        Shape parameter, must be > 0.
    beta : float
        Rate parameter, must be > 0. Mean = alpha/beta.

    Notes
    -----
    PDF: f(x) = (beta^alpha / Gamma(alpha)) * x^(alpha-1) * exp(-beta*x)
    Support: (0, inf)
    """

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.beta <= 0:
            raise ValueError("beta must be positive")

    @property
    def name(self) -> str:
        return f"Gamma(alpha={self.alpha:.2f}, beta={self.beta:.4f})"

    @property
    def n_params(self) -> int:
        return 2

    def pdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        # scipy uses shape-scale: shape=alpha, scale=1/beta
        return stats.gamma.pdf(x, a=self.alpha, scale=1.0 / self.beta)

    def cdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return stats.gamma.cdf(x, a=self.alpha, scale=1.0 / self.beta)

    def sample(self, n: int, seed: int | None = None) -> NDArray[np.float64]:
        rng = np.random.default_rng(seed)
        return rng.gamma(self.alpha, 1.0 / self.beta, size=n)

    def mean(self) -> float:
        return self.alpha / self.beta

    def var(self) -> float:
        return self.alpha / self.beta**2

    def skewness(self) -> float:
        return 2.0 / np.sqrt(self.alpha)

    def kurtosis_excess(self) -> float:
        return 6.0 / self.alpha


@dataclass(frozen=True)
class ParetoDist:
    """Pareto (Type I) distribution.

    Parameters
    ----------
    alpha : float
        Shape (tail index), must be > 0. Smaller alpha = heavier tail.
    x_m : float
        Scale (minimum value), must be > 0.

    Notes
    -----
    PDF: f(x) = (alpha * x_m^alpha) / x^(alpha+1),  x >= x_m
    CDF: F(x) = 1 - (x_m / x)^alpha
    Support: [x_m, inf)
    """

    alpha: float
    x_m: float

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.x_m <= 0:
            raise ValueError("x_m must be positive")

    @property
    def name(self) -> str:
        return f"Pareto(alpha={self.alpha:.2f}, x_m={self.x_m:.0f})"

    @property
    def n_params(self) -> int:
        return 2

    def pdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        # scipy.stats.pareto: pdf(x, b) = b / x^(b+1) for x >= 1
        # We use loc=0, scale=x_m to shift: support starts at x_m
        return stats.pareto.pdf(x, b=self.alpha, scale=self.x_m)

    def cdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return stats.pareto.cdf(x, b=self.alpha, scale=self.x_m)

    def sample(self, n: int, seed: int | None = None) -> NDArray[np.float64]:
        rng = np.random.default_rng(seed)
        # Inverse transform: X = x_m / U^(1/alpha), U ~ Uniform(0,1)
        u = rng.uniform(0, 1, size=n)
        return self.x_m / u ** (1.0 / self.alpha)

    def mean(self) -> float:
        if self.alpha <= 1:
            return float("inf")
        return self.alpha * self.x_m / (self.alpha - 1)

    def var(self) -> float:
        if self.alpha <= 2:
            return float("inf")
        return (
            self.alpha * self.x_m**2
            / ((self.alpha - 1) ** 2 * (self.alpha - 2))
        )

    def skewness(self) -> float:
        if self.alpha <= 3:
            return float("inf")
        return (
            2 * (1 + self.alpha) / (self.alpha - 3) * np.sqrt((self.alpha - 2) / self.alpha)
        )

    def kurtosis_excess(self) -> float:
        if self.alpha <= 4:
            return float("inf")
        a = self.alpha
        return 6 * (a**3 + a**2 - 6 * a - 2) / (a * (a - 3) * (a - 4))


@dataclass(frozen=True)
class WeibullDist:
    """Weibull distribution.

    Parameters
    ----------
    k : float
        Shape parameter, must be > 0.
    lam : float
        Scale parameter (lambda), must be > 0.

    Notes
    -----
    PDF: f(x) = (k/lam) * (x/lam)^(k-1) * exp(-(x/lam)^k)
    CDF: F(x) = 1 - exp(-(x/lam)^k)
    Support: [0, inf)
    """

    k: float
    lam: float

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.lam <= 0:
            raise ValueError("lam must be positive")

    @property
    def name(self) -> str:
        return f"Weibull(k={self.k:.2f}, lam={self.lam:.2f})"

    @property
    def n_params(self) -> int:
        return 2

    def pdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        # scipy.stats.weibull_min: c=k, scale=lam
        return stats.weibull_min.pdf(x, c=self.k, scale=self.lam)

    def cdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return stats.weibull_min.cdf(x, c=self.k, scale=self.lam)

    def sample(self, n: int, seed: int | None = None) -> NDArray[np.float64]:
        rng = np.random.default_rng(seed)
        # Inverse transform: X = lam * (-ln(U))^(1/k)
        u = rng.uniform(0, 1, size=n)
        return self.lam * (-np.log(u)) ** (1.0 / self.k)

    def mean(self) -> float:
        from scipy.special import gamma as gamma_fn

        return float(self.lam * gamma_fn(1 + 1 / self.k))

    def var(self) -> float:
        from scipy.special import gamma as gamma_fn

        g1 = gamma_fn(1 + 1 / self.k)
        g2 = gamma_fn(1 + 2 / self.k)
        return float(self.lam**2 * (g2 - g1**2))

    def skewness(self) -> float:
        from scipy.special import gamma as gamma_fn

        g1 = gamma_fn(1 + 1 / self.k)
        g2 = gamma_fn(1 + 2 / self.k)
        g3 = gamma_fn(1 + 3 / self.k)
        mu = self.lam * g1
        sigma2 = self.lam**2 * (g2 - g1**2)
        sigma = np.sqrt(sigma2)
        return float(
            (self.lam**3 * g3 - 3 * mu * sigma2 - mu**3) / sigma**3
        )

    def kurtosis_excess(self) -> float:
        from scipy.special import gamma as gamma_fn

        g1 = gamma_fn(1 + 1 / self.k)
        g2 = gamma_fn(1 + 2 / self.k)
        g3 = gamma_fn(1 + 3 / self.k)
        g4 = gamma_fn(1 + 4 / self.k)
        mu = self.lam * g1
        sigma2 = self.lam**2 * (g2 - g1**2)
        sigma = np.sqrt(sigma2)
        m4 = (
            self.lam**4 * g4
            - 4 * mu * self.lam**3 * g3
            + 6 * mu**2 * self.lam**2 * g2
            - 3 * mu**4
        )
        return float(m4 / sigma**4 - 3)


# --- Catalogue ---

DISTRIBUTION_FAMILIES: dict[str, type] = {
    "normal": NormalDist,
    "lognormal": LogNormalDist,
    "gamma": GammaDist,
    "pareto": ParetoDist,
    "weibull": WeibullDist,
}


def get_distribution(name: str, **params: float) -> Distribution:
    """Factory function to create a distribution by name.

    Parameters
    ----------
    name : str
        Distribution family name (case-insensitive).
        One of: 'normal', 'lognormal', 'gamma', 'pareto', 'weibull'.
    **params : float
        Distribution parameters (varies by family).

    Returns
    -------
    Distribution
        An instance implementing the Distribution protocol.

    Examples
    --------
    >>> dist = get_distribution("lognormal", mu=9.1, sigma=0.4)
    >>> dist.mean()
    9725.53...
    """
    key = name.lower().replace("-", "").replace("_", "")
    if key not in DISTRIBUTION_FAMILIES:
        valid = ", ".join(DISTRIBUTION_FAMILIES.keys())
        raise ValueError(f"Unknown distribution '{name}'. Valid: {valid}")
    return DISTRIBUTION_FAMILIES[key](**params)
