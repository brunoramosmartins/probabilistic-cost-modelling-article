"""Synthetic cost data generator with known ground truth.

Generates configurable samples for each cost component (salary, overtime,
severance, hiring) with controllable distribution, parameters, sample size,
noise level, and outlier contamination.
"""

import numpy as np
from numpy.typing import NDArray

from src.distributions import (
    Distribution,
    GammaDist,
    LogNormalDist,
    NormalDist,
    ParetoDist,
    WeibullDist,
)


def generate_salary_data(
    n: int,
    distribution: Distribution | None = None,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate synthetic salary data from a specified distribution.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    distribution : Distribution, optional
        Distribution to sample from. Defaults to LogNormal(mu=9.1, sigma=0.4),
        which produces salaries with median ~R$9,000.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    NDArray[np.float64]
        Array of n salary values.

    Examples
    --------
    >>> data = generate_salary_data(1000, seed=42)
    >>> data.shape
    (1000,)
    >>> 8000 < data.mean() < 12000
    True
    """
    if distribution is None:
        distribution = LogNormalDist(mu=9.1, sigma=0.4)
    return distribution.sample(n, seed=seed)


def generate_overtime_data(
    n: int,
    distribution: Distribution | None = None,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate synthetic overtime cost data.

    Parameters
    ----------
    n : int
        Number of samples.
    distribution : Distribution, optional
        Defaults to Gamma(alpha=4, beta=1/30), mean=R$120/hour.
    seed : int, optional
        Random seed.

    Returns
    -------
    NDArray[np.float64]
        Array of overtime cost values.
    """
    if distribution is None:
        distribution = GammaDist(alpha=4.0, beta=1.0 / 30.0)
    return distribution.sample(n, seed=seed)


def generate_severance_data(
    n: int,
    distribution: Distribution | None = None,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate synthetic severance cost data.

    Parameters
    ----------
    n : int
        Number of samples.
    distribution : Distribution, optional
        Defaults to Pareto(alpha=2.5, x_m=10000).
    seed : int, optional
        Random seed.

    Returns
    -------
    NDArray[np.float64]
        Array of severance cost values.
    """
    if distribution is None:
        distribution = ParetoDist(alpha=2.5, x_m=10000.0)
    return distribution.sample(n, seed=seed)


def generate_hiring_data(
    n: int,
    distribution: Distribution | None = None,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate synthetic hiring cost data.

    Parameters
    ----------
    n : int
        Number of samples.
    distribution : Distribution, optional
        Defaults to LogNormal(mu=9.5, sigma=0.7).
    seed : int, optional
        Random seed.

    Returns
    -------
    NDArray[np.float64]
        Array of hiring cost values.
    """
    if distribution is None:
        distribution = LogNormalDist(mu=9.5, sigma=0.7)
    return distribution.sample(n, seed=seed)


def generate_mixed_data(
    n: int,
    components: list[Distribution],
    weights: list[float],
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate data from a mixture of distributions.

    Useful for simulating multimodal salary distributions (e.g., junior vs
    senior clusters).

    Parameters
    ----------
    n : int
        Total number of samples.
    components : list[Distribution]
        List of K distribution components.
    weights : list[float]
        Mixing weights (must sum to 1.0). weights[k] is the probability
        that a sample comes from components[k].
    seed : int, optional
        Random seed.

    Returns
    -------
    NDArray[np.float64]
        Array of n samples drawn from the mixture.

    Examples
    --------
    >>> junior = NormalDist(mu=8000, sigma=1500)
    >>> senior = NormalDist(mu=18000, sigma=2500)
    >>> data = generate_mixed_data(1000, [junior, senior], [0.6, 0.4], seed=42)
    >>> data.shape
    (1000,)
    """
    if len(components) != len(weights):
        raise ValueError("components and weights must have the same length")

    weights_arr = np.array(weights, dtype=np.float64)
    if not np.isclose(weights_arr.sum(), 1.0):
        raise ValueError(f"weights must sum to 1.0, got {weights_arr.sum():.4f}")

    rng = np.random.default_rng(seed)

    # Determine how many samples from each component
    assignments = rng.choice(len(components), size=n, p=weights_arr)

    # Generate samples from each component
    samples = np.empty(n, dtype=np.float64)
    for k, component in enumerate(components):
        mask = assignments == k
        n_k = mask.sum()
        if n_k > 0:
            # Use a derived seed for reproducibility within each component
            component_seed = None if seed is None else seed + k + 1
            samples[mask] = component.sample(int(n_k), seed=component_seed)

    return samples


def inject_outliers(
    data: NDArray[np.float64],
    fraction: float = 0.05,
    multiplier: float = 3.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Inject synthetic outliers into cost data.

    Randomly selects a fraction of observations and multiplies them by
    a constant, simulating unexpected extreme costs.

    Parameters
    ----------
    data : NDArray[np.float64]
        Original data array.
    fraction : float
        Proportion of observations to contaminate (0 < fraction < 1).
    multiplier : float
        Factor to multiply selected observations by.
    seed : int, optional
        Random seed.

    Returns
    -------
    NDArray[np.float64]
        Data with outliers injected (copy, original unchanged).

    Examples
    --------
    >>> data = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    >>> contaminated = inject_outliers(data, fraction=0.4, multiplier=5.0, seed=0)
    >>> (contaminated > data).sum() > 0
    True
    """
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1 (exclusive)")
    if multiplier <= 0:
        raise ValueError("multiplier must be positive")

    rng = np.random.default_rng(seed)
    result = data.copy()
    n = len(data)
    n_outliers = max(1, int(n * fraction))

    outlier_indices = rng.choice(n, size=n_outliers, replace=False)
    result[outlier_indices] *= multiplier

    return result


def generate_team_data(
    team_size: int = 50,
    seed: int | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Generate a complete synthetic dataset for a team.

    Produces salary, overtime, severance, and hiring cost data using
    the default distributions defined in the cost model design document.

    Parameters
    ----------
    team_size : int
        Number of employees (used for salary and overtime).
    seed : int, optional
        Base random seed. Each component uses seed+offset for independence.

    Returns
    -------
    dict[str, NDArray[np.float64]]
        Dictionary with keys: 'salary', 'overtime', 'severance', 'hiring'.
    """
    base_seed = seed if seed is not None else None

    def offset_seed(offset: int) -> int | None:
        return None if base_seed is None else base_seed + offset

    return {
        "salary": generate_salary_data(team_size, seed=offset_seed(0)),
        "overtime": generate_overtime_data(team_size, seed=offset_seed(100)),
        "severance": generate_severance_data(10, seed=offset_seed(200)),
        "hiring": generate_hiring_data(5, seed=offset_seed(300)),
    }
