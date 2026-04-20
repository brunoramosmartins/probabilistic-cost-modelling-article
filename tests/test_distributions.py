"""Tests for the distribution catalogue (src/distributions.py)."""

import numpy as np
import pytest

from src.distributions import (
    GammaDist,
    LogNormalDist,
    NormalDist,
    ParetoDist,
    WeibullDist,
    get_distribution,
)


class TestNormalDist:
    """Tests for NormalDist."""

    def test_mean(self):
        dist = NormalDist(mu=5.0, sigma=2.0)
        assert dist.mean() == 5.0

    def test_var(self):
        dist = NormalDist(mu=5.0, sigma=2.0)
        assert dist.var() == 4.0

    def test_skewness(self):
        dist = NormalDist(mu=0.0, sigma=1.0)
        assert dist.skewness() == 0.0

    def test_kurtosis_excess(self):
        dist = NormalDist(mu=0.0, sigma=1.0)
        assert dist.kurtosis_excess() == 0.0

    def test_pdf_integrates_to_one(self):
        dist = NormalDist(mu=3.0, sigma=1.5)
        x = np.linspace(-10, 16, 10000)
        dx = x[1] - x[0]
        integral = np.sum(dist.pdf(x)) * dx
        assert abs(integral - 1.0) < 0.001

    def test_cdf_bounds(self):
        dist = NormalDist(mu=0.0, sigma=1.0)
        assert abs(dist.cdf(np.array([0.0]))[0] - 0.5) < 1e-10

    def test_sample_shape(self):
        dist = NormalDist(mu=0.0, sigma=1.0)
        samples = dist.sample(500, seed=42)
        assert samples.shape == (500,)

    def test_sample_moments_converge(self):
        dist = NormalDist(mu=10.0, sigma=3.0)
        samples = dist.sample(100_000, seed=123)
        assert abs(samples.mean() - 10.0) < 0.05
        assert abs(samples.var() - 9.0) < 0.1

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError):
            NormalDist(mu=0.0, sigma=0.0)
        with pytest.raises(ValueError):
            NormalDist(mu=0.0, sigma=-1.0)

    def test_n_params(self):
        dist = NormalDist(mu=0.0, sigma=1.0)
        assert dist.n_params == 2


class TestLogNormalDist:
    """Tests for LogNormalDist."""

    def test_mean(self):
        dist = LogNormalDist(mu=0.0, sigma=1.0)
        expected = np.exp(0.5)
        assert abs(dist.mean() - expected) < 1e-10

    def test_var(self):
        dist = LogNormalDist(mu=0.0, sigma=1.0)
        expected = np.exp(1.0) * (np.exp(1.0) - 1)
        assert abs(dist.var() - expected) < 1e-10

    def test_skewness_positive(self):
        dist = LogNormalDist(mu=9.1, sigma=0.4)
        assert dist.skewness() > 0

    def test_kurtosis_excess_positive(self):
        dist = LogNormalDist(mu=9.1, sigma=0.4)
        assert dist.kurtosis_excess() > 0

    def test_sample_all_positive(self):
        dist = LogNormalDist(mu=9.1, sigma=0.4)
        samples = dist.sample(10_000, seed=42)
        assert np.all(samples > 0)

    def test_sample_moments_converge(self):
        dist = LogNormalDist(mu=2.0, sigma=0.5)
        samples = dist.sample(200_000, seed=99)
        assert abs(samples.mean() - dist.mean()) / dist.mean() < 0.01

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError):
            LogNormalDist(mu=0.0, sigma=0.0)


class TestGammaDist:
    """Tests for GammaDist."""

    def test_mean(self):
        dist = GammaDist(alpha=4.0, beta=2.0)
        assert abs(dist.mean() - 2.0) < 1e-10

    def test_var(self):
        dist = GammaDist(alpha=4.0, beta=2.0)
        assert abs(dist.var() - 1.0) < 1e-10

    def test_skewness(self):
        dist = GammaDist(alpha=4.0, beta=2.0)
        assert abs(dist.skewness() - 1.0) < 1e-10

    def test_kurtosis_excess(self):
        dist = GammaDist(alpha=4.0, beta=2.0)
        assert abs(dist.kurtosis_excess() - 1.5) < 1e-10

    def test_exponential_special_case(self):
        """Gamma(1, beta) should equal Exponential(beta)."""
        dist = GammaDist(alpha=1.0, beta=0.5)
        assert abs(dist.mean() - 2.0) < 1e-10
        assert abs(dist.var() - 4.0) < 1e-10

    def test_sample_all_positive(self):
        dist = GammaDist(alpha=2.0, beta=1.0)
        samples = dist.sample(10_000, seed=42)
        assert np.all(samples > 0)

    def test_sample_moments_converge(self):
        dist = GammaDist(alpha=5.0, beta=0.1)
        samples = dist.sample(200_000, seed=77)
        assert abs(samples.mean() - dist.mean()) / dist.mean() < 0.01

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            GammaDist(alpha=0.0, beta=1.0)
        with pytest.raises(ValueError):
            GammaDist(alpha=1.0, beta=-1.0)


class TestParetoDist:
    """Tests for ParetoDist."""

    def test_mean(self):
        dist = ParetoDist(alpha=3.0, x_m=1000.0)
        assert abs(dist.mean() - 1500.0) < 1e-10

    def test_mean_infinite(self):
        dist = ParetoDist(alpha=1.0, x_m=1000.0)
        assert dist.mean() == float("inf")

    def test_var(self):
        dist = ParetoDist(alpha=3.0, x_m=1000.0)
        expected = 3.0 * 1000.0**2 / (4.0 * 1.0)  # alpha*xm^2/((a-1)^2*(a-2))
        assert abs(dist.var() - expected) < 1e-6

    def test_var_infinite(self):
        dist = ParetoDist(alpha=2.0, x_m=1000.0)
        assert dist.var() == float("inf")

    def test_sample_all_above_xm(self):
        dist = ParetoDist(alpha=2.5, x_m=10000.0)
        samples = dist.sample(10_000, seed=42)
        assert np.all(samples >= 10000.0)

    def test_tail_probability(self):
        """P(X > x) = (x_m/x)^alpha for Pareto."""
        dist = ParetoDist(alpha=2.5, x_m=10000.0)
        x = np.array([20000.0])
        survival = 1 - dist.cdf(x)
        expected = (10000.0 / 20000.0) ** 2.5
        assert abs(survival[0] - expected) < 1e-6

    def test_sample_moments_converge(self):
        dist = ParetoDist(alpha=4.0, x_m=1000.0)
        samples = dist.sample(500_000, seed=55)
        # alpha=4 has finite mean and variance
        assert abs(samples.mean() - dist.mean()) / dist.mean() < 0.02

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            ParetoDist(alpha=0.0, x_m=1000.0)
        with pytest.raises(ValueError):
            ParetoDist(alpha=2.0, x_m=0.0)


class TestWeibullDist:
    """Tests for WeibullDist."""

    def test_exponential_special_case(self):
        """Weibull(k=1, lam) should equal Exponential(1/lam)."""
        dist = WeibullDist(k=1.0, lam=5.0)
        assert abs(dist.mean() - 5.0) < 1e-6

    def test_sample_all_positive(self):
        dist = WeibullDist(k=2.0, lam=100.0)
        samples = dist.sample(10_000, seed=42)
        assert np.all(samples >= 0)

    def test_cdf_closed_form(self):
        """CDF should match 1 - exp(-(x/lam)^k)."""
        dist = WeibullDist(k=2.0, lam=10.0)
        x = np.array([5.0, 10.0, 15.0])
        expected = 1 - np.exp(-(x / 10.0) ** 2)
        np.testing.assert_allclose(dist.cdf(x), expected, rtol=1e-10)

    def test_sample_moments_converge(self):
        dist = WeibullDist(k=3.0, lam=50.0)
        samples = dist.sample(200_000, seed=33)
        assert abs(samples.mean() - dist.mean()) / dist.mean() < 0.01

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            WeibullDist(k=0.0, lam=1.0)
        with pytest.raises(ValueError):
            WeibullDist(k=1.0, lam=-1.0)


class TestGetDistribution:
    """Tests for the factory function."""

    def test_normal(self):
        dist = get_distribution("normal", mu=0.0, sigma=1.0)
        assert dist.mean() == 0.0

    def test_lognormal(self):
        dist = get_distribution("lognormal", mu=9.1, sigma=0.4)
        assert dist.mean() > 0

    def test_case_insensitive(self):
        dist = get_distribution("LogNormal", mu=0.0, sigma=1.0)
        assert dist.mean() > 0

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown distribution"):
            get_distribution("student_t", df=5.0)
