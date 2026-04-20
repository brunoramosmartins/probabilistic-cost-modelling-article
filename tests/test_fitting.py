"""Tests for the MLE fitting pipeline (src/fitting.py)."""

import numpy as np
import pytest

from src.fitting import (
    FitResult,
    fit_all,
    fit_gamma,
    fit_lognormal,
    fit_mle,
    fit_normal,
    fit_pareto,
    fit_weibull,
)


class TestFitNormal:
    """Tests for Normal MLE."""

    def test_recovers_mean(self):
        rng = np.random.default_rng(42)
        data = rng.normal(100.0, 10.0, size=10_000)
        result = fit_normal(data)
        assert abs(result.params["mu"] - 100.0) < 0.5

    def test_recovers_sigma(self):
        rng = np.random.default_rng(42)
        data = rng.normal(50.0, 5.0, size=10_000)
        result = fit_normal(data)
        assert abs(result.params["sigma"] - 5.0) < 0.2

    def test_se_shrinks_with_n(self):
        rng = np.random.default_rng(42)
        data_small = rng.normal(0, 1, size=100)
        data_large = rng.normal(0, 1, size=10_000)
        se_small = fit_normal(data_small).se["mu"]
        se_large = fit_normal(data_large).se["mu"]
        assert se_large < se_small / 5  # ~10x smaller for 100x data

    def test_ci_covers_true_param(self):
        """95% CI should cover the true parameter most of the time."""
        covers = 0
        for seed in range(100):
            rng = np.random.default_rng(seed)
            data = rng.normal(10.0, 2.0, size=50)
            result = fit_normal(data)
            if result.ci_lower["mu"] <= 10.0 <= result.ci_upper["mu"]:
                covers += 1
        # Should cover ~95% of the time (allow some slack)
        assert covers >= 85

    def test_n_params(self):
        data = np.array([1.0, 2.0, 3.0])
        result = fit_normal(data)
        assert result.n_params == 2

    def test_n_obs(self):
        data = np.ones(50)
        result = fit_normal(data)
        assert result.n_obs == 50


class TestFitLogNormal:
    """Tests for LogNormal MLE."""

    def test_recovers_mu(self):
        rng = np.random.default_rng(42)
        data = rng.lognormal(9.0, 0.5, size=10_000)
        result = fit_lognormal(data)
        assert abs(result.params["mu"] - 9.0) < 0.05

    def test_recovers_sigma(self):
        rng = np.random.default_rng(42)
        data = rng.lognormal(9.0, 0.5, size=10_000)
        result = fit_lognormal(data)
        assert abs(result.params["sigma"] - 0.5) < 0.02

    def test_rejects_negative_data(self):
        data = np.array([-1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="positive"):
            fit_lognormal(data)

    def test_ci_covers_true_mu(self):
        covers = 0
        for seed in range(100):
            rng = np.random.default_rng(seed)
            data = rng.lognormal(9.1, 0.4, size=100)
            result = fit_lognormal(data)
            if result.ci_lower["mu"] <= 9.1 <= result.ci_upper["mu"]:
                covers += 1
        assert covers >= 85


class TestFitGamma:
    """Tests for Gamma MLE."""

    def test_recovers_alpha(self):
        rng = np.random.default_rng(42)
        # Gamma with shape=5, scale=20 (rate=0.05)
        data = rng.gamma(5.0, 20.0, size=10_000)
        result = fit_gamma(data)
        assert abs(result.params["alpha"] - 5.0) < 0.3

    def test_recovers_beta(self):
        rng = np.random.default_rng(42)
        data = rng.gamma(5.0, 20.0, size=10_000)  # scale=20, rate=1/20=0.05
        result = fit_gamma(data)
        assert abs(result.params["beta"] - 0.05) < 0.005

    def test_rejects_negative_data(self):
        data = np.array([-1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="positive"):
            fit_gamma(data)

    def test_converges(self):
        rng = np.random.default_rng(42)
        data = rng.gamma(3.0, 10.0, size=500)
        result = fit_gamma(data)
        assert result.converged

    def test_exponential_special_case(self):
        """Gamma with alpha=1 should be Exponential."""
        rng = np.random.default_rng(42)
        data = rng.exponential(50.0, size=5000)
        result = fit_gamma(data)
        assert abs(result.params["alpha"] - 1.0) < 0.1


class TestFitPareto:
    """Tests for Pareto MLE."""

    def test_recovers_alpha(self):
        rng = np.random.default_rng(42)
        # Pareto: X = x_m / U^(1/alpha)
        x_m = 1000.0
        alpha_true = 3.0
        u = rng.uniform(0, 1, size=10_000)
        data = x_m / u ** (1.0 / alpha_true)
        result = fit_pareto(data, x_m=x_m)
        assert abs(result.params["alpha"] - alpha_true) < 0.1

    def test_se_shrinks_with_n(self):
        rng = np.random.default_rng(42)
        x_m = 1000.0
        u = rng.uniform(0, 1, size=100)
        data_small = x_m / u ** (1.0 / 2.5)
        u = rng.uniform(0, 1, size=10_000)
        data_large = x_m / u ** (1.0 / 2.5)
        se_small = fit_pareto(data_small, x_m=x_m).se["alpha"]
        se_large = fit_pareto(data_large, x_m=x_m).se["alpha"]
        assert se_large < se_small / 5

    def test_auto_xm_from_data(self):
        data = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = fit_pareto(data)
        assert result.params["x_m"] == 5.0

    def test_rejects_data_below_xm(self):
        data = np.array([5.0, 6.0, 7.0])
        with pytest.raises(ValueError, match=">= x_m"):
            fit_pareto(data, x_m=6.0)


class TestFitWeibull:
    """Tests for Weibull MLE."""

    def test_recovers_k(self):
        rng = np.random.default_rng(42)
        # Weibull: X = lam * (-log(U))^(1/k)
        k_true = 2.5
        lam_true = 100.0
        u = rng.uniform(0, 1, size=5000)
        data = lam_true * (-np.log(u)) ** (1.0 / k_true)
        result = fit_weibull(data)
        assert abs(result.params["k"] - k_true) < 0.2

    def test_recovers_lam(self):
        rng = np.random.default_rng(42)
        k_true = 2.5
        lam_true = 100.0
        u = rng.uniform(0, 1, size=5000)
        data = lam_true * (-np.log(u)) ** (1.0 / k_true)
        result = fit_weibull(data)
        assert abs(result.params["lam"] - lam_true) < 5.0

    def test_rejects_negative_data(self):
        data = np.array([-1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="positive"):
            fit_weibull(data)

    def test_exponential_special_case(self):
        """Weibull with k=1 should recover Exponential."""
        rng = np.random.default_rng(42)
        data = rng.exponential(50.0, size=5000)
        result = fit_weibull(data)
        assert abs(result.params["k"] - 1.0) < 0.1


class TestFitMLE:
    """Tests for the dispatch function."""

    def test_normal(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = fit_mle(data, "normal")
        assert result.distribution == "Normal"

    def test_case_insensitive(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = fit_mle(data, "LogNormal")
        assert result.distribution == "LogNormal"

    def test_unknown_raises(self):
        data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Unknown"):
            fit_mle(data, "student_t")


class TestFitAll:
    """Tests for fit_all."""

    def test_returns_sorted_by_loglik(self):
        rng = np.random.default_rng(42)
        data = rng.lognormal(9.0, 0.5, size=500)
        results = fit_all(data)
        logliks = [r.loglik for r in results]
        assert logliks == sorted(logliks, reverse=True)

    def test_lognormal_wins_for_lognormal_data(self):
        rng = np.random.default_rng(42)
        data = rng.lognormal(9.0, 0.5, size=1000)
        results = fit_all(data)
        # LogNormal should be among top 2 (it's the true model)
        top_names = [r.distribution for r in results[:2]]
        assert "LogNormal" in top_names

    def test_custom_candidates(self):
        rng = np.random.default_rng(42)
        data = rng.normal(100, 10, size=100)
        results = fit_all(data, candidates=["normal", "gamma"])
        assert len(results) == 2

    def test_fit_result_has_aic_bic(self):
        rng = np.random.default_rng(42)
        data = rng.normal(100, 10, size=100)
        results = fit_all(data, candidates=["normal"])
        assert results[0].aic < 0 or results[0].aic > 0  # Just check it exists
        assert results[0].bic < 0 or results[0].bic > 0
