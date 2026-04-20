"""Tests for model comparison tools (src/model_selection.py)."""

import numpy as np
import pytest

from src.fitting import fit_all, fit_gamma, fit_lognormal, fit_normal
from src.model_selection import (
    ad_test,
    aic_weights,
    compare_models,
    compute_aic,
    compute_aicc,
    compute_bic,
    ks_test,
    likelihood_ratio_test,
)


class TestComputeAIC:
    """Tests for compute_aic."""

    def test_formula(self):
        # AIC = 2k - 2*loglik
        assert compute_aic(loglik=-100.0, k=2) == 204.0

    def test_more_params_higher_aic(self):
        # Same loglik, more params -> higher AIC
        aic_simple = compute_aic(loglik=-100.0, k=2)
        aic_complex = compute_aic(loglik=-100.0, k=5)
        assert aic_complex > aic_simple

    def test_better_fit_lower_aic(self):
        # Same params, better loglik -> lower AIC
        aic_bad = compute_aic(loglik=-200.0, k=2)
        aic_good = compute_aic(loglik=-100.0, k=2)
        assert aic_good < aic_bad


class TestComputeAICc:
    """Tests for compute_aicc."""

    def test_converges_to_aic_for_large_n(self):
        aic = compute_aic(loglik=-100.0, k=2)
        aicc = compute_aicc(loglik=-100.0, k=2, n=10000)
        assert abs(aicc - aic) < 0.01

    def test_larger_than_aic(self):
        aic = compute_aic(loglik=-100.0, k=2)
        aicc = compute_aicc(loglik=-100.0, k=2, n=20)
        assert aicc > aic

    def test_infinite_when_n_too_small(self):
        # n - k - 1 <= 0
        aicc = compute_aicc(loglik=-100.0, k=5, n=5)
        assert aicc == float("inf")


class TestComputeBIC:
    """Tests for compute_bic."""

    def test_formula(self):
        # BIC = k*log(n) - 2*loglik
        expected = 2 * np.log(100) - 2 * (-100.0)
        assert abs(compute_bic(loglik=-100.0, k=2, n=100) - expected) < 1e-10

    def test_penalizes_more_than_aic_for_large_n(self):
        # For n > 7, BIC penalty k*log(n) > AIC penalty 2k
        aic = compute_aic(loglik=-100.0, k=3)
        bic = compute_bic(loglik=-100.0, k=3, n=100)
        assert bic > aic


class TestAICWeights:
    """Tests for aic_weights."""

    def test_sum_to_one(self):
        weights = aic_weights([100.0, 102.0, 110.0])
        assert abs(weights.sum() - 1.0) < 1e-10

    def test_best_model_highest_weight(self):
        weights = aic_weights([100.0, 105.0, 120.0])
        assert weights[0] > weights[1] > weights[2]

    def test_equal_aic_equal_weights(self):
        weights = aic_weights([100.0, 100.0, 100.0])
        np.testing.assert_allclose(weights, [1 / 3, 1 / 3, 1 / 3])

    def test_large_delta_negligible_weight(self):
        weights = aic_weights([100.0, 120.0])  # delta=20
        assert weights[1] < 0.001


class TestLikelihoodRatioTest:
    """Tests for likelihood_ratio_test."""

    def test_significant_result(self):
        # Large difference should reject
        result = likelihood_ratio_test(
            loglik_restricted=-4521.3,
            loglik_full=-4518.1,
            df=1,
        )
        assert result.statistic == pytest.approx(6.4, abs=0.01)
        assert result.reject is True
        assert result.p_value < 0.05

    def test_non_significant_result(self):
        # Small difference should not reject
        result = likelihood_ratio_test(
            loglik_restricted=-100.0,
            loglik_full=-99.8,
            df=1,
        )
        assert result.reject is False
        assert result.p_value > 0.05

    def test_statistic_non_negative(self):
        result = likelihood_ratio_test(
            loglik_restricted=-100.0,
            loglik_full=-100.0,
            df=1,
        )
        assert result.statistic >= 0

    def test_invalid_df_raises(self):
        with pytest.raises(ValueError):
            likelihood_ratio_test(-100.0, -99.0, df=0)

    def test_chi2_df(self):
        result = likelihood_ratio_test(-100.0, -95.0, df=3)
        assert result.df == 3


class TestKSTest:
    """Tests for ks_test."""

    def test_accepts_correct_distribution(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, size=500)
        from scipy.stats import norm

        result = ks_test(data, norm.cdf)
        assert result.reject is False

    def test_rejects_wrong_distribution(self):
        rng = np.random.default_rng(42)
        data = rng.lognormal(9, 0.5, size=500)
        from scipy.stats import norm

        # Fit Normal to clearly non-Normal data
        mu, sigma = data.mean(), data.std()
        result = ks_test(data, lambda x: norm.cdf(x, loc=mu, scale=sigma))
        assert result.reject is True

    def test_statistic_between_0_and_1(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, size=100)
        from scipy.stats import norm

        result = ks_test(data, norm.cdf)
        assert 0 <= result.statistic <= 1


class TestADTest:
    """Tests for ad_test."""

    def test_accepts_correct_distribution(self):
        rng = np.random.default_rng(42)
        data = rng.normal(5, 2, size=500)
        from scipy.stats import norm

        result = ad_test(data, lambda x: norm.cdf(x, loc=5, scale=2))
        assert result.reject is False

    def test_rejects_wrong_distribution(self):
        rng = np.random.default_rng(42)
        # Generate heavy-tailed data, test against Normal
        data = rng.standard_t(df=3, size=500) * 10 + 50
        from scipy.stats import norm

        mu, sigma = data.mean(), data.std()
        result = ad_test(data, lambda x: norm.cdf(x, loc=mu, scale=sigma))
        assert result.reject is True

    def test_statistic_positive(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, size=100)
        from scipy.stats import norm

        result = ad_test(data, norm.cdf)
        assert result.statistic > 0


class TestCompareModels:
    """Tests for compare_models."""

    def test_returns_dataframe(self):
        rng = np.random.default_rng(42)
        data = rng.lognormal(9.0, 0.5, size=200)
        results = fit_all(data, candidates=["normal", "lognormal", "gamma"])
        df = compare_models(results)
        assert "AIC" in df.columns
        assert "BIC" in df.columns
        assert "AIC_weight" in df.columns

    def test_sorted_by_aic(self):
        rng = np.random.default_rng(42)
        data = rng.lognormal(9.0, 0.5, size=500)
        results = fit_all(data, candidates=["normal", "lognormal", "gamma"])
        df = compare_models(results)
        assert df["AIC"].is_monotonic_increasing

    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(42)
        data = rng.lognormal(9.0, 0.5, size=200)
        results = fit_all(data, candidates=["normal", "lognormal", "gamma"])
        df = compare_models(results)
        assert abs(df["AIC_weight"].sum() - 1.0) < 1e-10

    def test_lognormal_wins_for_lognormal_data(self):
        rng = np.random.default_rng(42)
        data = rng.lognormal(9.0, 0.5, size=1000)
        results = fit_all(data, candidates=["normal", "lognormal", "gamma"])
        df = compare_models(results)
        assert df.iloc[0]["Distribution"] == "LogNormal"

    def test_delta_aic_min_is_zero(self):
        rng = np.random.default_rng(42)
        data = rng.gamma(5, 20, size=200)
        results = fit_all(data, candidates=["normal", "gamma"])
        df = compare_models(results)
        assert df["delta_AIC"].min() == 0.0
