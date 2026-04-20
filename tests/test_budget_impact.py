"""Tests for budget impact tools (src/budget_impact.py)."""

import numpy as np
import pytest

from src.budget_impact import (
    analytical_es_normal,
    analytical_es_pareto,
    analytical_var_normal,
    analytical_var_pareto,
    budget_reserve,
    compare_distributions_impact,
    expected_shortfall,
    var_at_level,
)
from src.fitting import fit_lognormal, fit_normal


class TestVarAtLevel:
    """Tests for var_at_level."""

    def test_median_at_50(self):
        data = np.arange(1.0, 101.0)
        var_50 = var_at_level(data, alpha=0.5)
        assert abs(var_50 - 50.5) < 1.0

    def test_higher_alpha_higher_var(self):
        rng = np.random.default_rng(42)
        data = rng.normal(100, 10, size=10_000)
        var_90 = var_at_level(data, 0.90)
        var_99 = var_at_level(data, 0.99)
        assert var_99 > var_90

    def test_normal_var_matches_analytical(self):
        rng = np.random.default_rng(42)
        data = rng.normal(100, 10, size=100_000)
        empirical = var_at_level(data, 0.95)
        analytical = analytical_var_normal(100, 10, 0.95)
        assert abs(empirical - analytical) < 0.5

    def test_invalid_alpha_raises(self):
        data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            var_at_level(data, 0.0)
        with pytest.raises(ValueError):
            var_at_level(data, 1.0)


class TestExpectedShortfall:
    """Tests for expected_shortfall."""

    def test_es_greater_than_var(self):
        rng = np.random.default_rng(42)
        data = rng.normal(100, 10, size=10_000)
        var = var_at_level(data, 0.95)
        es = expected_shortfall(data, 0.95)
        assert es >= var

    def test_normal_es_matches_analytical(self):
        rng = np.random.default_rng(42)
        data = rng.normal(50, 5, size=200_000)
        empirical = expected_shortfall(data, 0.95)
        analytical = analytical_es_normal(50, 5, 0.95)
        assert abs(empirical - analytical) < 0.2

    def test_heavier_tail_higher_es(self):
        """Pareto should have higher ES than Normal at same confidence."""
        rng = np.random.default_rng(42)
        normal_data = rng.normal(16667, 14907, size=100_000)
        u = rng.uniform(0, 1, size=100_000)
        pareto_data = 10000.0 / u ** (1.0 / 2.5)
        es_normal = expected_shortfall(normal_data, 0.99)
        es_pareto = expected_shortfall(pareto_data, 0.99)
        assert es_pareto > es_normal


class TestBudgetReserve:
    """Tests for budget_reserve."""

    def test_positive_for_right_skewed(self):
        rng = np.random.default_rng(42)
        data = rng.lognormal(9.1, 0.4, size=10_000)
        reserve = budget_reserve(data, 0.95)
        assert reserve > 0

    def test_zero_at_mean(self):
        """Reserve at ~50% confidence should be near zero for symmetric data."""
        rng = np.random.default_rng(42)
        data = rng.normal(100, 10, size=100_000)
        reserve = budget_reserve(data, 0.5)
        assert abs(reserve) < 1.0


class TestAnalyticalVarPareto:
    """Tests for analytical_var_pareto."""

    def test_known_value(self):
        # VaR_95 = x_m * (0.05)^{-1/alpha} = 10000 * 20^{1/2.5}
        var = analytical_var_pareto(2.5, 10000.0, 0.95)
        expected = 10000.0 * (0.05) ** (-1.0 / 2.5)
        assert abs(var - expected) < 0.01

    def test_higher_confidence_higher_var(self):
        var_95 = analytical_var_pareto(2.5, 10000.0, 0.95)
        var_99 = analytical_var_pareto(2.5, 10000.0, 0.99)
        assert var_99 > var_95


class TestAnalyticalEsPareto:
    """Tests for analytical_es_pareto."""

    def test_es_greater_than_var(self):
        var = analytical_var_pareto(2.5, 10000.0, 0.95)
        es = analytical_es_pareto(2.5, 10000.0, 0.95)
        assert es > var

    def test_ratio_to_var(self):
        """ES = (alpha/(alpha-1)) * VaR for Pareto."""
        var = analytical_var_pareto(3.0, 1000.0, 0.95)
        es = analytical_es_pareto(3.0, 1000.0, 0.95)
        assert abs(es / var - 3.0 / 2.0) < 1e-10

    def test_infinite_for_alpha_leq_1(self):
        es = analytical_es_pareto(1.0, 1000.0, 0.95)
        assert es == float("inf")


class TestCompareDistributionsImpact:
    """Tests for compare_distributions_impact."""

    def test_returns_dataframe(self):
        rng = np.random.default_rng(42)
        data = rng.lognormal(9.1, 0.4, size=500)
        from src.fitting import fit_all
        results = fit_all(data, candidates=["normal", "lognormal"])
        df = compare_distributions_impact(results, n_simulations=10_000, seed=42)
        assert "Distribution" in df.columns
        assert "VaR_95%" in df.columns
        assert "ES_95%" in df.columns
        assert len(df) == 2

    def test_pareto_higher_es_than_normal(self):
        """When Pareto is the true model, its ES should be higher than Normal's."""
        from src.fitting import FitResult
        results = [
            FitResult(
                distribution="Normal",
                params={"mu": 16667.0, "sigma": 14907.0},
                loglik=-1000.0, n_params=2, n_obs=100,
                se={}, ci_lower={}, ci_upper={},
            ),
            FitResult(
                distribution="Pareto",
                params={"alpha": 2.5, "x_m": 10000.0},
                loglik=-900.0, n_params=1, n_obs=100,
                se={}, ci_lower={}, ci_upper={},
            ),
        ]
        df = compare_distributions_impact(results, n_simulations=50_000, seed=42)
        es_normal = df[df["Distribution"] == "Normal"]["ES_99%"].iloc[0]
        es_pareto = df[df["Distribution"] == "Pareto"]["ES_99%"].iloc[0]
        assert es_pareto > es_normal
