"""Tests for tail analysis tools (src/heavy_tails.py)."""

import numpy as np
import pytest

from src.heavy_tails import (
    compare_tail_risk,
    hill_estimator,
    hill_plot_data,
    qq_plot_data,
    tail_probability,
)


class TestHillEstimator:
    """Tests for hill_estimator."""

    def test_recovers_known_alpha(self):
        """Hill estimator should recover alpha for Pareto data."""
        rng = np.random.default_rng(42)
        alpha_true = 2.5
        x_m = 1000.0
        u = rng.uniform(0, 1, size=10_000)
        data = x_m / u ** (1.0 / alpha_true)
        # Use k = 500 (5% of data)
        alpha_hat = hill_estimator(data, k=500)
        assert abs(alpha_hat - alpha_true) < 0.3

    def test_smaller_k_higher_variance(self):
        """Smaller k gives noisier estimates."""
        rng = np.random.default_rng(42)
        u = rng.uniform(0, 1, size=5000)
        data = 1000.0 / u ** (1.0 / 3.0)

        estimates_small_k = [
            hill_estimator(data[rng.choice(5000, 5000)], k=20) for _ in range(50)
        ]
        estimates_large_k = [
            hill_estimator(data[rng.choice(5000, 5000)], k=200) for _ in range(50)
        ]
        assert np.std(estimates_small_k) > np.std(estimates_large_k)

    def test_invalid_k_raises(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError):
            hill_estimator(data, k=0)
        with pytest.raises(ValueError):
            hill_estimator(data, k=5)

    def test_positive_result(self):
        rng = np.random.default_rng(42)
        data = rng.pareto(3.0, size=1000) + 1
        alpha = hill_estimator(data, k=100)
        assert alpha > 0


class TestHillPlotData:
    """Tests for hill_plot_data."""

    def test_returns_lists(self):
        rng = np.random.default_rng(42)
        data = rng.pareto(2.5, size=500) + 1
        k_values, alpha_values = hill_plot_data(data)
        assert len(k_values) > 0
        assert len(k_values) == len(alpha_values)

    def test_custom_k_range(self):
        rng = np.random.default_rng(42)
        data = rng.pareto(2.5, size=200) + 1
        k_values, alpha_values = hill_plot_data(data, k_range=[5, 10, 20, 30])
        assert len(k_values) <= 4


class TestTailProbability:
    """Tests for tail_probability."""

    def test_pareto_exact(self):
        """P(X > x) = (x_m/x)^alpha for Pareto."""
        prob = tail_probability("pareto", {"alpha": 2.5, "x_m": 10000.0}, 50000.0)
        expected = (10000.0 / 50000.0) ** 2.5
        assert abs(prob - expected) < 1e-6

    def test_normal_symmetric(self):
        """P(X > mu) = 0.5 for Normal."""
        prob = tail_probability("normal", {"mu": 100.0, "sigma": 10.0}, 100.0)
        assert abs(prob - 0.5) < 1e-10

    def test_higher_threshold_lower_prob(self):
        params = {"alpha": 2.5, "x_m": 1000.0}
        p1 = tail_probability("pareto", params, 5000.0)
        p2 = tail_probability("pareto", params, 10000.0)
        assert p1 > p2

    def test_unknown_distribution_raises(self):
        with pytest.raises(ValueError):
            tail_probability("student_t", {"df": 5.0}, 10.0)


class TestQQPlotData:
    """Tests for qq_plot_data."""

    def test_output_shapes_match(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, size=100)
        theoretical, empirical = qq_plot_data(data, "normal", {"mu": 0.0, "sigma": 1.0})
        assert theoretical.shape == (100,)
        assert empirical.shape == (100,)

    def test_sorted_ascending(self):
        rng = np.random.default_rng(42)
        data = rng.normal(5, 2, size=200)
        theoretical, empirical = qq_plot_data(data, "normal", {"mu": 5.0, "sigma": 2.0})
        assert np.all(np.diff(empirical) >= 0)
        assert np.all(np.diff(theoretical) >= 0)

    def test_good_fit_close_to_diagonal(self):
        """For data from the correct distribution, Q-Q should be near y=x."""
        rng = np.random.default_rng(42)
        data = rng.normal(10, 3, size=1000)
        theoretical, empirical = qq_plot_data(data, "normal", {"mu": 10.0, "sigma": 3.0})
        correlation = np.corrcoef(theoretical, empirical)[0, 1]
        assert correlation > 0.99


class TestCompareTailRisk:
    """Tests for compare_tail_risk."""

    def test_returns_dataframe(self):
        distributions = {
            "pareto:Pareto(2.5)": {"alpha": 2.5, "x_m": 10000.0},
            "normal:Normal": {"mu": 16667.0, "sigma": 14907.0},
        }
        df = compare_tail_risk([50000.0, 100000.0], distributions)
        assert len(df) == 2
        assert "Pareto(2.5)" in df.columns
        assert "Normal" in df.columns

    def test_pareto_higher_tail_than_normal(self):
        distributions = {
            "pareto:Pareto": {"alpha": 2.5, "x_m": 10000.0},
            "normal:Normal": {"mu": 16667.0, "sigma": 14907.0},
        }
        df = compare_tail_risk([80000.0], distributions)
        assert df["Pareto"].iloc[0] > df["Normal"].iloc[0]
