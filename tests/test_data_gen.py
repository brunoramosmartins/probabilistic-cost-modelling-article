"""Tests for the synthetic data generator (src/data_gen.py)."""

import numpy as np
import pytest

from src.data_gen import (
    generate_hiring_data,
    generate_mixed_data,
    generate_overtime_data,
    generate_salary_data,
    generate_severance_data,
    generate_team_data,
    inject_outliers,
)
from src.distributions import LogNormalDist, NormalDist


class TestGenerateSalaryData:
    """Tests for generate_salary_data."""

    def test_output_shape(self):
        data = generate_salary_data(500, seed=42)
        assert data.shape == (500,)

    def test_all_positive(self):
        data = generate_salary_data(1000, seed=42)
        assert np.all(data > 0)

    def test_reasonable_range(self):
        """Default LogNormal(9.1, 0.4) should give salaries in thousands."""
        data = generate_salary_data(10_000, seed=42)
        assert data.mean() > 5000
        assert data.mean() < 20000

    def test_custom_distribution(self):
        dist = NormalDist(mu=10000.0, sigma=1000.0)
        data = generate_salary_data(10_000, distribution=dist, seed=42)
        assert abs(data.mean() - 10000) < 100

    def test_reproducibility(self):
        data1 = generate_salary_data(100, seed=42)
        data2 = generate_salary_data(100, seed=42)
        np.testing.assert_array_equal(data1, data2)

    def test_different_seeds_differ(self):
        data1 = generate_salary_data(100, seed=42)
        data2 = generate_salary_data(100, seed=99)
        assert not np.array_equal(data1, data2)


class TestGenerateOvertimeData:
    """Tests for generate_overtime_data."""

    def test_output_shape(self):
        data = generate_overtime_data(200, seed=42)
        assert data.shape == (200,)

    def test_all_positive(self):
        data = generate_overtime_data(1000, seed=42)
        assert np.all(data > 0)

    def test_default_mean(self):
        """Default Gamma(4, 1/30) has mean=120."""
        data = generate_overtime_data(50_000, seed=42)
        assert abs(data.mean() - 120) < 5


class TestGenerateSeveranceData:
    """Tests for generate_severance_data."""

    def test_output_shape(self):
        data = generate_severance_data(100, seed=42)
        assert data.shape == (100,)

    def test_all_above_minimum(self):
        """Default Pareto(2.5, 10000) should have all values >= 10000."""
        data = generate_severance_data(1000, seed=42)
        assert np.all(data >= 10000)

    def test_heavy_tail(self):
        """Some values should be much larger than the mean."""
        data = generate_severance_data(10_000, seed=42)
        mean = data.mean()
        assert np.any(data > 3 * mean)


class TestGenerateHiringData:
    """Tests for generate_hiring_data."""

    def test_output_shape(self):
        data = generate_hiring_data(50, seed=42)
        assert data.shape == (50,)

    def test_all_positive(self):
        data = generate_hiring_data(1000, seed=42)
        assert np.all(data > 0)


class TestGenerateMixedData:
    """Tests for generate_mixed_data (mixture distributions)."""

    def test_output_shape(self):
        c1 = NormalDist(mu=8000, sigma=1500)
        c2 = NormalDist(mu=18000, sigma=2500)
        data = generate_mixed_data(1000, [c1, c2], [0.6, 0.4], seed=42)
        assert data.shape == (1000,)

    def test_bimodal_structure(self):
        """Mixture should have mass around both component means."""
        c1 = NormalDist(mu=8000, sigma=500)
        c2 = NormalDist(mu=18000, sigma=500)
        data = generate_mixed_data(10_000, [c1, c2], [0.5, 0.5], seed=42)
        # Significant mass in both regions
        low_count = np.sum((data > 7000) & (data < 9000))
        high_count = np.sum((data > 17000) & (data < 19000))
        assert low_count > 3000
        assert high_count > 3000

    def test_weights_must_sum_to_one(self):
        c1 = NormalDist(mu=100, sigma=10)
        c2 = NormalDist(mu=200, sigma=10)
        with pytest.raises(ValueError, match="sum to 1.0"):
            generate_mixed_data(100, [c1, c2], [0.3, 0.3], seed=42)

    def test_mismatched_lengths_raise(self):
        c1 = NormalDist(mu=100, sigma=10)
        with pytest.raises(ValueError, match="same length"):
            generate_mixed_data(100, [c1], [0.5, 0.5], seed=42)

    def test_three_components(self):
        c1 = NormalDist(mu=5000, sigma=500)
        c2 = NormalDist(mu=12000, sigma=1000)
        c3 = NormalDist(mu=25000, sigma=2000)
        data = generate_mixed_data(
            3000, [c1, c2, c3], [0.5, 0.3, 0.2], seed=42
        )
        assert data.shape == (3000,)


class TestInjectOutliers:
    """Tests for inject_outliers."""

    def test_output_shape_preserved(self):
        data = np.ones(100)
        result = inject_outliers(data, fraction=0.1, multiplier=5.0, seed=42)
        assert result.shape == (100,)

    def test_original_unchanged(self):
        data = np.arange(100, dtype=np.float64)
        _ = inject_outliers(data, fraction=0.1, multiplier=5.0, seed=42)
        np.testing.assert_array_equal(data, np.arange(100, dtype=np.float64))

    def test_some_values_increased(self):
        data = np.ones(100) * 1000.0
        result = inject_outliers(data, fraction=0.1, multiplier=3.0, seed=42)
        n_changed = np.sum(result != 1000.0)
        assert n_changed == 10  # 10% of 100

    def test_multiplier_applied(self):
        data = np.ones(100) * 100.0
        result = inject_outliers(data, fraction=0.05, multiplier=10.0, seed=42)
        outliers = result[result != 100.0]
        np.testing.assert_allclose(outliers, 1000.0)

    def test_invalid_fraction_raises(self):
        data = np.ones(10)
        with pytest.raises(ValueError):
            inject_outliers(data, fraction=0.0)
        with pytest.raises(ValueError):
            inject_outliers(data, fraction=1.0)

    def test_invalid_multiplier_raises(self):
        data = np.ones(10)
        with pytest.raises(ValueError):
            inject_outliers(data, fraction=0.1, multiplier=0.0)


class TestGenerateTeamData:
    """Tests for generate_team_data."""

    def test_returns_all_components(self):
        result = generate_team_data(team_size=50, seed=42)
        assert "salary" in result
        assert "overtime" in result
        assert "severance" in result
        assert "hiring" in result

    def test_salary_shape(self):
        result = generate_team_data(team_size=30, seed=42)
        assert result["salary"].shape == (30,)

    def test_overtime_shape(self):
        result = generate_team_data(team_size=50, seed=42)
        assert result["overtime"].shape == (50,)

    def test_reproducibility(self):
        r1 = generate_team_data(team_size=50, seed=42)
        r2 = generate_team_data(team_size=50, seed=42)
        np.testing.assert_array_equal(r1["salary"], r2["salary"])
