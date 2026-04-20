"""Tests for GMM fitting and multimodality detection (src/mixture.py)."""

import numpy as np
import pytest

from src.mixture import (
    GMMResult,
    detect_multimodality,
    fit_gmm,
    select_K,
)


class TestFitGMM:
    """Tests for fit_gmm."""

    def test_single_component(self):
        rng = np.random.default_rng(42)
        data = rng.normal(10.0, 2.0, size=500)
        result = fit_gmm(data, K=1, seed=42)
        assert result.K == 1
        assert abs(result.means[0] - 10.0) < 0.5
        assert abs(np.sqrt(result.variances[0]) - 2.0) < 0.3

    def test_two_components_recovery(self):
        """Recover parameters of a well-separated 2-component mixture."""
        rng = np.random.default_rng(42)
        n1 = 600
        n2 = 400
        data = np.concatenate([
            rng.normal(8.0, 1.0, size=n1),
            rng.normal(20.0, 1.5, size=n2),
        ])
        rng.shuffle(data)

        result = fit_gmm(data, K=2, seed=42)
        assert result.K == 2

        # Sort by means for comparison
        order = np.argsort(result.means)
        means = result.means[order]
        weights = result.weights[order]

        assert abs(means[0] - 8.0) < 1.0
        assert abs(means[1] - 20.0) < 1.0
        assert abs(weights[0] - 0.6) < 0.1
        assert abs(weights[1] - 0.4) < 0.1

    def test_responsibilities_sum_to_one(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, size=100)
        result = fit_gmm(data, K=2, seed=42)
        row_sums = result.responsibilities.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, size=200)
        result = fit_gmm(data, K=3, seed=42)
        assert abs(result.weights.sum() - 1.0) < 1e-10

    def test_loglik_increases_with_iterations(self):
        """EM should monotonically increase log-likelihood."""
        rng = np.random.default_rng(42)
        data = np.concatenate([
            rng.normal(5, 1, size=200),
            rng.normal(15, 2, size=200),
        ])
        # Fit with very few iterations to check monotonicity
        result_few = fit_gmm(data, K=2, max_iter=5, seed=42, n_restarts=1)
        result_many = fit_gmm(data, K=2, max_iter=200, seed=42, n_restarts=1)
        assert result_many.loglik >= result_few.loglik

    def test_converges_for_well_separated(self):
        rng = np.random.default_rng(42)
        data = np.concatenate([
            rng.normal(0, 0.5, size=300),
            rng.normal(10, 0.5, size=300),
        ])
        result = fit_gmm(data, K=2, seed=42, n_restarts=10, max_iter=500)
        assert result.converged

    def test_responsibilities_shape(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, size=150)
        result = fit_gmm(data, K=3, seed=42)
        assert result.responsibilities.shape == (150, 3)

    def test_bic_computed(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, size=100)
        result = fit_gmm(data, K=2, seed=42)
        assert np.isfinite(result.bic)


class TestSelectK:
    """Tests for select_K."""

    def test_selects_one_for_unimodal(self):
        rng = np.random.default_rng(42)
        data = rng.normal(10.0, 2.0, size=500)
        optimal_K, results = select_K(data, K_range=[1, 2, 3], seed=42)
        assert optimal_K == 1

    def test_selects_more_than_one_for_bimodal(self):
        rng = np.random.default_rng(42)
        data = np.concatenate([
            rng.normal(5.0, 1.0, size=500),
            rng.normal(20.0, 1.0, size=500),
        ])
        optimal_K, results = select_K(data, K_range=[1, 2, 3], seed=42)
        assert optimal_K >= 2

    def test_selects_three_for_trimodal(self):
        rng = np.random.default_rng(42)
        data = np.concatenate([
            rng.normal(5.0, 0.8, size=400),
            rng.normal(15.0, 0.8, size=300),
            rng.normal(25.0, 0.8, size=300),
        ])
        optimal_K, results = select_K(data, K_range=[1, 2, 3, 4], seed=42)
        assert optimal_K == 3

    def test_returns_all_results(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, size=200)
        _, results = select_K(data, K_range=[1, 2, 3], seed=42)
        assert len(results) == 3


class TestDetectMultimodality:
    """Tests for detect_multimodality."""

    def test_unimodal_not_detected(self):
        rng = np.random.default_rng(42)
        data = rng.normal(10.0, 2.0, size=500)
        result = detect_multimodality(data, seed=42)
        assert result.is_multimodal is False
        assert result.optimal_K == 1

    def test_bimodal_detected(self):
        rng = np.random.default_rng(42)
        data = np.concatenate([
            rng.normal(5.0, 1.0, size=500),
            rng.normal(20.0, 1.5, size=500),
        ])
        result = detect_multimodality(data, seed=42)
        assert result.is_multimodal == True
        assert result.optimal_K >= 2

    def test_evidence_string(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, size=200)
        result = detect_multimodality(data, seed=42)
        assert "BIC" in result.evidence

    def test_returns_gmm_results(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, size=200)
        result = detect_multimodality(data, seed=42)
        assert len(result.gmm_results) == 4  # K=1,2,3,4
