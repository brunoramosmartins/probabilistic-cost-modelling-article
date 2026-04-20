"""Gaussian Mixture Model fitting via the EM algorithm.

Implements the EM algorithm from scratch for the article's derivation,
with a scipy/sklearn fallback for validation. Includes BIC-based K selection
and multimodality detection.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from src.model_selection import compute_bic


@dataclass
class GMMResult:
    """Result of fitting a Gaussian Mixture Model.

    Attributes
    ----------
    K : int
        Number of components.
    weights : NDArray[np.float64]
        Mixing weights (shape: K,). Sum to 1.
    means : NDArray[np.float64]
        Component means (shape: K,).
    variances : NDArray[np.float64]
        Component variances (shape: K,).
    responsibilities : NDArray[np.float64]
        Responsibility matrix (shape: n, K). gamma[i,k] = P(z_i=k|x_i).
    loglik : float
        Final log-likelihood.
    n_iter : int
        Number of EM iterations performed.
    converged : bool
        Whether convergence criterion was met.
    bic : float
        BIC value for this model.
    """

    K: int
    weights: NDArray[np.float64]
    means: NDArray[np.float64]
    variances: NDArray[np.float64]
    responsibilities: NDArray[np.float64]
    loglik: float
    n_iter: int
    converged: bool
    bic: float


def _compute_responsibilities(
    data: NDArray[np.float64],
    weights: NDArray[np.float64],
    means: NDArray[np.float64],
    variances: NDArray[np.float64],
) -> NDArray[np.float64]:
    """E-step: compute responsibilities (posterior probabilities).

    Parameters
    ----------
    data : NDArray, shape (n,)
    weights : NDArray, shape (K,)
    means : NDArray, shape (K,)
    variances : NDArray, shape (K,)

    Returns
    -------
    NDArray, shape (n, K)
        Responsibility matrix.
    """
    n = len(data)
    K = len(weights)
    resp = np.zeros((n, K))

    for k in range(K):
        resp[:, k] = weights[k] * stats.norm.pdf(
            data, loc=means[k], scale=np.sqrt(variances[k])
        )

    # Normalize rows
    row_sums = resp.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums = np.maximum(row_sums, 1e-300)
    resp /= row_sums

    return resp


def _compute_loglikelihood(
    data: NDArray[np.float64],
    weights: NDArray[np.float64],
    means: NDArray[np.float64],
    variances: NDArray[np.float64],
) -> float:
    """Compute observed-data log-likelihood.

    Parameters
    ----------
    data : NDArray, shape (n,)
    weights, means, variances : NDArray, shape (K,)

    Returns
    -------
    float
        Log-likelihood value.
    """
    n = len(data)
    K = len(weights)
    ll = np.zeros(n)

    for k in range(K):
        ll += weights[k] * stats.norm.pdf(
            data, loc=means[k], scale=np.sqrt(variances[k])
        )

    return float(np.sum(np.log(np.maximum(ll, 1e-300))))


def fit_gmm(
    data: NDArray[np.float64],
    K: int,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int | None = None,
    n_restarts: int = 5,
) -> GMMResult:
    """Fit a K-component Gaussian Mixture Model via EM.

    Parameters
    ----------
    data : NDArray[np.float64]
        Observed data (1D array).
    K : int
        Number of mixture components.
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance on log-likelihood change.
    seed : int, optional
        Random seed for initialization.
    n_restarts : int
        Number of random restarts. Best result (highest loglik) is returned.

    Returns
    -------
    GMMResult
        Best fitting result across all restarts.

    Notes
    -----
    Implements the EM algorithm derived in notes/phase4-mixture.md:
    - E-step: gamma_ik = pi_k * N(x_i|mu_k, sigma_k^2) / sum_j(...)
    - M-step: pi_k = N_k/n, mu_k = sum(gamma*x)/N_k, sigma_k^2 = sum(gamma*(x-mu)^2)/N_k
    """
    n = len(data)
    rng = np.random.default_rng(seed)

    best_result = None

    for restart in range(n_restarts):
        # --- Initialization: k-means-like ---
        # Random partition of data into K groups
        indices = rng.permutation(n)
        chunk_size = n // K

        weights = np.ones(K) / K
        means = np.zeros(K)
        variances = np.zeros(K)

        for k in range(K):
            start = k * chunk_size
            end = start + chunk_size if k < K - 1 else n
            subset = data[indices[start:end]]
            means[k] = subset.mean()
            variances[k] = max(subset.var(), 1e-6)

        # Sort means for reproducibility
        order = np.argsort(means)
        means = means[order]
        variances = variances[order]

        # --- EM iterations ---
        prev_loglik = -np.inf
        converged = False

        for iteration in range(max_iter):
            # E-step
            resp = _compute_responsibilities(data, weights, means, variances)

            # M-step
            N_k = resp.sum(axis=0)  # shape (K,)
            N_k = np.maximum(N_k, 1e-10)  # avoid division by zero

            weights = N_k / n
            means = (resp.T @ data) / N_k

            for k in range(K):
                diff = data - means[k]
                variances[k] = float((resp[:, k] @ (diff**2)) / N_k[k])
                variances[k] = max(variances[k], 1e-6)  # floor to avoid collapse

            # Compute log-likelihood
            loglik = _compute_loglikelihood(data, weights, means, variances)

            # Check convergence
            if abs(loglik - prev_loglik) < tol:
                converged = True
                break
            prev_loglik = loglik

        n_iter = iteration + 1

        # Compute final responsibilities
        resp = _compute_responsibilities(data, weights, means, variances)

        # BIC: 3K - 1 free parameters (K means + K variances + K-1 weights)
        n_params = 3 * K - 1
        bic = compute_bic(loglik, n_params, n)

        result = GMMResult(
            K=K,
            weights=weights,
            means=means,
            variances=variances,
            responsibilities=resp,
            loglik=loglik,
            n_iter=n_iter,
            converged=converged,
            bic=bic,
        )

        if best_result is None or loglik > best_result.loglik:
            best_result = result

    return best_result


def select_K(
    data: NDArray[np.float64],
    K_range: range | list[int] | None = None,
    criterion: str = "bic",
    seed: int | None = None,
) -> tuple[int, list[GMMResult]]:
    """Select optimal number of components K using information criteria.

    Parameters
    ----------
    data : NDArray[np.float64]
        Observed data.
    K_range : range or list[int], optional
        Range of K values to try. Defaults to [1, 2, 3, 4, 5].
    criterion : str
        Selection criterion: 'bic' (default) or 'aic'.
    seed : int, optional
        Random seed.

    Returns
    -------
    tuple[int, list[GMMResult]]
        (optimal K, list of all GMMResults)
    """
    if K_range is None:
        K_range = [1, 2, 3, 4, 5]

    results = []
    for K in K_range:
        result = fit_gmm(data, K=K, seed=seed)
        results.append(result)

    if criterion == "bic":
        scores = [r.bic for r in results]
    else:
        # AIC: 2*n_params - 2*loglik
        scores = [2 * (3 * r.K - 1) - 2 * r.loglik for r in results]

    best_idx = int(np.argmin(scores))
    optimal_K = list(K_range)[best_idx]

    return optimal_K, results


@dataclass
class MultimodalityResult:
    """Result of multimodality detection.

    Attributes
    ----------
    is_multimodal : bool
        Whether the data appears multimodal.
    optimal_K : int
        Optimal number of components.
    evidence : str
        Description of the evidence.
    gmm_results : list[GMMResult]
        All fitted GMM results.
    """

    is_multimodal: bool
    optimal_K: int
    evidence: str
    gmm_results: list[GMMResult]


def detect_multimodality(
    data: NDArray[np.float64],
    seed: int | None = None,
) -> MultimodalityResult:
    """Detect whether data is multimodal using BIC-based GMM selection.

    Parameters
    ----------
    data : NDArray[np.float64]
        Observed data.
    seed : int, optional
        Random seed.

    Returns
    -------
    MultimodalityResult
        Detection result with evidence.

    Notes
    -----
    Fits GMMs with K=1,2,3,4 and selects the best K via BIC.
    Multimodality is declared if K > 1 AND the BIC improvement from K=1
    to optimal K is substantial (delta BIC > 10).
    """
    optimal_K, results = select_K(data, K_range=[1, 2, 3, 4], seed=seed)

    bic_1 = results[0].bic  # K=1
    bic_best = results[optimal_K - 1].bic
    delta_bic = bic_1 - bic_best  # positive means K>1 is better

    # Strong evidence threshold: delta BIC > 10 (Kass & Raftery, 1995)
    is_multimodal = optimal_K > 1 and delta_bic > 10

    if is_multimodal:
        evidence = (
            f"BIC selects K={optimal_K}. "
            f"BIC(K=1)={bic_1:.1f}, BIC(K={optimal_K})={bic_best:.1f}, "
            f"delta={delta_bic:.1f} > 10 (strong evidence)."
        )
    else:
        evidence = (
            f"BIC selects K={optimal_K}. "
            f"BIC(K=1)={bic_1:.1f}, delta={delta_bic:.1f}. "
            f"Insufficient evidence for multimodality."
        )

    return MultimodalityResult(
        is_multimodal=is_multimodal,
        optimal_K=optimal_K,
        evidence=evidence,
        gmm_results=results,
    )
