"""
Diagnostic metrics for the importance-sampling credit-risk estimator.

Covers:
  * Second-moment / LR-margin check for Gaussian proposals.
  * Spectral health of the fitted covariance matrix.
  * Production ESS for the final estimator.
  * Risk-contribution (expected shortfall allocation) estimates with
    delta-method standard errors.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Spectral / LR-margin diagnostics
# ---------------------------------------------------------------------------

def gaussian_lr_margin(Sigma: np.ndarray) -> float:
    """
    Compute the Gaussian likelihood-ratio second-moment margin.

    The margin is defined as:

        m_LR(Sigma) = lambda_min(2*I - Sigma^{-1})

    A positive value certifies that the second-moment condition
    E[R_Z^2] < infinity holds under the standard Gaussian prior, i.e. the
    proposal N(mu, Sigma) has a square-integrable likelihood ratio against
    N(0, I_d).

    Parameters
    ----------
    Sigma : np.ndarray, shape (d, d)
        Symmetric positive-definite covariance matrix of the proposal.

    Returns
    -------
    float
        Minimum eigenvalue of (2*I - Sigma^{-1}).  Positive = condition
        satisfied.
    """
    Sigma = np.asarray(Sigma, dtype=np.float64)
    d = Sigma.shape[0]
    Sigma_inv = np.linalg.inv(Sigma)
    M = 2.0 * np.eye(d, dtype=np.float64) - Sigma_inv
    eigvals = np.linalg.eigvalsh(M)
    return float(eigvals.min())


def min_eigenvalue(Sigma: np.ndarray) -> float:
    """
    Minimum eigenvalue of a matrix after symmetrisation.

    Parameters
    ----------
    Sigma : np.ndarray, shape (d, d)
        Square matrix (nominally symmetric; symmetrised before computation).

    Returns
    -------
    float
        lambda_min((Sigma + Sigma.T) / 2).
    """
    Sigma = np.asarray(Sigma, dtype=np.float64)
    Sigma_sym = (Sigma + Sigma.T) * 0.5
    return float(np.linalg.eigvalsh(Sigma_sym).min())


def condition_number(Sigma: np.ndarray) -> float:
    """
    Spectral condition number kappa_2(Sigma) = lambda_max / lambda_min.

    Parameters
    ----------
    Sigma : np.ndarray, shape (d, d)
        Symmetric positive-definite matrix.

    Returns
    -------
    float
        Ratio of the largest to the smallest eigenvalue.
    """
    Sigma = np.asarray(Sigma, dtype=np.float64)
    eigvals = np.linalg.eigvalsh(Sigma)
    return float(eigvals.max() / eigvals.min())


# ---------------------------------------------------------------------------
# Production ESS
# ---------------------------------------------------------------------------

def compute_production_ess(weights_times_indicator: np.ndarray) -> float:
    """
    Effective sample size for the production IS estimator.

    ESS = (sum w)^2 / (sum w^2)

    computed only over samples with non-zero weight (i.e. samples inside the
    rare event A = {L >= x}).

    Parameters
    ----------
    weights_times_indicator : np.ndarray, shape (M,)
        Product Lambda_m * 1_A(sample_m) for each pilot draw m.

    Returns
    -------
    float
        Production ESS.  Returns 0.0 when all weights are zero.
    """
    w = np.asarray(weights_times_indicator, dtype=np.float64)
    w_pos = w[w > 0.0]
    if w_pos.size == 0:
        return 0.0
    return float(w_pos.sum() ** 2 / np.dot(w_pos, w_pos))


# ---------------------------------------------------------------------------
# Risk-contribution estimates
# ---------------------------------------------------------------------------

def compute_contribution_estimates(
    Lambda: np.ndarray,
    L_obligor: np.ndarray,
    Y_obligor: np.ndarray,
    indicator: np.ndarray,
) -> dict:
    """
    Estimate per-obligor expected-loss contributions conditional on event A.

    The IS estimator of the conditional contribution for obligor k is:

        r_k = sum_m (Lambda_m * l_k * Y_{k,m} * 1_{A,m})
              / sum_m (Lambda_m * 1_{A,m})

    Standard errors are obtained via the delta method for ratio estimators:

        e_{k,m} = w_m * (l_k * Y_{k,m} - r_k),   w_m = Lambda_m * 1_{A,m}
        se_k    = sqrt(sum_m e_{k,m}^2) / sum_m w_m

    Parameters
    ----------
    Lambda : np.ndarray, shape (M,)
        Full likelihood-ratio values for each pilot draw.
    L_obligor : np.ndarray, shape (M, N)
        Loss-at-default for each obligor in each draw (constant across m,
        repeated to shape (M, N)).
    Y_obligor : np.ndarray, shape (M, N)
        Binary default indicators (1 = default, 0 = no default) for each
        obligor in each draw.
    indicator : np.ndarray, shape (M,)
        Event indicator 1_{A}(sample_m)  (1 or 0).

    Returns
    -------
    dict with keys:
        'r'               : np.ndarray, shape (N,) – contribution estimates.
        'se'              : np.ndarray, shape (N,) – delta-method standard
                            errors.
        'half_length_95'  : np.ndarray, shape (N,) – 1.96 * se (95 % CI
                            half-lengths).
        'mean_half_length': float – mean of half_length_95 over obligors.
        'prob_estimate'   : float – IS estimate of P(A) = mean(Lambda*1_A).
        'total_weight'    : float – sum(Lambda * 1_A).
    """
    Lambda = np.asarray(Lambda, dtype=np.float64)          # (M,)
    L_obligor = np.asarray(L_obligor, dtype=np.float64)    # (M, N)
    Y_obligor = np.asarray(Y_obligor, dtype=np.float64)    # (M, N)
    indicator = np.asarray(indicator, dtype=np.float64)    # (M,)
    M = Lambda.shape[0]

    # Effective weight per draw: w_m = Lambda_m * 1_{A,m}
    w = Lambda * indicator                                 # (M,)
    total_weight = float(w.sum())
    prob_estimate = total_weight / M

    if total_weight == 0.0:
        N = L_obligor.shape[1]
        zeros_N = np.zeros(N, dtype=np.float64)
        return {
            "r": zeros_N,
            "se": zeros_N,
            "half_length_95": zeros_N,
            "mean_half_length": 0.0,
            "prob_estimate": 0.0,
            "total_weight": 0.0,
        }

    # Numerator for each obligor: sum_m w_m * l_k * Y_{k,m}
    # L_obligor * Y_obligor: (M, N);  w[:, None] broadcasts over N
    weighted_loss = w[:, None] * L_obligor * Y_obligor    # (M, N)
    r = weighted_loss.sum(axis=0) / total_weight          # (N,)

    # Delta-method standard errors
    # e_{k,m} = w_m * (l_k * Y_{k,m} - r_k)
    residuals = w[:, None] * (L_obligor * Y_obligor - r[None, :])  # (M, N)
    se = np.sqrt((residuals ** 2).sum(axis=0)) / total_weight      # (N,)

    half_length_95 = 1.96 * se

    return {
        "r": r,
        "se": se,
        "half_length_95": half_length_95,
        "mean_half_length": float(half_length_95.mean()),
        "prob_estimate": prob_estimate,
        "total_weight": total_weight,
    }
