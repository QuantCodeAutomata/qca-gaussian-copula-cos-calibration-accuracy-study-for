"""
Weighted Gaussian moment fitting and importance-sampling calibration metrics.

Given a set of pilot factor draws Z (shape M × d) and associated non-negative
importance weights omega (shape M,), this module:

  1. Fits the first two moments of a Gaussian proposal distribution (mu, Sigma)
     to the weighted sample, with optional ridge regularisation of Sigma.

  2. Computes relative errors between fitted moments and reference moments.

  3. Evaluates the mean absolute weight error between a COS tail-probability
     estimate and an exact convolution reference.

  4. Computes the *spurious mass share*: the fraction of total COS importance
     weight concentrated at states where the exact tail probability is
     numerically negligible (below a threshold).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Weighted moment fitting
# ---------------------------------------------------------------------------

def fit_gaussian_moments(
    Z: np.ndarray,
    weights: np.ndarray,
    rho: float = 1e-8,
) -> dict:
    """
    Fit weighted mean and covariance of a Gaussian from importance-weighted
    pilot draws.

    The weighted estimators are:

        S_omega = sum_m  omega_m
        mu      = (1/S_omega) * sum_m omega_m * z_m
        Sigma   = (1/S_omega) * sum_m omega_m * (z_m - mu)(z_m - mu)^T

    A symmetrisation step and optional ridge correction are applied to Sigma
    to guarantee numerical positive-definiteness.

    Parameters
    ----------
    Z : np.ndarray, shape (M, d)
        Pilot factor draws.
    weights : np.ndarray, shape (M,)
        Non-negative importance weights (e.g. omega_m = q^ex(z_m)).
    rho : float, optional
        Ridge parameter added to the diagonal of Sigma.  Default: 1e-8.

    Returns
    -------
    dict with keys:
        'S_omega'     : float  – sum of weights.
        'mu'          : np.ndarray, shape (d,)  – weighted mean.
        'Sigma_raw'   : np.ndarray, shape (d,d) – raw weighted covariance.
        'Sigma_sym'   : np.ndarray, shape (d,d) – symmetrised covariance.
        'Sigma_ridge' : np.ndarray, shape (d,d) – ridge-regularised covariance.
        'ESS'         : float  – calibration ESS = (sum w)^2 / sum(w^2).
        'mean_weight' : float  – mean weight S_omega / M.
        'success'     : bool   – True when S_omega > 0.
    """
    Z = np.asarray(Z, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    M, d = Z.shape

    S_omega = float(weights.sum())
    success = S_omega > 0.0

    if not success:
        nan_vec = np.full(d, np.nan)
        nan_mat = np.full((d, d), np.nan)
        return {
            "S_omega": 0.0,
            "mu": nan_vec,
            "Sigma_raw": nan_mat,
            "Sigma_sym": nan_mat,
            "Sigma_ridge": nan_mat,
            "ESS": 0.0,
            "mean_weight": 0.0,
            "success": False,
        }

    # Weighted mean
    mu = (weights @ Z) / S_omega                               # shape (d,)

    # Weighted covariance  (biased, divided by S_omega)
    Z_c = Z - mu[None, :]                                      # (M, d)
    Sigma_raw = (weights[:, None] * Z_c).T @ Z_c / S_omega    # (d, d)

    # Symmetrise and add ridge
    Sigma_sym = (Sigma_raw + Sigma_raw.T) * 0.5
    Sigma_ridge = Sigma_sym + rho * np.eye(d, dtype=np.float64)

    # Calibration ESS
    ESS = S_omega ** 2 / float(np.dot(weights, weights))

    return {
        "S_omega": S_omega,
        "mu": mu,
        "Sigma_raw": Sigma_raw,
        "Sigma_sym": Sigma_sym,
        "Sigma_ridge": Sigma_ridge,
        "ESS": ESS,
        "mean_weight": S_omega / M,
        "success": True,
    }


# ---------------------------------------------------------------------------
# Calibration error diagnostics
# ---------------------------------------------------------------------------

def compute_calibration_errors(
    mu_K: np.ndarray,
    Sigma_K: np.ndarray,
    mu_ex: np.ndarray,
    Sigma_ex: np.ndarray,
) -> dict:
    """
    Compute relative moment errors between a fitted proposal and a reference.

    Parameters
    ----------
    mu_K : np.ndarray, shape (d,)
        Fitted mean vector.
    Sigma_K : np.ndarray, shape (d, d)
        Fitted covariance matrix.
    mu_ex : np.ndarray, shape (d,)
        Reference mean vector.
    Sigma_ex : np.ndarray, shape (d, d)
        Reference covariance matrix.

    Returns
    -------
    dict with keys:
        'e_mu'    : float – relative mean error  ||mu_K - mu_ex||_2
                                                 / ||mu_ex||_2.
        'e_Sigma' : float – relative covariance error  ||Sigma_K - Sigma_ex||_F
                                                       / ||Sigma_ex||_F.
    """
    mu_K = np.asarray(mu_K, dtype=np.float64)
    Sigma_K = np.asarray(Sigma_K, dtype=np.float64)
    mu_ex = np.asarray(mu_ex, dtype=np.float64)
    Sigma_ex = np.asarray(Sigma_ex, dtype=np.float64)

    norm_mu_ex = float(np.linalg.norm(mu_ex))
    norm_Sigma_ex = float(np.linalg.norm(Sigma_ex, "fro"))

    e_mu = (
        float(np.linalg.norm(mu_K - mu_ex)) / norm_mu_ex
        if norm_mu_ex > 0.0
        else float(np.linalg.norm(mu_K - mu_ex))
    )
    e_Sigma = (
        float(np.linalg.norm(Sigma_K - Sigma_ex, "fro")) / norm_Sigma_ex
        if norm_Sigma_ex > 0.0
        else float(np.linalg.norm(Sigma_K - Sigma_ex, "fro"))
    )

    return {"e_mu": e_mu, "e_Sigma": e_Sigma}


# ---------------------------------------------------------------------------
# Weight-error and spurious-mass diagnostics
# ---------------------------------------------------------------------------

def compute_mean_abs_weight_error(
    q_K: np.ndarray,
    q_ex: np.ndarray,
) -> float:
    """
    Mean absolute difference between two tail-probability estimate arrays.

    Parameters
    ----------
    q_K : np.ndarray, shape (M,)
        COS (or other approximate) tail probability estimates.
    q_ex : np.ndarray, shape (M,)
        Exact convolution tail probability estimates.

    Returns
    -------
    float
        mean |q_K[m] - q_ex[m]| over m = 1, …, M.
    """
    q_K = np.asarray(q_K, dtype=np.float64)
    q_ex = np.asarray(q_ex, dtype=np.float64)
    return float(np.mean(np.abs(q_K - q_ex)))


def compute_spurious_mass_share(
    q_K_clipped: np.ndarray,
    q_ex: np.ndarray,
    threshold: float = 1e-12,
) -> float:
    """
    Fraction of total COS importance weight concentrated at states where the
    exact tail probability is negligible.

    A state m is *spurious* if q_ex[m] <= threshold.  The spurious mass share
    is the sum of COS weights at spurious states divided by the total COS
    weight.

    Parameters
    ----------
    q_K_clipped : np.ndarray, shape (M,)
        Clipped COS tail probability estimates (used as IS weights).
    q_ex : np.ndarray, shape (M,)
        Exact tail probability estimates.
    threshold : float, optional
        Exact-probability cutoff below which a state is considered spurious.
        Default: 1e-12.

    Returns
    -------
    float
        Spurious mass share in [0, 1]; 0.0 when total weight is zero.
    """
    q_K_clipped = np.asarray(q_K_clipped, dtype=np.float64)
    q_ex = np.asarray(q_ex, dtype=np.float64)

    total_weight = float(q_K_clipped.sum())
    if total_weight == 0.0:
        return 0.0

    spurious_weight = float(q_K_clipped[q_ex <= threshold].sum())
    return spurious_weight / total_weight
