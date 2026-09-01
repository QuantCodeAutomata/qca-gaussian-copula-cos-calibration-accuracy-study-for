"""
Inverse-Gamma parameter fitting from weighted sufficient statistics.

For the Student-t credit model, the scale-mixing variable W follows an
inverse-Gamma distribution:

    f(w; alpha, beta) = beta^alpha / Gamma(alpha) * w^{-alpha-1} * exp(-beta/w)

Given importance-weighted pilot draws (W_m, omega_m), we match the following
weighted sufficient statistics to their theoretical expectations:

    m_log  = (1/S) * sum_m omega_m * log(W_m)   ≈  E[log W] = log(beta) - psi(alpha)
    m_neg1 = (1/S) * sum_m omega_m * W_m^{-1}   ≈  E[W^{-1}] = alpha / beta

where S = sum omega_m and psi denotes the digamma function.

The parameter alpha is the unique solution of:

    psi(alpha) - log(alpha) + m_log + log(m_neg1) = 0

and then beta = alpha / m_neg1.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import digamma
from scipy.stats import invgamma as sp_invgamma


# ---------------------------------------------------------------------------
# Sufficient statistics
# ---------------------------------------------------------------------------

def compute_invgamma_sufficient_stats(
    W: np.ndarray,
    weights: np.ndarray,
) -> dict:
    """
    Compute weighted sufficient statistics for inverse-Gamma fitting.

    Parameters
    ----------
    W : np.ndarray, shape (M,)
        Realised scale-mixing variable draws.
    weights : np.ndarray, shape (M,)
        Non-negative importance weights (calibration weights).

    Returns
    -------
    dict with keys:
        'm_log'  : float – weighted mean of log(W):  (1/S) * sum(w * log(W)).
        'm_neg1' : float – weighted mean of W^{-1}:  (1/S) * sum(w * W^{-1}).
        'S'      : float – sum of weights.
    """
    W = np.asarray(W, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    S = float(weights.sum())
    if S == 0.0:
        return {"m_log": np.nan, "m_neg1": np.nan, "S": 0.0}

    m_log = float(weights @ np.log(W)) / S
    m_neg1 = float(weights @ (1.0 / W)) / S

    return {"m_log": m_log, "m_neg1": m_neg1, "S": S}


# ---------------------------------------------------------------------------
# MOM parameter fitting
# ---------------------------------------------------------------------------

def fit_invgamma_params(
    m_log: float,
    m_neg1: float,
) -> tuple[float, float]:
    """
    Fit inverse-Gamma parameters by matching weighted sufficient statistics.

    Solves for alpha from the method-of-moments equation:

        g(alpha) = psi(alpha) - log(alpha) + m_log + log(m_neg1) = 0

    where psi is the digamma function.  The scale parameter is then:

        beta = alpha / m_neg1.

    Parameters
    ----------
    m_log : float
        Weighted mean of log(W).
    m_neg1 : float
        Weighted mean of W^{-1}  (must be strictly positive).

    Returns
    -------
    alpha : float
        Fitted shape parameter (> 0).
    beta : float
        Fitted scale parameter (> 0).

    Raises
    ------
    ValueError
        If the root-finding algorithm fails to bracket a root.
    """
    constant = m_log + np.log(m_neg1)   # fixed offset in the equation

    def equation(alpha: float) -> float:
        return digamma(alpha) - np.log(alpha) + constant

    # Bracket check:
    # As alpha → 0+: psi(alpha) ≈ -1/alpha → -inf, log(alpha) → -inf,
    #   psi(alpha) - log(alpha) → +inf  → equation > 0.
    # As alpha → +inf: psi(alpha) - log(alpha) → 0-  → equation → constant.
    # A root exists when constant < 0, which holds for the targeted regime.
    lo, hi = 1e-6, 1e6
    f_lo = equation(lo)
    f_hi = equation(hi)

    if f_lo * f_hi > 0.0:
        raise ValueError(
            f"Could not bracket root for alpha: g({lo})={f_lo:.4g}, "
            f"g({hi})={f_hi:.4g}.  Check m_log={m_log:.4g}, "
            f"m_neg1={m_neg1:.4g}."
        )

    alpha = float(brentq(equation, lo, hi, xtol=1e-12, rtol=1e-12))
    beta = alpha / m_neg1
    return alpha, beta


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_invgamma(
    alpha: float,
    beta: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample from InvGamma(alpha, beta) with density:

        f(w; alpha, beta) = beta^alpha / Gamma(alpha) * w^{-alpha-1}
                            * exp(-beta/w)

    Uses scipy.stats.invgamma(a=alpha, scale=beta).

    Parameters
    ----------
    alpha : float
        Shape parameter (> 0).
    beta : float
        Scale parameter (> 0).
    size : int
        Number of samples to draw.
    rng : np.random.Generator
        NumPy default-RNG instance for reproducibility.

    Returns
    -------
    np.ndarray, shape (size,)
        Samples from InvGamma(alpha, beta), dtype float64.
    """
    samples = sp_invgamma.rvs(
        a=alpha, scale=beta, size=size, random_state=rng
    )
    return np.asarray(samples, dtype=np.float64)
