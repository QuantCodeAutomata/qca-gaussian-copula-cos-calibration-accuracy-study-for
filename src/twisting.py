"""
Bernoulli exponential twisting and likelihood-ratio utilities.

Exponential twisting of a portfolio of independent Bernoulli obligors shifts
the mean loss to a target level.  For obligor n with default probability p_n
and loss-at-default l_n, the twisted probability at parameter theta is:

    p_n^theta = p_n * exp(theta * l_n) / (1 + p_n * (exp(theta * l_n) - 1))

The log-cumulant generating function (log-CGF) of the portfolio loss L under
the original measure is:

    psi(theta) = sum_n  log(1 + p_n * (exp(theta * l_n) - 1))

and the mean loss under the twisted measure is psi'(theta).

Likelihood-ratio functions are provided for:
  * Gaussian factor proposal  N(mu, Sigma) vs N(0, I_d).
  * Inverse-Gamma mixing-variable proposal IG(a_W, b_W) vs IG(alpha0, beta0).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# Bernoulli twisting
# ---------------------------------------------------------------------------

def twisted_probs(
    p_n: np.ndarray,
    l_n: np.ndarray,
    theta: float,
) -> np.ndarray:
    """
    Compute exponentially twisted Bernoulli default probabilities.

        p_n^theta = p_n * exp(theta * l_n) / (1 + p_n * (exp(theta*l_n) - 1))

    Parameters
    ----------
    p_n : np.ndarray, shape (N,)
        Conditional default probabilities under the original measure.
    l_n : np.ndarray, shape (N,)
        Loss-at-default values for each obligor.
    theta : float
        Twisting parameter.

    Returns
    -------
    np.ndarray, shape (N,)
        Twisted default probabilities, each in (0, 1).
    """
    p_n = np.asarray(p_n, dtype=np.float64)
    l_n = np.asarray(l_n, dtype=np.float64)
    # Numerically stable equivalent: p / (p + (1-p)*exp(-theta*l))
    # Avoids overflow of exp(theta*l) for large theta.
    # For l=0: exp(0)=1 → p/(p+(1-p))=p ✓
    return p_n / (p_n + (1.0 - p_n) * np.exp(-theta * l_n))


def log_cgf(
    theta: float,
    p_n: np.ndarray,
    l_n: np.ndarray,
) -> float:
    """
    Log cumulant generating function of the portfolio loss.

        psi(theta) = sum_n  log(1 + p_n * (exp(theta * l_n) - 1))

    Parameters
    ----------
    theta : float
        Twisting parameter.
    p_n : np.ndarray, shape (N,)
        Conditional default probabilities.
    l_n : np.ndarray, shape (N,)
        Loss-at-default values.

    Returns
    -------
    float
        psi(theta).
    """
    p_n = np.asarray(p_n, dtype=np.float64)
    l_n = np.asarray(l_n, dtype=np.float64)
    # Numerically stable form: log(1 + p*(e^{theta*l}-1))
    #   = theta*l + log(p + (1-p)*exp(-theta*l))
    # For l=0: 0 + log(1) = 0 ✓
    # For large theta,l: exp(-theta*l) → 0 so term → theta*l + log(p)  (finite)
    return float(
        np.sum(theta * l_n + np.log(p_n + (1.0 - p_n) * np.exp(-theta * l_n)))
    )


def twisted_mean(
    theta: float,
    p_n: np.ndarray,
    l_n: np.ndarray,
) -> float:
    """
    Expected portfolio loss under the twisted measure.

        E^theta[L] = sum_n  l_n * p_n^theta

    This equals psi'(theta), the derivative of the log-CGF.

    Parameters
    ----------
    theta : float
        Twisting parameter.
    p_n : np.ndarray, shape (N,)
        Conditional default probabilities.
    l_n : np.ndarray, shape (N,)
        Loss-at-default values.

    Returns
    -------
    float
        Mean portfolio loss under the twisted measure.
    """
    l_n = np.asarray(l_n, dtype=np.float64)
    return float(np.dot(l_n, twisted_probs(p_n, l_n, theta)))


def solve_theta(
    target_loss: float,
    p_n: np.ndarray,
    l_n: np.ndarray,
    theta_max: float = 100.0,
) -> float:
    """
    Solve for the twisting parameter theta that shifts the mean loss to a
    specified target level.

    Uses Brent's method on the interval [0, theta_max] to find theta such
    that twisted_mean(theta, p_n, l_n) == target_loss.

    Parameters
    ----------
    target_loss : float
        Desired mean loss under the twisted measure (typically the rare-event
        threshold x).
    p_n : np.ndarray, shape (N,)
        Conditional default probabilities.
    l_n : np.ndarray, shape (N,)
        Loss-at-default values.
    theta_max : float, optional
        Upper bracket for the search.  Default: 100.0.

    Returns
    -------
    float
        Optimal twisting parameter theta*.

    Raises
    ------
    ValueError
        If target_loss exceeds the maximum attainable twisted mean at
        theta_max, or if target_loss <= 0.
    """
    p_n = np.asarray(p_n, dtype=np.float64)
    l_n = np.asarray(l_n, dtype=np.float64)

    max_mean = twisted_mean(theta_max, p_n, l_n)
    if target_loss > max_mean:
        raise ValueError(
            f"target_loss={target_loss} exceeds maximum attainable twisted "
            f"mean {max_mean:.6g} at theta_max={theta_max}."
        )

    def objective(theta: float) -> float:
        return twisted_mean(theta, p_n, l_n) - target_loss

    theta_star = brentq(objective, 0.0, theta_max, xtol=1e-12, rtol=1e-12)
    return float(theta_star)


# ---------------------------------------------------------------------------
# Likelihood-ratio functions
# ---------------------------------------------------------------------------

def log_lr_gaussian(
    z: np.ndarray,
    mu: np.ndarray,
    Sigma_inv: np.ndarray,
    log_det_Sigma: float,
) -> float:
    """
    Log likelihood ratio of N(mu, Sigma) with respect to N(0, I_d).

        log R_Z(z) = 0.5 * (z^T z - (z-mu)^T Sigma^{-1} (z-mu) - log|Sigma|)

    Parameters
    ----------
    z : np.ndarray, shape (d,)
        Factor realisation.
    mu : np.ndarray, shape (d,)
        Mean of the Gaussian proposal.
    Sigma_inv : np.ndarray, shape (d, d)
        Pre-computed inverse of the proposal covariance.
    log_det_Sigma : float
        Pre-computed log-determinant of the proposal covariance, log|Sigma|.

    Returns
    -------
    float
        log[N(z; mu, Sigma) / N(z; 0, I_d)].
    """
    z = np.asarray(z, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    diff = z - mu                                              # (d,)
    quad_prop = float(diff @ Sigma_inv @ diff)                 # (z-mu)^T Sig^{-1}(z-mu)
    quad_prior = float(np.dot(z, z))                           # z^T z
    return 0.5 * (quad_prior - quad_prop - log_det_Sigma)


def log_lr_invgamma(
    w: float,
    alpha_prop: float,
    beta_prop: float,
    alpha0: float = 2.0,
    beta0: float = 2.0,
) -> float:
    """
    Log likelihood ratio of IG(alpha_prop, beta_prop) with respect to
    the prior IG(alpha0, beta0).

    Using the parameterisation
        f(w; alpha, beta) = beta^alpha / Gamma(alpha) * w^{-alpha-1}
                            * exp(-beta/w),
    the log ratio is:

        log[f(w; alpha_prop, beta_prop) / f(w; alpha0, beta0)]
          = alpha_prop * log(beta_prop) - lgamma(alpha_prop)
            - alpha0 * log(beta0) + lgamma(alpha0)
            + (alpha0 - alpha_prop) * log(w)
            + (beta0 - beta_prop) / w

    Parameters
    ----------
    w : float
        Realised inverse-Gamma variable (w > 0).
    alpha_prop : float
        Shape parameter of the proposal distribution.
    beta_prop : float
        Scale parameter of the proposal distribution.
    alpha0 : float, optional
        Shape parameter of the prior distribution.  Default: 2.0.
    beta0 : float, optional
        Scale parameter of the prior distribution.  Default: 2.0.

    Returns
    -------
    float
        log[IG(w; alpha_prop, beta_prop) / IG(w; alpha0, beta0)].
    """
    log_ratio = (
        alpha_prop * np.log(beta_prop)
        - gammaln(alpha_prop)
        - alpha0 * np.log(beta0)
        + gammaln(alpha0)
        + (alpha0 - alpha_prop) * np.log(w)
        + (beta0 - beta_prop) / w
    )
    return float(log_ratio)
