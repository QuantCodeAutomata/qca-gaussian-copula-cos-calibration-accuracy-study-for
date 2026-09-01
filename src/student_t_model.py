"""
Student-t copula model utilities for the benchmark credit portfolio.

Portfolio specification
-----------------------
* 100 obligors in 10 homogeneous blocks of 10 obligors each.
* Unconditional default probability p0 = 0.01.
* Block losses-at-default: (1, 1, 4, 4, 9, 9, 16, 16, 25, 25).
* Common state U = (Z, W): Z ~ N_11(0, I),  W ~ InvGamma(alpha0, beta0).
* nu = 4 degrees of freedom.

Conditional default probability (block g, 1-indexed):
    p_g(z, w) = Phi( (T_4^{-1}(0.01)/sqrt(w) - 0.3*z[0] - 0.8*z[g])
                     / sqrt(0.27) )
where z is 0-indexed (z[0] = market factor, z[g] = block-g factor).

T_4^{-1}(0.01) is obtained from scipy.stats.t.ppf(0.01, df=4).

InvGamma parameterisation used throughout:
    f(w; alpha, beta) ∝ w^{-alpha-1} exp(-beta/w)
which corresponds to scipy.stats.invgamma(a=alpha, scale=beta).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, t as t_dist, invgamma


# ---------------------------------------------------------------------------
# Module-level constant
# ---------------------------------------------------------------------------

_T4_INV_P0: float = float(t_dist.ppf(0.01, df=4))   # T_4^{-1}(0.01)


# ---------------------------------------------------------------------------
# Batch conditional block probabilities
# ---------------------------------------------------------------------------

def conditional_block_probs_t(
    Z: np.ndarray,
    W: np.ndarray | float,
    n_blocks: int = 10,
) -> np.ndarray:
    """
    Compute Student-t conditional block default probabilities (batch-capable).

    For block g (1-indexed), g = 1, ..., n_blocks:
        p_g(z, w) = Phi( (T_4^{-1}(0.01)/sqrt(w) - 0.3*z[0] - 0.8*z[g])
                         / sqrt(0.27) )
    with z 0-indexed (z[0] = market factor, z[g] = block-g factor).

    Parameters
    ----------
    Z : np.ndarray, shape (M, 11) or (11,)
        Systematic factor realisations.  Z[..., 0] is the market factor;
        Z[..., 1:n_blocks+1] are block-specific factors.
    W : np.ndarray, shape (M,) or scalar
        Realised inverse-Gamma scale-mixing variable (W > 0).
    n_blocks : int, optional
        Number of homogeneous blocks.  Default: 10.

    Returns
    -------
    np.ndarray, shape (M, n_blocks) or (n_blocks,)
        Conditional default probability for each block.
        Shape matches the leading dimension of Z (or scalar if Z is 1-D).
    """
    Z = np.asarray(Z, dtype=np.float64)
    scalar_input = Z.ndim == 1
    if scalar_input:
        Z = Z[None, :]                                   # (1, 11)

    W = np.asarray(W, dtype=np.float64).ravel()
    if W.size == 1:
        W = np.broadcast_to(W, Z.shape[0])              # broadcast scalar W

    market = Z[:, 0]                                     # (M,)
    blocks = Z[:, 1 : n_blocks + 1]                      # (M, n_blocks)

    # (T_4^{-1}(0.01)/sqrt(w) - 0.3*z[0] - 0.8*z[g]) / sqrt(0.27)
    t4_over_sqrtw = _T4_INV_P0 / np.sqrt(W)             # (M,)
    args = (
        t4_over_sqrtw[:, None]
        - 0.3 * market[:, None]
        - 0.8 * blocks
    ) / np.sqrt(0.27)                                    # (M, n_blocks)

    probs = norm.cdf(args)

    if scalar_input:
        return probs[0]                                  # (n_blocks,)
    return probs                                         # (M, n_blocks)


# ---------------------------------------------------------------------------
# Pilot sampling
# ---------------------------------------------------------------------------

def sample_pilot_student_t(
    M0: int,
    seed: int,
    alpha0: float = 2.0,
    beta0: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw M0 i.i.d. pilot samples for the Student-t model.

    Draws Z ~ N_11(0, I) and W ~ InvGamma(alpha0, beta0) independently using
    a single seeded NumPy default RNG for full reproducibility.

    Parameters
    ----------
    M0 : int
        Number of pilot draws.
    seed : int
        Integer seed for the NumPy default RNG.
    alpha0 : float, optional
        Shape parameter of the InvGamma prior.  Default: 2.0.
    beta0 : float, optional
        Scale parameter of the InvGamma prior.  Default: 2.0.

    Returns
    -------
    Z : np.ndarray, shape (M0, 11)
        Standard Gaussian factor draws, dtype float64.
    W : np.ndarray, shape (M0,)
        Inverse-Gamma scale-mixing variable draws, dtype float64.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(M0, 11)).astype(np.float64)
    W = invgamma.rvs(
        a=alpha0, scale=beta0, size=M0, random_state=rng
    ).astype(np.float64)
    return Z, W
