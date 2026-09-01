"""
Filtered COS method for conditional CDF / tail probability approximation.

The COS (Fourier COSine) method approximates the CDF of the portfolio loss L
conditional on a systematic factor draw z by expanding the density in a
truncated Fourier cosine series on the interval [a, b] and applying an
exponential filter to control the Gibbs phenomenon.

Interval
--------
[a, b] = [-0.5, 1100.5]

Filter
------
EXP-4 filter:  sigma_k = exp(-8 * (k/K)^4)   for k = 1, …, K-1.

Left-limit evaluation point
----------------------------
Gaussian event  {L >= 250}: y_x = 249.5  (left limit of x = 250)
Student-t event {L >= 504}: y_x = 503.5  (left limit of x = 504)

COS CDF approximation
---------------------
For frequency omega_k = k * pi / (b - a) and filter sigma_k:

  F^sigma(y) = (y - a)/(b - a)
             + (2/pi) * sum_{k=1}^{K-1}
               (sigma_k / k)
               * Re{ phi(omega_k) * exp(-i * omega_k * a) }
               * sin(k * pi * (y - a) / (b - a))

where the characteristic function of L | Z=z is

  phi(omega) = prod_{g=1}^{10} (1 - p_g + p_g * exp(i*omega*c_g))^10.

Raw and clipped tail probabilities
-----------------------------------
  q_tilde = 1 - F^sigma(y_x)          (raw; may fall outside [0, 1])
  q       = clip(q_tilde, 0, 1)       (production estimate)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Characteristic function
# ---------------------------------------------------------------------------

def compute_char_func_batch(
    probs_batch: np.ndarray,
    block_losses: np.ndarray,
    omega_k: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the portfolio-loss characteristic function at K-1 frequencies
    for a batch of M conditional probability vectors.

    The characteristic function for the conditional loss distribution is:

        phi(omega) = prod_{g=1}^{10}
                     (1 - p_g + p_g * exp(i * omega * c_g))^{n_per_block}

    where n_per_block = 10 (block size fixed by the portfolio model).

    Parameters
    ----------
    probs_batch : np.ndarray, shape (M, 10)
        Conditional block default probabilities for M factor draws.
    block_losses : np.ndarray, shape (10,)
        Block losses-at-default c_g.
    omega_k : np.ndarray, shape (K-1,)
        Evaluation frequencies omega_k = k * pi / (b - a), k = 1, …, K-1.

    Returns
    -------
    np.ndarray, shape (M, K-1), dtype complex128
        phi[m, k] = phi(omega_k[k] | z_m).
    """
    probs_batch = np.asarray(probs_batch, dtype=np.float64)    # (M, 10)
    block_losses = np.asarray(block_losses, dtype=np.float64)  # (10,)
    omega_k = np.asarray(omega_k, dtype=np.float64)            # (K-1,)
    n_per_block = 10

    # exp(i * omega_k[k] * c_g[g])  →  shape (K-1, 10)
    exp_term = np.exp(
        1j * omega_k[:, None] * block_losses[None, :]
    )                                                          # (K-1, 10)

    # p_g values broadcast to (M, K-1, 10)
    p = probs_batch[:, None, :]                                # (M, 1, 10)

    # Summand inside the product for each block
    # term[m, k, g] = 1 - p_g(z_m) + p_g(z_m) * exp(i * omega_k[k] * c_g)
    term = 1.0 - p + p * exp_term[None, :, :]                 # (M, K-1, 10)

    # Raise each block's term to n_per_block and take the product over blocks
    phi = np.prod(term ** n_per_block, axis=-1)                # (M, K-1)
    return phi.astype(np.complex128)


# ---------------------------------------------------------------------------
# COS tail probability
# ---------------------------------------------------------------------------

def compute_cos_tail_batch(
    probs_batch: np.ndarray,
    block_losses: np.ndarray,
    K: int,
    y_x: float,
    a: float,
    b: float,
    batch_size: int = 5000,
) -> dict:
    """
    Compute filtered COS tail probabilities for a batch of factor draws.

    For each draw m the method evaluates:

        q_tilde_m = 1 - F^sigma(y_x | z_m)
        q_m       = clip(q_tilde_m, 0, 1)

    Parameters
    ----------
    probs_batch : np.ndarray, shape (M, 10)
        Conditional block default probabilities.
    block_losses : np.ndarray, shape (10,)
        Block losses-at-default.
    K : int
        Number of COS terms (frequencies k = 1, …, K-1 are used).
    y_x : float
        Left-limit evaluation point for the CDF.
        Use 249.5 for the Gaussian event {L >= 250} and
        503.5 for the Student-t event {L >= 504}.
    a : float
        Left endpoint of the truncation interval (typically -0.5).
    b : float
        Right endpoint of the truncation interval (typically 1100.5).
    batch_size : int, optional
        Number of draws processed per pass.  Default: 5000.

    Returns
    -------
    dict with keys:
        'q_tilde'      : np.ndarray, shape (M,) – raw tail probabilities.
        'q_clipped'    : np.ndarray, shape (M,) – clipped to [0, 1].
        'frac_below_0' : float – fraction of q_tilde values < 0.
        'frac_above_1' : float – fraction of q_tilde values > 1.
        'frac_clipped' : float – fraction that needed any clipping.
    """
    probs_batch = np.asarray(probs_batch, dtype=np.float64)
    block_losses = np.asarray(block_losses, dtype=np.float64)
    M = probs_batch.shape[0]
    span = b - a                                               # b - a

    # Frequency array and filter (both shape (K-1,))
    k_arr = np.arange(1, K, dtype=np.float64)                 # [1, 2, …, K-1]
    omega_k = k_arr * np.pi / span                            # (K-1,)
    sigma_k = np.exp(-8.0 * (k_arr / K) ** 4)                 # EXP-4 filter

    # Pre-compute scalars that do not depend on the batch
    sin_term = np.sin(k_arr * np.pi * (y_x - a) / span)       # (K-1,)
    exp_neg_ia = np.exp(-1j * omega_k * a)                     # (K-1,), complex
    # Coefficient vector: (sigma_k / k) * sin_term, shape (K-1,)
    coeff = (sigma_k / k_arr) * sin_term                       # (K-1,)

    q_tilde = np.empty(M, dtype=np.float64)

    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        batch = probs_batch[start:end]                         # (B, 10)

        # Characteristic function: shape (B, K-1)
        phi = compute_char_func_batch(batch, block_losses, omega_k)

        # Re{ phi * exp(-i * omega_k * a) }  →  shape (B, K-1)
        re_phi_exp = np.real(phi * exp_neg_ia[None, :])

        # COS CDF at y_x:
        # F^sigma(y_x) = (y_x - a)/span + (2/pi) * sum_k coeff[k] * re_phi_exp[m,k]
        cos_sum = re_phi_exp @ coeff                           # (B,)
        F_sigma = (y_x - a) / span + (2.0 / np.pi) * cos_sum  # (B,)

        q_tilde[start:end] = 1.0 - F_sigma

    q_clipped = np.clip(q_tilde, 0.0, 1.0)

    frac_below_0 = float(np.mean(q_tilde < 0.0))
    frac_above_1 = float(np.mean(q_tilde > 1.0))
    frac_clipped = float(np.mean((q_tilde < 0.0) | (q_tilde > 1.0)))

    return {
        "q_tilde": q_tilde,
        "q_clipped": q_clipped,
        "frac_below_0": frac_below_0,
        "frac_above_1": frac_above_1,
        "frac_clipped": frac_clipped,
    }
