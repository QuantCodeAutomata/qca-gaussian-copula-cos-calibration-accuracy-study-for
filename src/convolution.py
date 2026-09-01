"""
Exact block-convolution tail probability for the credit portfolio.

Given a factor realisation z and its induced conditional block default
probabilities, this module computes

    q^ex(z) = P(L >= threshold | Z = z)

where the portfolio loss is

    L = sum_{g=1}^{10}  c_g * D_g,   D_g | Z=z ~ Binomial(10, p_g(z)),

via dynamic-programming convolution over the 10 independent blocks.

Algorithm
---------
The DP state is a probability-mass vector over loss values 0, 1, …,
threshold-1, plus a scalar *tail accumulator* for the event {L >= threshold}.

For each block g with n_per_block=10 obligors, loss c_g per default, and
conditional default probability p_g(z):

1.  Compute the Binomial(10, p_g) PMF on {0, 1, …, 10}.
2.  For every current loss l in 0..threshold-1 and every default count k=0..10:
        new_loss = l + k * c_g
        if new_loss < threshold:  new_mass[new_loss] += mass[l] * binom_pmf[k]
        else:                     tail_accum         += mass[l] * binom_pmf[k]
3.  Any probability already in the tail stays in the tail.

The returned value is the tail accumulator after processing all 10 blocks.
"""

from __future__ import annotations

import numpy as np
from scipy.special import comb


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _binom_pmf_batch(
    p: np.ndarray,
    n: int,
    k_vals: np.ndarray,
    binom_coeffs: np.ndarray,
) -> np.ndarray:
    """
    Compute Binomial(n, p) PMF for every k in k_vals, for a batch of M
    probability values.

    Parameters
    ----------
    p : np.ndarray, shape (M,)
        Success probabilities (one per draw in the batch).
    n : int
        Number of trials (n_per_block = 10).
    k_vals : np.ndarray, shape (n+1,)
        Default counts [0, 1, …, n].
    binom_coeffs : np.ndarray, shape (n+1,)
        Pre-computed binomial coefficients C(n, k).

    Returns
    -------
    np.ndarray, shape (M, n+1)
        PMF values; entry [m, k] = C(n,k) * p[m]^k * (1-p[m])^(n-k).
    """
    # Shapes: p -> (M, 1),  k_vals -> (1, n+1)
    p_col = p[:, None]                                         # (M, 1)
    q_col = (1.0 - p)[:, None]                                 # (M, 1)
    return (
        binom_coeffs[None, :]                                  # (1, n+1)
        * (p_col ** k_vals[None, :])                           # (M, n+1)
        * (q_col ** (n - k_vals)[None, :])                     # (M, n+1)
    ).astype(np.float64)


def _convolve_block_batch(
    mass: np.ndarray,
    tail_accum: np.ndarray,
    binom_probs: np.ndarray,
    c_g: int,
    threshold: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform the DP convolution step for one block over a batch of M draws.

    Parameters
    ----------
    mass : np.ndarray, shape (M, threshold)
        Current probability-mass vector for losses 0..threshold-1.
    tail_accum : np.ndarray, shape (M,)
        Current tail accumulator (probability of L >= threshold so far).
    binom_probs : np.ndarray, shape (M, n+1)
        Binomial PMF for the current block, for each draw in the batch.
    c_g : int
        Loss per default for the current block.
    threshold : int
        Tail threshold (mass at loss >= threshold goes to the tail).

    Returns
    -------
    new_mass : np.ndarray, shape (M, threshold)
    tail_accum : np.ndarray, shape (M,)  (updated in-place copy)
    """
    n_outcomes = binom_probs.shape[1]                          # n_per_block + 1
    new_mass = np.zeros_like(mass)

    for k in range(n_outcomes):
        shift = k * c_g
        bp = binom_probs[:, k]                                 # shape (M,)

        if shift >= threshold:
            # All current non-tail mass moves to tail.
            tail_accum = tail_accum + bp * mass.sum(axis=1)
        else:
            # Portion with l in [0, threshold-shift-1] → new_mass[shift:]
            new_mass[:, shift:] += bp[:, None] * mass[:, : threshold - shift]
            # Portion with l in [threshold-shift, threshold-1] → tail
            if shift > 0:
                tail_accum = (
                    tail_accum + bp * mass[:, threshold - shift :].sum(axis=1)
                )
            # (shift == 0: mass[:, threshold:] is empty; nothing added to tail)

    return new_mass, tail_accum


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_exact_tail_single(
    probs: np.ndarray,
    block_losses: np.ndarray,
    threshold: int,
) -> float:
    """
    Compute the exact conditional tail probability for a single factor draw.

    P(L >= threshold | Z = z) is obtained by DP convolution over all blocks.

    Parameters
    ----------
    probs : np.ndarray, shape (10,)
        Conditional block default probabilities p_g(z) for g = 1, …, 10.
    block_losses : np.ndarray, shape (10,)
        Block losses-at-default c_g for g = 1, …, 10.
    threshold : int
        Tail threshold; typically 250 for the Gaussian event {L >= 250}.

    Returns
    -------
    float
        P(L >= threshold | z).
    """
    probs_2d = probs.reshape(1, -1).astype(np.float64)
    result = compute_exact_tail_batch(probs_2d, block_losses, threshold, batch_size=1)
    return float(result[0])


def compute_exact_tail_batch(
    probs_batch: np.ndarray,
    block_losses: np.ndarray,
    threshold: int,
    batch_size: int = 5000,
) -> np.ndarray:
    """
    Compute exact conditional tail probabilities for a batch of factor draws.

    Parameters
    ----------
    probs_batch : np.ndarray, shape (M, 10)
        Conditional block default probabilities for every pilot draw.
    block_losses : np.ndarray, shape (10,)
        Block losses-at-default c_g for g = 1, …, 10.
    threshold : int
        Tail threshold; typically 250.
    batch_size : int, optional
        Number of draws to process in a single pass (memory control).
        Default: 5000.

    Returns
    -------
    np.ndarray, shape (M,)
        Exact tail probability q^ex(z_m) = P(L >= threshold | Z = z_m) for
        each pilot draw m = 1, …, M.
    """
    probs_batch = np.asarray(probs_batch, dtype=np.float64)
    block_losses = np.asarray(block_losses, dtype=np.int64)
    M = probs_batch.shape[0]
    n_blocks = probs_batch.shape[1]
    n_per_block = 10

    k_vals = np.arange(n_per_block + 1, dtype=np.float64)     # [0, 1, …, 10]
    binom_coeffs = comb(n_per_block, k_vals, exact=False)      # shape (11,)

    output = np.empty(M, dtype=np.float64)

    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        batch = probs_batch[start:end]                         # (B, 10)
        B = batch.shape[0]

        # Initialise DP state: all mass at loss = 0
        mass = np.zeros((B, threshold), dtype=np.float64)
        mass[:, 0] = 1.0
        tail_accum = np.zeros(B, dtype=np.float64)

        for g in range(n_blocks):
            p_g = batch[:, g]                                  # shape (B,)
            c_g = int(block_losses[g])
            binom_probs = _binom_pmf_batch(p_g, n_per_block, k_vals, binom_coeffs)
            mass, tail_accum = _convolve_block_batch(
                mass, tail_accum, binom_probs, c_g, threshold
            )

        output[start:end] = tail_accum

    return output
