"""
Synthetic Gaussian / Student-t credit portfolio model.

The portfolio consists of 100 obligors arranged in 10 homogeneous blocks of
10 obligors each.  Every obligor has the same unconditional default probability
p0 = 0.01.  Block-level losses-at-default are (1,1,4,4,9,9,16,16,25,25).

Factor structure
----------------
d = 11 Gaussian systematic factors:
  * z[0]  – market-wide factor
  * z[g]  – block-specific factor for block g  (g = 1, …, 10, 1-indexed)

Conditional default probabilities
----------------------------------
Gaussian model (for block g, 1-indexed):
  p_g(z) = Phi( (Phi^{-1}(p0) + 0.3*z[0] + 0.8*z[g]) / sqrt(0.27) )

Student-t model (scale mixture of Gaussians with W ~ InvGamma(2, 2)):
  p_g(z, w) = Phi( (T4^{-1}(p0)/sqrt(w) - 0.3*z[0] - 0.8*z[g]) / sqrt(0.27) )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from scipy.stats import norm, t as t_dist, invgamma


@dataclass
class Portfolio:
    """
    Synthetic credit portfolio with 100 obligors in 10 homogeneous blocks.

    Parameters
    ----------
    n_blocks : int
        Number of homogeneous blocks (default 10).
    n_per_block : int
        Number of obligors in each block (default 10).
    block_losses : tuple of int
        Loss-at-default per obligor for each block,
        shape (n_blocks,).  Default: (1,1,4,4,9,9,16,16,25,25).
    p0 : float
        Unconditional single-obligor default probability (default 0.01).
    d : int
        Total number of systematic factors (default 11: 1 market + 10 block).
    L_max : int
        Maximum attainable portfolio loss (default 1100).
    """

    n_blocks: int = 10
    n_per_block: int = 10
    block_losses: Tuple[int, ...] = (1, 1, 4, 4, 9, 9, 16, 16, 25, 25)
    p0: float = 0.01
    d: int = 11
    L_max: int = 1100

    # Pre-computed scalar constants – excluded from __init__ and __repr__
    _phi_inv_p0: float = field(init=False, repr=False)
    _t4_inv_p0: float = field(init=False, repr=False)
    _block_losses_arr: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._phi_inv_p0 = float(norm.ppf(self.p0))           # Phi^{-1}(0.01)
        self._t4_inv_p0 = float(t_dist.ppf(self.p0, df=4))    # T_4^{-1}(0.01)
        self._block_losses_arr = np.array(self.block_losses, dtype=np.float64)

    # ------------------------------------------------------------------
    # Conditional default probabilities
    # ------------------------------------------------------------------

    def conditional_block_probs_gaussian(self, z: np.ndarray) -> np.ndarray:
        """
        Gaussian conditional default probability for every block.

        For block g (1-indexed):
            p_g(z) = Phi( (Phi^{-1}(p0) + 0.3*z[0] + 0.8*z[g]) / sqrt(0.27) )

        Parameters
        ----------
        z : np.ndarray, shape (11,)
            Systematic factor realisation.  z[0] is the market factor;
            z[1] … z[10] are the block-specific factors.

        Returns
        -------
        np.ndarray, shape (10,)
            Conditional default probability for each of the 10 blocks.
        """
        z = np.asarray(z, dtype=np.float64)
        block_factors = z[1:]                                    # shape (10,)
        args = (
            self._phi_inv_p0 + 0.3 * z[0] + 0.8 * block_factors
        ) / np.sqrt(0.27)
        return norm.cdf(args)

    def conditional_block_probs_student_t(
        self, z: np.ndarray, w: float
    ) -> np.ndarray:
        """
        Student-t conditional default probability for every block.

        For block g (1-indexed):
            p_g(z,w) = Phi( (T4^{-1}(p0)/sqrt(w) - 0.3*z[0] - 0.8*z[g])
                            / sqrt(0.27) )

        Parameters
        ----------
        z : np.ndarray, shape (11,)
            Systematic factor realisation.
        w : float
            Realised inverse-Gamma scale-mixing variable (W > 0).

        Returns
        -------
        np.ndarray, shape (10,)
            Conditional default probability for each of the 10 blocks.
        """
        z = np.asarray(z, dtype=np.float64)
        block_factors = z[1:]                                    # shape (10,)
        args = (
            self._t4_inv_p0 / np.sqrt(w) - 0.3 * z[0] - 0.8 * block_factors
        ) / np.sqrt(0.27)
        return norm.cdf(args)

    # ------------------------------------------------------------------
    # Pilot sampling
    # ------------------------------------------------------------------

    def sample_pilot_gaussian(self, M0: int, seed: int) -> np.ndarray:
        """
        Draw M0 i.i.d. pilot factor samples from N(0, I_d).

        Parameters
        ----------
        M0 : int
            Number of pilot draws.
        seed : int
            Integer seed for the numpy default RNG (reproducibility).

        Returns
        -------
        np.ndarray, shape (M0, 11)
            Standard Gaussian factor draws, dtype float64.
        """
        rng = np.random.default_rng(seed)
        return rng.standard_normal(size=(M0, self.d)).astype(np.float64)

    def sample_pilot_student_t(
        self, M0: int, seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw M0 i.i.d. pilot samples for the Student-t model.

        Draws Z ~ N(0, I_11) and W ~ InvGamma(alpha=2, beta=2) independently.
        The InvGamma parameterisation used throughout is:
            f(w; alpha, beta) ∝ w^{-alpha-1} exp(-beta/w)
        which corresponds to scipy.stats.invgamma(a=alpha, scale=beta).

        Parameters
        ----------
        M0 : int
            Number of pilot draws.
        seed : int
            Integer seed for the numpy default RNG.

        Returns
        -------
        Z : np.ndarray, shape (M0, 11)
            Standard Gaussian factor draws.
        W : np.ndarray, shape (M0,)
            Inverse-Gamma scale-mixing variable draws.
        """
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal(size=(M0, self.d)).astype(np.float64)
        # InvGamma(alpha=2, beta=2) ↔ scipy.stats.invgamma(a=2, scale=2)
        W = invgamma.rvs(
            a=2, scale=2, size=M0, random_state=rng
        ).astype(np.float64)
        return Z, W
