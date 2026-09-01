"""
Experiment 3: Student-t Copula Matched-Budget Production Comparison.

Extends Experiment 2 to the Student-t factor model where the scale-mixing
variable W ~ InvGamma(2, 2) is part of the common state U = (Z, W).

Portfolio: same 100 obligors in 10 blocks, threshold x = 504.
Conditional default probability (block g, 1-indexed):
    p_g(z, w) = Phi( (T_4^{-1}(0.01)/sqrt(w) - 0.3*z[0] - 0.8*z[g])
                     / sqrt(0.27) )

Calibration methods
-------------------
CEIS-t  – binary indicator  omega_m = 1_{L_m >= 504} from one conditional
           loss simulation per (z_m, w_m) pilot draw.
ISCOS-t – filtered COS tail probability (K=64, EXP-4 filter) as weight.

Fitted proposal parameters
---------------------------
Both methods fit:
  Z part : mu (d=11,) and Sigma (11x11) via weighted Gaussian moments.
  W part : alpha_W, beta_W via InvGamma MOM (weighted sufficient statistics).

Production events (M = 250000 each)
-------------------------------------
A1: exact-level  {L = 504},  seed 420
A2: threshold-tail  {L >= 504},  seed 421

Full LR:
  log Lambda = log R_Z(z) + log R_W(w) + psi(theta, p_n) - theta * L

where
  log R_Z(z) = log[N(z;0,I) / N(z;mu,Sigma)]
  log R_W(w) = log[IG(w;alpha0,beta0) / IG(w;alpha_W,beta_W)]
             = -log_lr_invgamma(w, alpha_W, beta_W, alpha0=2, beta0=2)

Outputs (in results/)
---------------------
  exp3_calibration_table.csv
  exp3_production_table.csv
  exp3_contribution_halfwidths.png
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.portfolio import Portfolio
from src.cos import compute_cos_tail_batch
from src.calibration import fit_gaussian_moments
from src.invgamma_fit import (
    compute_invgamma_sufficient_stats,
    fit_invgamma_params,
    sample_invgamma,
)
from src.diagnostics import compute_contribution_estimates, compute_production_ess
from src.student_t_model import conditional_block_probs_t, sample_pilot_student_t


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

M0 = 250_000
M_PROD = 250_000
SEED_PILOT = 42
SEED_A1 = 420
SEED_A2 = 421
TARGET_LOSS = 504
K_ISCOS = 64
ALPHA0 = 2.0
BETA0 = 2.0
THETA_MAX = 100.0
BISECT_ITERS = 80
PROD_BATCH = 10_000
A_COS = -0.5
B_COS = 1100.5
Y_X_T = 503.5               # left-limit for Student-t threshold 504
RHO = 1e-8


# ---------------------------------------------------------------------------
# Vectorised helpers (mirrors exp2, but adapted for Student-t)
# ---------------------------------------------------------------------------

def _twisted_probs_batch(
    p_n: np.ndarray,
    l_n: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """Compute twisted Bernoulli probs for a batch; shapes (B,N), (N,), (B,)."""
    exp_neg = np.exp(-theta[:, None] * l_n[None, :])
    return p_n / (p_n + (1.0 - p_n) * exp_neg)


def _solve_theta_batch(
    target_loss: float,
    p_n: np.ndarray,
    l_n: np.ndarray,
    n_iter: int = BISECT_ITERS,
) -> np.ndarray:
    """
    Vectorised bisection for theta in [0, THETA_MAX].
    theta >= 0 since target_loss > E[L] at theta=0.
    p_n: (B, N), l_n: (N,) → theta: (B,).
    """
    B = p_n.shape[0]
    lo = np.zeros(B, dtype=np.float64)
    hi = np.full(B, THETA_MAX, dtype=np.float64)

    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        p_theta = _twisted_probs_batch(p_n, l_n, mid)
        means = (p_theta * l_n[None, :]).sum(axis=1)
        lo = np.where(means < target_loss, mid, lo)
        hi = np.where(means >= target_loss, mid, hi)

    return 0.5 * (lo + hi)


def _log_psi_batch(
    theta: np.ndarray,
    p_n: np.ndarray,
    l_n: np.ndarray,
) -> np.ndarray:
    """
    Vectorised log-CGF:  psi_m = sum_n [theta_m*l_n + log(p_n[m] + (1-p_n[m])*exp(-theta_m*l_n))].
    theta: (B,), p_n: (B,N), l_n: (N,) → (B,).
    """
    exp_neg = np.exp(-theta[:, None] * l_n[None, :])
    return (theta[:, None] * l_n[None, :] + np.log(p_n + (1.0 - p_n) * exp_neg)).sum(axis=1)


def _log_r_z_gaussian_batch(
    Z: np.ndarray,
    mu: np.ndarray,
    Sigma_inv: np.ndarray,
    log_det_Sigma: float,
) -> np.ndarray:
    """
    log[N(z;0,I) / N(z;mu,Sigma)] for each row of Z.
    = 0.5 * (-z^T z + (z-mu)^T Sigma^{-1}(z-mu) + log|Sigma|)
    """
    diff = Z - mu[None, :]
    quad_prop = ((diff @ Sigma_inv) * diff).sum(axis=1)
    quad_prior = (Z ** 2).sum(axis=1)
    return 0.5 * (-quad_prior + quad_prop + log_det_Sigma)


def _log_r_w_invgamma_batch(
    W: np.ndarray,
    alpha_prop: float,
    beta_prop: float,
    alpha0: float = ALPHA0,
    beta0: float = BETA0,
) -> np.ndarray:
    """
    log[IG(w; alpha0, beta0) / IG(w; alpha_prop, beta_prop)] for each w.

    Derivation: f(w;a,b) = b^a/Γ(a) · w^{-a-1} · exp(-b/w), so

        log[f(w;a0,b0)/f(w;ap,bp)]
            = a0·log(b0) - lgamma(a0) - ap·log(bp) + lgamma(ap)
              + (ap - a0)·log(w)          # from -(a0+1) - (-(ap+1)) = ap-a0
              + (bp - b0) / w             # from -b0/w - (-bp/w) = (bp-b0)/w
    """
    return (
        alpha0 * np.log(beta0)
        - gammaln(alpha0)
        - alpha_prop * np.log(beta_prop)
        + gammaln(alpha_prop)
        + (alpha_prop - alpha0) * np.log(W)
        + (beta_prop - beta0) / W          # + not − : (beta_prop−beta0)/w
    )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _calibrate_methods(
    Z: np.ndarray,
    W: np.ndarray,
    block_probs_pilot: np.ndarray,
    block_losses_arr: np.ndarray,
) -> dict:
    """
    Fit IS parameters for CEIS-t and ISCOS-t.

    Returns dict keyed by method name with:
        mu        : (d,) Gaussian mean for Z
        Sigma     : (d, d) Gaussian covariance for Z
        alpha_W   : float, InvGamma shape for W
        beta_W    : float, InvGamma scale for W
        ESS_cal   : calibration ESS
    """
    d = Z.shape[1]
    results = {}

    # --- CEIS-t ---
    print("[exp3] CEIS-t calibration: simulating conditional losses …")
    rng_ceis = np.random.default_rng(0)
    D_ceis = rng_ceis.binomial(n=10, p=block_probs_pilot)      # (M0, 10)
    L_ceis = (D_ceis * block_losses_arr[None, :]).sum(axis=1)  # (M0,)
    omega_ceis = (L_ceis >= TARGET_LOSS).astype(np.float64)

    mom_ceis = fit_gaussian_moments(Z, omega_ceis, rho=RHO)
    stats_w_ceis = compute_invgamma_sufficient_stats(W, omega_ceis)
    alpha_ceis, beta_ceis = fit_invgamma_params(
        stats_w_ceis["m_log"], stats_w_ceis["m_neg1"]
    )

    results["CEIS-t"] = {
        "mu": mom_ceis["mu"],
        "Sigma": mom_ceis["Sigma_ridge"],
        "alpha_W": alpha_ceis,
        "beta_W": beta_ceis,
        "ESS_cal": mom_ceis["ESS"],
    }
    print(
        f"         ESS_cal={mom_ceis['ESS']:.1f}, "
        f"alpha_W={alpha_ceis:.3f}, beta_W={beta_ceis:.3f}"
    )

    # --- ISCOS-t ---
    print(f"[exp3] ISCOS-t calibration: K={K_ISCOS} COS weights …")
    cos_out = compute_cos_tail_batch(
        block_probs_pilot,
        block_losses_arr,
        K=K_ISCOS,
        y_x=Y_X_T,
        a=A_COS,
        b=B_COS,
        batch_size=5000,
    )
    q_cos = cos_out["q_clipped"]

    mom_cos = fit_gaussian_moments(Z, q_cos, rho=RHO)
    stats_w_cos = compute_invgamma_sufficient_stats(W, q_cos)
    alpha_cos, beta_cos = fit_invgamma_params(
        stats_w_cos["m_log"], stats_w_cos["m_neg1"]
    )

    results["ISCOS-t"] = {
        "mu": mom_cos["mu"],
        "Sigma": mom_cos["Sigma_ridge"],
        "alpha_W": alpha_cos,
        "beta_W": beta_cos,
        "ESS_cal": mom_cos["ESS"],
    }
    print(
        f"         ESS_cal={mom_cos['ESS']:.1f}, "
        f"alpha_W={alpha_cos:.3f}, beta_W={beta_cos:.3f}"
    )

    return results


# ---------------------------------------------------------------------------
# Production simulation
# ---------------------------------------------------------------------------

def _run_production(
    method_name: str,
    mu: np.ndarray,
    Sigma: np.ndarray,
    alpha_W: float,
    beta_W: float,
    event_type: str,
    seed: int,
    block_losses_arr: np.ndarray,
) -> dict:
    """
    Run one production simulation for the Student-t copula model.

    Parameters
    ----------
    method_name : str
    mu : (d,) proposal mean for Z.
    Sigma : (d, d) proposal covariance for Z.
    alpha_W, beta_W : InvGamma proposal parameters for W.
    event_type : 'exact' or 'tail'.
    seed : int
    block_losses_arr : (10,) losses per block.

    Returns
    -------
    dict with prob_estimate, prod_ESS, mean_half_length, r, half_length_95.
    """
    d = mu.shape[0]
    N_OBL = 100
    n_blocks = len(block_losses_arr)

    l_n = np.repeat(block_losses_arr, 10).astype(np.float64)   # (100,)

    Sigma_inv = np.linalg.inv(Sigma)
    _, log_det_Sigma = np.linalg.slogdet(Sigma)

    Lambda_all = np.empty(M_PROD, dtype=np.float64)
    Y_all = np.empty((M_PROD, N_OBL), dtype=np.float64)
    indicator_all = np.empty(M_PROD, dtype=np.float64)

    rng = np.random.default_rng(seed)

    for start in range(0, M_PROD, PROD_BATCH):
        end = min(start + PROD_BATCH, M_PROD)
        B = end - start

        # Sample W ~ InvGamma(alpha_W, beta_W) and Z ~ N(mu, Sigma)
        W_b = sample_invgamma(alpha_W, beta_W, size=B, rng=rng)  # (B,)
        Z_b = rng.multivariate_normal(mu, Sigma, size=B)          # (B, d)

        # Student-t conditional block probs
        p_blocks = conditional_block_probs_t(Z_b, W_b, n_blocks=n_blocks)  # (B, 10)
        p_n = np.repeat(p_blocks, 10, axis=1)                              # (B, 100)

        # Solve theta (vectorised bisection)
        theta = _solve_theta_batch(TARGET_LOSS, p_n, l_n)                  # (B,)
        theta_eff = np.maximum(theta, 0.0) if event_type == "tail" else theta

        # Defaults under twisted measure
        p_n_theta = _twisted_probs_batch(p_n, l_n, theta_eff)              # (B, 100)
        Y_b = (rng.uniform(size=(B, N_OBL)) < p_n_theta).astype(np.float64)
        L_sim = (Y_b * l_n[None, :]).sum(axis=1)                           # (B,)

        # Log likelihood ratio
        log_R_Z = _log_r_z_gaussian_batch(Z_b, mu, Sigma_inv, log_det_Sigma)
        log_R_W = _log_r_w_invgamma_batch(W_b, alpha_W, beta_W)
        psi_b = _log_psi_batch(theta_eff, p_n, l_n)
        log_lambda = log_R_Z + log_R_W + psi_b - theta_eff * L_sim

        Lambda_all[start:end] = np.exp(log_lambda)
        Y_all[start:end] = Y_b

        # Integer rounding avoids float equality issues for exact event
        L_int = np.round(L_sim).astype(np.int64)
        if event_type == "exact":
            indicator_all[start:end] = (L_int == int(TARGET_LOSS)).astype(np.float64)
        else:
            indicator_all[start:end] = (L_int >= int(TARGET_LOSS)).astype(np.float64)

    L_obligor = np.broadcast_to(l_n[None, :], (M_PROD, N_OBL))
    contrib = compute_contribution_estimates(
        Lambda_all, L_obligor, Y_all, indicator_all
    )
    prod_ess = compute_production_ess(Lambda_all * indicator_all)

    print(
        f"     {method_name:8s} {event_type:5s}: "
        f"P(A)={contrib['prob_estimate']:.4e}, "
        f"ESS={prod_ess:.1f}, "
        f"HL={contrib['mean_half_length']:.4e}"
    )
    return {
        "method": method_name,
        "event": event_type,
        "prob_estimate": contrib["prob_estimate"],
        "prod_ESS": prod_ess,
        "mean_half_length": contrib["mean_half_length"],
        "r": contrib["r"],
        "half_length_95": contrib["half_length_95"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_exp3() -> dict:
    """
    Run Experiment 3 and return a results dictionary.

    Side-effects
    ------------
    Writes to ``results/``:
      * exp3_calibration_table.csv
      * exp3_production_table.csv
      * exp3_contribution_halfwidths.png
    """
    os.makedirs("results", exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Construct portfolio and draw common pilot sample
    # ------------------------------------------------------------------
    port = Portfolio(
        n_blocks=10,
        n_per_block=10,
        block_losses=(1, 1, 4, 4, 9, 9, 16, 16, 25, 25),
        p0=0.01,
        d=11,
    )
    block_losses_arr = np.array(port.block_losses, dtype=np.float64)
    n_blocks = port.n_blocks

    print(
        f"[exp3] Drawing M0={M0:,} pilot samples for Student-t model "
        f"(seed={SEED_PILOT}) …"
    )
    Z, W = sample_pilot_student_t(M0=M0, seed=SEED_PILOT, alpha0=ALPHA0, beta0=BETA0)

    # ------------------------------------------------------------------
    # 2. Vectorised Student-t conditional block probs for pilot
    # ------------------------------------------------------------------
    print("[exp3] Computing Student-t conditional block probabilities …")
    block_probs_pilot = conditional_block_probs_t(Z, W, n_blocks=n_blocks)  # (M0, 10)

    # ------------------------------------------------------------------
    # 3. Calibration
    # ------------------------------------------------------------------
    cal_params = _calibrate_methods(Z, W, block_probs_pilot, block_losses_arr)

    # ------------------------------------------------------------------
    # 4. Production simulations
    # ------------------------------------------------------------------
    method_names = list(cal_params.keys())                      # CEIS-t, ISCOS-t
    events = [("A1", "exact", SEED_A1), ("A2", "tail", SEED_A2)]

    prod_results: list[dict] = []
    print("[exp3] Running production simulations …")
    for mname in method_names:
        p = cal_params[mname]
        for ename, etype, eseed in events:
            res = _run_production(
                mname,
                p["mu"],
                p["Sigma"],
                p["alpha_W"],
                p["beta_W"],
                etype,
                eseed,
                block_losses_arr,
            )
            res["event_label"] = ename
            prod_results.append(res)

    # ------------------------------------------------------------------
    # 5. Save calibration table
    # ------------------------------------------------------------------
    cal_rows = [
        {
            "method": mname,
            "ESS_cal": float(cal_params[mname]["ESS_cal"]),
            "mu_norm": float(np.linalg.norm(cal_params[mname]["mu"])),
            "alpha_W": float(cal_params[mname]["alpha_W"]),
            "beta_W": float(cal_params[mname]["beta_W"]),
        }
        for mname in method_names
    ]
    pd.DataFrame(cal_rows).to_csv("results/exp3_calibration_table.csv", index=False)
    print("[exp3] Saved results/exp3_calibration_table.csv")

    # ------------------------------------------------------------------
    # 6. Save production table
    # ------------------------------------------------------------------
    prod_rows = [
        {
            "method": r["method"],
            "event": r["event_label"],
            "prob_estimate": r["prob_estimate"],
            "prod_ESS": r["prod_ESS"],
            "mean_half_length": r["mean_half_length"],
        }
        for r in prod_results
    ]
    pd.DataFrame(prod_rows).to_csv("results/exp3_production_table.csv", index=False)
    print("[exp3] Saved results/exp3_production_table.csv")

    # ------------------------------------------------------------------
    # 7. Contribution half-width figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    N_OBL = 100
    obligors = np.arange(1, N_OBL + 1)

    for ax, ename in zip(axes, ["A1", "A2"]):
        for r in prod_results:
            if r["event_label"] == ename:
                ax.plot(
                    obligors,
                    r["half_length_95"],
                    label=r["method"],
                    alpha=0.8,
                    linewidth=1.2,
                )
        ax.set_title(f"Event {ename}  (threshold={TARGET_LOSS})")
        ax.set_xlabel("Obligor index")
        ax.set_ylabel("95% CI half-length")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Risk-contribution 95% CI half-widths (Student-t copula)")
    fig.tight_layout()
    fig.savefig("results/exp3_contribution_halfwidths.png", dpi=150)
    plt.close(fig)
    print("[exp3] Saved results/exp3_contribution_halfwidths.png")

    print("[exp3] Done.")
    return {"calibration": cal_params, "production": prod_results}


if __name__ == "__main__":
    run_exp3()
