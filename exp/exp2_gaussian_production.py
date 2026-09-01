"""
Experiment 2: Gaussian Copula Matched-Budget Production Comparison.

Compares three IS calibration methods at a matched pilot budget (M0=250000)
for the Gaussian copula portfolio with threshold x=250.

Calibration methods
-------------------
CEIS   – Cross-Entropy IS: binary indicator  omega_m = 1_{L_m >= 250}
         obtained by simulating one conditional loss per pilot draw.
ISCOS  – IS via filtered COS approximation (K=32, EXP-4 filter).
GL     – Glasserman saddle-point: minimum-||mu||^2 Gaussian mean shift
         with Sigma = I, obtained by constrained numerical optimisation.

Production events
-----------------
A1: exact-level  {L = 250},  seed 420
A2: threshold-tail  {L >= 250},  seed 421

For each (method, event): M=250000 production samples, Bernoulli twisting.
Full LR = R_Z(z) * exp(-theta*L + psi(theta, p_n(z)))

Outputs (in results/)
---------------------
  exp2_calibration_table.csv
  exp2_production_table.csv
  exp2_contribution_halfwidths.png
  exp2_ess_comparison.png
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, multivariate_normal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.portfolio import Portfolio
from src.cos import compute_cos_tail_batch
from src.calibration import fit_gaussian_moments
from src.diagnostics import compute_contribution_estimates, compute_production_ess


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

M0 = 250_000
M_PROD = 250_000
SEED_PILOT = 42
SEED_A1 = 420
SEED_A2 = 421
TARGET_LOSS = 250
K_ISCOS = 32
PHI_INV_P0 = float(norm.ppf(0.01))
THETA_MAX = 100.0
BISECT_ITERS = 80         # bisection iterations → ~2^{-80} precision
PROD_BATCH = 10_000       # production samples processed per inner loop
A_COS = -0.5
B_COS = 1100.5
Y_X_GAUSS = 249.5
RHO = 1e-8


# ---------------------------------------------------------------------------
# Helpers: vectorised Bernoulli twisting
# ---------------------------------------------------------------------------

def _twisted_probs_batch(
    p_n: np.ndarray,
    l_n: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """
    Compute exponentially twisted probabilities for a batch of theta values.

    Parameters
    ----------
    p_n : (B, N) array of conditional default probabilities.
    l_n : (N,) array of loss-at-default values.
    theta : (B,) twisting parameters.

    Returns
    -------
    (B, N) array of twisted probabilities.
    """
    exp_neg = np.exp(-theta[:, None] * l_n[None, :])   # (B, N)
    return p_n / (p_n + (1.0 - p_n) * exp_neg)


def _solve_theta_batch(
    target_loss: float,
    p_n: np.ndarray,
    l_n: np.ndarray,
    n_iter: int = BISECT_ITERS,
) -> np.ndarray:
    """
    Vectorised bisection to find theta[m] such that
    twisted_mean(theta[m], p_n[m]) == target_loss for each row m.

    Searches over theta in [0, THETA_MAX] (theta >= 0 since target > E[L] at theta=0).

    Parameters
    ----------
    target_loss : float
        Desired mean portfolio loss under the twisted measure.
    p_n : (B, N) conditional default probabilities.
    l_n : (N,) loss-at-default values.
    n_iter : int
        Number of bisection iterations.

    Returns
    -------
    theta : (B,) optimal twisting parameters.
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
    Vectorised log-CGF:  psi_m = sum_n log[1 + p_n[m] * (exp(theta_m*l_n) - 1)]
    using the numerically stable form theta*l + log(p + (1-p)*exp(-theta*l)).

    Parameters
    ----------
    theta : (B,) twisting parameters.
    p_n   : (B, N) conditional probabilities.
    l_n   : (N,) loss-at-default values.

    Returns
    -------
    (B,) log-CGF values.
    """
    exp_neg = np.exp(-theta[:, None] * l_n[None, :])          # (B, N)
    log_terms = (
        theta[:, None] * l_n[None, :]
        + np.log(p_n + (1.0 - p_n) * exp_neg)
    )
    return log_terms.sum(axis=1)


def _log_r_z_gaussian_batch(
    Z: np.ndarray,
    mu: np.ndarray,
    Sigma_inv: np.ndarray,
    log_det_Sigma: float,
) -> np.ndarray:
    """
    Batch log likelihood ratio  log[N(z;0,I) / N(z;mu,Sigma)]  for each row.

    log R_Z = 0.5 * (-(z^T z) + (z-mu)^T Sigma^{-1}(z-mu) + log|Sigma|)

    Parameters
    ----------
    Z          : (B, d) factor draws.
    mu         : (d,) proposal mean.
    Sigma_inv  : (d, d) inverse proposal covariance.
    log_det_Sigma : float, log-determinant of proposal covariance.

    Returns
    -------
    (B,) log R_Z values.
    """
    diff = Z - mu[None, :]                                     # (B, d)
    quad_prop = ((diff @ Sigma_inv) * diff).sum(axis=1)        # (B,)
    quad_prior = (Z ** 2).sum(axis=1)                          # (B,)
    return 0.5 * (-quad_prior + quad_prop + log_det_Sigma)


# ---------------------------------------------------------------------------
# Glasserman minimum-||mu||^2 calibration (Gaussian copula)
# ---------------------------------------------------------------------------

def _expected_loss_gaussian(mu: np.ndarray, block_losses: np.ndarray) -> float:
    """
    Analytical E_{Z~N(mu,I)}[L] for the Gaussian copula portfolio.

    Marginalising z ~ N(mu, I) through the Gaussian CDF:
        E[p_g(Z)] = Phi(Phi^{-1}(p0) + 0.3*mu[0] + 0.8*mu[g])
    so
        E[L] = sum_g n_per_block * c_g * Phi(Phi^{-1}(p0) + 0.3*mu[0] + 0.8*mu[g]).
    """
    args = PHI_INV_P0 + 0.3 * mu[0] + 0.8 * mu[1:]    # (10,)
    return float(10.0 * np.dot(block_losses, norm.cdf(args)))


def _glasserman_mean_shift(
    target_loss: float,
    block_losses: np.ndarray,
    d: int = 11,
) -> np.ndarray:
    """
    Find the minimum-||mu||^2 Gaussian mean shift such that
    E_{Z~N(mu,I)}[L] == target_loss using SLSQP.

    Strategy
    --------
    1. Compute the gradient direction g of E_mu[L] at mu=0.
    2. Use 1-D bisection (brentq) along g/||g|| to find a feasible
       warm-start x0 that exactly satisfies the constraint.
    3. Polish with SLSQP (minimize ||mu||^2 s.t. E_mu[L] = target_loss)
       starting from x0.
    """
    from scipy.optimize import brentq

    def _jac_emu(mu: np.ndarray) -> np.ndarray:
        args = PHI_INV_P0 + 0.3 * mu[0] + 0.8 * mu[1:]    # (10,)
        phi_vals = norm.pdf(args)                            # (10,)
        grad = np.empty(d, dtype=np.float64)
        grad[0] = 10.0 * float(np.dot(block_losses, phi_vals)) * 0.3
        grad[1:] = 10.0 * block_losses * phi_vals * 0.8
        return grad

    # Step 1: gradient direction at mu=0
    g0 = _jac_emu(np.zeros(d))
    g_unit = g0 / np.linalg.norm(g0)

    # Step 2: 1-D bisection along g_unit
    def f1d(t: float) -> float:
        return _expected_loss_gaussian(t * g_unit, block_losses) - target_loss

    # Bracket: E[L] at mu=0 < target_loss; find t_hi where E[L] > target_loss
    t_hi = 1.0
    while f1d(t_hi) < 0.0 and t_hi < 1e5:
        t_hi *= 2.0
    t_star = brentq(f1d, 0.0, t_hi, xtol=1e-10, rtol=1e-10)
    x0 = t_star * g_unit                                    # feasible warm-start

    # Step 3: SLSQP polish
    def obj(mu: np.ndarray) -> float:
        return 0.5 * float(np.dot(mu, mu))

    def jac_obj(mu: np.ndarray) -> np.ndarray:
        return mu.copy()

    def con_eq(mu: np.ndarray) -> float:
        return _expected_loss_gaussian(mu, block_losses) - target_loss

    constraints = [{"type": "eq", "fun": con_eq, "jac": _jac_emu}]

    result = minimize(
        obj,
        x0=x0,
        jac=jac_obj,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 5000},
    )
    if not result.success:
        return x0          # return feasible warm-start if SLSQP fails
    return result.x


# ---------------------------------------------------------------------------
# Calibration (CEIS, ISCOS, Glasserman)
# ---------------------------------------------------------------------------

def _calibrate_methods(
    port: Portfolio,
    Z: np.ndarray,
    block_probs_pilot: np.ndarray,
    block_losses_arr: np.ndarray,
) -> dict:
    """
    Fit IS parameters for CEIS, ISCOS (K=32), and Glasserman.

    Returns a dict keyed by method name with entries:
        mu       : (d,) proposal mean
        Sigma    : (d, d) proposal covariance
        ESS_cal  : calibration ESS
    """
    d = Z.shape[1]
    results = {}

    # --- CEIS ---
    print("[exp2] CEIS calibration: simulating conditional losses …")
    rng_ceis = np.random.default_rng(0)
    # D_g ~ Binomial(10, p_g(z)) for each block and pilot draw
    D_ceis = rng_ceis.binomial(n=10, p=block_probs_pilot)      # (M0, 10)
    L_ceis = (D_ceis * block_losses_arr[None, :]).sum(axis=1)  # (M0,)
    omega_ceis = (L_ceis >= TARGET_LOSS).astype(np.float64)

    mom_ceis = fit_gaussian_moments(Z, omega_ceis, rho=RHO)
    results["CEIS"] = {
        "mu": mom_ceis["mu"],
        "Sigma": mom_ceis["Sigma_ridge"],
        "ESS_cal": mom_ceis["ESS"],
    }
    print(f"         ESS_cal={mom_ceis['ESS']:.1f}")

    # --- ISCOS ---
    print(f"[exp2] ISCOS calibration: K={K_ISCOS} COS weights …")
    cos_out = compute_cos_tail_batch(
        block_probs_pilot,
        block_losses_arr,
        K=K_ISCOS,
        y_x=Y_X_GAUSS,
        a=A_COS,
        b=B_COS,
        batch_size=5000,
    )
    q_cos = cos_out["q_clipped"]
    mom_cos = fit_gaussian_moments(Z, q_cos, rho=RHO)
    results["ISCOS"] = {
        "mu": mom_cos["mu"],
        "Sigma": mom_cos["Sigma_ridge"],
        "ESS_cal": mom_cos["ESS"],
    }
    print(f"         ESS_cal={mom_cos['ESS']:.1f}")

    # --- Glasserman ---
    print("[exp2] Glasserman calibration: solving minimum-norm mean shift …")
    mu_gl = _glasserman_mean_shift(TARGET_LOSS, block_losses_arr, d=d)
    e_loss = _expected_loss_gaussian(mu_gl, block_losses_arr)
    print(f"         E_mu[L]={e_loss:.2f}  (target={TARGET_LOSS})")
    results["GL"] = {
        "mu": mu_gl,
        "Sigma": np.eye(d, dtype=np.float64),
        "ESS_cal": np.nan,     # no pilot ESS for the analytical method
    }

    return results


# ---------------------------------------------------------------------------
# Production simulation
# ---------------------------------------------------------------------------

def _run_production(
    method_name: str,
    mu: np.ndarray,
    Sigma: np.ndarray,
    event_type: str,
    seed: int,
    block_losses_arr: np.ndarray,
) -> dict:
    """
    Run one production simulation (M_PROD samples).

    Parameters
    ----------
    method_name : str
        Label for logging.
    mu : (d,) proposal mean.
    Sigma : (d, d) proposal covariance.
    event_type : 'exact' or 'tail'
        A1 = exact (L == TARGET_LOSS), A2 = tail (L >= TARGET_LOSS).
    seed : int
        RNG seed.
    block_losses_arr : (n_blocks,) loss-at-default per block.

    Returns
    -------
    dict with keys prob_estimate, prod_ESS, mean_half_length, contrib.
    """
    d = mu.shape[0]
    N_OBL = 100                                                 # total obligors
    n_blocks = len(block_losses_arr)

    # Obligor loss-at-default: 10 obligors per block share the same c_g
    l_n = np.repeat(block_losses_arr, 10).astype(np.float64)   # (100,)

    # Pre-compute Gaussian LR quantities
    Sigma_inv = np.linalg.inv(Sigma)
    sign, log_det_Sigma = np.linalg.slogdet(Sigma)
    # log_det_Sigma = log|Sigma|

    # Allocate output arrays
    Lambda_all = np.empty(M_PROD, dtype=np.float64)
    Y_all = np.empty((M_PROD, N_OBL), dtype=np.float64)
    indicator_all = np.empty(M_PROD, dtype=np.float64)

    rng = np.random.default_rng(seed)

    for start in range(0, M_PROD, PROD_BATCH):
        end = min(start + PROD_BATCH, M_PROD)
        B = end - start

        # Sample Z from proposal N(mu, Sigma)
        Z_b = rng.multivariate_normal(mu, Sigma, size=B)       # (B, d)

        # Gaussian conditional block probs  p_g(z) = Phi((a + 0.3*z0 + 0.8*zg)/sqrt(0.27))
        p_blocks = norm.cdf(
            (PHI_INV_P0 + 0.3 * Z_b[:, 0:1] + 0.8 * Z_b[:, 1:n_blocks + 1])
            / np.sqrt(0.27)
        )                                                        # (B, 10)

        # Per-obligor probs: 10 obligors per block
        p_n = np.repeat(p_blocks, 10, axis=1)                  # (B, 100)

        # Solve theta (vectorised bisection)
        theta = _solve_theta_batch(TARGET_LOSS, p_n, l_n)       # (B,)

        # For tail event, use theta^+ = max(theta, 0)
        if event_type == "tail":
            theta_eff = np.maximum(theta, 0.0)
        else:
            theta_eff = theta

        # Twisted default probabilities and simulated defaults
        p_n_theta = _twisted_probs_batch(p_n, l_n, theta_eff)  # (B, 100)
        U = rng.uniform(size=(B, N_OBL))
        Y_b = (U < p_n_theta).astype(np.float64)               # (B, 100)
        L_sim = (Y_b * l_n[None, :]).sum(axis=1)               # (B,)

        # Log LR components
        log_R_Z = _log_r_z_gaussian_batch(Z_b, mu, Sigma_inv, log_det_Sigma)
        psi_b = _log_psi_batch(theta_eff, p_n, l_n)
        log_lambda = log_R_Z + psi_b - theta_eff * L_sim        # (B,)

        Lambda_b = np.exp(log_lambda)

        # Event indicator — use rounded integer comparison to avoid float equality issues
        L_int = np.round(L_sim).astype(np.int64)
        if event_type == "exact":
            ind_b = (L_int == int(TARGET_LOSS)).astype(np.float64)
        else:
            ind_b = (L_int >= int(TARGET_LOSS)).astype(np.float64)

        Lambda_all[start:end] = Lambda_b
        Y_all[start:end] = Y_b
        indicator_all[start:end] = ind_b

    # Contribution estimates (L_obligor is constant across samples)
    L_obligor = np.broadcast_to(l_n[None, :], (M_PROD, N_OBL))
    contrib = compute_contribution_estimates(
        Lambda_all, L_obligor, Y_all, indicator_all
    )
    prod_ess = compute_production_ess(Lambda_all * indicator_all)

    print(
        f"     {method_name:6s} {event_type:5s}: "
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
        "Lambda": Lambda_all,
        "indicator": indicator_all,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_exp2() -> dict:
    """
    Run Experiment 2 and return a results dictionary.

    Side-effects
    ------------
    Writes to ``results/``:
      * exp2_calibration_table.csv
      * exp2_production_table.csv
      * exp2_contribution_halfwidths.png
      * exp2_ess_comparison.png
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

    print(f"[exp2] Drawing M0={M0:,} pilot samples (seed={SEED_PILOT}) …")
    Z = port.sample_pilot_gaussian(M0=M0, seed=SEED_PILOT)     # (M0, 11)

    # Vectorised Gaussian block probs for pilot
    block_probs_pilot = norm.cdf(
        (PHI_INV_P0 + 0.3 * Z[:, 0:1] + 0.8 * Z[:, 1:]) / np.sqrt(0.27)
    )                                                            # (M0, 10)

    # ------------------------------------------------------------------
    # 2. Calibration
    # ------------------------------------------------------------------
    cal_params = _calibrate_methods(
        port, Z, block_probs_pilot, block_losses_arr
    )

    # ------------------------------------------------------------------
    # 3. Production simulations
    # ------------------------------------------------------------------
    method_names = list(cal_params.keys())                      # CEIS, ISCOS, GL
    events = [("A1", "exact", SEED_A1), ("A2", "tail", SEED_A2)]

    prod_results: list[dict] = []
    print("[exp2] Running production simulations …")
    for mname in method_names:
        mu_m = cal_params[mname]["mu"]
        Sigma_m = cal_params[mname]["Sigma"]
        for ename, etype, eseed in events:
            res = _run_production(
                mname, mu_m, Sigma_m, etype, eseed, block_losses_arr
            )
            res["event_label"] = ename
            prod_results.append(res)

    # ------------------------------------------------------------------
    # 4. Save calibration table
    # ------------------------------------------------------------------
    cal_rows = []
    for mname, params in cal_params.items():
        cal_rows.append(
            {
                "method": mname,
                "ESS_cal": float(params["ESS_cal"])
                if not np.isnan(params["ESS_cal"])
                else "N/A",
                "mu_norm": float(np.linalg.norm(params["mu"])),
                "E_mu_L": float(_expected_loss_gaussian(params["mu"], block_losses_arr)),
            }
        )
    pd.DataFrame(cal_rows).to_csv("results/exp2_calibration_table.csv", index=False)
    print("[exp2] Saved results/exp2_calibration_table.csv")

    # ------------------------------------------------------------------
    # 5. Save production table
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
    pd.DataFrame(prod_rows).to_csv("results/exp2_production_table.csv", index=False)
    print("[exp2] Saved results/exp2_production_table.csv")

    # ------------------------------------------------------------------
    # 6. Contribution half-width figure
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
        ax.set_title(f"Event {ename}")
        ax.set_xlabel("Obligor index")
        ax.set_ylabel("95% CI half-length")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Risk-contribution 95% CI half-widths")
    fig.tight_layout()
    fig.savefig("results/exp2_contribution_halfwidths.png", dpi=150)
    plt.close(fig)
    print("[exp2] Saved results/exp2_contribution_halfwidths.png")

    # ------------------------------------------------------------------
    # 7. ESS comparison figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, ename in zip(axes, ["A1", "A2"]):
        subset = [r for r in prod_results if r["event_label"] == ename]
        names = [r["method"] for r in subset]
        ess_vals = [r["prod_ESS"] for r in subset]
        ax.bar(names, ess_vals)
        ax.set_title(f"Production ESS – Event {ename}")
        ax.set_xlabel("Method")
        ax.set_ylabel("ESS")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Production ESS comparison (Gaussian copula)")
    fig.tight_layout()
    fig.savefig("results/exp2_ess_comparison.png", dpi=150)
    plt.close(fig)
    print("[exp2] Saved results/exp2_ess_comparison.png")

    print("[exp2] Done.")
    return {"calibration": cal_params, "production": prod_results}


if __name__ == "__main__":
    run_exp2()
