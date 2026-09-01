"""
Experiment 1: Gaussian Copula COS Calibration Accuracy Study.

Studies how well the filtered COS method approximates the exact conditional
tail probability q^ex(z) = P(L >= 250 | Z=z) for the Gaussian copula
portfolio, and how the quality of the resulting IS proposal (mu_K, Sigma_K)
converges as the number of COS terms K increases.

Pipeline
--------
1. Draw M0=250000 pilot samples Z ~ N_11(0,I) (seed 42).
2. Compute exact block probabilities and exact tail weights q_ex via DP
   convolution (threshold = 250).
3. Compute finite-pilot benchmark IS moments (mu_ex, Sigma_ex, ESS_ex).
4. For each K in [16, 32, 64, 128, 256, 512, 1024]:
   a. Compute COS tail weights q_K (EXP-4 filtered, clipped to [0,1]).
   b. Fit Gaussian proposal (mu_K, Sigma_K) via weighted moments.
   c. Record calibration diagnostics.
5. Save benchmark stats, K-sweep table, and three diagnostic figures.

COS settings
------------
a = -0.5,  b = 1100.5,  y_x = 249.5,  EXP-4 filter,  batch_size = 5000,
rho = 1e-8 (ridge regularisation for Sigma).
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

# Allow running from the repository root with  python -m exp.exp1_...
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.portfolio import Portfolio
from src.convolution import compute_exact_tail_batch
from src.cos import compute_cos_tail_batch
from src.calibration import (
    fit_gaussian_moments,
    compute_calibration_errors,
    compute_mean_abs_weight_error,
    compute_spurious_mass_share,
)
from src.diagnostics import gaussian_lr_margin, min_eigenvalue


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

M0 = 250_000
SEED = 42
THRESHOLD = 250
Y_X = 249.5
A_COS = -0.5
B_COS = 1100.5
BATCH_SIZE = 5000
RHO = 1e-8
K_LIST = [16, 32, 64, 128, 256, 512, 1024]
PHI_INV_P0 = float(norm.ppf(0.01))      # Phi^{-1}(0.01) ≈ -2.326


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_exp1() -> dict:
    """
    Run Experiment 1 and return a results dictionary.

    Side-effects
    ------------
    Writes to ``results/``:
      * exp1_benchmark_stats.json
      * exp1_k_sweep_table.csv
      * exp1_error_convergence.png
      * exp1_ess_vs_k.png
      * exp1_clipping_diagnostics.png
    """
    os.makedirs("results", exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Construct portfolio and draw pilot sample
    # ------------------------------------------------------------------
    port = Portfolio(
        n_blocks=10,
        n_per_block=10,
        block_losses=(1, 1, 4, 4, 9, 9, 16, 16, 25, 25),
        p0=0.01,
        d=11,
    )
    block_losses_arr = np.array(port.block_losses, dtype=np.float64)

    print(f"[exp1] Drawing M0={M0:,} pilot samples (seed={SEED}) …")
    Z = port.sample_pilot_gaussian(M0=M0, seed=SEED)            # (M0, 11)

    # ------------------------------------------------------------------
    # 2. Vectorised Gaussian conditional block probabilities
    #    p_g(z) = Phi( (Phi^{-1}(p0) + 0.3*z[0] + 0.8*z[g]) / sqrt(0.27) )
    # ------------------------------------------------------------------
    block_probs = norm.cdf(
        (PHI_INV_P0 + 0.3 * Z[:, 0:1] + 0.8 * Z[:, 1:]) / np.sqrt(0.27)
    )                                                            # (M0, 10)

    # ------------------------------------------------------------------
    # 3. Exact conditional tail probabilities via DP convolution
    # ------------------------------------------------------------------
    print("[exp1] Computing exact tail probabilities (DP convolution) …")
    q_ex = compute_exact_tail_batch(
        block_probs, block_losses_arr, threshold=THRESHOLD, batch_size=BATCH_SIZE
    )                                                            # (M0,)

    # ------------------------------------------------------------------
    # 4. Finite-pilot benchmark moments
    # ------------------------------------------------------------------
    S_ex = float(q_ex.sum())
    mu_ex = (q_ex @ Z) / S_ex                                   # (11,)
    Z_c = Z - mu_ex[None, :]                                    # (M0, 11)
    Sigma_ex = (q_ex[:, None] * Z_c).T @ Z_c / S_ex            # (11, 11)
    ESS_ex = S_ex ** 2 / float(np.dot(q_ex, q_ex))
    mean_weight_ex = float(q_ex.mean())

    print(
        f"[exp1] Benchmark — mean_weight_ex={mean_weight_ex:.4e}, "
        f"ESS_ex={ESS_ex:.1f}"
    )

    # ------------------------------------------------------------------
    # 5. K-sweep: COS calibration for each K
    # ------------------------------------------------------------------
    k_rows: list[dict] = []

    for K in K_LIST:
        print(f"[exp1]   K={K:5d} …", end=" ", flush=True)

        # COS tail probabilities (EXP-4 filtered, clipped)
        cos_out = compute_cos_tail_batch(
            block_probs,
            block_losses_arr,
            K=K,
            y_x=Y_X,
            a=A_COS,
            b=B_COS,
            batch_size=BATCH_SIZE,
        )
        q_K = cos_out["q_clipped"]                              # (M0,)

        # Fit Gaussian proposal moments from COS weights
        moments = fit_gaussian_moments(Z, q_K, rho=RHO)

        # Calibration errors relative to exact benchmark
        errors = compute_calibration_errors(
            moments["mu"], moments["Sigma_ridge"], mu_ex, Sigma_ex
        )

        # Additional weight-error diagnostics
        mae = compute_mean_abs_weight_error(q_K, q_ex)
        spurious = compute_spurious_mass_share(q_K, q_ex)
        lam_min = min_eigenvalue(moments["Sigma_ridge"])
        m_LR = gaussian_lr_margin(moments["Sigma_ridge"])

        row = {
            "K": K,
            "mean_weight": float(moments["mean_weight"]),
            "ESS": float(moments["ESS"]),
            "e_mu": float(errors["e_mu"]),
            "e_Sigma": float(errors["e_Sigma"]),
            "mean_abs_weight_error": float(mae),
            "spurious_mass_share": float(spurious),
            "frac_below_0": float(cos_out["frac_below_0"]),
            "frac_above_1": float(cos_out["frac_above_1"]),
            "frac_clipped": float(cos_out["frac_clipped"]),
            "lambda_min": float(lam_min),
            "m_LR": float(m_LR),
        }
        k_rows.append(row)

        print(
            f"ESS={row['ESS']:.1f}, e_mu={row['e_mu']:.3e}, "
            f"e_Sigma={row['e_Sigma']:.3e}"
        )

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------

    # 6a. Benchmark stats JSON
    bench_stats = {"mean_weight_ex": mean_weight_ex, "ESS_ex": ESS_ex}
    with open("results/exp1_benchmark_stats.json", "w") as fh:
        json.dump(bench_stats, fh, indent=2)
    print("[exp1] Saved results/exp1_benchmark_stats.json")

    # 6b. K-sweep CSV
    df_sweep = pd.DataFrame(k_rows)
    df_sweep.to_csv("results/exp1_k_sweep_table.csv", index=False)
    print("[exp1] Saved results/exp1_k_sweep_table.csv")

    # 6c. Error-convergence plot (log-log: e_mu and e_Sigma vs K)
    K_vals = [r["K"] for r in k_rows]
    e_mu_vals = [r["e_mu"] for r in k_rows]
    e_Sigma_vals = [r["e_Sigma"] for r in k_rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(K_vals, e_mu_vals, "o-", label=r"$e_\mu$")
    ax.loglog(K_vals, e_Sigma_vals, "s--", label=r"$e_\Sigma$")
    ax.set_xlabel("K  (number of COS terms)")
    ax.set_ylabel("Relative error")
    ax.set_title("COS calibration: moment errors vs K")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/exp1_error_convergence.png", dpi=150)
    plt.close(fig)
    print("[exp1] Saved results/exp1_error_convergence.png")

    # 6d. ESS vs K
    ess_vals = [r["ESS"] for r in k_rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogx(K_vals, ess_vals, "o-", label="ESS (COS weights)")
    ax.axhline(ESS_ex, color="k", linestyle="--", linewidth=1.2, label="ESS_ex")
    ax.set_xlabel("K  (number of COS terms)")
    ax.set_ylabel("Calibration ESS")
    ax.set_title("ESS vs K")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/exp1_ess_vs_k.png", dpi=150)
    plt.close(fig)
    print("[exp1] Saved results/exp1_ess_vs_k.png")

    # 6e. Clipping diagnostics
    frac_below = [r["frac_below_0"] for r in k_rows]
    frac_above = [r["frac_above_1"] for r in k_rows]
    frac_clip = [r["frac_clipped"] for r in k_rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogx(K_vals, frac_below, "o-", label="frac_below_0")
    ax.semilogx(K_vals, frac_above, "s--", label="frac_above_1")
    ax.semilogx(K_vals, frac_clip, "^:", label="frac_clipped")
    ax.set_xlabel("K  (number of COS terms)")
    ax.set_ylabel("Fraction of samples")
    ax.set_title("Clipping diagnostics vs K")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/exp1_clipping_diagnostics.png", dpi=150)
    plt.close(fig)
    print("[exp1] Saved results/exp1_clipping_diagnostics.png")

    print("[exp1] Done.")
    return {"benchmark": bench_stats, "k_sweep": k_rows}


if __name__ == "__main__":
    run_exp1()
