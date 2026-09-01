# Repository Notes

## Project Overview
Synthetic Gaussian/Student-t credit portfolio importance-sampling library.

## src/ Module Summary

| File | Purpose |
|------|---------|
| `portfolio.py` | `Portfolio` dataclass: conditional default probs (Gaussian & Student-t), pilot sampling |
| `convolution.py` | Exact tail probability via DP block-convolution (`threshold=250`) |
| `cos.py` | Filtered COS (EXP-4, K terms) conditional CDF → tail probability |
| `calibration.py` | Weighted Gaussian moment fitting, ESS, MAE, spurious-mass diagnostics |
| `diagnostics.py` | LR margin, eigenvalue, condition number, production ESS, contribution estimates |
| `twisting.py` | Bernoulli exponential twisting; Gaussian & InvGamma likelihood ratios |
| `invgamma_fit.py` | InvGamma MOM fitting via digamma equation; weighted sufficient stats |
| `student_t_model.py` | Batch `conditional_block_probs_t(Z,W)` and `sample_pilot_student_t(M0,seed)` |

## exp/ Module Summary

| File | Purpose |
|------|---------|
| `exp1_cos_calibration.py` | COS calibration accuracy study (K-sweep 16–1024, Gaussian copula) |
| `exp2_gaussian_production.py` | Matched-budget production comparison: CEIS, ISCOS (K=32), Glasserman (Gaussian) |
| `exp3_student_t_production.py` | Matched-budget production comparison: CEIS-t, ISCOS-t (K=64) (Student-t) |

## Key Design Decisions

### Numerical Stability
- `twisted_probs` and `log_cgf` in `twisting.py` use the form
  `p / (p + (1-p)*exp(-theta*l))` and `theta*l + log(p + (1-p)*exp(-theta*l))`
  to avoid `exp(theta*l)` overflow for large `theta`.
- COS `compute_char_func_batch` uses `term**10` where term is a complex
  array (M, K-1, 10); products are taken over the last axis.

### InvGamma Parameterisation
- Throughout: `InvGamma(alpha, beta)` means density
  `f(w) = beta^alpha / Gamma(alpha) * w^{-alpha-1} * exp(-beta/w)`.
- Sampling: `scipy.stats.invgamma(a=alpha, scale=beta)`.
- Prior for Student-t mixing: `InvGamma(2, 2)`.

### Convolution DP
- State vector length = `threshold` (index = loss value 0..threshold-1).
- Tail accumulator is a separate scalar updated during convolution.
- Batch dimension (M) is the first axis of all DP arrays.

### COS Parameters
- `a=-0.5`, `b=1100.5`, left-limit `y_x=249.5` (Gaussian) or `503.5` (Student-t).
- EXP-4 filter: `sigma_k = exp(-8*(k/K)^4)`.

## Dependencies
`numpy`, `scipy` (stats, special, optimize).
