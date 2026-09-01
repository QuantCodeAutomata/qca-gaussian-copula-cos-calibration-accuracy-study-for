# Gaussian Copula COS Calibration Accuracy Study

## Description

This project investigates the calibration accuracy of the **Importance Sampling via the Characteristic Function and Optimal Scaling (ISCOS)** methodology when applied to credit portfolio loss distributions modelled under a Gaussian copula framework. Three experiments probe distinct failure modes and robustness properties of the COS-based estimator.

### Experiment 1 – Baseline Gaussian Copula Calibration

Evaluates the standard ISCOS calibration pipeline on a homogeneous block-structured portfolio governed by a Gaussian copula with a single systematic factor per block. The experiment measures moment-matching errors in the twisted Gaussian proposal, the fraction of spurious mass produced by the COS tail approximation, and the sensitivity of calibration quality to the number of COS terms `K`.

### Experiment 2 – Heavy-Tail Misspecification (Student-t Copula)

Examines what happens when the true dependence structure is a Student-t copula (finite degrees of freedom) but the IS algorithm is calibrated assuming a Gaussian copula. Key metrics include the log-likelihood ratio margin (LR condition), the minimum eigenvalue of the moment-matched covariance, and the empirical coverage of the resulting confidence intervals.

### Experiment 3 – High-Dimensional Stability

Scales the portfolio to a large number of obligors and systematic factors to stress-test the numerical stability of the COS convolution step. The experiment tracks the condition number of the calibrated covariance matrix and the share of COS output values that must be clipped to `[0, 1]`, as a function of `K` and portfolio dimension `d`.

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/                  # Raw market data (git-ignored CSVs); init only
│   └── __init__.py
├── results/               # Output tables and figures (tracked by git)
├── src/                   # Core library modules
│   ├── __init__.py
│   ├── portfolio.py       # Portfolio class: block structure, copula sampling
│   ├── convolution.py     # Exact discrete convolution of block loss distributions
│   ├── cos.py             # COS-based tail probability approximation (ISCOS)
│   ├── calibration.py     # Moment-matching and calibration error metrics
│   ├── diagnostics.py     # LR-condition margin, eigenvalue, condition-number helpers
│   ├── twisting.py        # Exponential twisting: CGF, theta solver, twisted probabilities
│   └── invgamma_fit.py    # Inverse-Gamma sufficient-statistics and parameter fitting
├── experiments/           # Top-level experiment runner scripts
│   ├── exp1_baseline.py
│   ├── exp2_student_t.py
│   └── exp3_high_dim.py
└── tests/                 # pytest test suite
    ├── __init__.py
    ├── test_portfolio.py
    ├── test_convolution.py
    ├── test_cos.py
    ├── test_calibration.py
    ├── test_diagnostics.py
    ├── test_twisting.py
    └── test_invgamma_fit.py
```

---

## Installation

Python 3.10 or later is recommended.

```bash
# Clone the repository
git clone <repo-url>
cd gaussian-copula-cos-calibration

# (Optional) create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Running individual experiments

```bash
python experiments/exp1_baseline.py
python experiments/exp2_student_t.py
python experiments/exp3_high_dim.py
```

Each script writes summary tables (CSV) and diagnostic figures (PDF/PNG) to the `results/` directory.

### Running all experiments sequentially

```bash
for exp in experiments/exp*.py; do python "$exp"; done
```

### Running the test suite

```bash
pytest tests/ -v
```

To run a single test module:

```bash
pytest tests/test_cos.py -v
```

---

## Key References

1. **Fang, F., Oosterlee, C. W. (2008).** "A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions." *SIAM Journal on Scientific Computing*, 31(2), 826–848.  
   — Foundational COS method for characteristic-function-based density recovery.

2. **Glasserman, P., Li, J. (2005).** "Importance Sampling for Portfolio Credit Risk." *Management Science*, 51(11), 1643–1656.  
   — Exponential twisting framework for rare-event simulation in credit portfolios.

3. **ISCOS Methodology** — The Importance Sampling via Characteristic Function and Optimal Scaling approach combines the COS series expansion (reference 1) with exponential twisting (reference 2) to calibrate an efficient proposal distribution for estimating extreme credit loss probabilities. The COS step approximates the conditional tail probability `P(L > x | Z)` via the characteristic function of the block loss distribution, and the twisting step solves for the mean-shift parameter `θ` that centres the proposal on the target tail event.
