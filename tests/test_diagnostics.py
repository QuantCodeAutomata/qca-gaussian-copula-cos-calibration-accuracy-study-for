import pytest
import numpy as np
from src.diagnostics import gaussian_lr_margin, min_eigenvalue, condition_number


def test_gaussian_lr_margin_identity():
    """For identity covariance, 2I - I^{-1} = 2I - I = I, so m_LR = 1."""
    Sigma = np.eye(3)
    margin = gaussian_lr_margin(Sigma)
    assert margin == pytest.approx(1.0, abs=1e-10)


def test_gaussian_lr_margin_condition():
    """For a well-conditioned Sigma, margin should be positive (satisfies LR condition)."""
    Sigma = 0.8 * np.eye(5)  # lambda_min = 0.8, 2 - 1/0.8 = 2 - 1.25 = 0.75 > 0
    margin = gaussian_lr_margin(Sigma)
    assert margin == pytest.approx(0.75, abs=1e-8)


def test_gaussian_lr_margin_fail_condition():
    """For Sigma with small eigenvalue, margin may be negative."""
    Sigma = 0.4 * np.eye(5)  # lambda_min = 0.4, 2 - 1/0.4 = 2 - 2.5 = -0.5 < 0
    margin = gaussian_lr_margin(Sigma)
    assert margin == pytest.approx(-0.5, abs=1e-8)


def test_min_eigenvalue_positive():
    """Identity matrix should have min eigenvalue = 1."""
    assert min_eigenvalue(np.eye(4)) == pytest.approx(1.0, abs=1e-10)


def test_condition_number_identity():
    """Identity matrix should have condition number = 1."""
    assert condition_number(np.eye(5)) == pytest.approx(1.0, abs=1e-10)
