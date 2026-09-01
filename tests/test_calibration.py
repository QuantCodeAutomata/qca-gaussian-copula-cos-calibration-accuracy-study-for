import pytest
import numpy as np
from src.calibration import fit_gaussian_moments, compute_calibration_errors, compute_spurious_mass_share


def test_fit_gaussian_moments_uniform_weights():
    """With equal weights, moments should equal sample moments."""
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((1000, 5))
    weights = np.ones(1000)
    result = fit_gaussian_moments(Z, weights)
    assert result['success']
    np.testing.assert_allclose(result['mu'], Z.mean(axis=0), atol=1e-10)
    assert result['ESS'] == pytest.approx(1000.0, rel=1e-6)


def test_fit_gaussian_moments_zero_weights():
    """With all-zero weights, should return success=False."""
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((100, 5))
    weights = np.zeros(100)
    result = fit_gaussian_moments(Z, weights)
    assert not result['success']


def test_calibration_errors_identical():
    """If K and exact are identical, errors should be 0."""
    mu = np.array([1.0, 2.0, 3.0])
    Sigma = np.eye(3)
    errors = compute_calibration_errors(mu, Sigma, mu, Sigma)
    assert errors['e_mu'] == pytest.approx(0.0, abs=1e-12)
    assert errors['e_Sigma'] == pytest.approx(0.0, abs=1e-12)


def test_calibration_errors_range():
    """Errors should be non-negative."""
    mu1 = np.array([1.0, 2.0])
    mu2 = np.array([1.5, 2.5])
    S1 = np.eye(2)
    S2 = 2 * np.eye(2)
    errors = compute_calibration_errors(mu1, S1, mu2, S2)
    assert errors['e_mu'] >= 0
    assert errors['e_Sigma'] >= 0


def test_spurious_mass_share_all_zero_exact():
    """If all q_ex <= threshold, spurious mass share should be 1 or handle gracefully."""
    q_K = np.array([0.1, 0.2, 0.3])
    q_ex = np.array([0.0, 0.0, 0.0])
    result = compute_spurious_mass_share(q_K, q_ex, threshold=1e-12)
    # All COS weight is spurious
    assert result == pytest.approx(1.0)


def test_spurious_mass_share_none_spurious():
    """If no q_ex <= threshold, spurious share should be 0."""
    q_K = np.array([0.1, 0.2, 0.3])
    q_ex = np.array([0.5, 0.6, 0.7])
    result = compute_spurious_mass_share(q_K, q_ex, threshold=1e-12)
    assert result == pytest.approx(0.0)
