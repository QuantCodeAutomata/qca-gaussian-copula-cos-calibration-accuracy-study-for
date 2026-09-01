import pytest
import numpy as np
from src.twisting import twisted_probs, twisted_mean, solve_theta, log_cgf


def test_twisted_probs_theta_zero():
    """At theta=0, twisted probs should equal original probs."""
    p = np.array([0.01, 0.02, 0.05])
    l = np.array([1.0, 4.0, 9.0])
    result = twisted_probs(p, l, theta=0.0)
    np.testing.assert_allclose(result, p, rtol=1e-10)


def test_twisted_probs_range():
    """Twisted probs should be in [0,1]."""
    p = np.array([0.01, 0.05, 0.10])
    l = np.array([1.0, 4.0, 9.0])
    for theta in [-1.0, 0.0, 0.5, 2.0]:
        result = twisted_probs(p, l, theta=theta)
        assert np.all(result >= 0)
        assert np.all(result <= 1)


def test_twisted_mean_monotone():
    """Twisted mean should be monotone increasing in theta."""
    p = np.array([0.01, 0.05, 0.10])
    l = np.array([1.0, 4.0, 9.0])
    means = [twisted_mean(t, p, l) for t in [0.0, 0.5, 1.0, 2.0]]
    for i in range(len(means) - 1):
        assert means[i] <= means[i + 1]


def test_solve_theta_correctness():
    """Solving theta should produce twisted mean equal to target."""
    p = np.full(10, 0.05)
    l = np.array([1, 1, 4, 4, 9, 9, 16, 16, 25, 25], dtype=float)
    target = 10.0
    theta = solve_theta(target, p, l)
    achieved = twisted_mean(theta, p, l)
    assert achieved == pytest.approx(target, rel=1e-5)


def test_log_cgf_theta_zero():
    """At theta=0, psi(0) = sum_n log[1 + p_n * (1-1)] = sum_n log(1) = 0."""
    p = np.array([0.01, 0.05])
    l = np.array([1.0, 4.0])
    result = log_cgf(0.0, p, l)
    assert result == pytest.approx(0.0, abs=1e-10)
