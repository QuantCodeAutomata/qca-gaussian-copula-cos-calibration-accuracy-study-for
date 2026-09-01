import pytest
import numpy as np
from scipy.stats import invgamma
from src.invgamma_fit import compute_invgamma_sufficient_stats, fit_invgamma_params, sample_invgamma


def test_sufficient_stats_uniform_weights():
    """With uniform weights, should return sample averages of log(W) and 1/W."""
    rng = np.random.default_rng(42)
    W = invgamma.rvs(a=2.0, scale=2.0, size=10000, random_state=rng)
    weights = np.ones(10000)
    stats = compute_invgamma_sufficient_stats(W, weights)
    # For InvGamma(2,2): E[log W] = log(beta) - digamma(alpha) = log(2) - digamma(2)
    # E[1/W] = alpha/beta = 2/2 = 1
    assert stats['m_neg1'] == pytest.approx(1.0, rel=0.05)  # allow 5% numerical tolerance
    assert 'm_log' in stats
    assert stats['S'] == pytest.approx(10000.0)


def test_fit_invgamma_round_trip():
    """Fitting from true sufficient stats should recover true parameters."""
    from scipy.special import digamma
    alpha_true = 2.0
    beta_true = 2.0
    # True sufficient stats for InvGamma(alpha, beta):
    # E[log W] = log(beta) - digamma(alpha)
    # E[1/W] = alpha/beta
    m_log = np.log(beta_true) - digamma(alpha_true)
    m_neg1 = alpha_true / beta_true
    alpha_fit, beta_fit = fit_invgamma_params(m_log, m_neg1)
    assert alpha_fit == pytest.approx(alpha_true, rel=1e-4)
    assert beta_fit == pytest.approx(beta_true, rel=1e-4)


def test_sample_invgamma_shape():
    """Samples should have correct shape."""
    rng = np.random.default_rng(0)
    samples = sample_invgamma(2.0, 2.0, 100, rng)
    assert samples.shape == (100,)
    assert np.all(samples > 0)
