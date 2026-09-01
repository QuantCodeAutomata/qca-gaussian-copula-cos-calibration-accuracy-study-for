import pytest
import numpy as np
from src.portfolio import Portfolio


def test_portfolio_construction():
    """Test that portfolio is constructed with correct parameters."""
    port = Portfolio()
    assert port.n_blocks == 10
    assert port.n_per_block == 10
    assert port.d == 11
    assert port.p0 == 0.01
    assert port.L_max == 1100
    assert len(port.block_losses) == 10
    expected_losses = np.array([1, 1, 4, 4, 9, 9, 16, 16, 25, 25])
    np.testing.assert_array_equal(port.block_losses, expected_losses)


def test_conditional_block_probs_gaussian_zero_factors():
    """At z=0, conditional probs should equal p0=0.01 only if model is calibrated that way."""
    port = Portfolio()
    z = np.zeros(11)
    probs = port.conditional_block_probs_gaussian(z)
    assert probs.shape == (10,)
    # At z=0: p_g(0) = Phi(Phi^{-1}(0.01)/sqrt(0.27)) which is NOT exactly 0.01
    # because loading is not identity. Just check values are in (0,1)
    assert np.all(probs > 0)
    assert np.all(probs < 1)


def test_conditional_block_probs_gaussian_range():
    """Conditional probabilities should always be in (0,1)."""
    port = Portfolio()
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((100, 11))
    for i in range(100):
        probs = port.conditional_block_probs_gaussian(Z[i])
        assert np.all(probs >= 0) and np.all(probs <= 1)


def test_pilot_sample_reproducibility():
    """Same seed should produce same pilot sample."""
    port = Portfolio()
    Z1 = port.sample_pilot_gaussian(100, seed=42)
    Z2 = port.sample_pilot_gaussian(100, seed=42)
    np.testing.assert_array_equal(Z1, Z2)


def test_pilot_sample_shape():
    """Pilot sample should have correct shape."""
    port = Portfolio()
    Z = port.sample_pilot_gaussian(500, seed=42)
    assert Z.shape == (500, 11)


def test_conditional_probs_student_t_range():
    """Student-t conditional probabilities should be in (0,1)."""
    port = Portfolio()
    rng = np.random.default_rng(42)
    Z = rng.standard_normal((50, 11))
    W = rng.exponential(1.0, size=50)  # just test the range
    for i in range(50):
        probs = port.conditional_block_probs_student_t(Z[i], W[i])
        assert probs.shape == (10,)
        assert np.all(probs >= 0) and np.all(probs <= 1)
