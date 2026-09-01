import pytest
import numpy as np
from src.convolution import compute_exact_tail_single, compute_exact_tail_batch

BLOCK_LOSSES = np.array([1, 1, 4, 4, 9, 9, 16, 16, 25, 25])
THRESHOLD = 250


def test_exact_tail_zero_probs():
    """With all zero default probs, tail probability should be 0."""
    probs = np.zeros(10)
    result = compute_exact_tail_single(probs, BLOCK_LOSSES, THRESHOLD)
    assert result == pytest.approx(0.0, abs=1e-10)


def test_exact_tail_one_probs():
    """With all default probs=1, loss = sum(10*c_g) = 1100 >= 250, tail prob = 1."""
    probs = np.ones(10)
    result = compute_exact_tail_single(probs, BLOCK_LOSSES, THRESHOLD)
    assert result == pytest.approx(1.0, abs=1e-10)


def test_exact_tail_in_range():
    """Result should always be in [0,1]."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        probs = rng.uniform(0, 1, size=10)
        result = compute_exact_tail_single(probs, BLOCK_LOSSES, THRESHOLD)
        assert 0 <= result <= 1


def test_exact_tail_batch_consistency():
    """Batch computation should match single computation."""
    rng = np.random.default_rng(5)
    probs_batch = rng.uniform(0, 0.1, size=(10, 10))
    batch_result = compute_exact_tail_batch(probs_batch, BLOCK_LOSSES, THRESHOLD)
    for i in range(10):
        single_result = compute_exact_tail_single(probs_batch[i], BLOCK_LOSSES, THRESHOLD)
        assert batch_result[i] == pytest.approx(single_result, rel=1e-10)


def test_exact_tail_monotone_in_probs():
    """Higher default probs should lead to higher (or equal) tail probability."""
    probs_low = np.full(10, 0.01)
    probs_high = np.full(10, 0.10)
    result_low = compute_exact_tail_single(probs_low, BLOCK_LOSSES, THRESHOLD)
    result_high = compute_exact_tail_single(probs_high, BLOCK_LOSSES, THRESHOLD)
    assert result_high >= result_low
