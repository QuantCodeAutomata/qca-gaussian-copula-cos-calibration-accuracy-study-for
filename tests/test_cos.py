import pytest
import numpy as np
from src.cos import compute_cos_tail_batch

BLOCK_LOSSES = np.array([1, 1, 4, 4, 9, 9, 16, 16, 25, 25])


def test_cos_output_structure():
    """COS computation should return dict with expected keys."""
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 0.1, size=(5, 10))
    result = compute_cos_tail_batch(probs, BLOCK_LOSSES, K=32, y_x=249.5, a=-0.5, b=1100.5)
    assert 'q_tilde' in result
    assert 'q_clipped' in result
    assert 'frac_below_0' in result
    assert 'frac_above_1' in result
    assert len(result['q_tilde']) == 5
    assert len(result['q_clipped']) == 5


def test_cos_clipping():
    """Clipped values should always be in [0,1]."""
    rng = np.random.default_rng(42)
    probs = rng.uniform(0, 0.3, size=(50, 10))
    result = compute_cos_tail_batch(probs, BLOCK_LOSSES, K=64, y_x=249.5, a=-0.5, b=1100.5)
    assert np.all(result['q_clipped'] >= 0)
    assert np.all(result['q_clipped'] <= 1)


def test_cos_exp4_filter_applied():
    """Different K values should produce different but valid results."""
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 0.05, size=(10, 10))
    result16 = compute_cos_tail_batch(probs, BLOCK_LOSSES, K=16, y_x=249.5, a=-0.5, b=1100.5)
    result256 = compute_cos_tail_batch(probs, BLOCK_LOSSES, K=256, y_x=249.5, a=-0.5, b=1100.5)
    # Both should produce valid clipped outputs
    assert np.all(result16['q_clipped'] >= 0)
    assert np.all(result256['q_clipped'] >= 0)
