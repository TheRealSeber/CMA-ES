import numpy as np

from cmaes_ri.metrics import THRESHOLDS, evals_to_thresholds, thin_indices


def test_evals_to_thresholds_basic():
    evals = np.array([10, 20, 30, 40, 50])
    fbest = np.array([1.0, 1e-1, 1e-3, 1e-6, 1e-9])
    res = evals_to_thresholds(evals, fbest, THRESHOLDS)
    assert res[1e-2] == 30
    assert res[1e-5] == 40
    assert res[1e-8] == 50


def test_threshold_not_reached_is_none():
    evals = np.array([10, 20])
    fbest = np.array([5.0, 1.0])
    res = evals_to_thresholds(evals, fbest, THRESHOLDS)
    assert res[1e-2] is None
    assert res[1e-8] is None


def test_thin_indices_short_returns_all():
    idx = thin_indices(50, 300)
    assert np.array_equal(idx, np.arange(50))


def test_thin_indices_long_is_bounded_and_endpoints():
    idx = thin_indices(100000, 300)
    assert len(idx) <= 301
    assert idx[0] == 0
    assert idx[-1] == 99999
    assert np.all(np.diff(idx) > 0)
