import numpy as np
import pytest

from cmaes_ri.engine import build_es, ri_psigma_vector


def test_baseline_ps_is_zero():
    es = build_es(
        dim=10,
        x0=np.zeros(10),
        sigma0=1.0,
        variant="baseline",
        pycma_seed=1,
        perturb_rng=np.random.default_rng(0),
    )
    assert np.allclose(es.adapt_sigma.ps, 0.0)


def test_ri_ps_is_nonzero():
    es = build_es(
        dim=10,
        x0=np.zeros(10),
        sigma0=1.0,
        variant="ri",
        pycma_seed=1,
        perturb_rng=np.random.default_rng(0),
    )
    assert np.linalg.norm(es.adapt_sigma.ps) > 0.0


def test_ri_vector_is_approximately_standard_normal():
    dim = 8
    rng = np.random.default_rng(42)
    weights = np.array([0.5, 0.3, 0.2])  # mu=3, sums to 1
    mueff = 1.0 / np.sum(weights**2)
    draws = np.array([ri_psigma_vector(weights, mueff, dim, rng) for _ in range(20000)])
    assert np.allclose(draws.mean(axis=0), 0.0, atol=0.05)
    assert np.allclose(draws.var(axis=0), 1.0, atol=0.1)


def test_weights_first_mu_sum_to_one():
    es = build_es(
        dim=10,
        x0=np.zeros(10),
        sigma0=1.0,
        variant="baseline",
        pycma_seed=1,
        perturb_rng=np.random.default_rng(0),
    )
    w = np.asarray(es.sp.weights)
    mu = es.sp.weights.mu
    assert w[:mu].sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(w[:mu] > 0)


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        build_es(
            dim=3,
            x0=np.zeros(3),
            sigma0=1.0,
            variant="bogus",
            pycma_seed=1,
            perturb_rng=np.random.default_rng(0),
        )
