"""pycma extension: build a CMA-ES whose p_sigma init is either zero or randomized.

The ONLY difference between the two variants is the value written to
``es.adapt_sigma.ps`` immediately after construction, before the first tell().
"""

from __future__ import annotations

import cma
import numpy as np


def ri_psigma_vector(
    positive_weights: np.ndarray, mueff: float, dim: int, rng: np.random.Generator
) -> np.ndarray:
    """Return p_sigma^0 = sqrt(mueff) * sum_i w_i z_i, z_i ~ N(0, I).

    ``positive_weights`` must sum to 1. The result is distributed N(0, I) because
    Var = mueff * sum(w_i^2) * I = mueff * (1/mueff) * I = I.
    """
    mu = positive_weights.size
    z = rng.standard_normal((mu, dim))
    return np.sqrt(mueff) * (positive_weights[:, None] * z).sum(axis=0)


def build_es(
    dim: int,
    x0: np.ndarray,
    sigma0: float,
    variant: str,
    pycma_seed: int,
    perturb_rng: np.random.Generator,
    maxfevals: int | None = None,
    ftarget: float = 1e-8,
) -> cma.CMAEvolutionStrategy:
    """Construct a CMAEvolutionStrategy and apply the chosen p_sigma init."""
    opts = {
        "seed": int(pycma_seed),
        "ftarget": ftarget,
        "verbose": -9,
        "verb_log": 0,
        "verb_disp": 0,
    }
    if maxfevals is not None:
        opts["maxfevals"] = int(maxfevals)
    es = cma.CMAEvolutionStrategy(list(map(float, x0)), float(sigma0), opts)

    if variant == "ri":
        w = np.asarray(es.sp.weights)
        mu = es.sp.weights.mu
        positive = w[:mu]
        mueff = float(es.sp.weights.mueff)
        es.adapt_sigma.ps = ri_psigma_vector(positive, mueff, dim, perturb_rng)
    elif variant != "baseline":
        raise ValueError(f"unknown variant: {variant!r}")
    return es
