"""Execute a single CMA-ES run and record its trajectory."""

from __future__ import annotations

import numpy as np

from .engine import build_es
from .metrics import RunRecord
from .problems import get_problem
from .seeds import derive_streams

SIGMA0_FACTORS = (0.1, 0.5, 1.0, 2.0)
EVAL_BUDGET_PER_DIM = 10_000
FTARGET = 1e-8


def run_single(
    function: str,
    dim: int,
    sigma0_factor: float,
    variant: str,
    rep: int,
    seed: int,
    maxfevals: int | None = None,
) -> RunRecord:
    prob = get_problem(function)
    streams = derive_streams(seed, variant)

    # starting mean: uniform in domain, identical across variants for this seed
    x0 = streams.mean_rng.uniform(-prob.bound, prob.bound, size=dim)
    sigma0 = sigma0_factor * prob.L
    budget = maxfevals if maxfevals is not None else EVAL_BUDGET_PER_DIM * dim

    es = build_es(
        dim=dim,
        x0=x0,
        sigma0=sigma0,
        variant=variant,
        pycma_seed=streams.pycma_seed,
        perturb_rng=streams.perturb_rng,
        maxfevals=budget,
        ftarget=FTARGET,
    )

    rec = RunRecord(
        function=function,
        dim=dim,
        sigma0=sigma0,
        sigma0_factor=sigma0_factor,
        variant=variant,
        rep=rep,
        seed=seed,
    )

    while not es.stop():
        solutions = es.ask()
        values = [prob.func(np.asarray(x)) for x in solutions]
        es.tell(solutions, values)
        rec.evals.append(int(es.countevals))
        rec.fbest.append(float(es.best.f))
        rec.sigma.append(float(es.sigma))
        rec.ps_norm.append(float(np.linalg.norm(es.adapt_sigma.ps)))

    rec.total_evals = int(es.countevals)
    rec.final_fbest = float(es.best.f)
    rec.success = rec.final_fbest <= FTARGET
    return rec
