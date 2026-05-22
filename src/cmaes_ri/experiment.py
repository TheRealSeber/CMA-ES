"""Full experiment matrix: expansion, parallel dispatch, checkpoint/resume."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from .problems import PROBLEMS
from .runner import SIGMA0_FACTORS, run_single
from .seeds import make_seed_list

DEFAULT_FUNCTIONS = list(PROBLEMS)
DEFAULT_DIMS = (2, 10, 30)
DEFAULT_REPS = 31
DEFAULT_VARIANTS = ("baseline", "ri")
TRAJ_MAX_POINTS = 300


@dataclass(frozen=True)
class Config:
    function: str
    dim: int
    factor: float
    variant: str
    rep: int
    seed: int

    @property
    def key(self) -> tuple:
        return (self.function, self.dim, self.factor, self.variant, self.rep)


def build_matrix(
    functions=DEFAULT_FUNCTIONS,
    dims=DEFAULT_DIMS,
    factors=SIGMA0_FACTORS,
    variants=DEFAULT_VARIANTS,
    reps=DEFAULT_REPS,
) -> list[Config]:
    seeds = make_seed_list(reps)
    configs = []
    for fn in functions:
        for n in dims:
            for fac in factors:
                for var in variants:
                    for rep in range(reps):
                        configs.append(Config(fn, n, fac, var, rep, seeds[rep]))
    return configs


def _run_one(cfg: Config, maxfevals: int | None):
    rec = run_single(
        cfg.function, cfg.dim, cfg.factor, cfg.variant, cfg.rep, cfg.seed, maxfevals=maxfevals
    )
    summary = rec.summary_dict()
    arrays = rec.thinned_arrays(TRAJ_MAX_POINTS)
    traj = pd.DataFrame(
        {
            "function": cfg.function,
            "dim": cfg.dim,
            "sigma0_factor": cfg.factor,
            "variant": cfg.variant,
            "rep": cfg.rep,
            "gen": arrays["gen"],
            "evals": arrays["evals"],
            "fbest": arrays["fbest"],
            "sigma": arrays["sigma"],
            "ps_norm": arrays["ps_norm"],
        }
    )
    return summary, traj


def run_matrix(
    out_dir,
    functions=DEFAULT_FUNCTIONS,
    dims=DEFAULT_DIMS,
    factors=SIGMA0_FACTORS,
    variants=DEFAULT_VARIANTS,
    reps=DEFAULT_REPS,
    maxfevals=None,
    n_jobs=-1,
) -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.parquet"
    traj_path = out / "trajectories.parquet"

    matrix = build_matrix(functions, dims, factors, variants, reps)

    done_keys: set[tuple] = set()
    if summary_path.exists():
        prev = pd.read_parquet(summary_path)
        done_keys = {
            (r.function, r.dim, r.sigma0_factor, r.variant, r.rep) for r in prev.itertuples()
        }
    todo = [c for c in matrix if c.key not in done_keys]

    if todo:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_run_one)(cfg, maxfevals) for cfg in tqdm(todo, desc="runs", unit="run")
        )
        new_summary = pd.DataFrame([s for s, _ in results])
        new_traj = pd.concat([t for _, t in results], ignore_index=True)

        if summary_path.exists():
            new_summary = pd.concat([pd.read_parquet(summary_path), new_summary], ignore_index=True)
        if traj_path.exists():
            new_traj = pd.concat([pd.read_parquet(traj_path), new_traj], ignore_index=True)
        new_summary.to_parquet(summary_path, index=False)
        new_traj.to_parquet(traj_path, index=False)

    return pd.read_parquet(summary_path)
