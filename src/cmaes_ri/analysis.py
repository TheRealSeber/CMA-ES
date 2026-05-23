"""Paired statistics (Wilcoxon) and Expected Running Time (ERT)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .runner import EVAL_BUDGET_PER_DIM


def expected_running_time(evals: np.ndarray, success: np.ndarray, budget: float) -> float:
    """ERT = (sum of evals on success + budget * #failures) / #successes.

    Failures contribute the full budget. Returns inf if there are no successes.
    """
    evals = np.asarray(evals, dtype=float)
    success = np.asarray(success, dtype=bool)
    n_succ = int(success.sum())
    if n_succ == 0:
        return float("inf")
    succ_evals = evals[success].sum()
    fail_cost = budget * int((~success).sum())
    return float((succ_evals + fail_cost) / n_succ)


def paired_wilcoxon(
    summary: pd.DataFrame, function: str, dim: int, factor: float, metric: str
) -> dict:
    """Paired Wilcoxon (baseline vs RI) on ``metric``, paired by rep.

    Only pairs where BOTH variants reached the threshold (metric not null) are used.
    """
    sub = summary[
        (summary.function == function) & (summary.dim == dim) & (summary.sigma0_factor == factor)
    ]
    b = sub[sub.variant == "baseline"].set_index("rep")[metric]
    r = sub[sub.variant == "ri"].set_index("rep")[metric]
    paired = pd.concat([b.rename("baseline"), r.rename("ri")], axis=1).dropna()

    result = {
        "function": function,
        "dim": dim,
        "sigma0_factor": factor,
        "metric": metric,
        "n_pairs": int(len(paired)),
        "median_baseline": float(paired["baseline"].median()) if len(paired) else float("nan"),
        "median_ri": float(paired["ri"].median()) if len(paired) else float("nan"),
        "statistic": float("nan"),
        "pvalue": float("nan"),
    }
    if len(paired) >= 5 and np.any(paired["baseline"].values != paired["ri"].values):
        stat, p = wilcoxon(paired["baseline"].values, paired["ri"].values)
        result["statistic"] = float(stat)
        result["pvalue"] = float(p)
    return result


def wilcoxon_table(summary: pd.DataFrame, metric: str = "evals_to_1e-08") -> pd.DataFrame:
    """Run paired_wilcoxon across every (function, dim, factor) cell."""
    rows = []
    keys = summary[["function", "dim", "sigma0_factor"]].drop_duplicates()
    for k in keys.itertuples(index=False):
        rows.append(
            paired_wilcoxon(summary, k.function, int(k.dim), float(k.sigma0_factor), metric)
        )
    return pd.DataFrame(rows)


def ert_table(summary: pd.DataFrame, metric: str = "evals_to_1e-08") -> pd.DataFrame:
    """Per (function, dim, factor, variant) ERT for the given threshold metric."""
    rows = []
    grp = summary.groupby(["function", "dim", "sigma0_factor", "variant"])
    for (fn, dim, fac, var), g in grp:
        budget = EVAL_BUDGET_PER_DIM * int(dim)
        reached = g[metric].notna().values
        evals = g[metric].fillna(budget).values
        rows.append(
            {
                "function": fn,
                "dim": int(dim),
                "sigma0_factor": float(fac),
                "variant": var,
                "ert": expected_running_time(evals, reached, budget),
                "success_rate": float(reached.mean()),
            }
        )
    return pd.DataFrame(rows)
