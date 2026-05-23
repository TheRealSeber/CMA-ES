import numpy as np
import pandas as pd

from cmaes_ri.analysis import expected_running_time, paired_wilcoxon


def _toy_summary():
    rows = []
    for rep in range(10):
        rows.append(
            {
                "function": "sphere",
                "dim": 2,
                "sigma0_factor": 1.0,
                "variant": "baseline",
                "rep": rep,
                "seed": rep,
                "evals_to_1e-08": 100 + rep,
                "success": True,
            }
        )
        rows.append(
            {
                "function": "sphere",
                "dim": 2,
                "sigma0_factor": 1.0,
                "variant": "ri",
                "rep": rep,
                "seed": rep,
                "evals_to_1e-08": 90 + rep,
                "success": True,
            }
        )
    return pd.DataFrame(rows)


def test_paired_wilcoxon_detects_difference():
    df = _toy_summary()
    res = paired_wilcoxon(df, "sphere", 2, 1.0, "evals_to_1e-08")
    assert res["n_pairs"] == 10
    assert res["median_baseline"] > res["median_ri"]
    assert 0.0 <= res["pvalue"] <= 1.0


def test_ert_basic():
    evals = np.array([100, 200, 500])
    success = np.array([True, True, False])
    assert expected_running_time(evals, success, budget=500) == 400.0


def test_ert_no_success_is_inf():
    assert expected_running_time(np.array([10, 20]), np.array([False, False]), 20) == float("inf")
