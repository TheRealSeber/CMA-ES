import pandas as pd

from cmaes_ri.plots import plot_convergence, plot_ert_vs_sigma0, plot_sigma_psnorm


def _toy_traj():
    rows = []
    for variant, base in [("baseline", 1.0), ("ri", 0.8)]:
        for rep in range(3):
            for gen in range(1, 11):
                rows.append(
                    {
                        "function": "sphere",
                        "dim": 2,
                        "sigma0_factor": 1.0,
                        "variant": variant,
                        "rep": rep,
                        "gen": gen,
                        "evals": gen * 6,
                        "fbest": base / gen,
                        "sigma": base / gen,
                        "ps_norm": float(gen),
                    }
                )
    return pd.DataFrame(rows)


def _toy_summary():
    rows = []
    for variant, base in [("baseline", 200), ("ri", 150)]:
        for rep in range(5):
            rows.append(
                {
                    "function": "sphere",
                    "dim": 2,
                    "sigma0_factor": 1.0,
                    "variant": variant,
                    "rep": rep,
                    "evals_to_1e-08": base + rep,
                    "success": True,
                }
            )
    return pd.DataFrame(rows)


def test_plot_convergence_writes_file(tmp_path):
    out = plot_convergence(_toy_traj(), "sphere", 2, 1.0, tmp_path)
    assert out.exists() and out.stat().st_size > 0


def test_plot_sigma_psnorm_writes_file(tmp_path):
    out = plot_sigma_psnorm(_toy_traj(), "sphere", 2, 1.0, tmp_path)
    assert out.exists() and out.stat().st_size > 0


def test_plot_ert_writes_file(tmp_path):
    out = plot_ert_vs_sigma0(_toy_summary(), "sphere", 2, tmp_path)
    assert out.exists() and out.stat().st_size > 0
