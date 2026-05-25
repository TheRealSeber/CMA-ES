"""Command-line entrypoints: run / smoke / analyze / plot."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .analysis import ert_table, wilcoxon_table
from .experiment import run_matrix
from .plots import plot_convergence, plot_ert_vs_sigma0, plot_sigma_psnorm


def _cmd_run(args):
    run_matrix(args.out, n_jobs=args.jobs)


def _cmd_smoke(args):
    run_matrix(
        args.out, functions=["sphere"], dims=[2], factors=[1.0], reps=2, maxfevals=800, n_jobs=1
    )


def _cmd_analyze(args):
    out = Path(args.out)
    summary = pd.read_parquet(out / "summary.parquet")
    wt = wilcoxon_table(summary)
    wt.to_csv(out / "wilcoxon.csv", index=False)
    et = ert_table(summary)
    et.to_csv(out / "ert.csv", index=False)
    print(wt.to_string(index=False))


def _cmd_plot(args):
    out = Path(args.out)
    traj = pd.read_parquet(out / "trajectories.parquet")
    summary = pd.read_parquet(out / "summary.parquet")
    fig_dir = out / "figures"
    for fn in traj.function.unique():
        dims = sorted(traj[traj.function == fn].dim.unique())
        for n in dims:
            for fac in sorted(traj.sigma0_factor.unique()):
                plot_convergence(traj, fn, int(n), float(fac), fig_dir)
                plot_sigma_psnorm(traj, fn, int(n), float(fac), fig_dir)
            plot_ert_vs_sigma0(summary, fn, int(n), fig_dir)
    print(f"figures written to {fig_dir}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="cmaes-ri")
    p.add_argument("--out", default="results", help="output directory")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run the full matrix")
    pr.add_argument("--jobs", type=int, default=-1)
    pr.set_defaults(func=_cmd_run)

    ps = sub.add_parser("smoke", help="tiny validation run")
    ps.set_defaults(func=_cmd_smoke)

    pa = sub.add_parser("analyze", help="Wilcoxon + ERT tables from summary.parquet")
    pa.set_defaults(func=_cmd_analyze)

    pl = sub.add_parser("plot", help="figures from trajectories.parquet")
    pl.set_defaults(func=_cmd_plot)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
