"""Figures: convergence curves, sigma & ||p_sigma|| trajectories, ERT vs sigma0."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .analysis import ert_table  # noqa: E402

_PALETTE = {"baseline": "#1f77b4", "ri": "#d62728"}


def _median_band(ax, df, ycol, label, color, logy=True):
    """Plot median over reps vs evals, with inter-quartile band, per variant."""
    piv = df.pivot_table(index="gen", columns="rep", values=ycol)
    ev = df.groupby("gen")["evals"].median()
    med = piv.median(axis=1)
    q1 = piv.quantile(0.25, axis=1)
    q3 = piv.quantile(0.75, axis=1)
    ax.plot(ev, med, color=color, label=label)
    ax.fill_between(ev, q1, q3, color=color, alpha=0.2)
    if logy:
        ax.set_yscale("log")


def _subset(traj, function, dim, factor, variant):
    return traj[
        (traj.function == function)
        & (traj.dim == dim)
        & (traj.sigma0_factor == factor)
        & (traj.variant == variant)
    ]


def plot_convergence(traj, function, dim, factor, out_dir) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for variant in ("baseline", "ri"):
        sub = _subset(traj, function, dim, factor, variant)
        if len(sub):
            _median_band(ax, sub, "fbest", variant, _PALETTE[variant], logy=True)
    ax.set_xlabel("function evaluations")
    ax.set_ylabel("best f (median, IQR)")
    ax.set_title(f"{function} n={dim} sigma0={factor}L")
    ax.legend()
    path = out_dir / f"convergence_{function}_n{dim}_s{factor}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_sigma_psnorm(traj, function, dim, factor, out_dir) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for variant in ("baseline", "ri"):
        sub = _subset(traj, function, dim, factor, variant)
        if len(sub):
            _median_band(ax1, sub, "sigma", variant, _PALETTE[variant], logy=True)
            _median_band(ax2, sub, "ps_norm", variant, _PALETTE[variant], logy=False)
    ax1.set_xlabel("function evaluations")
    ax1.set_ylabel("sigma (median, IQR)")
    ax1.set_xscale("log")
    ax1.set_title("step size")
    ax2.set_xlabel("function evaluations")
    ax2.set_ylabel("||p_sigma|| (median, IQR)")
    ax2.set_xscale("log")
    ax2.set_title("evolution path norm")
    ax2.legend()
    path = out_dir / f"sigma_psnorm_{function}_n{dim}_s{factor}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_ert_vs_sigma0(summary, function, dim, out_dir, metric="evals_to_1e-08") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ert = ert_table(summary, metric=metric)
    ert = ert[(ert.function == function) & (ert.dim == dim)].sort_values("sigma0_factor")
    fig, ax = plt.subplots(figsize=(7, 5))
    for variant in ("baseline", "ri"):
        sub = ert[ert.variant == variant]
        ax.plot(sub.sigma0_factor, sub.ert, "o-", color=_PALETTE[variant], label=variant)
    ax.set_xlabel("sigma0 / L")
    ax.set_ylabel(f"ERT to {metric.split('_')[-1]} (evals)")
    ax.set_yscale("log")
    ax.set_title(f"ERT vs initial step size — {function} n={dim}")
    ax.legend()
    path = out_dir / f"ert_{function}_n{dim}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
