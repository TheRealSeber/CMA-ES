"""Per-run trajectories, threshold-crossing extraction, and trajectory thinning."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

THRESHOLDS: tuple[float, ...] = (1e-2, 1e-5, 1e-8)


def evals_to_thresholds(
    evals: np.ndarray, fbest: np.ndarray, thresholds=THRESHOLDS
) -> dict[float, int | None]:
    """For each threshold, the eval count at the first generation with fbest <= threshold."""
    out: dict[float, int | None] = {}
    for t in thresholds:
        hit = fbest <= t
        if np.any(hit):
            out[t] = int(evals[int(np.argmax(hit))])
        else:
            out[t] = None
    return out


def thin_indices(n: int, max_points: int) -> np.ndarray:
    """Return up to max_points indices into range(n), log-spaced, always incl. first/last.

    Log spacing keeps early-iteration detail (where the p_sigma effect lives) while
    bounding the number of persisted points for long runs.
    """
    if n <= max_points:
        return np.arange(n)
    raw = np.unique(np.round(np.geomspace(1, n, num=max_points)).astype(int) - 1)
    raw = np.clip(raw, 0, n - 1)
    if raw[-1] != n - 1:
        raw = np.append(raw, n - 1)
    return np.unique(raw)


@dataclass
class RunRecord:
    # identity
    function: str
    dim: int
    sigma0: float
    sigma0_factor: float
    variant: str
    rep: int
    seed: int
    # per-generation trajectories (full resolution during the run)
    evals: list[int] = field(default_factory=list)
    fbest: list[float] = field(default_factory=list)
    sigma: list[float] = field(default_factory=list)
    ps_norm: list[float] = field(default_factory=list)
    # outcome
    total_evals: int = 0
    final_fbest: float = float("inf")
    success: bool = False  # reached 1e-8

    def thresholds_row(self) -> dict[float, int | None]:
        return evals_to_thresholds(np.asarray(self.evals), np.asarray(self.fbest))

    def summary_dict(self) -> dict:
        row = {
            "function": self.function,
            "dim": self.dim,
            "sigma0": self.sigma0,
            "sigma0_factor": self.sigma0_factor,
            "variant": self.variant,
            "rep": self.rep,
            "seed": self.seed,
            "n_gen": len(self.evals),
            "total_evals": self.total_evals,
            "final_fbest": self.final_fbest,
            "success": self.success,
        }
        for t, v in self.thresholds_row().items():
            row[f"evals_to_{t:.0e}"] = v
        return row

    def thinned_arrays(self, max_points: int = 300) -> dict[str, np.ndarray]:
        idx = thin_indices(len(self.evals), max_points)
        return {
            "gen": idx + 1,
            "evals": np.asarray(self.evals)[idx],
            "fbest": np.asarray(self.fbest)[idx],
            "sigma": np.asarray(self.sigma)[idx],
            "ps_norm": np.asarray(self.ps_norm)[idx],
        }
