"""Deterministic seeds and per-repetition independent random streams (PCG64)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Fixed root so the whole seed list is reproducible across machines.
_ROOT_ENTROPY = 0xC0FFEE_21


def make_seed_list(n_reps: int) -> list[int]:
    """Return a stable list of n_reps distinct 32-bit base seeds."""
    ss = np.random.SeedSequence(_ROOT_ENTROPY)
    children = ss.spawn(n_reps)
    return [int(c.generate_state(1, dtype=np.uint32)[0]) for c in children]


@dataclass
class Streams:
    mean_rng: np.random.Generator  # draws m^0 (same for both variants)
    perturb_rng: np.random.Generator  # draws z_i for RI only
    pycma_seed: int  # pycma internal seed (same for both variants)


def derive_streams(base_seed: int, variant: str) -> Streams:
    """Derive independent streams from a base seed.

    mean_rng and pycma_seed depend ONLY on base_seed (not variant), so baseline
    and RI share the same starting mean and the same pycma sampling sequence.
    """
    ss = np.random.SeedSequence(base_seed)
    mean_ss, perturb_ss, pycma_ss = ss.spawn(3)
    pycma_seed = int(pycma_ss.generate_state(1, dtype=np.uint32)[0])
    return Streams(
        mean_rng=np.random.Generator(np.random.PCG64(mean_ss)),
        perturb_rng=np.random.Generator(np.random.PCG64(perturb_ss)),
        pycma_seed=pycma_seed,
    )
