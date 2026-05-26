# CMA-ES — Randomized Initialization of the Evolution Path `p_σ`

WAE 2026L, Task 21 — Sebastian Rydz, Jerzy Czarnecki.

A performance analysis of the CMA-ES algorithm evaluating a custom, weighted random
initialization of the evolution path vector `p_σ`.

## Idea

Standard CMA-ES initializes the evolution path with zeros, `p_σ⁰ = 0`. This artificially
deflates `‖p_σ‖` in the first iterations and can cause overly aggressive step-size shrinking.
This project studies a **randomized initialization (RI)**:

```
p_σ⁰ = √µ_eff · Σ_{i=1}^{µ} w_i z_i ,   Σ w_i = 1, w_i > 0,   z_i ~ N(0, I)
```

where `w_i` are CMA-ES's own recombination weights. The `√µ_eff` factor makes `p_σ⁰ ~ N(0, I)`
— distributionally neutral but dynamically different in the early iterations. The study
compares **baseline** (`p_σ⁰ = 0`) against **RI** across test functions, dimensions, and
initial step sizes `σ⁰`.

The CMA-ES engine is the reference implementation **[pycma](https://github.com/CMA-ES/pycma)**;
the modification is applied as an *extension* (only `es.adapt_sigma.ps` is overwritten right
after construction), so the only difference between variants is the `p_σ` initialization.

## Install

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
```

## Usage

```bash
# tiny validation run (seconds)
uv run cmaes-ri --out results smoke

# full experiment matrix (5 functions × 3 dims × 4 σ⁰ × 2 variants × 31 reps = 3720 runs)
uv run cmaes-ri --out results run --jobs -1

# statistics (paired Wilcoxon + ERT) -> results/wilcoxon.csv, results/ert.csv
uv run cmaes-ri --out results analyze

# figures -> results/figures/*.png
uv run cmaes-ri --out results plot
```

Or reproduce everything in one shot:

```bash
./scripts/reproduce.sh results
```

Runs are **checkpointed**: re-running `run` skips configurations already present in
`results/summary.parquet`, so an interrupted run resumes.

## Experiment design

| Aspect | Value |
|---|---|
| Test functions | Sphere, Rosenbrock (unimodal); Rastrigin, Ackley, Griewank (multimodal) |
| Dimensions | n ∈ {2, 10, 30} |
| Population | λ = 4 + ⌊3 ln n⌋, µ = ⌊λ/2⌋ (pycma defaults) |
| Initial step size | σ⁰ ∈ {0.1, 0.5, 1.0, 2.0} · L, L = half domain width |
| Repetitions | 31 per cell, deterministic PCG64 seeds (shared `m⁰`/pycma seed across variants) |
| Budget | 10⁴·n evaluations or f-precision 1e-8 (plus pycma default convergence criteria) |
| Recorded | best-f, σ(t), ‖p_σ(t)‖ trajectories; evals-to-{1e-2,1e-5,1e-8}; success |
| Analysis | paired Wilcoxon on evals-to-threshold; ERT; success rate |

## Project structure

```
src/cmaes_ri/
  problems.py     test functions + domains
  seeds.py        deterministic PCG64 seeds and per-rep streams
  engine.py       pycma extension: zero vs randomized p_σ init  (the studied effect)
  metrics.py      run record, threshold crossing, trajectory thinning
  runner.py       single run executor
  experiment.py   matrix expansion, joblib parallel dispatch, parquet checkpoint/resume
  analysis.py     paired Wilcoxon, ERT
  plots.py        convergence / σ & ‖p_σ‖ / ERT figures
  cli.py          run / smoke / analyze / plot entrypoints
tests/            pytest suite
scripts/reproduce.sh
report/           LaTeX report + compiled PDF
Containerfile     python:3.12-slim (Podman)
```

## Reproducibility

Each repetition is parameterized by a deterministic PCG64 seed (`numpy.random.Generator`),
derived so the starting mean `m⁰` and pycma's internal sampling are identical across the two
variants — isolating the `p_σ` initialization effect for paired statistics. Raw per-run data
is stored as parquet; large parquet files are kept out of git (see `.gitignore`) with a
checksum manifest in `results/MANIFEST.txt`.

## Tests

```bash
uv run pytest
```

Covers test-function optima, the RI vector's N(0,I) statistics, seed determinism, single-run
convergence, matrix resume, and the analysis/plot paths.
