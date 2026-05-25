#!/usr/bin/env bash
# Reproduce all runs, statistics and figures for WAE Task 21.
set -euo pipefail
OUT="${1:-results}"
uv sync --extra dev
uv run cmaes-ri --out "$OUT" run
uv run cmaes-ri --out "$OUT" analyze
uv run cmaes-ri --out "$OUT" plot
echo "Done. Results in $OUT/"
