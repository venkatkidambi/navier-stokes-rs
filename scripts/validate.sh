#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p results results/fields figures
export MPLCONFIGDIR="$ROOT_DIR/.matplotlib"
mkdir -p "$MPLCONFIGDIR"

cargo run --release -- \
  --nx 64 \
  --ny 64 \
  --re 100 \
  --nt 100 \
  --mode optimized_tolerance \
  --poisson-solver multigrid \
  --poisson-tol 1e-8 \
  --max-vcycles 20 \
  --out results/validation_re100.csv \
  --write-fields

if [[ -x .venv/bin/python ]]; then
  .venv/bin/python scripts/validate_ghia_centerlines.py --re 100
else
  python3 scripts/validate_ghia_centerlines.py --re 100
fi
