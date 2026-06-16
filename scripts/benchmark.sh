#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${NX:-}" ]]; then
  GRIDS="${GRIDS:-$NX}"
else
  GRIDS="${GRIDS:-64 128 256}"
fi
RE="${RE:-100}"
NT="${NT:-200}"
POISSON_TOL="${POISSON_TOL:-1e-8}"
MAX_VCYCLES="${MAX_VCYCLES:-20}"

mkdir -p results results/fields figures
export MPLCONFIGDIR="$ROOT_DIR/.matplotlib"
mkdir -p "$MPLCONFIGDIR"

find results -mindepth 1 ! -name ".gitkeep" -exec rm -rf {} +
find figures -mindepth 1 ! -name ".gitkeep" -exec rm -rf {} +
mkdir -p results/fields

cargo build --release

run_case() {
  local label="$1"
  local out="$2"
  local nx="$3"
  local ny="$4"
  local nt="$5"
  shift 5

  "$@" cargo run --release -- \
    --nx "$nx" \
    --ny "$ny" \
    --re "$RE" \
    --nt "$nt" \
    --mode optimized_tolerance \
    --poisson-solver multigrid \
    --poisson-tol "$POISSON_TOL" \
    --max-vcycles "$MAX_VCYCLES" \
    --out "$out" \
    --field-tag "$label" \
    --write-fields
}

HOST_TRIPLE="$(rustc -vV | awk '/host:/ {print $2}')"
HAVE_NEON=false
if [[ "$HOST_TRIPLE" == aarch64-* ]]; then
  HAVE_NEON=true
fi

for nx in $GRIDS; do
  ny="${NY:-$nx}"
  echo "== Benchmark grid ${nx}x${ny}, Re=${RE}, nt=${NT} =="
  run_case scalar "results/benchmark_scalar_${nx}.csv" "$nx" "$ny" "$NT" env NS_FORCE_SCALAR=1
  if [[ "$HAVE_NEON" == true ]]; then
    run_case neon "results/benchmark_neon_${nx}.csv" "$nx" "$ny" "$NT" env
  else
    echo "Skipping NEON run: host is not aarch64."
  fi
done

if [[ -x .venv/bin/python ]]; then
  PYTHON=.venv/bin/python
else
  PYTHON=python3
fi

"$PYTHON" scripts/validate_ghia_centerlines.py --re "$RE"

AGGREGATE="results/benchmark_comparison.csv"
rm -f "$AGGREGATE"
first=true

for nx in $GRIDS; do
  out="results/benchmark_comparison_${nx}.csv"
  args=(
    --nx "$nx"
    --nt "$NT"
    --re "$RE"
    --scalar-csv "results/benchmark_scalar_${nx}.csv"
    --out "$out"
  )

  baseline_csv="results/profile_baseline_${nx}.csv"
  if [[ -f "$baseline_csv" ]]; then
    args+=(--baseline-csv "$baseline_csv")
  fi
  if [[ "$HAVE_NEON" == true ]]; then
    args+=(--simd-csv "results/benchmark_neon_${nx}.csv")
  fi

  "$PYTHON" scripts/summarize_benchmark.py "${args[@]}"

  if [[ "$first" == true ]]; then
    cat "$out" > "$AGGREGATE"
    first=false
  else
    tail -n +2 "$out" >> "$AGGREGATE"
  fi
done

echo
echo "Combined benchmark summary:"
cat "$AGGREGATE"

"$PYTHON" scripts/plot_benchmark_stats.py --summary "$AGGREGATE"
