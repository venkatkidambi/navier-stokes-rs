# navier-stokes-rs

Rust-native, optimized 2D incompressible Navier-Stokes solver for the lid-driven cavity problem.

The objective of this project is to write a native Rust CFD solver for the classic 2D lid-driven cavity problem and then push its performance as far as is reasonable without changing the numerical method. The solver uses a staggered MAC-grid finite-difference projection method with explicit advection/diffusion and a pressure Poisson projection. It includes Jacobi, red-black Gauss-Seidel, and geometric multigrid pressure solvers, plus CSV export for summaries, timestep diagnostics, and final fields.

Performance work is focused on the pressure projection path, which dominates runtime. The current optimized path uses:

- Flat `Vec<f64>` structure-of-arrays field storage for `u`, `v`, and `p`.
- Geometric multigrid V-cycles for pressure convergence.
- Red-black Gauss-Seidel smoothing with branch-light row/parity loops.
- Small isolated unsafe pointer kernels for the hottest RBGS interior loop.
- Apple Silicon `aarch64` NEON kernels for contiguous reductions, mean subtraction, and pressure residual stencils.
- Scalar fallback paths and equivalence tests for SIMD-accelerated kernels.

## Performance Snapshot

Run:

```bash
scripts/benchmark.sh
```

The benchmark compares the optimized scalar path against the Apple Silicon NEON/SIMD path, validates the exported fields against Ghia centerline data, and writes a compact summary to `results/benchmark_comparison.csv`.

Benchmark runs clean generated files under `results/` and `figures/` first, so each summary reflects the current run rather than accumulated historical files.

## Problem

The default validation target is the classic square lid-driven cavity:

- Domain: `[0, 1] x [0, 1]`
- No-slip walls
- Top lid moves right with `u = 1`, `v = 0`
- Other walls use `u = 0`, `v = 0`
- Pressure uses homogeneous Neumann boundaries with one pressure nullspace removal
- Built-in validation data: Ghia et al. centerline profiles for `Re=100`

## Numerical Method

The solver stores fields on a MAC grid:

- Pressure `p`: cell centers, `nx by ny`
- Horizontal velocity `u`: vertical faces, `(nx + 1) by ny`
- Vertical velocity `v`: horizontal faces, `nx by (ny + 1)`

Each timestep:

1. Apply velocity and pressure boundary conditions.
2. Build tentative velocities from explicit advection and diffusion.
3. Build the pressure Poisson right-hand side from tentative face-velocity divergence.
4. Solve the pressure Poisson equation using the MAC `D*G` operator.
5. Project velocity with the pressure gradient.
6. Reapply boundary conditions.
7. Report divergence and kinetic energy diagnostics.

The timestep is selected automatically:

```text
dt = min(0.25 h, 0.20 h^2 / nu), h = min(dx, dy), nu = 1 / Re
```

## Build And Test

```bash
cargo test
cargo build --release
```

Run a small `Re=100` smoke case:

```bash
cargo run --release -- \
  --nx 32 --ny 32 --re 100 --nt 20 \
  --mode optimized_tolerance \
  --poisson-solver multigrid \
  --poisson-tol 1e-8 \
  --max-vcycles 20 \
  --out results/smoke_re100.csv
```

## Validation

Run the compact validation script:

```bash
scripts/validate.sh
```

The script runs a small `Re=100` Rust case, exports final MAC fields, and compares centerline velocities against the Ghia et al. reference data. It writes:

- `results/validation_re100.csv`
- `results/ghia_validation_summary.csv`
- `results/fields/rust_mac_N64_Re100_*`

If `matplotlib` is available, it also writes:

- `figures/ghia_u_centerline_Re100.png`
- `figures/ghia_v_centerline_Re100.png`
- `figures/ghia_error_summary_Re100.png`

For a higher-resolution validation run:

```bash
cargo run --release -- \
  --nx 256 --ny 256 --re 100 --nt 1000 \
  --mode optimized_tolerance \
  --poisson-solver multigrid \
  --poisson-tol 1e-8 \
  --max-vcycles 20 \
  --out results/validation_re100_256.csv \
  --write-fields

python3 scripts/validate_ghia_centerlines.py
```

Ghia profiles are steady-state reference data. Short smoke runs are useful for checking the pipeline, but longer runs are needed for meaningful centerline agreement. When multiple exported field cases exist, the validation plots show one representative case per run label to keep the figures readable. Use `--plot-all` with `scripts/validate_ghia_centerlines.py` if you want every exported case on the plots.

## Profiling And Benchmarks

Print a multigrid kernel timing breakdown for one run:

```bash
NS_PROFILE=1 cargo run --release -- \
  --nx 128 --ny 128 --re 100 --nt 200 \
  --mode optimized_tolerance \
  --poisson-solver multigrid \
  --out results/profile_kernel_128.csv
```

This prints the multigrid kernel timing breakdown, including RBGS smoothing, residual computation, restriction, prolongation, and mean subtraction.

Compare the current optimized scalar path against the NEON/SIMD path on Apple Silicon:

```bash
scripts/benchmark.sh
```

Useful environment knobs:

- `NS_FORCE_SCALAR=1`: disable NEON kernels and use scalar implementations.
- `scripts/benchmark.sh`: run the default `64 128 256` grid sweep.
- `NX=256 NT=500 scripts/benchmark.sh`: run one grid size.
- `GRIDS="64 128" NT=500 scripts/benchmark.sh`: run a custom sweep.

The benchmark writes `results/benchmark_comparison.csv` with runtime, speedup, Ghia centerline errors, pressure residual, and divergence metrics. It also writes local runtime plots under `figures/`.

Benchmark runs start by cleaning generated files under `results/` and `figures/`, preserving only `.gitkeep`, so each run leaves a compact current result set.

## Output

Summary CSVs include timing, pressure-solver, divergence, and kinetic-energy diagnostics. Key fields include:

- `total_s`, `ms_per_step`
- `poisson_solver`, `pressure_residual_final`
- `divergence_rms_before_projection`
- `divergence_rms_after_projection`
- `divergence_reduction_factor`
- `max_abs_divergence_after_projection`
- `kinetic_energy`

Final field export writes MAC-grid `u`, `v`, `p`, and metadata files under `results/fields/`.

## Project Layout

```text
.
├── Cargo.toml
├── README.md
├── figures/
├── results/
├── scripts/
└── src/
```

The Rust tests include projection and divergence checks, multigrid residual reduction, operator consistency, and a small cavity sanity run.
