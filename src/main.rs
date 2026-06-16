mod io;
mod kernels;
mod metrics;
mod multigrid;
mod poisson;
mod solver;

use clap::{Parser, ValueEnum};
use poisson::PoissonSolverKind;
use solver::{run_cavity, AdaptiveToleranceSchedule, Mode, SimulationConfig, ToleranceStrategy};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Rust 2D lid-driven cavity Navier-Stokes solver"
)]
struct Cli {
    #[arg(long, default_value_t = 256)]
    nx: usize,
    #[arg(long, default_value_t = 256)]
    ny: usize,
    #[arg(long, default_value_t = 400.0)]
    re: f64,
    #[arg(long, default_value_t = 1000)]
    nt: usize,
    #[arg(long, value_enum, default_value_t = CliMode::OptimizedTolerance)]
    mode: CliMode,
    #[arg(long, default_value = "results/rust_summary.csv")]
    out: PathBuf,
    #[arg(long)]
    fields_prefix: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    write_fields: bool,
    #[arg(long, default_value = "")]
    field_tag: String,
    #[arg(long)]
    timestep_history: Option<PathBuf>,
    #[arg(
        long = "poisson-iters-per-step",
        alias = "poisson-iters",
        default_value_t = 200
    )]
    poisson_iters_per_step: usize,
    #[arg(long, default_value_t = 1.0e-8)]
    poisson_tol: f64,
    #[arg(long, default_value_t = false)]
    allow_early_stop: bool,
    #[arg(long, value_enum, default_value_t = CliPoissonSolver::Multigrid)]
    poisson_solver: CliPoissonSolver,
    #[arg(long, default_value = "auto")]
    mg_levels: String,
    #[arg(long, default_value_t = 5)]
    mg_vcycles: usize,
    #[arg(long, default_value_t = 2)]
    mg_pre_smooth: usize,
    #[arg(long, default_value_t = 2)]
    mg_post_smooth: usize,
    #[arg(long, default_value_t = 50)]
    mg_coarse_iters: usize,
    #[arg(long, default_value_t = 20)]
    max_vcycles: usize,
    #[arg(long, value_enum, default_value_t = CliToleranceStrategy::Fixed)]
    tolerance_strategy: CliToleranceStrategy,
    #[arg(long, default_value_t = 2.0)]
    adaptive_t1: f64,
    #[arg(long, default_value_t = 5.0)]
    adaptive_t2: f64,
    #[arg(long, default_value_t = 1.0e-6)]
    adaptive_tol_early: f64,
    #[arg(long, default_value_t = 1.0e-8)]
    adaptive_tol_middle: f64,
    #[arg(long, default_value_t = 1.0e-10)]
    adaptive_tol_late: f64,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum CliMode {
    #[value(
        name = "algorithm_fair_fixed_iters",
        alias = "algorithm_fair",
        alias = "algorithm-fair"
    )]
    AlgorithmFairFixedIters,
    #[value(
        name = "optimized_fixed_iters",
        alias = "optimized_fixed",
        alias = "optimized"
    )]
    OptimizedFixedIters,
    #[value(name = "optimized_tolerance")]
    OptimizedTolerance,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum CliPoissonSolver {
    Jacobi,
    Rbgs,
    Multigrid,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum CliToleranceStrategy {
    Fixed,
    Adaptive,
}

impl From<CliPoissonSolver> for PoissonSolverKind {
    fn from(value: CliPoissonSolver) -> Self {
        match value {
            CliPoissonSolver::Jacobi => PoissonSolverKind::Jacobi,
            CliPoissonSolver::Rbgs => PoissonSolverKind::Rbgs,
            CliPoissonSolver::Multigrid => PoissonSolverKind::Multigrid,
        }
    }
}

impl From<CliToleranceStrategy> for ToleranceStrategy {
    fn from(value: CliToleranceStrategy) -> Self {
        match value {
            CliToleranceStrategy::Fixed => ToleranceStrategy::Fixed,
            CliToleranceStrategy::Adaptive => ToleranceStrategy::Adaptive,
        }
    }
}

impl From<CliMode> for Mode {
    fn from(value: CliMode) -> Self {
        match value {
            CliMode::AlgorithmFairFixedIters => Mode::AlgorithmFairFixedIters,
            CliMode::OptimizedFixedIters => Mode::OptimizedFixedIters,
            CliMode::OptimizedTolerance => Mode::OptimizedTolerance,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let config = SimulationConfig {
        nx: cli.nx,
        ny: cli.ny,
        re: cli.re,
        nt: cli.nt,
        mode: cli.mode.into(),
        poisson_iters_per_step: cli.poisson_iters_per_step,
        poisson_tol: cli.poisson_tol,
        allow_early_stop: cli.allow_early_stop,
        poisson_solver: cli.poisson_solver.into(),
        mg_levels: cli.mg_levels,
        mg_vcycles: cli.mg_vcycles,
        mg_pre_smooth: cli.mg_pre_smooth,
        mg_post_smooth: cli.mg_post_smooth,
        mg_coarse_iters: cli.mg_coarse_iters,
        max_vcycles: cli.max_vcycles,
        tolerance_strategy: cli.tolerance_strategy.into(),
        adaptive_schedule: AdaptiveToleranceSchedule {
            first_time: cli.adaptive_t1,
            second_time: cli.adaptive_t2,
            early_tol: cli.adaptive_tol_early,
            middle_tol: cli.adaptive_tol_middle,
            late_tol: cli.adaptive_tol_late,
        },
        timestep_history: cli.timestep_history.clone(),
    };

    let result = run_cavity(&config);
    io::write_summary(&cli.out, &result.summary)?;
    if let Some(history_path) = &config.timestep_history {
        io::write_timestep_history(history_path, &result.timestep_history)?;
    }
    eprintln!(
        "divergence_rms before projection: {:.6e}, after projection: {:.6e}",
        result.summary.divergence_rms_before_projection,
        result.summary.divergence_rms_after_projection
    );

    if let Some(prefix) = cli.fields_prefix {
        io::write_fields(
            &prefix, config.nx, config.ny, &result.u, &result.v, &result.p,
        )?;
    }

    if cli.write_fields {
        io::write_validation_fields(
            &PathBuf::from("results/fields"),
            "rust",
            &cli.field_tag,
            config.tolerance_strategy.as_str(),
            config.nx,
            config.ny,
            config.re,
            result.summary.dt,
            config.nt,
            &result.u,
            &result.v,
            &result.p,
        )?;
    }

    Ok(())
}
