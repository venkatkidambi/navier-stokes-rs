use crate::multigrid::{self, MultigridConfig, MultigridSolver};
use crate::solver::{idx_p, Mode};
use rayon::prelude::*;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PoissonSolverKind {
    Jacobi,
    Rbgs,
    Multigrid,
}

impl PoissonSolverKind {
    pub fn as_str(self) -> &'static str {
        match self {
            PoissonSolverKind::Jacobi => "jacobi",
            PoissonSolverKind::Rbgs => "rbgs",
            PoissonSolverKind::Multigrid => "multigrid",
        }
    }
}

pub struct PoissonConfig {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
    pub mode: Mode,
    pub solver: PoissonSolverKind,
    pub max_iters: usize,
    pub tolerance: f64,
    pub allow_early_stop: bool,
    pub mg_vcycles: usize,
    pub mg_pre_smooth: usize,
    pub mg_post_smooth: usize,
    pub mg_coarse_iters: usize,
    pub max_vcycles: usize,
}

pub struct PoissonSolveStats {
    pub iterations: usize,
    pub residual_initial: f64,
    pub residual_l2: f64,
    pub mg_vcycles: usize,
}

pub fn pressure_poisson(
    config: &PoissonConfig,
    rhs: &[f64],
    p: &mut [f64],
    p_next: &mut [f64],
    lap: &mut [f64],
    multigrid: Option<&mut MultigridSolver>,
) -> PoissonSolveStats {
    match config.solver {
        PoissonSolverKind::Jacobi => match config.mode {
            Mode::OptimizedTolerance => weighted_jacobi_tolerance(config, rhs, p, p_next, lap),
            _ if config.allow_early_stop => weighted_jacobi_tolerance(config, rhs, p, p_next, lap),
            _ => weighted_jacobi_fixed(config, rhs, p, p_next, lap),
        },
        PoissonSolverKind::Rbgs => match config.mode {
            Mode::OptimizedTolerance => rbgs_tolerance(config, rhs, p, lap),
            _ if config.allow_early_stop => rbgs_tolerance(config, rhs, p, lap),
            _ => rbgs_fixed(config, rhs, p, lap),
        },
        PoissonSolverKind::Multigrid => {
            let mg = multigrid.expect("multigrid solver requested without hierarchy");
            multigrid_solve(config, rhs, p, mg)
        }
    }
}

#[allow(dead_code)]
pub fn apply_mac_laplacian(config: &PoissonConfig, p: &[f64], out: &mut [f64]) {
    multigrid::apply_operator(config.nx, config.ny, config.dx, config.dy, p, out);
}

pub fn pressure_residual_l2(
    config: &PoissonConfig,
    rhs: &[f64],
    p: &[f64],
    lap: &mut [f64],
) -> f64 {
    multigrid::residual_l2(config.nx, config.ny, config.dx, config.dy, rhs, p, lap)
}

fn multigrid_solve(
    config: &PoissonConfig,
    rhs: &[f64],
    p: &mut [f64],
    multigrid: &mut MultigridSolver,
) -> PoissonSolveStats {
    let mg_cfg = MultigridConfig {
        pre_smooth: config.mg_pre_smooth,
        post_smooth: config.mg_post_smooth,
        coarse_iters: config.mg_coarse_iters,
    };
    let (cycles, initial, final_residual) = match config.mode {
        Mode::OptimizedTolerance => {
            multigrid.solve_tolerance(rhs, p, &mg_cfg, config.tolerance, config.max_vcycles)
        }
        _ if config.allow_early_stop => {
            multigrid.solve_tolerance(rhs, p, &mg_cfg, config.tolerance, config.max_vcycles)
        }
        _ => multigrid.solve_fixed(rhs, p, &mg_cfg, config.mg_vcycles),
    };
    PoissonSolveStats {
        iterations: cycles,
        residual_initial: initial,
        residual_l2: final_residual,
        mg_vcycles: cycles,
    }
}

fn weighted_jacobi_fixed(
    config: &PoissonConfig,
    rhs: &[f64],
    p: &mut [f64],
    p_next: &mut [f64],
    lap: &mut [f64],
) -> PoissonSolveStats {
    let initial = pressure_residual_l2(config, rhs, p, lap);
    for _ in 0..config.max_iters {
        jacobi_sweep(config, rhs, p, p_next, 0.85);
        p.copy_from_slice(p_next);
        subtract_mean_pressure(p);
    }
    PoissonSolveStats {
        iterations: config.max_iters,
        residual_initial: initial,
        residual_l2: pressure_residual_l2(config, rhs, p, lap),
        mg_vcycles: 0,
    }
}

fn weighted_jacobi_tolerance(
    config: &PoissonConfig,
    rhs: &[f64],
    p: &mut [f64],
    p_next: &mut [f64],
    lap: &mut [f64],
) -> PoissonSolveStats {
    let initial = pressure_residual_l2(config, rhs, p, lap);
    let mut performed = 0usize;
    let check_every = 25usize;
    let mut residual = initial;
    for iter in 0..config.max_iters {
        jacobi_sweep(config, rhs, p, p_next, 0.85);
        p.copy_from_slice(p_next);
        subtract_mean_pressure(p);
        performed = iter + 1;
        if performed % check_every == 0 {
            residual = pressure_residual_l2(config, rhs, p, lap);
            if residual < config.tolerance {
                break;
            }
        }
    }
    if performed % check_every != 0 {
        residual = pressure_residual_l2(config, rhs, p, lap);
    }
    PoissonSolveStats {
        iterations: performed,
        residual_initial: initial,
        residual_l2: residual,
        mg_vcycles: 0,
    }
}

fn rbgs_fixed(
    config: &PoissonConfig,
    rhs: &[f64],
    p: &mut [f64],
    lap: &mut [f64],
) -> PoissonSolveStats {
    let initial = pressure_residual_l2(config, rhs, p, lap);
    multigrid::rbgs_smooth(
        config.nx,
        config.ny,
        config.dx,
        config.dy,
        rhs,
        p,
        config.max_iters,
    );
    PoissonSolveStats {
        iterations: config.max_iters,
        residual_initial: initial,
        residual_l2: pressure_residual_l2(config, rhs, p, lap),
        mg_vcycles: 0,
    }
}

fn rbgs_tolerance(
    config: &PoissonConfig,
    rhs: &[f64],
    p: &mut [f64],
    lap: &mut [f64],
) -> PoissonSolveStats {
    let initial = pressure_residual_l2(config, rhs, p, lap);
    let mut residual = initial;
    let mut performed = 0usize;
    for _ in 0..config.max_iters {
        multigrid::rbgs_smooth(config.nx, config.ny, config.dx, config.dy, rhs, p, 1);
        performed += 1;
        if performed % 25 == 0 {
            residual = pressure_residual_l2(config, rhs, p, lap);
            if residual < config.tolerance {
                break;
            }
        }
    }
    if performed % 25 != 0 {
        residual = pressure_residual_l2(config, rhs, p, lap);
    }
    PoissonSolveStats {
        iterations: performed,
        residual_initial: initial,
        residual_l2: residual,
        mg_vcycles: 0,
    }
}

fn jacobi_sweep(config: &PoissonConfig, rhs: &[f64], p: &[f64], p_next: &mut [f64], omega: f64) {
    let nx = config.nx;
    let ny = config.ny;
    let inv_dx2 = 1.0 / (config.dx * config.dx);
    let inv_dy2 = 1.0 / (config.dy * config.dy);
    p_next.par_chunks_mut(nx).enumerate().for_each(|(j, row)| {
        for i in 0..nx {
            let c = idx_p(i, j, nx);
            let mut coeff_sum = 0.0;
            let mut neighbor_sum = 0.0;
            if i > 0 {
                coeff_sum += inv_dx2;
                neighbor_sum += p[idx_p(i - 1, j, nx)] * inv_dx2;
            }
            if i + 1 < nx {
                coeff_sum += inv_dx2;
                neighbor_sum += p[idx_p(i + 1, j, nx)] * inv_dx2;
            }
            if j > 0 {
                coeff_sum += inv_dy2;
                neighbor_sum += p[idx_p(i, j - 1, nx)] * inv_dy2;
            }
            if j + 1 < ny {
                coeff_sum += inv_dy2;
                neighbor_sum += p[idx_p(i, j + 1, nx)] * inv_dy2;
            }
            let jacobi = (neighbor_sum - rhs[c]) / coeff_sum;
            row[i] = (1.0 - omega) * p[c] + omega * jacobi;
        }
    });
}

fn subtract_mean_pressure(p: &mut [f64]) {
    let mean = p.par_iter().sum::<f64>() / p.len() as f64;
    p.par_iter_mut().for_each(|value| *value -= mean);
}
