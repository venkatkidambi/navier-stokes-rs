use crate::metrics::{compute_mac_divergence, divergence_rms, divergence_stats, kinetic_energy};
use crate::multigrid::MultigridSolver;
use crate::poisson::{pressure_poisson, PoissonConfig, PoissonSolverKind};
use rayon::prelude::*;
use serde::Serialize;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Mode {
    AlgorithmFairFixedIters,
    OptimizedFixedIters,
    OptimizedTolerance,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::AlgorithmFairFixedIters => "algorithm_fair_fixed_iters",
            Mode::OptimizedFixedIters => "optimized_fixed_iters",
            Mode::OptimizedTolerance => "optimized_tolerance",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ToleranceStrategy {
    Fixed,
    Adaptive,
}

impl ToleranceStrategy {
    pub fn as_str(self) -> &'static str {
        match self {
            ToleranceStrategy::Fixed => "fixed",
            ToleranceStrategy::Adaptive => "adaptive",
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct AdaptiveToleranceSchedule {
    pub first_time: f64,
    pub second_time: f64,
    pub early_tol: f64,
    pub middle_tol: f64,
    pub late_tol: f64,
}

impl AdaptiveToleranceSchedule {
    pub fn tolerance_at(self, physical_time: f64) -> f64 {
        if physical_time < self.first_time {
            self.early_tol
        } else if physical_time < self.second_time {
            self.middle_tol
        } else {
            self.late_tol
        }
    }
}

#[derive(Clone, Debug)]
pub struct SimulationConfig {
    pub nx: usize,
    pub ny: usize,
    pub re: f64,
    pub nt: usize,
    pub mode: Mode,
    pub poisson_iters_per_step: usize,
    pub poisson_tol: f64,
    pub allow_early_stop: bool,
    pub poisson_solver: PoissonSolverKind,
    pub mg_levels: String,
    pub mg_vcycles: usize,
    pub mg_pre_smooth: usize,
    pub mg_post_smooth: usize,
    pub mg_coarse_iters: usize,
    pub max_vcycles: usize,
    pub tolerance_strategy: ToleranceStrategy,
    pub adaptive_schedule: AdaptiveToleranceSchedule,
    pub timestep_history: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
pub struct BenchmarkSummary {
    pub language: String,
    pub run_label: String,
    pub mode: String,
    pub tolerance_strategy: String,
    pub nx: usize,
    pub ny: usize,
    pub re: f64,
    pub dt: f64,
    pub nt: usize,
    pub total_s: f64,
    pub ms_per_step: f64,
    pub update_ms: f64,
    pub poisson_ms: f64,
    pub projection_ms: f64,
    pub bc_ms: f64,
    pub poisson_solver: String,
    pub poisson_iters_per_step: usize,
    pub total_poisson_iterations: usize,
    pub mg_vcycles_per_step: usize,
    pub total_mg_vcycles: usize,
    pub avg_mg_vcycles_per_step: f64,
    pub mg_pre_smooth: usize,
    pub mg_post_smooth: usize,
    pub mg_coarse_iters: usize,
    pub pressure_residual_initial: f64,
    pub pressure_residual_avg: f64,
    pub pressure_residual_final: f64,
    pub pressure_residual_reduction_factor: f64,
    pub avg_poisson_residual: f64,
    pub final_poisson_residual: f64,
    pub divergence_rms_before_projection: f64,
    pub divergence_rms_after_projection: f64,
    pub divergence_reduction_factor: f64,
    pub max_abs_divergence_after_projection: f64,
    pub divergence_l2_raw: f64,
    pub divergence_rms: f64,
    pub max_abs_divergence: f64,
    pub divergence_l2: f64,
    pub kinetic_energy: f64,
}

pub struct SimulationResult {
    pub summary: BenchmarkSummary,
    pub timestep_history: Vec<TimestepDiagnostic>,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub p: Vec<f64>,
}

#[derive(Debug, Serialize)]
pub struct TimestepDiagnostic {
    pub timestep: usize,
    pub physical_time: f64,
    pub active_tolerance: f64,
    pub mg_vcycles_used: usize,
    pub final_poisson_residual: f64,
    pub divergence_rms_after_projection: f64,
}

#[inline]
pub fn idx_p(i: usize, j: usize, nx: usize) -> usize {
    i + nx * j
}

#[inline]
pub fn idx_u(i: usize, j: usize, nx: usize) -> usize {
    i + (nx + 1) * j
}

#[inline]
pub fn idx_v(i: usize, j: usize, nx: usize) -> usize {
    i + nx * j
}

pub fn stable_dt(nx: usize, ny: usize, re: f64) -> f64 {
    let dx = 1.0 / nx as f64;
    let dy = 1.0 / ny as f64;
    let h = dx.min(dy);
    let nu = 1.0 / re;
    let convective = 0.25 * h;
    let diffusive = 0.20 * h * h / nu;
    convective.min(diffusive)
}

pub fn run_cavity(config: &SimulationConfig) -> SimulationResult {
    assert!(
        config.nx >= 4 && config.ny >= 4,
        "grid must be at least 4x4"
    );
    assert!(config.re > 0.0, "Reynolds number must be positive");
    if config.poisson_solver == PoissonSolverKind::Multigrid {
        assert!(
            config.mg_levels == "auto",
            "only --mg-levels auto is currently implemented"
        );
    }

    let dx = 1.0 / config.nx as f64;
    let dy = 1.0 / config.ny as f64;
    let nu = 1.0 / config.re;
    let dt = stable_dt(config.nx, config.ny, config.re);

    let mut u = vec![0.0; (config.nx + 1) * config.ny];
    let mut v = vec![0.0; config.nx * (config.ny + 1)];
    let mut p = vec![0.0; config.nx * config.ny];
    let mut u_tmp = vec![0.0; u.len()];
    let mut v_tmp = vec![0.0; v.len()];
    let mut rhs = vec![0.0; p.len()];
    let mut div = vec![0.0; p.len()];
    let mut p_next = vec![0.0; p.len()];
    let mut lap = vec![0.0; p.len()];

    let mut update_time = Duration::ZERO;
    let mut poisson_time = Duration::ZERO;
    let mut projection_time = Duration::ZERO;
    let mut bc_time = Duration::ZERO;
    let mut total_poisson_iterations = 0usize;
    let mut total_mg_vcycles = 0usize;
    let mut pressure_residual_initial = 0.0;
    let mut pressure_residual_sum = 0.0;
    let mut pressure_residual_final = 0.0;
    let mut divergence_rms_before_projection = 0.0;
    let mut divergence_rms_after_projection = 0.0;
    let mut max_abs_divergence_after_projection = 0.0;
    let mut timestep_history = if config.timestep_history.is_some() {
        Vec::with_capacity(config.nt)
    } else {
        Vec::new()
    };

    let t0 = Instant::now();
    apply_velocity_bc(&mut u, &mut v, config.nx, config.ny);
    let mut multigrid = if config.poisson_solver == PoissonSolverKind::Multigrid {
        Some(MultigridSolver::new(config.nx, config.ny, dx, dy))
    } else {
        None
    };

    for step in 0..config.nt {
        let physical_time = step as f64 * dt;
        let active_tolerance = match config.tolerance_strategy {
            ToleranceStrategy::Fixed => config.poisson_tol,
            ToleranceStrategy::Adaptive => config.adaptive_schedule.tolerance_at(physical_time),
        };

        let tb = Instant::now();
        apply_velocity_bc(&mut u, &mut v, config.nx, config.ny);
        bc_time += tb.elapsed();

        let tu = Instant::now();
        update_tentative_mac(
            config.nx, config.ny, dx, dy, dt, nu, &u, &v, &mut u_tmp, &mut v_tmp,
        );
        apply_velocity_bc(&mut u_tmp, &mut v_tmp, config.nx, config.ny);
        build_pressure_rhs(
            config.nx, config.ny, dx, dy, dt, &u_tmp, &v_tmp, &mut rhs, &mut div,
        );
        divergence_rms_before_projection = divergence_rms(&div);
        update_time += tu.elapsed();

        let tp = Instant::now();
        let poisson_config = PoissonConfig {
            nx: config.nx,
            ny: config.ny,
            dx,
            dy,
            mode: config.mode,
            solver: config.poisson_solver,
            max_iters: config.poisson_iters_per_step,
            tolerance: active_tolerance,
            allow_early_stop: config.allow_early_stop,
            mg_vcycles: config.mg_vcycles,
            mg_pre_smooth: config.mg_pre_smooth,
            mg_post_smooth: config.mg_post_smooth,
            mg_coarse_iters: config.mg_coarse_iters,
            max_vcycles: config.max_vcycles,
        };
        let poisson_stats = pressure_poisson(
            &poisson_config,
            &rhs,
            &mut p,
            &mut p_next,
            &mut lap,
            multigrid.as_mut(),
        );
        total_poisson_iterations += poisson_stats.iterations;
        total_mg_vcycles += poisson_stats.mg_vcycles;
        pressure_residual_initial = poisson_stats.residual_initial;
        pressure_residual_sum += poisson_stats.residual_l2;
        pressure_residual_final = poisson_stats.residual_l2;
        poisson_time += tp.elapsed();

        let tproj = Instant::now();
        project_velocity(
            config.nx, config.ny, dx, dy, dt, &p, &u_tmp, &v_tmp, &mut u, &mut v,
        );
        projection_time += tproj.elapsed();

        let tb2 = Instant::now();
        apply_velocity_bc(&mut u, &mut v, config.nx, config.ny);
        bc_time += tb2.elapsed();

        compute_mac_divergence(config.nx, config.ny, dx, dy, &u, &v, &mut div);
        let after_stats = divergence_stats(&div);
        divergence_rms_after_projection = after_stats.rms;
        max_abs_divergence_after_projection = after_stats.max_abs;

        if config.timestep_history.is_some() {
            timestep_history.push(TimestepDiagnostic {
                timestep: step + 1,
                physical_time: (step + 1) as f64 * dt,
                active_tolerance,
                mg_vcycles_used: poisson_stats.mg_vcycles,
                final_poisson_residual: poisson_stats.residual_l2,
                divergence_rms_after_projection,
            });
        }
    }

    let total = t0.elapsed();
    compute_mac_divergence(config.nx, config.ny, dx, dy, &u, &v, &mut div);
    let final_div = divergence_stats(&div);
    let ke = kinetic_energy(config.nx, config.ny, dx, dy, &u, &v);
    let reduction = if divergence_rms_after_projection > 0.0 {
        divergence_rms_before_projection / divergence_rms_after_projection
    } else {
        f64::INFINITY
    };
    let pressure_reduction = if pressure_residual_final > 0.0 {
        pressure_residual_initial / pressure_residual_final
    } else {
        f64::INFINITY
    };

    SimulationResult {
        summary: BenchmarkSummary {
            language: "rust".to_string(),
            run_label: config.tolerance_strategy.as_str().to_string(),
            mode: config.mode.as_str().to_string(),
            tolerance_strategy: config.tolerance_strategy.as_str().to_string(),
            nx: config.nx,
            ny: config.ny,
            re: config.re,
            dt,
            nt: config.nt,
            total_s: total.as_secs_f64(),
            ms_per_step: 1000.0 * total.as_secs_f64() / config.nt as f64,
            update_ms: 1000.0 * update_time.as_secs_f64(),
            poisson_ms: 1000.0 * poisson_time.as_secs_f64(),
            projection_ms: 1000.0 * projection_time.as_secs_f64(),
            bc_ms: 1000.0 * bc_time.as_secs_f64(),
            poisson_solver: config.poisson_solver.as_str().to_string(),
            poisson_iters_per_step: config.poisson_iters_per_step,
            total_poisson_iterations,
            mg_vcycles_per_step: config.mg_vcycles,
            total_mg_vcycles,
            avg_mg_vcycles_per_step: total_mg_vcycles as f64 / config.nt as f64,
            mg_pre_smooth: config.mg_pre_smooth,
            mg_post_smooth: config.mg_post_smooth,
            mg_coarse_iters: config.mg_coarse_iters,
            pressure_residual_initial,
            pressure_residual_avg: pressure_residual_sum / config.nt as f64,
            pressure_residual_final,
            pressure_residual_reduction_factor: pressure_reduction,
            avg_poisson_residual: pressure_residual_sum / config.nt as f64,
            final_poisson_residual: pressure_residual_final,
            divergence_rms_before_projection,
            divergence_rms_after_projection,
            divergence_reduction_factor: reduction,
            max_abs_divergence_after_projection,
            divergence_l2_raw: final_div.raw_l2,
            divergence_rms: final_div.rms,
            max_abs_divergence: final_div.max_abs,
            divergence_l2: final_div.rms,
            kinetic_energy: ke,
        },
        timestep_history,
        u,
        v,
        p,
    }
}

fn update_tentative_mac(
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    dt: f64,
    nu: f64,
    u: &[f64],
    v: &[f64],
    u_tmp: &mut [f64],
    v_tmp: &mut [f64],
) {
    u_tmp.copy_from_slice(u);
    v_tmp.copy_from_slice(v);

    let inv_2dx = 0.5 / dx;
    let inv_2dy = 0.5 / dy;
    let inv_dx2 = 1.0 / (dx * dx);
    let inv_dy2 = 1.0 / (dy * dy);

    u_tmp
        .par_chunks_mut(nx + 1)
        .enumerate()
        .for_each(|(j, row)| {
            for i in 1..nx {
                let c = idx_u(i, j, nx);
                let u_c = u[c];
                let du_dx = (u[idx_u(i + 1, j, nx)] - u[idx_u(i - 1, j, nx)]) * inv_2dx;
                let u_s = if j > 0 { u[idx_u(i, j - 1, nx)] } else { -u_c };
                let u_n = if j + 1 < ny {
                    u[idx_u(i, j + 1, nx)]
                } else {
                    2.0 - u_c
                };
                let du_dy = (u_n - u_s) * inv_2dy;
                let v_at_u = 0.25
                    * (v[idx_v(i - 1, j, nx)]
                        + v[idx_v(i, j, nx)]
                        + v[idx_v(i - 1, j + 1, nx)]
                        + v[idx_v(i, j + 1, nx)]);
                let lap = (u[idx_u(i + 1, j, nx)] - 2.0 * u_c + u[idx_u(i - 1, j, nx)]) * inv_dx2
                    + (u_n - 2.0 * u_c + u_s) * inv_dy2;
                row[i] = u_c + dt * (-(u_c * du_dx + v_at_u * du_dy) + nu * lap);
            }
        });

    v_tmp.par_chunks_mut(nx).enumerate().for_each(|(j, row)| {
        if j == 0 || j == ny {
            return;
        }
        for i in 0..nx {
            let c = idx_v(i, j, nx);
            let v_c = v[c];
            let v_w = if i > 0 { v[idx_v(i - 1, j, nx)] } else { -v_c };
            let v_e = if i + 1 < nx {
                v[idx_v(i + 1, j, nx)]
            } else {
                -v_c
            };
            let dv_dx = (v_e - v_w) * inv_2dx;
            let dv_dy = (v[idx_v(i, j + 1, nx)] - v[idx_v(i, j - 1, nx)]) * inv_2dy;
            let u_at_v = 0.25
                * (u[idx_u(i, j - 1, nx)]
                    + u[idx_u(i + 1, j - 1, nx)]
                    + u[idx_u(i, j, nx)]
                    + u[idx_u(i + 1, j, nx)]);
            let lap = (v_e - 2.0 * v_c + v_w) * inv_dx2
                + (v[idx_v(i, j + 1, nx)] - 2.0 * v_c + v[idx_v(i, j - 1, nx)]) * inv_dy2;
            row[i] = v_c + dt * (-(u_at_v * dv_dx + v_c * dv_dy) + nu * lap);
        }
    });
}

fn build_pressure_rhs(
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    dt: f64,
    u_tmp: &[f64],
    v_tmp: &[f64],
    rhs: &mut [f64],
    div: &mut [f64],
) {
    compute_mac_divergence(nx, ny, dx, dy, u_tmp, v_tmp, div);
    rhs.par_iter_mut()
        .zip(div.par_iter())
        .for_each(|(rhs_value, div_value)| *rhs_value = *div_value / dt);
    let mean = rhs.par_iter().sum::<f64>() / rhs.len() as f64;
    rhs.par_iter_mut().for_each(|value| *value -= mean);
}

fn project_velocity(
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    dt: f64,
    p: &[f64],
    u_tmp: &[f64],
    v_tmp: &[f64],
    u: &mut [f64],
    v: &mut [f64],
) {
    u.copy_from_slice(u_tmp);
    v.copy_from_slice(v_tmp);
    u.par_chunks_mut(nx + 1).enumerate().for_each(|(j, row)| {
        for i in 1..nx {
            row[i] =
                u_tmp[idx_u(i, j, nx)] - dt * (p[idx_p(i, j, nx)] - p[idx_p(i - 1, j, nx)]) / dx;
        }
    });
    v.par_chunks_mut(nx).enumerate().for_each(|(j, row)| {
        if j == 0 || j == ny {
            return;
        }
        for i in 0..nx {
            row[i] =
                v_tmp[idx_v(i, j, nx)] - dt * (p[idx_p(i, j, nx)] - p[idx_p(i, j - 1, nx)]) / dy;
        }
    });
}

pub fn apply_velocity_bc(u: &mut [f64], v: &mut [f64], nx: usize, ny: usize) {
    for j in 0..ny {
        u[idx_u(0, j, nx)] = 0.0;
        u[idx_u(nx, j, nx)] = 0.0;
    }
    for i in 0..nx {
        v[idx_v(i, 0, nx)] = 0.0;
        v[idx_v(i, ny, nx)] = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poisson::apply_mac_laplacian;

    #[test]
    fn lid_driven_cavity_sanity_run_is_finite() {
        let config = SimulationConfig {
            nx: 32,
            ny: 32,
            re: 100.0,
            nt: 100,
            mode: Mode::OptimizedFixedIters,
            poisson_iters_per_step: 500,
            poisson_tol: 1.0e-10,
            allow_early_stop: false,
            poisson_solver: PoissonSolverKind::Multigrid,
            mg_levels: "auto".to_string(),
            mg_vcycles: 8,
            mg_pre_smooth: 2,
            mg_post_smooth: 2,
            mg_coarse_iters: 100,
            max_vcycles: 20,
            tolerance_strategy: ToleranceStrategy::Fixed,
            adaptive_schedule: AdaptiveToleranceSchedule {
                first_time: 2.0,
                second_time: 5.0,
                early_tol: 1.0e-6,
                middle_tol: 1.0e-8,
                late_tol: 1.0e-10,
            },
            timestep_history: None,
        };
        let result = run_cavity(&config);
        assert!(result.summary.total_s > 0.0);
        assert!(result.summary.kinetic_energy.is_finite());
        assert!(result.summary.kinetic_energy > 0.0);
        assert!(result.summary.divergence_rms_after_projection.is_finite());
        assert!(result.summary.divergence_rms_after_projection < 1.0e-4);
    }

    #[test]
    fn projection_only_random_face_velocity_reduces_divergence_by_six_orders() {
        let nx = 16;
        let ny = 16;
        let dx = 1.0 / nx as f64;
        let dy = 1.0 / ny as f64;
        let dt = 0.01;
        let mut u = vec![0.0; (nx + 1) * ny];
        let mut v = vec![0.0; nx * (ny + 1)];
        let mut p = vec![0.0; nx * ny];
        let mut p_next = vec![0.0; nx * ny];
        let mut rhs = vec![0.0; nx * ny];
        let mut div = vec![0.0; nx * ny];
        let mut lap = vec![0.0; nx * ny];
        let mut u_projected = u.clone();
        let mut v_projected = v.clone();
        let mut seed = 0x1234_5678_u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let unit = ((seed >> 32) as f64) / (u32::MAX as f64);
            2.0 * unit - 1.0
        };

        for j in 0..ny {
            for i in 1..nx {
                u[idx_u(i, j, nx)] = next();
            }
        }
        for j in 1..ny {
            for i in 0..nx {
                v[idx_v(i, j, nx)] = next();
            }
        }
        apply_velocity_bc(&mut u, &mut v, nx, ny);
        build_pressure_rhs(nx, ny, dx, dy, dt, &u, &v, &mut rhs, &mut div);
        let before = divergence_rms(&div);
        let config = PoissonConfig {
            nx,
            ny,
            dx,
            dy,
            mode: Mode::OptimizedTolerance,
            solver: PoissonSolverKind::Multigrid,
            max_iters: 200_000,
            tolerance: 1.0e-10,
            allow_early_stop: true,
            mg_vcycles: 5,
            mg_pre_smooth: 3,
            mg_post_smooth: 3,
            mg_coarse_iters: 200,
            max_vcycles: 50,
        };
        let mut multigrid = MultigridSolver::new(nx, ny, dx, dy);
        let stats = pressure_poisson(
            &config,
            &rhs,
            &mut p,
            &mut p_next,
            &mut lap,
            Some(&mut multigrid),
        );
        project_velocity(
            nx,
            ny,
            dx,
            dy,
            dt,
            &p,
            &u,
            &v,
            &mut u_projected,
            &mut v_projected,
        );
        apply_velocity_bc(&mut u_projected, &mut v_projected, nx, ny);
        compute_mac_divergence(nx, ny, dx, dy, &u_projected, &v_projected, &mut div);
        let after = divergence_rms(&div);
        assert!(
            after < before * 1.0e-6,
            "projection did not reduce divergence by 1e6: before={before}, after={after}, pressure_residual={}",
            stats.residual_l2
        );
    }

    #[test]
    fn mac_laplacian_matches_discrete_divergence_of_gradient() {
        let nx = 10;
        let ny = 9;
        let dx = 1.0 / nx as f64;
        let dy = 1.0 / ny as f64;
        let mut p = vec![0.0; nx * ny];
        let mut u_grad = vec![0.0; (nx + 1) * ny];
        let mut v_grad = vec![0.0; nx * (ny + 1)];
        let mut div_grad = vec![0.0; nx * ny];
        let mut lap = vec![0.0; nx * ny];

        for j in 0..ny {
            for i in 0..nx {
                p[idx_p(i, j, nx)] =
                    ((i as f64 + 0.25) * 0.73).sin() + ((j as f64 + 0.5) * 0.41).cos();
            }
        }
        for j in 0..ny {
            for i in 1..nx {
                u_grad[idx_u(i, j, nx)] = (p[idx_p(i, j, nx)] - p[idx_p(i - 1, j, nx)]) / dx;
            }
        }
        for j in 1..ny {
            for i in 0..nx {
                v_grad[idx_v(i, j, nx)] = (p[idx_p(i, j, nx)] - p[idx_p(i, j - 1, nx)]) / dy;
            }
        }
        compute_mac_divergence(nx, ny, dx, dy, &u_grad, &v_grad, &mut div_grad);
        let config = PoissonConfig {
            nx,
            ny,
            dx,
            dy,
            mode: Mode::OptimizedFixedIters,
            solver: PoissonSolverKind::Rbgs,
            max_iters: 1,
            tolerance: 0.0,
            allow_early_stop: false,
            mg_vcycles: 0,
            mg_pre_smooth: 0,
            mg_post_smooth: 0,
            mg_coarse_iters: 0,
            max_vcycles: 0,
        };
        apply_mac_laplacian(&config, &p, &mut lap);
        for k in 0..lap.len() {
            assert!(
                (lap[k] - div_grad[k]).abs() < 1.0e-10,
                "D(G(p)) != laplacian at {k}: {} vs {}",
                div_grad[k],
                lap[k]
            );
        }
    }

    #[test]
    fn multigrid_residual_drops_strongly_over_vcycles() {
        let nx = 32;
        let ny = 32;
        let dx = 1.0 / nx as f64;
        let dy = 1.0 / ny as f64;
        let mut rhs = vec![0.0; nx * ny];
        let mut p = vec![0.0; nx * ny];
        let mut seed = 0xfeed_beef_u64;
        for value in &mut rhs {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let unit = ((seed >> 32) as f64) / (u32::MAX as f64);
            *value = 2.0 * unit - 1.0;
        }
        let mean = rhs.iter().sum::<f64>() / rhs.len() as f64;
        for value in &mut rhs {
            *value -= mean;
        }

        let mut multigrid = MultigridSolver::new(nx, ny, dx, dy);
        let config = crate::multigrid::MultigridConfig {
            pre_smooth: 2,
            post_smooth: 2,
            coarse_iters: 100,
        };
        let initial_check = multigrid.residual_l2_for(&rhs, &p);
        let (_, initial, after_one) = multigrid.solve_fixed(&rhs, &mut p, &config, 1);
        let (_, _, after_several) = multigrid.solve_fixed(&rhs, &mut p, &config, 5);
        assert!((initial - initial_check).abs() < 1.0e-12);
        assert!(after_one < initial, "one V-cycle did not reduce residual");
        assert!(
            after_several < initial * 1.0e-2,
            "several V-cycles did not reduce residual strongly: initial={initial}, final={after_several}"
        );
    }
}
