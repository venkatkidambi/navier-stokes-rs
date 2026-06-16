use crate::solver::{idx_u, idx_v};
use rayon::prelude::*;

#[derive(Copy, Clone, Debug)]
pub struct DivergenceStats {
    pub raw_l2: f64,
    pub rms: f64,
    pub max_abs: f64,
}

pub fn compute_mac_divergence(
    nx: usize,
    _ny: usize,
    dx: f64,
    dy: f64,
    u: &[f64],
    v: &[f64],
    div: &mut [f64],
) {
    let inv_dx = 1.0 / dx;
    let inv_dy = 1.0 / dy;
    div.par_chunks_mut(nx).enumerate().for_each(|(j, row)| {
        for i in 0..nx {
            row[i] = (u[idx_u(i + 1, j, nx)] - u[idx_u(i, j, nx)]) * inv_dx
                + (v[idx_v(i, j + 1, nx)] - v[idx_v(i, j, nx)]) * inv_dy;
        }
    });
}

pub fn divergence_l2_raw(div: &[f64]) -> f64 {
    let sum: f64 = div.par_iter().map(|x| x * x).sum();
    sum.sqrt()
}

pub fn divergence_rms(div: &[f64]) -> f64 {
    divergence_l2_raw(div) / (div.len() as f64).sqrt()
}

pub fn max_abs_divergence(div: &[f64]) -> f64 {
    div.par_iter()
        .map(|x| x.abs())
        .reduce(|| 0.0, |a, b| a.max(b))
}

pub fn divergence_stats(div: &[f64]) -> DivergenceStats {
    DivergenceStats {
        raw_l2: divergence_l2_raw(div),
        rms: divergence_rms(div),
        max_abs: max_abs_divergence(div),
    }
}

pub fn kinetic_energy(nx: usize, ny: usize, dx: f64, dy: f64, u: &[f64], v: &[f64]) -> f64 {
    let sum: f64 = (0..ny)
        .into_par_iter()
        .map(|j| {
            let mut local = 0.0;
            for i in 0..nx {
                let uc = 0.5 * (u[idx_u(i, j, nx)] + u[idx_u(i + 1, j, nx)]);
                let vc = 0.5 * (v[idx_v(i, j, nx)] + v[idx_v(i, j + 1, nx)]);
                local += 0.5 * (uc * uc + vc * vc);
            }
            local
        })
        .sum();
    sum * dx * dy
}
