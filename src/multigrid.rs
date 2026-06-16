use crate::kernels;
use crate::solver::idx_p;
use std::time::{Duration, Instant};

pub struct MultigridConfig {
    pub pre_smooth: usize,
    pub post_smooth: usize,
    pub coarse_iters: usize,
}

pub struct MultigridSolver {
    levels: Vec<Level>,
    profile: KernelProfile,
}

#[derive(Default)]
struct KernelProfile {
    enabled: bool,
    smooth: Duration,
    residual_operator: Duration,
    residual_norm: Duration,
    restriction: Duration,
    prolongation: Duration,
    subtract_mean: Duration,
    load_finest: Duration,
    copies: Duration,
}

struct Level {
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    x: Vec<f64>,
    rhs: Vec<f64>,
    residual: Vec<f64>,
}

impl MultigridSolver {
    pub fn new(nx: usize, ny: usize, dx: f64, dy: f64) -> Self {
        assert!(nx == ny, "multigrid currently expects square grids");
        assert!(nx >= 4 && ny >= 4, "multigrid grid must be at least 4x4");
        let mut levels = Vec::new();
        let mut nxl = nx;
        let mut nyl = ny;
        let mut dxl = dx;
        let mut dyl = dy;
        loop {
            levels.push(Level::new(nxl, nyl, dxl, dyl));
            if nxl <= 4 || nyl <= 4 {
                break;
            }
            assert!(
                nxl % 2 == 0 && nyl % 2 == 0,
                "multigrid requires grid dimensions divisible by powers of 2 down to 4x4"
            );
            nxl /= 2;
            nyl /= 2;
            dxl *= 2.0;
            dyl *= 2.0;
        }
        Self {
            levels,
            profile: KernelProfile {
                enabled: std::env::var_os("NS_PROFILE").is_some(),
                ..KernelProfile::default()
            },
        }
    }

    pub fn solve_fixed(
        &mut self,
        rhs: &[f64],
        p: &mut [f64],
        cfg: &MultigridConfig,
        vcycles: usize,
    ) -> (usize, f64, f64) {
        self.load_finest(rhs, p);
        let initial = self.residual_l2_finest();
        for _ in 0..vcycles {
            self.v_cycle(0, cfg);
            let t = Instant::now();
            subtract_mean(&mut self.levels[0].x);
            self.profile.add_subtract_mean(t.elapsed());
        }
        let final_residual = self.residual_l2_finest();
        let t = Instant::now();
        p.copy_from_slice(&self.levels[0].x);
        self.profile.add_copies(t.elapsed());
        (vcycles, initial, final_residual)
    }

    pub fn solve_tolerance(
        &mut self,
        rhs: &[f64],
        p: &mut [f64],
        cfg: &MultigridConfig,
        tol: f64,
        max_vcycles: usize,
    ) -> (usize, f64, f64) {
        self.load_finest(rhs, p);
        let initial = self.residual_l2_finest();
        let mut final_residual = initial;
        let mut cycles = 0usize;
        for _ in 0..max_vcycles {
            self.v_cycle(0, cfg);
            let t = Instant::now();
            subtract_mean(&mut self.levels[0].x);
            self.profile.add_subtract_mean(t.elapsed());
            cycles += 1;
            final_residual = self.residual_l2_finest();
            if final_residual < tol {
                break;
            }
        }
        let t = Instant::now();
        p.copy_from_slice(&self.levels[0].x);
        self.profile.add_copies(t.elapsed());
        (cycles, initial, final_residual)
    }

    #[allow(dead_code)]
    pub fn residual_l2_for(&mut self, rhs: &[f64], p: &[f64]) -> f64 {
        self.levels[0].x.copy_from_slice(p);
        self.levels[0].rhs.copy_from_slice(rhs);
        self.residual_l2_finest()
    }

    fn load_finest(&mut self, rhs: &[f64], p: &[f64]) {
        let t = Instant::now();
        self.levels[0].x.copy_from_slice(p);
        self.levels[0].rhs.copy_from_slice(rhs);
        self.profile.add_load_finest(t.elapsed());
        let t = Instant::now();
        subtract_mean(&mut self.levels[0].x);
        subtract_mean(&mut self.levels[0].rhs);
        self.profile.add_subtract_mean(t.elapsed());
    }

    fn residual_l2_finest(&mut self) -> f64 {
        let t = Instant::now();
        compute_residual_level(&mut self.levels[0]);
        self.profile.add_residual_operator(t.elapsed());
        let t = Instant::now();
        let norm = l2_rms(&self.levels[0].residual);
        self.profile.add_residual_norm(t.elapsed());
        norm
    }

    fn v_cycle(&mut self, level_idx: usize, cfg: &MultigridConfig) {
        let is_coarse = level_idx + 1 == self.levels.len();
        if is_coarse {
            let t = Instant::now();
            rbgs_smooth_level(&mut self.levels[level_idx], cfg.coarse_iters);
            self.profile.add_smooth(t.elapsed());
            let t = Instant::now();
            subtract_mean(&mut self.levels[level_idx].x);
            self.profile.add_subtract_mean(t.elapsed());
            return;
        }

        let t = Instant::now();
        rbgs_smooth_level(&mut self.levels[level_idx], cfg.pre_smooth);
        self.profile.add_smooth(t.elapsed());
        let t = Instant::now();
        subtract_mean(&mut self.levels[level_idx].x);
        self.profile.add_subtract_mean(t.elapsed());
        let t = Instant::now();
        compute_residual_level(&mut self.levels[level_idx]);
        self.profile.add_residual_operator(t.elapsed());

        let (fine, rest) = self.levels.split_at_mut(level_idx + 1);
        let fine_level = &fine[level_idx];
        let coarse = &mut rest[0];
        let t = Instant::now();
        restrict_full_weighting(
            fine_level.nx,
            fine_level.ny,
            &fine_level.residual,
            &mut coarse.rhs,
        );
        self.profile.add_restriction(t.elapsed());
        let t = Instant::now();
        subtract_mean(&mut coarse.rhs);
        self.profile.add_subtract_mean(t.elapsed());
        let t = Instant::now();
        coarse.x.fill(0.0);
        self.profile.add_copies(t.elapsed());

        self.v_cycle(level_idx + 1, cfg);

        let (fine, rest) = self.levels.split_at_mut(level_idx + 1);
        let fine_level = &mut fine[level_idx];
        let coarse = &rest[0];
        let t = Instant::now();
        prolong_bilinear_add(fine_level.nx, fine_level.ny, &coarse.x, &mut fine_level.x);
        self.profile.add_prolongation(t.elapsed());
        let t = Instant::now();
        subtract_mean(&mut fine_level.x);
        self.profile.add_subtract_mean(t.elapsed());
        let t = Instant::now();
        rbgs_smooth_level(fine_level, cfg.post_smooth);
        self.profile.add_smooth(t.elapsed());
        let t = Instant::now();
        subtract_mean(&mut fine_level.x);
        self.profile.add_subtract_mean(t.elapsed());
    }
}

impl KernelProfile {
    fn add_smooth(&mut self, elapsed: Duration) {
        if self.enabled {
            self.smooth += elapsed;
        }
    }

    fn add_residual_operator(&mut self, elapsed: Duration) {
        if self.enabled {
            self.residual_operator += elapsed;
        }
    }

    fn add_residual_norm(&mut self, elapsed: Duration) {
        if self.enabled {
            self.residual_norm += elapsed;
        }
    }

    fn add_restriction(&mut self, elapsed: Duration) {
        if self.enabled {
            self.restriction += elapsed;
        }
    }

    fn add_prolongation(&mut self, elapsed: Duration) {
        if self.enabled {
            self.prolongation += elapsed;
        }
    }

    fn add_subtract_mean(&mut self, elapsed: Duration) {
        if self.enabled {
            self.subtract_mean += elapsed;
        }
    }

    fn add_load_finest(&mut self, elapsed: Duration) {
        if self.enabled {
            self.load_finest += elapsed;
        }
    }

    fn add_copies(&mut self, elapsed: Duration) {
        if self.enabled {
            self.copies += elapsed;
        }
    }

    fn total(&self) -> Duration {
        self.smooth
            + self.residual_operator
            + self.residual_norm
            + self.restriction
            + self.prolongation
            + self.subtract_mean
            + self.load_finest
            + self.copies
    }
}

impl Drop for MultigridSolver {
    fn drop(&mut self) {
        if !self.profile.enabled {
            return;
        }
        let total = self.profile.total().as_secs_f64();
        if total == 0.0 {
            return;
        }
        let print_row = |name: &str, duration: Duration| {
            let seconds = duration.as_secs_f64();
            eprintln!(
                "  {name:<18} {:>10.3} ms {:>6.1}%",
                1000.0 * seconds,
                100.0 * seconds / total
            );
        };
        eprintln!("Multigrid kernel profile:");
        print_row("rbgs_smooth", self.profile.smooth);
        print_row("residual_operator", self.profile.residual_operator);
        print_row("residual_norm", self.profile.residual_norm);
        print_row("restriction", self.profile.restriction);
        print_row("prolongation", self.profile.prolongation);
        print_row("subtract_mean", self.profile.subtract_mean);
        print_row("load_finest", self.profile.load_finest);
        print_row("copies/fill", self.profile.copies);
    }
}

impl Level {
    fn new(nx: usize, ny: usize, dx: f64, dy: f64) -> Self {
        let n = nx * ny;
        Self {
            nx,
            ny,
            dx,
            dy,
            x: vec![0.0; n],
            rhs: vec![0.0; n],
            residual: vec![0.0; n],
        }
    }
}

pub fn apply_operator(nx: usize, ny: usize, dx: f64, dy: f64, p: &[f64], out: &mut [f64]) {
    let inv_dx2 = 1.0 / (dx * dx);
    let inv_dy2 = 1.0 / (dy * dy);
    for j in 0..ny {
        for i in 0..nx {
            let c = idx_p(i, j, nx);
            let mut value = 0.0;
            if i > 0 {
                value += (p[idx_p(i - 1, j, nx)] - p[c]) * inv_dx2;
            }
            if i + 1 < nx {
                value += (p[idx_p(i + 1, j, nx)] - p[c]) * inv_dx2;
            }
            if j > 0 {
                value += (p[idx_p(i, j - 1, nx)] - p[c]) * inv_dy2;
            }
            if j + 1 < ny {
                value += (p[idx_p(i, j + 1, nx)] - p[c]) * inv_dy2;
            }
            out[c] = value;
        }
    }
}

pub fn residual_l2(
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    rhs: &[f64],
    p: &[f64],
    tmp: &mut [f64],
) -> f64 {
    apply_operator(nx, ny, dx, dy, p, tmp);
    let mut sum = 0.0;
    for k in 0..rhs.len() {
        let r = rhs[k] - tmp[k];
        sum += r * r;
    }
    (sum / rhs.len() as f64).sqrt()
}

pub fn rbgs_smooth(
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    rhs: &[f64],
    p: &mut [f64],
    sweeps: usize,
) {
    let inv_dx2 = 1.0 / (dx * dx);
    let inv_dy2 = 1.0 / (dy * dy);
    for _ in 0..sweeps {
        for color in 0..2 {
            rbgs_smooth_color(nx, ny, inv_dx2, inv_dy2, rhs, p, color);
        }
        subtract_mean(p);
    }
}

fn rbgs_smooth_color(
    nx: usize,
    ny: usize,
    inv_dx2: f64,
    inv_dy2: f64,
    rhs: &[f64],
    p: &mut [f64],
    color: usize,
) {
    if nx == 0 || ny == 0 {
        return;
    }

    let interior_coeff_inv = 1.0 / (2.0 * (inv_dx2 + inv_dy2));

    update_boundary_row(nx, ny, 0, color, inv_dx2, inv_dy2, rhs, p);
    if ny > 1 {
        update_boundary_row(nx, ny, ny - 1, color, inv_dx2, inv_dy2, rhs, p);
    }

    for j in 1..ny.saturating_sub(1) {
        let row = j * nx;
        let parity = color ^ (j & 1);

        if parity == 0 {
            update_left_boundary(row, nx, inv_dx2, inv_dy2, rhs, p);
        }
        if ((nx - 1) & 1) == parity {
            update_right_boundary(row, nx, inv_dx2, inv_dy2, rhs, p);
        }

        let start = if parity == 1 { 1 } else { 2 };
        // SAFETY: `j` is restricted to interior rows and `start..nx-1`
        // restricts `i` to interior columns, so c +/- 1 and c +/- nx are
        // in-bounds. Red/black ordering updates only one color at a time; all
        // stencil neighbors are the opposite color and are not written by this
        // loop, preserving the scalar RBGS update order.
        unsafe {
            rbgs_smooth_interior_row_unchecked(
                row,
                start,
                nx,
                inv_dx2,
                inv_dy2,
                interior_coeff_inv,
                rhs,
                p,
            );
        }
    }
}

unsafe fn rbgs_smooth_interior_row_unchecked(
    row: usize,
    start: usize,
    nx: usize,
    inv_dx2: f64,
    inv_dy2: f64,
    coeff_inv: f64,
    rhs: &[f64],
    p: &mut [f64],
) {
    let p_ptr = p.as_mut_ptr();
    let rhs_ptr = rhs.as_ptr();
    let mut i = start;
    while i < nx - 1 {
        let c = row + i;
        let neigh = (*p_ptr.add(c - 1) + *p_ptr.add(c + 1)) * inv_dx2
            + (*p_ptr.add(c - nx) + *p_ptr.add(c + nx)) * inv_dy2;
        *p_ptr.add(c) = (neigh - *rhs_ptr.add(c)) * coeff_inv;
        i += 2;
    }
}

fn update_boundary_row(
    nx: usize,
    ny: usize,
    j: usize,
    color: usize,
    inv_dx2: f64,
    inv_dy2: f64,
    rhs: &[f64],
    p: &mut [f64],
) {
    let row = j * nx;
    let parity = color ^ (j & 1);
    for i in (parity..nx).step_by(2) {
        let c = row + i;
        let mut coeff = 0.0;
        let mut neigh = 0.0;
        if i > 0 {
            coeff += inv_dx2;
            neigh += p[c - 1] * inv_dx2;
        }
        if i + 1 < nx {
            coeff += inv_dx2;
            neigh += p[c + 1] * inv_dx2;
        }
        if j > 0 {
            coeff += inv_dy2;
            neigh += p[c - nx] * inv_dy2;
        }
        if j + 1 < ny {
            coeff += inv_dy2;
            neigh += p[c + nx] * inv_dy2;
        }
        p[c] = (neigh - rhs[c]) / coeff;
    }
}

fn update_left_boundary(
    row: usize,
    nx: usize,
    inv_dx2: f64,
    inv_dy2: f64,
    rhs: &[f64],
    p: &mut [f64],
) {
    let c = row;
    let coeff_inv = 1.0 / (inv_dx2 + 2.0 * inv_dy2);
    let neigh = p[c + 1] * inv_dx2 + (p[c - nx] + p[c + nx]) * inv_dy2;
    p[c] = (neigh - rhs[c]) * coeff_inv;
}

fn update_right_boundary(
    row: usize,
    nx: usize,
    inv_dx2: f64,
    inv_dy2: f64,
    rhs: &[f64],
    p: &mut [f64],
) {
    let c = row + nx - 1;
    let coeff_inv = 1.0 / (inv_dx2 + 2.0 * inv_dy2);
    let neigh = p[c - 1] * inv_dx2 + (p[c - nx] + p[c + nx]) * inv_dy2;
    p[c] = (neigh - rhs[c]) * coeff_inv;
}

fn rbgs_smooth_level(level: &mut Level, sweeps: usize) {
    rbgs_smooth(
        level.nx,
        level.ny,
        level.dx,
        level.dy,
        &level.rhs,
        &mut level.x,
        sweeps,
    );
}

fn compute_residual_level(level: &mut Level) {
    kernels::pressure_residual(
        level.nx,
        level.ny,
        level.dx,
        level.dy,
        &level.rhs,
        &level.x,
        &mut level.residual,
    );
}

fn restrict_full_weighting(fnx: usize, fny: usize, fine: &[f64], coarse: &mut [f64]) {
    let cnx = fnx / 2;
    let cny = fny / 2;
    for jc in 0..cny {
        for ic in 0..cnx {
            if ic > 0 && ic + 1 < cnx && jc > 0 && jc + 1 < cny {
                let fi = 2 * ic;
                let fj = 2 * jc;
                let fm = idx_p(fi, fj - 1, fnx);
                let fc = idx_p(fi, fj, fnx);
                let fp = idx_p(fi, fj + 1, fnx);
                coarse[idx_p(ic, jc, cnx)] = (4.0 * fine[fc]
                    + 2.0 * (fine[fc - 1] + fine[fc + 1] + fine[fm] + fine[fp])
                    + fine[fm - 1]
                    + fine[fm + 1]
                    + fine[fp - 1]
                    + fine[fp + 1])
                    * (1.0 / 16.0);
                continue;
            }
            let fi = 2 * ic;
            let fj = 2 * jc;
            let mut sum = 0.0;
            let mut weight = 0.0;
            for dj in -1isize..=1 {
                for di in -1isize..=1 {
                    let i = fi as isize + di;
                    let j = fj as isize + dj;
                    if i < 0 || j < 0 || i >= fnx as isize || j >= fny as isize {
                        continue;
                    }
                    let w = match (di.abs(), dj.abs()) {
                        (0, 0) => 4.0,
                        (1, 0) | (0, 1) => 2.0,
                        _ => 1.0,
                    };
                    sum += w * fine[idx_p(i as usize, j as usize, fnx)];
                    weight += w;
                }
            }
            coarse[idx_p(ic, jc, cnx)] = sum / weight;
        }
    }
}

fn prolong_bilinear_add(fnx: usize, fny: usize, coarse: &[f64], fine: &mut [f64]) {
    let cnx = fnx / 2;
    let cny = fny / 2;
    for j in 0..fny {
        let (j0, j1, wy0, wy1) = prolong_indices(j, cny);
        let fine_row = j * fnx;
        let coarse_row0 = j0 * cnx;
        let coarse_row1 = j1 * cnx;
        for i in 0..fnx {
            let (i0, i1, wx0, wx1) = prolong_indices(i, cnx);
            let coarse0 = wx0 * coarse[coarse_row0 + i0] + wx1 * coarse[coarse_row0 + i1];
            let coarse1 = wx0 * coarse[coarse_row1 + i0] + wx1 * coarse[coarse_row1 + i1];
            fine[fine_row + i] += wy0 * coarse0 + wy1 * coarse1;
        }
    }
}

#[inline]
fn prolong_indices(fine_index: usize, coarse_len: usize) -> (usize, usize, f64, f64) {
    if fine_index == 0 {
        return (0, 0, 1.0, 0.0);
    }
    let coarse = fine_index / 2;
    if fine_index & 1 == 0 {
        (coarse - 1, coarse, 0.25, 0.75)
    } else if coarse + 1 < coarse_len {
        (coarse, coarse + 1, 0.75, 0.25)
    } else {
        (coarse, coarse, 1.0, 0.0)
    }
}

fn l2_rms(values: &[f64]) -> f64 {
    let sum = kernels::sum_squares_f64(values);
    (sum / values.len() as f64).sqrt()
}

fn subtract_mean(values: &mut [f64]) {
    let mean = kernels::sum_f64(values) / values.len() as f64;
    kernels::subtract_scalar_in_place(values, mean);
}
