use std::sync::OnceLock;

pub fn sum_f64(values: &[f64]) -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        if force_scalar_kernels() {
            return values.iter().sum();
        }
        // SAFETY: The NEON implementation only performs unaligned in-bounds loads
        // from chunks proven by the loop bounds, then finishes the tail scalarly.
        unsafe { neon::sum_f64(values) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        values.iter().sum()
    }
}

pub fn sum_squares_f64(values: &[f64]) -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        if force_scalar_kernels() {
            return values.iter().map(|x| x * x).sum();
        }
        // SAFETY: The NEON implementation only performs unaligned in-bounds loads
        // from chunks proven by the loop bounds, then finishes the tail scalarly.
        unsafe { neon::sum_squares_f64(values) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        values.iter().map(|x| x * x).sum()
    }
}

pub fn subtract_scalar_in_place(values: &mut [f64], scalar: f64) {
    #[cfg(target_arch = "aarch64")]
    {
        if force_scalar_kernels() {
            for value in values {
                *value -= scalar;
            }
            return;
        }
        // SAFETY: The NEON implementation only performs unaligned in-bounds
        // loads/stores to disjoint mutable chunks, then finishes the tail scalarly.
        unsafe { neon::subtract_scalar_in_place(values, scalar) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for value in values {
            *value -= scalar;
        }
    }
}

pub fn pressure_residual(
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    rhs: &[f64],
    p: &[f64],
    residual: &mut [f64],
) {
    debug_assert_eq!(rhs.len(), nx * ny);
    debug_assert_eq!(p.len(), nx * ny);
    debug_assert_eq!(residual.len(), nx * ny);

    #[cfg(target_arch = "aarch64")]
    {
        if force_scalar_kernels() {
            scalar::pressure_residual(nx, ny, dx, dy, rhs, p, residual);
            return;
        }
        // SAFETY: The NEON implementation handles boundaries scalarly and only
        // vectorizes interior cells with in-bounds contiguous loads.
        unsafe { neon::pressure_residual(nx, ny, dx, dy, rhs, p, residual) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar::pressure_residual(nx, ny, dx, dy, rhs, p, residual);
    }
}

fn force_scalar_kernels() -> bool {
    static FORCE_SCALAR: OnceLock<bool> = OnceLock::new();
    *FORCE_SCALAR.get_or_init(|| std::env::var_os("NS_FORCE_SCALAR").is_some())
}

mod scalar {
    pub fn pressure_residual(
        nx: usize,
        ny: usize,
        dx: f64,
        dy: f64,
        rhs: &[f64],
        p: &[f64],
        residual: &mut [f64],
    ) {
        let inv_dx2 = 1.0 / (dx * dx);
        let inv_dy2 = 1.0 / (dy * dy);
        for j in 0..ny {
            for i in 0..nx {
                pressure_residual_cell(nx, ny, inv_dx2, inv_dy2, rhs, p, residual, i, j);
            }
        }
    }

    pub fn pressure_residual_cell(
        nx: usize,
        ny: usize,
        inv_dx2: f64,
        inv_dy2: f64,
        rhs: &[f64],
        p: &[f64],
        residual: &mut [f64],
        i: usize,
        j: usize,
    ) {
        let c = i + nx * j;
        let mut value = 0.0;
        if i > 0 {
            value += (p[c - 1] - p[c]) * inv_dx2;
        }
        if i + 1 < nx {
            value += (p[c + 1] - p[c]) * inv_dx2;
        }
        if j > 0 {
            value += (p[c - nx] - p[c]) * inv_dy2;
        }
        if j + 1 < ny {
            value += (p[c + nx] - p[c]) * inv_dy2;
        }
        residual[c] = rhs[c] - value;
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use crate::kernels::scalar;
    use std::arch::aarch64::*;

    pub unsafe fn sum_f64(values: &[f64]) -> f64 {
        let mut acc0 = vdupq_n_f64(0.0);
        let mut acc1 = vdupq_n_f64(0.0);
        let mut acc2 = vdupq_n_f64(0.0);
        let mut acc3 = vdupq_n_f64(0.0);
        let chunks = values.len() / 8;
        for k in 0..chunks {
            let ptr = values.as_ptr().add(8 * k);
            acc0 = vaddq_f64(acc0, vld1q_f64(ptr));
            acc1 = vaddq_f64(acc1, vld1q_f64(ptr.add(2)));
            acc2 = vaddq_f64(acc2, vld1q_f64(ptr.add(4)));
            acc3 = vaddq_f64(acc3, vld1q_f64(ptr.add(6)));
        }
        let acc = vaddq_f64(vaddq_f64(acc0, acc1), vaddq_f64(acc2, acc3));
        let lanes = [vgetq_lane_f64::<0>(acc), vgetq_lane_f64::<1>(acc)];
        let mut sum = lanes[0] + lanes[1];
        for value in &values[8 * chunks..] {
            sum += *value;
        }
        sum
    }

    pub unsafe fn sum_squares_f64(values: &[f64]) -> f64 {
        let mut acc0 = vdupq_n_f64(0.0);
        let mut acc1 = vdupq_n_f64(0.0);
        let mut acc2 = vdupq_n_f64(0.0);
        let mut acc3 = vdupq_n_f64(0.0);
        let chunks = values.len() / 8;
        for k in 0..chunks {
            let ptr = values.as_ptr().add(8 * k);
            let x0 = vld1q_f64(ptr);
            let x1 = vld1q_f64(ptr.add(2));
            let x2 = vld1q_f64(ptr.add(4));
            let x3 = vld1q_f64(ptr.add(6));
            acc0 = vaddq_f64(acc0, vmulq_f64(x0, x0));
            acc1 = vaddq_f64(acc1, vmulq_f64(x1, x1));
            acc2 = vaddq_f64(acc2, vmulq_f64(x2, x2));
            acc3 = vaddq_f64(acc3, vmulq_f64(x3, x3));
        }
        let acc = vaddq_f64(vaddq_f64(acc0, acc1), vaddq_f64(acc2, acc3));
        let lanes = [vgetq_lane_f64::<0>(acc), vgetq_lane_f64::<1>(acc)];
        let mut sum = lanes[0] + lanes[1];
        for value in &values[8 * chunks..] {
            sum += *value * *value;
        }
        sum
    }

    pub unsafe fn subtract_scalar_in_place(values: &mut [f64], scalar: f64) {
        let shift = vdupq_n_f64(scalar);
        let chunks = values.len() / 8;
        for k in 0..chunks {
            let ptr = values.as_mut_ptr().add(8 * k);
            let x0 = vld1q_f64(ptr);
            let x1 = vld1q_f64(ptr.add(2));
            let x2 = vld1q_f64(ptr.add(4));
            let x3 = vld1q_f64(ptr.add(6));
            vst1q_f64(ptr, vsubq_f64(x0, shift));
            vst1q_f64(ptr.add(2), vsubq_f64(x1, shift));
            vst1q_f64(ptr.add(4), vsubq_f64(x2, shift));
            vst1q_f64(ptr.add(6), vsubq_f64(x3, shift));
        }
        for value in &mut values[8 * chunks..] {
            *value -= scalar;
        }
    }

    pub unsafe fn pressure_residual(
        nx: usize,
        ny: usize,
        dx: f64,
        dy: f64,
        rhs: &[f64],
        p: &[f64],
        residual: &mut [f64],
    ) {
        let inv_dx2_scalar = 1.0 / (dx * dx);
        let inv_dy2_scalar = 1.0 / (dy * dy);
        for i in 0..nx {
            scalar::pressure_residual_cell(
                nx,
                ny,
                inv_dx2_scalar,
                inv_dy2_scalar,
                rhs,
                p,
                residual,
                i,
                0,
            );
            if ny > 1 {
                scalar::pressure_residual_cell(
                    nx,
                    ny,
                    inv_dx2_scalar,
                    inv_dy2_scalar,
                    rhs,
                    p,
                    residual,
                    i,
                    ny - 1,
                );
            }
        }
        for j in 1..ny.saturating_sub(1) {
            scalar::pressure_residual_cell(
                nx,
                ny,
                inv_dx2_scalar,
                inv_dy2_scalar,
                rhs,
                p,
                residual,
                0,
                j,
            );
            if nx > 1 {
                scalar::pressure_residual_cell(
                    nx,
                    ny,
                    inv_dx2_scalar,
                    inv_dy2_scalar,
                    rhs,
                    p,
                    residual,
                    nx - 1,
                    j,
                );
            }
        }
        if nx < 4 || ny < 3 {
            return;
        }

        let inv_dx2 = vdupq_n_f64(1.0 / (dx * dx));
        let inv_dy2 = vdupq_n_f64(1.0 / (dy * dy));
        for j in 1..ny - 1 {
            let row = j * nx;
            let mut i = 1usize;
            while i + 2 < nx {
                let c = row + i;
                let center = vld1q_f64(p.as_ptr().add(c));
                let west = vld1q_f64(p.as_ptr().add(c - 1));
                let east = vld1q_f64(p.as_ptr().add(c + 1));
                let south = vld1q_f64(p.as_ptr().add(c - nx));
                let north = vld1q_f64(p.as_ptr().add(c + nx));
                let rhs_v = vld1q_f64(rhs.as_ptr().add(c));

                let x_part = vmulq_f64(
                    vaddq_f64(vsubq_f64(west, center), vsubq_f64(east, center)),
                    inv_dx2,
                );
                let y_part = vmulq_f64(
                    vaddq_f64(vsubq_f64(south, center), vsubq_f64(north, center)),
                    inv_dy2,
                );
                let lap = vaddq_f64(x_part, y_part);
                vst1q_f64(residual.as_mut_ptr().add(c), vsubq_f64(rhs_v, lap));
                i += 2;
            }
            if i < nx - 1 {
                scalar::pressure_residual_cell(
                    nx,
                    ny,
                    inv_dx2_scalar,
                    inv_dy2_scalar,
                    rhs,
                    p,
                    residual,
                    i,
                    j,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded_values(len: usize) -> Vec<f64> {
        let mut seed = 0x6a09_e667_f3bc_c909_u64;
        (0..len)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let unit = ((seed >> 32) as f64) / (u32::MAX as f64);
                2.0 * unit - 1.0
            })
            .collect()
    }

    #[test]
    fn sums_match_scalar_reference() {
        let values = seeded_values(257);
        let scalar_sum: f64 = values.iter().sum();
        let scalar_sq: f64 = values.iter().map(|x| x * x).sum();
        assert!((sum_f64(&values) - scalar_sum).abs() < 1.0e-12);
        assert!((sum_squares_f64(&values) - scalar_sq).abs() < 1.0e-12);
    }

    #[test]
    fn subtract_scalar_matches_reference() {
        let mut values = seeded_values(257);
        let mut expected = values.clone();
        subtract_scalar_in_place(&mut values, 0.125);
        for value in &mut expected {
            *value -= 0.125;
        }
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((*actual - *expected).abs() < 1.0e-15);
        }
    }

    #[test]
    fn pressure_residual_matches_scalar_reference() {
        let nx = 17;
        let ny = 13;
        let dx = 1.0 / nx as f64;
        let dy = 1.0 / ny as f64;
        let rhs = seeded_values(nx * ny);
        let p = seeded_values(nx * ny);
        let mut actual = vec![0.0; nx * ny];
        let mut expected = vec![0.0; nx * ny];
        pressure_residual(nx, ny, dx, dy, &rhs, &p, &mut actual);
        scalar::pressure_residual(nx, ny, dx, dy, &rhs, &p, &mut expected);
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((*actual - *expected).abs() < 1.0e-9);
        }
    }
}
