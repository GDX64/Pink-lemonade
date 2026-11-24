//! How much of a Haar basis function's L2 energy survives the render blur.
//!
//! All lengths are in kernel standard deviations (σ = 1). Rendering convolves the density with a
//! unit Gaussian `G`, so the rendered energy of an L2-normalized basis function `B` is
//! `‖B * G‖² = ⟨B, B * G₂⟩` with `G₂ = G * G` of variance 2. For indicator functions of intervals
//! this has a closed form through the second antiderivative of `G₂`.

use std::f64::consts::PI;

/// Variance of `G * G` for a unit-variance render kernel.
const VAR2: f64 = 2.0;

/// Error function (Abramowitz & Stegun 7.1.26, |error| ≤ 1.5e-7), odd by construction.
pub fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let y = 1.0 - poly * (-x * x).exp();
    if x >= 0.0 { y } else { -y }
}

fn g2(z: f64) -> f64 {
    (-z * z / (2.0 * VAR2)).exp() / (2.0 * PI * VAR2).sqrt()
}

/// CDF of `G₂` at `z`.
fn cdf2(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / (2.0 * VAR2).sqrt()))
}

/// Second antiderivative of `G₂`: `K'' = G₂`.
fn k2(z: f64) -> f64 {
    z * cdf2(z) + VAR2 * g2(z)
}

/// `∫_a^b ∫_c^d G₂(s - t) ds dt`.
fn rect_overlap(a: f64, b: f64, c: f64, d: f64) -> f64 {
    k2(d - a) - k2(d - b) - k2(c - a) + k2(c - b)
}

/// Surviving energy fraction of an L2-normalized box of width `p` (Haar scaling function).
pub fn box_gain(p: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p < 1e-3 {
        return p * g2(0.0);
    }
    (rect_overlap(0.0, p, 0.0, p) / p).clamp(0.0, 1.0)
}

/// Surviving energy fraction of an L2-normalized Haar wavelet with support `p` (halves `p / 2`).
pub fn wavelet_gain(p: f64) -> f64 {
    if p <= 1e-2 {
        // ~p³ and below every threshold that matters; avoids cancellation.
        return 0.0;
    }
    let h = 0.5 * p;
    let same = rect_overlap(0.0, h, 0.0, h);
    let cross = rect_overlap(0.0, h, h, 2.0 * h);
    ((2.0 * same - 2.0 * cross) / (2.0 * h)).clamp(0.0, 1.0)
}

/// `‖ψ * G‖₁ / ‖ψ‖₁` for a Haar wavelet with support `p`: the fraction of a dropped detail's
/// moved mass that still shows after blurring. The integrand is antisymmetric about the
/// wavelet's centre and one-signed on each side, which gives the closed form below.
pub fn wavelet_l1_gain(p: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    let h = 0.5 * p;
    if h < 1e-3 {
        return p / (2.0 * (2.0 * PI).sqrt());
    }
    let phi = |z: f64| (-0.5 * z * z).exp() / (2.0 * PI).sqrt();
    let v = h * erf(h / std::f64::consts::SQRT_2) + 2.0 * phi(h) - 2.0 * phi(0.0);
    (2.0 / p * v).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force `‖B * G‖²` on a fine grid for comparison.
    fn numeric_gain(p: f64, wavelet: bool) -> f64 {
        let n = 4000;
        let lo = -8.0;
        let hi = p + 8.0;
        let dz = (hi - lo) / n as f64;
        let norm = 1.0 / p.sqrt();
        let basis = |s: f64| {
            if !(0.0..p).contains(&s) {
                0.0
            } else if wavelet && s >= 0.5 * p {
                -norm
            } else {
                norm
            }
        };
        // blurred(z) = ∫ B(s) G(z - s) ds
        let sub = 2000;
        let ds = p / sub as f64;
        let mut energy = 0.0;
        for i in 0..n {
            let z = lo + (i as f64 + 0.5) * dz;
            let mut v = 0.0;
            for j in 0..sub {
                let s = (j as f64 + 0.5) * ds;
                let u = z - s;
                v += basis(s) * (-0.5 * u * u).exp() / (2.0 * PI).sqrt() * ds;
            }
            energy += v * v * dz;
        }
        energy
    }

    #[test]
    fn closed_forms_match_numeric_integration() {
        for p in [0.3, 1.0, 2.5, 8.0] {
            let b = box_gain(p);
            let w = wavelet_gain(p);
            assert!((b - numeric_gain(p, false)).abs() < 2e-4, "box p={p}: {b}");
            assert!(
                (w - numeric_gain(p, true)).abs() < 2e-4,
                "wavelet p={p}: {w}"
            );
        }
    }

    #[test]
    fn l1_gain_matches_numeric_integration() {
        let cdf = |z: f64| 0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2));
        for p in [0.05, 0.5, 2.0, 10.0] {
            let h = 0.5 * p;
            let n = 200_000;
            let (lo, hi) = (-10.0, p + 10.0);
            let dz = (hi - lo) / n as f64;
            let num: f64 = (0..n)
                .map(|i| {
                    let z = lo + (i as f64 + 0.5) * dz;
                    ((cdf(z) - 2.0 * cdf(z - h) + cdf(z - p)) / p).abs() * dz
                })
                .sum();
            let g = wavelet_l1_gain(p);
            assert!(
                (g - num).abs() < 1e-3 * num.max(1e-3),
                "p={p}: {g} vs {num}"
            );
        }
    }

    #[test]
    fn limits() {
        assert!(box_gain(1e4) > 0.999);
        assert!(wavelet_gain(1e4) > 0.999);
        assert!(wavelet_gain(0.5) < box_gain(0.5));
        assert!(box_gain(0.01) < 0.01);
    }
}
