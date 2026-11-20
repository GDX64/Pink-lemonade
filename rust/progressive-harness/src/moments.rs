//! Exact mass, first-moment, and second-moment preservation checks for arbitrary viewports.
//!
//! Generalizes `rust/benches/common/moments.rs` so both full views and clipped sub-views
//! can be checked against ground-truth input points.

use pink_lemonade_progressive_core::{Frame, Point, ViewQuery};

/// Mass, first moments, and uncentered second moments of a 2D Gaussian mixture.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Moments {
    pub mass: f64,
    pub first_x: f64,
    pub first_y: f64,
    pub second_xx: f64,
    pub second_xy: f64,
    pub second_yy: f64,
}

/// Reference moments of all points whose `x` lies in `[query.min_x, query.max_x]`.
pub fn expected_moments_for_view(points: &[Point], query: &ViewQuery, y_range: [f64; 2]) -> Moments {
    let (x_factor, y_factor) = query.factors(query.effective_y_bounds(y_range));
    let sigma_x = 1.0 / x_factor;
    let sigma_y = 1.0 / y_factor;

    let mut expected = Moments::default();
    for p in points {
        if p.x < query.min_x || p.x > query.max_x {
            continue;
        }
        expected.mass += p.weight;
        expected.first_x += p.weight * p.x;
        expected.first_y += p.weight * p.y;
        expected.second_xx += p.weight * (p.x * p.x + sigma_x * sigma_x);
        expected.second_xy += p.weight * p.x * p.y;
        expected.second_yy += p.weight * (p.y * p.y + sigma_y * sigma_y);
    }
    expected
}

/// Recover total mass, first moments, and second moments from a rendered [`Frame`]'s
/// packed 7-float Gaussian kernels.
pub fn frame_moments(frame: &Frame, query: &ViewQuery, y_range: [f64; 2]) -> Result<Moments, String> {
    if !frame.gpu_instances.len().is_multiple_of(7) {
        return Err(format!(
            "gpu_instances length {} is not a multiple of 7",
            frame.gpu_instances.len()
        ));
    }
    let (x_factor, y_factor) = query.factors(query.effective_y_bounds(y_range));
    let sigma_x = 1.0 / x_factor;
    let sigma_y = 1.0 / y_factor;

    let mut actual = Moments::default();
    for (idx, kernel) in frame.gpu_instances.as_chunks::<7>().0.iter().enumerate() {
        let (x, y) = (kernel[0] as f64, kernel[1] as f64);
        let (amp, p00, p01, p10, p11) = (
            kernel[2] as f64,
            kernel[3] as f64,
            kernel[4] as f64,
            kernel[5] as f64,
            kernel[6] as f64,
        );
        if (p01 - p10).abs() > 1e-5 * p01.abs().max(1.0) {
            return Err(format!("kernel {idx}: asymmetric covariance p01={p01}, p10={p10}"));
        }
        let det = p00 * p11 - p01 * p10;
        if !det.is_finite() || det <= 0.0 {
            return Err(format!("kernel {idx}: non-positive covariance det={det}"));
        }
        let mass = amp * 2.0 * std::f64::consts::PI * det.sqrt();
        if !mass.is_finite() {
            return Err(format!("kernel {idx}: non-finite mass"));
        }
        actual.mass += mass;
        actual.first_x += mass * x;
        actual.first_y += mass * y;
        actual.second_xx += mass * (x * x + p00 * sigma_x * sigma_x);
        actual.second_xy += mass * (x * y + p01 * sigma_x * sigma_y);
        actual.second_yy += mass * (y * y + p11 * sigma_y * sigma_y);
    }
    Ok(actual)
}

/// Check that `actual` matches `expected` across all 6 moment quantities within `rel_tol`.
pub fn check_moments_close(
    label: &str,
    actual: &Moments,
    expected: &Moments,
    rel_tol: f64,
) -> Result<(), String> {
    for (quantity, obs, ref_val) in [
        ("mass", actual.mass, expected.mass),
        ("first moment x", actual.first_x, expected.first_x),
        ("first moment y", actual.first_y, expected.first_y),
        ("second moment xx", actual.second_xx, expected.second_xx),
        ("second moment xy", actual.second_xy, expected.second_xy),
        ("second moment yy", actual.second_yy, expected.second_yy),
    ] {
        let tol = rel_tol * ref_val.abs().max(1.0);
        if !obs.is_finite() || (obs - ref_val).abs() > tol {
            return Err(format!(
                "{label}: {quantity} = {obs}, expected {ref_val} (tolerance {tol})"
            ));
        }
    }
    Ok(())
}

/// Convenience helper combining [`expected_moments_for_view`], [`frame_moments`], and [`check_moments_close`].
pub fn assert_moments_preserved(
    label: &str,
    points: &[Point],
    query: &ViewQuery,
    y_range: [f64; 2],
    frame: &Frame,
    rel_tol: f64,
) -> Result<(), String> {
    let expected = expected_moments_for_view(points, query, y_range);
    let actual = frame_moments(frame, query, y_range)?;
    check_moments_close(label, &actual, &expected, rel_tol)
}
