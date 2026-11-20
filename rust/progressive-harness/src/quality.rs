//! Density quality (NMAE), `y_range`, and `line_points` verification helpers.

use pink_lemonade_progressive_core::{Frame, Point, ViewQuery};

use crate::fixture::SplitMix64;

/// A pixel coordinate and its exact reference KDE density from unmerged input points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PixelSample {
    pub px: f64,
    pub py: f64,
    pub reference_density: f64,
}

/// Summary of density approximation quality for a rendered frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityMetrics {
    pub input_points: usize,
    pub output_kernels: usize,
    pub reduction_percent: f64,
    pub normalized_mean_absolute_error_percent: f64,
}

/// Compute exact visible `[min_y, max_y]` of points in `[min_x, max_x]`.
///
/// Returns `None` if no points fall in the interval.
pub fn exact_visible_y_range(points: &[Point], min_x: f64, max_x: f64) -> Option<[f64; 2]> {
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for p in points {
        if p.x >= min_x && p.x <= max_x {
            min_y = min_y.min(p.y);
            max_y = max_y.max(p.y);
        }
    }
    if min_y.is_finite() {
        Some([min_y, max_y])
    } else {
        None
    }
}

/// Sample occupied pixels near visible points and compute their exact ground-truth KDE density.
pub fn sample_reference_pixels(
    points: &[Point],
    query: &ViewQuery,
    y_range: [f64; 2],
    sample_count: usize,
    seed: u64,
) -> Vec<PixelSample> {
    let visible: Vec<&Point> = points
        .iter()
        .filter(|p| p.x >= query.min_x && p.x <= query.max_x)
        .collect();
    if visible.is_empty() || sample_count == 0 {
        return Vec::new();
    }

    let bounds_y = query.effective_y_bounds(y_range);
    let x_span = (query.max_x - query.min_x).max(1e-12);
    let y_span = (bounds_y[1] - bounds_y[0]).max(1e-12);
    let (x_factor, y_factor) = query.factors(bounds_y);
    let inv_two_pi = 1.0 / (2.0 * std::f64::consts::PI);

    let mut rng = SplitMix64(seed);
    let mut samples = Vec::with_capacity(sample_count);

    for _ in 0..sample_count {
        let idx = (rng.next_u64() as usize) % visible.len();
        let anchor = visible[idx];
        let base_px = (anchor.x - query.min_x) / x_span * query.screen_w;
        let base_py = (anchor.y - bounds_y[0]) / y_span * query.screen_h;
        let px = (base_px + (rng.unit() * 3.0 - 1.5) * query.sigma_px).clamp(0.0, query.screen_w);
        let py = (base_py + (rng.unit() * 3.0 - 1.5) * query.sigma_px).clamp(0.0, query.screen_h);

        let data_x = query.min_x + (px / query.screen_w) * x_span;
        let data_y = bounds_y[0] + (py / query.screen_h) * y_span;

        let reference_density: f64 = visible
            .iter()
            .map(|p| {
                let dx = (data_x - p.x) * x_factor;
                let dy = (data_y - p.y) * y_factor;
                let r2 = dx * dx + dy * dy;
                if r2 > 36.0 {
                    0.0
                } else {
                    p.weight * inv_two_pi * (-0.5 * r2).exp()
                }
            })
            .sum();

        samples.push(PixelSample {
            px,
            py,
            reference_density,
        });
    }
    samples
}

/// Evaluate Normalized Mean Absolute Density Error (NMAE %) of a [`Frame`] against
/// precomputed [`PixelSample`]s, matching `rust/benches/common/quality.rs`.
pub fn evaluate_density_nmae(
    input_points: usize,
    frame: &Frame,
    query: &ViewQuery,
    y_range: [f64; 2],
    samples: &[PixelSample],
) -> Result<QualityMetrics, String> {
    if !frame.gpu_instances.len().is_multiple_of(7) {
        return Err(format!(
            "gpu_instances length {} is not a multiple of 7",
            frame.gpu_instances.len()
        ));
    }
    if samples.is_empty() {
        return Err("cannot evaluate density NMAE with zero pixel samples".to_owned());
    }

    let bounds_y = query.effective_y_bounds(y_range);
    let x_span = (query.max_x - query.min_x).max(1e-12);
    let y_span = (bounds_y[1] - bounds_y[0]).max(1e-12);
    let (x_factor, y_factor) = query.factors(bounds_y);

    let mut abs_error = 0.0;
    let mut ref_total = 0.0;
    for sample in samples {
        let data_x = query.min_x + (sample.px / query.screen_w) * x_span;
        let data_y = bounds_y[0] + (sample.py / query.screen_h) * y_span;
        let reduced = reduced_density(&frame.gpu_instances, data_x, data_y, x_factor, y_factor)?;
        abs_error += (reduced - sample.reference_density).abs();
        ref_total += sample.reference_density;
    }

    let nmae = if ref_total > 0.0 {
        100.0 * abs_error / ref_total
    } else {
        0.0
    };
    let output_kernels = frame.kernel_count();
    let reduction = if input_points > 0 {
        100.0 * (1.0 - output_kernels as f64 / input_points as f64)
    } else {
        0.0
    };

    Ok(QualityMetrics {
        input_points,
        output_kernels,
        reduction_percent: reduction,
        normalized_mean_absolute_error_percent: nmae,
    })
}

fn reduced_density(
    instances: &[f32],
    data_x: f64,
    data_y: f64,
    x_factor: f64,
    y_factor: f64,
) -> Result<f64, String> {
    let mut sum = 0.0;
    for (idx, kernel) in instances.as_chunks::<7>().0.iter().enumerate() {
        let dx = (data_x - kernel[0] as f64) * x_factor;
        let dy = (data_y - kernel[1] as f64) * y_factor;
        let p00 = kernel[3] as f64;
        let p01 = kernel[4] as f64;
        let p11 = kernel[6] as f64;
        let det = p00 * p11 - p01 * p01;
        if !det.is_finite() || det <= 0.0 {
            return Err(format!("kernel {idx}: non-positive covariance det={det}"));
        }
        let d2 = (p11 * dx * dx - 2.0 * p01 * dx * dy + p00 * dy * dy) / det;
        if d2 <= 36.0 {
            sum += kernel[2] as f64 * (-0.5 * d2).exp();
        }
    }
    Ok(sum)
}

/// Verify `frame.y_range` against exact visible `[min_y, max_y]`.
pub fn verify_y_range(
    points: &[Point],
    query: &ViewQuery,
    frame: &Frame,
    abs_tol: f64,
) -> Result<(), String> {
    let expected = exact_visible_y_range(points, query.min_x, query.max_x);
    match (expected, frame.y_range) {
        (None, None) => Ok(()),
        (None, Some(got)) => Err(format!(
            "expected None y_range for empty view, got {got:?}"
        )),
        (Some(exp), None) => Err(format!("expected y_range {exp:?}, got None")),
        (Some([exp_min, exp_max]), Some([got_min, got_max])) => {
            if (got_min - exp_min).abs() > abs_tol || (got_max - exp_max).abs() > abs_tol {
                Err(format!(
                    "y_range mismatch: got [{got_min}, {got_max}], expected [{exp_min}, {exp_max}] (tol {abs_tol})"
                ))
            } else {
                Ok(())
            }
        }
    }
}

/// Verify `frame.line_points` invariants when non-empty:
/// even length, finite, non-decreasing `x`, and preserves visible `min_y` and `max_y`
/// (within f32 precision).
pub fn verify_line_points(points: &[Point], query: &ViewQuery, frame: &Frame) -> Result<(), String> {
    if frame.line_points.is_empty() {
        return Ok(());
    }
    if !frame.line_points.len().is_multiple_of(2) {
        return Err(format!(
            "line_points length {} must be even",
            frame.line_points.len()
        ));
    }

    let mut prev_x = f32::NEG_INFINITY;
    let mut seen_min_y = f32::INFINITY;
    let mut seen_max_y = f32::NEG_INFINITY;

    for (i, pair) in frame.line_points.as_chunks::<2>().0.iter().enumerate() {
        let (x, y) = (pair[0], pair[1]);
        if !x.is_finite() || !y.is_finite() {
            return Err(format!("line_points[{i}] contains non-finite value ({x}, {y})"));
        }
        if x < prev_x {
            return Err(format!(
                "line_points x not sorted at index {i}: {x} < {prev_x}"
            ));
        }
        prev_x = x;
        if (x as f64) >= query.min_x - 1e-6 && (x as f64) <= query.max_x + 1e-6 {
            seen_min_y = seen_min_y.min(y);
            seen_max_y = seen_max_y.max(y);
        }
    }

    if let Some([exp_min, exp_max]) = exact_visible_y_range(points, query.min_x, query.max_x) {
        let exp_min_f32 = exp_min as f32;
        let exp_max_f32 = exp_max as f32;
        let min_tol = 1e-4 * exp_min_f32.abs().max(1.0);
        let max_tol = 1e-4 * exp_max_f32.abs().max(1.0);
        if (seen_min_y - exp_min_f32).abs() > min_tol {
            return Err(format!(
                "line_points missed visible min_y: got {seen_min_y}, expected {exp_min_f32}"
            ));
        }
        if (seen_max_y - exp_max_f32).abs() > max_tol {
            return Err(format!(
                "line_points missed visible max_y: got {seen_max_y}, expected {exp_max_f32}"
            ));
        }
    }

    Ok(())
}
