use super::fixture::{SCREEN_H, SCREEN_W, SIGMA_PX};
use pink_lemonade_wasm::MergeResult;

#[derive(Default)]
struct Moments {
    mass: f64,
    first_x: f64,
    first_y: f64,
    second_xx: f64,
    second_xy: f64,
    second_yy: f64,
}

pub fn assert_preserved(name: &str, input: &[f64], result: &MergeResult) {
    let sigma_x = SIGMA_PX / SCREEN_W;
    let sigma_y = SIGMA_PX / SCREEN_H;
    let mut expected = Moments::default();
    for point in input.chunks_exact(3) {
        let (x, y, mass) = (point[0], point[1], point[2]);
        expected.mass += mass;
        expected.first_x += mass * x;
        expected.first_y += mass * y;
        expected.second_xx += mass * (x * x + sigma_x * sigma_x);
        expected.second_xy += mass * x * y;
        expected.second_yy += mass * (y * y + sigma_y * sigma_y);
    }

    let instances = result.gpu_instances();
    assert_eq!(
        instances.len(),
        result.count() as usize * 7,
        "{name}: invalid output length"
    );
    let mut actual = Moments::default();
    for kernel in instances.chunks_exact(7) {
        let (x, y) = (kernel[0] as f64, kernel[1] as f64);
        let (p00, p01, p10, p11) = (
            kernel[3] as f64,
            kernel[4] as f64,
            kernel[5] as f64,
            kernel[6] as f64,
        );
        assert_close(name, "covariance symmetry", p01, p10);
        let det = p00 * p11 - p01 * p10;
        assert!(
            det.is_finite() && det > 0.0,
            "{name}: invalid covariance determinant {det}"
        );
        let mass = kernel[2] as f64 * 2.0 * std::f64::consts::PI * det.sqrt();
        actual.mass += mass;
        actual.first_x += mass * x;
        actual.first_y += mass * y;
        actual.second_xx += mass * (x * x + p00 * sigma_x * sigma_x);
        actual.second_xy += mass * (x * y + p01 * sigma_x * sigma_y);
        actual.second_yy += mass * (y * y + p11 * sigma_y * sigma_y);
    }

    for (quantity, observed, reference) in [
        ("mass", actual.mass, expected.mass),
        ("first moment x", actual.first_x, expected.first_x),
        ("first moment y", actual.first_y, expected.first_y),
        ("second moment xx", actual.second_xx, expected.second_xx),
        ("second moment xy", actual.second_xy, expected.second_xy),
        ("second moment yy", actual.second_yy, expected.second_yy),
    ] {
        assert_close(name, quantity, observed, reference);
    }
}

fn assert_close(name: &str, quantity: &str, actual: f64, expected: f64) {
    let tolerance = 2e-5 * expected.abs().max(1.0);
    assert!(
        actual.is_finite() && (actual - expected).abs() <= tolerance,
        "{name}: full-view {quantity} = {actual}, expected {expected} (tolerance {tolerance})"
    );
}
