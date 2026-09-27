use pink_lemonade_wasm::downsampler::Downsampler as DownsamplerTrait;
pub type Factory = fn() -> Box<dyn DownsamplerTrait>;

fn configure(d: &mut dyn DownsamplerTrait) {
    // Data units and sigma-normalised screen units coincide in this fixture.
    d.set_view_min_x(0.0);
    d.set_view_max_x(10.0);
    d.set_view_min_y(0.0);
    d.set_view_max_y(10.0);
    d.set_screen_w(10.0);
    d.set_screen_h(10.0);
    d.set_sigma_size_px(1.0);
}

fn assert_close(strategy: &str, quantity: &str, actual: f64, expected: f64) {
    let tolerance = 2e-5 * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "{strategy}: {quantity} = {actual}, expected {expected} (tolerance {tolerance})"
    );
}

pub fn empty_input_returns_no_instances(name: &str, factory: Factory) {
    let mut d = factory();
    configure(d.as_mut());
    d.set_data_f64(Vec::new().into_boxed_slice());
    let result = d.merge_points();
    assert_eq!(result.count(), 0, "{name}");
    assert!(result.gpu_instances().is_empty(), "{name}");
}

pub fn output_contains_only_points_inside_the_view(name: &str, factory: Factory) {
    let mut d = factory();
    configure(d.as_mut());
    d.set_data_f64(
        vec![
            -1.0, 4.0, 8.0, // before the x window
            3.0, 5.0, 2.0, // only visible point
            11.0, 6.0, 9.0, // after the x window
        ]
        .into_boxed_slice(),
    );
    let result = d.merge_points();
    assert_eq!(result.count(), 1, "{name}");
    let gpu = result.gpu_instances();
    assert_close(name, "visible x", gpu[0] as f64, 3.0);
    assert_close(name, "visible y", gpu[1] as f64, 5.0);
    assert_close(name, "visible mass", gpu[2] as f64 * 2.0 * std::f64::consts::PI, 2.0);
}

pub fn merged_output_preserves_mass_first_and_second_moments(name: &str, factory: Factory) {
    // Sorted by x, as required by the view-window search. Unequal weights and
    // asymmetric positions exercise both diagonal and cross covariance terms.
    let points: [(f64, f64, f64); 4] = [
        (1.0, 2.0, 1.0),
        (2.0, 5.0, 3.0),
        (4.0, 3.0, 2.0),
        (7.0, 8.0, 4.0),
    ];
    let mass: f64 = points.iter().map(|p| p.2).sum();
    let mean_x = points.iter().map(|p| p.0 * p.2).sum::<f64>() / mass;
    let mean_y = points.iter().map(|p| p.1 * p.2).sum::<f64>() / mass;
    let second_xx = points.iter().map(|p| p.2 * (1.0 + p.0 * p.0)).sum::<f64>() / mass;
    let second_xy = points.iter().map(|p| p.2 * p.0 * p.1).sum::<f64>() / mass;
    let second_yy = points.iter().map(|p| p.2 * (1.0 + p.1 * p.1)).sum::<f64>() / mass;
    let data: Vec<f64> = points.iter().flat_map(|p| [p.0, p.1, p.2]).collect();

    let mut d = factory();
    configure(d.as_mut());
    d.set_data_f64(data.into_boxed_slice());
    let result = d.merge_points();
    assert_eq!(result.count(), 1, "{name}: fixture must actually merge");
    let gpu = result.gpu_instances();
    assert_eq!(gpu.len(), 7, "{name}");

    let x = gpu[0] as f64;
    let y = gpu[1] as f64;
    let amplitude = gpu[2] as f64;
    let p00 = gpu[3] as f64;
    let p01 = gpu[4] as f64;
    let p10 = gpu[5] as f64;
    let p11 = gpu[6] as f64;
    assert_close(name, "covariance symmetry", p01, p10);
    let det = p00 * p11 - p01 * p10;
    assert!(det > 0.0, "{name}: covariance must be positive definite");
    let output_mass = amplitude * 2.0 * std::f64::consts::PI * det.sqrt();

    assert_close(name, "mass", output_mass, mass);
    assert_close(name, "mean x", x, mean_x);
    assert_close(name, "mean y", y, mean_y);
    assert_close(name, "second moment xx", p00 + x * x, second_xx);
    assert_close(name, "second moment xy", p01 + x * y, second_xy);
    assert_close(name, "second moment yy", p11 + y * y, second_yy);
}
