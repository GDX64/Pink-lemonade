#[allow(dead_code)]
#[path = "../benches/common/fixture.rs"]
mod fixture;

#[path = "../benches/common/quality.rs"]
mod quality;

use pink_lemonade_wasm::Downsampler;

#[test]
fn identical_points_reduce_without_density_error() {
    let data = vec![0.5, 0.5, 1.0, 0.5, 0.5, 2.0];
    let mut downsampler = Downsampler::new();
    downsampler.set_view_min_x(0.0);
    downsampler.set_view_max_x(1.0);
    downsampler.set_view_min_y(0.0);
    downsampler.set_view_max_y(1.0);
    downsampler.set_screen_w(1920.0);
    downsampler.set_screen_h(1080.0);
    downsampler.set_sigma_size_px(16.0);
    downsampler.set_merge_threshold(1.0);
    downsampler.set_data_f64(data.clone().into_boxed_slice());

    let sample_x: f64 = 960.5;
    let sample_y: f64 = 540.5;
    let distance_squared = ((sample_x - 960.0) / 16.0).powi(2)
        + ((sample_y - 540.0) / 16.0).powi(2);
    let samples = [quality::Sample {
        x: sample_x,
        y: sample_y,
        original_density: 3.0 * (-0.5 * distance_squared).exp() / (2.0 * std::f64::consts::PI),
    }];
    let metrics = quality::evaluate(data.len() / 3, &downsampler.merge_points(), &samples);
    assert_eq!(metrics.output_count, 1);
    assert_eq!(metrics.reduction_percent, 50.0);
    assert!(metrics.normalized_mean_absolute_error_percent < 0.0001);
}

#[test]
fn saved_reference_contains_occupied_pixels() {
    let samples = quality::saved_samples();
    assert_eq!(samples.len(), 1_000);
    assert!(samples.iter().all(|sample| sample.original_density > 0.0));
}
