mod common;

use pink_lemonade_wasm::TreeDownsampler;
use pink_lemonade_wasm::downsampler::Downsampler as DownsamplerTrait;

fn factory() -> Box<dyn DownsamplerTrait> {
    let mut d = TreeDownsampler::new();
    d.set_merge_threshold(10_000.0);
    Box::new(d)
}

#[test]
fn empty_input_returns_no_instances() {
    common::empty_input_returns_no_instances("tree_merge", factory);
}

#[test]
fn output_contains_only_points_inside_the_view() {
    common::output_contains_only_points_inside_the_view("tree_merge", factory);
}

#[test]
fn merged_output_preserves_mass_first_and_second_moments() {
    common::merged_output_preserves_mass_first_and_second_moments("tree_merge", factory);
}

#[test]
fn changing_views_clip_at_both_boundaries_and_reuse_the_tree() {
    let mut d = TreeDownsampler::new();
    d.set_merge_threshold(10_000.0);
    d.set_view_min_y(0.0);
    d.set_view_max_y(1.0);
    d.set_screen_w(1920.0);
    d.set_screen_h(1080.0);
    d.set_sigma_size_px(16.0);
    d.set_data_f64(
        vec![0.0, 0.2, 1.0, 0.25, 0.4, 2.0, 0.5, 0.6, 3.0, 0.75, 0.8, 4.0].into_boxed_slice(),
    );

    for (min_x, max_x, expected_mass, expected_x) in [
        (0.25, 0.5, 5.0, 0.4),
        (0.5, 0.75, 7.0, (0.5 * 3.0 + 0.75 * 4.0) / 7.0),
        (0.0, 0.25, 3.0, (0.25 * 2.0) / 3.0),
    ] {
        d.set_view_min_x(min_x);
        d.set_view_max_x(max_x);
        let result = d.merge_points();
        let gpu = result.gpu_instances();
        let (mass, first_x) = gpu
            .chunks_exact(7)
            .fold((0.0, 0.0), |(mass, first_x), kernel| {
                let det = kernel[3] as f64 * kernel[6] as f64 - (kernel[4] as f64).powi(2);
                let kernel_mass = kernel[2] as f64 * 2.0 * std::f64::consts::PI * det.sqrt();
                (mass + kernel_mass, first_x + kernel_mass * kernel[0] as f64)
            });
        assert!((mass - expected_mass).abs() < 1e-4, "mass {mass}, expected {expected_mass}");
        assert!((first_x / mass - expected_x).abs() < 1e-6);
    }
}

#[test]
fn clipped_view_across_leaf_blocks_preserves_visible_mass() {
    let mut d = TreeDownsampler::new();
    d.set_view_min_x(15.0);
    d.set_view_max_x(80.0);
    d.set_view_min_y(0.0);
    d.set_view_max_y(1.0);
    d.set_screen_w(1920.0);
    d.set_screen_h(1080.0);
    d.set_sigma_size_px(16.0);
    d.set_merge_threshold(10_000.0);
    let data: Vec<f64> = (0..100)
        .flat_map(|i| [i as f64, 0.5, (i % 3 + 1) as f64])
        .collect();
    let expected_mass: f64 = (15..=80).map(|i| (i % 3 + 1) as f64).sum();
    d.set_data_f64(data.into_boxed_slice());
    let mass: f64 = d
        .merge_points()
        .gpu_instances()
        .chunks_exact(7)
        .map(|kernel| {
            let det = kernel[3] as f64 * kernel[6] as f64 - (kernel[4] as f64).powi(2);
            kernel[2] as f64 * 2.0 * std::f64::consts::PI * det.sqrt()
        })
        .sum();
    assert!((mass - expected_mass).abs() < 1e-4);

    d.set_merge_threshold(0.0);
    assert_eq!(d.merge_points().count(), 66);
}
