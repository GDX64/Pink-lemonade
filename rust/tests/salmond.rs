mod common;

use pink_lemonade_wasm::SalmondDownsampler;
use pink_lemonade_wasm::downsampler::Downsampler as DownsamplerTrait;

fn factory() -> Box<dyn DownsamplerTrait> {
    let mut d = SalmondDownsampler::new();
    d.set_merge_threshold(1_000.0);
    d.set_target_count(1.0);
    Box::new(d)
}

#[test]
fn empty_input_returns_no_instances() {
    common::empty_input_returns_no_instances("salmond", factory);
}

#[test]
fn output_contains_only_points_inside_the_view() {
    common::output_contains_only_points_inside_the_view("salmond", factory);
}

#[test]
fn merged_output_preserves_mass_first_and_second_moments() {
    common::merged_output_preserves_mass_first_and_second_moments("salmond", factory);
}
