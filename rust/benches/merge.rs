mod common;

use pink_lemonade_wasm::Downsampler;

fn main() {
    let mut downsampler = Downsampler::new();
    downsampler.set_merge_threshold(1.0);
    common::benchmark("merge", &mut downsampler);
}
