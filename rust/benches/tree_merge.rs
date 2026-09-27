mod common;

use pink_lemonade_wasm::TreeDownsampler;

fn main() {
    let mut downsampler = TreeDownsampler::new();
    downsampler.set_merge_threshold(1.0);
    common::benchmark("tree_merge", &mut downsampler);
}
