mod fixture;
mod moments;
mod mybench;
mod quality;

use fixture::{POINTS, Random, SCREEN_H, SCREEN_W, SIGMA_PX, input_data};
use mybench::Mybench;
use pink_lemonade_wasm::downsampler::Downsampler;
use std::time::Instant;

const VIEW_POSITIONS: usize = 10;
const SEED: u64 = 0x6a09_e667_f3bc_c909;

/// Run the same input and timing procedure for any downsampling strategy.
pub fn benchmark(name: &str, downsampler: &mut dyn Downsampler) {
    let data = input_data();

    downsampler.set_view_min_y(0.0);
    downsampler.set_view_max_y(1.0);
    downsampler.set_screen_w(SCREEN_W);
    downsampler.set_screen_h(SCREEN_H);
    downsampler.set_sigma_size_px(SIGMA_PX);
    downsampler.set_data_f64(data.clone().into_boxed_slice());

    // Generate ranges outside the timed section. Widths vary from 10% to 90%
    // of the data range, and each start is chosen within the remaining space.
    let mut random = Random(SEED);
    let views: [(f64, f64); VIEW_POSITIONS] = std::array::from_fn(|_| {
        let width = 0.1 + 0.8 * random.unit();
        let start = (1.0 - width) * random.unit();
        (start, start + width)
    });

    // Warm up and measure the same deterministic collection of random views.
    let bench = Mybench { warmup_runs: VIEW_POSITIONS, measured_runs: VIEW_POSITIONS };
    let mut run = 0;
    let summary = bench.run(|| {
        let (view_min_x, view_max_x) = views[run % VIEW_POSITIONS];
        run += 1;
        downsampler.set_view_min_x(view_min_x);
        downsampler.set_view_max_x(view_max_x);
        downsampler.merge_points()
    });
    println!(
        "{name}, {POINTS} points, 1920x1080, sigma 16px, {} random x views (width 0.1-0.9, seed {SEED:#x}), {} runs: mean {:.3} ms, 95% CI [{:.3}, {:.3}] ms",
        VIEW_POSITIONS,
        summary.measured_runs,
        summary.mean.as_secs_f64() * 1_000.0,
        summary.ci95_low.as_secs_f64() * 1_000.0,
        summary.ci95_high.as_secs_f64() * 1_000.0,
    );

    // Quality is evaluated separately from timing, on the complete view.
    downsampler.set_view_min_x(0.0);
    downsampler.set_view_max_x(1.0);
    let full_view = downsampler.merge_points();
    moments::assert_preserved(name, &data, &full_view);
    let samples = quality::saved_samples();
    let quality_start = Instant::now();
    let quality = quality::evaluate(POINTS, &full_view, &samples);
    let quality_elapsed = quality_start.elapsed();
    println!(
        "full view: {POINTS} -> {} kernels, reduction {:.2}%, normalized mean absolute density error {:.3}% ({} occupied pixels, quality evaluation {:.3} ms)",
        quality.output_count,
        quality.reduction_percent,
        quality.normalized_mean_absolute_error_percent,
        samples.len(),
        quality_elapsed.as_secs_f64() * 1_000.0,
    );
}
