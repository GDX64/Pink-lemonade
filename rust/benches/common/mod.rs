mod mybench;

use mybench::Mybench;
use pink_lemonade_wasm::downsampler::Downsampler;

const POINTS: usize = 100_000;
const VIEW_POSITIONS: usize = 10;
const SEED: u64 = 0x6a09_e667_f3bc_c909;

/// Small deterministic generator so every strategy sees the same view ranges.
struct Random(u64);

impl Random {
    fn unit(&mut self) -> f64 {
        // SplitMix64, using the high 53 bits to form a number in [0, 1).
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1_u64 << 53) as f64
    }
}

/// Run the same input and timing procedure for any downsampling strategy.
pub fn benchmark(name: &str, downsampler: &mut dyn Downsampler) {
    let mut data = Vec::with_capacity(POINTS * 3);
    for i in 0..POINTS {
        let x = i as f64 / POINTS as f64;
        let y = 0.5 + 0.25 * (i as f64 * 0.003).sin();
        data.extend_from_slice(&[x, y, 1.0]);
    }

    downsampler.set_view_min_y(0.0);
    downsampler.set_view_max_y(1.0);
    downsampler.set_screen_w(1_000.0);
    downsampler.set_screen_h(1_000.0);
    downsampler.set_sigma_size_px(1.0);
    downsampler.set_data_f64(data.into_boxed_slice());

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
        "{name}, {POINTS} points, {} random x views (width 0.1–0.9, seed {SEED:#x}), {} runs: mean {:.3} ms, 95% CI [{:.3}, {:.3}] ms",
        VIEW_POSITIONS,
        summary.measured_runs,
        summary.mean.as_secs_f64() * 1_000.0,
        summary.ci95_low.as_secs_f64() * 1_000.0,
        summary.ci95_high.as_secs_f64() * 1_000.0,
    );
}
