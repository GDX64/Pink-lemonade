//! Deterministic streaming dataset generators for tests and gigabyte-scale benchmarks.
//!
//! Generators yield bounded chunks via [`DatasetStream`] so arbitrarily large datasets
//! (tens of millions to billions of points) can be fed into a [`DatasetBuilder`]
//! in O(chunk_size) RAM.

use pink_lemonade_progressive_core::{BuildError, DatasetBuilder, Point};

/// Deterministic SplitMix64 PRNG matching `rust/benches/common/fixture.rs`.
#[derive(Debug, Clone)]
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform float in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64
    }

    /// Standard normal `N(0, 1)` sample via Box-Muller transform (matching `src/data/noise.ts`).
    pub fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-15);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Built-in deterministic dataset shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetShape {
    /// Same curve as `rust/benches/common/fixture.rs`:
    /// `x = i / n`, `y = 0.5 + 0.25 * sin(i * 300 / n)`, `weight = 1.0`.
    SineWave,
    /// Integrated Gaussian noise (Brownian random walk) with bimodal weight distribution,
    /// matching `createNoiseFloatData` in `src/data/noise.ts`.
    RandomWalk { seed: u64 },
    /// Uniform noise in `[-10, 10]` plus tall periodic spikes (`+80` every 997th point),
    /// matching `rust/tests/tree_merge.rs` for testing `y_range` and `line_points` extrema.
    NoisySpikes { seed: u64 },
    /// Non-uniform x spacing with dense bursts, drifting baseline, and varying weights.
    MultiRegime { seed: u64 },
    /// Strongly non-uniform x: 90% of points in ten 1%-wide bursts, 10% spread over the
    /// background, and three 7%-wide stretches with no points at all. `y` is a Brownian random
    /// walk with unit weights. Separates count-based x-partitions from geometric columns.
    BurstyX { seed: u64 },
}

impl DatasetShape {
    pub const fn name(self) -> &'static str {
        match self {
            Self::SineWave => "sine_wave",
            Self::RandomWalk { .. } => "random_walk",
            Self::NoisySpikes { .. } => "noisy_spikes",
            Self::MultiRegime { .. } => "multi_regime",
            Self::BurstyX { .. } => "bursty_x",
        }
    }
}

/// Streaming generator yielding x-sorted chunks of [`Point`]s in O(chunk_size) memory.
#[derive(Debug, Clone)]
pub struct DatasetStream {
    shape: DatasetShape,
    total_points: u64,
    chunk_size: usize,
    cursor: u64,
    rng: SplitMix64,
    walk_acc: f64,
}

impl DatasetStream {
    pub fn new(shape: DatasetShape, total_points: u64, chunk_size: usize) -> Self {
        let seed = match shape {
            DatasetShape::SineWave => 0,
            DatasetShape::RandomWalk { seed }
            | DatasetShape::NoisySpikes { seed }
            | DatasetShape::MultiRegime { seed }
            | DatasetShape::BurstyX { seed } => seed,
        };
        let walk_acc = total_points as f64 / 1000.0;
        Self {
            shape,
            total_points,
            chunk_size: chunk_size.max(1),
            cursor: 0,
            rng: SplitMix64(seed),
            walk_acc,
        }
    }

    pub fn total_points(&self) -> u64 {
        self.total_points
    }

    /// Collect the entire stream into memory (intended for small/medium test fixtures).
    pub fn collect_all(self) -> Vec<Point> {
        let cap = usize::try_from(self.total_points).expect("dataset too large to collect in RAM");
        let mut all = Vec::with_capacity(cap);
        for chunk in self {
            all.extend_from_slice(&chunk);
        }
        all
    }

    fn point_at(&mut self, index: u64) -> Point {
        let n = self.total_points.max(1) as f64;
        match self.shape {
            DatasetShape::SineWave => {
                let x = index as f64 / n;
                let y = 0.5 + 0.25 * (index as f64 * 300.0 / n).sin();
                Point::new(x, y, 1.0)
            }
            DatasetShape::RandomWalk { .. } => {
                let x = index as f64 / n;
                self.walk_acc += self.rng.normal();
                let density = gaussian_sample(0.9, 0.05, x) + gaussian_sample(0.3, 0.1, x) * 0.8;
                let s = density + self.rng.normal() * 0.5;
                let weight = (s.powi(5) * 100.0).abs();
                Point::new(x, self.walk_acc, if weight > 0.0 { weight } else { 1.0 })
            }
            DatasetShape::NoisySpikes { .. } => {
                let x = index as f64 / n;
                let noise = self.rng.unit() * 20.0 - 10.0;
                let spike = if index % 997 == 0 { 80.0 } else { 0.0 };
                Point::new(x, noise + spike, 1.0)
            }
            DatasetShape::MultiRegime { .. } => {
                let t = index as f64 / n;
                let x = (t + 0.04 * (t * std::f64::consts::TAU * 5.0).sin() / std::f64::consts::TAU)
                    .clamp(0.0, 1.0);
                let regime = ((t * 8.0).floor() as u64) % 3;
                let base = 0.5 + 0.18 * (t * 12.0).sin();
                let jitter = (self.rng.unit() - 0.5) * match regime {
                    0 => 0.05,
                    1 => 0.30,
                    _ => 0.12,
                };
                let weight = 0.5 + self.rng.unit() * 4.5;
                Point::new(x, base + jitter, weight)
            }
            DatasetShape::BurstyX { .. } => {
                self.walk_acc += self.rng.normal();
                Point::new(bursty_x(index as f64 / n), self.walk_acc, 1.0)
            }
        }
    }
}

impl Iterator for DatasetStream {
    type Item = Vec<Point>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.total_points {
            return None;
        }
        let remaining = (self.total_points - self.cursor) as usize;
        let count = remaining.min(self.chunk_size);
        let mut chunk = Vec::with_capacity(count);
        for _ in 0..count {
            let pt = self.point_at(self.cursor);
            self.cursor += 1;
            chunk.push(pt);
        }
        Some(chunk)
    }
}

/// Stream a [`DatasetStream`] into any [`DatasetBuilder`] in bounded chunks.
pub fn stream_into_builder(
    stream: DatasetStream,
    builder: &mut dyn DatasetBuilder,
) -> Result<u64, BuildError> {
    let mut count = 0u64;
    for chunk in stream {
        count += chunk.len() as u64;
        builder.push_chunk(&chunk)?;
    }
    Ok(count)
}

/// `BurstyX` layout: ten bursts of 9% of the points each, 1% of x wide, centred on
/// `0.05 + 0.1k`; three empty stretches; the remaining 10% spread uniformly over the rest.
const BURST_HALF_WIDTH: f64 = 0.005;
const BURST_MASS: f64 = 0.09;
const BURSTY_GAPS: [(f64, f64); 3] = [(0.165, 0.235), (0.465, 0.535), (0.765, 0.835)];

/// Piecewise-linear segments `(x0, x1, cumulative mass at x0, mass)` of the `BurstyX` CDF.
fn bursty_segments() -> &'static [(f64, f64, f64, f64)] {
    static SEGMENTS: std::sync::OnceLock<Vec<(f64, f64, f64, f64)>> = std::sync::OnceLock::new();
    SEGMENTS.get_or_init(|| {
        let bursts: Vec<(f64, f64)> = (0..10)
            .map(|k| {
                let c = 0.05 + 0.1 * k as f64;
                (c - BURST_HALF_WIDTH, c + BURST_HALF_WIDTH)
            })
            .collect();
        let mut cuts = vec![0.0, 1.0];
        for &(a, b) in bursts.iter().chain(BURSTY_GAPS.iter()) {
            cuts.extend([a, b]);
        }
        cuts.sort_by(f64::total_cmp);
        cuts.dedup();
        let inside = |set: &[(f64, f64)], m: f64| set.iter().any(|&(a, b)| a <= m && m <= b);
        let gap_len: f64 = BURSTY_GAPS.iter().map(|(a, b)| b - a).sum();
        let background_len = 1.0 - gap_len - bursts.len() as f64 * 2.0 * BURST_HALF_WIDTH;
        let background_mass = 1.0 - bursts.len() as f64 * BURST_MASS;
        let mut segments = Vec::new();
        let mut cumulative = 0.0;
        for w in cuts.windows(2) {
            let (a, b) = (w[0], w[1]);
            let mid = 0.5 * (a + b);
            let mass = if inside(&bursts, mid) {
                BURST_MASS * (b - a) / (2.0 * BURST_HALF_WIDTH)
            } else if inside(&BURSTY_GAPS, mid) {
                continue;
            } else {
                background_mass * (b - a) / background_len
            };
            segments.push((a, b, cumulative, mass));
            cumulative += mass;
        }
        segments
    })
}

/// Monotone inverse CDF of the `BurstyX` layout: `t ∈ [0, 1)` → `x ∈ [0, 1]`.
pub fn bursty_x(t: f64) -> f64 {
    let segments = bursty_segments();
    let i = segments
        .partition_point(|&(_, _, cumulative, _)| cumulative <= t)
        .saturating_sub(1);
    let (x0, x1, cumulative, mass) = segments[i];
    (x0 + (t - cumulative) / mass * (x1 - x0)).clamp(x0, x1)
}

fn gaussian_sample(mean: f64, std_dev: f64, x: f64) -> f64 {
    let z = (x - mean) / std_dev;
    (-0.5 * z * z).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bursty_x_is_sorted_bursty_and_has_empty_gaps() {
        let n = 200_000u64;
        let points = DatasetStream::new(DatasetShape::BurstyX { seed: 1 }, n, 8_192).collect_all();
        assert!(points.windows(2).all(|w| w[0].x <= w[1].x), "x must be sorted");
        assert!(points.iter().all(|p| (0.0..=1.0).contains(&p.x)));
        let in_bursts = points
            .iter()
            .filter(|p| (0..10).any(|k| (p.x - (0.05 + 0.1 * k as f64)).abs() <= BURST_HALF_WIDTH))
            .count() as f64
            / n as f64;
        assert!((in_bursts - 0.9).abs() < 0.005, "burst fraction {in_bursts}");
        for (a, b) in BURSTY_GAPS {
            assert!(points.iter().all(|p| p.x <= a || p.x >= b), "points inside gap ({a}, {b})");
        }
    }
}
