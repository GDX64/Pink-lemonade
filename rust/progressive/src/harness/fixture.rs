//! Deterministic streaming dataset generators for tests and gigabyte-scale benchmarks.
//!
//! Generators yield bounded chunks via [`DatasetStream`] so arbitrarily large datasets
//! (tens of millions to billions of points) can be fed into a [`crate::DatasetBuilder`]
//! in O(chunk_size) RAM.

use crate::strategy::DatasetBuilder;
use crate::types::{BuildError, Point};

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
}

/// Built-in deterministic dataset shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetShape {
    /// Same curve as `rust/benches/common/fixture.rs`:
    /// `x = i / n`, `y = 0.5 + 0.25 * sin(i * 300 / n)`, `weight = 1.0`.
    SineWave,
    /// Uniform noise in `[-10, 10]` plus tall periodic spikes (`+80` every 997th point),
    /// matching `rust/tests/tree_merge.rs` for testing `y_range` and `line_points` extrema.
    NoisySpikes { seed: u64 },
    /// Non-uniform x spacing with dense bursts, drifting baseline, and varying weights.
    MultiRegime { seed: u64 },
}

/// Streaming generator yielding x-sorted chunks of [`Point`]s in O(chunk_size) memory.
#[derive(Debug, Clone)]
pub struct DatasetStream {
    shape: DatasetShape,
    total_points: u64,
    chunk_size: usize,
    cursor: u64,
    rng: SplitMix64,
}

impl DatasetStream {
    pub fn new(shape: DatasetShape, total_points: u64, chunk_size: usize) -> Self {
        let seed = match shape {
            DatasetShape::SineWave => 0,
            DatasetShape::NoisySpikes { seed } | DatasetShape::MultiRegime { seed } => seed,
        };
        Self {
            shape,
            total_points,
            chunk_size: chunk_size.max(1),
            cursor: 0,
            rng: SplitMix64(seed),
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
            DatasetShape::NoisySpikes { .. } => {
                let x = index as f64 / n;
                let noise = self.rng.unit() * 20.0 - 10.0;
                let spike = if index % 997 == 0 { 80.0 } else { 0.0 };
                Point::new(x, noise + spike, 1.0)
            }
            DatasetShape::MultiRegime { .. } => {
                // Warp x slightly with a smooth monotonic function to create dense/sparse bands.
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
pub fn stream_into_builder<B: DatasetBuilder>(
    stream: DatasetStream,
    builder: &mut B,
) -> Result<u64, BuildError> {
    let mut count = 0u64;
    for chunk in stream {
        count += chunk.len() as u64;
        builder.push_chunk(&chunk)?;
    }
    Ok(count)
}
