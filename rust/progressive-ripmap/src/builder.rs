//! Offline builder: x-ordered moment hierarchy + dense per-block y-bin moment planes.

use std::io::Write;

use pink_lemonade_progressive_common::{
    MomentCell, Moments, PointCollector, level_counts, write_hierarchy,
};
use pink_lemonade_progressive_core::{
    BuildError, DatasetBuilder, ObjectWriter, Point, StoreError,
};

use crate::RipmapStrategy;
use crate::format::{
    HIERARCHY_MAGIC, RIPMAP_KEY, RipmapHeader, bin_geometry, max_depth, plane_key,
};

pub struct RipmapBuilder {
    config: RipmapStrategy,
    points: PointCollector,
}

impl RipmapBuilder {
    pub fn new(config: RipmapStrategy) -> Self {
        Self {
            config: RipmapStrategy {
                leaf_points: config.leaf_points.max(1),
                fanout: config.fanout.max(2),
                root_cap: config.root_cap.max(1),
                min_points_per_bin: config.min_points_per_bin.max(1),
            },
            points: PointCollector::default(),
        }
    }
}

impl DatasetBuilder for RipmapBuilder {
    fn push_chunk(&mut self, points: &[Point]) -> Result<(), BuildError> {
        self.points.push_chunk(points)
    }

    fn finish(self: Box<Self>, writer: &mut dyn ObjectWriter) -> Result<(), BuildError> {
        let RipmapStrategy {
            leaf_points,
            fanout,
            root_cap,
            min_points_per_bin,
        } = self.config;
        let points = self.points.points;

        write_hierarchy(&points, HIERARCHY_MAGIC, leaf_points, fanout, root_cap, writer)?;

        let counts = level_counts(points.len() as u64, leaf_points, fanout, root_cap);
        let mut kmax = Vec::with_capacity(counts.len());
        for level in 0..counts.len() {
            let block_points = leaf_points as u64 * (fanout as u64).pow(level as u32);
            let depth = max_depth(block_points, min_points_per_bin);
            if depth > 0 {
                write_planes(writer, &points, level, block_points as usize, depth)?;
            }
            kmax.push(depth);
        }

        let header = RipmapHeader {
            min_points_per_bin,
            kmax,
        };
        writer.write_object(RIPMAP_KEY, &header.encode())?;
        Ok(())
    }
}

/// Write `plane_{level}_{k}.bin` for `k = 1..=kmax`, one dense `2^k`-cell slot array per block.
fn write_planes(
    writer: &mut dyn ObjectWriter,
    points: &[Point],
    level: usize,
    block_points: usize,
    kmax: u8,
) -> Result<(), BuildError> {
    let mut sinks: Vec<Box<dyn Write + Send>> = (1..=kmax)
        .map(|k| writer.create_object(&plane_key(level, k)))
        .collect::<Result<_, _>>()?;

    let finest = 1usize << kmax;
    let mut bins = vec![Moments::default(); finest];
    let mut scratch: Vec<u8> = Vec::with_capacity(finest * std::mem::size_of::<MomentCell>());

    for block in points.chunks(block_points) {
        let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
        for p in block {
            min_x = min_x.min(p.x);
            max_x = max_x.max(p.x);
            min_y = min_y.min(p.y);
            max_y = max_y.max(p.y);
        }
        let (dx, dy) = (max_x - min_x, max_y - min_y);

        bins.fill(Moments::default());
        for &p in block {
            let b = if dy > 0.0 {
                (((p.y - min_y) / dy * finest as f64) as usize).min(finest - 1)
            } else {
                0
            };
            bins[b].push(p);
        }

        // Emit the finest depth first, then halve by merging sibling bins in place.
        let mut len = finest;
        for k in (1..=kmax).rev() {
            scratch.clear();
            for (b, &m) in bins[..len].iter().enumerate() {
                let (bin_lo, bin_h) = bin_geometry(min_y, max_y, k, b);
                scratch.extend_from_slice(&MomentCell::pack(m, min_x, dx, bin_lo, bin_h).to_bytes());
            }
            sinks[k as usize - 1]
                .write_all(&scratch)
                .map_err(StoreError::from)?;

            len /= 2;
            for b in 0..len {
                let mut merged = bins[2 * b];
                merged.merge(bins[2 * b + 1]);
                bins[b] = merged;
            }
        }
    }

    for mut sink in sinks {
        sink.flush().map_err(StoreError::from)?;
    }
    Ok(())
}
