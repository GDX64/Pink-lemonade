//! Paged moment k-d tree strategy (2D space partitioning) for `pink-lemonade-progressive-core`.
//!
//! The builder permutes the input so that every chunk of `L·Fᵏ` consecutive points forms a
//! 2D k-d cell: each chunk is split by repeated binary splits along its longest side (measured
//! in overview screen units, i.e. x scaled by the screen aspect), with split positions snapped
//! to multiples of the child size. Each leaf of `L` points is then sorted by x.
//!
//! Because the permutation keeps the implicit `node j → children [j·F, (j+1)·F)` indexing,
//! the static layout (`manifest.bin` + `level_*.bin` + `points.bin`) and the online reader
//! are exactly those of the 1D paged moment tree (`pink-lemonade-progressive-common`):
//! - cells resolve as one kernel once their 2D bbox fits the merge threshold on screen, so
//!   white noise no longer forces descent to raw points (cells are short in y);
//! - cells straddling the x-viewport boundary still descend to raw leaves for exact clipping,
//!   which costs `O(√cells)` boundary leaves per edge (the known tradeoff vs x-ordering);
//! - with explicit `y_bounds`, cells far off-screen in y are culled into single kernels.
//!
//! Line samples are re-sorted by x since k-d order is not x-monotonic.

use std::sync::Arc;

use pink_lemonade_progressive_common::{
    HierarchyReader, PointCollector, level_counts, write_hierarchy,
};
use pink_lemonade_progressive_core::{
    BuildError, ByteRangeStore, ClientConfig, ClientError, DatasetBuilder, Frame, ObjectWriter,
    Point, ProgressiveClient, ProgressiveStrategy, ViewQuery, async_trait,
};

pub const DEFAULT_LEAF_POINTS: u32 = 32;
/// 8 children per node = 3 binary splits per level (cells shrink ~2.8× in area-equivalent
/// side per level), balancing over-refinement against tree height (RTTs).
pub const DEFAULT_FANOUT: u32 = 8;
/// Inline the top level in `manifest.bin` once it has at most this many nodes (≤ 64 KiB).
pub const DEFAULT_ROOT_CAP: u32 = 1024;
/// x:y weighting when choosing the split axis (16:9 overview screen).
pub const DEFAULT_SPLIT_ASPECT: f64 = 16.0 / 9.0;

pub const MAGIC: [u8; 8] = *b"PLPKDTR1";

#[derive(Debug, Clone, Copy)]
pub struct KdTreeStrategy {
    pub leaf_points: u32,
    pub fanout: u32,
    pub root_cap: u32,
    pub split_aspect: f64,
}

impl Default for KdTreeStrategy {
    fn default() -> Self {
        Self {
            leaf_points: DEFAULT_LEAF_POINTS,
            fanout: DEFAULT_FANOUT,
            root_cap: DEFAULT_ROOT_CAP,
            split_aspect: DEFAULT_SPLIT_ASPECT,
        }
    }
}

#[async_trait]
impl ProgressiveStrategy for KdTreeStrategy {
    fn name(&self) -> &str {
        "moment_kdtree"
    }

    fn create_builder(&self) -> Box<dyn DatasetBuilder> {
        Box::new(KdTreeBuilder {
            config: *self,
            points: PointCollector::default(),
        })
    }

    async fn open_client(
        &self,
        store: Arc<dyn ByteRangeStore>,
        config: ClientConfig,
    ) -> Result<Box<dyn ProgressiveClient>, ClientError> {
        let reader = HierarchyReader::open(store, MAGIC, config.memory_budget_bytes).await?;
        Ok(Box::new(KdTreeClient { reader }))
    }
}

pub struct KdTreeBuilder {
    config: KdTreeStrategy,
    points: PointCollector,
}

impl DatasetBuilder for KdTreeBuilder {
    fn push_chunk(&mut self, points: &[Point]) -> Result<(), BuildError> {
        self.points.push_chunk(points)
    }

    fn finish(self: Box<Self>, writer: &mut dyn ObjectWriter) -> Result<(), BuildError> {
        let KdTreeStrategy {
            leaf_points,
            fanout,
            root_cap,
            split_aspect,
        } = self.config;
        let (leaf_points, fanout) = (leaf_points.max(1), fanout.max(2));
        let mut points = self.points.points;

        let counts = level_counts(points.len() as u64, leaf_points, fanout, root_cap);
        if !counts.is_empty() {
            // Child sizes top-down: root nodes hold L·F^R points, ..., leaves hold L.
            let root_level = counts.len() - 1;
            let sizes: Vec<usize> = (0..=root_level)
                .rev()
                .map(|level| leaf_points as usize * (fanout as usize).pow(level as u32))
                .collect();
            let scale = axis_scale(&points, split_aspect);
            kd_order(&mut points, &sizes, scale);
        }

        write_hierarchy(&points, MAGIC, leaf_points, fanout, root_cap, writer)?;
        Ok(())
    }
}

/// Per-axis weights turning data extents into overview-screen units.
fn axis_scale(points: &[Point], aspect: f64) -> (f64, f64) {
    let (lo, hi) = bbox(points);
    let span = |a: f64, b: f64| if b > a { b - a } else { 1.0 };
    (aspect / span(lo[0], hi[0]), 1.0 / span(lo[1], hi[1]))
}

fn bbox(points: &[Point]) -> ([f64; 2], [f64; 2]) {
    points.iter().fold(
        ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]),
        |(lo, hi), p| {
            (
                [lo[0].min(p.x), lo[1].min(p.y)],
                [hi[0].max(p.x), hi[1].max(p.y)],
            )
        },
    )
}

/// Recursively permute `points` so each `sizes[0]`-chunk is a k-d cell, then recurse into
/// every chunk with `sizes[1..]`; the final (leaf) chunks are sorted by x.
fn kd_order(points: &mut [Point], sizes: &[usize], scale: (f64, f64)) {
    let Some((&size, rest)) = sizes.split_first() else {
        return;
    };
    split_into_cells(points, size, scale);
    for chunk in points.chunks_mut(size) {
        if rest.is_empty() {
            chunk.sort_by(|a, b| a.x.total_cmp(&b.x));
        } else {
            kd_order(chunk, rest, scale);
        }
    }
}

/// Binary-split `points` along the longest scaled side until every group of `cell` points
/// (with the remainder last) is spatially separated from the others.
fn split_into_cells(points: &mut [Point], cell: usize, scale: (f64, f64)) {
    if points.len() <= cell {
        return;
    }
    let children = points.len().div_ceil(cell);
    let mid = (children / 2) * cell;
    let (lo, hi) = bbox(points);
    let split_on_x = (hi[0] - lo[0]) * scale.0 >= (hi[1] - lo[1]) * scale.1;
    if split_on_x {
        points.select_nth_unstable_by(mid, |a, b| a.x.total_cmp(&b.x));
    } else {
        points.select_nth_unstable_by(mid, |a, b| a.y.total_cmp(&b.y));
    }
    let (left, right) = points.split_at_mut(mid);
    split_into_cells(left, cell, scale);
    split_into_cells(right, cell, scale);
}

pub struct KdTreeClient {
    reader: HierarchyReader,
}

#[async_trait]
impl ProgressiveClient for KdTreeClient {
    async fn query(&mut self, query: ViewQuery) -> Result<Frame, ClientError> {
        self.reader.standard_query(query, true).await
    }

    fn memory_bytes(&self) -> usize {
        self.reader.memory_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kd_order_produces_disjoint_cells() {
        // 4×4 grid of points, cells of 4 points: each cell must be a 2×2 block.
        let mut pts: Vec<Point> = (0..16)
            .map(|i| Point::new((i / 4) as f64, (i % 4) as f64, 1.0))
            .collect();
        kd_order(&mut pts, &[4], (1.0, 1.0));
        for cell in pts.chunks(4) {
            let (lo, hi) = bbox(cell);
            assert!(hi[0] - lo[0] <= 1.0 && hi[1] - lo[1] <= 1.0, "{cell:?}");
            assert!(cell.windows(2).all(|w| w[0].x <= w[1].x));
        }
    }
}
