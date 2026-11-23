//! Sparse moment grid pyramid strategy for `pink-lemonade-progressive-core`.
//!
//! The data bounding box is cut into power-of-two grids: `2^i` columns × `2^j` rows. Only
//! non-empty cells are stored (24-byte moments + row index), column-major, with per-column
//! start pointers (CSC). Instead of one square-cell pyramid, grids are stored for several cell
//! aspect ratios `d = i - j` (by default `-2, 0, 2, …, 12`: from rows 4× finer than columns
//! to columns 4096× finer than rows), so an x-zoomed view of dense data picks tall-thin cells
//! instead of over-refining in y like a quadtree.
//!
//! Every cell is direct-addressed, so a query needs no tree walk:
//! 1. one round of reads fetches the column records of the visible columns at the two
//!    candidate x-levels (exact column nodes, point starts, and the CSC pointers of every
//!    plane at that level) plus the edge columns of the finer levels;
//! 2. the client computes the exact y-range, picks the plane with the fewest bytes whose
//!    cells fit the merge threshold on screen, and reads the visible cells (one contiguous
//!    range per plane) together with the raw points of the edge columns.
//!
//! Exactness follows the shared rule: a column is summarized only when its exact node lies
//! fully inside `[min_x, max_x]`; a straddling column is split into its two children down to
//! the deepest level (or `col_leaf` points), whose points are read raw and clipped.

use std::sync::Arc;

use pink_lemonade_progressive_common::PointCollector;
use pink_lemonade_progressive_core::{
    ByteRangeStore, ClientConfig, ClientError, DatasetBuilder, ProgressiveClient,
    ProgressiveStrategy, async_trait,
};

mod builder;
mod client;
pub mod format;

pub use builder::SparseGridBuilder;
pub use client::SparseGridClient;
pub use format::MAGIC;

/// Deepest columns average at most this many points (raw edge reads stay this small).
pub const DEFAULT_COL_LEAF: u32 = 256;
/// Each plane stores at most `N / min_points_per_cell` cells (its byte budget).
pub const DEFAULT_MIN_POINTS_PER_CELL: u32 = 16;
/// A column qualifies for a plane only if its cells average at least this many points
/// (below that, its raw points cost about as little as its cells).
pub const DEFAULT_MIN_COLUMN_POINTS_PER_CELL: u32 = 4;
pub const DEFAULT_MIN_ASPECT: i32 = -2;
pub const DEFAULT_MAX_ASPECT: i32 = 12;
pub const DEFAULT_ASPECT_STEP: u32 = 2;

#[derive(Debug, Clone, Copy)]
pub struct SparseGridStrategy {
    pub col_leaf: u32,
    pub min_points_per_cell: u32,
    /// Per-column floor: qualifying columns average at least this many points per cell.
    pub min_column_points_per_cell: u32,
    /// Stored aspects `d = i - j` (columns `2^d`× finer than rows) run from `min_aspect` to
    /// `max_aspect` in steps of `aspect_step`.
    pub min_aspect: i32,
    pub max_aspect: i32,
    pub aspect_step: u32,
}

impl Default for SparseGridStrategy {
    fn default() -> Self {
        Self {
            col_leaf: DEFAULT_COL_LEAF,
            min_points_per_cell: DEFAULT_MIN_POINTS_PER_CELL,
            min_column_points_per_cell: DEFAULT_MIN_COLUMN_POINTS_PER_CELL,
            min_aspect: DEFAULT_MIN_ASPECT,
            max_aspect: DEFAULT_MAX_ASPECT,
            aspect_step: DEFAULT_ASPECT_STEP,
        }
    }
}

impl SparseGridStrategy {
    pub(crate) fn aspects(&self) -> impl Iterator<Item = i64> {
        (self.min_aspect as i64..=self.max_aspect as i64).step_by(self.aspect_step.max(1) as usize)
    }
}

#[async_trait]
impl ProgressiveStrategy for SparseGridStrategy {
    fn name(&self) -> &str {
        "sparse_grid"
    }

    fn create_builder(&self) -> Box<dyn DatasetBuilder> {
        Box::new(SparseGridBuilder {
            config: *self,
            points: PointCollector::default(),
        })
    }

    async fn open_client(
        &self,
        store: Arc<dyn ByteRangeStore>,
        config: ClientConfig,
    ) -> Result<Box<dyn ProgressiveClient>, ClientError> {
        Ok(Box::new(SparseGridClient::open(store, config.memory_budget_bytes).await?))
    }
}
