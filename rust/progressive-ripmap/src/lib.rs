//! Moment ripmap strategy (`PLPRIPM2`): decoupled x/y-resolution sparse moment planes.
//!
//! Static layout (at most 4 objects):
//! - `manifest.bin` (`PLPRIPM2`): header, per-level offsets and plane directory metadata, plus
//!   the inlined top (root) level records — loaded in a single EOF-clamped `read_range` (1 RTT).
//! - `levels.bin`: non-root x-ordered hierarchy levels `0..root_level` packed back-to-back.
//!   Each record at level `ℓ` holds a 64-byte `PackedNode` followed by inline 6-byte
//!   `(cell_offset: u32, cell_count: u16)` slots locating that block's sparse cells for every
//!   stored y-depth `k` at level `ℓ` (zero extra RTTs for sparse offset lookup).
//! - `planes.bin`: all stored `(ℓ, k)` sparse moment planes packed back-to-back as 16-byte
//!   quantized `SparseCell`s (non-empty bins only, sorted by `bin: u16`). Per-block adaptive
//!   gating stores a block in plane `(ℓ, k)` whenever its non-empty bin count is at most
//!   `block_points / min_points_per_bin`, letting outlier-stretched blocks reach +2 extra depths
//!   without paying for empty bins.
//! - `points.bin`: flat packed 24-byte `(x, y, weight)` samples in x-sorted order.

mod builder;
mod client;
mod format;

use std::sync::Arc;

use pink_lemonade_progressive_core::{
    ByteRangeStore, ClientConfig, ClientError, DatasetBuilder, ProgressiveClient,
    ProgressiveStrategy, async_trait,
};

pub use builder::RipmapBuilder;
pub use client::RipmapClient;

pub const DEFAULT_LEAF_POINTS: u32 = 32;
/// Small x-fanout keeps consecutive x-block widths within 4× of each other, bounding the
/// over-refinement from choosing a block narrower than ideal.
pub const DEFAULT_FANOUT: u32 = 4;
pub const DEFAULT_ROOT_CAP: u32 = 1024;
/// Per-block non-empty bin budget: a block is stored in plane `(ℓ, k)` only when its
/// non-empty bin count is at most `block_points / min_points_per_bin`.
pub const DEFAULT_MIN_POINTS_PER_BIN: u32 = 16;
/// Hard cap on y-depth (`2^14` bins per block).
pub const MAX_Y_DEPTH: u8 = 14;

#[derive(Debug, Clone, Copy)]
pub struct RipmapStrategy {
    pub leaf_points: u32,
    pub fanout: u32,
    pub root_cap: u32,
    pub min_points_per_bin: u32,
}

impl Default for RipmapStrategy {
    fn default() -> Self {
        Self {
            leaf_points: DEFAULT_LEAF_POINTS,
            fanout: DEFAULT_FANOUT,
            root_cap: DEFAULT_ROOT_CAP,
            min_points_per_bin: DEFAULT_MIN_POINTS_PER_BIN,
        }
    }
}

#[async_trait]
impl ProgressiveStrategy for RipmapStrategy {
    fn name(&self) -> &str {
        "moment_ripmap"
    }

    fn create_builder(&self) -> Box<dyn DatasetBuilder> {
        Box::new(RipmapBuilder::new(*self))
    }

    async fn open_client(
        &self,
        store: Arc<dyn ByteRangeStore>,
        config: ClientConfig,
    ) -> Result<Box<dyn ProgressiveClient>, ClientError> {
        Ok(Box::new(RipmapClient::open(store, config).await?))
    }
}
