//! Moment ripmap strategy: decoupled x/y-resolution dense moment planes.
//!
//! Static layout:
//! - An x-ordered paged moment hierarchy (`manifest.bin` `PLPRIPM1`, `level_*.bin`,
//!   `points.bin`) with a small fanout (default 4) so x-block widths come in fine steps.
//!   It provides exact `y_range`, line extrema, and boundary clipping exactly like the 1D tree.
//! - `ripmap.bin`: plane parameters and the deepest y-depth `kmax(ℓ)` stored per x-level.
//! - `plane_{ℓ}_{k}.bin`: for every x-block (node) `j` at level `ℓ`, `2^k` dense 24-byte
//!   [`MomentCell`](pink_lemonade_progressive_common::MomentCell) slots splitting the block's own
//!   `[min_y, max_y]` into equal bins (empty bins are zero). Block `j`'s cells are the
//!   contiguous byte range `[j·2^k, (j+1)·2^k) × 24`. Planes are only stored while blocks hold
//!   at least `min_points_per_bin` points per bin on average (sparser data is cheaper raw).
//!
//! Online, a fully-inside x-node at level `ℓ` with on-screen width `w ≤ 0.9σ` (and that does
//! not already fit as one kernel) is rendered from plane `(ℓ, k)` with the smallest `k` whose
//! bin height `h` satisfies `w² + h² ≤ 1` (in merge-threshold units). The whole view thus costs
//! roughly one contiguous byte range per `(ℓ, k)` run of blocks regardless of signal shape,
//! instead of descending to raw points on vertically incoherent data (white noise).

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
/// Planes are stored only while `block_points / 2^k >= min_points_per_bin`.
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
