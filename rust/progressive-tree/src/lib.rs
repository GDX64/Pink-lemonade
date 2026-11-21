//! Paged multi-level moment tree strategy for `pink-lemonade-progressive-core`.
//!
//! Organizes x-sorted points into fixed-size leaf blocks (`points.bin`) topped by a
//! high-fanout hierarchy of 128-byte moment + extrema summary nodes (`level_*.bin`), with
//! the top root level inlined in `manifest.bin`.
//!
//! Queries descend only into nodes overlapping `[min_x, max_x]` whose screen-space span
//! exceeds `merge_threshold`, coalescing missing pages into concurrent byte-range requests
//! and caching them in a strictly bounded LRU page cache (`<= memory_budget_bytes`).

mod builder;
mod cache;
mod fetch;
mod format;
mod node;
mod client;

use std::sync::Arc;

use pink_lemonade_progressive_core::{
    ByteRangeStore, ClientConfig, ClientError, DatasetBuilder, ProgressiveClient,
    ProgressiveStrategy, async_trait,
};

pub use builder::PagedTreeBuilder;
pub use client::PagedTreeClient;

/// Default raw points per leaf block (32 points = 768 B per leaf).
pub const DEFAULT_LEAF_POINTS: u32 = 32;
/// Default branching factor per internal tree page (32 nodes = 2 KiB per page).
pub const DEFAULT_FANOUT: u32 = 32;

/// Resolution-driven paged moment tree strategy.
#[derive(Debug, Clone, Copy)]
pub struct PagedTreeStrategy {
    pub leaf_points: u32,
    pub fanout: u32,
}

impl Default for PagedTreeStrategy {
    fn default() -> Self {
        Self {
            leaf_points: DEFAULT_LEAF_POINTS,
            fanout: DEFAULT_FANOUT,
        }
    }
}

impl PagedTreeStrategy {
    pub const fn new(leaf_points: u32, fanout: u32) -> Self {
        Self {
            leaf_points,
            fanout,
        }
    }
}

#[async_trait]
impl ProgressiveStrategy for PagedTreeStrategy {
    fn name(&self) -> &str {
        "paged_moment_tree"
    }

    fn create_builder(&self) -> Box<dyn DatasetBuilder> {
        Box::new(PagedTreeBuilder::new(self.leaf_points, self.fanout))
    }

    async fn open_client(
        &self,
        store: Arc<dyn ByteRangeStore>,
        config: ClientConfig,
    ) -> Result<Box<dyn ProgressiveClient>, ClientError> {
        let client = PagedTreeClient::open(store, config).await?;
        Ok(Box::new(client))
    }
}
