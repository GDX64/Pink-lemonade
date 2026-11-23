//! Classic moment quadtree strategy for `pink-lemonade-progressive-core`.
//!
//! A point-region quadtree over the data bounding box: each node splits its region at the
//! midpoint into 4 equal quadrants (squares in normalized data coordinates), empty quadrants
//! are dropped, and nodes with at most `leaf_points` points become leaves. Every node stores
//! exact moments (the shared 64-byte [`PackedNode`](pink_lemonade_progressive_common::PackedNode)),
//! so the client resolves a node as one kernel once it lies fully inside the x-viewport and
//! fits the merge threshold on screen, descending otherwise (straddling nodes always descend
//! to raw leaves for exact clipping).
//!
//! Layout: the top depths (≤ `root_cap` nodes) are inlined in `manifest.bin`; deeper depths
//! are packed into `nodes_{g}.bin` objects of at least `min_object_nodes` records each, so
//! deep trees never turn into thousands of tiny objects; raw points live in one `points.bin`.
//!
//! Known tradeoffs this baseline is meant to measure:
//! - one round trip per depth, and thin curves (random walks) make the tree deep;
//! - square cells overshoot in y when a dense dataset is zoomed in x only.

use std::sync::Arc;

use pink_lemonade_progressive_common::PointCollector;
use pink_lemonade_progressive_core::{
    ByteRangeStore, ClientConfig, ClientError, DatasetBuilder, ProgressiveClient,
    ProgressiveStrategy, async_trait,
};

mod builder;
mod client;
pub mod format;

pub use builder::QuadtreeBuilder;
pub use client::QuadtreeClient;
pub use format::MAGIC;

pub const DEFAULT_LEAF_POINTS: u32 = 32;
/// Inline the top depths in `manifest.bin` while they total at most this many nodes.
pub const DEFAULT_ROOT_CAP: u32 = 1024;
/// Pack consecutive depths into one object until it holds at least this many nodes.
pub const DEFAULT_MIN_OBJECT_NODES: u32 = 64 * 1024;
/// Guards against unbounded splitting of near-coincident points.
pub const DEFAULT_MAX_DEPTH: u32 = 40;

#[derive(Debug, Clone, Copy)]
pub struct QuadtreeStrategy {
    pub leaf_points: u32,
    pub root_cap: u32,
    pub min_object_nodes: u32,
    /// Capped at 62 (the client's frontier ordering key holds 63 depths).
    pub max_depth: u32,
}

impl Default for QuadtreeStrategy {
    fn default() -> Self {
        Self {
            leaf_points: DEFAULT_LEAF_POINTS,
            root_cap: DEFAULT_ROOT_CAP,
            min_object_nodes: DEFAULT_MIN_OBJECT_NODES,
            max_depth: DEFAULT_MAX_DEPTH,
        }
    }
}

#[async_trait]
impl ProgressiveStrategy for QuadtreeStrategy {
    fn name(&self) -> &str {
        "moment_quadtree"
    }

    fn create_builder(&self) -> Box<dyn DatasetBuilder> {
        let mut config = *self;
        config.leaf_points = config.leaf_points.max(1);
        config.max_depth = config.max_depth.min(client::Z_KEY_DEPTHS - 1);
        Box::new(QuadtreeBuilder {
            config,
            points: PointCollector::default(),
        })
    }

    async fn open_client(
        &self,
        store: Arc<dyn ByteRangeStore>,
        config: ClientConfig,
    ) -> Result<Box<dyn ProgressiveClient>, ClientError> {
        Ok(Box::new(QuadtreeClient::open(store, config.memory_budget_bytes).await?))
    }
}
