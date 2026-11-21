//! Paged multi-level moment tree strategy for `pink-lemonade-progressive-core`.
//!
//! Organizes x-sorted points into fixed-size leaf blocks (`points.bin`) topped by a
//! high-fanout hierarchy of 64-byte moment + extrema summary nodes (`level_*.bin`), with
//! the top root level inlined in `manifest.bin` (`PLPTREE2`).
//!
//! Queries descend only into nodes overlapping `[min_x, max_x]` whose screen-space span
//! exceeds `merge_threshold`, coalescing missing pages into concurrent byte-range requests
//! and caching them in a strictly bounded LRU page cache (`<= memory_budget_bytes`).
//!
//! The format, reader, and frontier traversal live in `pink-lemonade-progressive-common` and
//! are shared with the 2D moment k-d tree; this crate only fixes the x-sorted point order.

use std::sync::Arc;

use pink_lemonade_progressive_common::{HierarchyReader, PointCollector, write_hierarchy};
use pink_lemonade_progressive_core::{
    BuildError, ByteRangeStore, ClientConfig, ClientError, DatasetBuilder, Frame, ObjectWriter,
    Point, ProgressiveClient, ProgressiveStrategy, ViewQuery, async_trait,
};

/// Default raw points per leaf block (32 points = 768 B per leaf).
pub const DEFAULT_LEAF_POINTS: u32 = 32;
/// Default branching factor per internal tree page (32 nodes = 2 KiB per page).
pub const DEFAULT_FANOUT: u32 = 32;

/// Manifest magic for the x-ordered moment tree (format unchanged since v2).
pub const MAGIC: [u8; 8] = *b"PLPTREE2";

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

/// Offline builder: x-sorted points, rolled up until at most `fanout` root nodes remain.
pub struct PagedTreeBuilder {
    leaf_points: u32,
    fanout: u32,
    points: PointCollector,
}

impl PagedTreeBuilder {
    pub fn new(leaf_points: u32, fanout: u32) -> Self {
        Self {
            leaf_points: leaf_points.max(1),
            fanout: fanout.max(2),
            points: PointCollector::default(),
        }
    }
}

impl DatasetBuilder for PagedTreeBuilder {
    fn push_chunk(&mut self, points: &[Point]) -> Result<(), BuildError> {
        self.points.push_chunk(points)
    }

    fn finish(self: Box<Self>, writer: &mut dyn ObjectWriter) -> Result<(), BuildError> {
        write_hierarchy(
            &self.points.points,
            MAGIC,
            self.leaf_points,
            self.fanout,
            self.fanout,
            writer,
        )?;
        Ok(())
    }
}

/// Online client: shared hierarchy reader with x-ordered (already sorted) line output.
pub struct PagedTreeClient {
    reader: HierarchyReader,
}

impl PagedTreeClient {
    pub async fn open(
        store: Arc<dyn ByteRangeStore>,
        config: ClientConfig,
    ) -> Result<Self, ClientError> {
        Ok(Self {
            reader: HierarchyReader::open(store, MAGIC, config.memory_budget_bytes).await?,
        })
    }
}

#[async_trait]
impl ProgressiveClient for PagedTreeClient {
    async fn query(&mut self, query: ViewQuery) -> Result<Frame, ClientError> {
        self.reader.standard_query(query, false).await
    }

    fn memory_bytes(&self) -> usize {
        self.reader.memory_bytes()
    }
}
