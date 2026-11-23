//! Online client: depth-by-depth descent over the quadtree with byte-range reads.
//!
//! Same exactness rules as every other strategy: a node is summarized only when it lies
//! fully inside `[min_x, max_x]`; straddling nodes descend down to raw leaves, which are
//! clipped by x. Each depth below the inlined top costs one round trip.

use std::collections::HashMap;
use std::sync::Arc;

use pink_lemonade_progressive_common::{
    FrontierItem, LruPageCache, Node, POINT_BYTES, assemble_frame, coalesce_spans, decode_points,
    read_whole_object, standard_decision, try_join_all,
};
use pink_lemonade_progressive_core::{
    ByteRangeStore, ClientError, Frame, Point, ProgressiveClient, ViewQuery, async_trait,
};

use crate::format::{
    Link, MANIFEST_KEY, Manifest, NODES_KEY, POINTS_KEY, QuadRecord, RECORD_BYTES, decode_records,
};

/// Bridge gaps of up to 16 KiB between child pages; cap merged reads at 256 KiB.
const MAX_GAP_RECORD_BYTES: u64 = 16 * 1024;
const MAX_RUN_RECORD_BYTES: u64 = 256 * 1024;
/// Bridge gaps of up to 6 KiB (256 points) between raw leaves; cap merged reads at 192 KiB.
const MAX_GAP_POINTS: u64 = 256;
const MAX_RUN_POINTS: u64 = 8 * 1024;
/// Depth capacity of the `u128` frontier ordering key (2 bits per depth); the builder caps
/// `max_depth` below it.
pub(crate) const Z_KEY_DEPTHS: u32 = 63;

/// Cache space for child pages is the depth; leaves use the point-start id.
pub struct QuadtreeClient {
    store: Arc<dyn ByteRangeStore>,
    manifest: Manifest,
    cache: LruPageCache,
}

/// A node reached during descent.
#[derive(Clone, Copy)]
struct Reached {
    node: Node,
    link: Link,
}

impl Reached {
    fn new(record: &QuadRecord) -> Self {
        Self {
            node: record.node.unpack(),
            link: record.link(),
        }
    }
}

impl QuadtreeClient {
    pub async fn open(store: Arc<dyn ByteRangeStore>, memory_budget_bytes: usize) -> Result<Self, ClientError> {
        let raw = read_whole_object(store.as_ref(), MANIFEST_KEY).await?;
        let manifest = Manifest::decode(&raw)?;
        let cache = LruPageCache::new(memory_budget_bytes.saturating_sub(manifest.approx_memory_bytes()));
        Ok(Self { store, manifest, cache })
    }

    /// Children pages `(first, count)` at `depth`, keyed by `first` (inline, cached, or fetched
    /// with gap-bridging coalesced range reads).
    async fn fetch_children(
        &mut self,
        depth: usize,
        pages: &[(u64, u64)],
    ) -> Result<HashMap<u64, Vec<QuadRecord>>, ClientError> {
        let mut out = HashMap::with_capacity(pages.len());
        if let Some(level) = self.manifest.inline.get(depth) {
            for &(first, count) in pages {
                out.insert(first, level[first as usize..(first + count) as usize].to_vec());
            }
            return Ok(out);
        }
        let info = *self
            .manifest
            .depths
            .get(depth)
            .ok_or_else(|| ClientError::CorruptData(format!("child depth {depth} out of range")))?;

        let mut missing = Vec::new();
        for &(first, count) in pages {
            if out.contains_key(&first) {
                continue;
            }
            match self.cache.get_blob(depth as u16, first) {
                Some(bytes) => {
                    out.insert(first, decode_records(&bytes).map_err(ClientError::CorruptData)?);
                }
                None => missing.push((first, count)),
            }
        }
        if missing.is_empty() {
            return Ok(out);
        }
        if missing.iter().any(|&(first, count)| first + count > info.count) {
            return Err(ClientError::CorruptData(format!("child page beyond depth {depth}")));
        }

        let reads = coalesce_spans(
            missing.clone(),
            MAX_GAP_RECORD_BYTES / RECORD_BYTES as u64,
            MAX_RUN_RECORD_BYTES / RECORD_BYTES as u64,
        );
        let futures: Vec<_> = reads
            .iter()
            .map(|&(start, end)| {
                let store = Arc::clone(&self.store);
                let offset = info.offset + start * RECORD_BYTES as u64;
                let len = (end - start) as usize * RECORD_BYTES;
                async move {
                    let bytes = store.read_range(NODES_KEY, offset, len).await?;
                    Ok::<_, ClientError>((start, bytes))
                }
            })
            .collect();
        let fetched = try_join_all(futures).await?;

        for (first, count) in missing {
            let (start, bytes) = fetched
                .iter()
                .find(|(start, bytes)| first >= *start && first + count <= start + (bytes.len() / RECORD_BYTES) as u64)
                .ok_or_else(|| ClientError::CorruptData("short child page read".to_owned()))?;
            let lo = (first - start) as usize * RECORD_BYTES;
            let page: Arc<[u8]> = Arc::from(&bytes[lo..lo + count as usize * RECORD_BYTES]);
            out.insert(first, decode_records(&page).map_err(ClientError::CorruptData)?);
            self.cache.insert_blob(depth as u16, first, page);
        }
        Ok(out)
    }

    /// Raw leaves `(start, count)` keyed by point start.
    async fn fetch_leaves(&mut self, leaves: &[(u64, u64)]) -> Result<HashMap<u64, Arc<[Point]>>, ClientError> {
        let mut out = HashMap::with_capacity(leaves.len());
        let mut missing = Vec::new();
        for &(start, count) in leaves {
            if out.contains_key(&start) {
                continue;
            }
            match self.cache.get_leaf_points(start) {
                Some(points) => {
                    out.insert(start, points);
                }
                None => missing.push((start, count)),
            }
        }
        if missing.is_empty() {
            return Ok(out);
        }
        missing.sort_unstable();
        missing.dedup();
        if missing.iter().any(|&(s, c)| s + c > self.manifest.total_points) {
            return Err(ClientError::CorruptData("leaf beyond points.bin".to_owned()));
        }

        let reads = coalesce_spans(missing.clone(), MAX_GAP_POINTS, MAX_RUN_POINTS);
        let futures: Vec<_> = reads
            .iter()
            .map(|&(start, end)| {
                let store = Arc::clone(&self.store);
                async move {
                    let bytes = store
                        .read_range(POINTS_KEY, start * POINT_BYTES as u64, (end - start) as usize * POINT_BYTES)
                        .await?;
                    Ok::<_, ClientError>((start, bytes))
                }
            })
            .collect();
        let fetched = try_join_all(futures).await?;

        for (start, count) in missing {
            let (read_start, bytes) = fetched
                .iter()
                .find(|(s, bytes)| start >= *s && start + count <= s + (bytes.len() / POINT_BYTES) as u64)
                .ok_or_else(|| ClientError::CorruptData("short leaf read".to_owned()))?;
            let lo = (start - read_start) as usize * POINT_BYTES;
            let points: Arc<[Point]> = decode_points(&bytes[lo..lo + count as usize * POINT_BYTES])
                .map_err(ClientError::CorruptData)?
                .into();
            self.cache.insert_leaf_points(start, Arc::clone(&points));
            out.insert(start, points);
        }
        Ok(out)
    }

    /// Expand `parents` (all at `depth`) into their children at `depth + 1`, in order.
    async fn expand(&mut self, depth: usize, parents: &[Reached]) -> Result<Vec<Reached>, ClientError> {
        let pages: Vec<(u64, u64)> = parents
            .iter()
            .filter_map(|r| match r.link {
                Link::Children { first, count } => Some((first, count)),
                Link::Points { .. } => None,
            })
            .collect();
        if pages.is_empty() {
            return Ok(Vec::new());
        }
        let fetched = self.fetch_children(depth + 1, &pages).await?;
        let mut children = Vec::with_capacity(pages.len() * 4);
        for (first, _) in pages {
            let page = fetched
                .get(&first)
                .ok_or_else(|| ClientError::CorruptData("missing child page".to_owned()))?;
            children.extend(page.iter().map(Reached::new));
        }
        Ok(children)
    }

    /// Exact visible `[min_y, max_y]`: fold fully-inside nodes, and descend straddling nodes
    /// only while their y-extent reaches beyond the range known so far.
    async fn visible_y_range(&mut self, min_x: f64, max_x: f64) -> Result<Option<[f64; 2]>, ClientError> {
        let Some(root) = self.manifest.root() else {
            return Ok(None);
        };
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        let mut level = vec![Reached::new(&root)];
        let mut straddling_leaves: Vec<Reached> = Vec::new();
        let mut depth = 0;
        while !level.is_empty() {
            let mut straddling = Vec::new();
            for r in level {
                let n = r.node;
                if n.is_empty() || n.max_x < min_x || n.min_x > max_x {
                    continue;
                }
                if min_x <= n.min_x && n.max_x <= max_x {
                    lo = lo.min(n.min_y);
                    hi = hi.max(n.max_y);
                } else {
                    straddling.push(r);
                }
            }
            straddling.retain(|r| r.node.min_y < lo || r.node.max_y > hi);
            let (leaves, inner): (Vec<Reached>, Vec<Reached>) = straddling
                .into_iter()
                .partition(|r| matches!(r.link, Link::Points { .. }));
            straddling_leaves.extend(leaves);
            level = self.expand(depth, &inner).await?;
            depth += 1;
        }

        straddling_leaves.retain(|r| r.node.min_y < lo || r.node.max_y > hi);
        let spans = leaf_spans(&straddling_leaves);
        if !spans.is_empty() {
            let leaves = self.fetch_leaves(&spans).await?;
            for (start, _) in spans {
                for p in leaves[&start].iter().filter(|p| p.x >= min_x && p.x <= max_x) {
                    lo = lo.min(p.y);
                    hi = hi.max(p.y);
                }
            }
        }
        Ok(lo.is_finite().then_some([lo, hi]))
    }
}

fn leaf_spans(reached: &[Reached]) -> Vec<(u64, u64)> {
    reached
        .iter()
        .filter_map(|r| match r.link {
            Link::Points { start, count } => Some((start, count)),
            Link::Children { .. } => None,
        })
        .collect()
}

#[async_trait]
impl ProgressiveClient for QuadtreeClient {
    async fn query(&mut self, query: ViewQuery) -> Result<Frame, ClientError> {
        let Some(root) = self.manifest.root() else {
            return Ok(Frame::empty());
        };
        if query.min_x > query.max_x {
            return Ok(Frame::empty());
        }
        let Some(y_range) = self.visible_y_range(query.min_x, query.max_x).await? else {
            return Ok(Frame::empty());
        };
        let (x_factor, y_factor) = query.factors(query.effective_y_bounds(y_range));

        // Frontier in quadtree order: each item carries a Z-key (2 bits per depth, left-aligned)
        // so expanded children keep their parent's slot; greedy kernel merging works best on
        // spatially adjacent items.
        let mut done: Vec<(u128, FrontierItem)> = Vec::new();
        let mut leaf_items: Vec<(u64, u64)> = Vec::new();
        let mut level: Vec<(u128, Reached)> = vec![(0, Reached::new(&root))];
        let mut depth = 0;
        while !level.is_empty() {
            let mut expand: Vec<(u128, Reached)> = Vec::new();
            for (pos, r) in level {
                let n = r.node;
                if n.is_empty() || n.max_x < query.min_x || n.min_x > query.max_x {
                    continue;
                }
                let inside = query.min_x <= n.min_x && n.max_x <= query.max_x;
                let resolve = inside
                    && matches!(
                        standard_decision(&n, &query, x_factor, y_factor),
                        pink_lemonade_progressive_common::Decision::Resolve
                    );
                match (resolve, r.link) {
                    (true, _) => done.push((pos, FrontierItem::Resolved(n))),
                    (false, Link::Points { start, count }) => {
                        leaf_items.push((start, count));
                        done.push((pos, FrontierItem::Leaf(start)));
                    }
                    (false, Link::Children { .. }) => expand.push((pos, r)),
                }
            }
            let parents: Vec<Reached> = expand.iter().map(|(_, r)| *r).collect();
            let children = self.expand(depth, &parents).await?;
            let mut next = Vec::with_capacity(children.len());
            let mut it = children.into_iter();
            let shift = 2 * (Z_KEY_DEPTHS - (depth as u32 + 1));
            for (pos, r) in expand {
                let Link::Children { count, .. } = r.link else { unreachable!() };
                for i in 0..count as u128 {
                    next.push((pos | (i << shift), it.next().expect("child count mismatch")));
                }
            }
            level = next;
            depth += 1;
        }
        done.sort_by_key(|&(key, _)| key);

        let leaves = if leaf_items.is_empty() {
            HashMap::new()
        } else {
            self.fetch_leaves(&leaf_items).await?
        };
        Ok(assemble_frame(
            done.into_iter().map(|(_, item)| item).collect(),
            &leaves,
            &query,
            (x_factor, y_factor),
            y_range,
            true,
            &mut |_, _| unreachable!("the quadtree never emits plane items"),
        ))
    }

    fn memory_bytes(&self) -> usize {
        self.manifest.approx_memory_bytes() + self.cache.used_bytes()
    }
}
