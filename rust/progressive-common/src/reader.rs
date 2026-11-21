//! Progressive, memory-bounded reader for the paged multi-level moment hierarchy.
//!
//! Provides the two resolution-aware phases shared by hierarchy-based strategies:
//! 1. **Pruned exact `visible_y_range`**: folds interior nodes' `[min_y, max_y]` first and
//!    descends straddling boundary nodes *only* when their vertical bounds extend beyond the
//!    already-known interior range.
//! 2. **Level-by-level frontier traversal**: descends only into nodes overlapping
//!    `[min_x, max_x]` that straddle the viewport boundary or that a caller-provided
//!    [`Decision`] policy chooses to expand, coalescing missing pages across small gaps into
//!    concurrent byte-range requests and caching compact 64-byte pages in a strictly bounded
//!    LRU cache.
//!
//! Correctness never depends on the point order: a node is only resolved (summarized) when it
//! lies fully inside `[min_x, max_x]`, so moments are exact for x-ordered and 2D orders alike.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use pink_lemonade_progressive_core::{
    ByteRangeStore, ClientError, Frame, Point, ViewQuery,
};

use crate::cache::LruPageCache;
use crate::fetch::{coalesce_runs, try_join_all};
use crate::format::{MANIFEST_KEY, Manifest, POINTS_KEY, level_key};
use crate::node::{NODE_BYTES, Node, POINT_BYTES, PackedNode, decode_packed_nodes, decode_points};
use crate::render::{assemble_frame, standard_decision};

/// Bridge small gaps of up to 16 KiB of tree pages to avoid extra RTTs; cap runs at 256 KiB.
/// Byte-based so every fanout gets the same request sizes (at fanout=32 a 2 KiB page gives
/// the original 8-page gap / 128-page run limits).
const MAX_GAP_TREE_BYTES: usize = 16 * 1024;
const MAX_COALESCE_TREE_BYTES: usize = 256 * 1024;

/// At leaf_points=32 and 24 B/point, 1 leaf = 768 B.
/// Bridge small gaps up to 8 leaves (6 KiB) to collapse scattered raw-leaf splits without over-fetching.
const MAX_GAP_LEAVES: u64 = 8;
const MAX_COALESCE_LEAVES: usize = 256;

/// What to do with a node that lies fully inside the queried x-range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// Emit the node itself as one (mergeable) Gaussian kernel.
    Resolve,
    /// Emit the node through an auxiliary moment plane at y-depth `k` (strategy-specific).
    Plane(u8),
    /// Descend into the node's children (or its raw points at level 0).
    Expand,
}

/// Fully expanded traversal output, in hierarchy order.
#[derive(Debug, Clone, Copy)]
pub enum FrontierItem {
    Resolved(Node),
    Plane {
        level: usize,
        index: u64,
        node: Node,
        k: u8,
    },
    /// Raw points of leaf `i` must be fetched and clipped to `[min_x, max_x]`.
    Leaf(u64),
}

pub struct HierarchyReader {
    store: Arc<dyn ByteRangeStore>,
    manifest: Manifest,
    pub cache: LruPageCache,
}

impl HierarchyReader {
    /// Fetch and decode `manifest.bin`, sizing the cache to the budget left after the manifest.
    pub async fn open(
        store: Arc<dyn ByteRangeStore>,
        magic: [u8; 8],
        memory_budget_bytes: usize,
    ) -> Result<Self, ClientError> {
        let raw = read_whole_object(store.as_ref(), MANIFEST_KEY).await?;
        let manifest = Manifest::decode(&raw, magic)?;
        let cache_budget = memory_budget_bytes.saturating_sub(manifest.approx_memory_bytes());
        Ok(Self {
            store,
            manifest,
            cache: LruPageCache::new(cache_budget),
        })
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    pub fn store(&self) -> &Arc<dyn ByteRangeStore> {
        &self.store
    }

    /// Manifest + cached bytes (excluding any caller-owned state).
    pub fn memory_bytes(&self) -> usize {
        self.manifest.approx_memory_bytes() + self.cache.used_bytes()
    }

    /// Fetch missing pages at `level` via gap-bridging coalesced concurrent range reads.
    pub async fn fetch_tree_pages(
        &mut self,
        level: usize,
        page_indices: &[u64],
    ) -> Result<HashMap<u64, Arc<[PackedNode]>>, ClientError> {
        let mut resolved = HashMap::with_capacity(page_indices.len());
        let mut missing = Vec::new();

        for &page in page_indices {
            if resolved.contains_key(&page) {
                continue;
            }
            if let Some(cached) = self.cache.get_tree_page(level, page) {
                resolved.insert(page, cached);
            } else {
                missing.push(page);
            }
        }

        if missing.is_empty() {
            return Ok(resolved);
        }

        let fanout = self.manifest.fanout as u64;
        let page_bytes = fanout as usize * NODE_BYTES;
        let runs = coalesce_runs(
            missing,
            (MAX_GAP_TREE_BYTES / page_bytes) as u64,
            (MAX_COALESCE_TREE_BYTES / page_bytes).max(1),
        );
        let total_nodes = self.manifest.level_counts[level];
        let key = level_key(level);

        let futures: Vec<_> = runs
            .iter()
            .map(|&(start_page, page_count)| {
                let store = Arc::clone(&self.store);
                let key = key.clone();
                let start_node = start_page * fanout;
                let end_node = ((start_page + page_count as u64) * fanout).min(total_nodes);
                let byte_offset = start_node * NODE_BYTES as u64;
                let byte_len = (end_node - start_node) as usize * NODE_BYTES;
                async move {
                    let bytes = store.read_range(&key, byte_offset, byte_len).await?;
                    Ok::<_, ClientError>((start_page, page_count, bytes))
                }
            })
            .collect();

        let fetched = try_join_all(futures).await?;
        let fanout_usize = self.manifest.fanout as usize;

        for (start_page, page_count, bytes) in fetched {
            let nodes = decode_packed_nodes(&bytes).map_err(ClientError::CorruptData)?;
            for i in 0..page_count {
                let page = start_page + i as u64;
                let lo = i * fanout_usize;
                let hi = ((i + 1) * fanout_usize).min(nodes.len());
                if lo >= hi {
                    break;
                }
                let slice: Arc<[PackedNode]> = Arc::from(&nodes[lo..hi]);
                self.cache.insert_tree_page(level, page, Arc::clone(&slice));
                resolved.insert(page, slice);
            }
        }

        Ok(resolved)
    }

    /// Split `leaf_indices` into cache hits and (deduplicated) misses.
    pub fn cached_leaves(&mut self, leaf_indices: &[u64]) -> (HashMap<u64, Arc<[Point]>>, Vec<u64>) {
        let mut resolved = HashMap::with_capacity(leaf_indices.len());
        let mut missing = Vec::new();
        let mut seen = HashSet::with_capacity(leaf_indices.len());
        for &leaf in leaf_indices {
            if !seen.insert(leaf) {
                continue;
            }
            if let Some(cached) = self.cache.get_leaf_points(leaf) {
                resolved.insert(leaf, cached);
            } else {
                missing.push(leaf);
            }
        }
        (resolved, missing)
    }

    /// Download `missing` leaves from `points.bin` via gap-bridging coalesced range reads,
    /// returning only the requested leaves (bridged gap leaves are discarded).
    ///
    /// Borrows `self` immutably so callers can overlap it with other downloads; pass the result
    /// to [`Self::cache_leaves`] afterwards.
    pub async fn download_leaves(
        &self,
        missing: Vec<u64>,
    ) -> Result<Vec<(u64, Arc<[Point]>)>, ClientError> {
        if missing.is_empty() {
            return Ok(Vec::new());
        }
        let requested: HashSet<u64> = missing.iter().copied().collect();
        let runs = coalesce_runs(missing, MAX_GAP_LEAVES, MAX_COALESCE_LEAVES);
        let leaf_size = self.manifest.leaf_points as u64;
        let total_points = self.manifest.total_points;

        let futures: Vec<_> = runs
            .iter()
            .map(|&(start_leaf, leaf_count)| {
                let store = Arc::clone(&self.store);
                let start_pt = start_leaf * leaf_size;
                let end_pt = ((start_leaf + leaf_count as u64) * leaf_size).min(total_points);
                let byte_offset = start_pt * POINT_BYTES as u64;
                let byte_len = (end_pt - start_pt) as usize * POINT_BYTES;
                async move {
                    let bytes = store.read_range(POINTS_KEY, byte_offset, byte_len).await?;
                    Ok::<_, ClientError>((start_leaf, leaf_count, bytes))
                }
            })
            .collect();

        let fetched = try_join_all(futures).await?;
        let leaf_size_usize = self.manifest.leaf_points as usize;
        let mut out = Vec::with_capacity(requested.len());

        for (start_leaf, leaf_count, bytes) in fetched {
            let points = decode_points(&bytes).map_err(ClientError::CorruptData)?;
            for i in 0..leaf_count {
                let leaf = start_leaf + i as u64;
                if !requested.contains(&leaf) {
                    continue;
                }
                let lo = i * leaf_size_usize;
                let hi = ((i + 1) * leaf_size_usize).min(points.len());
                if lo >= hi {
                    break;
                }
                out.push((leaf, Arc::from(&points[lo..hi])));
            }
        }
        Ok(out)
    }

    /// Insert downloaded leaves into the LRU cache and merge them into `resolved`.
    pub fn cache_leaves(
        &mut self,
        downloaded: Vec<(u64, Arc<[Point]>)>,
        resolved: &mut HashMap<u64, Arc<[Point]>>,
    ) {
        for (leaf, points) in downloaded {
            self.cache.insert_leaf_points(leaf, Arc::clone(&points));
            resolved.insert(leaf, points);
        }
    }

    /// Fetch leaf point blocks (cache first, then coalesced range reads).
    pub async fn fetch_leaves(
        &mut self,
        leaf_indices: &[u64],
    ) -> Result<HashMap<u64, Arc<[Point]>>, ClientError> {
        let (mut resolved, missing) = self.cached_leaves(leaf_indices);
        let downloaded = self.download_leaves(missing).await?;
        self.cache_leaves(downloaded, &mut resolved);
        Ok(resolved)
    }

    /// Exact visible `[min_y, max_y]` in `[min_x, max_x]`.
    ///
    /// Folds fully-contained interior nodes first and skips descending any straddling boundary
    /// node whose `[min_y, max_y]` already sits inside the known interior range.
    pub async fn visible_y_range(
        &mut self,
        min_x: f64,
        max_x: f64,
    ) -> Result<Option<[f64; 2]>, ClientError> {
        let Some(root_level) = self.manifest.root_level() else {
            return Ok(None);
        };
        if min_x > max_x {
            return Ok(None);
        }

        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        // First fold all interior root nodes so `min_y`/`max_y` are as tight as possible
        // before deciding whether straddling boundary nodes even need descending.
        let mut candidates: Vec<(u64, f64, f64)> = Vec::new();
        for (idx, &node) in self.manifest.root_nodes.iter().enumerate() {
            if node.max_x < min_x || node.min_x > max_x {
                continue;
            }
            if min_x <= node.min_x && node.max_x <= max_x {
                min_y = min_y.min(node.min_y);
                max_y = max_y.max(node.max_y);
            } else {
                candidates.push((idx as u64, node.min_y, node.max_y));
            }
        }

        let mut straddling: Vec<u64> = candidates
            .into_iter()
            .filter(|&(_, c_min, c_max)| c_min < min_y || c_max > max_y)
            .map(|(idx, _, _)| idx)
            .collect();

        let fanout = self.manifest.fanout as u64;
        for child_level in (0..root_level).rev() {
            if straddling.is_empty() {
                break;
            }
            let pages = self.fetch_tree_pages(child_level, &straddling).await?;
            let mut next_candidates: Vec<(u64, f64, f64)> = Vec::new();

            for parent_idx in straddling {
                let Some(children) = pages.get(&parent_idx) else {
                    continue;
                };
                let base_child = parent_idx * fanout;
                for (offset, packed) in children.iter().enumerate() {
                    let node = packed.unpack();
                    if node.max_x < min_x || node.min_x > max_x {
                        continue;
                    }
                    if min_x <= node.min_x && node.max_x <= max_x {
                        min_y = min_y.min(node.min_y);
                        max_y = max_y.max(node.max_y);
                    } else {
                        next_candidates.push((base_child + offset as u64, node.min_y, node.max_y));
                    }
                }
            }

            straddling = next_candidates
                .into_iter()
                .filter(|&(_, c_min, c_max)| c_min < min_y || c_max > max_y)
                .map(|(idx, _, _)| idx)
                .collect();
        }

        if !straddling.is_empty() {
            let leaves = self.fetch_leaves(&straddling).await?;
            for leaf_idx in straddling {
                if let Some(points) = leaves.get(&leaf_idx) {
                    for p in points.iter() {
                        if p.x >= min_x && p.x <= max_x {
                            min_y = min_y.min(p.y);
                            max_y = max_y.max(p.y);
                        }
                    }
                }
            }
        }

        if min_y.is_finite() {
            Ok(Some([min_y, max_y]))
        } else {
            Ok(None)
        }
    }

    /// Level-by-level frontier traversal over nodes overlapping `[min_x, max_x]`.
    ///
    /// `decide(node, level)` is consulted only for nodes fully inside the x-range; straddling
    /// nodes always expand (down to raw leaves) so clipping stays exact. Raw leaves are *not*
    /// fetched here: callers fetch [`FrontierItem::Leaf`]s (possibly overlapped with other work).
    pub async fn frontier(
        &mut self,
        min_x: f64,
        max_x: f64,
        decide: &(dyn Fn(&Node, usize) -> Decision + Sync),
    ) -> Result<Vec<FrontierItem>, ClientError> {
        enum Item {
            Done(FrontierItem),
            ExpandNode(u64),
        }

        let Some(root_level) = self.manifest.root_level() else {
            return Ok(Vec::new());
        };
        let classify = |node: Node, level: usize, index: u64| -> Option<Item> {
            if node.max_x < min_x || node.min_x > max_x {
                return None;
            }
            let fully_inside = min_x <= node.min_x && node.max_x <= max_x;
            let decision = if fully_inside {
                decide(&node, level)
            } else {
                Decision::Expand
            };
            Some(match decision {
                Decision::Resolve => Item::Done(FrontierItem::Resolved(node)),
                Decision::Plane(k) => Item::Done(FrontierItem::Plane {
                    level,
                    index,
                    node,
                    k,
                }),
                Decision::Expand if level > 0 => Item::ExpandNode(index),
                Decision::Expand => Item::Done(FrontierItem::Leaf(index)),
            })
        };

        let mut frontier: Vec<Item> = self
            .manifest
            .root_nodes
            .iter()
            .enumerate()
            .filter_map(|(idx, &node)| classify(node, root_level, idx as u64))
            .collect();

        let fanout = self.manifest.fanout as u64;
        for child_level in (0..root_level).rev() {
            let needed_pages: Vec<u64> = frontier
                .iter()
                .filter_map(|item| match item {
                    Item::ExpandNode(idx) => Some(*idx),
                    Item::Done(_) => None,
                })
                .collect();
            if needed_pages.is_empty() {
                break;
            }

            let pages = self.fetch_tree_pages(child_level, &needed_pages).await?;
            let mut next = Vec::with_capacity(frontier.len() + needed_pages.len() * 4);
            for item in frontier {
                match item {
                    Item::Done(done) => next.push(Item::Done(done)),
                    Item::ExpandNode(node_idx) => {
                        let Some(children) = pages.get(&node_idx) else {
                            continue;
                        };
                        let base_child = node_idx * fanout;
                        next.extend(children.iter().enumerate().filter_map(|(offset, packed)| {
                            classify(packed.unpack(), child_level, base_child + offset as u64)
                        }));
                    }
                }
            }
            frontier = next;
        }

        Ok(frontier
            .into_iter()
            .map(|item| match item {
                Item::Done(done) => done,
                Item::ExpandNode(_) => unreachable!("all tree levels expanded"),
            })
            .collect())
    }

    /// Standard query: exact `y_range`, threshold-driven frontier (with off-screen y culling),
    /// raw boundary leaves, and greedy kernel merging.
    ///
    /// `sort_lines` must be set when the hierarchy order is not x-monotonic (e.g. k-d order).
    pub async fn standard_query(
        &mut self,
        query: ViewQuery,
        sort_lines: bool,
    ) -> Result<Frame, ClientError> {
        if self.manifest.root_level().is_none() || query.min_x > query.max_x {
            return Ok(Frame::empty());
        }
        let Some(y_range) = self.visible_y_range(query.min_x, query.max_x).await? else {
            return Ok(Frame::empty());
        };
        let (x_factor, y_factor) = query.factors(query.effective_y_bounds(y_range));
        let decide = |node: &Node, _level: usize| standard_decision(node, &query, x_factor, y_factor);
        let items = self.frontier(query.min_x, query.max_x, &decide).await?;

        let needed: Vec<u64> = items
            .iter()
            .filter_map(|item| match item {
                FrontierItem::Leaf(idx) => Some(*idx),
                _ => None,
            })
            .collect();
        let leaves = if needed.is_empty() {
            HashMap::new()
        } else {
            self.fetch_leaves(&needed).await?
        };

        Ok(assemble_frame(
            items,
            &leaves,
            &query,
            (x_factor, y_factor),
            y_range,
            sort_lines,
            &mut |_, _| unreachable!("standard_decision never emits plane items"),
        ))
    }
}

/// Read a complete small object (e.g. a manifest) in one range request.
pub async fn read_whole_object(store: &dyn ByteRangeStore, key: &str) -> Result<Vec<u8>, ClientError> {
    let len = store.object_len(key).await?;
    let len_usize = usize::try_from(len)
        .map_err(|_| ClientError::CorruptData(format!("{key} exceeds usize")))?;
    Ok(store.read_range(key, 0, len_usize).await?)
}
