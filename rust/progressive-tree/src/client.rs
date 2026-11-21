//! Progressive, memory-bounded client for the paged multi-level moment tree.
//!
//! Each query executes in two resolution-aware phases:
//! 1. **Pruned exact `visible_y_range`**: folds interior nodes' `[min_y, max_y]` first and
//!    descends straddling boundary nodes *only* when their vertical bounds extend beyond the
//!    already-known interior range.
//! 2. **Level-by-level frontier traversal**: descends only into nodes overlapping
//!    `[min_x, max_x]` that fail `fits_threshold` (or straddle viewport boundaries),
//!    coalescing missing pages across small gaps into concurrent byte-range requests and
//!    caching compact 64-byte pages in a strictly bounded LRU cache.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use pink_lemonade_progressive_core::{
    ByteRangeStore, ClientConfig, ClientError, Frame, Point, ProgressiveClient, ViewQuery,
    async_trait,
};

use crate::cache::LruPageCache;
use crate::fetch::{coalesce_runs, try_join_all};
use crate::format::{MANIFEST_KEY, Manifest, POINTS_KEY, level_key};
use crate::node::{NODE_BYTES, Node, POINT_BYTES, PackedNode, decode_packed_nodes, decode_points};

/// At fanout=32 and 64 B/node, 1 tree page = 2 KiB.
/// Bridge small gaps up to 8 pages (16 KiB) to avoid extra RTTs; cap runs at 128 pages (256 KiB).
const MAX_GAP_TREE_PAGES: u64 = 8;
const MAX_COALESCE_TREE_PAGES: usize = 128;

/// At leaf_points=32 and 24 B/point, 1 leaf = 768 B.
/// Bridge small gaps up to 8 leaves (6 KiB) to collapse scattered raw-leaf splits without over-fetching.
const MAX_GAP_LEAVES: u64 = 8;
const MAX_COALESCE_LEAVES: usize = 256;

pub struct PagedTreeClient {
    store: Arc<dyn ByteRangeStore>,
    manifest: Manifest,
    cache: LruPageCache,
}

impl PagedTreeClient {
    pub async fn open(
        store: Arc<dyn ByteRangeStore>,
        config: ClientConfig,
    ) -> Result<Self, ClientError> {
        let len = store.object_len(MANIFEST_KEY).await?;
        let len_usize = usize::try_from(len)
            .map_err(|_| ClientError::CorruptData("manifest.bin exceeds usize".to_owned()))?;
        let raw = store.read_range(MANIFEST_KEY, 0, len_usize).await?;
        let manifest = Manifest::decode(&raw)?;

        let cache_budget = config
            .memory_budget_bytes
            .saturating_sub(manifest.approx_memory_bytes());
        Ok(Self {
            store,
            manifest,
            cache: LruPageCache::new(cache_budget),
        })
    }

    /// Fetch missing pages at `level` via gap-bridging coalesced concurrent range reads.
    async fn fetch_tree_pages(
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

        let runs = coalesce_runs(missing, MAX_GAP_TREE_PAGES, MAX_COALESCE_TREE_PAGES);
        let fanout = self.manifest.fanout as u64;
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

    /// Fetch missing leaf point blocks from `points.bin` via gap-bridging coalesced range reads.
    async fn fetch_leaves(
        &mut self,
        leaf_indices: &[u64],
    ) -> Result<HashMap<u64, Arc<[Point]>>, ClientError> {
        let mut resolved = HashMap::with_capacity(leaf_indices.len());
        let mut missing = Vec::new();

        for &leaf in leaf_indices {
            if resolved.contains_key(&leaf) {
                continue;
            }
            if let Some(cached) = self.cache.get_leaf_points(leaf) {
                resolved.insert(leaf, cached);
            } else {
                missing.push(leaf);
            }
        }

        if missing.is_empty() {
            return Ok(resolved);
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
                let slice: Arc<[Point]> = Arc::from(&points[lo..hi]);
                self.cache.insert_leaf_points(leaf, Arc::clone(&slice));
                resolved.insert(leaf, slice);
            }
        }

        Ok(resolved)
    }

    /// Exact visible `[min_y, max_y]` in `[min_x, max_x]`.
    ///
    /// Folds fully-contained interior nodes first and skips descending any straddling boundary
    /// node whose `[min_y, max_y]` already sits inside the known interior range.
    async fn visible_y_range(
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
}

enum FrontierItem {
    Resolved(Node),
    ExpandNode { node_idx: u64 },
    ExpandLeaf { leaf_idx: u64 },
}

#[async_trait]
impl ProgressiveClient for PagedTreeClient {
    async fn query(&mut self, query: ViewQuery) -> Result<Frame, ClientError> {
        let Some(root_level) = self.manifest.root_level() else {
            return Ok(Frame::empty());
        };
        if query.min_x > query.max_x {
            return Ok(Frame::empty());
        }

        let Some(y_range) = self.visible_y_range(query.min_x, query.max_x).await? else {
            return Ok(Frame::empty());
        };

        let (x_factor, y_factor) = query.factors(query.effective_y_bounds(y_range));
        let threshold = query.merge_threshold;
        let fanout = self.manifest.fanout as u64;

        let mut frontier: Vec<FrontierItem> = Vec::with_capacity(self.manifest.root_nodes.len());
        for (idx, &node) in self.manifest.root_nodes.iter().enumerate() {
            if node.max_x < query.min_x || node.min_x > query.max_x {
                continue;
            }
            let fully_inside = query.min_x <= node.min_x && node.max_x <= query.max_x;
            if fully_inside && node.fits_threshold(x_factor, y_factor, threshold) {
                frontier.push(FrontierItem::Resolved(node));
            } else if root_level > 0 {
                frontier.push(FrontierItem::ExpandNode {
                    node_idx: idx as u64,
                });
            } else {
                frontier.push(FrontierItem::ExpandLeaf {
                    leaf_idx: idx as u64,
                });
            }
        }

        for child_level in (0..root_level).rev() {
            let needed_pages: Vec<u64> = frontier
                .iter()
                .filter_map(|item| match item {
                    FrontierItem::ExpandNode { node_idx } => Some(*node_idx),
                    _ => None,
                })
                .collect();
            if needed_pages.is_empty() {
                break;
            }

            let pages = self.fetch_tree_pages(child_level, &needed_pages).await?;
            let mut next_frontier = Vec::with_capacity(frontier.len() + needed_pages.len() * 4);

            for item in frontier {
                match item {
                    FrontierItem::Resolved(node) => next_frontier.push(FrontierItem::Resolved(node)),
                    FrontierItem::ExpandLeaf { leaf_idx } => {
                        next_frontier.push(FrontierItem::ExpandLeaf { leaf_idx })
                    }
                    FrontierItem::ExpandNode { node_idx } => {
                        let Some(children) = pages.get(&node_idx) else {
                            continue;
                        };
                        let base_child = node_idx * fanout;
                        for (offset, packed) in children.iter().enumerate() {
                            let child = packed.unpack();
                            if child.max_x < query.min_x || child.min_x > query.max_x {
                                continue;
                            }
                            let fully_inside =
                                query.min_x <= child.min_x && child.max_x <= query.max_x;
                            if fully_inside && child.fits_threshold(x_factor, y_factor, threshold) {
                                next_frontier.push(FrontierItem::Resolved(child));
                            } else if child_level > 0 {
                                next_frontier.push(FrontierItem::ExpandNode {
                                    node_idx: base_child + offset as u64,
                                });
                            } else {
                                next_frontier.push(FrontierItem::ExpandLeaf {
                                    leaf_idx: base_child + offset as u64,
                                });
                            }
                        }
                    }
                }
            }
            frontier = next_frontier;
        }

        let needed_leaves: Vec<u64> = frontier
            .iter()
            .filter_map(|item| match item {
                FrontierItem::ExpandLeaf { leaf_idx } => Some(*leaf_idx),
                _ => None,
            })
            .collect();
        let leaves = if needed_leaves.is_empty() {
            HashMap::new()
        } else {
            self.fetch_leaves(&needed_leaves).await?
        };

        let mut groups: Vec<Node> = Vec::with_capacity(frontier.len());
        for item in frontier {
            match item {
                FrontierItem::Resolved(node) => {
                    push_group(node, x_factor, y_factor, threshold, &mut groups);
                }
                FrontierItem::ExpandLeaf { leaf_idx } => {
                    if let Some(points) = leaves.get(&leaf_idx) {
                        let start = points.partition_point(|p| p.x < query.min_x);
                        let end = points.partition_point(|p| p.x <= query.max_x);
                        if start < end {
                            collect_points(
                                &points[start..end],
                                x_factor,
                                y_factor,
                                threshold,
                                &mut groups,
                            );
                        }
                    }
                }
                FrontierItem::ExpandNode { .. } => unreachable!("all tree levels expanded"),
            }
        }

        let mut gpu_instances = Vec::with_capacity(groups.len() * 7);
        let mut line_points = Vec::with_capacity(groups.len() * 8);
        for node in groups {
            node.emit_gpu(x_factor, y_factor, &mut gpu_instances);
            node.emit_line(&mut line_points);
        }

        Ok(Frame {
            gpu_instances,
            line_points,
            y_range: Some(y_range),
        })
    }

    fn memory_bytes(&self) -> usize {
        self.manifest.approx_memory_bytes() + self.cache.used_bytes()
    }
}

fn collect_points(
    points: &[Point],
    x_factor: f64,
    y_factor: f64,
    threshold: f64,
    out: &mut Vec<Node>,
) {
    if points.is_empty() {
        return;
    }
    let node = Node::from_points(points);
    if points.len() == 1 || node.fits_threshold(x_factor, y_factor, threshold) {
        push_group(node, x_factor, y_factor, threshold, out);
    } else {
        let mid = points.len() / 2;
        collect_points(&points[..mid], x_factor, y_factor, threshold, out);
        collect_points(&points[mid..], x_factor, y_factor, threshold, out);
    }
}

fn push_group(node: Node, x_factor: f64, y_factor: f64, threshold: f64, out: &mut Vec<Node>) {
    if node.is_empty() {
        return;
    }
    if let Some(last) = out.last_mut() {
        let combined = Node::combined(*last, node);
        if combined.fits_merged_group(x_factor, y_factor, threshold) {
            *last = combined;
            return;
        }
    }
    out.push(node);
}
