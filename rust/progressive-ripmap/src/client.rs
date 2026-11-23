//! Online `PLPRIPM2` ripmap client: single-RTT manifest open, `x_factor`-guided level jumping,
//! sparse 16-byte moment planes with inline per-node directories, cross-`k` cache reuse &
//! promotion, and greedy kernel merging.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use pink_lemonade_progressive_common::{
    DEFAULT_MANIFEST_READ_BYTES, LruPageCache, Node, POINT_BYTES, SPARSE_CELL_BYTES, SparseCell,
    coalesce_runs, coalesce_spans, decode_points, decode_sparse_cells, join2, offscreen_y,
    push_group, read_manifest, sort_line_points, try_join_all,
};
use pink_lemonade_progressive_core::{
    ByteRangeStore, ClientConfig, ClientError, Frame, Point, ProgressiveClient, ViewQuery,
    async_trait,
};

use crate::format::{
    LEVELS_KEY, MANIFEST_KEY, PLANES_KEY, POINTS_KEY, PlaneDirEntry, RipmapManifest, RipmapRecord,
    node_bin_geometry, record_bytes,
};

/// Resolve via planes once an x-block is at most this fraction of the merge threshold wide.
const MAX_PLANE_BLOCK_WIDTH: f64 = 0.9;
/// Bridge gaps of up to 16 KiB between tree pages; cap merged reads at 256 KiB.
const MAX_GAP_TREE_BYTES: usize = 16 * 1024;
const MAX_RUN_TREE_BYTES: usize = 256 * 1024;
/// Bridge gaps of up to 16 KiB between sparse plane blocks; cap merged reads at 1 MiB.
const MAX_GAP_PLANE_BYTES: u64 = 16 * 1024;
const MAX_RUN_PLANE_BYTES: u64 = 1024 * 1024;
/// Bridge gaps of up to 8 leaves (6 KiB) between raw leaves; cap merged reads at 256 leaves.
const MAX_GAP_LEAVES: u64 = 8;
const MAX_RUN_LEAVES: usize = 256;
/// Max extra bytes tolerated to promote a small coarse-k run into an adjacent finer plane slot
/// already being fetched at the same level, saving a network request.
const PROMOTE_RUN_EXTRA_BYTES: u64 = 4 * 1024;
/// Max hierarchy levels to skip in one jump over straddling boundary nodes (`4^3 = 64` nodes).
const MAX_BOUNDARY_JUMP: usize = 3;

/// Cache space ID for tree pages at `level` (`0..0x7FFF`).
const fn tree_space(level: usize) -> u16 {
    level as u16
}

/// Cache space ID for sparse plane blocks at `(level, slot)` (`0x8000..`).
const fn plane_space(level: usize, slot: usize) -> u16 {
    0x8000 | ((level as u16) << 5) | (slot as u16)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PlaneBlockKey {
    level: usize,
    slot: usize,
    index: u64,
}

#[derive(Debug, Clone)]
enum RipmapItem {
    Resolved(Node),
    Plane {
        level: usize,
        slot: usize,
        index: u64,
        node: Node,
        dir: PlaneDirEntry,
        /// All inline plane directory entries for this block (used for cross-`k` promotion).
        all_dirs: Vec<PlaneDirEntry>,
    },
    Leaf(u64),
}

enum RipmapDecision {
    Resolve,
    Plane(usize),
    Expand,
}

pub struct RipmapClient {
    store: Arc<dyn ByteRangeStore>,
    manifest: RipmapManifest,
    cache: LruPageCache,
}

impl RipmapClient {
    pub async fn open(
        store: Arc<dyn ByteRangeStore>,
        config: ClientConfig,
    ) -> Result<Self, ClientError> {
        let raw = read_manifest(
            store.as_ref(),
            MANIFEST_KEY,
            DEFAULT_MANIFEST_READ_BYTES,
            RipmapManifest::declared_len,
        )
        .await?;
        let manifest = RipmapManifest::decode(&raw)?;
        let cache_budget = config
            .memory_budget_bytes
            .saturating_sub(manifest.approx_memory_bytes());
        Ok(Self {
            store,
            manifest,
            cache: LruPageCache::new(cache_budget),
        })
    }

    /// Deepest stored y-depth at `level` (exposed for diagnostics).
    pub fn plane_depth(&self, level: usize) -> u8 {
        self.manifest
            .levels
            .get(level)
            .and_then(|l| l.planes.last())
            .map(|p| p.k)
            .unwrap_or(0)
    }

    /// Fetch missing tree pages at `level` from `levels.bin` via gap-bridging coalesced reads.
    async fn fetch_level_pages(
        &mut self,
        level: usize,
        page_indices: &[u64],
    ) -> Result<HashMap<u64, Vec<RipmapRecord>>, ClientError> {
        let info = &self.manifest.levels[level];
        let plane_count = info.planes.len();
        let stride = record_bytes(plane_count);
        let fanout = self.manifest.fanout as u64;
        let fanout_usize = self.manifest.fanout as usize;
        let space = tree_space(level);

        let mut resolved = HashMap::with_capacity(page_indices.len());
        let mut missing = Vec::new();
        for &page in page_indices {
            if resolved.contains_key(&page) {
                continue;
            }
            if let Some(bytes) = self.cache.get_blob(space, page) {
                let recs = RipmapRecord::decode_slice(&bytes, plane_count)
                    .map_err(ClientError::CorruptData)?;
                resolved.insert(page, recs);
            } else {
                missing.push(page);
            }
        }
        if missing.is_empty() {
            return Ok(resolved);
        }

        let page_bytes = fanout_usize * stride;
        let runs = coalesce_runs(
            missing,
            (MAX_GAP_TREE_BYTES / page_bytes) as u64,
            (MAX_RUN_TREE_BYTES / page_bytes).max(1),
        );
        let total_nodes = info.node_count;
        let base_off = info.level_byte_offset;

        let futures: Vec<_> = runs
            .iter()
            .map(|&(start_page, page_count)| {
                let store = Arc::clone(&self.store);
                let start_node = start_page * fanout;
                let end_node = ((start_page + page_count as u64) * fanout).min(total_nodes);
                let offset = base_off + start_node * stride as u64;
                let len = (end_node - start_node) as usize * stride;
                async move {
                    let bytes = store.read_range(LEVELS_KEY, offset, len).await?;
                    Ok::<_, ClientError>((start_page, page_count, bytes))
                }
            })
            .collect();

        let fetched = try_join_all(futures).await?;
        for (start_page, page_count, bytes) in fetched {
            for i in 0..page_count {
                let page = start_page + i as u64;
                let lo = i * page_bytes;
                let hi = ((i + 1) * page_bytes).min(bytes.len());
                if lo >= hi {
                    break;
                }
                let slice: Arc<[u8]> = Arc::from(&bytes[lo..hi]);
                let recs = RipmapRecord::decode_slice(&slice, plane_count)
                    .map_err(ClientError::CorruptData)?;
                self.cache.insert_blob(space, page, slice);
                resolved.insert(page, recs);
            }
        }
        Ok(resolved)
    }

    /// Fetch all descendants at `target_level` (`< parent_level`) for `parent_indices` in 1 RTT.
    async fn fetch_descendants(
        &mut self,
        parent_level: usize,
        target_level: usize,
        parent_indices: &[u64],
    ) -> Result<Vec<(u64, RipmapRecord)>, ClientError> {
        debug_assert!(target_level < parent_level);
        let fanout = self.manifest.fanout as u64;
        let step = (parent_level - target_level) as u32;
        let pages_per_parent = fanout.pow(step - 1);
        let total_nodes = self.manifest.levels[target_level].node_count;
        let total_pages = total_nodes.div_ceil(fanout);

        let mut needed_pages = Vec::with_capacity(parent_indices.len() * pages_per_parent as usize);
        for &p in parent_indices {
            let p_lo = p * pages_per_parent;
            let p_hi = ((p + 1) * pages_per_parent).min(total_pages);
            for page in p_lo..p_hi {
                needed_pages.push(page);
            }
        }
        let fetched = self.fetch_level_pages(target_level, &needed_pages).await?;

        let mut out = Vec::with_capacity(needed_pages.len() * fanout as usize);
        for page in needed_pages {
            let Some(recs) = fetched.get(&page) else {
                continue;
            };
            let base = page * fanout;
            for (offset, rec) in recs.iter().enumerate() {
                out.push((base + offset as u64, rec.clone()));
            }
        }
        Ok(out)
    }

    /// Fetch all levels `0..parent_level` for `parent_indices` concurrently in a single RTT,
    /// returning `levels[lvl]` as a map `node_index -> RipmapRecord` for `lvl in 0..parent_level`.
    async fn fetch_boundary_subtrees(
        &mut self,
        parent_level: usize,
        parent_indices: &[u64],
    ) -> Result<Vec<HashMap<u64, RipmapRecord>>, ClientError> {
        let fanout = self.manifest.fanout as u64;
        let fanout_usize = self.manifest.fanout as usize;
        let mut results: Vec<HashMap<u64, RipmapRecord>> =
            (0..parent_level).map(|_| HashMap::new()).collect();

        // Collect cache hits and plan coalesced reads across all levels `0..parent_level` at once.
        let mut futures = Vec::new();
        for lvl in 0..parent_level {
            let info = &self.manifest.levels[lvl];
            let plane_count = info.planes.len();
            let stride = record_bytes(plane_count);
            let page_bytes = fanout_usize * stride;
            let space = tree_space(lvl);
            let step = (parent_level - lvl) as u32;
            let pages_per_parent = fanout.pow(step - 1);
            let total_pages = info.node_count.div_ceil(fanout);

            let mut missing = Vec::new();
            for &p in parent_indices {
                let p_lo = p * pages_per_parent;
                let p_hi = ((p + 1) * pages_per_parent).min(total_pages);
                for page in p_lo..p_hi {
                    if let Some(bytes) = self.cache.get_blob(space, page) {
                        let recs = RipmapRecord::decode_slice(&bytes, plane_count)
                            .map_err(ClientError::CorruptData)?;
                        let base = page * fanout;
                        for (off, rec) in recs.into_iter().enumerate() {
                            results[lvl].insert(base + off as u64, rec);
                        }
                    } else {
                        missing.push(page);
                    }
                }
            }
            if missing.is_empty() {
                continue;
            }
            let runs = coalesce_runs(
                missing,
                (MAX_GAP_TREE_BYTES / page_bytes) as u64,
                (MAX_RUN_TREE_BYTES / page_bytes).max(1),
            );
            let base_off = info.level_byte_offset;
            let total_nodes = info.node_count;
            for (start_page, page_count) in runs {
                let store = Arc::clone(&self.store);
                let start_node = start_page * fanout;
                let end_node = ((start_page + page_count as u64) * fanout).min(total_nodes);
                let offset = base_off + start_node * stride as u64;
                let len = (end_node - start_node) as usize * stride;
                futures.push(async move {
                    let bytes = store.read_range(LEVELS_KEY, offset, len).await?;
                    Ok::<_, ClientError>((lvl, start_page, page_count, bytes))
                });
            }
        }

        for (lvl, start_page, page_count, bytes) in try_join_all(futures).await? {
            let plane_count = self.manifest.levels[lvl].planes.len();
            let stride = record_bytes(plane_count);
            let page_bytes = fanout_usize * stride;
            let space = tree_space(lvl);
            for i in 0..page_count {
                let page = start_page + i as u64;
                let lo = i * page_bytes;
                let hi = ((i + 1) * page_bytes).min(bytes.len());
                if lo >= hi {
                    break;
                }
                let slice: Arc<[u8]> = Arc::from(&bytes[lo..hi]);
                let recs = RipmapRecord::decode_slice(&slice, plane_count)
                    .map_err(ClientError::CorruptData)?;
                self.cache.insert_blob(space, page, slice);
                let base = page * fanout;
                for (off, rec) in recs.into_iter().enumerate() {
                    results[lvl].insert(base + off as u64, rec);
                }
            }
        }

        Ok(results)
    }

    /// Jump each overlapping root node down only as far as its *own* x-width requires
    /// (`w <= MAX_PLANE_BLOCK_WIDTH * threshold`), keeping narrow burst nodes (and off-screen
    /// y-nodes when `y_bounds` is set) at their higher level so one wide gap-straddling node
    /// cannot drag the whole view down to level 0.
    async fn jump_to_initial_frontier(
        &mut self,
        query: &ViewQuery,
        x_factor: f64,
        explicit_y_factor: Option<f64>,
        threshold: f64,
    ) -> Result<Vec<(usize, u64, RipmapRecord)>, ClientError> {
        let Some(root_level) = self.manifest.root_level() else {
            return Ok(Vec::new());
        };
        let min_x = query.min_x;
        let max_x = query.max_x;
        let num_levels = self.manifest.levels.len();
        let mut by_level: Vec<Vec<(u64, RipmapRecord)>> = vec![Vec::new(); num_levels];

        by_level[root_level] = self
            .manifest
            .root_records
            .iter()
            .enumerate()
            .filter_map(|(idx, r)| {
                if r.node.max_x < min_x || r.node.min_x > max_x {
                    None
                } else {
                    Some((idx as u64, r.clone()))
                }
            })
            .collect();
        if by_level[root_level].is_empty() {
            return Ok(Vec::new());
        }

        let target_w = MAX_PLANE_BLOCK_WIDTH * threshold;
        let fanout_f = self.manifest.fanout as f64;

        for cur_level in (1..=root_level).rev() {
            let nodes = std::mem::take(&mut by_level[cur_level]);
            if nodes.is_empty() {
                continue;
            }
            let mut ready = Vec::new();
            let mut by_drop: BTreeMap<usize, Vec<u64>> = BTreeMap::new();

            for (idx, rec) in nodes {
                let n = &rec.node;
                let fully_inside = min_x <= n.min_x && n.max_x <= max_x;
                let is_offscreen = fully_inside
                    && explicit_y_factor.is_some_and(|yf| offscreen_y(n, query, yf));
                let w = (n.max_x - n.min_x) * x_factor;
                if w <= target_w || is_offscreen {
                    ready.push((idx, rec));
                } else {
                    let ideal_drop = (w / target_w).log(fanout_f).ceil().max(1.0) as usize;
                    let drop = ideal_drop.min(MAX_BOUNDARY_JUMP).min(cur_level);
                    by_drop.entry(drop).or_default().push(idx);
                }
            }
            by_level[cur_level] = ready;

            for (drop, parents) in by_drop {
                let target_level = cur_level - drop;
                let descendants = self
                    .fetch_descendants(cur_level, target_level, &parents)
                    .await?;
                for (d_idx, d_rec) in descendants {
                    if !(d_rec.node.max_x < min_x || d_rec.node.min_x > max_x) {
                        by_level[target_level].push((d_idx, d_rec));
                    }
                }
            }
        }

        let fanout = self.manifest.fanout as u64;
        let mut out: Vec<(usize, u64, RipmapRecord)> = Vec::new();
        for (level, nodes) in by_level.into_iter().enumerate() {
            for (idx, rec) in nodes {
                out.push((level, idx, rec));
            }
        }
        // Exact x-monotonic order across mixed levels using each node's starting leaf index.
        out.sort_unstable_by_key(|&(lvl, idx, _)| idx * fanout.pow(lvl as u32));
        Ok(out)
    }

    fn cached_leaves(&mut self, leaf_indices: &[u64]) -> (HashMap<u64, Arc<[Point]>>, Vec<u64>) {
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

    async fn fetch_leaves(
        &mut self,
        leaf_indices: &[u64],
    ) -> Result<HashMap<u64, Arc<[Point]>>, ClientError> {
        let (mut resolved, missing) = self.cached_leaves(leaf_indices);
        let downloaded = download_leaves(
            &self.store,
            self.manifest.leaf_points,
            self.manifest.total_points,
            missing,
        )
        .await?;
        for (leaf, points) in downloaded {
            self.cache.insert_leaf_points(leaf, Arc::clone(&points));
            resolved.insert(leaf, points);
        }
        Ok(resolved)
    }

    /// Exact visible `[min_y, max_y]` starting from mixed-level initial frontier nodes.
    async fn visible_y_range_from(
        &mut self,
        init_nodes: &[(usize, u64, RipmapRecord)],
        min_x: f64,
        max_x: f64,
    ) -> Result<Option<[f64; 2]>, ClientError> {
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        let num_levels = self.manifest.levels.len();
        let mut straddling_by_level: Vec<Vec<(u64, f64, f64)>> = vec![Vec::new(); num_levels];

        for (lvl, idx, rec) in init_nodes {
            let n = rec.node;
            if min_x <= n.min_x && n.max_x <= max_x {
                min_y = min_y.min(n.min_y);
                max_y = max_y.max(n.max_y);
            } else {
                straddling_by_level[*lvl].push((*idx, n.min_y, n.max_y));
            }
        }

        for cur_level in (1..num_levels).rev() {
            let ids: Vec<u64> = std::mem::take(&mut straddling_by_level[cur_level])
                .into_iter()
                .filter(|&(_, lo, hi)| lo < min_y || hi > max_y)
                .map(|(idx, _, _)| idx)
                .collect();
            if ids.is_empty() {
                continue;
            }
            let drop = cur_level.min(MAX_BOUNDARY_JUMP);
            let target_level = cur_level - drop;
            let descendants = self
                .fetch_descendants(cur_level, target_level, &ids)
                .await?;
            for (idx, rec) in descendants {
                let n = rec.node;
                if n.max_x < min_x || n.min_x > max_x {
                    continue;
                }
                if min_x <= n.min_x && n.max_x <= max_x {
                    min_y = min_y.min(n.min_y);
                    max_y = max_y.max(n.max_y);
                } else {
                    straddling_by_level[target_level].push((idx, n.min_y, n.max_y));
                }
            }
        }

        let leaf_ids: Vec<u64> = std::mem::take(&mut straddling_by_level[0])
            .into_iter()
            .filter(|&(_, lo, hi)| lo < min_y || hi > max_y)
            .map(|(idx, _, _)| idx)
            .collect();
        if !leaf_ids.is_empty() {
            let leaves = self.fetch_leaves(&leaf_ids).await?;
            for leaf_idx in leaf_ids {
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
        Ok(min_y.is_finite().then_some([min_y, max_y]))
    }

    /// Expand frontier starting from mixed-level `init_nodes`, preserving exact x-order.
    async fn frontier_from(
        &mut self,
        init_nodes: Vec<(usize, u64, RipmapRecord)>,
        query: &ViewQuery,
        x_factor: f64,
        y_factor: f64,
        threshold: f64,
    ) -> Result<Vec<RipmapItem>, ClientError> {
        let min_x = query.min_x;
        let max_x = query.max_x;
        let fanout = self.manifest.fanout as u64;
        let num_levels = self.manifest.levels.len();

        let mut done: Vec<(u64, RipmapItem)> = Vec::with_capacity(init_nodes.len());
        let mut expand_by_level: Vec<Vec<(u64, bool, f64)>> = vec![Vec::new(); num_levels];

        let classify_into = |rec: RipmapRecord,
                             level: usize,
                             index: u64,
                             manifest: &RipmapManifest,
                             done: &mut Vec<(u64, RipmapItem)>,
                             expand_by_level: &mut [Vec<(u64, bool, f64)>]| {
            let node = rec.node;
            if node.max_x < min_x || node.min_x > max_x {
                return;
            }
            let fully_inside = min_x <= node.min_x && node.max_x <= max_x;
            let decision = if fully_inside {
                decide_record(&rec, level, manifest, query, x_factor, y_factor, threshold)
            } else {
                RipmapDecision::Expand
            };
            let order_key = index * fanout.pow(level as u32);
            match decision {
                RipmapDecision::Resolve => done.push((order_key, RipmapItem::Resolved(node))),
                RipmapDecision::Plane(slot) => {
                    let dir = rec.planes[slot];
                    done.push((
                        order_key,
                        RipmapItem::Plane {
                            level,
                            slot,
                            index,
                            node,
                            dir,
                            all_dirs: rec.planes,
                        },
                    ));
                }
                RipmapDecision::Expand if level > 0 => {
                    expand_by_level[level].push((
                        index,
                        !fully_inside,
                        (node.max_y - node.min_y) * y_factor,
                    ));
                }
                RipmapDecision::Expand => done.push((order_key, RipmapItem::Leaf(index))),
            }
        };

        for (lvl, idx, rec) in init_nodes {
            classify_into(
                rec,
                lvl,
                idx,
                &self.manifest,
                &mut done,
                &mut expand_by_level,
            );
        }

        for cur_level in (1..num_levels).rev() {
            let expanding = std::mem::take(&mut expand_by_level[cur_level]);
            if expanding.is_empty() {
                continue;
            }
            let only_straddling = expanding.iter().all(|&(_, s, _)| s);
            let max_span_sigma = expanding
                .iter()
                .map(|&(_, _, sp)| sp)
                .fold(0.0_f64, f64::max);
            let expand_all: Vec<u64> = expanding.iter().map(|&(idx, _, _)| idx).collect();

            // Fast path: when only straddling boundary nodes remain within `MAX_BOUNDARY_JUMP`
            // levels of level 0 and no lower levels have pending expansions:
            // - If vertical span `<= 2 * 2^cur_level` (Brownian leaf-span bound: ~2σ per leaf),
            //   jump straight to `level_0` (2 requests, 1 RTT) since interior leaves fit directly.
            // - Otherwise fetch `0..cur_level` concurrently in 1 RTT so interior siblings still
            //   use their sparse planes.
            let lower_empty = expand_by_level[1..cur_level].iter().all(Vec::is_empty);
            if only_straddling && lower_empty && cur_level <= MAX_BOUNDARY_JUMP && cur_level > 1 {
                let leaf_span_bound = (2u64 << cur_level) as f64;
                if max_span_sigma <= leaf_span_bound {
                    let leaves_desc = self.fetch_descendants(cur_level, 0, &expand_all).await?;
                    for (leaf_idx, rec) in leaves_desc {
                        classify_into(
                            rec,
                            0,
                            leaf_idx,
                            &self.manifest,
                            &mut done,
                            &mut expand_by_level,
                        );
                    }
                } else {
                    let subtrees = self.fetch_boundary_subtrees(cur_level, &expand_all).await?;
                    for &idx in &expand_all {
                        let mut sub_items = Vec::new();
                        walk_boundary_subtree(
                            cur_level,
                            idx,
                            &subtrees,
                            &self.manifest,
                            query,
                            x_factor,
                            y_factor,
                            threshold,
                            &mut sub_items,
                        );
                        done.extend(sub_items);
                    }
                }
                break;
            }

            let child_level = cur_level - 1;
            let pages = self.fetch_level_pages(child_level, &expand_all).await?;
            for idx in expand_all {
                let Some(children) = pages.get(&idx) else {
                    continue;
                };
                let base_child = idx * fanout;
                for (offset, rec) in children.iter().enumerate() {
                    classify_into(
                        rec.clone(),
                        child_level,
                        base_child + offset as u64,
                        &self.manifest,
                        &mut done,
                        &mut expand_by_level,
                    );
                }
            }
        }

        done.sort_unstable_by_key(|(order_key, _)| *order_key);
        Ok(done.into_iter().map(|(_, item)| item).collect())
    }
}

#[allow(clippy::too_many_arguments)]
fn walk_boundary_subtree(
    parent_level: usize,
    parent_idx: u64,
    subtrees: &[HashMap<u64, RipmapRecord>],
    manifest: &RipmapManifest,
    query: &ViewQuery,
    x_factor: f64,
    y_factor: f64,
    threshold: f64,
    out: &mut Vec<(u64, RipmapItem)>,
) {
    #[allow(clippy::too_many_arguments)]
    fn visit(
        level: usize,
        node_idx: u64,
        subtrees: &[HashMap<u64, RipmapRecord>],
        manifest: &RipmapManifest,
        query: &ViewQuery,
        x_factor: f64,
        y_factor: f64,
        threshold: f64,
        out: &mut Vec<(u64, RipmapItem)>,
    ) {
        let Some(rec) = subtrees[level].get(&node_idx) else {
            return;
        };
        let node = rec.node;
        if node.max_x < query.min_x || node.min_x > query.max_x {
            return;
        }
        let fully_inside = query.min_x <= node.min_x && node.max_x <= query.max_x;
        let decision = if fully_inside {
            decide_record(rec, level, manifest, query, x_factor, y_factor, threshold)
        } else {
            RipmapDecision::Expand
        };
        let fanout = manifest.fanout as u64;
        let order_key = node_idx * fanout.pow(level as u32);
        match decision {
            RipmapDecision::Resolve => out.push((order_key, RipmapItem::Resolved(node))),
            RipmapDecision::Plane(slot) => out.push((
                order_key,
                RipmapItem::Plane {
                    level,
                    slot,
                    index: node_idx,
                    node,
                    dir: rec.planes[slot],
                    all_dirs: rec.planes.clone(),
                },
            )),
            RipmapDecision::Expand if level == 0 => out.push((order_key, RipmapItem::Leaf(node_idx))),
            RipmapDecision::Expand => {
                let base = node_idx * fanout;
                for off in 0..fanout {
                    visit(
                        level - 1,
                        base + off,
                        subtrees,
                        manifest,
                        query,
                        x_factor,
                        y_factor,
                        threshold,
                        out,
                    );
                }
            }
        }
    }

    let fanout = manifest.fanout as u64;
    let base = parent_idx * fanout;
    for off in 0..fanout {
        visit(
            parent_level - 1,
            base + off,
            subtrees,
            manifest,
            query,
            x_factor,
            y_factor,
            threshold,
            out,
        );
    }
}

fn decide_record(
    rec: &RipmapRecord,
    level: usize,
    manifest: &RipmapManifest,
    query: &ViewQuery,
    x_factor: f64,
    y_factor: f64,
    threshold: f64,
) -> RipmapDecision {
    let node = &rec.node;
    if offscreen_y(node, query, y_factor) || node.fits_threshold(x_factor, y_factor, threshold) {
        return RipmapDecision::Resolve;
    }
    let w = (node.max_x - node.min_x) * x_factor;
    if w <= MAX_PLANE_BLOCK_WIDTH * threshold {
        let k_need = needed_depth(node, w, y_factor, threshold);
        let planes = &manifest.levels[level].planes;
        for (slot, pinfo) in planes.iter().enumerate() {
            if pinfo.k as u32 >= k_need && rec.planes[slot].cell_count > 0 {
                return RipmapDecision::Plane(slot);
            }
        }
    }
    RipmapDecision::Expand
}

/// Smallest y-depth `k >= 1` whose bins fit beside an x-block of on-screen width `w`.
fn needed_depth(node: &Node, w: f64, y_factor: f64, threshold: f64) -> u32 {
    let h_max = (threshold * threshold - w * w).max(0.0).sqrt();
    let span = (node.max_y - node.min_y) * y_factor;
    if span <= h_max {
        return 1;
    }
    let bins = (span / h_max.max(1e-12)).ceil();
    (bins.log2().ceil() as u32).clamp(1, 63)
}

async fn download_leaves(
    store: &Arc<dyn ByteRangeStore>,
    leaf_points: u32,
    total_points: u64,
    missing: Vec<u64>,
) -> Result<Vec<(u64, Arc<[Point]>)>, ClientError> {
    if missing.is_empty() {
        return Ok(Vec::new());
    }
    let requested: HashSet<u64> = missing.iter().copied().collect();
    let runs = coalesce_runs(missing, MAX_GAP_LEAVES, MAX_RUN_LEAVES);
    let leaf_size = leaf_points as u64;

    let futures: Vec<_> = runs
        .iter()
        .map(|&(start_leaf, leaf_count)| {
            let store = Arc::clone(store);
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
    let leaf_size_usize = leaf_points as usize;
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

struct MissingPlaneBlock {
    key: PlaneBlockKey,
    dir: PlaneDirEntry,
    all_dirs: Vec<PlaneDirEntry>,
}

/// Planned read in `planes.bin`: `(start_byte, end_byte)` and the list of `(key, rel_byte_offset, byte_len)`
/// blocks satisfied by that read.
struct PlannedPlaneRead {
    start_byte: u64,
    end_byte: u64,
    blocks: Vec<(PlaneBlockKey, usize, usize)>,
}

/// Plan coalesced byte-range reads in `planes.bin` across all missing sparse plane blocks,
/// promoting tiny coarse-slot runs into an adjacent already-fetched finer slot at the same level
/// when it saves a request for `<= PROMOTE_RUN_EXTRA_BYTES`.
fn plan_sparse_plane_reads(
    manifest: &RipmapManifest,
    missing: Vec<MissingPlaneBlock>,
) -> (Vec<PlannedPlaneRead>, HashMap<PlaneBlockKey, PlaneBlockKey>) {
    if missing.is_empty() {
        return (Vec::new(), HashMap::new());
    }
    let mut by_level: BTreeMap<usize, BTreeMap<usize, Vec<MissingPlaneBlock>>> = BTreeMap::new();
    for m in missing {
        by_level
            .entry(m.key.level)
            .or_default()
            .entry(m.key.slot)
            .or_default()
            .push(m);
    }

    let mut promoted: HashMap<PlaneBlockKey, PlaneBlockKey> = HashMap::new();
    let mut all_targets: Vec<(PlaneBlockKey, u64, u64)> = Vec::new(); // (key, byte_off, byte_len)

    for (level, mut by_slot) in by_level {
        let plane_infos = &manifest.levels[level].planes;
        let mut slots: Vec<usize> = by_slot.keys().copied().collect();
        slots.sort_unstable();

        // Small-run promotion upward into an already-present finer slot `next_slot = slot + 1`
        // when every block in the coarse slot is stored in `next_slot` and the extra payload is tiny.
        for &slot in &slots {
            let next_slot = slot + 1;
            if !by_slot.contains_key(&next_slot) || next_slot >= plane_infos.len() {
                continue;
            }
            let Some(blocks) = by_slot.get(&slot) else {
                continue;
            };
            let mut can_promote_all = true;
            let mut extra_bytes = 0u64;
            for b in blocks {
                let next_dir = b.all_dirs[next_slot];
                if next_dir.cell_count == 0 {
                    can_promote_all = false;
                    break;
                }
                extra_bytes += (next_dir.cell_count.saturating_sub(b.dir.cell_count)) as u64
                    * SPARSE_CELL_BYTES as u64;
            }
            if can_promote_all && extra_bytes <= PROMOTE_RUN_EXTRA_BYTES {
                let moved = by_slot.remove(&slot).unwrap();
                let dst = by_slot.entry(next_slot).or_default();
                for mut b in moved {
                    let orig = b.key;
                    let new_key = PlaneBlockKey {
                        level,
                        slot: next_slot,
                        index: orig.index,
                    };
                    promoted.insert(orig, new_key);
                    b.key = new_key;
                    b.dir = b.all_dirs[next_slot];
                    dst.push(b);
                }
            }
        }

        // Phase B (finest -> coarsest): coalesce each slot's byte ranges and freely absorb any
        // coarser-slot block whose finer-slot cell range already falls inside a planned finer read!
        let mut active_slots: Vec<usize> = by_slot.keys().copied().collect();
        active_slots.sort_unstable_by(|a, b| b.cmp(a)); // descending: finest first

        // Planned byte intervals `(start_byte, end_byte, slot)` already scheduled in `planes.bin`
        // for finer slots at this level.
        let mut covered_ranges: Vec<(u64, u64, usize)> = Vec::new();
        for slot in active_slots {
            let Some(blocks) = by_slot.remove(&slot) else {
                continue;
            };
            let base = plane_infos[slot].byte_offset;
            let mut slot_targets: Vec<(PlaneBlockKey, u64, u64)> = Vec::with_capacity(blocks.len());
            for b in blocks {
                // Check if this block already sits inside a scheduled finer-slot read at this level.
                let mut absorbed = false;
                for &(r_start, r_end, finer_slot) in &covered_ranges {
                    let fdir = b.all_dirs[finer_slot];
                    if fdir.cell_count == 0 {
                        continue;
                    }
                    let fbase = plane_infos[finer_slot].byte_offset;
                    let foff = fbase + fdir.cell_offset as u64 * SPARSE_CELL_BYTES as u64;
                    let flen = fdir.cell_count as u64 * SPARSE_CELL_BYTES as u64;
                    if foff >= r_start && foff + flen <= r_end {
                        let new_key = PlaneBlockKey {
                            level,
                            slot: finer_slot,
                            index: b.key.index,
                        };
                        promoted.insert(b.key, new_key);
                        all_targets.push((new_key, foff, flen));
                        absorbed = true;
                        break;
                    }
                }
                if !absorbed {
                    let off = base + b.dir.cell_offset as u64 * SPARSE_CELL_BYTES as u64;
                    let len = b.dir.cell_count as u64 * SPARSE_CELL_BYTES as u64;
                    if len > 0 {
                        slot_targets.push((b.key, off, len));
                    }
                }
            }
            if slot_targets.is_empty() {
                continue;
            }
            let spans: Vec<(u64, u64)> =
                slot_targets.iter().map(|&(_, off, len)| (off, len)).collect();
            for (s, e) in coalesce_spans(spans, MAX_GAP_PLANE_BYTES, MAX_RUN_PLANE_BYTES) {
                covered_ranges.push((s, e, slot));
            }
            all_targets.extend(slot_targets);
        }
    }

    // Resolve transitive promotions.
    let pkeys: Vec<PlaneBlockKey> = promoted.keys().copied().collect();
    for orig in pkeys {
        let mut cur = promoted[&orig];
        while let Some(&next) = promoted.get(&cur) {
            if next == cur {
                break;
            }
            cur = next;
        }
        promoted.insert(orig, cur);
    }

    // Deduplicate targets and coalesce across `planes.bin`!
    all_targets.sort_unstable_by_key(|&(_, off, len)| (off, len));
    all_targets.dedup_by_key(|t| t.0);

    let spans: Vec<(u64, u64)> = all_targets.iter().map(|&(_, off, len)| (off, len)).collect();
    let reads = coalesce_spans(spans, MAX_GAP_PLANE_BYTES, MAX_RUN_PLANE_BYTES);

    let mut planned = Vec::with_capacity(reads.len());
    let mut t_idx = 0;
    for (start_byte, end_byte) in reads {
        let mut blocks = Vec::new();
        while t_idx < all_targets.len() {
            let (key, off, len) = all_targets[t_idx];
            if off >= start_byte && off + len <= end_byte {
                blocks.push((key, (off - start_byte) as usize, len as usize));
                t_idx += 1;
            } else {
                break;
            }
        }
        planned.push(PlannedPlaneRead {
            start_byte,
            end_byte,
            blocks,
        });
    }

    (planned, promoted)
}

async fn download_sparse_planes(
    store: &Arc<dyn ByteRangeStore>,
    planned: Vec<PlannedPlaneRead>,
) -> Result<Vec<(PlaneBlockKey, Arc<[u8]>)>, ClientError> {
    if planned.is_empty() {
        return Ok(Vec::new());
    }
    let futures: Vec<_> = planned
        .into_iter()
        .map(|read| {
            let store = Arc::clone(store);
            async move {
                let len = (read.end_byte - read.start_byte) as usize;
                let bytes = store.read_range(PLANES_KEY, read.start_byte, len).await?;
                Ok::<_, ClientError>((read.blocks, bytes))
            }
        })
        .collect();

    let fetched = try_join_all(futures).await?;
    let mut out = Vec::new();
    for (blocks, bytes) in fetched {
        for (key, rel_off, len) in blocks {
            let slice = bytes
                .get(rel_off..rel_off + len)
                .ok_or_else(|| ClientError::CorruptData("short sparse plane read".to_owned()))?;
            out.push((key, Arc::from(slice)));
        }
    }
    Ok(out)
}

fn y_range_from_items(
    items: &[RipmapItem],
    leaves: &HashMap<u64, Arc<[Point]>>,
    min_x: f64,
    max_x: f64,
) -> Option<[f64; 2]> {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for item in items {
        match item {
            RipmapItem::Resolved(node) | RipmapItem::Plane { node, .. } => {
                if !node.is_empty() {
                    lo = lo.min(node.min_y);
                    hi = hi.max(node.max_y);
                }
            }
            RipmapItem::Leaf(idx) => {
                if let Some(points) = leaves.get(idx) {
                    let start = points.partition_point(|p| p.x < min_x);
                    let end = points.partition_point(|p| p.x <= max_x);
                    for p in &points[start..end] {
                        lo = lo.min(p.y);
                        hi = hi.max(p.y);
                    }
                }
            }
        }
    }
    lo.is_finite().then_some([lo, hi])
}

#[async_trait]
impl ProgressiveClient for RipmapClient {
    async fn query(&mut self, query: ViewQuery) -> Result<Frame, ClientError> {
        if self.manifest.root_level().is_none() || query.min_x > query.max_x {
            return Ok(Frame::empty());
        }
        let threshold = query.merge_threshold.max(1e-12);
        let x_den = (query.max_x - query.min_x).max(1e-12);
        let sigma = query.sigma_px.max(1e-12);
        let x_factor_est = query.screen_w / x_den / sigma;
        let explicit_y_factor = query.y_bounds.map(|yb| query.factors(yb).1);

        let init_nodes = self
            .jump_to_initial_frontier(&query, x_factor_est, explicit_y_factor, threshold)
            .await?;
        if init_nodes.is_empty() {
            return Ok(Frame::empty());
        }

        let (known_y_range, effective_y) = if let Some(explicit) = query.y_bounds {
            (None, explicit)
        } else {
            let Some(yr) = self
                .visible_y_range_from(&init_nodes, query.min_x, query.max_x)
                .await?
            else {
                return Ok(Frame::empty());
            };
            (Some(yr), yr)
        };
        let (x_factor, y_factor) = query.factors(effective_y);

        let items = self
            .frontier_from(init_nodes, &query, x_factor, y_factor, threshold)
            .await?;

        let leaf_indices: Vec<u64> = items
            .iter()
            .filter_map(|item| match item {
                RipmapItem::Leaf(idx) => Some(*idx),
                _ => None,
            })
            .collect();
        let (mut leaves, missing_leaves) = self.cached_leaves(&leaf_indices);

        // Lookup sparse plane blocks in cache, checking `slot..num_planes` for cross-k cache reuse!
        let mut planes: HashMap<PlaneBlockKey, (u8, Vec<SparseCell>)> = HashMap::new();
        let mut missing_planes: Vec<MissingPlaneBlock> = Vec::new();
        let mut seen_missing: HashSet<PlaneBlockKey> = HashSet::new();

        for item in &items {
            if let RipmapItem::Plane {
                level,
                slot,
                index,
                dir,
                all_dirs,
                ..
            } = item
            {
                let req_key = PlaneBlockKey {
                    level: *level,
                    slot: *slot,
                    index: *index,
                };
                if planes.contains_key(&req_key) || !seen_missing.insert(req_key) {
                    continue;
                }
                let level_planes = &self.manifest.levels[*level].planes;
                let mut hit = None;
                for (try_slot, pinfo) in level_planes.iter().enumerate().skip(*slot) {
                    if let Some(bytes) = self.cache.get_blob(plane_space(*level, try_slot), *index) {
                        let cells = decode_sparse_cells(&bytes).map_err(ClientError::CorruptData)?;
                        hit = Some((pinfo.k, cells));
                        break;
                    }
                }
                match hit {
                    Some(found) => {
                        planes.insert(req_key, found);
                    }
                    None => missing_planes.push(MissingPlaneBlock {
                        key: req_key,
                        dir: *dir,
                        all_dirs: all_dirs.clone(),
                    }),
                }
            }
        }

        let orig_missing_keys: Vec<PlaneBlockKey> = missing_planes.iter().map(|m| m.key).collect();
        let (planned_reads, promoted_map) = plan_sparse_plane_reads(&self.manifest, missing_planes);

        let (downloaded_leaves, downloaded_planes) = join2(
            download_leaves(
                &self.store,
                self.manifest.leaf_points,
                self.manifest.total_points,
                missing_leaves,
            ),
            download_sparse_planes(&self.store, planned_reads),
        )
        .await?;

        for (leaf, points) in downloaded_leaves {
            self.cache.insert_leaf_points(leaf, Arc::clone(&points));
            leaves.insert(leaf, points);
        }

        let mut fetched_map: HashMap<PlaneBlockKey, Arc<[u8]>> =
            HashMap::with_capacity(downloaded_planes.len());
        for (key, bytes) in downloaded_planes {
            self.cache
                .insert_blob(plane_space(key.level, key.slot), key.index, Arc::clone(&bytes));
            fetched_map.insert(key, bytes);
        }
        for orig in orig_missing_keys {
            let actual = promoted_map.get(&orig).copied().unwrap_or(orig);
            if let Some(bytes) = fetched_map.get(&actual) {
                let k = self.manifest.levels[actual.level].planes[actual.slot].k;
                let cells = decode_sparse_cells(bytes).map_err(ClientError::CorruptData)?;
                planes.insert(orig, (k, cells));
            }
        }

        let Some(y_range) =
            known_y_range.or_else(|| y_range_from_items(&items, &leaves, query.min_x, query.max_x))
        else {
            return Ok(Frame::empty());
        };

        // Assemble frame: merge resolved/leaf groups, emit merged sparse plane cells, and line extrema.
        let mut groups: Vec<Node> = Vec::with_capacity(items.len());
        let mut gpu_instances: Vec<f32> = Vec::with_capacity(items.len() * 7);
        let mut plane_line_nodes: Vec<Node> = Vec::new();

        for item in &items {
            match item {
                RipmapItem::Resolved(node) => {
                    push_group(*node, x_factor, y_factor, threshold, &mut groups);
                }
                RipmapItem::Leaf(leaf_idx) => {
                    if let Some(points) = leaves.get(leaf_idx) {
                        let start = points.partition_point(|p| p.x < query.min_x);
                        let end = points.partition_point(|p| p.x <= query.max_x);
                        if start < end {
                            pink_lemonade_progressive_common::collect_points(
                                &points[start..end],
                                x_factor,
                                y_factor,
                                threshold,
                                &mut groups,
                            );
                        }
                    }
                }
                RipmapItem::Plane {
                    level,
                    slot,
                    index,
                    node,
                    ..
                } => {
                    let key = PlaneBlockKey {
                        level: *level,
                        slot: *slot,
                        index: *index,
                    };
                    if let Some((actual_k, cells)) = planes.get(&key) {
                        emit_sparse_plane(
                            node,
                            *actual_k,
                            cells,
                            &query,
                            x_factor,
                            y_factor,
                            threshold,
                            &mut gpu_instances,
                        );
                    }
                    plane_line_nodes.push(*node);
                }
            }
        }

        let mut line_points = Vec::with_capacity((groups.len() + plane_line_nodes.len()) * 8);
        for node in &groups {
            node.emit_gpu(x_factor, y_factor, &mut gpu_instances);
            node.emit_line(&mut line_points);
        }
        for node in &plane_line_nodes {
            node.emit_line(&mut line_points);
        }
        line_points = sort_line_points(&line_points);

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

#[allow(clippy::too_many_arguments)]
fn emit_sparse_plane(
    node: &Node,
    actual_k: u8,
    cells: &[SparseCell],
    query: &ViewQuery,
    x_factor: f64,
    y_factor: f64,
    threshold: f64,
    gpu: &mut Vec<f32>,
) {
    let dx = node.max_x - node.min_x;
    let mut below = Node::default();
    let mut above = Node::default();
    let mut on_screen: Vec<Node> = Vec::new();
    let y_bounds = query.y_bounds;

    for cell in cells {
        if cell.is_empty() {
            continue;
        }
        let (bin_lo, bin_h) = node_bin_geometry(node, actual_k, cell.bin as usize);
        let cell_node = cell.unpack(node.min_x, dx, bin_lo, bin_h);
        if let Some([ylo, yhi]) = y_bounds {
            if cell_node.max_y < ylo && offscreen_y(&cell_node, query, y_factor) {
                let candidate = Node::combined(below, cell_node);
                if offscreen_y(&candidate, query, y_factor) {
                    below = candidate;
                    continue;
                }
            } else if cell_node.min_y > yhi && offscreen_y(&cell_node, query, y_factor) {
                let candidate = Node::combined(above, cell_node);
                if offscreen_y(&candidate, query, y_factor) {
                    above = candidate;
                    continue;
                }
            }
        }
        push_group(cell_node, x_factor, y_factor, threshold, &mut on_screen);
    }

    if !below.is_empty() {
        below.emit_gpu(x_factor, y_factor, gpu);
    }
    for g in &on_screen {
        g.emit_gpu(x_factor, y_factor, gpu);
    }
    if !above.is_empty() {
        above.emit_gpu(x_factor, y_factor, gpu);
    }
}
