//! Online ripmap client: x-hierarchy traversal that resolves σ-wide x-blocks through dense
//! y-bin moment planes, fetching planes and raw boundary leaves concurrently.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use pink_lemonade_progressive_common::{
    Decision, FrontierItem, HierarchyReader, MomentCell, Node, assemble_frame, coalesce_runs,
    decode_cells, join2, offscreen_y, read_whole_object, try_join_all,
};
use pink_lemonade_progressive_core::{
    ByteRangeStore, ClientConfig, ClientError, Frame, ProgressiveClient, ViewQuery, async_trait,
};

use crate::format::{
    HIERARCHY_MAGIC, RIPMAP_KEY, RipmapHeader, block_bytes, node_bin_geometry, plane_key,
};

/// Resolve via planes once an x-block is at most this fraction of the merge threshold wide.
const MAX_PLANE_BLOCK_WIDTH: f64 = 0.9;
/// Bridge gaps of up to 16 KiB between needed blocks of one plane; cap single reads at 1 MiB.
const MAX_GAP_BYTES: usize = 16 * 1024;
const MAX_RUN_BYTES: usize = 1024 * 1024;
/// Budget reserved for ripmap metadata held outside the hierarchy reader's cache.
const METADATA_RESERVE_BYTES: usize = 1024;

type PlaneKey = (usize, u8, u64);

pub struct RipmapClient {
    reader: HierarchyReader,
    header: RipmapHeader,
}

impl RipmapClient {
    pub async fn open(
        store: Arc<dyn ByteRangeStore>,
        config: ClientConfig,
    ) -> Result<Self, ClientError> {
        let budget = config
            .memory_budget_bytes
            .saturating_sub(METADATA_RESERVE_BYTES);
        let (reader, header_bytes) = join2(
            HierarchyReader::open(Arc::clone(&store), HIERARCHY_MAGIC, budget),
            read_whole_object(store.as_ref(), RIPMAP_KEY),
        )
        .await?;
        let header = RipmapHeader::decode(&header_bytes)?;
        if header.kmax.len() > METADATA_RESERVE_BYTES / 2
            || header.kmax.len() != reader.manifest().level_counts.len()
        {
            return Err(ClientError::CorruptData(
                "ripmap.bin level count does not match manifest".to_owned(),
            ));
        }
        Ok(Self { reader, header })
    }

    fn kmax(&self, level: usize) -> u8 {
        self.header.kmax.get(level).copied().unwrap_or(0)
    }
}

/// Smallest y-depth `k >= 1` whose bins fit beside an x-block of on-screen width `w`.
fn needed_depth(node: &Node, w: f64, y_factor: f64, threshold: f64) -> u32 {
    let h_max = (threshold * threshold - w * w).max(0.0).sqrt();
    let span = (node.max_y - node.min_y) * y_factor;
    if span <= h_max {
        return 1;
    }
    let bins = (span / h_max.max(1e-12)).ceil();
    // ceil(log2(bins)) via f64 exponent arithmetic; capped well above MAX_Y_DEPTH.
    (bins.log2().ceil() as u32).clamp(1, 63)
}

/// Download plane blocks grouped per `(level, k)`, bridging small gaps between blocks.
async fn download_planes(
    store: &Arc<dyn ByteRangeStore>,
    level_counts: &[u64],
    missing: Vec<PlaneKey>,
) -> Result<Vec<(PlaneKey, Arc<[MomentCell]>)>, ClientError> {
    if missing.is_empty() {
        return Ok(Vec::new());
    }
    let requested: HashSet<PlaneKey> = missing.iter().copied().collect();
    let mut by_plane: BTreeMap<(usize, u8), Vec<u64>> = BTreeMap::new();
    for (level, k, block) in missing {
        by_plane.entry((level, k)).or_default().push(block);
    }

    let mut futures = Vec::new();
    for ((level, k), blocks) in by_plane {
        let bytes_per_block = block_bytes(k);
        let max_gap = (MAX_GAP_BYTES / bytes_per_block) as u64;
        let max_run = (MAX_RUN_BYTES / bytes_per_block).max(1);
        let total_blocks = level_counts[level];
        let key = plane_key(level, k);
        for (start, count) in coalesce_runs(blocks, max_gap, max_run) {
            let store = Arc::clone(store);
            let key = key.clone();
            let end = (start + count as u64).min(total_blocks);
            let offset = start * bytes_per_block as u64;
            let len = (end - start) as usize * bytes_per_block;
            futures.push(async move {
                let bytes = store.read_range(&key, offset, len).await?;
                Ok::<_, ClientError>((level, k, start, bytes))
            });
        }
    }

    let mut out = Vec::with_capacity(requested.len());
    for (level, k, start, bytes) in try_join_all(futures).await? {
        let cells = decode_cells(&bytes).map_err(ClientError::CorruptData)?;
        for (i, chunk) in cells.chunks(1usize << k).enumerate() {
            let key = (level, k, start + i as u64);
            if requested.contains(&key) {
                out.push((key, Arc::from(chunk)));
            }
        }
    }
    Ok(out)
}

#[async_trait]
impl ProgressiveClient for RipmapClient {
    async fn query(&mut self, query: ViewQuery) -> Result<Frame, ClientError> {
        if self.reader.manifest().root_level().is_none() || query.min_x > query.max_x {
            return Ok(Frame::empty());
        }
        let Some(y_range) = self
            .reader
            .visible_y_range(query.min_x, query.max_x)
            .await?
        else {
            return Ok(Frame::empty());
        };
        let (x_factor, y_factor) = query.factors(query.effective_y_bounds(y_range));
        let threshold = query.merge_threshold.max(1e-12);

        let kmax = self.header.kmax.clone();
        let decide = |node: &Node, level: usize| {
            if offscreen_y(node, &query, y_factor)
                || node.fits_threshold(x_factor, y_factor, threshold)
            {
                return Decision::Resolve;
            }
            let w = (node.max_x - node.min_x) * x_factor;
            if w <= MAX_PLANE_BLOCK_WIDTH * threshold {
                let k = needed_depth(node, w, y_factor, threshold);
                if k <= kmax.get(level).copied().unwrap_or(0) as u32 {
                    return Decision::Plane(k as u8);
                }
            }
            Decision::Expand
        };
        let items = self
            .reader
            .frontier(query.min_x, query.max_x, &decide)
            .await?;

        // Split needed leaves / plane blocks into cache hits and misses.
        let leaf_indices: Vec<u64> = items
            .iter()
            .filter_map(|item| match item {
                FrontierItem::Leaf(idx) => Some(*idx),
                _ => None,
            })
            .collect();
        let (mut leaves, missing_leaves) = self.reader.cached_leaves(&leaf_indices);

        let mut planes: HashMap<PlaneKey, Arc<[MomentCell]>> = HashMap::new();
        let mut missing_planes = Vec::new();
        for item in &items {
            if let FrontierItem::Plane { level, index, k, .. } = *item {
                let key = (level, k, index);
                if planes.contains_key(&key) {
                    continue;
                }
                match self.reader.cache.get_plane_block(level, k, index) {
                    Some(cells) => {
                        planes.insert(key, cells);
                    }
                    None => missing_planes.push(key),
                }
            }
        }

        let (downloaded_leaves, downloaded_planes) = join2(
            self.reader.download_leaves(missing_leaves),
            download_planes(
                self.reader.store(),
                &self.reader.manifest().level_counts,
                missing_planes,
            ),
        )
        .await?;
        self.reader.cache_leaves(downloaded_leaves, &mut leaves);
        for ((level, k, block), cells) in downloaded_planes {
            self.reader
                .cache
                .insert_plane_block(level, k, block, Arc::clone(&cells));
            planes.insert((level, k, block), cells);
        }

        let mut emit_plane = |item: &FrontierItem, gpu: &mut Vec<f32>| {
            let FrontierItem::Plane {
                level,
                index,
                node,
                k,
            } = *item
            else {
                return;
            };
            let Some(cells) = planes.get(&(level, k, index)) else {
                return;
            };
            let dx = node.max_x - node.min_x;
            for (b, cell) in cells.iter().enumerate() {
                if cell.is_empty() {
                    continue;
                }
                let (bin_lo, bin_h) = node_bin_geometry(&node, k, b);
                cell.unpack(node.min_x, dx, bin_lo, bin_h)
                    .emit_gpu(x_factor, y_factor, gpu);
            }
        };

        Ok(assemble_frame(
            items,
            &leaves,
            &query,
            (x_factor, y_factor),
            y_range,
            true,
            &mut emit_plane,
        ))
    }

    fn memory_bytes(&self) -> usize {
        self.reader.memory_bytes() + std::mem::size_of::<RipmapHeader>() + self.header.kmax.len()
    }
}

impl RipmapClient {
    /// Deepest stored y-depth at `level` (exposed for diagnostics).
    pub fn plane_depth(&self, level: usize) -> u8 {
        self.kmax(level)
    }
}
