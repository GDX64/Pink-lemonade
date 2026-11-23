//! Offline builder for `PLPRIPM2`: x-ordered moment hierarchy packed into `levels.bin` with
//! inline per-node plane directories, sparse 16-byte moment planes packed into `planes.bin`,
//! raw points in `points.bin`, and root records + offsets in `manifest.bin` (≤ 4 objects total).

use std::io::Write;

use pink_lemonade_progressive_common::{
    Moments, Node, POINT_BYTES, PointCollector, SPARSE_CELL_BYTES, SparseCell,
};
use pink_lemonade_progressive_core::{
    BuildError, DatasetBuilder, ObjectWriter, Point, StoreError,
};

use crate::RipmapStrategy;
use crate::format::{
    LEVELS_KEY, LevelInfo, MANIFEST_KEY, PLANES_KEY, POINTS_KEY, PlaneDirEntry, PlaneInfo,
    RipmapManifest, RipmapRecord, bin_geometry, max_bins_per_block, plane_depths_for_level,
    record_bytes,
};

/// Shifted raw-moment accumulator (`u = x - min_x`, `v = y - min_y`): avoids divisions in the
/// inner point loop and during binary bin halving while keeping >10 digits of `f64` precision.
#[derive(Default, Clone, Copy)]
struct RawAcc {
    w: f64,
    wu: f64,
    wv: f64,
    wuu: f64,
    wuv: f64,
    wvv: f64,
}

impl RawAcc {
    #[inline]
    fn push(&mut self, u: f64, v: f64, w: f64) {
        self.w += w;
        let wu = w * u;
        let wv = w * v;
        self.wu += wu;
        self.wv += wv;
        self.wuu += wu * u;
        self.wuv += wu * v;
        self.wvv += wv * v;
    }

    #[inline]
    fn add(self, b: Self) -> Self {
        Self {
            w: self.w + b.w,
            wu: self.wu + b.wu,
            wv: self.wv + b.wv,
            wuu: self.wuu + b.wuu,
            wuv: self.wuv + b.wuv,
            wvv: self.wvv + b.wvv,
        }
    }

    #[inline]
    fn into_moments(self, min_x: f64, min_y: f64) -> Moments {
        if self.w <= 0.0 {
            return Moments::default();
        }
        let inv = 1.0 / self.w;
        let mean_u = self.wu * inv;
        let mean_v = self.wv * inv;
        Moments {
            mass: self.w,
            x: min_x + mean_u,
            y: min_y + mean_v,
            m2_xx: (self.wuu - self.wu * mean_u).max(0.0),
            m2_xy: self.wuv - self.wu * mean_v,
            m2_yy: (self.wvv - self.wv * mean_v).max(0.0),
        }
    }
}

pub struct RipmapBuilder {
    config: RipmapStrategy,
    points: PointCollector,
}

impl RipmapBuilder {
    pub fn new(config: RipmapStrategy) -> Self {
        Self {
            config: RipmapStrategy {
                leaf_points: config.leaf_points.max(1),
                fanout: config.fanout.max(2),
                root_cap: config.root_cap.max(1),
                min_points_per_bin: config.min_points_per_bin.max(1),
            },
            points: PointCollector::default(),
        }
    }
}

impl DatasetBuilder for RipmapBuilder {
    fn push_chunk(&mut self, points: &[Point]) -> Result<(), BuildError> {
        self.points.push_chunk(points)
    }

    fn finish(self: Box<Self>, writer: &mut dyn ObjectWriter) -> Result<(), BuildError> {
        let RipmapStrategy {
            leaf_points,
            fanout,
            root_cap,
            min_points_per_bin,
        } = self.config;
        let points = self.points.points;

        if points.is_empty() {
            let manifest = RipmapManifest::empty(leaf_points, fanout, min_points_per_bin);
            writer.write_object(MANIFEST_KEY, &manifest.encode())?;
            return Ok(());
        }

        // 1. Write `points.bin` in buffered chunks.
        write_points(writer, &points)?;

        // 2. Build all hierarchy levels bottom-up in memory.
        let mut level_nodes: Vec<Vec<Node>> = Vec::new();
        let mut cur: Vec<Node> = points
            .chunks(leaf_points as usize)
            .map(Node::from_points)
            .collect();
        while cur.len() > root_cap as usize {
            let next: Vec<Node> = cur
                .chunks(fanout as usize)
                .map(|g| g.iter().fold(Node::default(), |acc, &n| Node::combined(acc, n)))
                .collect();
            level_nodes.push(cur);
            cur = next;
        }
        level_nodes.push(cur);

        let num_levels = level_nodes.len();
        let root_level = num_levels - 1;

        // 3. Emit `levels.bin` (non-root levels) and `planes.bin` (all stored sparse planes).
        let mut levels_sink = if root_level > 0 {
            Some(writer.create_object(LEVELS_KEY)?)
        } else {
            None
        };
        let mut planes_sink: Option<Box<dyn Write + Send>> = None;
        let mut levels_byte_offset = 0u64;
        let mut planes_byte_offset = 0u64;

        let mut level_infos = Vec::with_capacity(num_levels);
        let mut root_records = Vec::new();

        for (level, nodes) in level_nodes.iter().enumerate() {
            let block_points = leaf_points as u64 * (fanout as u64).pow(level as u32);
            let ks = plane_depths_for_level(level, block_points, min_points_per_bin);
            let max_bins = max_bins_per_block(block_points, min_points_per_bin);

            let (dirs, plane_infos) = if ks.is_empty() {
                (vec![Vec::new(); nodes.len()], Vec::new())
            } else {
                let (dirs, plane_bufs) =
                    build_sparse_planes(&points, nodes, block_points as usize, &ks, max_bins)?;
                let mut infos = Vec::with_capacity(ks.len());
                for (&k, buf) in ks.iter().zip(plane_bufs.iter()) {
                    let cell_count = (buf.len() / SPARSE_CELL_BYTES) as u64;
                    if !buf.is_empty() {
                        let sink = match planes_sink.as_mut() {
                            Some(s) => s,
                            None => planes_sink.insert(writer.create_object(PLANES_KEY)?),
                        };
                        sink.write_all(buf).map_err(StoreError::from)?;
                    }
                    infos.push(PlaneInfo {
                        k,
                        byte_offset: planes_byte_offset,
                        cell_count,
                    });
                    planes_byte_offset += buf.len() as u64;
                }
                (dirs, infos)
            };

            let node_count = nodes.len() as u64;
            if level < root_level {
                let sink = levels_sink.as_mut().expect("levels_sink created for root_level > 0");
                let mut buf = Vec::with_capacity(nodes.len() * record_bytes(ks.len()));
                for (&node, planes) in nodes.iter().zip(dirs) {
                    let rec = RipmapRecord { node, planes };
                    rec.encode_into(&mut buf);
                }
                sink.write_all(&buf).map_err(StoreError::from)?;
                level_infos.push(LevelInfo {
                    node_count,
                    level_byte_offset: levels_byte_offset,
                    planes: plane_infos,
                });
                levels_byte_offset += buf.len() as u64;
            } else {
                root_records = nodes
                    .iter()
                    .zip(dirs)
                    .map(|(&node, planes)| RipmapRecord { node, planes })
                    .collect();
                level_infos.push(LevelInfo {
                    node_count,
                    level_byte_offset: 0,
                    planes: plane_infos,
                });
            }
        }

        if let Some(mut sink) = levels_sink {
            sink.flush().map_err(StoreError::from)?;
        }
        if let Some(mut sink) = planes_sink {
            sink.flush().map_err(StoreError::from)?;
        }

        let manifest = RipmapManifest {
            total_points: points.len() as u64,
            leaf_points,
            fanout,
            min_points_per_bin,
            levels: level_infos,
            root_records,
        };
        writer.write_object(MANIFEST_KEY, &manifest.encode())?;
        Ok(())
    }
}

fn write_points(writer: &mut dyn ObjectWriter, points: &[Point]) -> Result<(), BuildError> {
    let mut sink = writer.create_object(POINTS_KEY)?;
    const CHUNK_POINTS: usize = 4096;
    let mut buf = Vec::with_capacity(CHUNK_POINTS * POINT_BYTES);
    for chunk in points.chunks(CHUNK_POINTS) {
        buf.clear();
        for p in chunk {
            buf.extend_from_slice(&p.x.to_le_bytes());
            buf.extend_from_slice(&p.y.to_le_bytes());
            buf.extend_from_slice(&p.weight.to_le_bytes());
        }
        sink.write_all(&buf).map_err(StoreError::from)?;
    }
    sink.flush().map_err(StoreError::from)?;
    Ok(())
}

/// Build sparse 16-byte cell streams for all requested `ks` (strictly ascending) at one level.
///
/// For each block `j` and plane `p_slot`, emits non-empty bins only if their count is at most
/// `max_bins` (otherwise records `cell_count = 0`, spending 0 bytes in `planes.bin`).
#[allow(clippy::type_complexity)]
fn build_sparse_planes(
    points: &[Point],
    nodes: &[Node],
    block_points: usize,
    ks: &[u8],
    max_bins: usize,
) -> Result<(Vec<Vec<PlaneDirEntry>>, Vec<Vec<u8>>), BuildError> {
    let num_planes = ks.len();
    let k_hi = *ks.last().expect("ks non-empty");
    let k_lo = ks[0];
    let finest = 1usize << k_hi;

    let mut dirs = vec![vec![PlaneDirEntry::default(); num_planes]; nodes.len()];
    let mut plane_bufs: Vec<Vec<u8>> = vec![Vec::new(); num_planes];
    let mut bins = vec![RawAcc::default(); finest];

    for (block_idx, (block, node)) in points.chunks(block_points).zip(nodes.iter()).enumerate() {
        let (min_x, max_x) = (node.min_x, node.max_x);
        let (min_y, max_y) = (node.min_y, node.max_y);
        let (dx, dy) = (max_x - min_x, max_y - min_y);
        let inv_dy_finest = if dy > 0.0 {
            finest as f64 / dy
        } else {
            0.0
        };

        bins.fill(RawAcc::default());
        for &p in block {
            let b = if dy > 0.0 {
                (((p.y - min_y) * inv_dy_finest) as usize).min(finest - 1)
            } else {
                0
            };
            bins[b].push(p.x - min_x, p.y - min_y, p.weight);
        }

        let mut len = finest;
        let mut slot = num_planes - 1;
        for k in (k_lo..=k_hi).rev() {
            if ks[slot] == k {
                let non_empty = bins[..len].iter().filter(|a| a.w > 0.0).count();
                if non_empty > 0 && non_empty <= max_bins {
                    let buf = &mut plane_bufs[slot];
                    let cell_offset = u32::try_from(buf.len() / SPARSE_CELL_BYTES).map_err(|_| {
                        BuildError::Format("sparse plane exceeds u32 cell offset capacity".to_owned())
                    })?;
                    dirs[block_idx][slot] = PlaneDirEntry {
                        cell_offset,
                        cell_count: non_empty as u16,
                    };
                    buf.reserve(non_empty * SPARSE_CELL_BYTES);
                    for (b, &acc) in bins[..len].iter().enumerate() {
                        if acc.w <= 0.0 {
                            continue;
                        }
                        let (bin_lo, bin_h) = bin_geometry(min_y, max_y, k, b);
                        let m = acc.into_moments(min_x, min_y);
                        let cell = SparseCell::pack(b as u16, m, min_x, dx, bin_lo, bin_h);
                        buf.extend_from_slice(&cell.to_bytes());
                    }
                }
                if slot == 0 {
                    break;
                }
                slot -= 1;
            }

            if k > k_lo {
                len /= 2;
                for b in 0..len {
                    bins[b] = bins[2 * b].add(bins[2 * b + 1]);
                }
            }
        }
    }

    Ok((dirs, plane_bufs))
}
