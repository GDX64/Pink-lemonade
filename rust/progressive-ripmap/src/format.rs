//! `PLPRIPM2` binary format: single-manifest sparse moment ripmap with 16-byte quantized
//! [`SparseCell`]s, inline per-node plane directories in `levels.bin`, and packed `planes.bin`.
//!
//! Layout emitted into `ObjectWriter` (at most 4 objects):
//! - `manifest.bin`: header + per-level layout & plane offsets + inlined top (root) level records.
//! - `levels.bin`: non-root levels `0..root_level` packed back-to-back. Each record at level `ℓ`
//!   is a 64-byte [`PackedNode`] followed by `P_ℓ` 6-byte `(cell_offset: u32, cell_count: u16)`
//!   directory slots (one per stored y-depth `k` at level `ℓ`; `cell_count == 0` = absent/empty).
//! - `planes.bin`: all `(level, k)` sparse cell arrays packed back-to-back, each block's
//!   non-empty bins sorted by `bin: u16` as 16-byte [`SparseCell`]s.
//! - `points.bin`: flat packed 24-byte `(x, y, weight)` records in x-sorted order.

use pink_lemonade_progressive_common::{Moments, NODE_BYTES, Node, PackedNode};
use pink_lemonade_progressive_core::ClientError;

use crate::MAX_Y_DEPTH;

pub const MANIFEST_KEY: &str = "manifest.bin";
pub const LEVELS_KEY: &str = "levels.bin";
pub const PLANES_KEY: &str = "planes.bin";
pub const POINTS_KEY: &str = "points.bin";

pub const RIPMAP_MAGIC: [u8; 8] = *b"PLPRIPM2";
const RIPMAP_VERSION: u32 = 1;

/// One sparse plane directory slot attached to a node record: `(cell_offset: u32, cell_count: u16)`.
pub const DIR_ENTRY_BYTES: usize = 6;
/// 16-byte quantized sparse moment cell (4-byte aligned).
pub const SPARSE_CELL_BYTES: usize = 16;

/// Deepest `k` depth considered for mid-level windowed planes (handles 2×–4× y-zoom on spiky data).
const MAX_STORED_K_WINDOWED: u8 = 9;
/// Deepest `k` depth for coarse near-overview blocks (`>= 8192` points/block).
const MAX_STORED_K_COARSE: u8 = 8;
/// Extra y-depth beyond the dense budget allowed for sparse blocks (e.g., outlier-stretched blocks
/// whose occupied bins remain `<= block_points / min_points_per_bin`).
const SPARSE_EXTRA_DEPTH: u8 = 2;

/// Candidate y-depths `k` stored at `level` for blocks of `block_points` points.
///
/// - Level 0 (raw 32-point leaves) stores no planes.
/// - Cheap depths (`k <= k_dense - 2`, costing `<= 1%` of raw storage each) are stored at stride 1
///   so vertically coherent signals hit their exact `k` without 2× rounding overhead.
/// - Expensive top depths use stride 2 up to `min(k_dense + 2, k_max_level)`, giving sparse
///   outlier-stretched blocks +2 extra depth while capping coarse near-overview blocks at `k <= 8`.
pub fn plane_depths_for_level(
    level: usize,
    block_points: u64,
    min_points_per_bin: u32,
) -> Vec<u8> {
    if level == 0 {
        return Vec::new();
    }
    let ratio = block_points / min_points_per_bin.max(1) as u64;
    if ratio < 2 {
        return Vec::new();
    }
    let k_dense = (ratio.ilog2() as u8).min(MAX_Y_DEPTH);
    let max_k = if block_points >= 8_192 {
        MAX_STORED_K_COARSE
    } else {
        MAX_STORED_K_WINDOWED
    };
    let k_cap = k_dense
        .saturating_add(SPARSE_EXTRA_DEPTH)
        .min(max_k)
        .min(MAX_Y_DEPTH);
    let stride1_end = k_dense.saturating_sub(2).min(k_cap);

    let mut ks = Vec::new();
    for k in 1..=stride1_end {
        ks.push(k);
    }
    let mut k = stride1_end.saturating_add(2).min(k_cap);
    if stride1_end == 0 {
        k = 1;
    }
    while k <= k_cap {
        if ks.last().copied() != Some(k) {
            ks.push(k);
        }
        if k == k_cap {
            break;
        }
        k = k.saturating_add(2).min(k_cap);
    }
    ks
}

/// Per-block cap on non-empty bins stored in any plane at `level`.
pub fn max_bins_per_block(block_points: u64, min_points_per_bin: u32) -> usize {
    let cap = (block_points / min_points_per_bin.max(1) as u64) as usize;
    cap.min(u16::MAX as usize)
}

/// Byte size of one node record (packed node + inline plane directory) at a level with `plane_count` planes.
pub const fn record_bytes(plane_count: usize) -> usize {
    NODE_BYTES + plane_count * DIR_ENTRY_BYTES
}

/// Bin geometry `(bin_lo, bin_h)` of bin `b` at depth `k` within `[min_y, max_y]`.
///
/// Builder and client evaluate this identical expression on identical exact `f64` bounds.
pub fn bin_geometry(min_y: f64, max_y: f64, k: u8, b: usize) -> (f64, f64) {
    let bin_h = (max_y - min_y) / (1u64 << k) as f64;
    (min_y + b as f64 * bin_h, bin_h)
}

pub fn node_bin_geometry(node: &Node, k: u8, b: usize) -> (f64, f64) {
    bin_geometry(node.min_y, node.max_y, k, b)
}

/// Directory entry locating one block's sparse cells within a plane's cell stream in `planes.bin`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PlaneDirEntry {
    pub cell_offset: u32,
    pub cell_count: u16,
}

impl PlaneDirEntry {
    pub fn to_bytes(self) -> [u8; DIR_ENTRY_BYTES] {
        let mut buf = [0u8; DIR_ENTRY_BYTES];
        buf[0..4].copy_from_slice(&self.cell_offset.to_le_bytes());
        buf[4..6].copy_from_slice(&self.cell_count.to_le_bytes());
        buf
    }

    pub fn from_bytes(buf: &[u8; DIR_ENTRY_BYTES]) -> Self {
        Self {
            cell_offset: u32::from_le_bytes(buf[0..4].try_into().unwrap()),
            cell_count: u16::from_le_bytes(buf[4..6].try_into().unwrap()),
        }
    }
}

/// Decoded node record: exact `Node` + inline sparse-plane directory entries for this level.
#[derive(Debug, Clone)]
pub struct RipmapRecord {
    pub node: Node,
    pub planes: Vec<PlaneDirEntry>,
}

impl RipmapRecord {
    pub fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.node.to_bytes());
        for entry in &self.planes {
            out.extend_from_slice(&entry.to_bytes());
        }
    }

    pub fn decode_slice(bytes: &[u8], plane_count: usize) -> Result<Vec<Self>, String> {
        let stride = record_bytes(plane_count);
        if !bytes.len().is_multiple_of(stride) {
            return Err(format!(
                "record byte length {} is not a multiple of {stride}",
                bytes.len()
            ));
        }
        let count = bytes.len() / stride;
        let mut out = Vec::with_capacity(count);
        for chunk in bytes.chunks_exact(stride) {
            let node = PackedNode::from_bytes(chunk[0..NODE_BYTES].try_into().unwrap()).unpack();
            let mut planes = Vec::with_capacity(plane_count);
            for p in 0..plane_count {
                let off = NODE_BYTES + p * DIR_ENTRY_BYTES;
                planes.push(PlaneDirEntry::from_bytes(
                    chunk[off..off + DIR_ENTRY_BYTES].try_into().unwrap(),
                ));
            }
            out.push(Self { node, planes });
        }
        Ok(out)
    }
}

/// 16-byte quantized sparse moment cell using Popoviciu's `[0, 0.25]` normalized variance bound.
///
/// - `tx, ty ∈ [0, 1]` quantized to `u16` (`0..=65535`)
/// - `norm_var_x, norm_var_y ∈ [0, 0.25]` quantized to `u16` (`0..=65535`)
/// - `norm_cov_xy ∈ [-0.25, 0.25]` quantized to `i16` (`-32767..=32767`)
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct SparseCell {
    pub bin: u16,
    pub q_tx: u16,
    pub q_ty: u16,
    pub q_var_x: u16,
    pub q_cov_xy: i16,
    pub q_var_y: u16,
    pub mass: f32,
}

const _: () = assert!(std::mem::size_of::<SparseCell>() == SPARSE_CELL_BYTES);

impl SparseCell {
    /// Pack normalized moments `m` for bin `bin` within `[x_lo, x_lo + dx] × [y_lo, y_lo + dy]`.
    pub fn pack(bin: u16, m: Moments, x_lo: f64, dx: f64, y_lo: f64, dy: f64) -> Self {
        if m.mass == 0.0 {
            return Self::default();
        }
        let inv_mass = 1.0 / m.mass;
        let tx = if dx > 0.0 {
            ((m.x - x_lo) / dx).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let ty = if dy > 0.0 {
            ((m.y - y_lo) / dy).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let nvx = if dx > 0.0 {
            (m.m2_xx * inv_mass / (dx * dx)).clamp(0.0, 0.25)
        } else {
            0.0
        };
        let nvy = if dy > 0.0 {
            (m.m2_yy * inv_mass / (dy * dy)).clamp(0.0, 0.25)
        } else {
            0.0
        };
        let ncxy = if dx > 0.0 && dy > 0.0 {
            (m.m2_xy * inv_mass / (dx * dy)).clamp(-0.25, 0.25)
        } else {
            0.0
        };
        Self {
            bin,
            q_tx: (tx * 65535.0).round() as u16,
            q_ty: (ty * 65535.0).round() as u16,
            q_var_x: (nvx * 4.0 * 65535.0).round() as u16,
            q_cov_xy: (ncxy * 4.0 * 32767.0).round() as i16,
            q_var_y: (nvy * 4.0 * 65535.0).round() as u16,
            mass: m.mass as f32,
        }
    }

    pub fn unpack(self, x_lo: f64, dx: f64, y_lo: f64, dy: f64) -> Node {
        let mass = self.mass as f64;
        let tx = self.q_tx as f64 / 65535.0;
        let ty = self.q_ty as f64 / 65535.0;
        let nvx = self.q_var_x as f64 / (4.0 * 65535.0);
        let nvy = self.q_var_y as f64 / (4.0 * 65535.0);
        let max_cov = (nvx * nvy).sqrt();
        let ncxy = (self.q_cov_xy as f64 / (4.0 * 32767.0)).clamp(-max_cov, max_cov);

        let (x_hi, y_hi) = (x_lo + dx, y_lo + dy);
        Node {
            mass,
            x: x_lo + tx * dx,
            y: y_lo + ty * dy,
            m2_xx: nvx * mass * dx * dx,
            m2_xy: ncxy * mass * dx * dy,
            m2_yy: nvy * mass * dy * dy,
            min_x: x_lo,
            max_x: x_hi,
            min_y: y_lo,
            max_y: y_hi,
            first_x: x_lo,
            first_y: y_lo,
            last_x: x_hi,
            last_y: y_lo,
            min_y_x: x_lo,
            max_y_x: x_lo,
        }
    }

    pub fn to_bytes(self) -> [u8; SPARSE_CELL_BYTES] {
        let mut buf = [0u8; SPARSE_CELL_BYTES];
        buf[0..2].copy_from_slice(&self.bin.to_le_bytes());
        buf[2..4].copy_from_slice(&self.q_tx.to_le_bytes());
        buf[4..6].copy_from_slice(&self.q_ty.to_le_bytes());
        buf[6..8].copy_from_slice(&self.q_var_x.to_le_bytes());
        buf[8..10].copy_from_slice(&self.q_cov_xy.to_le_bytes());
        buf[10..12].copy_from_slice(&self.q_var_y.to_le_bytes());
        buf[12..16].copy_from_slice(&self.mass.to_le_bytes());
        buf
    }

    pub fn from_bytes(buf: &[u8; SPARSE_CELL_BYTES]) -> Self {
        Self {
            bin: u16::from_le_bytes(buf[0..2].try_into().unwrap()),
            q_tx: u16::from_le_bytes(buf[2..4].try_into().unwrap()),
            q_ty: u16::from_le_bytes(buf[4..6].try_into().unwrap()),
            q_var_x: u16::from_le_bytes(buf[6..8].try_into().unwrap()),
            q_cov_xy: i16::from_le_bytes(buf[8..10].try_into().unwrap()),
            q_var_y: u16::from_le_bytes(buf[10..12].try_into().unwrap()),
            mass: f32::from_le_bytes(buf[12..16].try_into().unwrap()),
        }
    }
}

pub fn decode_sparse_cells(bytes: &[u8]) -> Result<Vec<SparseCell>, String> {
    if !bytes.len().is_multiple_of(SPARSE_CELL_BYTES) {
        return Err(format!(
            "sparse cell slice length {} is not a multiple of {SPARSE_CELL_BYTES}",
            bytes.len()
        ));
    }
    Ok(bytes
        .as_chunks::<SPARSE_CELL_BYTES>()
        .0
        .iter()
        .map(SparseCell::from_bytes)
        .collect())
}

/// Metadata for one stored plane `(level, k)` packed inside `planes.bin`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaneInfo {
    pub k: u8,
    pub byte_offset: u64,
    pub cell_count: u64,
}

/// Metadata for one hierarchy level packed inside `levels.bin` (or inlined in `manifest.bin` for root).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelInfo {
    pub node_count: u64,
    /// Byte offset in `levels.bin` (0 for the inlined root level).
    pub level_byte_offset: u64,
    /// Planes stored at this level, in strictly ascending `k` order.
    pub planes: Vec<PlaneInfo>,
}

/// Complete static ripmap manifest loaded in a single `read_range` on `open`.
#[derive(Debug, Clone)]
pub struct RipmapManifest {
    pub total_points: u64,
    pub leaf_points: u32,
    pub fanout: u32,
    pub min_points_per_bin: u32,
    pub levels: Vec<LevelInfo>,
    pub root_records: Vec<RipmapRecord>,
}

/// Fixed header length in `manifest.bin` before variable-length tables.
pub const MANIFEST_HEADER_BYTES: usize = 40;

impl RipmapManifest {
    pub fn empty(leaf_points: u32, fanout: u32, min_points_per_bin: u32) -> Self {
        Self {
            total_points: 0,
            leaf_points,
            fanout,
            min_points_per_bin,
            levels: Vec::new(),
            root_records: Vec::new(),
        }
    }

    pub fn root_level(&self) -> Option<usize> {
        self.levels.len().checked_sub(1)
    }

    pub fn approx_memory_bytes(&self) -> usize {
        let levels_bytes: usize = self
            .levels
            .iter()
            .map(|l| std::mem::size_of::<LevelInfo>() + l.planes.len() * std::mem::size_of::<PlaneInfo>())
            .sum();
        let root_bytes: usize = self
            .root_records
            .iter()
            .map(|r| {
                std::mem::size_of::<RipmapRecord>()
                    + r.planes.len() * std::mem::size_of::<PlaneDirEntry>()
            })
            .sum();
        std::mem::size_of::<Self>() + levels_bytes + root_bytes
    }

    /// Read the declared total byte length from the first 40 bytes of `manifest.bin`.
    pub fn declared_len(prefix: &[u8]) -> Result<usize, ClientError> {
        let corrupt = |msg: &str| ClientError::CorruptData(format!("{MANIFEST_KEY}: {msg}"));
        if prefix.len() < MANIFEST_HEADER_BYTES || prefix[0..8] != RIPMAP_MAGIC {
            return Err(corrupt("bad magic or too short"));
        }
        let version = u32::from_le_bytes(prefix[8..12].try_into().unwrap());
        if version != RIPMAP_VERSION {
            return Err(corrupt(&format!("unsupported version {version}")));
        }
        let total_len = u32::from_le_bytes(prefix[36..40].try_into().unwrap()) as usize;
        if total_len < MANIFEST_HEADER_BYTES {
            return Err(corrupt("invalid declared manifest length"));
        }
        Ok(total_len)
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(MANIFEST_HEADER_BYTES + 1024);
        out.extend_from_slice(&RIPMAP_MAGIC);
        out.extend_from_slice(&RIPMAP_VERSION.to_le_bytes());
        out.extend_from_slice(&self.leaf_points.to_le_bytes());
        out.extend_from_slice(&self.fanout.to_le_bytes());
        out.extend_from_slice(&self.min_points_per_bin.to_le_bytes());
        out.extend_from_slice(&(self.levels.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.total_points.to_le_bytes());
        // Placeholder for total_manifest_bytes at [36..40].
        out.extend_from_slice(&0u32.to_le_bytes());

        for lvl in &self.levels {
            out.extend_from_slice(&lvl.node_count.to_le_bytes());
            out.extend_from_slice(&lvl.level_byte_offset.to_le_bytes());
            out.push(lvl.planes.len() as u8);
            for p in &lvl.planes {
                out.push(p.k);
                out.extend_from_slice(&p.byte_offset.to_le_bytes());
                out.extend_from_slice(&p.cell_count.to_le_bytes());
            }
        }
        for rec in &self.root_records {
            rec.encode_into(&mut out);
        }

        let total_len = u32::try_from(out.len()).expect("manifest exceeds 4 GiB");
        out[36..40].copy_from_slice(&total_len.to_le_bytes());
        out
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, ClientError> {
        let corrupt = |msg: &str| ClientError::CorruptData(format!("{MANIFEST_KEY}: {msg}"));
        let declared = Self::declared_len(bytes)?;
        if bytes.len() != declared {
            return Err(corrupt(&format!(
                "length mismatch: got {}, declared {declared}",
                bytes.len()
            )));
        }
        let leaf_points = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let fanout = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let min_points_per_bin = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
        let num_levels = u32::from_le_bytes(bytes[24..28].try_into().unwrap()) as usize;
        let total_points = u64::from_le_bytes(bytes[28..36].try_into().unwrap());

        if leaf_points == 0 || fanout < 2 || min_points_per_bin == 0 {
            return Err(corrupt("invalid parameters in manifest"));
        }

        let mut cursor = MANIFEST_HEADER_BYTES;
        let mut levels = Vec::with_capacity(num_levels);
        for _ in 0..num_levels {
            let hdr = bytes
                .get(cursor..cursor + 17)
                .ok_or_else(|| corrupt("truncated level metadata"))?;
            let node_count = u64::from_le_bytes(hdr[0..8].try_into().unwrap());
            let level_byte_offset = u64::from_le_bytes(hdr[8..16].try_into().unwrap());
            let plane_count = hdr[16] as usize;
            cursor += 17;

            let mut planes = Vec::with_capacity(plane_count);
            for _ in 0..plane_count {
                let entry = bytes
                    .get(cursor..cursor + 17)
                    .ok_or_else(|| corrupt("truncated plane metadata"))?;
                let k = entry[0];
                if k == 0 || k > MAX_Y_DEPTH {
                    return Err(corrupt("invalid plane y-depth k"));
                }
                let byte_offset = u64::from_le_bytes(entry[1..9].try_into().unwrap());
                let cell_count = u64::from_le_bytes(entry[9..17].try_into().unwrap());
                cursor += 17;
                planes.push(PlaneInfo {
                    k,
                    byte_offset,
                    cell_count,
                });
            }
            levels.push(LevelInfo {
                node_count,
                level_byte_offset,
                planes,
            });
        }

        let root_records = if let Some(root_info) = levels.last() {
            let recs = RipmapRecord::decode_slice(&bytes[cursor..], root_info.planes.len())
                .map_err(ClientError::CorruptData)?;
            if recs.len() as u64 != root_info.node_count {
                return Err(corrupt("root record count mismatch"));
            }
            recs
        } else {
            if cursor != bytes.len() {
                return Err(corrupt("trailing bytes in empty manifest"));
            }
            Vec::new()
        };

        Ok(Self {
            total_points,
            leaf_points,
            fanout,
            min_points_per_bin,
            levels,
            root_records,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pink_lemonade_progressive_core::Point;

    #[test]
    fn sparse_cell_roundtrip_preserves_moments() {
        let pts = [
            Point::new(0.1, 2.0, 1.0),
            Point::new(0.2, 2.5, 3.0),
            Point::new(0.35, 2.2, 0.5),
        ];
        let mut m = Moments::default();
        for p in pts {
            m.push(p);
        }
        let packed = SparseCell::from_bytes(&SparseCell::pack(7, m, 0.0, 0.4, 2.0, 0.5).to_bytes());
        assert_eq!(packed.bin, 7);
        let n = packed.unpack(0.0, 0.4, 2.0, 0.5);
        for (got, want) in [
            (n.mass, m.mass),
            (n.x, m.x),
            (n.y, m.y),
            (n.m2_xx, m.m2_xx),
            (n.m2_xy, m.m2_xy),
            (n.m2_yy, m.m2_yy),
        ] {
            assert!(
                (got - want).abs() <= 1e-4 * want.abs().max(1.0),
                "{got} vs {want}"
            );
        }
    }
}
