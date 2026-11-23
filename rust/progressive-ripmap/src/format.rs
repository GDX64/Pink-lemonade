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

use pink_lemonade_progressive_common::{NODE_BYTES, Node, PackedNode};
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

/// Candidate y-depths `k` stored at `level` for blocks of `block_points` points.
///
/// Purely data-derived (no screen-resolution constants):
/// - Level 0 (raw `leaf_points`-point leaves) stores no planes (raw leaves are tiny and only
///   reached at extreme zoom).
/// - Cheap depths (`k <= k_dense - 2`, costing `<= 1%` of raw storage each) are stored at stride 1
///   so vertically coherent signals hit their exact `k` without 2× rounding overhead.
/// - Expensive top depths use stride 2 up to `min(k_dense + sparse_extra_depth, MAX_Y_DEPTH)`.
///   The per-block gate (`non_empty <= block_points / min_points_per_bin`) bounds every stored
///   plane to `<= N` bytes regardless of depth.
pub fn plane_depths_for_level(
    level: usize,
    block_points: u64,
    min_points_per_bin: u32,
    sparse_extra_depth: u8,
) -> Vec<u8> {
    if level == 0 {
        return Vec::new();
    }
    let ratio = block_points / min_points_per_bin.max(1) as u64;
    if ratio < 2 {
        return Vec::new();
    }
    let k_dense = (ratio.ilog2() as u8).min(MAX_Y_DEPTH);
    let k_cap = k_dense.saturating_add(sparse_extra_depth).min(MAX_Y_DEPTH);
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
