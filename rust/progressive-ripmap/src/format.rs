//! `ripmap.bin` header and plane naming / geometry shared by builder and client.

use pink_lemonade_progressive_common::{CELL_BYTES, Node};
use pink_lemonade_progressive_core::ClientError;

use crate::MAX_Y_DEPTH;

pub const HIERARCHY_MAGIC: [u8; 8] = *b"PLPRIPM1";
pub const RIPMAP_KEY: &str = "ripmap.bin";
const RIPMAP_MAGIC: &[u8; 8] = b"PLPRIPH1";
const RIPMAP_VERSION: u32 = 1;

pub fn plane_key(level: usize, k: u8) -> String {
    format!("plane_{level}_{k}.bin")
}

/// Byte size of one block's `2^k` cells.
pub fn block_bytes(k: u8) -> usize {
    (1usize << k) * CELL_BYTES
}

/// Deepest stored y-depth for blocks of `block_points` points (0 = no planes).
pub fn max_depth(block_points: u64, min_points_per_bin: u32) -> u8 {
    let ratio = block_points / min_points_per_bin.max(1) as u64;
    if ratio < 2 {
        0
    } else {
        (ratio.ilog2() as u8).min(MAX_Y_DEPTH)
    }
}

/// Bin geometry `(bin_lo, bin_h)` of bin `b` at depth `k` within `node`'s y-range.
///
/// Builder and client evaluate this identical expression on identical exact `f64` bounds, so
/// cell normalization rectangles agree bit-for-bit.
pub fn bin_geometry(min_y: f64, max_y: f64, k: u8, b: usize) -> (f64, f64) {
    let bin_h = (max_y - min_y) / (1u64 << k) as f64;
    (min_y + b as f64 * bin_h, bin_h)
}

pub fn node_bin_geometry(node: &Node, k: u8, b: usize) -> (f64, f64) {
    bin_geometry(node.min_y, node.max_y, k, b)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RipmapHeader {
    pub min_points_per_bin: u32,
    /// `kmax[ℓ]` for every hierarchy level (`0` = no planes at that level).
    pub kmax: Vec<u8>,
}

impl RipmapHeader {
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(20 + self.kmax.len());
        out.extend_from_slice(RIPMAP_MAGIC);
        out.extend_from_slice(&RIPMAP_VERSION.to_le_bytes());
        out.extend_from_slice(&self.min_points_per_bin.to_le_bytes());
        out.extend_from_slice(&(self.kmax.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.kmax);
        out
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, ClientError> {
        let corrupt = |msg: &str| ClientError::CorruptData(format!("{RIPMAP_KEY}: {msg}"));
        if bytes.len() < 20 || &bytes[0..8] != RIPMAP_MAGIC {
            return Err(corrupt("bad magic or too short"));
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != RIPMAP_VERSION {
            return Err(corrupt(&format!("unsupported version {version}")));
        }
        let min_points_per_bin = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let levels = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
        let kmax = bytes
            .get(20..20 + levels)
            .ok_or_else(|| corrupt("truncated kmax table"))?
            .to_vec();
        if kmax.iter().any(|&k| k > MAX_Y_DEPTH) {
            return Err(corrupt("y-depth exceeds MAX_Y_DEPTH"));
        }
        Ok(Self {
            min_points_per_bin,
            kmax,
        })
    }
}
