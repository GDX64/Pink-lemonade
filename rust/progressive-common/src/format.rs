//! Binary manifest & object naming conventions for the paged moment hierarchy.
//!
//! Each strategy stamps its own 8-byte magic (e.g. `PLPTREE2`, `PLPKDTR1`) so artifacts from
//! different point orderings can never be opened by the wrong client.
//!
//! Layout emitted into `ObjectWriter`:
//! - `manifest.bin`: header + per-level node counts + inlined top (root) level nodes.
//! - `level_0.bin`, `level_1.bin`, ...: fixed-width 64-byte [`PackedNode`] arrays for every
//!   level below the inlined root level.
//! - `points.bin`: flat packed 24-byte `(x, y, weight)` records (in hierarchy order), grouped
//!   into leaves of `leaf_points` points.

use pink_lemonade_progressive_core::ClientError;

use crate::node::{NODE_BYTES, Node, PackedNode, decode_packed_nodes};

pub const MANIFEST_KEY: &str = "manifest.bin";
pub const POINTS_KEY: &str = "points.bin";

const VERSION: u32 = 2;

pub fn level_key(level: usize) -> String {
    format!("level_{level}.bin")
}

/// Static index metadata loaded once on `open_client`.
#[derive(Debug, Clone)]
pub struct Manifest {
    pub magic: [u8; 8],
    pub total_points: u64,
    pub leaf_points: u32,
    pub fanout: u32,
    /// Number of nodes at each level `0..num_levels`.
    /// Level 0 holds leaf summaries; the last level (`num_levels - 1`) is inlined in `root_nodes`.
    pub level_counts: Vec<u64>,
    pub root_nodes: Vec<Node>,
}

impl Manifest {
    pub fn empty(magic: [u8; 8], leaf_points: u32, fanout: u32) -> Self {
        Self {
            magic,
            total_points: 0,
            leaf_points,
            fanout,
            level_counts: Vec::new(),
            root_nodes: Vec::new(),
        }
    }

    pub fn root_level(&self) -> Option<usize> {
        self.level_counts.len().checked_sub(1)
    }

    pub fn approx_memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.level_counts.len() * std::mem::size_of::<u64>()
            + self.root_nodes.len() * std::mem::size_of::<Node>()
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            32 + self.level_counts.len() * 8 + self.root_nodes.len() * NODE_BYTES,
        );
        out.extend_from_slice(&self.magic);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&self.leaf_points.to_le_bytes());
        out.extend_from_slice(&self.fanout.to_le_bytes());
        out.extend_from_slice(&(self.level_counts.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.total_points.to_le_bytes());
        for &count in &self.level_counts {
            out.extend_from_slice(&count.to_le_bytes());
        }
        for &node in &self.root_nodes {
            out.extend_from_slice(&node.to_bytes());
        }
        out
    }

    pub fn decode(bytes: &[u8], magic: [u8; 8]) -> Result<Self, ClientError> {
        if bytes.len() < 32 {
            return Err(ClientError::CorruptData("manifest.bin too short".to_owned()));
        }
        if bytes[0..8] != magic {
            return Err(ClientError::CorruptData(format!(
                "invalid manifest magic (expected {:?})",
                String::from_utf8_lossy(&magic)
            )));
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != VERSION {
            return Err(ClientError::CorruptData(format!(
                "unsupported manifest version {version}"
            )));
        }
        let leaf_points = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let fanout = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let num_levels = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize;
        let total_points = u64::from_le_bytes(bytes[24..32].try_into().unwrap());

        if leaf_points == 0 || fanout < 2 {
            return Err(ClientError::CorruptData(
                "invalid leaf_points or fanout in manifest".to_owned(),
            ));
        }

        let counts_end = 32 + num_levels * 8;
        if bytes.len() < counts_end {
            return Err(ClientError::CorruptData(
                "manifest.bin truncated in level_counts".to_owned(),
            ));
        }
        let mut level_counts = Vec::with_capacity(num_levels);
        for i in 0..num_levels {
            let off = 32 + i * 8;
            level_counts.push(u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()));
        }

        let root_nodes: Vec<Node> = decode_packed_nodes(&bytes[counts_end..])
            .map_err(ClientError::CorruptData)?
            .into_iter()
            .map(PackedNode::unpack)
            .collect();
        let expected_root = level_counts.last().copied().unwrap_or(0) as usize;
        if root_nodes.len() != expected_root {
            return Err(ClientError::CorruptData(format!(
                "manifest root node count mismatch: got {}, expected {expected_root}",
                root_nodes.len()
            )));
        }

        Ok(Self {
            magic,
            total_points,
            leaf_points,
            fanout,
            level_counts,
            root_nodes,
        })
    }
}
