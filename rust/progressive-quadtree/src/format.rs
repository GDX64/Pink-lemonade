//! Static layout of the moment quadtree.
//!
//! - `manifest.bin`: header, one [`DepthInfo`] per depth, and the inlined top depths.
//! - `nodes_{g}.bin`: consecutive depths packed together until the object holds at least
//!   `min_object_nodes` records (only the last object may be smaller), so a deep tree never
//!   becomes thousands of tiny objects. Each depth is a contiguous array of 72-byte
//!   [`QuadRecord`]s ordered by parent, so the children of one node are adjacent.
//! - `points.bin`: 24-byte `(x, y, weight)` records in quadtree (Z) order; every subtree is a
//!   contiguous run and every leaf is sorted by x.

use pink_lemonade_progressive_common::{NODE_BYTES, Node, PackedNode};
use pink_lemonade_progressive_core::ClientError;

pub const MAGIC: [u8; 8] = *b"PLPQUAD1";
const VERSION: u32 = 1;

pub const MANIFEST_KEY: &str = "manifest.bin";
pub const POINTS_KEY: &str = "points.bin";

/// 64-byte moment node + `first: u32` + `count: u32`.
pub const RECORD_BYTES: usize = NODE_BYTES + 8;
/// High bit of `count` marks a leaf (`first` = point start, `count & !LEAF_BIT` = points).
pub const LEAF_BIT: u32 = 1 << 31;
/// `DepthInfo::object` value for depths inlined in the manifest.
pub const INLINE_OBJECT: u32 = u32::MAX;

const HEADER_BYTES: usize = 32;
const DEPTH_INFO_BYTES: usize = 24;

pub fn nodes_key(object: u32) -> String {
    format!("nodes_{object}.bin")
}

/// One quadtree node: exact moments plus either its children or its raw points.
#[derive(Debug, Clone, Copy)]
pub struct QuadRecord {
    pub node: PackedNode,
    first: u32,
    count: u32,
}

/// What lies below a [`QuadRecord`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Link {
    /// Children `[first, first + count)` in the next depth.
    Children { first: u64, count: u64 },
    /// Raw points `[start, start + count)` of `points.bin`.
    Points { start: u64, count: u64 },
}

impl QuadRecord {
    pub fn internal(node: Node, first_child: u32, child_count: u32) -> Self {
        debug_assert!((1..=4).contains(&child_count));
        Self {
            node: PackedNode::pack(node),
            first: first_child,
            count: child_count,
        }
    }

    pub fn leaf(node: Node, point_start: u32, point_count: u32) -> Self {
        debug_assert!(point_count < LEAF_BIT);
        Self {
            node: PackedNode::pack(node),
            first: point_start,
            count: point_count | LEAF_BIT,
        }
    }

    pub fn link(&self) -> Link {
        if self.count & LEAF_BIT != 0 {
            Link::Points {
                start: self.first as u64,
                count: (self.count & !LEAF_BIT) as u64,
            }
        } else {
            Link::Children {
                first: self.first as u64,
                count: self.count as u64,
            }
        }
    }

    pub fn to_bytes(self) -> [u8; RECORD_BYTES] {
        let mut buf = [0u8; RECORD_BYTES];
        buf[..NODE_BYTES].copy_from_slice(&self.node.to_bytes());
        buf[NODE_BYTES..NODE_BYTES + 4].copy_from_slice(&self.first.to_le_bytes());
        buf[NODE_BYTES + 4..].copy_from_slice(&self.count.to_le_bytes());
        buf
    }

    pub fn from_bytes(buf: &[u8; RECORD_BYTES]) -> Self {
        Self {
            node: PackedNode::from_bytes(buf[..NODE_BYTES].try_into().unwrap()),
            first: u32::from_le_bytes(buf[NODE_BYTES..NODE_BYTES + 4].try_into().unwrap()),
            count: u32::from_le_bytes(buf[NODE_BYTES + 4..].try_into().unwrap()),
        }
    }
}

pub fn decode_records(bytes: &[u8]) -> Result<Vec<QuadRecord>, String> {
    if !bytes.len().is_multiple_of(RECORD_BYTES) {
        return Err(format!(
            "record slice length {} is not a multiple of {RECORD_BYTES}",
            bytes.len()
        ));
    }
    Ok(bytes
        .as_chunks::<RECORD_BYTES>()
        .0
        .iter()
        .map(QuadRecord::from_bytes)
        .collect())
}

/// Where the records of one depth live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepthInfo {
    pub count: u64,
    /// `nodes_{object}.bin`, or [`INLINE_OBJECT`] for depths stored in the manifest.
    pub object: u32,
    /// Byte offset of the depth's first record inside its object.
    pub offset: u64,
}

#[derive(Debug, Clone)]
pub struct Manifest {
    pub total_points: u64,
    pub leaf_points: u32,
    pub depths: Vec<DepthInfo>,
    /// Records of the inlined top depths (`inline[d]` = depth `d`).
    pub inline: Vec<Vec<QuadRecord>>,
}

impl Manifest {
    pub fn approx_memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.depths.len() * std::mem::size_of::<DepthInfo>()
            + self
                .inline
                .iter()
                .map(|d| std::mem::size_of::<Vec<QuadRecord>>() + d.len() * std::mem::size_of::<QuadRecord>())
                .sum::<usize>()
    }

    pub fn root(&self) -> Option<QuadRecord> {
        self.inline.first().and_then(|d| d.first()).copied()
    }

    pub fn encode(&self) -> Vec<u8> {
        let inline_records: usize = self.inline.iter().map(Vec::len).sum();
        let mut out = Vec::with_capacity(
            HEADER_BYTES + self.depths.len() * DEPTH_INFO_BYTES + inline_records * RECORD_BYTES,
        );
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&self.leaf_points.to_le_bytes());
        out.extend_from_slice(&(self.depths.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.inline.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.total_points.to_le_bytes());
        for d in &self.depths {
            out.extend_from_slice(&d.count.to_le_bytes());
            out.extend_from_slice(&d.object.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&d.offset.to_le_bytes());
        }
        for record in self.inline.iter().flatten() {
            out.extend_from_slice(&record.to_bytes());
        }
        out
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, ClientError> {
        let corrupt = |msg: &str| ClientError::CorruptData(format!("quadtree manifest: {msg}"));
        if bytes.len() < HEADER_BYTES {
            return Err(corrupt("too short"));
        }
        if bytes[0..8] != MAGIC {
            return Err(corrupt("invalid magic (expected PLPQUAD1)"));
        }
        let u32_at = |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        let u64_at = |off: usize| u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        if u32_at(8) != VERSION {
            return Err(corrupt("unsupported version"));
        }
        let leaf_points = u32_at(12);
        let num_depths = u32_at(16) as usize;
        let inline_depths = u32_at(20) as usize;
        let total_points = u64_at(24);
        if inline_depths > num_depths || (num_depths > 0 && inline_depths == 0) {
            return Err(corrupt("invalid inline depth count"));
        }

        let infos_end = HEADER_BYTES + num_depths * DEPTH_INFO_BYTES;
        if bytes.len() < infos_end {
            return Err(corrupt("truncated depth table"));
        }
        let depths: Vec<DepthInfo> = (0..num_depths)
            .map(|d| {
                let off = HEADER_BYTES + d * DEPTH_INFO_BYTES;
                DepthInfo {
                    count: u64_at(off),
                    object: u32_at(off + 8),
                    offset: u64_at(off + 16),
                }
            })
            .collect();

        let mut records = decode_records(&bytes[infos_end..])
            .map_err(ClientError::CorruptData)?
            .into_iter();
        let mut inline = Vec::with_capacity(inline_depths);
        for info in &depths[..inline_depths] {
            if info.object != INLINE_OBJECT {
                return Err(corrupt("inline depth points at an object"));
            }
            let depth: Vec<QuadRecord> = records.by_ref().take(info.count as usize).collect();
            if depth.len() as u64 != info.count {
                return Err(corrupt("truncated inline records"));
            }
            inline.push(depth);
        }
        if records.next().is_some() {
            return Err(corrupt("trailing inline records"));
        }
        if depths[inline_depths..].iter().any(|d| d.object == INLINE_OBJECT) {
            return Err(corrupt("remote depth marked inline"));
        }

        Ok(Self {
            total_points,
            leaf_points,
            depths,
            inline,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pink_lemonade_progressive_core::Point;

    #[test]
    fn record_and_manifest_round_trip() {
        let node = Node::from_points(&[Point::new(0.0, 1.0, 1.0), Point::new(2.0, 3.0, 2.0)]);
        let leaf = QuadRecord::leaf(node, 7, 2);
        let internal = QuadRecord::internal(node, 3, 4);
        assert_eq!(QuadRecord::from_bytes(&leaf.to_bytes()).link(), Link::Points { start: 7, count: 2 });
        assert_eq!(
            QuadRecord::from_bytes(&internal.to_bytes()).link(),
            Link::Children { first: 3, count: 4 }
        );

        let manifest = Manifest {
            total_points: 2,
            leaf_points: 32,
            depths: vec![
                DepthInfo { count: 1, object: INLINE_OBJECT, offset: 0 },
                DepthInfo { count: 4, object: 0, offset: 0 },
            ],
            inline: vec![vec![internal]],
        };
        let decoded = Manifest::decode(&manifest.encode()).unwrap();
        assert_eq!(decoded.depths, manifest.depths);
        assert_eq!(decoded.root().unwrap().link(), internal.link());
    }
}
