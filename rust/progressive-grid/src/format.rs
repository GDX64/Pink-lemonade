//! Static layout of the sparse moment grid pyramid.
//!
//! Geometry: the data bounding box `[x0, x0 + w] × [y0, y0 + h]` is split into `2^i` columns
//! (x-level `i`) and `2^j` rows (y-level `j`). Coordinates are quantized once to `Q_BITS`-bit
//! fixed point, so the column of a point at level `i` is `qx >> (Q_BITS - i)` and every level
//! nests exactly inside the coarser one, in builder and client alike.
//!
//! - `manifest.bin`: header, bounding box, and per x-level the byte offset of its column
//!   records plus the planes `(i, j)` stored at that level (byte offset and cell count in
//!   `cells.bin`).
//! - `columns.bin`: per x-level, `2^i + 1` column records back to back (the last one is a
//!   sentinel holding the totals). A record is the column's exact 64-byte moment node, its
//!   first point in `points.bin` (`u32`), and its first cell in each plane of the level
//!   (`u32` each): the CSC column pointers of every plane at that level, in one read.
//! - `cells.bin`: every plane back to back; a plane lists its non-empty cells column-major
//!   (column, then row) as shared 16-byte [`SparseCell`]s (`bin` = row, quantized moments
//!   normalized against the cell rectangle). A column whose rows fail the per-column density
//!   gate has an empty run in that plane even though it holds points.
//! - `points.bin`: 24-byte `(x, y, weight)` records sorted by x, so every column at every
//!   level is one contiguous run.

use pink_lemonade_progressive_common::{NODE_BYTES, Node, PackedNode, SPARSE_CELL_BYTES, SparseCell};
use pink_lemonade_progressive_core::ClientError;

pub const MAGIC: [u8; 8] = *b"PLPGRID1";
const VERSION: u32 = 2;

pub const MANIFEST_KEY: &str = "manifest.bin";
pub const COLUMNS_KEY: &str = "columns.bin";
pub const CELLS_KEY: &str = "cells.bin";
pub const POINTS_KEY: &str = "points.bin";

/// Fixed-point precision of the quantized coordinates.
pub const Q_BITS: u32 = 30;
/// Deepest x-level the builder creates.
pub const MAX_LEVEL: u32 = 24;
/// Deepest y-level (row indices are stored in the cell's `u16` bin).
pub const MAX_ROW_LEVEL: u32 = 16;
/// Shared 16-byte sparse cell.
pub const GRID_CELL_BYTES: usize = SPARSE_CELL_BYTES;
const HEADER_BYTES: usize = 64;
const PLANE_INFO_BYTES: usize = 20;

/// Quantize `v` to `Q_BITS` bits within `[lo, lo + span]` (clamped; `0` for a degenerate span).
pub fn quantize(v: f64, lo: f64, span: f64) -> u32 {
    if span <= 0.0 {
        return 0;
    }
    let scaled = ((v - lo) / span * (1u64 << Q_BITS) as f64).floor();
    scaled.clamp(0.0, ((1u64 << Q_BITS) - 1) as f64) as u32
}

/// Index of quantized coordinate `q` at `level` (`2^level` slots).
pub fn index_at(q: u32, level: u32) -> u64 {
    (q >> (Q_BITS - level)) as u64
}

/// Width of one slot at `level`.
pub fn slot(span: f64, level: u32) -> f64 {
    span / (1u64 << level) as f64
}

/// Lower edge of slot `idx` at `level` (the same expression in builder and client, so cell
/// normalization rectangles agree bit for bit).
pub fn slot_lo(lo: f64, span: f64, idx: u64, level: u32) -> f64 {
    lo + slot(span, level) * idx as f64
}

/// Data bounding box `[x0, x0 + w] × [y0, y0 + h]`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Bounds {
    pub x0: f64,
    pub w: f64,
    pub y0: f64,
    pub h: f64,
}

impl Bounds {
    pub fn col(&self, x: f64, level: u32) -> u64 {
        index_at(quantize(x, self.x0, self.w), level)
    }

    /// Normalization rectangle `(x_lo, dx, y_lo, dy)` of cell `(c, r)` in plane `(i, j)`.
    pub fn cell_rect(&self, i: u32, c: u64, j: u32, r: u64) -> (f64, f64, f64, f64) {
        (
            slot_lo(self.x0, self.w, c, i),
            slot(self.w, i),
            slot_lo(self.y0, self.h, r, j),
            slot(self.h, j),
        )
    }
}

/// One stored grid `(i, j)`: its y-level and where its cells live in `cells.bin`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaneInfo {
    pub j: u32,
    /// Byte offset of the plane's first cell in `cells.bin`.
    pub offset: u64,
    pub cells: u64,
    /// Global plane index (cache namespace), assigned in manifest order on decode.
    pub id: u32,
}

impl PlaneInfo {
    /// Absolute index (in `GRID_CELL_BYTES` units) of the plane's cell `k` in `cells.bin`.
    pub fn absolute(&self, k: u64) -> u64 {
        self.offset / GRID_CELL_BYTES as u64 + k
    }
}

/// One x-level: where its column records start and the planes stored at it (sorted by `j`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelInfo {
    /// Byte offset of the level's first column record in `columns.bin`.
    pub offset: u64,
    pub planes: Vec<PlaneInfo>,
}

impl LevelInfo {
    pub fn record_bytes(&self) -> usize {
        column_record_bytes(self.planes.len())
    }
}

pub fn column_record_bytes(planes: usize) -> usize {
    NODE_BYTES + 4 + 4 * planes
}

/// Records of a level: `2^level` columns plus the sentinel.
pub fn records_at(level: u32) -> u64 {
    (1u64 << level) + 1
}

/// Append one column record.
pub fn write_record(node: Node, point_start: u32, cell_starts: &[u32], out: &mut Vec<u8>) {
    out.extend_from_slice(&PackedNode::pack(node).to_bytes());
    out.extend_from_slice(&point_start.to_le_bytes());
    for s in cell_starts {
        out.extend_from_slice(&s.to_le_bytes());
    }
}

/// Column node of the record at the start of `bytes`.
pub fn record_node(bytes: &[u8]) -> Node {
    PackedNode::from_bytes(bytes[..NODE_BYTES].try_into().unwrap()).unpack()
}

/// First point of the record at the start of `bytes`.
pub fn record_point_start(bytes: &[u8]) -> u64 {
    u32::from_le_bytes(bytes[NODE_BYTES..NODE_BYTES + 4].try_into().unwrap()) as u64
}

/// First cell in plane `p` of the record at the start of `bytes`.
pub fn record_cell_start(bytes: &[u8], p: usize) -> u64 {
    let off = NODE_BYTES + 4 + 4 * p;
    u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as u64
}

/// Append one stored cell.
pub fn write_cell(cell: SparseCell, out: &mut Vec<u8>) {
    out.extend_from_slice(&cell.to_bytes());
}

/// Decode one stored cell.
pub fn read_cell(bytes: &[u8; GRID_CELL_BYTES]) -> SparseCell {
    SparseCell::from_bytes(bytes)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Manifest {
    pub total_points: u64,
    pub col_leaf: u32,
    pub min_points_per_cell: u32,
    pub bounds: Bounds,
    /// One entry per x-level `0..=max_level` (empty for an empty dataset).
    pub levels: Vec<LevelInfo>,
}

impl Manifest {
    pub fn approx_memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self
                .levels
                .iter()
                .map(|l| std::mem::size_of::<LevelInfo>() + l.planes.len() * std::mem::size_of::<PlaneInfo>())
                .sum::<usize>()
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_BYTES + self.levels.len() * 12);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&self.col_leaf.to_le_bytes());
        out.extend_from_slice(&self.min_points_per_cell.to_le_bytes());
        out.extend_from_slice(&(self.levels.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.total_points.to_le_bytes());
        for v in [self.bounds.x0, self.bounds.w, self.bounds.y0, self.bounds.h] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for level in &self.levels {
            out.extend_from_slice(&level.offset.to_le_bytes());
            out.extend_from_slice(&(level.planes.len() as u32).to_le_bytes());
            for p in &level.planes {
                out.extend_from_slice(&p.j.to_le_bytes());
                out.extend_from_slice(&p.offset.to_le_bytes());
                out.extend_from_slice(&p.cells.to_le_bytes());
            }
        }
        out
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, ClientError> {
        let corrupt = |msg: &str| ClientError::CorruptData(format!("grid manifest: {msg}"));
        if bytes.len() < HEADER_BYTES {
            return Err(corrupt("too short"));
        }
        if bytes[0..8] != MAGIC {
            return Err(corrupt("invalid magic (expected PLPGRID1)"));
        }
        let u32_at = |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        let u64_at = |off: usize| u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        let f64_at = |off: usize| f64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        if u32_at(8) != VERSION {
            return Err(corrupt("unsupported version"));
        }
        let num_levels = u32_at(20) as usize;
        if num_levels > MAX_LEVEL as usize + 1 {
            return Err(corrupt("too many levels"));
        }
        let mut off = HEADER_BYTES;
        let mut levels = Vec::with_capacity(num_levels);
        let mut id = 0u32;
        for _ in 0..num_levels {
            if bytes.len() < off + 12 {
                return Err(corrupt("truncated level table"));
            }
            let offset = u64_at(off);
            let count = u32_at(off + 8) as usize;
            off += 12;
            if bytes.len() < off + count * PLANE_INFO_BYTES {
                return Err(corrupt("truncated plane table"));
            }
            let mut planes = Vec::with_capacity(count);
            for _ in 0..count {
                let j = u32_at(off);
                if j > MAX_ROW_LEVEL {
                    return Err(corrupt("plane y-level out of range"));
                }
                planes.push(PlaneInfo {
                    j,
                    offset: u64_at(off + 4),
                    cells: u64_at(off + 12),
                    id,
                });
                id += 1;
                off += PLANE_INFO_BYTES;
            }
            levels.push(LevelInfo { offset, planes });
        }
        if off != bytes.len() {
            return Err(corrupt("trailing bytes"));
        }
        Ok(Self {
            total_points: u64_at(24),
            col_leaf: u32_at(12),
            min_points_per_cell: u32_at(16),
            bounds: Bounds {
                x0: f64_at(32),
                w: f64_at(40),
                y0: f64_at(48),
                h: f64_at(56),
            },
            levels,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_round_trip() {
        let manifest = Manifest {
            total_points: 1000,
            col_leaf: 256,
            min_points_per_cell: 16,
            bounds: Bounds { x0: -1.0, w: 2.0, y0: 3.0, h: 4.5 },
            levels: vec![
                LevelInfo {
                    offset: 0,
                    planes: vec![PlaneInfo { j: 0, offset: 0, cells: 1, id: 0 }],
                },
                LevelInfo {
                    offset: 140,
                    planes: vec![
                        PlaneInfo { j: 1, offset: 28, cells: 3, id: 1 },
                        PlaneInfo { j: 3, offset: 112, cells: 9, id: 2 },
                    ],
                },
            ],
        };
        assert_eq!(Manifest::decode(&manifest.encode()).unwrap(), manifest);
    }

    #[test]
    fn levels_nest_exactly() {
        for v in [0.0, 0.1, 0.25, 0.333_333, 0.5, 0.999_999_9, 1.0] {
            let q = quantize(v, 0.0, 1.0);
            for level in 1..=MAX_LEVEL {
                assert_eq!(index_at(q, level) >> 1, index_at(q, level - 1));
            }
            assert_eq!(index_at(q, 0), 0);
        }
        assert_eq!(index_at(quantize(1.0, 0.0, 1.0), 4), 15);
        assert_eq!(quantize(5.0, 2.0, 0.0), 0);
    }
}
