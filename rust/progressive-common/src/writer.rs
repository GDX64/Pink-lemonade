//! Offline writer producing a paged multi-level moment hierarchy
//! (`manifest.bin`, `level_*.bin`, `points.bin`) over an already-ordered point array.
//!
//! The point *order* defines the strategy: x-sorted order yields the 1D x-ordered moment tree,
//! while a k-d permutation (where every `L·Fᵏ`-point chunk is a spatial cell) yields a 2D
//! moment k-d tree. Both are read by the same [`crate::HierarchyReader`].

use pink_lemonade_progressive_core::{BuildError, ObjectWriter, Point, StoreError};

use crate::format::{MANIFEST_KEY, Manifest, POINTS_KEY, level_key};
use crate::node::Node;

/// Accumulates validated, x-sorted input chunks for builders that need the whole dataset.
#[derive(Debug)]
pub struct PointCollector {
    pub points: Vec<Point>,
    last_x: f64,
}

impl Default for PointCollector {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            last_x: f64::NEG_INFINITY,
        }
    }
}

impl PointCollector {
    pub fn push_chunk(&mut self, points: &[Point]) -> Result<(), BuildError> {
        self.last_x = validate_points(points, self.last_x)?;
        self.points.extend_from_slice(points);
        Ok(())
    }
}

/// Check that all points are finite and x-sorted (continuing from `last_x`), returning the
/// new last x.
pub fn validate_points(points: &[Point], mut last_x: f64) -> Result<f64, BuildError> {
    for &p in points {
        if !p.x.is_finite() || !p.y.is_finite() || !p.weight.is_finite() {
            return Err(BuildError::InvalidInput(
                "point coordinates and weight must be finite".to_owned(),
            ));
        }
        if p.x < last_x {
            return Err(BuildError::InvalidInput(format!(
                "input points must be sorted by non-decreasing x ({} < {last_x})",
                p.x
            )));
        }
        last_x = p.x;
    }
    Ok(last_x)
}

/// Node counts per level (`[leaves, ..., root]`) for `point_count` points, rolling up by
/// `fanout` until a level has at most `root_cap` nodes. Empty for zero points.
pub fn level_counts(point_count: u64, leaf_points: u32, fanout: u32, root_cap: u32) -> Vec<u64> {
    if point_count == 0 {
        return Vec::new();
    }
    let fanout = fanout.max(2) as u64;
    let root_cap = root_cap.max(1) as u64;
    let mut counts = vec![point_count.div_ceil(leaf_points.max(1) as u64)];
    while *counts.last().unwrap() > root_cap {
        let next = counts.last().unwrap().div_ceil(fanout);
        counts.push(next);
    }
    counts
}

/// Write `points` (in hierarchy order) plus all summary levels, inlining the top level in
/// `manifest.bin`. Returns the written manifest.
///
/// `root_cap = fanout` reproduces the original `PLPTREE2` layout exactly.
pub fn write_hierarchy(
    points: &[Point],
    magic: [u8; 8],
    leaf_points: u32,
    fanout: u32,
    root_cap: u32,
    writer: &mut dyn ObjectWriter,
) -> Result<Manifest, BuildError> {
    let leaf_points = leaf_points.max(1);
    let fanout = fanout.max(2);
    if points.is_empty() {
        let manifest = Manifest::empty(magic, leaf_points, fanout);
        writer.write_object(MANIFEST_KEY, &manifest.encode())?;
        return Ok(manifest);
    }

    // 1. Stream raw points into `points.bin`.
    {
        let mut sink = writer.create_object(POINTS_KEY)?;
        for p in points {
            sink.write_all(&p.x.to_le_bytes()).map_err(StoreError::from)?;
            sink.write_all(&p.y.to_le_bytes()).map_err(StoreError::from)?;
            sink.write_all(&p.weight.to_le_bytes())
                .map_err(StoreError::from)?;
        }
        sink.flush().map_err(StoreError::from)?;
    }

    // 2. Build level 0 (leaf summaries) and iteratively roll up until <= root_cap root nodes.
    let mut current_level: Vec<Node> = points
        .chunks(leaf_points as usize)
        .map(Node::from_points)
        .collect();

    let mut counts = Vec::new();
    let mut level_idx = 0usize;
    while current_level.len() > root_cap.max(1) as usize {
        counts.push(current_level.len() as u64);
        write_level(writer, level_idx, &current_level)?;
        current_level = current_level
            .chunks(fanout as usize)
            .map(|group| group.iter().fold(Node::default(), |acc, &n| Node::combined(acc, n)))
            .collect();
        level_idx += 1;
    }

    // Inline the final top (root) level directly in `manifest.bin`.
    counts.push(current_level.len() as u64);
    debug_assert_eq!(
        counts,
        level_counts(points.len() as u64, leaf_points, fanout, root_cap)
    );
    let manifest = Manifest {
        magic,
        total_points: points.len() as u64,
        leaf_points,
        fanout,
        level_counts: counts,
        root_nodes: current_level,
    };
    writer.write_object(MANIFEST_KEY, &manifest.encode())?;
    Ok(manifest)
}

fn write_level(
    writer: &mut dyn ObjectWriter,
    level: usize,
    nodes: &[Node],
) -> Result<(), BuildError> {
    let mut sink = writer.create_object(&level_key(level))?;
    for node in nodes {
        sink.write_all(&node.to_bytes()).map_err(StoreError::from)?;
    }
    sink.flush().map_err(StoreError::from)?;
    Ok(())
}
