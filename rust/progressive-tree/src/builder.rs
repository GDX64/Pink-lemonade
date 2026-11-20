//! Offline builder producing a paged multi-level moment tree (`manifest.bin`, `level_*.bin`, `points.bin`).

use pink_lemonade_progressive_core::{BuildError, DatasetBuilder, ObjectWriter, Point, StoreError};

use crate::format::{MANIFEST_KEY, Manifest, POINTS_KEY, level_key};
use crate::node::Node;

pub struct PagedTreeBuilder {
    leaf_points: u32,
    fanout: u32,
    points: Vec<Point>,
    last_x: f64,
}

impl PagedTreeBuilder {
    pub fn new(leaf_points: u32, fanout: u32) -> Self {
        Self {
            leaf_points: leaf_points.max(1),
            fanout: fanout.max(2),
            points: Vec::new(),
            last_x: f64::NEG_INFINITY,
        }
    }
}

impl DatasetBuilder for PagedTreeBuilder {
    fn push_chunk(&mut self, points: &[Point]) -> Result<(), BuildError> {
        for &p in points {
            if !p.x.is_finite() || !p.y.is_finite() || !p.weight.is_finite() {
                return Err(BuildError::InvalidInput(
                    "point coordinates and weight must be finite".to_owned(),
                ));
            }
            if p.x < self.last_x {
                return Err(BuildError::InvalidInput(format!(
                    "input points must be sorted by non-decreasing x ({} < {})",
                    p.x, self.last_x
                )));
            }
            self.last_x = p.x;
        }
        self.points.extend_from_slice(points);
        Ok(())
    }

    fn finish(self: Box<Self>, writer: &mut dyn ObjectWriter) -> Result<(), BuildError> {
        if self.points.is_empty() {
            let manifest = Manifest::empty(self.leaf_points, self.fanout);
            writer.write_object(MANIFEST_KEY, &manifest.encode())?;
            return Ok(());
        }

        // 1. Stream raw points into `points.bin`.
        {
            let mut sink = writer.create_object(POINTS_KEY)?;
            for p in &self.points {
                sink.write_all(&p.x.to_le_bytes()).map_err(StoreError::from)?;
                sink.write_all(&p.y.to_le_bytes()).map_err(StoreError::from)?;
                sink.write_all(&p.weight.to_le_bytes())
                    .map_err(StoreError::from)?;
            }
            sink.flush().map_err(StoreError::from)?;
        }

        // 2. Build level 0 (leaf summaries) and iteratively roll up until <= fanout root nodes.
        let leaf_size = self.leaf_points as usize;
        let fanout = self.fanout as usize;

        let mut current_level: Vec<Node> = self
            .points
            .chunks(leaf_size)
            .map(Node::from_points)
            .collect();

        let mut level_counts = Vec::new();
        let mut level_idx = 0usize;

        while current_level.len() > fanout {
            level_counts.push(current_level.len() as u64);
            write_level(writer, level_idx, &current_level)?;

            let next_level: Vec<Node> = current_level
                .chunks(fanout)
                .map(|group| {
                    let mut acc = Node::default();
                    for &node in group {
                        acc = Node::combined(acc, node);
                    }
                    acc
                })
                .collect();

            current_level = next_level;
            level_idx += 1;
        }

        // Inline the final top (root) level directly in `manifest.bin`.
        level_counts.push(current_level.len() as u64);
        let manifest = Manifest {
            total_points: self.points.len() as u64,
            leaf_points: self.leaf_points,
            fanout: self.fanout,
            level_counts,
            root_nodes: current_level,
        };
        writer.write_object(MANIFEST_KEY, &manifest.encode())?;

        Ok(())
    }
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
