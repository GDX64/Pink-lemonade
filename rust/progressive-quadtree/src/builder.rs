//! Offline builder: classic point-region quadtree over the data bounding box.
//!
//! Every node splits its region at the midpoint into 4 equal quadrants (squares in
//! normalized `[0, 1]²` data coordinates) until it holds at most `leaf_points` points, reaches
//! `max_depth`, or all its points coincide. Empty quadrants are not stored. Points are
//! permuted in place into quadrant (Z) order, so every subtree is a contiguous run of
//! `points.bin`; each leaf is then sorted by x.

use pink_lemonade_progressive_common::{Node, PointCollector};
use pink_lemonade_progressive_core::{BuildError, DatasetBuilder, ObjectWriter, Point, StoreError};

use crate::QuadtreeStrategy;
use crate::format::{
    DepthInfo, INLINE_OBJECT, LEAF_BIT, MANIFEST_KEY, Manifest, POINTS_KEY, QuadRecord,
    RECORD_BYTES, nodes_key,
};

pub struct QuadtreeBuilder {
    pub(crate) config: QuadtreeStrategy,
    pub(crate) points: PointCollector,
}

impl DatasetBuilder for QuadtreeBuilder {
    fn push_chunk(&mut self, points: &[Point]) -> Result<(), BuildError> {
        self.points.push_chunk(points)
    }

    fn finish(self: Box<Self>, writer: &mut dyn ObjectWriter) -> Result<(), BuildError> {
        let config = self.config;
        let mut points = self.points.points;
        if points.len() as u64 >= LEAF_BIT as u64 {
            return Err(BuildError::InvalidInput(format!(
                "moment quadtree supports < 2^31 points (got {})",
                points.len()
            )));
        }

        let depths = if points.is_empty() {
            Vec::new()
        } else {
            build_records(&mut points, config.leaf_points.max(1) as usize, config.max_depth)
        };
        write_layout(&points, &depths, &config, writer)
    }
}

/// Structural node before moments are attached.
#[derive(Debug, Clone, Copy)]
struct Proto {
    start: usize,
    end: usize,
    /// `(first_child, child_count)` in the next depth, or `None` for leaves.
    children: Option<(usize, usize)>,
}

#[derive(Debug, Clone, Copy)]
struct Region {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
}

impl Region {
    fn mid(&self) -> (f64, f64) {
        (
            self.min_x + (self.max_x - self.min_x) * 0.5,
            self.min_y + (self.max_y - self.min_y) * 0.5,
        )
    }

    fn quadrant(&self, q: u8, (mx, my): (f64, f64)) -> Self {
        let (min_x, max_x) = if q & 1 == 0 { (self.min_x, mx) } else { (mx, self.max_x) };
        let (min_y, max_y) = if q & 2 == 0 { (self.min_y, my) } else { (my, self.max_y) };
        Self { min_x, max_x, min_y, max_y }
    }
}

fn quadrant_of(p: &Point, (mx, my): (f64, f64)) -> u8 {
    u8::from(p.x >= mx) | (u8::from(p.y >= my) << 1)
}

fn all_coincide(points: &[Point]) -> bool {
    let first = points[0];
    points.iter().all(|p| p.x == first.x && p.y == first.y)
}

/// Build the tree breadth-first (children of each node adjacent in the next depth) and return
/// the records of every depth, with exact moments rolled up from the leaves.
fn build_records(points: &mut [Point], leaf_points: usize, max_depth: u32) -> Vec<Vec<QuadRecord>> {
    let root = points.iter().fold(
        Region {
            min_x: f64::INFINITY,
            max_x: f64::NEG_INFINITY,
            min_y: f64::INFINITY,
            max_y: f64::NEG_INFINITY,
        },
        |r, p| Region {
            min_x: r.min_x.min(p.x),
            max_x: r.max_x.max(p.x),
            min_y: r.min_y.min(p.y),
            max_y: r.max_y.max(p.y),
        },
    );

    let mut protos: Vec<Vec<Proto>> = Vec::new();
    let mut current = vec![(0usize, points.len(), root)];
    let mut depth = 0u32;
    while !current.is_empty() {
        let mut level = Vec::with_capacity(current.len());
        let mut next = Vec::new();
        for (start, end, region) in current {
            let run = &mut points[start..end];
            if run.len() <= leaf_points || depth >= max_depth || all_coincide(run) {
                run.sort_by(|a, b| a.x.total_cmp(&b.x));
                level.push(Proto { start, end, children: None });
                continue;
            }
            let mid = region.mid();
            run.sort_unstable_by_key(|p| quadrant_of(p, mid));
            let first = next.len();
            let mut lo = 0;
            for q in 0..4u8 {
                let hi = run.partition_point(|p| quadrant_of(p, mid) <= q);
                if hi > lo {
                    next.push((start + lo, start + hi, region.quadrant(q, mid)));
                }
                lo = hi;
            }
            level.push(Proto {
                start,
                end,
                children: Some((first, next.len() - first)),
            });
        }
        protos.push(level);
        current = next;
        depth += 1;
    }

    // Roll moments up from the deepest depth.
    let mut nodes: Vec<Vec<Node>> = vec![Vec::new(); protos.len()];
    for d in (0..protos.len()).rev() {
        let level: Vec<Node> = protos[d]
            .iter()
            .map(|p| match p.children {
                None => Node::from_points(&points[p.start..p.end]),
                Some((first, count)) => nodes[d + 1][first..first + count]
                    .iter()
                    .fold(Node::default(), |acc, &n| Node::combined(acc, n)),
            })
            .collect();
        nodes[d] = level;
    }

    protos
        .iter()
        .zip(&nodes)
        .map(|(level, nodes)| {
            level
                .iter()
                .zip(nodes)
                .map(|(p, &node)| match p.children {
                    None => QuadRecord::leaf(node, p.start as u32, (p.end - p.start) as u32),
                    Some((first, count)) => QuadRecord::internal(node, first as u32, count as u32),
                })
                .collect()
        })
        .collect()
}

/// Assign depths to the manifest (top depths totalling ≤ `root_cap` records) or to packed
/// `nodes_{g}.bin` objects of at least `min_object_nodes` records; a short final group is
/// merged into the previous object. Returns `(inline_depths, object index per remote depth)`.
pub(crate) fn plan_objects(counts: &[u64], root_cap: u64, min_object_nodes: u64) -> (usize, Vec<u32>) {
    let mut inline = 0;
    let mut inlined = 0u64;
    for &c in counts {
        if inline > 0 && inlined + c > root_cap {
            break;
        }
        inlined += c;
        inline += 1;
    }

    let mut objects = Vec::with_capacity(counts.len() - inline);
    let (mut object, mut filled) = (0u32, 0u64);
    for &c in &counts[inline..] {
        if filled >= min_object_nodes {
            object += 1;
            filled = 0;
        }
        objects.push(object);
        filled += c;
    }
    if object > 0 && filled < min_object_nodes {
        for o in objects.iter_mut().filter(|o| **o == object) {
            *o -= 1;
        }
    }
    (inline, objects)
}

fn write_layout(
    points: &[Point],
    depths: &[Vec<QuadRecord>],
    config: &QuadtreeStrategy,
    writer: &mut dyn ObjectWriter,
) -> Result<(), BuildError> {
    let counts: Vec<u64> = depths.iter().map(|d| d.len() as u64).collect();
    let (inline_depths, objects) = if counts.is_empty() {
        (0, Vec::new())
    } else {
        plan_objects(&counts, config.root_cap as u64, config.min_object_nodes as u64)
    };

    if !points.is_empty() {
        let mut sink = writer.create_object(POINTS_KEY)?;
        for p in points {
            sink.write_all(&p.x.to_le_bytes()).map_err(StoreError::from)?;
            sink.write_all(&p.y.to_le_bytes()).map_err(StoreError::from)?;
            sink.write_all(&p.weight.to_le_bytes()).map_err(StoreError::from)?;
        }
        sink.flush().map_err(StoreError::from)?;
    }

    let mut infos: Vec<DepthInfo> = counts[..inline_depths]
        .iter()
        .map(|&count| DepthInfo { count, object: INLINE_OBJECT, offset: 0 })
        .collect();
    let remote = &depths[inline_depths..];
    let mut d = 0;
    while d < remote.len() {
        let object = objects[d];
        let mut sink = writer.create_object(&nodes_key(object))?;
        let mut offset = 0u64;
        while d < remote.len() && objects[d] == object {
            infos.push(DepthInfo {
                count: remote[d].len() as u64,
                object,
                offset,
            });
            for record in &remote[d] {
                sink.write_all(&record.to_bytes()).map_err(StoreError::from)?;
            }
            offset += (remote[d].len() * RECORD_BYTES) as u64;
            d += 1;
        }
        sink.flush().map_err(StoreError::from)?;
    }

    let manifest = Manifest {
        total_points: points.len() as u64,
        leaf_points: config.leaf_points,
        depths: infos,
        inline: depths[..inline_depths].to_vec(),
    };
    writer.write_object(MANIFEST_KEY, &manifest.encode())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::Link;

    #[test]
    fn plan_packs_depths_into_large_objects() {
        // 1 + 4 + 16 + 64 + 256 = 341 ≤ 1024 inline; then 1024, 4096, 16384, 65536, 3000.
        let counts = [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 3000];
        let (inline, objects) = plan_objects(&counts, 1024, 65_536);
        assert_eq!(inline, 5);
        // Depths 5..=8 reach 87,040 ≥ 64K; the 3,000-node tail joins them.
        assert_eq!(objects, vec![0, 0, 0, 0, 0]);
        let (_, objects) = plan_objects(&[1, 70_000, 70_000, 10], 1024, 65_536);
        assert_eq!(objects, vec![0, 1, 1]);
        // The root depth is always inline, even above the cap.
        assert_eq!(plan_objects(&[1, 5000], 1, 10).0, 1);
    }

    #[test]
    fn tree_partitions_points_exactly() {
        let mut points: Vec<Point> = (0..5_000)
            .map(|i| {
                let x = i as f64;
                Point::new(x, (x * 0.37).sin() * 10.0 + (i % 7) as f64, 1.0 + (i % 3) as f64)
            })
            .collect();
        let total_mass: f64 = points.iter().map(|p| p.weight).sum();
        let depths = build_records(&mut points, 32, 32);

        let mut covered = vec![0u32; points.len()];
        for (d, level) in depths.iter().enumerate() {
            for record in level {
                match record.link() {
                    Link::Points { start, count } => {
                        assert!(count <= 32);
                        let run = &points[start as usize..(start + count) as usize];
                        assert!(run.windows(2).all(|w| w[0].x <= w[1].x), "leaf not x-sorted");
                        for c in &mut covered[start as usize..(start + count) as usize] {
                            *c += 1;
                        }
                    }
                    Link::Children { first, count } => {
                        assert!((1..=4).contains(&count));
                        assert!(((first + count) as usize) <= depths[d + 1].len());
                    }
                }
            }
        }
        assert!(covered.iter().all(|&c| c == 1), "every point in exactly one leaf");
        let root = depths[0][0].node.unpack();
        assert!((root.mass - total_mass).abs() / total_mass < 1e-6);
    }

    #[test]
    fn coincident_points_terminate() {
        let mut points = vec![Point::new(1.0, 1.0, 1.0); 100];
        points.push(Point::new(2.0, 2.0, 1.0));
        let depths = build_records(&mut points, 4, 32);
        assert!(depths.len() <= 3, "{} depths", depths.len());
    }
}
