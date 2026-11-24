//! Baseline: binary moment x-tree with today's bbox resolve rule (`standard_decision`).
//!
//! Same grouping (x-runs halved down to 32-point leaves) and the same rule as the paged
//! strategies, without pages or a network: a view's cost is the number of nodes the client must
//! read (every child of an expanded node) plus the raw points of the leaves it opens.

use pink_lemonade_progressive_common::{
    Decision, Node, collect_points, push_group, standard_decision,
};
use pink_lemonade_progressive_core::{Frame, Point, ViewQuery};

/// Bytes charged per tree node: `f32` bbox (16 B) + 16-byte quantized moments.
pub const TREE_NODE_BYTES: usize = 32;
/// Bytes charged per raw point (`f64` x, y, weight), as in `points.bin`.
pub const RAW_POINT_BYTES: usize = 24;
pub const LEAF_POINTS: usize = 32;

const NO_CHILD: u32 = u32::MAX;

struct TreeNode {
    node: Node,
    start: u32,
    end: u32,
    left: u32,
    right: u32,
}

pub struct XTree {
    nodes: Vec<TreeNode>,
    points: Vec<Point>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct XTreeStats {
    pub nodes_read: usize,
    pub raw_points: usize,
    pub bytes: usize,
}

impl XTree {
    /// Build over x-sorted points.
    pub fn build(points: Vec<Point>) -> Self {
        let mut tree = Self {
            nodes: Vec::with_capacity(2 * points.len() / LEAF_POINTS + 1),
            points,
        };
        if !tree.points.is_empty() {
            tree.build_range(0, tree.points.len());
        }
        tree
    }

    fn build_range(&mut self, start: usize, end: usize) -> u32 {
        let idx = self.nodes.len() as u32;
        self.nodes.push(TreeNode {
            node: Node::default(),
            start: start as u32,
            end: end as u32,
            left: NO_CHILD,
            right: NO_CHILD,
        });
        let node = if end - start <= LEAF_POINTS {
            Node::from_points(&self.points[start..end])
        } else {
            let mid = start + (end - start) / 2;
            let left = self.build_range(start, mid);
            let right = self.build_range(mid, end);
            self.nodes[idx as usize].left = left;
            self.nodes[idx as usize].right = right;
            Node::combined(
                self.nodes[left as usize].node,
                self.nodes[right as usize].node,
            )
        };
        self.nodes[idx as usize].node = node;
        idx
    }

    pub fn render(&self, query: &ViewQuery, y_range: [f64; 2]) -> (Frame, XTreeStats) {
        let bounds = query.effective_y_bounds(y_range);
        let (fx, fy) = query.factors(bounds);
        let mut stats = XTreeStats::default();
        let mut groups: Vec<Node> = Vec::new();
        let mut stack = if self.nodes.is_empty() {
            vec![]
        } else {
            vec![0u32]
        };
        while let Some(idx) = stack.pop() {
            stats.nodes_read += 1;
            let t = &self.nodes[idx as usize];
            let n = &t.node;
            if n.max_x < query.min_x || n.min_x > query.max_x {
                continue;
            }
            let inside = n.min_x >= query.min_x && n.max_x <= query.max_x;
            if inside && matches!(standard_decision(n, query, fx, fy), Decision::Resolve) {
                push_group(*n, fx, fy, query.merge_threshold, &mut groups);
            } else if t.left == NO_CHILD {
                let leaf = &self.points[t.start as usize..t.end as usize];
                stats.raw_points += leaf.len();
                let a = leaf.partition_point(|p| p.x < query.min_x);
                let b = leaf.partition_point(|p| p.x <= query.max_x);
                collect_points(&leaf[a..b], fx, fy, query.merge_threshold, &mut groups);
            } else {
                // Children are pushed right-first so they pop in x order (for `push_group`).
                stack.push(t.right);
                stack.push(t.left);
            }
        }
        stats.bytes = stats.nodes_read * TREE_NODE_BYTES + stats.raw_points * RAW_POINT_BYTES;
        let mut gpu_instances = Vec::with_capacity(groups.len() * 7);
        for g in &groups {
            g.emit_gpu(fx, fy, &mut gpu_instances);
        }
        let frame = Frame {
            gpu_instances,
            line_points: Vec::new(),
            y_range: Some(y_range),
        };
        (frame, stats)
    }
}
