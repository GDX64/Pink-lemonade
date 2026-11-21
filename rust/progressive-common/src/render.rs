//! Frontier resolution policies and frame assembly (Gaussian kernels + line overlay).

use std::collections::HashMap;
use std::sync::Arc;

use pink_lemonade_progressive_core::{Frame, Point, ViewQuery};

use crate::node::{Node, push_line_point};
use crate::reader::{Decision, FrontierItem};

/// Distance (in kernel standard deviations / σ) beyond which an off-screen node's merged
/// kernel provably contributes nothing on screen (matches the 6σ density cutoff).
const OFFSCREEN_MARGIN_SIGMA: f64 = 6.0;

/// Default policy: resolve a fully-inside node when it fits the merge threshold on screen or
/// lies entirely off-screen in y (explicit `y_bounds` only); otherwise expand.
pub fn standard_decision(node: &Node, query: &ViewQuery, x_factor: f64, y_factor: f64) -> Decision {
    if offscreen_y(node, query, y_factor)
        || node.fits_threshold(x_factor, y_factor, query.merge_threshold)
    {
        Decision::Resolve
    } else {
        Decision::Expand
    }
}

/// `true` when `query` has explicit `y_bounds` and `node` is far enough above or below the
/// screen that both its points (bbox ≥ 6σ away) and its merged Gaussian (mean ≥ 6 marginal
/// standard deviations away) are invisible, so it can be emitted as a single kernel
/// regardless of size without changing on-screen density. Moments stay exact either way.
pub fn offscreen_y(node: &Node, query: &ViewQuery, y_factor: f64) -> bool {
    let Some([lo, hi]) = query.y_bounds else {
        return false;
    };
    // Marginal kernel std-dev in σ units: sqrt(p11) = sqrt(1 + var_y · fy²).
    let spread = (1.0 + node.var_y() * y_factor * y_factor).sqrt();
    let (edge_gap, mean_gap) = if node.min_y > hi {
        (node.min_y - hi, node.y - hi)
    } else if node.max_y < lo {
        (lo - node.max_y, lo - node.y)
    } else {
        return false;
    };
    edge_gap * y_factor >= OFFSCREEN_MARGIN_SIGMA
        && mean_gap * y_factor >= OFFSCREEN_MARGIN_SIGMA * spread
}

/// Turn fully expanded frontier items into a [`Frame`].
///
/// - `Resolved` nodes and clipped raw leaves are greedily merged in item order (`push_group`)
///   and emit both kernels and line extrema.
/// - `Plane` items emit their kernels through `emit_plane` and their line extrema from the
///   item's exact node.
///
/// When `sort_lines` is set, line samples are stably sorted by x (required whenever item
/// order is not x-monotonic, e.g. k-d order or mixed plane/node output).
pub fn assemble_frame(
    items: Vec<FrontierItem>,
    leaves: &HashMap<u64, Arc<[Point]>>,
    query: &ViewQuery,
    (x_factor, y_factor): (f64, f64),
    y_range: [f64; 2],
    sort_lines: bool,
    emit_plane: &mut dyn FnMut(&FrontierItem, &mut Vec<f32>),
) -> Frame {
    let threshold = query.merge_threshold;
    let mut groups: Vec<Node> = Vec::with_capacity(items.len());
    let mut gpu_instances = Vec::with_capacity(items.len() * 7);
    let mut plane_line_nodes: Vec<Node> = Vec::new();

    for item in &items {
        match *item {
            FrontierItem::Resolved(node) => {
                push_group(node, x_factor, y_factor, threshold, &mut groups);
            }
            FrontierItem::Leaf(leaf_idx) => {
                if let Some(points) = leaves.get(&leaf_idx) {
                    // Leaves are x-sorted internally in every hierarchy order.
                    let start = points.partition_point(|p| p.x < query.min_x);
                    let end = points.partition_point(|p| p.x <= query.max_x);
                    if start < end {
                        collect_points(&points[start..end], x_factor, y_factor, threshold, &mut groups);
                    }
                }
            }
            FrontierItem::Plane { node, .. } => {
                emit_plane(item, &mut gpu_instances);
                plane_line_nodes.push(node);
            }
        }
    }

    let mut line_points = Vec::with_capacity((groups.len() + plane_line_nodes.len()) * 8);
    for node in &groups {
        node.emit_gpu(x_factor, y_factor, &mut gpu_instances);
        node.emit_line(&mut line_points);
    }
    for node in &plane_line_nodes {
        node.emit_line(&mut line_points);
    }
    if sort_lines {
        line_points = sort_line_points(&line_points);
    }

    Frame {
        gpu_instances,
        line_points,
        y_range: Some(y_range),
    }
}

/// Stable sort of flat `[x, y, ...]` pairs by x, dropping consecutive duplicates.
pub fn sort_line_points(flat: &[f32]) -> Vec<f32> {
    let mut pairs: Vec<[f32; 2]> = flat.as_chunks::<2>().0.to_vec();
    pairs.sort_by(|a, b| a[0].total_cmp(&b[0]));
    let mut out = Vec::with_capacity(flat.len());
    for [x, y] in pairs {
        push_line_point(&mut out, x as f64, y as f64);
    }
    out
}

/// Recursively halve an x-sorted run of raw points until each half fits the threshold.
pub fn collect_points(
    points: &[Point],
    x_factor: f64,
    y_factor: f64,
    threshold: f64,
    out: &mut Vec<Node>,
) {
    if points.is_empty() {
        return;
    }
    let node = Node::from_points(points);
    if points.len() == 1 || node.fits_threshold(x_factor, y_factor, threshold) {
        push_group(node, x_factor, y_factor, threshold, out);
    } else {
        let mid = points.len() / 2;
        collect_points(&points[..mid], x_factor, y_factor, threshold, out);
        collect_points(&points[mid..], x_factor, y_factor, threshold, out);
    }
}

/// Greedily merge `node` into the previous output group when the result still fits on screen.
pub fn push_group(node: Node, x_factor: f64, y_factor: f64, threshold: f64, out: &mut Vec<Node>) {
    if node.is_empty() {
        return;
    }
    if let Some(last) = out.last_mut() {
        let combined = Node::combined(*last, node);
        if combined.fits_merged_group(x_factor, y_factor, threshold) {
            *last = combined;
            return;
        }
    }
    out.push(node);
}
