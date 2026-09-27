//! Reusable hierarchy of consecutive x-sorted points for changing view ranges.
//!
//! Building the tree costs O(N) when data changes. A view visits only nodes
//! needed to cover its visible x interval at its current screen resolution.

use wasm_bindgen::prelude::*;

use crate::common::{MergeResult, View, lower_bound_triplets, upper_bound_triplets};
use crate::downsampler::Downsampler as DownsamplerTrait;

const LEAF_POINTS: usize = 32;

#[derive(Clone, Copy)]
struct Node {
    mass: f64,
    x: f64,
    y: f64,
    m2_xx: f64,
    m2_xy: f64,
    m2_yy: f64,
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
}

impl Default for Node {
    fn default() -> Self {
        Self {
            mass: 0.0,
            x: 0.0,
            y: 0.0,
            m2_xx: 0.0,
            m2_xy: 0.0,
            m2_yy: 0.0,
            min_x: f64::INFINITY,
            max_x: f64::NEG_INFINITY,
            min_y: f64::INFINITY,
            max_y: f64::NEG_INFINITY,
        }
    }
}

impl Node {
    fn from_points(data: &[f64], start: usize, end: usize) -> Self {
        let mut node = Self::default();
        for point in data[start * 3..end * 3].chunks_exact(3) {
            let (x, y, mass) = (point[0], point[1], point[2]);
            node = Self::combined(
                node,
                Self {
                    mass,
                    x,
                    y,
                    min_x: x,
                    max_x: x,
                    min_y: y,
                    max_y: y,
                    ..Self::default()
                },
            );
        }
        node
    }

    fn combined(a: Self, b: Self) -> Self {
        if a.min_x > a.max_x {
            return b;
        }
        if b.min_x > b.max_x {
            return a;
        }
        let mut merged = if a.mass == 0.0 {
            b
        } else if b.mass == 0.0 {
            a
        } else {
            let mass = a.mass + b.mass;
            let fraction = b.mass / mass;
            let dx = b.x - a.x;
            let dy = b.y - a.y;
            let between = a.mass * b.mass / mass;
            Self {
                mass,
                x: a.x + fraction * dx,
                y: a.y + fraction * dy,
                m2_xx: a.m2_xx + b.m2_xx + between * dx * dx,
                m2_xy: a.m2_xy + b.m2_xy + between * dx * dy,
                m2_yy: a.m2_yy + b.m2_yy + between * dy * dy,
                ..Self::default()
            }
        };
        merged.min_x = a.min_x.min(b.min_x);
        merged.max_x = a.max_x.max(b.max_x);
        merged.min_y = a.min_y.min(b.min_y);
        merged.max_y = a.max_y.max(b.max_y);
        merged
    }
}

#[wasm_bindgen]
pub struct TreeDownsampler {
    view: View,
    merge_threshold: f64,
    data_f64: Vec<f64>,
    nodes: Vec<Node>,
    leaf_base: usize,
}

impl Default for TreeDownsampler {
    fn default() -> Self {
        Self {
            view: View::default(),
            merge_threshold: 1.0,
            data_f64: Vec::new(),
            nodes: Vec::new(),
            leaf_base: 0,
        }
    }
}

#[wasm_bindgen]
impl TreeDownsampler {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    #[wasm_bindgen(js_name = setViewMinX)]
    pub fn set_view_min_x(&mut self, value: f64) {
        self.view.view_min_x = value;
    }
    #[wasm_bindgen(js_name = setViewMaxX)]
    pub fn set_view_max_x(&mut self, value: f64) {
        self.view.view_max_x = value;
    }
    #[wasm_bindgen(js_name = setViewMinY)]
    pub fn set_view_min_y(&mut self, value: f64) {
        self.view.view_min_y = value;
    }
    #[wasm_bindgen(js_name = setViewMaxY)]
    pub fn set_view_max_y(&mut self, value: f64) {
        self.view.view_max_y = value;
    }
    #[wasm_bindgen(js_name = setScreenW)]
    pub fn set_screen_w(&mut self, value: f64) {
        self.view.screen_w = value;
    }
    #[wasm_bindgen(js_name = setScreenH)]
    pub fn set_screen_h(&mut self, value: f64) {
        self.view.screen_h = value;
    }
    #[wasm_bindgen(js_name = setMergeThreshold)]
    pub fn set_merge_threshold(&mut self, value: f64) {
        self.merge_threshold = value;
    }
    #[wasm_bindgen(js_name = setSigmaSizePx)]
    pub fn set_sigma_size_px(&mut self, value: f64) {
        self.view.sigma_size_px = value;
    }

    #[wasm_bindgen(js_name = setDataF64)]
    pub fn set_data_f64(&mut self, data: Box<[f64]>) {
        self.data_f64 = data.into_vec();
        let count = self.data_f64.len() / 3;
        let leaf_count = count.div_ceil(LEAF_POINTS);
        self.leaf_base = leaf_count.max(1).next_power_of_two();
        self.nodes = vec![Node::default(); self.leaf_base * 2];
        for index in 0..leaf_count {
            let start = index * LEAF_POINTS;
            let end = (start + LEAF_POINTS).min(count);
            self.nodes[self.leaf_base + index] = Node::from_points(&self.data_f64, start, end);
        }
        for index in (1..self.leaf_base).rev() {
            self.nodes[index] = Node::combined(self.nodes[index * 2], self.nodes[index * 2 + 1]);
        }
    }

    #[wasm_bindgen(js_name = mergePoints)]
    pub fn merge_points(&self) -> MergeResult {
        if self.data_f64.len() < 3 {
            return MergeResult::empty();
        }
        let start = lower_bound_triplets(&self.data_f64, self.view.view_min_x);
        let end = upper_bound_triplets(&self.data_f64, self.view.view_max_x);
        if start >= end {
            return MergeResult::empty();
        }
        let (x_factor, y_factor) = self.view.factors();
        let mut groups = Vec::new();
        self.collect(
            1,
            0,
            self.leaf_base,
            start,
            end,
            x_factor,
            y_factor,
            &mut groups,
        );
        let mut instances = Vec::with_capacity(groups.len() * 7);
        for node in groups {
            Self::emit(node, x_factor, y_factor, &mut instances);
        }
        MergeResult {
            count: (instances.len() / 7) as u32,
            gpu_instances: instances.into_boxed_slice(),
        }
    }

    /// Exact y extrema of points whose x lies in the inclusive view interval.
    /// An empty interval uses the full-data range, matching ViewManager.
    #[wasm_bindgen(js_name = visibleYRange)]
    pub fn visible_y_range(&self, min_x: f64, max_x: f64) -> Box<[f64]> {
        if self.data_f64.len() < 3 {
            return Box::new([f64::INFINITY, f64::NEG_INFINITY]);
        }
        let start = lower_bound_triplets(&self.data_f64, min_x);
        let end = upper_bound_triplets(&self.data_f64, max_x);
        if start >= end {
            let root = self.nodes[1];
            return Box::new([root.min_y, root.max_y]);
        }
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        self.y_range(1, 0, self.leaf_base, start, end, &mut min_y, &mut max_y);
        Box::new([min_y, max_y])
    }
}

impl TreeDownsampler {
    #[allow(clippy::too_many_arguments)]
    fn y_range(
        &self,
        index: usize,
        low: usize,
        high: usize,
        start: usize,
        end: usize,
        min_y: &mut f64,
        max_y: &mut f64,
    ) {
        let node_start = low * LEAF_POINTS;
        let node_end = (high * LEAF_POINTS).min(self.data_f64.len() / 3);
        if end <= node_start || node_end <= start {
            return;
        }
        if start <= node_start && node_end <= end {
            let node = self.nodes[index];
            *min_y = min_y.min(node.min_y);
            *max_y = max_y.max(node.max_y);
            return;
        }
        if high - low == 1 {
            for point in
                self.data_f64[start.max(node_start) * 3..end.min(node_end) * 3].chunks_exact(3)
            {
                *min_y = min_y.min(point[1]);
                *max_y = max_y.max(point[1]);
            }
            return;
        }
        let middle = (low + high) / 2;
        self.y_range(index * 2, low, middle, start, end, min_y, max_y);
        self.y_range(index * 2 + 1, middle, high, start, end, min_y, max_y);
    }

    #[allow(clippy::too_many_arguments)]
    fn collect(
        &self,
        index: usize,
        low: usize,
        high: usize,
        start: usize,
        end: usize,
        x_factor: f64,
        y_factor: f64,
        out: &mut Vec<Node>,
    ) {
        if end <= low * LEAF_POINTS || high * LEAF_POINTS <= start {
            return;
        }
        let node = self.nodes[index];
        if node.mass == 0.0 {
            return;
        }
        let node_end = (high * LEAF_POINTS).min(self.data_f64.len() / 3);
        if start <= low * LEAF_POINTS
            && node_end <= end
            && self.fits_threshold(node, x_factor, y_factor)
        {
            self.push_group(node, x_factor, y_factor, out);
            return;
        }
        if high - low == 1 {
            self.collect_points(
                start.max(low * LEAF_POINTS),
                end.min(node_end),
                x_factor,
                y_factor,
                out,
            );
            return;
        }
        let middle = (low + high) / 2;
        self.collect(index * 2, low, middle, start, end, x_factor, y_factor, out);
        self.collect(
            index * 2 + 1,
            middle,
            high,
            start,
            end,
            x_factor,
            y_factor,
            out,
        );
    }

    fn collect_points(
        &self,
        start: usize,
        end: usize,
        x_factor: f64,
        y_factor: f64,
        out: &mut Vec<Node>,
    ) {
        if start >= end {
            return;
        }
        let node = Node::from_points(&self.data_f64, start, end);
        if end - start == 1 || self.fits_threshold(node, x_factor, y_factor) {
            self.push_group(node, x_factor, y_factor, out);
        } else {
            let middle = (start + end) / 2;
            self.collect_points(start, middle, x_factor, y_factor, out);
            self.collect_points(middle, end, x_factor, y_factor, out);
        }
    }

    fn fits_threshold(&self, node: Node, x_factor: f64, y_factor: f64) -> bool {
        let dx = (node.max_x - node.min_x) * x_factor;
        let dy = (node.max_y - node.min_y) * y_factor;
        let threshold = self.merge_threshold.max(1e-12);
        dx * dx + dy * dy < threshold * threshold
    }

    fn fits_merged_group(&self, node: Node, x_factor: f64, y_factor: f64) -> bool {
        let dx = (node.x - node.min_x).max(node.max_x - node.x) * x_factor;
        let dy = (node.y - node.min_y).max(node.max_y - node.y) * y_factor;
        let threshold = self.merge_threshold.max(1e-12);
        dx * dx + dy * dy < threshold * threshold
    }

    fn push_group(&self, node: Node, x_factor: f64, y_factor: f64, out: &mut Vec<Node>) {
        // The tree's node boundaries are fixed. Coalesce each candidate with
        // the preceding group when every member remains near the new center.
        if let Some(last) = out.last_mut() {
            let combined = Node::combined(*last, node);
            if self.fits_merged_group(combined, x_factor, y_factor) {
                *last = combined;
                return;
            }
        }
        out.push(node);
    }

    fn emit(node: Node, x_factor: f64, y_factor: f64, out: &mut Vec<f32>) {
        let p00 = 1.0 + node.m2_xx / node.mass * x_factor * x_factor;
        let p01 = node.m2_xy / node.mass * x_factor * y_factor;
        let p11 = 1.0 + node.m2_yy / node.mass * y_factor * y_factor;
        let det = (p00 * p11 - p01 * p01).max(1e-12);
        let amplitude = node.mass / (2.0 * std::f64::consts::PI * det.sqrt());
        out.extend_from_slice(&[
            node.x as f32,
            node.y as f32,
            amplitude as f32,
            p00 as f32,
            p01 as f32,
            p01 as f32,
            p11 as f32,
        ]);
    }
}

impl DownsamplerTrait for TreeDownsampler {
    fn set_view_min_x(&mut self, value: f64) {
        Self::set_view_min_x(self, value);
    }
    fn set_view_max_x(&mut self, value: f64) {
        Self::set_view_max_x(self, value);
    }
    fn set_view_min_y(&mut self, value: f64) {
        Self::set_view_min_y(self, value);
    }
    fn set_view_max_y(&mut self, value: f64) {
        Self::set_view_max_y(self, value);
    }
    fn set_screen_w(&mut self, value: f64) {
        Self::set_screen_w(self, value);
    }
    fn set_screen_h(&mut self, value: f64) {
        Self::set_screen_h(self, value);
    }
    fn set_merge_threshold(&mut self, value: f64) {
        Self::set_merge_threshold(self, value);
    }
    fn set_sigma_size_px(&mut self, value: f64) {
        Self::set_sigma_size_px(self, value);
    }
    fn set_data_f64(&mut self, data: Box<[f64]>) {
        Self::set_data_f64(self, data);
    }
    fn merge_points(&self) -> MergeResult {
        Self::merge_points(self)
    }
}
