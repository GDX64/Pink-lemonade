//! Reusable hierarchy of consecutive x-sorted points for changing view ranges.
//!
//! Building the tree costs O(N) when data changes. A view visits only nodes
//! needed to cover its visible x interval at its current screen resolution.

use wasm_bindgen::prelude::*;

use crate::common::{MergeResult, View};
use crate::downsampler::Downsampler as DownsamplerTrait;

#[derive(Clone, Copy, Default)]
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

impl Node {
    fn combined(a: Self, b: Self) -> Self {
        if a.mass == 0.0 {
            return b;
        }
        if b.mass == 0.0 {
            return a;
        }
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
            min_x: a.min_x.min(b.min_x),
            max_x: a.max_x.max(b.max_x),
            min_y: a.min_y.min(b.min_y),
            max_y: a.max_y.max(b.max_y),
        }
    }
}

#[wasm_bindgen]
pub struct TreeDownsampler {
    view: View,
    merge_threshold: f64,
    xs: Vec<f64>,
    nodes: Vec<Node>,
    leaf_base: usize,
}

impl Default for TreeDownsampler {
    fn default() -> Self {
        Self {
            view: View::default(),
            merge_threshold: 1.0,
            xs: Vec::new(),
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
        let count = data.len() / 3;
        self.xs = Vec::with_capacity(count);
        self.leaf_base = count.max(1).next_power_of_two();
        self.nodes = vec![Node::default(); self.leaf_base * 2];
        for (index, point) in data.chunks_exact(3).enumerate() {
            let (x, y, mass) = (point[0], point[1], point[2]);
            self.xs.push(x);
            self.nodes[self.leaf_base + index] = Node {
                mass,
                x,
                y,
                min_x: x,
                max_x: x,
                min_y: y,
                max_y: y,
                ..Node::default()
            };
        }
        for index in (1..self.leaf_base).rev() {
            self.nodes[index] = Node::combined(self.nodes[index * 2], self.nodes[index * 2 + 1]);
        }
    }

    #[wasm_bindgen(js_name = mergePoints)]
    pub fn merge_points(&self) -> MergeResult {
        if self.xs.is_empty() {
            return MergeResult::empty();
        }
        let start = self.xs.partition_point(|&x| x < self.view.view_min_x);
        let end = self.xs.partition_point(|&x| x <= self.view.view_max_x);
        if start >= end {
            return MergeResult::empty();
        }
        let (x_factor, y_factor) = self.view.factors();
        let mut instances = Vec::new();
        self.collect(
            1,
            0,
            self.leaf_base,
            start,
            end,
            x_factor,
            y_factor,
            &mut instances,
        );
        MergeResult {
            count: (instances.len() / 7) as u32,
            gpu_instances: instances.into_boxed_slice(),
        }
    }
}

impl TreeDownsampler {
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
        out: &mut Vec<f32>,
    ) {
        if end <= low || high <= start {
            return;
        }
        let node = self.nodes[index];
        if node.mass == 0.0 {
            return;
        }
        let dx = (node.max_x - node.min_x) * x_factor;
        let dy = (node.max_y - node.min_y) * y_factor;
        let threshold = self.merge_threshold.max(1e-12);
        if high - low == 1
            || (start <= low && high <= end && dx * dx + dy * dy < threshold * threshold)
        {
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
