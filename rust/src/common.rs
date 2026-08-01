//! Pieces shared by every downsampling strategy: the value returned to JS, the
//! packed-triplet input helpers, and the screen-space mixture the two Gaussian
//! mixture reduction methods operate on.

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct MergeResult {
    pub(crate) gpu_instances: Box<[f32]>,
    pub(crate) count: u32,
}

#[wasm_bindgen]
impl MergeResult {
    #[wasm_bindgen(getter, js_name = gpuInstances)]
    pub fn gpu_instances(&self) -> Box<[f32]> {
        self.gpu_instances.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn count(&self) -> u32 {
        self.count
    }
}

impl MergeResult {
    pub(crate) fn empty() -> MergeResult {
        MergeResult {
            gpu_instances: Vec::<f32>::new().into_boxed_slice(),
            count: 0,
        }
    }
}

/// Screen-space mixture in structure-of-arrays form. P is symmetric: p01 = p10.
#[derive(Clone)]
pub(crate) struct Mixture {
    pub(crate) x: Vec<f64>,
    pub(crate) y: Vec<f64>,
    pub(crate) w: Vec<f64>,
    pub(crate) p00: Vec<f64>,
    pub(crate) p01: Vec<f64>,
    pub(crate) p11: Vec<f64>,
}

impl Mixture {
    pub(crate) fn len(&self) -> usize {
        self.x.len()
    }

    /// log det of the symmetric 2x2 covariance, guarded against degeneracy.
    pub(crate) fn log_det(&self, i: usize) -> f64 {
        let det = self.p00[i] * self.p11[i] - self.p01[i] * self.p01[i];
        det.max(1e-300).ln()
    }

    /// Moment-preserving merge of i and j into i: mass, mean and covariance of
    /// the two-component sub-mixture are preserved exactly. Applying this
    /// repeatedly merges a whole group, since the operation is associative.
    pub(crate) fn merge_pair_into(&mut self, i: usize, j: usize) {
        let wi = self.w[i];
        let wj = self.w[j];
        let w = wi + wj;
        let a = wi / w;
        let b = wj / w;
        let dx = self.x[i] - self.x[j];
        let dy = self.y[i] - self.y[j];
        let k = a * b;

        self.x[i] = a * self.x[i] + b * self.x[j];
        self.y[i] = a * self.y[i] + b * self.y[j];
        self.p00[i] = a * self.p00[i] + b * self.p00[j] + k * dx * dx;
        self.p01[i] = a * self.p01[i] + b * self.p01[j] + k * dx * dy;
        self.p11[i] = a * self.p11[i] + b * self.p11[j] + k * dy * dy;
        self.w[i] = w;
    }

    pub(crate) fn total_weight(&self, alive: &[bool]) -> f64 {
        (0..self.len())
            .filter(|i| alive[*i])
            .map(|i| self.w[i])
            .sum()
    }
}

/// The view transform every downsampler is configured with. Shared so the three
/// `#[wasm_bindgen]` classes present exactly the same setters to JS.
pub(crate) struct View {
    pub(crate) view_min_x: f64,
    pub(crate) view_max_x: f64,
    pub(crate) view_min_y: f64,
    pub(crate) view_max_y: f64,
    pub(crate) screen_w: f64,
    pub(crate) screen_h: f64,
    pub(crate) sigma_size_px: f64,
}

impl Default for View {
    fn default() -> Self {
        Self {
            view_min_x: 0.0,
            view_max_x: 1.0,
            view_min_y: 0.0,
            view_max_y: 1.0,
            screen_w: 1.0,
            screen_h: 1.0,
            sigma_size_px: 1.0,
        }
    }
}

impl View {
    /// Data units -> sigma-normalised screen units, per axis.
    pub(crate) fn factors(&self) -> (f64, f64) {
        let x_den = (self.view_max_x - self.view_min_x).max(1e-12);
        let y_den = (self.view_max_y - self.view_min_y).max(1e-12);
        let sigma = self.sigma_size_px.max(1e-12);
        (
            self.screen_w / x_den / sigma,
            self.screen_h / y_den / sigma,
        )
    }

    /// One unit-covariance kernel per visible sample, in screen space. This is
    /// the mixture both Runnalls and Salmond start from, so the two methods see
    /// byte-identical input.
    pub(crate) fn build_mixture(&self, data: &[f64], start_idx: usize, end_idx: usize) -> Mixture {
        let (x_factor, y_factor) = self.factors();
        let n = end_idx - start_idx;
        let mut m = Mixture {
            x: Vec::with_capacity(n),
            y: Vec::with_capacity(n),
            w: Vec::with_capacity(n),
            p00: vec![1.0; n],
            p01: vec![0.0; n],
            p11: vec![1.0; n],
        };
        for i in start_idx..end_idx {
            let base = i * 3;
            m.x.push((data[base] - self.view_min_x) * x_factor);
            m.y.push((data[base + 1] - self.view_min_y) * y_factor);
            m.w.push(data[base + 2]);
        }
        m
    }

    /// Back to data space, with the amplitude convention used by the GPU path:
    /// A = w / (2*pi*sqrt(det P)).
    pub(crate) fn pack_instances(&self, m: &Mixture, alive: &[bool]) -> MergeResult {
        let (x_factor, y_factor) = self.factors();
        let mut gpu_instances = Vec::<f32>::with_capacity(m.len() * 7);
        let mut count = 0u32;
        for i in 0..m.len() {
            if !alive[i] {
                continue;
            }
            let p00 = m.p00[i];
            let p01 = m.p01[i];
            let p11 = m.p11[i];
            let det = (p00 * p11 - p01 * p01).max(1e-12);
            let amplitude = m.w[i] / (2.0 * std::f64::consts::PI * det.sqrt());

            gpu_instances.push((self.view_min_x + m.x[i] / x_factor) as f32);
            gpu_instances.push((self.view_min_y + m.y[i] / y_factor) as f32);
            gpu_instances.push(amplitude as f32);
            gpu_instances.push(p00 as f32);
            gpu_instances.push(p01 as f32);
            gpu_instances.push(p01 as f32);
            gpu_instances.push(p11 as f32);
            count += 1;
        }

        MergeResult {
            gpu_instances: gpu_instances.into_boxed_slice(),
            count,
        }
    }
}

pub(crate) fn lower_bound_triplets(data: &[f64], target_x: f64) -> usize {
    let n = data.len() / 3;
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let x = data[mid * 3];
        if x < target_x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

pub(crate) fn upper_bound_triplets(data: &[f64], target_x: f64) -> usize {
    let n = data.len() / 3;
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let x = data[mid * 3];
        if x <= target_x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[cfg(test)]
pub(crate) fn unit_mixture(points: &[(f64, f64, f64)]) -> Mixture {
    Mixture {
        x: points.iter().map(|p| p.0).collect(),
        y: points.iter().map(|p| p.1).collect(),
        w: points.iter().map(|p| p.2).collect(),
        p00: vec![1.0; points.len()],
        p01: vec![0.0; points.len()],
        p11: vec![1.0; points.len()],
    }
}

/// Mass, mean and covariance of the whole mixture: the invariant every
/// moment-preserving reduction must hold, whatever path it takes.
#[cfg(test)]
pub(crate) fn moments(m: &Mixture, alive: &[bool]) -> (f64, f64, f64, f64, f64, f64) {
    let mut w = 0.0;
    let mut mx = 0.0;
    let mut my = 0.0;
    for i in 0..m.len() {
        if !alive[i] {
            continue;
        }
        w += m.w[i];
        mx += m.w[i] * m.x[i];
        my += m.w[i] * m.y[i];
    }
    mx /= w;
    my /= w;

    let (mut p00, mut p01, mut p11) = (0.0, 0.0, 0.0);
    for i in 0..m.len() {
        if !alive[i] {
            continue;
        }
        let dx = m.x[i] - mx;
        let dy = m.y[i] - my;
        p00 += m.w[i] * (m.p00[i] + dx * dx);
        p01 += m.w[i] * (m.p01[i] + dx * dy);
        p11 += m.w[i] * (m.p11[i] + dy * dy);
    }
    (w, mx, my, p00 / w, p01 / w, p11 / w)
}
