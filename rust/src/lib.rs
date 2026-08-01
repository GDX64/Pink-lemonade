use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct MergeResult {
    gpu_instances: Box<[f32]>,
    count: u32,
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

#[wasm_bindgen]
pub struct Downsampler {
    view_min_x: f64,
    view_max_x: f64,
    view_min_y: f64,
    view_max_y: f64,
    screen_w: f64,
    screen_h: f64,
    merge_threshold: f64,
    sigma_size_px: f64,
    data_f64: Vec<f64>,
}

impl Default for Downsampler {
    fn default() -> Self {
        Self {
            view_min_x: 0.0,
            view_max_x: 1.0,
            view_min_y: 0.0,
            view_max_y: 1.0,
            screen_w: 1.0,
            screen_h: 1.0,
            merge_threshold: 1.0,
            sigma_size_px: 1.0,
            data_f64: Vec::new(),
        }
    }
}

#[wasm_bindgen]
impl Downsampler {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Downsampler {
        Downsampler::default()
    }

    #[wasm_bindgen(js_name = setViewMinX)]
    pub fn set_view_min_x(&mut self, value: f64) {
        self.view_min_x = value;
    }

    #[wasm_bindgen(js_name = setViewMaxX)]
    pub fn set_view_max_x(&mut self, value: f64) {
        self.view_max_x = value;
    }

    #[wasm_bindgen(js_name = setViewMinY)]
    pub fn set_view_min_y(&mut self, value: f64) {
        self.view_min_y = value;
    }

    #[wasm_bindgen(js_name = setViewMaxY)]
    pub fn set_view_max_y(&mut self, value: f64) {
        self.view_max_y = value;
    }

    #[wasm_bindgen(js_name = setScreenW)]
    pub fn set_screen_w(&mut self, value: f64) {
        self.screen_w = value;
    }

    #[wasm_bindgen(js_name = setScreenH)]
    pub fn set_screen_h(&mut self, value: f64) {
        self.screen_h = value;
    }

    #[wasm_bindgen(js_name = setMergeThreshold)]
    pub fn set_merge_threshold(&mut self, value: f64) {
        self.merge_threshold = value;
    }

    #[wasm_bindgen(js_name = setSigmaSizePx)]
    pub fn set_sigma_size_px(&mut self, value: f64) {
        self.sigma_size_px = value;
    }

    #[wasm_bindgen(js_name = setDataF64)]
    pub fn set_data_f64(&mut self, data: Box<[f64]>) {
        self.data_f64 = data.into_vec();
    }

    #[wasm_bindgen(js_name = mergePoints)]
    pub fn merge_points(&self) -> MergeResult {
        if self.data_f64.len() < 3 {
            return MergeResult {
                gpu_instances: Vec::<f32>::new().into_boxed_slice(),
                count: 0,
            };
        }

        let x_den = (self.view_max_x - self.view_min_x).max(1e-12);
        let y_den = (self.view_max_y - self.view_min_y).max(1e-12);
        let sigma = self.sigma_size_px.max(1e-12);
        let x_factor = self.screen_w / x_den / sigma;
        let y_factor = self.screen_h / y_den / sigma;

        let to_sx = |x: f64| (x - self.view_min_x) * x_factor;
        let to_sy = |y: f64| (y - self.view_min_y) * y_factor;

        let start_idx = lower_bound_triplets(&self.data_f64, self.view_min_x);
        let end_idx = upper_bound_triplets(&self.data_f64, self.view_max_x);

        if end_idx <= start_idx {
            return MergeResult {
                gpu_instances: Vec::<f32>::new().into_boxed_slice(),
                count: 0,
            };
        }

        let merged = merge_pass(
            &self.data_f64,
            start_idx,
            end_idx,
            self.merge_threshold.max(1e-12),
            to_sx,
            to_sy,
        );

        let count = (merged.len() / 7) as u32;
        let mut gpu_instances = vec![0.0_f32; merged.len()];
        for (i, value) in merged.iter().enumerate() {
            gpu_instances[i] = *value as f32;
        }

        MergeResult {
            gpu_instances: gpu_instances.into_boxed_slice(),
            count,
        }
    }
}

/// Runnalls' Kullback-Leibler based Gaussian mixture reduction.
///
/// A. R. Runnalls, "Kullback-Leibler Approach to Gaussian Mixture Reduction",
/// IEEE Trans. Aerospace and Electronic Systems, 43(3), 2007.
///
/// Components are merged greedily in pairs, always choosing the pair whose
/// moment-preserving merge minimises an upper bound on the KL discrimination
/// between the mixture before and after merging. Used as a quality reference
/// for the screen-space merge: it is O(N^2) to set up and O(N) per merge, so it
/// cannot meet a per-frame budget, but it is order-independent and accounts for
/// component covariances.
#[wasm_bindgen]
pub struct KlDownsampler {
    view_min_x: f64,
    view_max_x: f64,
    view_min_y: f64,
    view_max_y: f64,
    screen_w: f64,
    screen_h: f64,
    kl_threshold: f64,
    target_count: f64,
    sigma_size_px: f64,
    data_f64: Vec<f64>,
}

impl Default for KlDownsampler {
    fn default() -> Self {
        Self {
            view_min_x: 0.0,
            view_max_x: 1.0,
            view_min_y: 0.0,
            view_max_y: 1.0,
            screen_w: 1.0,
            screen_h: 1.0,
            kl_threshold: f64::INFINITY,
            target_count: f64::INFINITY,
            sigma_size_px: 1.0,
            data_f64: Vec::new(),
        }
    }
}

/// Screen-space mixture in structure-of-arrays form. P is symmetric: p01 = p10.
struct Mixture {
    x: Vec<f64>,
    y: Vec<f64>,
    w: Vec<f64>,
    p00: Vec<f64>,
    p01: Vec<f64>,
    p11: Vec<f64>,
}

impl Mixture {
    fn len(&self) -> usize {
        self.x.len()
    }

    /// log det of the symmetric 2x2 covariance, guarded against degeneracy.
    fn log_det(&self, i: usize) -> f64 {
        let det = self.p00[i] * self.p11[i] - self.p01[i] * self.p01[i];
        det.max(1e-300).ln()
    }

    /// Moment-preserving merge of i and j: mass, mean and covariance of the
    /// two-component sub-mixture are preserved exactly.
    fn merge_pair_into(&mut self, i: usize, j: usize) {
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

    /// Runnalls' merging cost B(i,j): the increase in the KL upper bound caused
    /// by replacing i and j with their moment-preserving merge. Always >= 0,
    /// and exactly 0 for identical components.
    fn runnalls_cost(&self, i: usize, j: usize) -> f64 {
        let wi = self.w[i];
        let wj = self.w[j];
        let w = wi + wj;
        if w <= 0.0 {
            return 0.0;
        }
        let a = wi / w;
        let b = wj / w;
        let dx = self.x[i] - self.x[j];
        let dy = self.y[i] - self.y[j];
        let k = a * b;

        let p00 = a * self.p00[i] + b * self.p00[j] + k * dx * dx;
        let p01 = a * self.p01[i] + b * self.p01[j] + k * dx * dy;
        let p11 = a * self.p11[i] + b * self.p11[j] + k * dy * dy;
        let det = (p00 * p11 - p01 * p01).max(1e-300);

        0.5 * (w * det.ln() - wi * self.log_det(i) - wj * self.log_det(j))
    }
}

#[wasm_bindgen]
impl KlDownsampler {
    #[wasm_bindgen(constructor)]
    pub fn new() -> KlDownsampler {
        KlDownsampler::default()
    }

    #[wasm_bindgen(js_name = setViewMinX)]
    pub fn set_view_min_x(&mut self, value: f64) {
        self.view_min_x = value;
    }

    #[wasm_bindgen(js_name = setViewMaxX)]
    pub fn set_view_max_x(&mut self, value: f64) {
        self.view_max_x = value;
    }

    #[wasm_bindgen(js_name = setViewMinY)]
    pub fn set_view_min_y(&mut self, value: f64) {
        self.view_min_y = value;
    }

    #[wasm_bindgen(js_name = setViewMaxY)]
    pub fn set_view_max_y(&mut self, value: f64) {
        self.view_max_y = value;
    }

    #[wasm_bindgen(js_name = setScreenW)]
    pub fn set_screen_w(&mut self, value: f64) {
        self.screen_w = value;
    }

    #[wasm_bindgen(js_name = setScreenH)]
    pub fn set_screen_h(&mut self, value: f64) {
        self.screen_h = value;
    }

    /// KL cost bound: merging stops once the cheapest merge exceeds it. Named
    /// for parity with `Downsampler`, but it is not a pixel radius --- the two
    /// algorithms are steered by different quantities.
    #[wasm_bindgen(js_name = setMergeThreshold)]
    pub fn set_merge_threshold(&mut self, value: f64) {
        self.kl_threshold = value;
    }

    /// Component budget: merging stops once the mixture is this small.
    #[wasm_bindgen(js_name = setTargetCount)]
    pub fn set_target_count(&mut self, value: f64) {
        self.target_count = value;
    }

    #[wasm_bindgen(js_name = setSigmaSizePx)]
    pub fn set_sigma_size_px(&mut self, value: f64) {
        self.sigma_size_px = value;
    }

    #[wasm_bindgen(js_name = setDataF64)]
    pub fn set_data_f64(&mut self, data: Box<[f64]>) {
        self.data_f64 = data.into_vec();
    }

    #[wasm_bindgen(js_name = mergePoints)]
    pub fn merge_points(&self) -> MergeResult {
        let empty = MergeResult {
            gpu_instances: Vec::<f32>::new().into_boxed_slice(),
            count: 0,
        };

        if self.data_f64.len() < 3 {
            return empty;
        }

        let x_den = (self.view_max_x - self.view_min_x).max(1e-12);
        let y_den = (self.view_max_y - self.view_min_y).max(1e-12);
        let sigma = self.sigma_size_px.max(1e-12);
        let x_factor = self.screen_w / x_den / sigma;
        let y_factor = self.screen_h / y_den / sigma;

        let start_idx = lower_bound_triplets(&self.data_f64, self.view_min_x);
        let end_idx = upper_bound_triplets(&self.data_f64, self.view_max_x);
        if end_idx <= start_idx {
            return empty;
        }

        // One unit-covariance kernel per visible sample, in screen space.
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
            m.x.push((self.data_f64[base] - self.view_min_x) * x_factor);
            m.y.push((self.data_f64[base + 1] - self.view_min_y) * y_factor);
            m.w.push(self.data_f64[base + 2]);
        }

        let alive = runnalls_reduce(&mut m, self.target_count, self.kl_threshold);

        // Back to data space, with the amplitude convention used by the GPU path.
        let mut gpu_instances = Vec::<f32>::with_capacity(n * 7);
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

/// Greedy pairwise KL-optimal reduction of `m`, in place. Returns the liveness
/// mask over the original component slots. Stops when either the component
/// budget or the KL cost bound is reached.
fn runnalls_reduce(m: &mut Mixture, target_count: f64, kl_threshold: f64) -> Vec<bool> {
    let n = m.len();
    let mut alive = vec![true; n];
    if n == 0 {
        return alive;
    }

    let mut best = vec![usize::MAX; n];
    let mut best_cost = vec![f64::INFINITY; n];
    let mut live = n as f64;

    // Cheapest partner for `i` among the live components.
    fn recompute_best(
        m: &Mixture,
        alive: &[bool],
        best: &mut [usize],
        best_cost: &mut [f64],
        i: usize,
    ) {
        let mut bi = usize::MAX;
        let mut bc = f64::INFINITY;
        for j in 0..m.len() {
            if j == i || !alive[j] {
                continue;
            }
            let c = m.runnalls_cost(i, j);
            if c < bc {
                bc = c;
                bi = j;
            }
        }
        best[i] = bi;
        best_cost[i] = bc;
    }

    if n > 1 {
        for i in 0..n {
            recompute_best(m, &alive, &mut best, &mut best_cost, i);
        }
    }

    while live > target_count && live > 1.0 {
        // Global minimum-cost pair.
        let mut a = usize::MAX;
        let mut ac = f64::INFINITY;
        for i in 0..n {
            if alive[i] && best_cost[i] < ac {
                ac = best_cost[i];
                a = i;
            }
        }
        if a == usize::MAX || !ac.is_finite() || ac > kl_threshold {
            break;
        }

        let b = best[a];
        m.merge_pair_into(a, b);
        alive[b] = false;
        live -= 1.0;

        if live <= 1.0 {
            break;
        }

        // `a` changed, so every cost involving `a` is stale.
        recompute_best(m, &alive, &mut best, &mut best_cost, a);
        for k in 0..n {
            if !alive[k] || k == a {
                continue;
            }
            let c = m.runnalls_cost(k, a);
            if c < best_cost[k] {
                best_cost[k] = c;
                best[k] = a;
            } else if best[k] == a || best[k] == b {
                // Previous partner is gone or got more expensive; rescan.
                recompute_best(m, &alive, &mut best, &mut best_cost, k);
            }
        }
    }

    alive
}

fn lower_bound_triplets(data: &[f64], target_x: f64) -> usize {
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

fn upper_bound_triplets(data: &[f64], target_x: f64) -> usize {
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

fn merge_pass<FSX, FSY>(
    pts: &[f64],
    start_idx: usize,
    end_idx: usize,
    threshold_px: f64,
    to_sx: FSX,
    to_sy: FSY,
) -> Vec<f64>
where
    FSX: Fn(f64) -> f64,
    FSY: Fn(f64) -> f64,
{
    let mut out = Vec::<f64>::with_capacity((end_idx - start_idx) * 7);
    let t2 = threshold_px * threshold_px;

    let mut has_cluster = false;
    let mut first_x = 0.0;
    let mut first_y = 0.0;
    let mut cluster_x = 0.0;
    let mut cluster_y = 0.0;
    let mut cluster_w = 0.0;
    let mut cluster_start = 0usize;
    let mut cluster_end = 0usize;

    let flush_cluster = |out: &mut Vec<f64>,
                         cluster_x: f64,
                         cluster_y: f64,
                         cluster_w: f64,
                         cluster_start: usize,
                         cluster_end: usize| {
        let cx = to_sx(cluster_x);
        let cy = to_sy(cluster_y);
        let mut weighted_xx = 0.0;
        let mut weighted_xy = 0.0;
        let mut weighted_yy = 0.0;

        for i in cluster_start..=cluster_end {
            let base = i * 3;
            let x = pts[base];
            let y = pts[base + 1];
            let w = pts[base + 2];
            let dx = to_sx(x) - cx;
            let dy = to_sy(y) - cy;
            weighted_xx += w * dx * dx;
            weighted_xy += w * dx * dy;
            weighted_yy += w * dy * dy;
        }

        let inv_w = 1.0 / cluster_w.max(1e-12);
        let p00 = 1.0 + weighted_xx * inv_w;
        let p01 = weighted_xy * inv_w;
        let p10 = p01;
        let p11 = 1.0 + weighted_yy * inv_w;
        let det_of_p = (p00 * p11 - p01 * p10).max(1e-12);
        let w = cluster_w / (2.0 * std::f64::consts::PI * det_of_p.sqrt());

        out.push(cluster_x);
        out.push(cluster_y);
        out.push(w);
        out.push(p00);
        out.push(p01);
        out.push(p10);
        out.push(p11);
    };

    for i in start_idx..end_idx {
        let base = i * 3;
        let xi = pts[base];
        let yi = pts[base + 1];
        let wi = pts[base + 2];

        if !has_cluster {
            has_cluster = true;
            first_x = xi;
            first_y = yi;
            cluster_x = xi;
            cluster_y = yi;
            cluster_w = wi;
            cluster_start = i;
            cluster_end = i;
            continue;
        }

        let new_w = cluster_w + wi;
        let new_x = (cluster_x * cluster_w + xi * wi) / new_w;
        let new_y = (cluster_y * cluster_w + yi * wi) / new_w;
        let new_sx = to_sx(new_x);
        let new_sy = to_sy(new_y);

        let dfx = to_sx(first_x) - new_sx;
        let dfy = to_sy(first_y) - new_sy;
        let dix = to_sx(xi) - new_sx;
        let diy = to_sy(yi) - new_sy;

        if dfx * dfx + dfy * dfy < t2 && dix * dix + diy * diy < t2 {
            cluster_x = new_x;
            cluster_y = new_y;
            cluster_w = new_w;
            cluster_end = i;
        } else {
            flush_cluster(
                &mut out,
                cluster_x,
                cluster_y,
                cluster_w,
                cluster_start,
                cluster_end,
            );

            has_cluster = true;
            first_x = xi;
            first_y = yi;
            cluster_x = xi;
            cluster_y = yi;
            cluster_w = wi;
            cluster_start = i;
            cluster_end = i;
        }
    }

    if has_cluster {
        flush_cluster(
            &mut out,
            cluster_x,
            cluster_y,
            cluster_w,
            cluster_start,
            cluster_end,
        );
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_mixture(points: &[(f64, f64, f64)]) -> Mixture {
        Mixture {
            x: points.iter().map(|p| p.0).collect(),
            y: points.iter().map(|p| p.1).collect(),
            w: points.iter().map(|p| p.2).collect(),
            p00: vec![1.0; points.len()],
            p01: vec![0.0; points.len()],
            p11: vec![1.0; points.len()],
        }
    }

    /// Mass, mean and covariance of the whole mixture: the merge invariant.
    fn moments(m: &Mixture, alive: &[bool]) -> (f64, f64, f64, f64, f64, f64) {
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

    #[test]
    fn cost_is_zero_for_identical_components() {
        let m = unit_mixture(&[(3.0, -2.0, 5.0), (3.0, -2.0, 7.0)]);
        assert!(m.runnalls_cost(0, 1).abs() < 1e-12);
    }

    #[test]
    fn cost_is_symmetric() {
        let mut m = unit_mixture(&[(0.0, 0.0, 2.0), (1.5, -0.5, 5.0)]);
        m.p00[0] = 1.0;
        m.p01[0] = 0.3;
        m.p11[0] = 2.0;
        m.p00[1] = 3.0;
        m.p01[1] = -0.4;
        m.p11[1] = 1.0;
        assert!((m.runnalls_cost(0, 1) - m.runnalls_cost(1, 0)).abs() < 1e-12);
    }

    #[test]
    fn cost_matches_closed_form_for_equal_unit_kernels() {
        // wi = wj = 1, Pi = Pj = I, separation d along x:
        //   Pij = I + diag(d^2/4, 0), so B = 1/2 * 2 * log(1 + d^2/4).
        for d in [0.5, 1.0, 3.0] {
            let m = unit_mixture(&[(0.0, 0.0, 1.0), (d, 0.0, 1.0)]);
            let expected = (1.0 + d * d / 4.0).ln();
            assert!((m.runnalls_cost(0, 1) - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn cost_grows_with_separation() {
        let mut previous = -1.0;
        for d in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0] {
            let m = unit_mixture(&[(0.0, 0.0, 1.0), (d, 0.0, 1.0)]);
            let cost = m.runnalls_cost(0, 1);
            assert!(cost >= 0.0 && cost > previous);
            previous = cost;
        }
    }

    #[test]
    fn full_reduction_preserves_moments() {
        let points: Vec<(f64, f64, f64)> = (0..40)
            .map(|i| {
                let f = i as f64;
                ((f * 1.7).sin() * 5.0, (f * 0.9).cos() * 3.0, 1.0 + (i % 7) as f64)
            })
            .collect();

        let mut m = unit_mixture(&points);
        let expected = moments(&m, &vec![true; points.len()]);

        let alive = runnalls_reduce(&mut m, 1.0, f64::INFINITY);
        let live: Vec<usize> = (0..m.len()).filter(|i| alive[*i]).collect();
        assert_eq!(live.len(), 1);

        let i = live[0];
        assert!((m.w[i] - expected.0).abs() < 1e-9);
        assert!((m.x[i] - expected.1).abs() < 1e-9);
        assert!((m.y[i] - expected.2).abs() < 1e-9);
        assert!((m.p00[i] - expected.3).abs() < 1e-9);
        assert!((m.p01[i] - expected.4).abs() < 1e-9);
        assert!((m.p11[i] - expected.5).abs() < 1e-9);
    }

    #[test]
    fn conserves_total_weight_at_every_budget() {
        let points: Vec<(f64, f64, f64)> = (0..60)
            .map(|i| (i as f64 * 0.13, (i % 5) as f64 * 0.4, (i + 1) as f64))
            .collect();
        let total: f64 = points.iter().map(|p| p.2).sum();

        for target in [1.0, 3.0, 10.0, 25.0, 60.0] {
            let mut m = unit_mixture(&points);
            let alive = runnalls_reduce(&mut m, target, f64::INFINITY);
            let live = alive.iter().filter(|a| **a).count();
            assert_eq!(live, target as usize);

            let sum: f64 = (0..m.len()).filter(|i| alive[*i]).map(|i| m.w[i]).sum();
            assert!((sum - total).abs() < 1e-9);
        }
    }

    #[test]
    fn merges_the_closest_components_first() {
        // Three tight pairs, far apart. Reducing 6 -> 3 must recover the pairs.
        let mut m = unit_mixture(&[
            (0.0, 0.0, 1.0),
            (0.1, 0.0, 1.0),
            (50.0, 0.0, 1.0),
            (50.1, 0.0, 1.0),
            (0.0, 80.0, 1.0),
            (0.1, 80.0, 1.0),
        ]);
        let alive = runnalls_reduce(&mut m, 3.0, f64::INFINITY);

        let mut centers: Vec<String> = (0..m.len())
            .filter(|i| alive[*i])
            .map(|i| format!("{:.2},{:.2}", m.x[i], m.y[i]))
            .collect();
        centers.sort();
        assert_eq!(centers, vec!["0.05,0.00", "0.05,80.00", "50.05,0.00"]);
    }

    #[test]
    fn is_independent_of_input_ordering() {
        let points: Vec<(f64, f64, f64)> = (0..30)
            .map(|i| {
                let f = i as f64;
                ((f * 2.3).sin() * 4.0, (f * 1.1).cos() * 4.0, 1.0 + (i % 3) as f64)
            })
            .collect();
        let reversed: Vec<(f64, f64, f64)> = points.iter().rev().copied().collect();

        let key = |pts: &[(f64, f64, f64)]| {
            let mut m = unit_mixture(pts);
            let alive = runnalls_reduce(&mut m, 5.0, f64::INFINITY);
            let mut k: Vec<String> = (0..m.len())
                .filter(|i| alive[*i])
                .map(|i| format!("{:.6}|{:.6}|{:.6}", m.x[i], m.y[i], m.w[i]))
                .collect();
            k.sort();
            k
        };

        assert_eq!(key(&points), key(&reversed));
    }

    #[test]
    fn stops_on_the_kl_threshold_before_the_budget() {
        let mut m = unit_mixture(&[(0.0, 0.0, 1.0), (1000.0, 0.0, 1.0)]);
        let alive = runnalls_reduce(&mut m, 1.0, 1e-6);
        assert_eq!(alive.iter().filter(|a| **a).count(), 2);
    }

    #[test]
    fn merge_costs_are_monotonically_non_decreasing() {
        // Runnalls / Salmond: minimum merge cost rises as reduction proceeds.
        let points: Vec<(f64, f64, f64)> = (0..50)
            .map(|i| {
                let f = i as f64;
                ((f * 3.1).sin() * 6.0, (f * 2.7).cos() * 6.0, 1.0 + (i % 4) as f64)
            })
            .collect();

        let mut previous = 0.0;
        for target in (1..50).rev() {
            let mut m = unit_mixture(&points);
            let alive = runnalls_reduce(&mut m, target as f64, f64::INFINITY);
            // Cheapest remaining merge is a lower bound on the next cost.
            let live: Vec<usize> = (0..m.len()).filter(|i| alive[*i]).collect();
            let mut cheapest = f64::INFINITY;
            for a in 0..live.len() {
                for b in (a + 1)..live.len() {
                    cheapest = cheapest.min(m.runnalls_cost(live[a], live[b]));
                }
            }
            assert!(cheapest >= previous - 1e-9);
            previous = cheapest;
        }
    }
}
