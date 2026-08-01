//! Runnalls' Kullback-Leibler based Gaussian mixture reduction.
//!
//! A. R. Runnalls, "Kullback-Leibler Approach to Gaussian Mixture Reduction",
//! IEEE Trans. Aerospace and Electronic Systems, 43(3), 2007.
//!
//! Components are merged greedily in pairs, always choosing the pair whose
//! moment-preserving merge minimises an upper bound on the KL discrimination
//! between the mixture before and after merging. Used as a quality reference for
//! the screen-space merge: it is O(N^2) to set up and O(N) per merge, so it
//! cannot meet a per-frame budget, but it is order-independent and accounts for
//! component covariances.

use wasm_bindgen::prelude::*;

use crate::common::{lower_bound_triplets, upper_bound_triplets, MergeResult, Mixture, View};

impl Mixture {
    /// Runnalls' merging cost B(i,j): the increase in the KL upper bound caused
    /// by replacing i and j with their moment-preserving merge. Always >= 0, and
    /// exactly 0 for identical components.
    pub(crate) fn runnalls_cost(&self, i: usize, j: usize) -> f64 {
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
pub struct KlDownsampler {
    view: View,
    kl_threshold: f64,
    target_count: f64,
    data_f64: Vec<f64>,
}

impl Default for KlDownsampler {
    fn default() -> Self {
        Self {
            view: View::default(),
            kl_threshold: f64::INFINITY,
            target_count: f64::INFINITY,
            data_f64: Vec::new(),
        }
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
        self.view.sigma_size_px = value;
    }

    #[wasm_bindgen(js_name = setDataF64)]
    pub fn set_data_f64(&mut self, data: Box<[f64]>) {
        self.data_f64 = data.into_vec();
    }

    #[wasm_bindgen(js_name = mergePoints)]
    pub fn merge_points(&self) -> MergeResult {
        if self.data_f64.len() < 3 {
            return MergeResult::empty();
        }

        let start_idx = lower_bound_triplets(&self.data_f64, self.view.view_min_x);
        let end_idx = upper_bound_triplets(&self.data_f64, self.view.view_max_x);
        if end_idx <= start_idx {
            return MergeResult::empty();
        }

        let mut m = self.view.build_mixture(&self.data_f64, start_idx, end_idx);
        let alive = runnalls_reduce(&mut m, self.target_count, self.kl_threshold);
        self.view.pack_instances(&m, &alive)
    }
}

/// Greedy pairwise KL-optimal reduction of `m`, in place. Returns the liveness
/// mask over the original component slots. Stops when either the component
/// budget or the KL cost bound is reached.
pub(crate) fn runnalls_reduce(m: &mut Mixture, target_count: f64, kl_threshold: f64) -> Vec<bool> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{moments, unit_mixture};

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
                (
                    (f * 1.7).sin() * 5.0,
                    (f * 0.9).cos() * 3.0,
                    1.0 + (i % 7) as f64,
                )
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
                (
                    (f * 2.3).sin() * 4.0,
                    (f * 1.1).cos() * 4.0,
                    1.0 + (i % 3) as f64,
                )
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
                (
                    (f * 3.1).sin() * 6.0,
                    (f * 2.7).cos() * 6.0,
                    1.0 + (i % 4) as f64,
                )
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
