//! Salmond's clustering algorithm (the "CAF" reduction).
//!
//! D. J. Salmond, "Mixture Reduction Algorithms for Point and Extended Object
//! Tracking in Clutter", IEEE Trans. Aerospace and Electronic Systems, 45(2),
//! 2009 (revising the 1990 conference paper).
//!
//! Unlike Runnalls, components are combined in *groups* rather than pairs. The
//! heaviest remaining component becomes a cluster centre and absorbs every
//! component that falls inside
//!
//! ```text
//! D_i^2 = (b_i b_c / (b_i + b_c)) (x_i - x_c)^T Pc^-1 (x_i - x_c) < T1
//! ```
//!
//! where b are normalised probability masses and Pc is the covariance of the
//! *principal* component (not of the whole mixture --- that is the only formal
//! difference from the joining algorithm's d_ij^2). The b_i b_c / (b_i + b_c)
//! factor biases the measure so light components are absorbed readily while
//! heavy ones keep their identity.
//!
//! Reduction proceeds one whole pass at a time, so it is far cheaper than
//! Runnalls' O(N) rescan per merge --- but it cannot hit an exact component
//! budget: a pass may absorb more than strictly necessary. `merge_points` may
//! therefore return fewer than `target_count` kernels.

use wasm_bindgen::prelude::*;

use crate::common::{lower_bound_triplets, upper_bound_triplets, MergeResult, Mixture, View};

/// Squared Mahalanobis radius of the 2D ellipse holding 1% of a Gaussian's mass:
/// -2 ln(0.99). The paper suggests T1 = 0.05 * that.
const R2_1_PERCENT: f64 = 0.020100671707002903;
/// Same for 6%: -2 ln(0.94). The paper suggests dT = 0.05 * (R2_6% - R2_1%).
const R2_6_PERCENT: f64 = 0.12375080743617465;

pub(crate) const DEFAULT_T1: f64 = 0.05 * R2_1_PERCENT;
pub(crate) const DEFAULT_DELTA_T: f64 = 0.05 * (R2_6_PERCENT - R2_1_PERCENT);

/// Runaway guard. Each pass at least doubles the increment when it makes no
/// progress, so the threshold grows geometrically and this is never reached in
/// practice.
const MAX_PASSES: usize = 4_096;

/// A budgeted run accepts a pass that lands at or above this fraction of NT;
/// below it, the threshold is bisected back to trim the overshoot. See
/// [`salmond_cluster_reduce`] for why this exists.
const BUDGET_TOLERANCE: f64 = 0.95;

#[wasm_bindgen]
pub struct SalmondDownsampler {
    view: View,
    t1: f64,
    delta_t: f64,
    target_count: f64,
    data_f64: Vec<f64>,
}

impl Default for SalmondDownsampler {
    fn default() -> Self {
        Self {
            view: View::default(),
            t1: DEFAULT_T1,
            delta_t: DEFAULT_DELTA_T,
            target_count: f64::INFINITY,
            data_f64: Vec::new(),
        }
    }
}

#[wasm_bindgen]
impl SalmondDownsampler {
    #[wasm_bindgen(constructor)]
    pub fn new() -> SalmondDownsampler {
        SalmondDownsampler::default()
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

    /// Initial clustering threshold T1. Named for parity with `Downsampler`,
    /// but it is a squared, mass-weighted Mahalanobis radius, not a pixel count.
    #[wasm_bindgen(js_name = setMergeThreshold)]
    pub fn set_merge_threshold(&mut self, value: f64) {
        self.t1 = value;
    }

    /// Threshold increment applied between passes when the budget is not yet met.
    #[wasm_bindgen(js_name = setThresholdIncrement)]
    pub fn set_threshold_increment(&mut self, value: f64) {
        self.delta_t = value;
    }

    /// Component budget NT. Clustering repeats with a widened threshold until the
    /// mixture is at most this large --- possibly overshooting below it.
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
        let alive = salmond_cluster_reduce(&mut m, self.target_count, self.t1, self.delta_t);
        self.view.pack_instances(&m, &alive)
    }
}

/// Salmond's clustering reduction of `m`, in place. Returns the liveness mask
/// over the original component slots.
///
/// One pass is always performed at `t1` --- the threshold, not the budget, is
/// this algorithm's primary criterion. Further passes run only while the
/// component count still exceeds `target_count`.
///
/// When a budget *is* set, the threshold is bracketed rather than merely
/// widened. The paper accepts that a pass may cluster "more components than is
/// necessary", which is harmless for its tracking mixtures of a few dozen
/// components but severe here: at N = 10,000 a single widened pass overshot a
/// 621-kernel budget down to 146, which would make any equal-budget comparison
/// meaningless. So a pass that lands below `BUDGET_TOLERANCE * NT` is rolled
/// back and retried at the midpoint of the bracketing thresholds. This changes
/// only *which* threshold is used, never how a pass clusters.
pub(crate) fn salmond_cluster_reduce(
    m: &mut Mixture,
    target_count: f64,
    t1: f64,
    delta_t: f64,
) -> Vec<bool> {
    let n = m.len();
    let mut alive = vec![true; n];
    if n <= 1 {
        return alive;
    }

    // Normalising mass is the mixture total, which every merge preserves.
    let total = m.total_weight(&alive).max(1e-300);
    let budgeted = target_count.is_finite();

    // Largest threshold known to leave the mixture above budget, and the
    // smallest known to overshoot below it.
    let mut lo = 0.0_f64;
    let mut hi = f64::INFINITY;

    let mut threshold = t1;
    let mut increment = delta_t.max(1e-12);
    let mut count = n;

    for _ in 0..MAX_PASSES {
        let saved = (m.clone(), alive.clone(), count);
        let before = count;
        let after = cluster_pass(m, &mut alive, total, threshold);

        if (after as f64) <= target_count {
            let tight = (after as f64) >= target_count * BUDGET_TOLERANCE;
            if !budgeted || tight || after <= 1 || hi - lo <= 1e-12 {
                break;
            }
            // Overshot the budget: undo the pass and try a gentler threshold.
            hi = threshold;
            let (saved_m, saved_alive, saved_count) = saved;
            *m = saved_m;
            alive = saved_alive;
            count = saved_count;
            threshold = 0.5 * (lo + hi);
            continue;
        }

        count = after;
        if count <= 1 {
            break;
        }
        lo = threshold;
        if count == before {
            // No progress at this threshold. The paper leaves dT a free
            // compromise "between the number of iterations required and the
            // possibility of clustering more components than necessary"; growing
            // it geometrically on a stalled pass keeps the pass count bounded
            // for the mixture sizes here (N in the thousands, not dozens).
            increment *= 2.0;
        }
        threshold = if hi.is_finite() {
            0.5 * (lo + hi)
        } else {
            threshold + increment
        };
    }

    alive
}

/// One clustering sweep: repeatedly take the heaviest unclustered component as
/// the principal one and absorb everything within `threshold` of it. Returns the
/// number of components left alive.
fn cluster_pass(m: &mut Mixture, alive: &mut [bool], total: f64, threshold: f64) -> usize {
    let n = m.len();

    // Heaviest first; index breaks ties so the sweep is deterministic and
    // independent of how equal-weight components happen to be stored.
    let mut order: Vec<usize> = (0..n).filter(|i| alive[*i]).collect();
    order.sort_by(|a, b| {
        m.w[*b]
            .partial_cmp(&m.w[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    });

    let mut clustered = vec![false; n];
    let mut live = 0usize;

    for pos in 0..order.len() {
        let c = order[pos];
        if clustered[c] {
            continue;
        }
        clustered[c] = true;
        live += 1;

        // Snapshot of the principal component: distances are measured against
        // it as selected, not against the cluster as it grows.
        let cx = m.x[c];
        let cy = m.y[c];
        let bc = m.w[c] / total;
        let p00 = m.p00[c];
        let p01 = m.p01[c];
        let p11 = m.p11[c];
        let det = p00 * p11 - p01 * p01;
        if det.abs() < 1e-300 {
            continue;
        }
        let inv_det = 1.0 / det;

        for &i in &order[pos + 1..] {
            if clustered[i] {
                continue;
            }
            let bi = m.w[i] / total;
            let sum = bi + bc;
            if sum <= 0.0 {
                continue;
            }

            let dx = m.x[i] - cx;
            let dy = m.y[i] - cy;
            // (dx,dy)^T Pc^-1 (dx,dy) for the symmetric 2x2 Pc.
            let maha = inv_det * (p11 * dx * dx - 2.0 * p01 * dx * dy + p00 * dy * dy);
            let d2 = (bi * bc / sum) * maha;

            if d2 < threshold {
                m.merge_pair_into(c, i);
                alive[i] = false;
                clustered[i] = true;
            }
        }
    }

    live
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{moments, unit_mixture};

    fn live_count(alive: &[bool]) -> usize {
        alive.iter().filter(|a| **a).count()
    }

    #[test]
    fn coincident_components_always_cluster() {
        // D^2 = 0 for a zero separation, so any positive threshold absorbs it.
        let mut m = unit_mixture(&[(2.0, -1.0, 3.0), (2.0, -1.0, 1.0)]);
        let alive = salmond_cluster_reduce(&mut m, f64::INFINITY, 1e-12, DEFAULT_DELTA_T);
        assert_eq!(live_count(&alive), 1);
    }

    #[test]
    fn distance_is_normalised_by_the_principal_covariance() {
        // Inflating Pc by 4 quarters D^2, so a separation that was rejected at a
        // given threshold is accepted once the principal component is wider.
        // This is what distinguishes the clustering measure from the joining
        // one, which normalises by the covariance of the whole mixture.
        let points = [(0.0, 0.0, 10.0), (3.0, 0.0, 1.0)];

        let mut tight = unit_mixture(&points);
        let alive = salmond_cluster_reduce(&mut tight, f64::INFINITY, 0.5, DEFAULT_DELTA_T);
        assert_eq!(live_count(&alive), 2);

        let mut wide = unit_mixture(&points);
        wide.p00[0] = 4.0;
        wide.p11[0] = 4.0;
        let alive = salmond_cluster_reduce(&mut wide, f64::INFINITY, 0.5, DEFAULT_DELTA_T);
        assert_eq!(live_count(&alive), 1);
    }

    #[test]
    fn light_components_are_absorbed_more_readily_than_heavy_ones() {
        // The b_i b_c / (b_i + b_c) factor: at the same separation from the same
        // principal component, the lighter neighbour clusters and the heavy one
        // does not.
        let light = {
            let mut m = unit_mixture(&[(0.0, 0.0, 100.0), (2.0, 0.0, 0.01)]);
            let total = m.total_weight(&vec![true; 2]);
            let alive = salmond_cluster_reduce(&mut m, f64::INFINITY, 1e-4 * total, 1.0);
            live_count(&alive)
        };
        let heavy = {
            let mut m = unit_mixture(&[(0.0, 0.0, 100.0), (2.0, 0.0, 100.0)]);
            let total = m.total_weight(&vec![true; 2]);
            let alive = salmond_cluster_reduce(&mut m, f64::INFINITY, 1e-4 * total, 1.0);
            live_count(&alive)
        };
        assert_eq!(light, 1);
        assert_eq!(heavy, 2);
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

        let alive = salmond_cluster_reduce(&mut m, 1.0, DEFAULT_T1, DEFAULT_DELTA_T);
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

        for target in [1.0, 3.0, 10.0, 25.0] {
            let mut m = unit_mixture(&points);
            let alive = salmond_cluster_reduce(&mut m, target, DEFAULT_T1, DEFAULT_DELTA_T);

            // Clustering combines in groups, so it may overshoot the budget --
            // it must never undershoot it.
            assert!(live_count(&alive) as f64 <= target);

            let sum: f64 = (0..m.len()).filter(|i| alive[*i]).map(|i| m.w[i]).sum();
            assert!((sum - total).abs() < 1e-9);
        }
    }

    #[test]
    fn recovers_well_separated_groups() {
        // Three tight pairs, far apart. The heaviest of each pair is picked as
        // the principal component and absorbs only its own partner.
        let mut m = unit_mixture(&[
            (0.0, 0.0, 3.0),
            (0.1, 0.0, 1.0),
            (500.0, 0.0, 3.0),
            (500.1, 0.0, 1.0),
            (0.0, 800.0, 3.0),
            (0.1, 800.0, 1.0),
        ]);
        let alive = salmond_cluster_reduce(&mut m, 3.0, DEFAULT_T1, DEFAULT_DELTA_T);

        let mut centers: Vec<String> = (0..m.len())
            .filter(|i| alive[*i])
            .map(|i| format!("{:.3},{:.3}", m.x[i], m.y[i]))
            .collect();
        centers.sort();
        assert_eq!(
            centers,
            vec!["0.025,0.000", "0.025,800.000", "500.025,0.000"]
        );
    }

    #[test]
    fn is_independent_of_input_ordering() {
        // Distinct weights, so the heaviest-first sweep has no ties to break.
        let points: Vec<(f64, f64, f64)> = (0..30)
            .map(|i| {
                let f = i as f64;
                (
                    (f * 2.3).sin() * 4.0,
                    (f * 1.1).cos() * 4.0,
                    1.0 + f * 0.37,
                )
            })
            .collect();
        let reversed: Vec<(f64, f64, f64)> = points.iter().rev().copied().collect();

        let key = |pts: &[(f64, f64, f64)]| {
            let mut m = unit_mixture(pts);
            let alive = salmond_cluster_reduce(&mut m, 5.0, DEFAULT_T1, DEFAULT_DELTA_T);
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
    fn a_single_pass_runs_when_no_budget_is_set() {
        // Without NT the threshold alone governs: one pass at T1, no widening.
        // A dense cloud must lose components; a sparse one must keep all of them.
        let dense: Vec<(f64, f64, f64)> = (0..50)
            .map(|i| (i as f64 * 1e-4, 0.0, 1.0 + (i % 3) as f64))
            .collect();
        let mut m = unit_mixture(&dense);
        let alive = salmond_cluster_reduce(&mut m, f64::INFINITY, DEFAULT_T1, DEFAULT_DELTA_T);
        assert_eq!(live_count(&alive), 1);

        let sparse: Vec<(f64, f64, f64)> = (0..50)
            .map(|i| (i as f64 * 40.0, 0.0, 1.0 + (i % 3) as f64))
            .collect();
        let mut m = unit_mixture(&sparse);
        let alive = salmond_cluster_reduce(&mut m, f64::INFINITY, DEFAULT_T1, DEFAULT_DELTA_T);
        assert_eq!(live_count(&alive), 50);
    }

    #[test]
    fn always_terminates_on_a_degenerate_mixture() {
        // Every component identical: the first pass collapses them regardless of
        // how small the threshold is, because D^2 = 0.
        let points: Vec<(f64, f64, f64)> = (0..25).map(|_| (7.0, 7.0, 2.0)).collect();
        let mut m = unit_mixture(&points);
        let alive = salmond_cluster_reduce(&mut m, 4.0, 1e-300, 1e-300);
        assert_eq!(live_count(&alive), 1);
    }

    #[test]
    fn default_thresholds_match_the_paper() {
        // T1 = 0.05 * T1', where T1' is the ellipse holding 1% of the principal
        // component's mass; dT = 0.05 * dT', where T1' + dT' holds 6%.
        assert!((R2_1_PERCENT - (-2.0 * 0.99_f64.ln())).abs() < 1e-15);
        assert!((R2_6_PERCENT - (-2.0 * 0.94_f64.ln())).abs() < 1e-15);
        assert!(DEFAULT_T1 > 0.0 && DEFAULT_DELTA_T > DEFAULT_T1);
    }
}
