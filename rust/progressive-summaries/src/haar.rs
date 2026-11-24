//! Haar-coded density pyramid with view-driven thresholding.
//!
//! **Grid.** Dyadic cells over the data bbox: column level `a` splits x into `2^a` columns and
//! row level `b` splits y into `2^b` rows.
//!
//! **Decomposition.** For a view the client picks an *emit level* `(Ex, Ey)` whose cells are at
//! most `emit_cell_sigma` kernel widths on screen, and a *finest level* `(Fx, Fy) = (Ex, Ey) + D`.
//! Between a root level `(Fx - n, Fy - n)` (`n = min(Fx, Fy)`, so one axis is a single
//! column or row) and `(Fx, Fy)` the density is coded by the non-standard 2D Haar transform:
//! the root cells' masses (scaling coefficients), then three detail coefficients per parent
//! cell for each step that halves both axes. Every basis function of one step has the
//! cell shape of the view's emit level, so the coefficients a view needs are local to it.
//!
//! **Error estimate.** Dropping a coefficient `c` of basis `B` changes the rendered density by
//! `c · (B * G)`. Its energy is `c² · gain_x · gain_y` in σ units ([`crate::blur`]).
//! The client drops the lowest-energy coefficients while their summed energy stays within
//! `ε²` of the total (a relative L2 bound that ignores cross terms between blurred basis
//! functions).
//!
//! **Emission.** The kept coefficients reconstruct cell masses down to the finest level. Each
//! emit cell is drawn as one Gaussian matching the moments of its reconstructed sub-cells
//! (boxes with variance `h²/12`).
//!
//! **Exactness.** As in the other strategies, only finest-level columns fully inside
//! `[min_x, max_x]` are summarized; the points of the partial columns at both edges are drawn
//! raw. With explicit `y_bounds`, only rows within 6σ of the screen are coded.

use std::collections::HashMap;

use pink_lemonade_progressive_common::{Node, collect_points, push_group};
use pink_lemonade_progressive_core::{Frame, Point, ViewQuery};

use crate::blur::{box_gain, wavelet_gain, wavelet_l1_gain};

/// Bytes charged per kept detail coefficient (`f32` value).
pub const COEF_BYTES: usize = 4;
/// Bytes charged per parent cell with at least one kept detail (its position).
pub const PARENT_BYTES: usize = 2;
/// Bytes charged per root cell (`f32` mass + position).
pub const ROOT_BYTES: usize = 6;
/// Bytes charged per raw point (`f64` x, y, weight).
pub const RAW_POINT_BYTES: usize = 24;

/// Distance in σ beyond which off-screen rows contribute nothing visible.
const OFFSCREEN_MARGIN_SIGMA: f64 = 6.0;

/// Which error the client budgets when dropping coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Budget {
    /// Relative L2 energy, assuming dropped blurred basis functions are orthogonal.
    L2,
    /// Relative L1 upper bound: `Σ |d| · gain₁ ≤ ε · mass` (triangle inequality, no assumption).
    L1,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HaarParams {
    /// Largest emit-cell side on screen, in kernel widths.
    pub emit_cell_sigma: f64,
    /// Detail levels below the emit level (sub-cell resolution `2^-D` of an emit cell).
    pub detail_levels: u32,
    /// Target relative L2 error of the rendered density (`0` keeps every coefficient).
    pub epsilon: f64,
    /// Also threshold the details below the emit level. Their blurred basis functions are
    /// nearly parallel, so their errors add coherently and the energy estimate undercounts them.
    pub threshold_sub_emit: bool,
    pub budget: Budget,
}

impl Default for HaarParams {
    fn default() -> Self {
        Self {
            emit_cell_sigma: 0.5,
            detail_levels: 3,
            epsilon: 0.05,
            threshold_sub_emit: true,
            budget: Budget::L1,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HaarStats {
    pub emit_level: (u32, u32),
    pub finest_level: (u32, u32),
    pub root_cells: usize,
    /// Non-zero detail coefficients in the view's region.
    pub total_coefs: usize,
    pub kept_coefs: usize,
    pub kept_parents: usize,
    pub raw_points: usize,
    /// Emit cells whose reconstructed mass is negative (drawn with negative amplitude).
    pub negative_kernels: usize,
    pub bytes: usize,
    /// Estimated relative error of the dropped coefficients (L2: `sqrt(dropped / total
    /// energy)`; L1: bound on `‖error‖₁ / mass`).
    pub estimated_error: f64,
}

/// Points of one column level, y-sorted within each column, with prefix weights.
struct ColumnLevel {
    starts: Vec<u32>,
    ys: Vec<f64>,
    cw: Vec<f64>,
}

/// In-memory oracle for any dyadic cell's mass (stands in for stored coefficients).
pub struct HaarIndex {
    x0: f64,
    x_span: f64,
    y0: f64,
    y_span: f64,
    levels: Vec<ColumnLevel>,
    points: Vec<Point>,
}

fn pow2(level: u32) -> f64 {
    (1u64 << level) as f64
}

fn cell_of(u: f64, level: u32) -> u64 {
    let n = 1u64 << level;
    ((u * n as f64) as u64).min(n - 1)
}

/// Detail coefficients of one parent cell, unnormalized (sums and differences of child masses).
#[derive(Clone, Copy)]
struct Parent {
    d: [f64; 3],
    keep: [bool; 3],
}

#[derive(Clone, Copy)]
struct Region {
    /// Finest-level column range `[cx0, cx1)` and row range `[ry0, ry1)`.
    cx0: u64,
    cx1: u64,
    ry0: u64,
    ry1: u64,
    fx: u32,
    fy: u32,
    /// Root level `(a0, b0)`; step `s` is level `(a0 + s, b0 + s)`.
    a0: u32,
    b0: u32,
}

struct CellsInput {
    region: Region,
    emit_step: u32,
    params: HaarParams,
    /// Data → σ-unit factors of the view.
    factors: (f64, f64),
    /// Data extent in σ units.
    widths: (f64, f64),
    /// Mass of the points inside the coded region.
    region_mass: f64,
}

impl Region {
    fn overlaps(&self, s: u32, kx: u64, ky: u64) -> bool {
        let sx = self.fx - (self.a0 + s);
        let sy = self.fy - (self.b0 + s);
        (kx << sx) < self.cx1
            && ((kx + 1) << sx) > self.cx0
            && (ky << sy) < self.ry1
            && ((ky + 1) << sy) > self.ry0
    }
}

impl HaarIndex {
    /// Build over x-sorted points with column levels `0..=max_level`.
    pub fn build(points: Vec<Point>, max_level: u32) -> Self {
        assert!(!points.is_empty(), "empty dataset");
        assert!(
            points.windows(2).all(|w| w[0].x <= w[1].x),
            "points must be x-sorted"
        );
        let x0 = points[0].x;
        let x_span = (points[points.len() - 1].x - x0).max(1e-12);
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for p in &points {
            lo = lo.min(p.y);
            hi = hi.max(p.y);
        }
        let (y0, y_span) = (lo, (hi - lo).max(1e-12));
        let n = points.len();

        // Finest level: columns are contiguous in x order; sort each by normalized y.
        let cols = 1usize << max_level;
        let mut starts = vec![0u32; cols + 1];
        for p in &points {
            starts[cell_of((p.x - x0) / x_span, max_level) as usize + 1] += 1;
        }
        for k in 0..cols {
            starts[k + 1] += starts[k];
        }
        let mut entries: Vec<(f64, f64)> = points
            .iter()
            .map(|p| (((p.y - y0) / y_span).clamp(0.0, 1.0), p.weight))
            .collect();
        for k in 0..cols {
            entries[starts[k] as usize..starts[k + 1] as usize].sort_by(|a, b| a.0.total_cmp(&b.0));
        }

        let mut levels_rev = Vec::with_capacity(max_level as usize + 1);
        let mut level = max_level;
        loop {
            let mut cw = Vec::with_capacity(n + 1);
            cw.push(0.0);
            let mut acc = 0.0;
            for e in &entries {
                acc += e.1;
                cw.push(acc);
            }
            let ys = entries.iter().map(|e| e.0).collect();
            levels_rev.push(ColumnLevel {
                starts: starts.clone(),
                ys,
                cw,
            });
            if level == 0 {
                break;
            }
            // Merge column pairs (2k, 2k+1) into column k of the next coarser level.
            let parent_cols = 1usize << (level - 1);
            let mut merged = Vec::with_capacity(n);
            let mut parent_starts = vec![0u32; parent_cols + 1];
            for k in 0..parent_cols {
                let (a, m, b) = (
                    starts[2 * k] as usize,
                    starts[2 * k + 1] as usize,
                    starts[2 * k + 2] as usize,
                );
                let (mut i, mut j) = (a, m);
                while i < m || j < b {
                    if j >= b || (i < m && entries[i].0 <= entries[j].0) {
                        merged.push(entries[i]);
                        i += 1;
                    } else {
                        merged.push(entries[j]);
                        j += 1;
                    }
                }
                parent_starts[k + 1] = b as u32;
            }
            entries = merged;
            starts = parent_starts;
            level -= 1;
        }
        levels_rev.reverse();
        Self {
            x0,
            x_span,
            y0,
            y_span,
            levels: levels_rev,
            points,
        }
    }

    pub fn max_level(&self) -> u32 {
        self.levels.len() as u32 - 1
    }

    pub fn points(&self) -> &[Point] {
        &self.points
    }

    /// Masses of the lower and upper halves of row `ky` (level `b`) in column `kx` (level `a`).
    fn split_rows(&self, a: u32, kx: u64, b: u32, ky: u64) -> (f64, f64) {
        let lv = &self.levels[a as usize];
        let (s, e) = (
            lv.starts[kx as usize] as usize,
            lv.starts[kx as usize + 1] as usize,
        );
        if s == e {
            return (0.0, 0.0);
        }
        let ys = &lv.ys[s..e];
        let rows = pow2(b);
        let i0 = if ky == 0 {
            0
        } else {
            ys.partition_point(|&v| v < ky as f64 / rows)
        };
        let im = ys.partition_point(|&v| v < (2 * ky + 1) as f64 / (2.0 * rows));
        let i1 = if ky + 1 >= 1u64 << b {
            ys.len()
        } else {
            ys.partition_point(|&v| v < (ky + 1) as f64 / rows)
        };
        (lv.cw[s + im] - lv.cw[s + i0], lv.cw[s + i1] - lv.cw[s + im])
    }

    fn cell_mass(&self, a: u32, kx: u64, b: u32, ky: u64) -> f64 {
        if b == 0 {
            let lv = &self.levels[a as usize];
            return lv.cw[lv.starts[kx as usize + 1] as usize]
                - lv.cw[lv.starts[kx as usize] as usize];
        }
        let (lo, hi) = self.split_rows(a, kx, b - 1, ky >> 1);
        if ky & 1 == 0 { lo } else { hi }
    }

    /// Render one view at the given parameters.
    pub fn render(
        &self,
        query: &ViewQuery,
        y_range: [f64; 2],
        params: HaarParams,
    ) -> (Frame, HaarStats) {
        let bounds = query.effective_y_bounds(y_range);
        let (fxf, fyf) = query.factors(bounds);
        let wx = self.x_span * fxf;
        let wy = self.y_span * fyf;
        let level_for = |w: f64| (w / params.emit_cell_sigma).max(1.0).log2().ceil() as u32;
        let ex = level_for(wx).min(self.max_level());
        let ey = level_for(wy);
        let detail = params.detail_levels.min(self.max_level() - ex);
        let (fx, fy) = (ex + detail, ey + detail);
        let n = fx.min(fy);
        let (a0, b0) = (fx - n, fy - n);
        let emit_step = n - detail;

        let mut stats = HaarStats {
            emit_level: (ex, ey),
            finest_level: (fx, fy),
            ..HaarStats::default()
        };

        // Region in finest-level columns and rows.
        let u_min = (query.min_x - self.x0) / self.x_span;
        let u_max = (query.max_x - self.x0) / self.x_span;
        let fcols = 1u64 << fx;
        let cx0 = ((u_min * pow2(fx)).ceil().max(0.0) as u64).min(fcols);
        let cx1 = if u_max >= 1.0 {
            fcols
        } else {
            ((u_max * pow2(fx)).floor().max(0.0) as u64).min(fcols)
        };
        let frows = 1u64 << fy;
        let (ry0, ry1) = match query.y_bounds {
            None => (0, frows),
            Some([lo, hi]) => {
                let margin = OFFSCREEN_MARGIN_SIGMA / fyf;
                let vlo = (lo - margin - self.y0) / self.y_span;
                let vhi = (hi + margin - self.y0) / self.y_span;
                if vhi < 0.0 || vlo > 1.0 {
                    (0, 0)
                } else {
                    (cell_of(vlo.max(0.0), fy), cell_of(vhi.min(1.0), fy) + 1)
                }
            }
        };
        let region = Region {
            cx0,
            cx1,
            ry0,
            ry1,
            fx,
            fy,
            a0,
            b0,
        };

        let mut gpu = Vec::new();
        let mut groups: Vec<Node> = Vec::new();

        // Raw points in the partial columns at both edges.
        let pts = &self.points;
        let vis_a = pts.partition_point(|p| p.x < query.min_x);
        let vis_b = pts.partition_point(|p| p.x <= query.max_x);
        let col = |p: &Point| cell_of((p.x - self.x0) / self.x_span, fx);
        let inner_a = vis_a + pts[vis_a..vis_b].partition_point(|p| col(p) < cx0);
        let inner_b = vis_a + pts[vis_a..vis_b].partition_point(|p| col(p) < cx1);
        let inner_b = inner_b.max(inner_a);
        for slice in [&pts[vis_a..inner_a], &pts[inner_b..vis_b]] {
            stats.raw_points += slice.len();
            collect_points(slice, fxf, fyf, query.merge_threshold, &mut groups);
        }

        if cx0 < cx1 && ry0 < ry1 {
            // The error budget is relative to the mass actually drawn from cells (the client
            // would read it from the reconstruction; it is exact here).
            let region_mass: f64 = pts[inner_a..inner_b]
                .iter()
                .filter(|p| {
                    let row = cell_of(((p.y - self.y0) / self.y_span).clamp(0.0, 1.0), fy);
                    (ry0..ry1).contains(&row)
                })
                .map(|p| p.weight)
                .sum();
            let cells = CellsInput {
                region,
                emit_step,
                params,
                factors: (fxf, fyf),
                widths: (wx, wy),
                region_mass,
            };
            self.render_cells(&cells, &mut stats, &mut gpu);
        }

        // Raw-point groups are drawn as the other strategies draw them.
        let mut merged: Vec<Node> = Vec::with_capacity(groups.len());
        for g in groups {
            push_group(g, fxf, fyf, query.merge_threshold, &mut merged);
        }
        for g in &merged {
            g.emit_gpu(fxf, fyf, &mut gpu);
        }

        stats.bytes = stats.kept_coefs * COEF_BYTES
            + stats.kept_parents * PARENT_BYTES
            + stats.root_cells * ROOT_BYTES
            + stats.raw_points * RAW_POINT_BYTES;
        let frame = Frame {
            gpu_instances: gpu,
            line_points: Vec::new(),
            y_range: Some(y_range),
        };
        (frame, stats)
    }

    fn render_cells(&self, input: &CellsInput, stats: &mut HaarStats, gpu: &mut Vec<f32>) {
        let CellsInput {
            region: ref r,
            emit_step,
            params,
            factors: (fxf, fyf),
            widths: (wx, wy),
            region_mass,
        } = *input;
        let steps = r.fx - r.a0;
        let epsilon = params.epsilon;
        let mut forced_keep = 0usize;
        // Per-step widths (σ units) of a parent cell and the energy gains of its basis functions.
        let gains: Vec<[f64; 3]> = (0..steps)
            .map(|s| {
                let px = wx / pow2(r.a0 + s);
                let py = wy / pow2(r.b0 + s);
                match params.budget {
                    Budget::L2 => {
                        let norm = 1.0 / (px * py);
                        [
                            norm * wavelet_gain(px) * box_gain(py),
                            norm * box_gain(px) * wavelet_gain(py),
                            norm * wavelet_gain(px) * wavelet_gain(py),
                        ]
                    }
                    // A detail `d` moves `|d| / 2` of mass by half a cell; per unit of the
                    // unblurred L1 change (`|d|`), the blurred change is the L1 gain.
                    Budget::L1 => [
                        wavelet_l1_gain(px),
                        wavelet_l1_gain(py),
                        wavelet_l1_gain(px) * wavelet_l1_gain(py),
                    ],
                }
            })
            .collect();

        // 1. Roots and every non-zero detail coefficient in the region (true values).
        let shift_x = r.fx - r.a0;
        let shift_y = r.fy - r.b0;
        let mut roots = Vec::new();
        for kx in (r.cx0 >> shift_x)..=((r.cx1 - 1) >> shift_x) {
            for ky in (r.ry0 >> shift_y)..=((r.ry1 - 1) >> shift_y) {
                let m = self.cell_mass(r.a0, kx, r.b0, ky);
                if m != 0.0 {
                    roots.push((kx, ky, m));
                }
            }
        }
        stats.root_cells = roots.len();
        let root_px = wx / pow2(r.a0);
        let root_py = wy / pow2(r.b0);
        let root_gain = box_gain(root_px) * box_gain(root_py) / (root_px * root_py);
        let total_mass: f64 = roots.iter().map(|&(_, _, m)| m).sum();
        let mut total_energy: f64 = match params.budget {
            Budget::L2 => roots.iter().map(|&(_, _, m)| m * m * root_gain).sum(),
            Budget::L1 => 0.0,
        };

        let mut parents: HashMap<(u32, u64, u64), Parent> = HashMap::new();
        let mut energies: Vec<(f64, (u32, u64, u64), u8)> = Vec::new();
        let mut stack: Vec<(u32, u64, u64)> =
            roots.iter().map(|&(kx, ky, _)| (0, kx, ky)).collect();
        while let Some((s, kx, ky)) = stack.pop() {
            if s == steps {
                continue;
            }
            let (a, b) = (r.a0 + s, r.b0 + s);
            let (lb, lt) = self.split_rows(a + 1, 2 * kx, b, ky);
            let (rb, rt) = self.split_rows(a + 1, 2 * kx + 1, b, ky);
            let d = [lb + lt - rb - rt, lb + rb - lt - rt, lb - rb - lt + rt];
            let key = (s, kx, ky);
            let forced = s >= emit_step && !params.threshold_sub_emit;
            let mut keep = [false; 3];
            for (w, &dw) in d.iter().enumerate() {
                if dw != 0.0 {
                    let e = match params.budget {
                        Budget::L2 => dw * dw * gains[s as usize][w],
                        Budget::L1 => dw.abs() * gains[s as usize][w],
                    };
                    total_energy += e;
                    if forced {
                        keep[w] = true;
                        forced_keep += 1;
                    } else {
                        energies.push((e, key, w as u8));
                    }
                }
            }
            parents.insert(key, Parent { d, keep });
            for (cx, cy, m) in [
                (2 * kx, 2 * ky, lb),
                (2 * kx + 1, 2 * ky, rb),
                (2 * kx, 2 * ky + 1, lt),
                (2 * kx + 1, 2 * ky + 1, rt),
            ] {
                if m != 0.0 && r.overlaps(s + 1, cx, cy) {
                    stack.push((s + 1, cx, cy));
                }
            }
        }
        stats.total_coefs = energies.len() + forced_keep;

        // 2. Drop the lowest-energy coefficients within the ε² budget.
        energies.sort_by(|x, y| x.0.total_cmp(&y.0));
        let budget = match params.budget {
            Budget::L2 => epsilon * epsilon * total_energy,
            Budget::L1 => epsilon * region_mass,
        };
        let mut dropped = 0.0;
        let mut first_kept = energies.len();
        for (i, &(e, _, _)) in energies.iter().enumerate() {
            if dropped + e > budget {
                first_kept = i;
                break;
            }
            dropped += e;
        }
        for &(_, key, w) in &energies[first_kept..] {
            parents.get_mut(&key).expect("coefficient parent").keep[w as usize] = true;
        }
        stats.kept_coefs = energies.len() - first_kept + forced_keep;
        stats.kept_parents = parents
            .values()
            .filter(|p| p.keep.iter().any(|&k| k))
            .count();
        stats.estimated_error = match params.budget {
            Budget::L2 if total_energy > 0.0 => (dropped / total_energy).sqrt(),
            Budget::L1 if region_mass > 0.0 => dropped / region_mass,
            _ => 0.0,
        };

        // 3. Reconstruct with the kept coefficients and emit one Gaussian per emit cell.
        let tiny = 1e-12 * total_mass.abs().max(1e-300);
        let children = |s: u32, kx: u64, ky: u64, m: f64| -> [(u64, u64, f64); 4] {
            let mut d = [0.0; 3];
            if let Some(p) = parents.get(&(s, kx, ky)) {
                for w in 0..3 {
                    if p.keep[w] {
                        d[w] = p.d[w];
                    }
                }
            }
            let [dx, dy, dxy] = d;
            [
                (2 * kx, 2 * ky, 0.25 * (m + dx + dy + dxy)),
                (2 * kx + 1, 2 * ky, 0.25 * (m - dx + dy - dxy)),
                (2 * kx, 2 * ky + 1, 0.25 * (m + dx - dy - dxy)),
                (2 * kx + 1, 2 * ky + 1, 0.25 * (m - dx - dy + dxy)),
            ]
        };

        let hx = self.x_span / pow2(r.fx);
        let hy = self.y_span / pow2(r.fy);
        let mut stack: Vec<(u32, u64, u64, f64)> =
            roots.iter().map(|&(kx, ky, m)| (0, kx, ky, m)).collect();
        let mut sub: Vec<(u32, u64, u64, f64)> = Vec::new();
        while let Some((s, kx, ky, m)) = stack.pop() {
            if s < emit_step {
                for (cx, cy, cm) in children(s, kx, ky, m) {
                    if cm.abs() > tiny && r.overlaps(s + 1, cx, cy) {
                        stack.push((s + 1, cx, cy, cm));
                    }
                }
                continue;
            }
            // Emit cell: accumulate the moments of its reconstructed finest sub-cells.
            let cell_x = self.x0 + (kx as f64 + 0.5) * self.x_span / pow2(r.a0 + s);
            let cell_y = self.y0 + (ky as f64 + 0.5) * self.y_span / pow2(r.b0 + s);
            let (mut sm, mut sx, mut sy, mut sxx, mut sxy, mut syy) =
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            sub.clear();
            sub.push((s, kx, ky, m));
            while let Some((t, qx, qy, qm)) = sub.pop() {
                if t == steps {
                    let ox = self.x0 + (qx as f64 + 0.5) * hx - cell_x;
                    let oy = self.y0 + (qy as f64 + 0.5) * hy - cell_y;
                    sm += qm;
                    sx += qm * ox;
                    sy += qm * oy;
                    sxx += qm * (ox * ox + hx * hx / 12.0);
                    sxy += qm * ox * oy;
                    syy += qm * (oy * oy + hy * hy / 12.0);
                    continue;
                }
                for (cx, cy, cm) in children(t, qx, qy, qm) {
                    if cm.abs() > tiny && r.overlaps(t + 1, cx, cy) {
                        sub.push((t + 1, cx, cy, cm));
                    }
                }
            }
            if sm.abs() <= tiny {
                continue;
            }
            if sm < 0.0 {
                // Truncation ringing: drawn as a signed box-matched kernel so it still cancels
                // against its neighbours after the blur (dropping it would add its mass).
                let ew = self.x_span / pow2(r.a0 + s);
                let eh = self.y_span / pow2(r.b0 + s);
                emit_signed(
                    sm,
                    cell_x,
                    cell_y,
                    [ew * ew / 12.0, 0.0, eh * eh / 12.0],
                    (fxf, fyf),
                    gpu,
                );
                stats.negative_kernels += 1;
                continue;
            }
            let (mx, my) = (sx / sm, sy / sm);
            let min_vx = hx * hx / 12.0;
            let min_vy = hy * hy / 12.0;
            let vx = (sxx / sm - mx * mx).max(min_vx);
            let vy = (syy / sm - my * my).max(min_vy);
            let lim = 0.999 * (vx * vy).sqrt();
            let vxy = (sxy / sm - mx * my).clamp(-lim, lim);
            emit_signed(sm, cell_x + mx, cell_y + my, [vx, vxy, vy], (fxf, fyf), gpu);
        }
    }
}

/// Like `Node::emit_gpu`, but keeps the sign of `mass` in the amplitude.
fn emit_signed(
    mass: f64,
    x: f64,
    y: f64,
    [vx, vxy, vy]: [f64; 3],
    (fx, fy): (f64, f64),
    out: &mut Vec<f32>,
) {
    let p00 = 1.0 + vx * fx * fx;
    let p01 = vxy * fx * fy;
    let p11 = 1.0 + vy * fy * fy;
    let det = (p00 * p11 - p01 * p01).max(1e-12);
    let amplitude = mass / (2.0 * std::f64::consts::PI * det.sqrt());
    out.extend_from_slice(&[
        x as f32,
        y as f32,
        amplitude as f32,
        p00 as f32,
        p01 as f32,
        p01 as f32,
        p11 as f32,
    ]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_points() -> Vec<Point> {
        let mut pts = Vec::new();
        let mut s = 7u64;
        for i in 0..5000 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let y = ((s >> 11) as f64 / (1u64 << 53) as f64) * 10.0 + (i as f64 * 0.01).sin() * 5.0;
            pts.push(Point::new(i as f64 / 4999.0, y, 1.0 + (i % 3) as f64));
        }
        pts
    }

    #[test]
    fn cell_masses_add_up() {
        let pts = grid_points();
        let total: f64 = pts.iter().map(|p| p.weight).sum();
        let idx = HaarIndex::build(pts, 8);
        for (a, b) in [(0, 0), (3, 2), (8, 5)] {
            let mut sum = 0.0;
            for kx in 0..(1u64 << a) {
                for ky in 0..(1u64 << b) {
                    sum += idx.cell_mass(a, kx, b, ky);
                }
            }
            assert!(
                (sum - total).abs() < 1e-6,
                "level ({a},{b}): {sum} vs {total}"
            );
        }
    }

    #[test]
    fn lossless_view_preserves_inside_mass() {
        let pts = grid_points();
        let idx = HaarIndex::build(pts.clone(), 12);
        let q = ViewQuery::new(0.2, 0.7, 800.0, 400.0, 16.0, 1.0);
        let vis: Vec<&Point> = pts.iter().filter(|p| p.x >= 0.2 && p.x <= 0.7).collect();
        let lo = vis.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
        let hi = vis.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);
        let (frame, stats) = idx.render(
            &q,
            [lo, hi],
            HaarParams {
                epsilon: 0.0,
                ..HaarParams::default()
            },
        );
        assert_eq!(stats.kept_coefs, stats.total_coefs);
        assert_eq!(stats.estimated_error, 0.0);
        // Total emitted mass = visible mass (amplitude · 2π · sqrt(det)).
        let mass: f64 = frame
            .gpu_instances
            .as_chunks::<7>()
            .0
            .iter()
            .map(|k| {
                k[2] as f64
                    * 2.0
                    * std::f64::consts::PI
                    * ((k[3] * k[6] - k[4] * k[4]) as f64).sqrt()
            })
            .sum();
        let want: f64 = vis.iter().map(|p| p.weight).sum();
        assert!((mass - want).abs() / want < 1e-4, "{mass} vs {want}");
    }
}
