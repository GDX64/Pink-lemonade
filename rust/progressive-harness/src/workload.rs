//! Scripted interactive user-navigation workloads for benchmarking progressive strategies.

use std::time::Duration;

use pink_lemonade_progressive_core::ViewQuery;

/// A single viewport step in a simulated user session.
#[derive(Debug, Clone)]
pub struct WorkloadStep {
    pub label: String,
    pub query: ViewQuery,
    /// When `Some(duration)`, the harness cancels this query after `duration` of simulated
    /// time (modeling rapid pan/zoom before the previous view converges). When `None`, the
    /// query runs to completion.
    pub max_dwell: Option<Duration>,
    /// Explicit Y viewport as fractions `[lo, hi]` of the `y_range` returned by the most recent
    /// completed auto-y step of the same workload (models a box-zoom on what the user sees).
    /// The runner resolves it into `query.y_bounds` before issuing the query.
    pub relative_y_bounds: Option<[f64; 2]>,
}

impl WorkloadStep {
    /// Auto-y step that runs to completion.
    pub fn new(label: impl Into<String>, query: ViewQuery) -> Self {
        Self {
            label: label.into(),
            query,
            max_dwell: None,
            relative_y_bounds: None,
        }
    }

    /// Resolve `relative_y_bounds` against an auto-y `anchor` range.
    pub fn resolved_query(&self, anchor: Option<[f64; 2]>) -> Result<ViewQuery, String> {
        let Some([lo, hi]) = self.relative_y_bounds else {
            return Ok(self.query);
        };
        let [a, b] = anchor.ok_or_else(|| {
            format!(
                "step '{}' uses relative_y_bounds without a preceding auto-y frame",
                self.label
            )
        })?;
        let span = b - a;
        Ok(self.query.with_y_bounds(a + lo * span, a + hi * span))
    }
}

/// Named sequence of user viewport interactions run against a single persistent client
/// (preserving client cache state across steps within the workload).
#[derive(Debug, Clone)]
pub struct Workload {
    pub name: String,
    pub steps: Vec<WorkloadStep>,
}

impl Workload {
    /// Initial cold load of the full `[0, 1]` overview at screen resolution.
    pub fn cold_overview(
        screen_w: f64,
        screen_h: f64,
        sigma_px: f64,
        merge_threshold: f64,
    ) -> Self {
        Self {
            name: "cold_overview".to_owned(),
            steps: vec![WorkloadStep::new(
                "overview [0.000, 1.000]",
                ViewQuery::new(0.0, 1.0, screen_w, screen_h, sigma_px, merge_threshold),
            )],
        }
    }

    /// Progressive zoom-in from full overview down to 0.1% width around `anchor_x`.
    pub fn progressive_zoom_in(
        anchor_x: f64,
        screen_w: f64,
        screen_h: f64,
        sigma_px: f64,
        merge_threshold: f64,
    ) -> Self {
        let widths = [1.0, 0.5, 0.2, 0.05, 0.01, 0.001];
        let steps = widths
            .into_iter()
            .map(|width| {
                let min_x = (anchor_x - width * 0.5).clamp(0.0, 1.0 - width);
                let max_x = (min_x + width).min(1.0);
                WorkloadStep::new(
                    format!("zoom width={width:.3} [{min_x:.4}, {max_x:.4}]"),
                    ViewQuery::new(min_x, max_x, screen_w, screen_h, sigma_px, merge_threshold),
                )
            })
            .collect();
        Self {
            name: "progressive_zoom_in".to_owned(),
            steps,
        }
    }

    /// Overlapping lateral pan across adjacent slices (tests cache reuse / delta fetching).
    pub fn lateral_pan(
        window_width: f64,
        step_count: usize,
        screen_w: f64,
        screen_h: f64,
        sigma_px: f64,
        merge_threshold: f64,
    ) -> Self {
        let stride = window_width * 0.35;
        let steps = (0..step_count)
            .map(|i| {
                let min_x = (0.1 + i as f64 * stride).min(1.0 - window_width);
                let max_x = (min_x + window_width).min(1.0);
                WorkloadStep::new(
                    format!("pan#{i} [{min_x:.4}, {max_x:.4}]"),
                    ViewQuery::new(min_x, max_x, screen_w, screen_h, sigma_px, merge_threshold),
                )
            })
            .collect();
        Self {
            name: "lateral_pan".to_owned(),
            steps,
        }
    }

    /// Jump across distant regions and revisit earlier ones (exercises LRU eviction & re-fetch).
    pub fn jump_and_revisit(
        screen_w: f64,
        screen_h: f64,
        sigma_px: f64,
        merge_threshold: f64,
    ) -> Self {
        let ranges = [
            ("region_A", 0.05, 0.10),
            ("region_B", 0.45, 0.50),
            ("region_C", 0.85, 0.90),
            ("region_D", 0.25, 0.30),
            ("revisit_A", 0.05, 0.10),
            ("revisit_C", 0.85, 0.90),
        ];
        let steps = ranges
            .into_iter()
            .map(|(tag, min_x, max_x)| {
                WorkloadStep::new(
                    format!("{tag} [{min_x:.2}, {max_x:.2}]"),
                    ViewQuery::new(min_x, max_x, screen_w, screen_h, sigma_px, merge_threshold),
                )
            })
            .collect();
        Self {
            name: "jump_and_revisit".to_owned(),
            steps,
        }
    }

    /// Rapid pan where intermediate frames are interrupted mid-flight after `dwell` simulated time,
    /// followed by a final settled view that runs to completion.
    pub fn rapid_interrupt_pan(
        dwell: Duration,
        screen_w: f64,
        screen_h: f64,
        sigma_px: f64,
        merge_threshold: f64,
    ) -> Self {
        let mut steps = Vec::new();
        for i in 0..5 {
            let min_x = 0.10 + i as f64 * 0.08;
            let max_x = min_x + 0.05;
            steps.push(WorkloadStep {
                max_dwell: Some(dwell),
                ..WorkloadStep::new(
                    format!("interrupt#{i} [{min_x:.2}, {max_x:.2}]"),
                    ViewQuery::new(min_x, max_x, screen_w, screen_h, sigma_px, merge_threshold),
                )
            });
        }
        steps.push(WorkloadStep::new(
            "settled [0.55, 0.60]",
            ViewQuery::new(0.55, 0.60, screen_w, screen_h, sigma_px, merge_threshold),
        ));
        Self {
            name: "rapid_interrupt_pan".to_owned(),
            steps,
        }
    }

    /// Explicit vertical zoom on a fixed 5% x-window: auto-y anchor, then 2×, 8×, 32× box-zooms
    /// centered at `y_center_frac` of the anchor's `y_range`, an 8× vertical pan, and a reset
    /// back to auto-y (cache reuse). Exercises `ViewQuery::y_bounds` and off-screen culling.
    pub fn y_zoom(
        y_center_frac: f64,
        screen_w: f64,
        screen_h: f64,
        sigma_px: f64,
        merge_threshold: f64,
    ) -> Self {
        let (min_x, max_x) = (0.40, 0.45);
        let query = ViewQuery::new(min_x, max_x, screen_w, screen_h, sigma_px, merge_threshold);
        let zoom = |tag: &str, factor: f64, center: f64| {
            let half = 0.5 / factor;
            let (lo, hi) = (center - half, center + half);
            WorkloadStep {
                relative_y_bounds: Some([lo, hi]),
                ..WorkloadStep::new(format!("{tag} y{factor}x [{lo:.3}, {hi:.3}]"), query)
            }
        };
        let auto = |tag: &str| WorkloadStep::new(format!("{tag} [{min_x:.2}, {max_x:.2}]"), query);
        Self {
            name: "y_zoom".to_owned(),
            steps: vec![
                auto("auto_y"),
                zoom("zoom", 2.0, y_center_frac),
                zoom("zoom", 8.0, y_center_frac),
                zoom("zoom", 32.0, y_center_frac),
                zoom("pan", 8.0, y_center_frac + 1.0 / 16.0),
                auto("reset"),
            ],
        }
    }

    /// Full standard benchmark suite covering all 6 navigation patterns.
    ///
    /// `y_center_frac` positions the `y_zoom` workload's vertical zoom within the auto-y range
    /// (e.g. `0.5` for symmetric signals, `0.1` for `NoisySpikes` whose spikes stretch the top).
    pub fn standard_suite(
        screen_w: f64,
        screen_h: f64,
        sigma_px: f64,
        merge_threshold: f64,
        y_center_frac: f64,
    ) -> Vec<Self> {
        vec![
            Self::cold_overview(screen_w, screen_h, sigma_px, merge_threshold),
            Self::progressive_zoom_in(0.37, screen_w, screen_h, sigma_px, merge_threshold),
            Self::lateral_pan(0.05, 8, screen_w, screen_h, sigma_px, merge_threshold),
            Self::jump_and_revisit(screen_w, screen_h, sigma_px, merge_threshold),
            Self::rapid_interrupt_pan(
                Duration::from_millis(20),
                screen_w,
                screen_h,
                sigma_px,
                merge_threshold,
            ),
            Self::y_zoom(y_center_frac, screen_w, screen_h, sigma_px, merge_threshold),
        ]
    }
}
