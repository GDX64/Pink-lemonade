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
    /// query runs until `FrameStatus::Complete` or stream close.
    pub max_dwell: Option<Duration>,
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
            steps: vec![WorkloadStep {
                label: "overview [0.000, 1.000]".to_owned(),
                query: ViewQuery::new(0.0, 1.0, screen_w, screen_h, sigma_px, merge_threshold),
                max_dwell: None,
            }],
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
                WorkloadStep {
                    label: format!("zoom width={width:.3} [{min_x:.4}, {max_x:.4}]"),
                    query: ViewQuery::new(
                        min_x,
                        max_x,
                        screen_w,
                        screen_h,
                        sigma_px,
                        merge_threshold,
                    ),
                    max_dwell: None,
                }
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
                WorkloadStep {
                    label: format!("pan#{i} [{min_x:.4}, {max_x:.4}]"),
                    query: ViewQuery::new(
                        min_x,
                        max_x,
                        screen_w,
                        screen_h,
                        sigma_px,
                        merge_threshold,
                    ),
                    max_dwell: None,
                }
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
            .map(|(tag, min_x, max_x)| WorkloadStep {
                label: format!("{tag} [{min_x:.2}, {max_x:.2}]"),
                query: ViewQuery::new(min_x, max_x, screen_w, screen_h, sigma_px, merge_threshold),
                max_dwell: None,
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
                label: format!("interrupt#{i} [{min_x:.2}, {max_x:.2}]"),
                query: ViewQuery::new(min_x, max_x, screen_w, screen_h, sigma_px, merge_threshold),
                max_dwell: Some(dwell),
            });
        }
        steps.push(WorkloadStep {
            label: "settled [0.55, 0.60]".to_owned(),
            query: ViewQuery::new(0.55, 0.60, screen_w, screen_h, sigma_px, merge_threshold),
            max_dwell: None,
        });
        Self {
            name: "rapid_interrupt_pan".to_owned(),
            steps,
        }
    }

    /// Full standard benchmark suite covering all 5 navigation patterns.
    pub fn standard_suite(
        screen_w: f64,
        screen_h: f64,
        sigma_px: f64,
        merge_threshold: f64,
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
        ]
    }
}
