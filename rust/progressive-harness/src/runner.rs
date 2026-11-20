//! Step, workload, and full benchmark execution runners.

use std::sync::Arc;
use std::time::Duration;
use tokio::time::Instant;

use pink_lemonade_progressive_core::{
    ByteRangeStore, ClientConfig, Frame, Point, ProgressiveClient, ProgressiveStrategy,
};

use crate::fixture::{DatasetStream, stream_into_builder};
use crate::quality::{evaluate_density_nmae, exact_visible_y_range, sample_reference_pixels};
use crate::report::{BenchmarkReport, StepReport, WorkloadReport};
use crate::store::BenchmarkStore;
use crate::transport::{NetworkProfile, SimulatedTransport};
use crate::workload::{Workload, WorkloadStep};

/// Configuration for running a strategy through the benchmark harness.
#[derive(Debug, Clone)]
pub struct HarnessConfig {
    pub network: NetworkProfile,
    pub client: ClientConfig,
    pub screen_w: f64,
    pub screen_h: f64,
    pub sigma_px: f64,
    pub merge_threshold: f64,
    /// Number of reference pixels per step for density NMAE evaluation
    /// (only used when ground-truth points are provided).
    pub nmae_pixel_samples: usize,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            network: NetworkProfile::broadband(),
            client: ClientConfig::default(),
            screen_w: 1920.0,
            screen_h: 1080.0,
            sigma_px: 16.0,
            merge_threshold: 1.0,
            nmae_pixel_samples: 64,
        }
    }
}

/// Outcome of executing one [`WorkloadStep`] against a [`ProgressiveClient`].
#[derive(Debug, Clone)]
pub struct StepOutcome {
    /// Returned frame if the query finished before any `max_dwell` timeout.
    pub frame: Option<Frame>,
    /// Simulated elapsed time until completion or timeout.
    pub elapsed: Duration,
    pub completed: bool,
}

/// Execute a single [`WorkloadStep`] against `client`, honoring `max_dwell` cancellation.
pub async fn run_step(
    client: &mut dyn ProgressiveClient,
    step: &WorkloadStep,
) -> Result<StepOutcome, String> {
    let start = Instant::now();
    match step.max_dwell {
        None => {
            let frame = client
                .query(step.query)
                .await
                .map_err(|err| format!("step '{}' failed: {err}", step.label))?;
            Ok(StepOutcome {
                frame: Some(frame),
                elapsed: start.elapsed(),
                completed: true,
            })
        }
        Some(dwell) => match tokio::time::timeout(dwell, client.query(step.query)).await {
            Ok(Ok(frame)) => Ok(StepOutcome {
                frame: Some(frame),
                elapsed: start.elapsed(),
                completed: true,
            }),
            Ok(Err(err)) => Err(format!("step '{}' failed: {err}", step.label)),
            Err(_) => Ok(StepOutcome {
                frame: None,
                elapsed: start.elapsed(),
                completed: false,
            }),
        },
    }
}

/// Run one [`Workload`] against a freshly opened client wrapped in [`SimulatedTransport`].
pub async fn run_workload(
    strategy: &dyn ProgressiveStrategy,
    store: Arc<dyn ByteRangeStore>,
    workload: &Workload,
    config: &HarnessConfig,
    ground_truth: Option<&[Point]>,
) -> Result<WorkloadReport, String> {
    let transport = Arc::new(SimulatedTransport::new(store, config.network));
    let before_open = transport.metrics();
    let mut client = strategy
        .open_client(transport.clone(), config.client)
        .await
        .map_err(|e| format!("open_client failed: {e}"))?;
    let after_open = transport.metrics();
    let open_transport = after_open.delta_since(&before_open);

    let mut peak_client_memory_bytes = client.memory_bytes();
    let mut step_reports = Vec::with_capacity(workload.steps.len());

    for (idx, step) in workload.steps.iter().enumerate() {
        let before_step = transport.metrics();
        let outcome = run_step(client.as_mut(), step).await?;
        let after_step = transport.metrics();
        let step_transport = after_step.delta_since(&before_step);

        let mem = client.memory_bytes();
        peak_client_memory_bytes = peak_client_memory_bytes.max(mem);

        let latency = outcome.completed.then_some(outcome.elapsed);
        let kernel_count = outcome
            .frame
            .as_ref()
            .map(Frame::kernel_count)
            .unwrap_or(0);

        let nmae_percent = match (ground_truth, outcome.frame.as_ref()) {
            (Some(points), Some(frame)) => compute_nmae(
                points,
                step,
                frame,
                config.nmae_pixel_samples,
                0xA11CE_u64.wrapping_add(idx as u64),
            )?,
            _ => None,
        };

        step_reports.push(StepReport {
            label: step.label.clone(),
            completed: outcome.completed,
            latency,
            kernel_count,
            client_memory_bytes: mem,
            transport: step_transport,
            nmae_percent,
        });
    }

    let total_transport = transport.metrics().delta_since(&before_open);

    Ok(WorkloadReport {
        name: workload.name.clone(),
        open_transport,
        steps: step_reports,
        total_transport,
        peak_client_memory_bytes,
    })
}

/// Build a dataset from `stream` into `store`, then execute `workloads` and assemble a
/// [`BenchmarkReport`].
///
/// For deterministic virtual-time timing, run inside a Tokio runtime with
/// `#[tokio::test(start_paused = true)]` or call `tokio::time::pause()` beforehand.
pub async fn run_benchmark(
    strategy: &dyn ProgressiveStrategy,
    stream: DatasetStream,
    store: &mut dyn BenchmarkStore,
    workloads: &[Workload],
    config: &HarnessConfig,
    ground_truth: Option<&[Point]>,
) -> Result<BenchmarkReport, String> {
    let build_start = std::time::Instant::now();
    let mut builder = strategy.create_builder();
    let total_points = stream_into_builder(stream, builder.as_mut())
        .map_err(|e| format!("build push failed: {e}"))?;
    builder
        .finish(store)
        .map_err(|e| format!("build finish failed: {e}"))?;
    let build_duration = build_start.elapsed();

    let objects = store
        .list_objects()
        .await
        .map_err(|e| format!("list_objects failed: {e}"))?;
    let storage_bytes = objects.iter().map(|o| o.len).sum();
    let object_count = objects.len();
    let reader = store.reader();

    let mut workload_reports = Vec::with_capacity(workloads.len());
    for workload in workloads {
        let report = run_workload(strategy, reader.clone(), workload, config, ground_truth).await?;
        workload_reports.push(report);
    }

    Ok(BenchmarkReport {
        strategy_name: strategy.name().to_owned(),
        total_points,
        build_duration,
        storage_bytes,
        object_count,
        network: config.network,
        memory_budget_bytes: config.client.memory_budget_bytes,
        workloads: workload_reports,
    })
}

fn compute_nmae(
    points: &[Point],
    step: &WorkloadStep,
    frame: &Frame,
    sample_count: usize,
    seed: u64,
) -> Result<Option<f64>, String> {
    if sample_count == 0 {
        return Ok(None);
    }
    let Some(y_range) = exact_visible_y_range(points, step.query.min_x, step.query.max_x) else {
        return Ok(None);
    };
    let samples = sample_reference_pixels(points, &step.query, y_range, sample_count, seed);
    if samples.is_empty() {
        return Ok(None);
    }
    let visible_count = points
        .iter()
        .filter(|p| p.x >= step.query.min_x && p.x <= step.query.max_x)
        .count();
    let q = evaluate_density_nmae(visible_count, frame, &step.query, y_range, &samples)?;
    Ok(Some(q.normalized_mean_absolute_error_percent))
}
