//! Simulation, conformance, and benchmarking harness for progressive strategies.

pub mod conformance;
pub mod fixture;
pub mod moments;
pub mod quality;
pub mod report;
pub mod workload;

use std::sync::Arc;
use std::time::Duration;
use tokio::time::Instant;

use crate::harness::fixture::{DatasetStream, stream_into_builder};
use crate::harness::quality::{
    evaluate_density_nmae, exact_visible_y_range, sample_reference_pixels,
};
use crate::harness::report::{BenchmarkReport, StepReport, WorkloadReport};
use crate::harness::workload::{Workload, WorkloadStep};
use crate::store::{ByteRangeStore, ObjectWriter};
use crate::strategy::{DatasetBuilder, FrameSink, ProgressiveClient, ProgressiveStrategy};
use crate::transport::{NetworkProfile, SimulatedTransport};
use crate::types::{ClientConfig, ClientError, Frame, FrameStatus, Point};

/// Configuration for running a strategy through the benchmark harness.
#[derive(Debug, Clone)]
pub struct HarnessConfig {
    pub network: NetworkProfile,
    pub client: ClientConfig,
    pub screen_w: f64,
    pub screen_h: f64,
    pub sigma_px: f64,
    pub merge_threshold: f64,
    /// Optional number of reference pixels per step for progressive NMAE evaluation
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

/// Raw outcome of executing one [`WorkloadStep`] against a [`ProgressiveClient`].
#[derive(Debug, Clone)]
pub struct StepOutcome {
    /// Timestamped `(elapsed_since_step_start, Frame)` sequence emitted by the client.
    pub frames: Vec<(Duration, Frame)>,
    /// True if the query reached completion before any `max_dwell` cancellation.
    pub completed: bool,
}

/// Execute a single [`WorkloadStep`] against `client`, honoring `max_dwell` cancellation
/// and recording the simulated timestamp of every emitted frame.
pub async fn run_step<C: ProgressiveClient>(
    client: &mut C,
    step: &WorkloadStep,
) -> Result<StepOutcome, String> {
    let (sink, mut rx) = FrameSink::channel(32);
    let start = Instant::now();
    let deadline = step.max_dwell.map(|d| start + d);
    let mut frames = Vec::new();
    let mut cancelled_by_dwell = false;

    let query_result: Result<(), ClientError> = {
        let query_fut = client.query(&step.query, sink);
        tokio::pin!(query_fut);

        loop {
            tokio::select! {
                biased;
                maybe_frame = rx.recv() => {
                    if let Some(frame) = maybe_frame {
                        frames.push((start.elapsed(), frame));
                    } else {
                        break Ok(());
                    }
                }
                res = &mut query_fut => {
                    break res;
                }
                _ = async {
                    if let Some(at) = deadline {
                        tokio::time::sleep_until(at).await;
                    } else {
                        std::future::pending::<()>().await;
                    }
                } => {
                    cancelled_by_dwell = true;
                    break Ok(());
                }
            }
        }
        // Dropping `query_fut` here cancels any in-flight `ByteRangeStore` futures,
        // triggering `SimulatedTransport`'s cancellation drop guard.
    };

    // Drain any frames already queued right as `query_fut` completed.
    while let Ok(frame) = rx.try_recv() {
        frames.push((start.elapsed(), frame));
    }

    match query_result {
        Ok(()) | Err(ClientError::Cancelled) => {}
        Err(err) => return Err(format!("step '{}' failed: {err}", step.label)),
    }

    let completed = !cancelled_by_dwell
        && frames
            .last()
            .is_some_and(|(_, f)| f.status == FrameStatus::Complete);

    Ok(StepOutcome { frames, completed })
}

/// Run one [`Workload`] against a freshly opened client wrapped in [`SimulatedTransport`].
///
/// When `ground_truth` is `Some(points)`, the harness evaluates density NMAE (%) for every
/// emitted frame (tracking progressive convergence from coarse to fine).
pub async fn run_workload<P, S>(
    strategy: &P,
    store: S,
    workload: &Workload,
    config: &HarnessConfig,
    ground_truth: Option<&[Point]>,
) -> Result<WorkloadReport, String>
where
    P: ProgressiveStrategy,
    S: ByteRangeStore + 'static,
{
    let transport = Arc::new(SimulatedTransport::new(store, config.network));
    let before_open = transport.metrics();
    let mut client = strategy
        .open_client(Arc::clone(&transport), config.client)
        .await
        .map_err(|e| format!("open_client failed: {e}"))?;
    let after_open = transport.metrics();
    let open_transport = after_open.delta_since(&before_open);

    let mut peak_client_memory_bytes = client.memory_bytes();
    let mut step_reports = Vec::with_capacity(workload.steps.len());

    for (idx, step) in workload.steps.iter().enumerate() {
        let before_step = transport.metrics();
        let outcome = run_step(&mut client, step).await?;
        let after_step = transport.metrics();
        let step_transport = after_step.delta_since(&before_step);

        let mem = client.memory_bytes();
        peak_client_memory_bytes = peak_client_memory_bytes.max(mem);

        let ttff = outcome.frames.first().map(|(t, _)| *t);
        let ttc = outcome
            .frames
            .iter()
            .find(|(_, f)| f.status == FrameStatus::Complete)
            .map(|(t, _)| *t);
        let final_kernel_count = outcome
            .frames
            .last()
            .map(|(_, f)| f.kernel_count())
            .unwrap_or(0);

        let nmae_progression = if let Some(points) = ground_truth {
            compute_nmae_progression(
                points,
                step,
                &outcome.frames,
                config.nmae_pixel_samples,
                0xA11CE_u64.wrapping_add(idx as u64),
            )?
        } else {
            Vec::new()
        };

        step_reports.push(StepReport {
            label: step.label.clone(),
            frames_emitted: outcome.frames.len(),
            completed: outcome.completed,
            ttff,
            ttc,
            final_kernel_count,
            client_memory_bytes: mem,
            transport: step_transport,
            nmae_progression,
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
pub async fn run_benchmark<P, S>(
    strategy: &P,
    stream: DatasetStream,
    mut store: S,
    workloads: &[Workload],
    config: &HarnessConfig,
    ground_truth: Option<&[Point]>,
) -> Result<BenchmarkReport, String>
where
    P: ProgressiveStrategy,
    S: ByteRangeStore + ObjectWriter + Clone + 'static,
{
    let build_start = std::time::Instant::now();
    let mut builder = strategy.create_builder();
    let total_points =
        stream_into_builder(stream, &mut builder).map_err(|e| format!("build push failed: {e}"))?;
    builder
        .finish(&mut store)
        .map_err(|e| format!("build finish failed: {e}"))?;
    let build_duration = build_start.elapsed();

    let objects = store
        .list_objects()
        .await
        .map_err(|e| format!("list_objects failed: {e}"))?;
    let storage_bytes = objects.iter().map(|o| o.len).sum();
    let object_count = objects.len();

    let mut workload_reports = Vec::with_capacity(workloads.len());
    for workload in workloads {
        let report =
            run_workload(strategy, store.clone(), workload, config, ground_truth).await?;
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

fn compute_nmae_progression(
    points: &[Point],
    step: &WorkloadStep,
    frames: &[(Duration, Frame)],
    sample_count: usize,
    seed: u64,
) -> Result<Vec<f64>, String> {
    if frames.is_empty() || sample_count == 0 {
        return Ok(Vec::new());
    }
    let Some(y_range) = exact_visible_y_range(points, step.query.min_x, step.query.max_x) else {
        return Ok(Vec::new());
    };
    let samples = sample_reference_pixels(points, &step.query, y_range, sample_count, seed);
    if samples.is_empty() {
        return Ok(Vec::new());
    }
    let visible_count = points
        .iter()
        .filter(|p| p.x >= step.query.min_x && p.x <= step.query.max_x)
        .count();

    let mut progression = Vec::with_capacity(frames.len());
    for (_, frame) in frames {
        let q = evaluate_density_nmae(visible_count, frame, &step.query, y_range, &samples)?;
        progression.push(q.normalized_mean_absolute_error_percent);
    }
    Ok(progression)
}
