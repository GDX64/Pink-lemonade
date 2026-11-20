//! Reusable conformance test suite for validating any [`ProgressiveStrategy`].
//!
//! Plug any strategy into [`run_conformance_suite`] from a `#[tokio::test]` to verify:
//! - Empty dataset and empty-interval queries
//! - Moment preservation (mass, 1st, 2nd moments) on full and clipped views
//! - Exact visible `y_range` reporting
//! - `line_points` invariants (sorted x, visible extrema preservation)
//! - Density NMAE against ground truth
//! - Client memory budget compliance (`memory_bytes() <= memory_budget_bytes`)
//! - Mid-flight query cancellation safety
//! - Parity across both [`MemoryStore`] and [`DiskStore`]

use std::sync::Arc;
use std::time::Duration;

use pink_lemonade_progressive_core::{
    ByteRangeStore, ClientConfig, Frame, ObjectWriter, Point, ProgressiveClient,
    ProgressiveStrategy, ViewQuery,
};

use crate::fixture::{DatasetShape, DatasetStream, stream_into_builder};
use crate::moments::assert_moments_preserved;
use crate::quality::{
    evaluate_density_nmae, exact_visible_y_range, sample_reference_pixels, verify_line_points,
    verify_y_range,
};
use crate::runner::run_step;
use crate::store::{BenchmarkStore, DiskStore, MemoryStore};
use crate::transport::{NetworkProfile, SimulatedTransport};
use crate::workload::WorkloadStep;

/// Configurable tolerances and sizes for [`run_conformance_suite`].
#[derive(Debug, Clone)]
pub struct ConformanceConfig {
    pub point_count: u64,
    pub chunk_size: usize,
    pub screen_w: f64,
    pub screen_h: f64,
    pub sigma_px: f64,
    pub merge_threshold: f64,
    pub memory_budget_bytes: usize,
    /// Relative tolerance for mass / 1st / 2nd moment preservation (default `2e-4`).
    pub moment_rel_tol: f64,
    /// Absolute tolerance for visible `y_range` bounds (default `1e-6`).
    pub y_range_abs_tol: f64,
    /// Upper bound on density NMAE (%) (default `5.0`).
    pub max_nmae_percent: f64,
    /// Number of occupied pixels sampled for density NMAE evaluation.
    pub nmae_pixel_samples: usize,
    /// Whether to require non-empty `line_points` on non-empty views.
    pub require_line_points: bool,
}

impl Default for ConformanceConfig {
    fn default() -> Self {
        Self {
            point_count: 4_096,
            chunk_size: 512,
            screen_w: 1920.0,
            screen_h: 1080.0,
            sigma_px: 16.0,
            merge_threshold: 1.0,
            memory_budget_bytes: 8 * 1024 * 1024,
            moment_rel_tol: 2e-4,
            y_range_abs_tol: 1e-6,
            max_nmae_percent: 5.0,
            nmae_pixel_samples: 128,
            require_line_points: false,
        }
    }
}

/// Run the full conformance suite against `strategy`.
///
/// Returns `Ok(())` when all invariants pass, or a descriptive failure message.
pub async fn run_conformance_suite(
    strategy: &dyn ProgressiveStrategy,
    config: &ConformanceConfig,
) -> Result<(), String> {
    check_empty_dataset(strategy, config).await?;
    check_dataset_on_memory_store(strategy, config).await?;
    check_cancellation_safety(strategy, config).await?;
    check_disk_store_parity(strategy, config).await?;
    Ok(())
}

async fn check_empty_dataset(
    strategy: &dyn ProgressiveStrategy,
    config: &ConformanceConfig,
) -> Result<(), String> {
    let mut store = MemoryStore::new();
    let builder = strategy.create_builder();
    builder
        .finish(&mut store)
        .map_err(|e| format!("empty build failed: {e}"))?;

    let transport = Arc::new(SimulatedTransport::new(
        store.reader(),
        NetworkProfile::instant(),
    ));
    let mut client = strategy
        .open_client(
            transport,
            ClientConfig {
                memory_budget_bytes: config.memory_budget_bytes,
            },
        )
        .await
        .map_err(|e| format!("empty open_client failed: {e}"))?;

    let query = ViewQuery::new(
        0.0,
        1.0,
        config.screen_w,
        config.screen_h,
        config.sigma_px,
        config.merge_threshold,
    );
    let step = WorkloadStep {
        label: "empty_dataset".to_owned(),
        query,
        max_dwell: None,
    };
    let outcome = run_step(client.as_mut(), &step).await?;
    let frame = outcome
        .frame
        .as_ref()
        .ok_or_else(|| "empty dataset returned no frame".to_owned())?;

    if frame.kernel_count() != 0 {
        return Err(format!(
            "empty dataset expected 0 kernels, got {}",
            frame.kernel_count()
        ));
    }
    if frame.y_range.is_some() {
        return Err(format!(
            "empty dataset expected y_range=None, got {:?}",
            frame.y_range
        ));
    }
    check_memory_budget(client.as_ref(), config.memory_budget_bytes)?;
    Ok(())
}

async fn check_dataset_on_memory_store(
    strategy: &dyn ProgressiveStrategy,
    config: &ConformanceConfig,
) -> Result<(), String> {
    let shape = DatasetShape::NoisySpikes { seed: 42 };
    let points = DatasetStream::new(shape, config.point_count, config.chunk_size).collect_all();
    let mut store = MemoryStore::new();
    build_dataset(
        strategy,
        shape,
        config.point_count,
        config.chunk_size,
        &mut store,
    )?;

    let transport = Arc::new(SimulatedTransport::new(
        store.reader(),
        NetworkProfile::instant(),
    ));
    let mut client = strategy
        .open_client(
            transport,
            ClientConfig {
                memory_budget_bytes: config.memory_budget_bytes,
            },
        )
        .await
        .map_err(|e| format!("open_client failed: {e}"))?;

    let views = [
        ("full_view", 0.0, 1.0),
        ("left_slice", 0.05, 0.35),
        ("middle_slice", 0.30, 0.62),
        ("zoomed_slice", 0.72, 0.80),
        ("empty_out_of_range", 2.0, 3.0),
    ];

    for (label, min_x, max_x) in views {
        let query = ViewQuery::new(
            min_x,
            max_x,
            config.screen_w,
            config.screen_h,
            config.sigma_px,
            config.merge_threshold,
        );
        let step = WorkloadStep {
            label: label.to_owned(),
            query,
            max_dwell: None,
        };
        let outcome = run_step(client.as_mut(), &step).await?;
        let frame = outcome
            .frame
            .as_ref()
            .ok_or_else(|| format!("{label}: returned no frame"))?;

        verify_frame(label, &points, &query, frame, config)?;
        check_memory_budget(client.as_ref(), config.memory_budget_bytes)?;
    }

    Ok(())
}

async fn check_cancellation_safety(
    strategy: &dyn ProgressiveStrategy,
    config: &ConformanceConfig,
) -> Result<(), String> {
    let shape = DatasetShape::SineWave;
    let points = DatasetStream::new(shape, config.point_count, config.chunk_size).collect_all();
    let mut store = MemoryStore::new();
    build_dataset(
        strategy,
        shape,
        config.point_count,
        config.chunk_size,
        &mut store,
    )?;

    let transport = Arc::new(SimulatedTransport::new(
        store.reader(),
        NetworkProfile::broadband(),
    ));
    let mut client = strategy
        .open_client(
            transport,
            ClientConfig {
                memory_budget_bytes: config.memory_budget_bytes,
            },
        )
        .await
        .map_err(|e| format!("cancellation open_client failed: {e}"))?;

    let interrupted = WorkloadStep {
        label: "interrupted_view".to_owned(),
        query: ViewQuery::new(
            0.1,
            0.4,
            config.screen_w,
            config.screen_h,
            config.sigma_px,
            config.merge_threshold,
        ),
        max_dwell: Some(Duration::from_millis(5)),
    };
    let _ = run_step(client.as_mut(), &interrupted).await?;

    let settled_query = ViewQuery::new(
        0.5,
        0.8,
        config.screen_w,
        config.screen_h,
        config.sigma_px,
        config.merge_threshold,
    );
    let settled = WorkloadStep {
        label: "post_cancel_settled".to_owned(),
        query: settled_query,
        max_dwell: None,
    };
    let outcome = run_step(client.as_mut(), &settled).await?;
    let frame = outcome
        .frame
        .as_ref()
        .ok_or_else(|| "post_cancel_settled returned no frame".to_owned())?;

    verify_frame("post_cancel_settled", &points, &settled_query, frame, config)?;
    check_memory_budget(client.as_ref(), config.memory_budget_bytes)?;
    Ok(())
}

async fn check_disk_store_parity(
    strategy: &dyn ProgressiveStrategy,
    config: &ConformanceConfig,
) -> Result<(), String> {
    let unique = format!(
        "pink_lemonade_progressive_conformance_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let temp_dir = std::env::temp_dir().join(unique);
    let mut disk = DiskStore::open(&temp_dir).map_err(|e| format!("DiskStore::open failed: {e}"))?;

    let shape = DatasetShape::SineWave;
    let points = DatasetStream::new(shape, config.point_count, config.chunk_size).collect_all();
    let build_res = build_dataset(
        strategy,
        shape,
        config.point_count,
        config.chunk_size,
        &mut disk,
    );
    if let Err(err) = build_res {
        let _ = disk.remove_dir_all();
        return Err(err);
    }

    let objects = disk
        .list_objects()
        .await
        .map_err(|e| format!("DiskStore::list_objects failed: {e}"))?;
    if objects.is_empty() && config.point_count > 0 {
        let _ = disk.remove_dir_all();
        return Err("strategy wrote zero objects for non-empty dataset".to_owned());
    }

    let transport = Arc::new(SimulatedTransport::new(
        disk.reader(),
        NetworkProfile::instant(),
    ));
    let query = ViewQuery::new(
        0.2,
        0.7,
        config.screen_w,
        config.screen_h,
        config.sigma_px,
        config.merge_threshold,
    );
    let check_result = async {
        let mut client = strategy
            .open_client(
                transport,
                ClientConfig {
                    memory_budget_bytes: config.memory_budget_bytes,
                },
            )
            .await
            .map_err(|e| format!("DiskStore open_client failed: {e}"))?;
        let step = WorkloadStep {
            label: "disk_parity".to_owned(),
            query,
            max_dwell: None,
        };
        let outcome = run_step(client.as_mut(), &step).await?;
        let frame = outcome
            .frame
            .as_ref()
            .ok_or_else(|| "disk_parity returned no frame".to_owned())?;
        verify_frame("disk_parity", &points, &query, frame, config)?;
        check_memory_budget(client.as_ref(), config.memory_budget_bytes)?;
        Ok::<(), String>(())
    }
    .await;

    let _ = disk.remove_dir_all();
    check_result
}

fn build_dataset(
    strategy: &dyn ProgressiveStrategy,
    shape: DatasetShape,
    point_count: u64,
    chunk_size: usize,
    writer: &mut dyn ObjectWriter,
) -> Result<(), String> {
    let mut builder = strategy.create_builder();
    let stream = DatasetStream::new(shape, point_count, chunk_size);
    stream_into_builder(stream, builder.as_mut())
        .map_err(|e| format!("push_chunk failed: {e}"))?;
    builder
        .finish(writer)
        .map_err(|e| format!("finish failed: {e}"))?;
    Ok(())
}

fn verify_frame(
    label: &str,
    points: &[Point],
    query: &ViewQuery,
    frame: &Frame,
    config: &ConformanceConfig,
) -> Result<(), String> {
    verify_y_range(points, query, frame, config.y_range_abs_tol)
        .map_err(|e| format!("{label}: {e}"))?;

    let expected_y = exact_visible_y_range(points, query.min_x, query.max_x);
    let Some(y_range) = expected_y else {
        if frame.kernel_count() != 0 {
            return Err(format!(
                "{label}: empty view expected 0 kernels, got {}",
                frame.kernel_count()
            ));
        }
        return Ok(());
    };

    assert_moments_preserved(
        label,
        points,
        query,
        y_range,
        frame,
        config.moment_rel_tol,
    )?;

    if config.require_line_points && frame.line_points.is_empty() {
        return Err(format!("{label}: expected non-empty line_points"));
    }
    verify_line_points(points, query, frame).map_err(|e| format!("{label}: {e}"))?;

    let samples = sample_reference_pixels(points, query, y_range, config.nmae_pixel_samples, 0x5eed);
    if !samples.is_empty() {
        let visible_count = points
            .iter()
            .filter(|p| p.x >= query.min_x && p.x <= query.max_x)
            .count();
        let quality = evaluate_density_nmae(visible_count, frame, query, y_range, &samples)
            .map_err(|e| format!("{label}: {e}"))?;
        if quality.normalized_mean_absolute_error_percent > config.max_nmae_percent {
            return Err(format!(
                "{label}: density NMAE {:.3}% exceeded limit {:.3}%",
                quality.normalized_mean_absolute_error_percent, config.max_nmae_percent
            ));
        }
    }

    Ok(())
}

fn check_memory_budget(client: &dyn ProgressiveClient, budget: usize) -> Result<(), String> {
    let used = client.memory_bytes();
    if used > budget {
        return Err(format!(
            "client memory_bytes ({used}) exceeded memory_budget_bytes ({budget})"
        ));
    }
    Ok(())
}
