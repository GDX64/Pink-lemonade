//! Standard benchmark scenarios and one-line entry points (`run_on`, `compare`).
//!
//! Example:
//! ```ignore
//! use pink_lemonade_progressive_harness::benchmarks::{self, BenchmarkScenario};
//!
//! let report = benchmarks::run_on(&MyStrategy, BenchmarkScenario::Light).await?;
//! println!("{report}");
//! ```

use std::time::{SystemTime, UNIX_EPOCH};

use pink_lemonade_progressive_core::{ClientConfig, ProgressiveStrategy};

use crate::fixture::{DatasetShape, DatasetStream};
use crate::report::BenchmarkReport;
use crate::runner::{HarnessConfig, run_benchmark};
use crate::store::{DiskStore, MemoryStore};
use crate::transport::NetworkProfile;
use crate::workload::Workload;

pub use crate::report::BenchmarkReport as Report;

/// Storage backend used when building and serving a benchmark scenario.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreBacking {
    /// Fast in-memory `MemoryStore` (suitable for Light / Medium scenarios).
    Memory,
    /// Streaming directory-backed `DiskStore` in a temporary directory, removed after the run.
    Disk,
}

/// Full specification of a benchmark scenario.
#[derive(Debug, Clone)]
pub struct ScenarioSpec {
    pub name: String,
    pub shape: DatasetShape,
    pub point_count: u64,
    pub chunk_size: usize,
    pub backing: StoreBacking,
    pub harness: HarnessConfig,
    pub workloads: Vec<Workload>,
    /// When `true`, collects ground-truth points in memory to compute density NMAE (%)
    /// on every step. Only enable on small/medium scenarios that fit in RAM.
    pub evaluate_nmae: bool,
}

/// Standard progressive benchmark scenarios spanning fast unit-test runs to GB-scale out-of-core runs.
#[derive(Debug, Clone)]
pub enum BenchmarkScenario {
    /// 10K points (~234 KiB raw), `MemoryStore`, broadband, 1 MiB client budget. Fast smoke benchmark.
    Light,
    /// 1M points (~24 MiB raw), `MemoryStore`, broadband, 4 MiB client budget (smaller than raw data).
    Medium,
    /// 1M points of uncorrelated `NoisySpikes` white noise, broadband, 4 MiB client budget.
    /// Exposes 1D x-partitioning's failure mode at a size cheap enough for `--fast` runs.
    MediumWhiteNoise,
    /// 10M points (~240 MiB raw, integrated Gaussian `RandomWalk`), `MemoryStore`, broadband, 16 MiB client budget.
    Heavy,
    /// 10M points (~240 MiB raw) of uncorrelated `NoisySpikes` white noise, `MemoryStore`,
    /// broadband, 16 MiB client budget (2D space-partitioning stress test).
    HeavyWhiteNoise,
    /// `Heavy` (10M `RandomWalk`) rendered with a 4 px kernel instead of 16 px: every view needs
    /// ~16× more kernels, exposing constants tuned to one kernel size.
    HeavySmallKernel,
    /// `HeavyWhiteNoise` (10M `NoisySpikes`) rendered with a 4 px kernel.
    HeavyWhiteNoiseSmallKernel,
    /// 1M points of `BurstyX` (90% of points in 10% of x, empty stretches), broadband, 4 MiB budget.
    MediumBursty,
    /// 10M points of `BurstyX`, broadband, 16 MiB budget: count-based vs geometric x-partitions.
    HeavyBursty,
    /// 50M points (~1.2 GiB raw, integrated Gaussian `RandomWalk`), `MemoryStore`, broadband, 32 MiB client budget.
    Gigabyte,
    /// 1M points (integrated Gaussian `RandomWalk`), `MemoryStore`, 5 Mbps / 100 ms mobile link, 2 MiB client budget.
    ConstrainedMobile,
    /// User-defined scenario specification.
    Custom(Box<ScenarioSpec>),
}

/// Vertical zoom center (fraction of auto-y range) for `NoisySpikes`: the `+80` spikes stretch
/// the range to `[-10, 90]`, so the dense noise band sits around `0.1`.
const NOISY_SPIKES_Y_CENTER: f64 = 0.1;
const DEFAULT_Y_CENTER: f64 = 0.5;

fn preset(
    name: &str,
    shape: DatasetShape,
    point_count: u64,
    chunk_size: usize,
    network: NetworkProfile,
    memory_budget_bytes: usize,
) -> ScenarioSpec {
    preset_with_sigma(
        name,
        shape,
        point_count,
        chunk_size,
        network,
        memory_budget_bytes,
        HarnessConfig::default().sigma_px,
    )
}

/// Kernel size used by the `*SmallKernel` scenarios (vs the default 16 px).
pub const SMALL_KERNEL_SIGMA_PX: f64 = 4.0;

fn preset_with_sigma(
    name: &str,
    shape: DatasetShape,
    point_count: u64,
    chunk_size: usize,
    network: NetworkProfile,
    memory_budget_bytes: usize,
    sigma_px: f64,
) -> ScenarioSpec {
    let harness = HarnessConfig {
        network,
        client: ClientConfig {
            memory_budget_bytes,
        },
        sigma_px,
        ..HarnessConfig::default()
    };
    let y_center = match shape {
        DatasetShape::NoisySpikes { .. } => NOISY_SPIKES_Y_CENTER,
        _ => DEFAULT_Y_CENTER,
    };
    ScenarioSpec {
        name: name.to_owned(),
        shape,
        point_count,
        chunk_size,
        backing: StoreBacking::Memory,
        workloads: Workload::standard_suite(
            harness.screen_w,
            harness.screen_h,
            harness.sigma_px,
            harness.merge_threshold,
            y_center,
        ),
        harness,
        evaluate_nmae: false,
    }
}

const MIB: usize = 1024 * 1024;

impl BenchmarkScenario {
    /// Expand this preset into a concrete [`ScenarioSpec`] (can be modified and wrapped in `Custom`).
    pub fn spec(&self) -> ScenarioSpec {
        let walk = DatasetShape::RandomWalk { seed: 578_211 };
        let noise = DatasetShape::NoisySpikes { seed: 4_242 };
        let bursty = DatasetShape::BurstyX { seed: 9_001 };
        let broadband = NetworkProfile::broadband();
        match self {
            Self::Light => preset("light", DatasetShape::SineWave, 10_000, 1_000, broadband, MIB),
            Self::Medium => preset(
                "medium",
                DatasetShape::SineWave,
                1_000_000,
                65_536,
                broadband,
                4 * MIB,
            ),
            Self::MediumWhiteNoise => preset(
                "medium_white_noise",
                noise,
                1_000_000,
                65_536,
                broadband,
                4 * MIB,
            ),
            Self::Heavy => preset("heavy", walk, 10_000_000, 131_072, broadband, 16 * MIB),
            Self::HeavyWhiteNoise => preset(
                "heavy_white_noise",
                noise,
                10_000_000,
                131_072,
                broadband,
                16 * MIB,
            ),
            Self::HeavySmallKernel => preset_with_sigma(
                "heavy_sigma4",
                walk,
                10_000_000,
                131_072,
                broadband,
                16 * MIB,
                SMALL_KERNEL_SIGMA_PX,
            ),
            Self::HeavyWhiteNoiseSmallKernel => preset_with_sigma(
                "heavy_white_noise_sigma4",
                noise,
                10_000_000,
                131_072,
                broadband,
                16 * MIB,
                SMALL_KERNEL_SIGMA_PX,
            ),
            Self::MediumBursty => preset("medium_bursty", bursty, 1_000_000, 65_536, broadband, 4 * MIB),
            Self::HeavyBursty => preset("heavy_bursty", bursty, 10_000_000, 131_072, broadband, 16 * MIB),
            Self::Gigabyte => preset("gigabyte", walk, 50_000_000, 262_144, broadband, 32 * MIB),
            Self::ConstrainedMobile => preset(
                "constrained_mobile",
                walk,
                1_000_000,
                65_536,
                NetworkProfile::mobile_5mbps(),
                2 * MIB,
            ),
            Self::Custom(spec) => (**spec).clone(),
        }
    }
}

/// Run a standard [`BenchmarkScenario`] against `strategy` and return its [`Report`].
///
/// For fast deterministic virtual-time network simulation, call inside
/// `#[tokio::test(start_paused = true)]` or after `tokio::time::pause()`.
pub async fn run_on(
    strategy: &dyn ProgressiveStrategy,
    scenario: BenchmarkScenario,
) -> Result<Report, String> {
    let spec = scenario.spec();
    let ground_truth = if spec.evaluate_nmae {
        Some(DatasetStream::new(spec.shape, spec.point_count, spec.chunk_size).collect_all())
    } else {
        None
    };
    let stream = DatasetStream::new(spec.shape, spec.point_count, spec.chunk_size);

    let mut report = match spec.backing {
        StoreBacking::Memory => {
            let mut store = MemoryStore::new();
            run_benchmark(
                strategy,
                stream,
                &mut store,
                &spec.workloads,
                &spec.harness,
                ground_truth.as_deref(),
            )
            .await?
        }
        StoreBacking::Disk => {
            let unique = format!(
                "pink_lemonade_bench_{}_{}_{}",
                spec.name,
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0)
            );
            let dir = std::env::temp_dir().join(unique);
            let mut store =
                DiskStore::open(&dir).map_err(|e| format!("DiskStore::open failed: {e}"))?;
            let result = run_benchmark(
                strategy,
                stream,
                &mut store,
                &spec.workloads,
                &spec.harness,
                ground_truth.as_deref(),
            )
            .await;
            let _ = store.remove_dir_all();
            result?
        }
    };

    report.scenario_name = Some(spec.name);
    report.dataset_shape = Some(spec.shape.name().to_owned());
    Ok(report)
}

/// Run the same [`BenchmarkScenario`] across multiple strategies for side-by-side comparison.
pub async fn compare(
    strategies: &[&dyn ProgressiveStrategy],
    scenario: BenchmarkScenario,
) -> Result<Vec<BenchmarkReport>, String> {
    let mut reports = Vec::with_capacity(strategies.len());
    for strategy in strategies {
        reports.push(run_on(*strategy, scenario.clone()).await?);
    }
    Ok(reports)
}
