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
    /// 10M points (~240 MiB raw, `MultiRegime`), `DiskStore`, broadband, 16 MiB client budget.
    Heavy,
    /// 50M points (~1.2 GiB raw, `MultiRegime`), `DiskStore`, broadband, 32 MiB client budget.
    Gigabyte,
    /// 1M points (`NoisySpikes`), `MemoryStore`, 5 Mbps / 100 ms mobile link, 2 MiB client budget.
    ConstrainedMobile,
    /// User-defined scenario specification.
    Custom(Box<ScenarioSpec>),
}

impl BenchmarkScenario {
    /// Expand this preset into a concrete [`ScenarioSpec`] (can be modified and wrapped in `Custom`).
    pub fn spec(&self) -> ScenarioSpec {
        match self {
            Self::Light => {
                let harness = HarnessConfig {
                    network: NetworkProfile::broadband(),
                    client: ClientConfig {
                        memory_budget_bytes: 1024 * 1024, // 1 MiB
                    },
                    ..HarnessConfig::default()
                };
                ScenarioSpec {
                    name: "light".to_owned(),
                    shape: DatasetShape::SineWave,
                    point_count: 10_000,
                    chunk_size: 1_000,
                    backing: StoreBacking::Memory,
                    workloads: Workload::standard_suite(
                        harness.screen_w,
                        harness.screen_h,
                        harness.sigma_px,
                        harness.merge_threshold,
                    ),
                    harness,
                    evaluate_nmae: false,
                }
            }
            Self::Medium => {
                let harness = HarnessConfig {
                    network: NetworkProfile::broadband(),
                    client: ClientConfig {
                        memory_budget_bytes: 4 * 1024 * 1024, // 4 MiB (< 24 MiB raw)
                    },
                    ..HarnessConfig::default()
                };
                ScenarioSpec {
                    name: "medium".to_owned(),
                    shape: DatasetShape::SineWave,
                    point_count: 1_000_000,
                    chunk_size: 65_536,
                    backing: StoreBacking::Memory,
                    workloads: Workload::standard_suite(
                        harness.screen_w,
                        harness.screen_h,
                        harness.sigma_px,
                        harness.merge_threshold,
                    ),
                    harness,
                    evaluate_nmae: false,
                }
            }
            Self::Heavy => {
                let harness = HarnessConfig {
                    network: NetworkProfile::broadband(),
                    client: ClientConfig {
                        memory_budget_bytes: 16 * 1024 * 1024, // 16 MiB (< 240 MiB raw)
                    },
                    ..HarnessConfig::default()
                };
                ScenarioSpec {
                    name: "heavy".to_owned(),
                    shape: DatasetShape::MultiRegime { seed: 0x51a7_cafe },
                    point_count: 10_000_000,
                    chunk_size: 131_072,
                    backing: StoreBacking::Disk,
                    workloads: Workload::standard_suite(
                        harness.screen_w,
                        harness.screen_h,
                        harness.sigma_px,
                        harness.merge_threshold,
                    ),
                    harness,
                    evaluate_nmae: false,
                }
            }
            Self::Gigabyte => {
                let harness = HarnessConfig {
                    network: NetworkProfile::broadband(),
                    client: ClientConfig {
                        memory_budget_bytes: 32 * 1024 * 1024, // 32 MiB (< 1.2 GiB raw)
                    },
                    ..HarnessConfig::default()
                };
                ScenarioSpec {
                    name: "gigabyte".to_owned(),
                    shape: DatasetShape::MultiRegime { seed: 0x616a_b77e },
                    point_count: 50_000_000,
                    chunk_size: 262_144,
                    backing: StoreBacking::Disk,
                    workloads: Workload::standard_suite(
                        harness.screen_w,
                        harness.screen_h,
                        harness.sigma_px,
                        harness.merge_threshold,
                    ),
                    harness,
                    evaluate_nmae: false,
                }
            }
            Self::ConstrainedMobile => {
                let harness = HarnessConfig {
                    network: NetworkProfile::mobile_5mbps(),
                    client: ClientConfig {
                        memory_budget_bytes: 2 * 1024 * 1024, // 2 MiB
                    },
                    ..HarnessConfig::default()
                };
                ScenarioSpec {
                    name: "constrained_mobile".to_owned(),
                    shape: DatasetShape::NoisySpikes { seed: 0x4242 },
                    point_count: 1_000_000,
                    chunk_size: 65_536,
                    backing: StoreBacking::Memory,
                    workloads: Workload::standard_suite(
                        harness.screen_w,
                        harness.screen_h,
                        harness.sigma_px,
                        harness.merge_threshold,
                    ),
                    harness,
                    evaluate_nmae: false,
                }
            }
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
