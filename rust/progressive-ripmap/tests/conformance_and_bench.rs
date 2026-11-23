use pink_lemonade_progressive_core::ProgressiveStrategy;
use pink_lemonade_progressive_harness::benchmarks::{self, BenchmarkScenario};
use pink_lemonade_progressive_harness::{ConformanceConfig, run_conformance_suite};
use pink_lemonade_progressive_ripmap::RipmapStrategy;
use pink_lemonade_progressive_tree::PagedTreeStrategy;

#[tokio::test(start_paused = true)]
async fn passes_full_conformance_suite() {
    let config = ConformanceConfig {
        require_line_points: true,
        ..ConformanceConfig::default()
    };
    run_conformance_suite(&RipmapStrategy::default(), &config)
        .await
        .expect("RipmapStrategy should satisfy all conformance invariants");
}

/// 200K points makes x-blocks wide enough (thousands of points) that views are rendered
/// through moment planes at several `(ℓ, k)` depths rather than raw leaves.
#[tokio::test(start_paused = true)]
async fn passes_conformance_through_planes() {
    let config = ConformanceConfig {
        point_count: 200_000,
        chunk_size: 8_192,
        require_line_points: true,
        ..ConformanceConfig::default()
    };
    run_conformance_suite(&RipmapStrategy::default(), &config)
        .await
        .expect("RipmapStrategy should satisfy conformance at 200K points");
}

/// Small `root_cap` forces multi-level jumping (`levels.bin`), boundary subtree reads,
/// and sparse plane caching even at 50K points.
#[tokio::test(start_paused = true)]
async fn passes_conformance_with_small_root_cap() {
    let strategy = RipmapStrategy {
        root_cap: 16,
        ..RipmapStrategy::default()
    };
    let config = ConformanceConfig {
        point_count: 50_000,
        chunk_size: 4_096,
        require_line_points: true,
        ..ConformanceConfig::default()
    };
    run_conformance_suite(&strategy, &config)
        .await
        .expect("RipmapStrategy with small root_cap should satisfy all conformance invariants");
}

/// Bursty non-uniform x (`BurstyX`) must keep overview wire tiny (regression guard against
/// one wide gap-straddling node dragging all burst nodes down to level 0).
#[tokio::test(start_paused = true)]
async fn bursty_x_overview_stays_compact() {
    let report = benchmarks::run_on(&RipmapStrategy::default(), BenchmarkScenario::MediumBursty)
        .await
        .expect("medium_bursty benchmark failed");
    let overview = report
        .workloads
        .iter()
        .find(|w| w.name == "cold_overview")
        .unwrap();
    let wire = overview.total_transport.total_wire_bytes();
    assert!(
        wire < 1024 * 1024,
        "medium_bursty cold_overview should stay well under 1 MiB, got {wire} B"
    );
}

#[tokio::test(start_paused = true)]
async fn beats_x_tree_on_white_noise() {
    let ripmap = RipmapStrategy::default();
    let tree = PagedTreeStrategy::default();
    let strategies: [&dyn ProgressiveStrategy; 2] = [&tree, &ripmap];
    let reports = benchmarks::compare(&strategies, BenchmarkScenario::MediumWhiteNoise)
        .await
        .expect("medium white-noise comparison failed");
    for report in &reports {
        println!("\n{report}");
    }

    let wire = |r: usize, name: &str| {
        reports[r]
            .workloads
            .iter()
            .find(|w| w.name == name)
            .unwrap()
            .total_transport
            .total_wire_bytes()
    };
    // At 1M points a 5% window is only ~1.2 MiB raw, so windowed workloads stay near raw for
    // every strategy; the whole-range workloads are where x-only partitioning collapses.
    for name in ["cold_overview", "progressive_zoom_in"] {
        assert!(
            wire(1, name) * 2 < wire(0, name),
            "{name}: ripmap ({} B) should transfer < half of the x-tree ({} B) on white noise",
            wire(1, name),
            wire(0, name)
        );
    }
    for w in &reports[1].workloads {
        assert!(w.peak_client_memory_bytes <= reports[1].memory_budget_bytes);
    }
}

/// Every step of every standard workload (including explicit `y_zoom` views) stays within the
/// 5% density NMAE tolerance for the x-tree, k-d tree, and ripmap on both signal families.
///
/// Set `PLP_NMAE_POINTS` (e.g. `1000000`, with `--profile fast`) to check at sizes where most
/// windowed views are served from ripmap planes instead of raw leaves.
#[tokio::test(start_paused = true)]
async fn density_nmae_within_tolerance_on_all_workloads() {
    use pink_lemonade_progressive_harness::DatasetShape;
    use pink_lemonade_progressive_kdtree::KdTreeStrategy;

    let tree = PagedTreeStrategy::default();
    let kd = KdTreeStrategy::default();
    let ripmap = RipmapStrategy::default();
    let strategies: [&dyn ProgressiveStrategy; 3] = [&tree, &kd, &ripmap];

    for (base, shape) in [
        (BenchmarkScenario::MediumWhiteNoise, DatasetShape::NoisySpikes { seed: 4_242 }),
        (BenchmarkScenario::Medium, DatasetShape::RandomWalk { seed: 578_211 }),
    ] {
        let mut spec = base.spec();
        spec.shape = shape;
        spec.point_count = std::env::var("PLP_NMAE_POINTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100_000);
        spec.chunk_size = 8_192;
        spec.evaluate_nmae = true;
        spec.harness.nmae_pixel_samples = 48;
        let reports = benchmarks::compare(&strategies, BenchmarkScenario::Custom(Box::new(spec)))
            .await
            .expect("NMAE comparison failed");

        for report in &reports {
            for w in &report.workloads {
                let worst = w
                    .steps
                    .iter()
                    .filter_map(|s| s.nmae_percent)
                    .fold(0.0_f64, f64::max);
                println!(
                    "{:>18} {:>12} {:<20} worst NMAE {worst:.3}%",
                    report.strategy_name,
                    shape.name(),
                    w.name
                );
                assert!(
                    worst <= 5.0,
                    "{} on {} / {}: NMAE {worst:.3}% > 5%",
                    report.strategy_name,
                    shape.name(),
                    w.name
                );
            }
        }
    }
}
