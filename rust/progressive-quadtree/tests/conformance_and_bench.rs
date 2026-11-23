use pink_lemonade_progressive_core::ProgressiveStrategy;
use pink_lemonade_progressive_harness::benchmarks::{self, BenchmarkScenario};
use pink_lemonade_progressive_harness::{ConformanceConfig, DatasetShape, run_conformance_suite};
use pink_lemonade_progressive_kdtree::KdTreeStrategy;
use pink_lemonade_progressive_quadtree::QuadtreeStrategy;
use pink_lemonade_progressive_ripmap::RipmapStrategy;
use pink_lemonade_progressive_tree::PagedTreeStrategy;

#[tokio::test(start_paused = true)]
async fn passes_full_conformance_suite() {
    let config = ConformanceConfig {
        require_line_points: true,
        ..ConformanceConfig::default()
    };
    run_conformance_suite(&QuadtreeStrategy::default(), &config)
        .await
        .expect("QuadtreeStrategy should satisfy all conformance invariants");
}

/// 200K points makes the tree deep enough that remote depths (beyond the inlined top) and
/// resolved interior nodes are exercised.
#[tokio::test(start_paused = true)]
async fn passes_conformance_on_deep_tree() {
    let config = ConformanceConfig {
        point_count: 200_000,
        chunk_size: 8_192,
        require_line_points: true,
        ..ConformanceConfig::default()
    };
    run_conformance_suite(&QuadtreeStrategy::default(), &config)
        .await
        .expect("QuadtreeStrategy should satisfy conformance at 200K points");
}

/// Inlining only the root puts every other depth in `nodes.bin`, exercising per-depth offsets
/// from the very first descent.
#[tokio::test(start_paused = true)]
async fn passes_conformance_with_only_the_root_inlined() {
    let strategy = QuadtreeStrategy {
        root_cap: 1,
        ..QuadtreeStrategy::default()
    };
    let config = ConformanceConfig {
        point_count: 50_000,
        chunk_size: 4_096,
        require_line_points: true,
        ..ConformanceConfig::default()
    };
    run_conformance_suite(&strategy, &config)
        .await
        .expect("QuadtreeStrategy should satisfy conformance with only the root inlined");
}

/// Density NMAE stays within 5% on every standard workload for both signal families.
#[tokio::test(start_paused = true)]
async fn density_nmae_within_tolerance_on_all_workloads() {
    let quad = QuadtreeStrategy::default();
    let strategies: [&dyn ProgressiveStrategy; 1] = [&quad];
    for (base, shape) in [
        (BenchmarkScenario::MediumWhiteNoise, DatasetShape::NoisySpikes { seed: 4_242 }),
        (BenchmarkScenario::Medium, DatasetShape::RandomWalk { seed: 578_211 }),
    ] {
        let mut spec = base.spec();
        spec.shape = shape;
        spec.point_count = 100_000;
        spec.chunk_size = 8_192;
        spec.evaluate_nmae = true;
        spec.harness.nmae_pixel_samples = 48;
        let reports = benchmarks::compare(&strategies, BenchmarkScenario::Custom(Box::new(spec)))
            .await
            .expect("NMAE run failed");
        for w in &reports[0].workloads {
            let worst = w.steps.iter().filter_map(|s| s.nmae_percent).fold(0.0_f64, f64::max);
            assert!(worst <= 5.0, "{} / {}: NMAE {worst:.3}% > 5%", shape.name(), w.name);
        }
    }
}

/// Side-by-side numbers against the other strategies (printed with `--nocapture`); only the
/// memory budget is asserted, since this baseline exists to be measured.
#[tokio::test(start_paused = true)]
async fn compare_with_other_strategies() {
    let tree = PagedTreeStrategy::default();
    let kd = KdTreeStrategy::default();
    let ripmap = RipmapStrategy::default();
    let quad = QuadtreeStrategy::default();
    let strategies: [&dyn ProgressiveStrategy; 4] = [&tree, &kd, &ripmap, &quad];
    for scenario in [BenchmarkScenario::Medium, BenchmarkScenario::MediumWhiteNoise] {
        let reports = benchmarks::compare(&strategies, scenario)
            .await
            .expect("comparison failed");
        for report in &reports {
            println!("\n{report}");
        }
        for w in &reports[3].workloads {
            assert!(w.peak_client_memory_bytes <= reports[3].memory_budget_bytes);
        }
    }
}
