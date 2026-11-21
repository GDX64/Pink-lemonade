use pink_lemonade_progressive_core::ProgressiveStrategy;
use pink_lemonade_progressive_harness::benchmarks::{self, BenchmarkScenario};
use pink_lemonade_progressive_harness::{ConformanceConfig, run_conformance_suite};
use pink_lemonade_progressive_kdtree::KdTreeStrategy;
use pink_lemonade_progressive_tree::PagedTreeStrategy;

#[tokio::test(start_paused = true)]
async fn passes_full_conformance_suite() {
    let config = ConformanceConfig {
        require_line_points: true,
        ..ConformanceConfig::default()
    };
    run_conformance_suite(&KdTreeStrategy::default(), &config)
        .await
        .expect("KdTreeStrategy should satisfy all conformance invariants");
}

/// 200K points spans several remote k-d levels, so interior cells are genuinely resolved
/// (not just raw leaves) under both auto-y and explicit y-bounds views.
#[tokio::test(start_paused = true)]
async fn passes_conformance_on_multi_level_tree() {
    let config = ConformanceConfig {
        point_count: 200_000,
        chunk_size: 8_192,
        require_line_points: true,
        ..ConformanceConfig::default()
    };
    run_conformance_suite(&KdTreeStrategy::default(), &config)
        .await
        .expect("KdTreeStrategy should satisfy conformance at 200K points");
}

#[tokio::test(start_paused = true)]
async fn beats_x_tree_on_white_noise() {
    let kd = KdTreeStrategy::default();
    let tree = PagedTreeStrategy::default();
    let strategies: [&dyn ProgressiveStrategy; 2] = [&tree, &kd];
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
            "{name}: k-d tree ({} B) should transfer < half of the x-tree ({} B) on white noise",
            wire(1, name),
            wire(0, name)
        );
    }
    for w in &reports[1].workloads {
        assert!(w.peak_client_memory_bytes <= reports[1].memory_budget_bytes);
    }
}
