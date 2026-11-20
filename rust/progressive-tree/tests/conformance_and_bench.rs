use pink_lemonade_progressive_core::ProgressiveStrategy;
use pink_lemonade_progressive_harness::benchmarks::{self, BenchmarkScenario};
use pink_lemonade_progressive_harness::{ConformanceConfig, run_conformance_suite};
use pink_lemonade_progressive_naive::NaiveStrategy;
use pink_lemonade_progressive_tree::PagedTreeStrategy;

#[tokio::test(start_paused = true)]
async fn passes_full_conformance_suite() {
    let strategy = PagedTreeStrategy::default();
    let config = ConformanceConfig {
        require_line_points: true,
        ..ConformanceConfig::default()
    };
    run_conformance_suite(&strategy, &config)
        .await
        .expect("PagedTreeStrategy should satisfy all conformance invariants");
}

#[tokio::test(start_paused = true)]
async fn compare_light_against_naive() {
    let tree = PagedTreeStrategy::default();
    let strategies: [&dyn ProgressiveStrategy; 2] = [&NaiveStrategy, &tree];
    let reports = benchmarks::compare(&strategies, BenchmarkScenario::Light)
        .await
        .expect("light comparison failed");

    for report in &reports {
        println!("\n{report}");
    }

    // On Light (10K points), zoomed workloads (lateral_pan, jump_and_revisit, rapid_interrupt_pan)
    // fetch only the visible slices instead of the whole 234 KiB file.
    for workload_idx in [2, 3, 4] {
        let naive_wire = reports[0].workloads[workload_idx]
            .total_transport
            .total_wire_bytes();
        let tree_wire = reports[1].workloads[workload_idx]
            .total_transport
            .total_wire_bytes();
        assert!(
            tree_wire < naive_wire / 2,
            "workload {} expected paged tree ({tree_wire} B) < half of naive ({naive_wire} B)",
            reports[1].workloads[workload_idx].name
        );
    }
}

#[tokio::test(start_paused = true)]
async fn medium_1m_points_benchmark() {
    let tree = PagedTreeStrategy::default();
    let report = benchmarks::run_on(&tree, BenchmarkScenario::Medium)
        .await
        .expect("medium 1M benchmark failed");

    println!("\n{report}");

    // 1M raw points = 24 MiB; paged tree should keep client memory <= 4 MiB budget
    // and transfer far less than 24 MiB on every workload.
    let raw_bytes = 1_000_000 * 24;
    for w in &report.workloads {
        assert!(
            w.peak_client_memory_bytes <= report.memory_budget_bytes,
            "workload '{}' exceeded memory budget: {} > {}",
            w.name,
            w.peak_client_memory_bytes,
            report.memory_budget_bytes
        );
        assert!(
            w.total_transport.total_wire_bytes() < raw_bytes / 4,
            "workload '{}' downloaded too much: {} B vs {} B raw",
            w.name,
            w.total_transport.total_wire_bytes(),
            raw_bytes
        );
    }
}
