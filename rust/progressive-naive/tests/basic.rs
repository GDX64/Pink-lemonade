use pink_lemonade_progressive_harness::benchmarks::{self, BenchmarkScenario};
use pink_lemonade_progressive_harness::{ConformanceConfig, run_conformance_suite};
use pink_lemonade_progressive_naive::NaiveStrategy;

#[tokio::test(start_paused = true)]
async fn passes_conformance_suite() {
    let config = ConformanceConfig {
        require_line_points: true,
        ..ConformanceConfig::default()
    };
    run_conformance_suite(&NaiveStrategy, &config)
        .await
        .expect("NaiveStrategy should satisfy all conformance invariants");
}

#[tokio::test(start_paused = true)]
async fn light_benchmark() {
    let report = benchmarks::run_on(&NaiveStrategy, BenchmarkScenario::Light)
        .await
        .expect("light benchmark failed");

    println!("\n{report}");

    assert_eq!(report.total_points, 10_000);
    assert_eq!(report.workloads.len(), 5);
}
