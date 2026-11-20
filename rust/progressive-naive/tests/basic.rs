use pink_lemonade_progressive_core::ClientConfig;
use pink_lemonade_progressive_harness::{
    ConformanceConfig, DatasetShape, DatasetStream, HarnessConfig, MemoryStore, NetworkProfile,
    Workload, run_benchmark, run_conformance_suite,
};
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
async fn runs_standard_benchmark_and_prints_report() {
    let point_count = 10_000;
    let stream = DatasetStream::new(DatasetShape::SineWave, point_count, 1_000);
    let config = HarnessConfig {
        network: NetworkProfile::broadband(),
        client: ClientConfig {
            memory_budget_bytes: 1024 * 1024, // 1 MiB
        },
        ..HarnessConfig::default()
    };
    let workloads = Workload::standard_suite(
        config.screen_w,
        config.screen_h,
        config.sigma_px,
        config.merge_threshold,
    );

    let mut store = MemoryStore::new();
    let report = run_benchmark(
        &NaiveStrategy,
        stream,
        &mut store,
        &workloads,
        &config,
        None,
    )
    .await
    .expect("benchmark failed");

    println!("\n{report}");

    assert_eq!(report.total_points, point_count);
    assert_eq!(report.workloads.len(), 5);
}
