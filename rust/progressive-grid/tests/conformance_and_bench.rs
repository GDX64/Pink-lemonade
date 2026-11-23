use pink_lemonade_progressive_core::ProgressiveStrategy;
use pink_lemonade_progressive_grid::SparseGridStrategy;
use pink_lemonade_progressive_harness::benchmarks::{self, BenchmarkScenario};
use pink_lemonade_progressive_harness::{ConformanceConfig, DatasetShape, run_conformance_suite};
use pink_lemonade_progressive_quadtree::QuadtreeStrategy;
use pink_lemonade_progressive_ripmap::RipmapStrategy;
use pink_lemonade_progressive_tree::PagedTreeStrategy;

#[tokio::test(start_paused = true)]
async fn passes_full_conformance_suite() {
    let config = ConformanceConfig {
        require_line_points: true,
        ..ConformanceConfig::default()
    };
    run_conformance_suite(&SparseGridStrategy::default(), &config)
        .await
        .expect("SparseGridStrategy should satisfy all conformance invariants");
}

/// 200K points gives ~10 x-levels and planes at many aspects, so inside columns are rendered
/// from cells and edge chains descend several levels before going raw.
#[tokio::test(start_paused = true)]
async fn passes_conformance_with_deep_levels() {
    let config = ConformanceConfig {
        point_count: 200_000,
        chunk_size: 8_192,
        require_line_points: true,
        ..ConformanceConfig::default()
    };
    run_conformance_suite(&SparseGridStrategy::default(), &config)
        .await
        .expect("SparseGridStrategy should satisfy conformance at 200K points");
}

/// Tiny columns and a single aspect force deep edge chains and frequent raw fallback.
#[tokio::test(start_paused = true)]
async fn passes_conformance_with_small_columns_and_one_aspect() {
    let strategy = SparseGridStrategy {
        col_leaf: 8,
        min_aspect: 0,
        max_aspect: 0,
        ..SparseGridStrategy::default()
    };
    let config = ConformanceConfig {
        point_count: 50_000,
        chunk_size: 4_096,
        require_line_points: true,
        ..ConformanceConfig::default()
    };
    run_conformance_suite(&strategy, &config)
        .await
        .expect("SparseGridStrategy should satisfy conformance with 8-point columns");
}

/// Density NMAE stays within 5% on every standard workload for both signal families.
#[tokio::test(start_paused = true)]
async fn density_nmae_within_tolerance_on_all_workloads() {
    let grid = SparseGridStrategy::default();
    let strategies: [&dyn ProgressiveStrategy; 1] = [&grid];
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
/// memory budget is asserted.
#[tokio::test(start_paused = true)]
async fn compare_with_other_strategies() {
    let tree = PagedTreeStrategy::default();
    let ripmap = RipmapStrategy::default();
    let quad = QuadtreeStrategy::default();
    let grid = SparseGridStrategy::default();
    let strategies: [&dyn ProgressiveStrategy; 4] = [&tree, &ripmap, &quad, &grid];
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

/// Per-object storage and the stored plane table at 10M points (run with `--ignored
/// --nocapture`; builds two 10M datasets in memory).
#[tokio::test]
#[ignore = "diagnostic: 10M builds"]
async fn storage_breakdown_at_10m() {
    use pink_lemonade_progressive_core::ByteRangeStore;
    use pink_lemonade_progressive_grid::format::{CELLS_KEY, COLUMNS_KEY, MANIFEST_KEY, Manifest, POINTS_KEY};
    use pink_lemonade_progressive_harness::{DatasetStream, MemoryStore, stream_into_builder};

    for shape in [DatasetShape::RandomWalk { seed: 578_211 }, DatasetShape::NoisySpikes { seed: 4_242 }] {
        let mut builder = SparseGridStrategy::default().create_builder();
        stream_into_builder(DatasetStream::new(shape, 10_000_000, 131_072), builder.as_mut()).unwrap();
        let mut store = MemoryStore::new();
        builder.finish(&mut store).unwrap();
        println!("\n{}:", shape.name());
        for key in [MANIFEST_KEY, COLUMNS_KEY, CELLS_KEY, POINTS_KEY] {
            let len = store.object_len(key).await.unwrap();
            println!("  {key:<13} {:>10.2} MiB", len as f64 / 1048576.0);
        }
        let raw = store.read_range(MANIFEST_KEY, 0, store.object_len(MANIFEST_KEY).await.unwrap() as usize).await.unwrap();
        let manifest = Manifest::decode(&raw).unwrap();
        for (i, level) in manifest.levels.iter().enumerate() {
            let planes: Vec<String> = level.planes.iter().map(|p| format!("j{}:{}", p.j, p.cells)).collect();
            println!("  i={i:<2} {}", planes.join(" "));
        }
    }
}
