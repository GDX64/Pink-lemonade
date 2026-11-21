//! Runs standard benchmark scenarios across strategies and writes a structured JSON report
//! for interactive analysis in `benchmarks/analysis.ipynb`.
//!
//! Usage:
//!   cargo run --profile fast -p pink-lemonade-progressive-tree --example export_reports --manifest-path rust/Cargo.toml -- --fast
//!   cargo run --profile fast -p pink-lemonade-progressive-tree --example export_reports --manifest-path rust/Cargo.toml
//!   ... -- --only heavy_white_noise --out benchmarks/results/heavy_white_noise.json

use pink_lemonade_progressive_core::ProgressiveStrategy;
use pink_lemonade_progressive_harness::benchmarks::{self, BenchmarkScenario};
use pink_lemonade_progressive_harness::write_reports_json;
use pink_lemonade_progressive_kdtree::KdTreeStrategy;
use pink_lemonade_progressive_naive::NaiveStrategy;
use pink_lemonade_progressive_ripmap::RipmapStrategy;
use pink_lemonade_progressive_tree::PagedTreeStrategy;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Run network simulation in deterministic virtual time.
    tokio::time::pause();

    let mut fast = false;
    let mut only: Vec<String> = Vec::new();
    let mut out_path = "benchmarks/results/latest.json".to_owned();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--fast" => fast = true,
            "--only" => only.push(
                args.next()
                    .ok_or_else(|| "missing scenario name after --only".to_owned())?,
            ),
            "--out" => {
                out_path = args
                    .next()
                    .ok_or_else(|| "missing value after --out".to_owned())?;
            }
            other if !other.starts_with('-') => {
                out_path = other.to_owned();
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    let tree = PagedTreeStrategy::default();
    let kdtree = KdTreeStrategy::default();
    let ripmap = RipmapStrategy::default();
    let all_strategies: [&dyn ProgressiveStrategy; 4] = [&NaiveStrategy, &tree, &kdtree, &ripmap];
    // Heavy (10M) scenarios skip the naive full download (240 MiB per workload).
    let progressive_only: [&dyn ProgressiveStrategy; 3] = [&tree, &kdtree, &ripmap];

    let mut runs: Vec<(BenchmarkScenario, &[&dyn ProgressiveStrategy])> = vec![
        (BenchmarkScenario::Light, &all_strategies),
        (BenchmarkScenario::Medium, &all_strategies),
        (BenchmarkScenario::MediumWhiteNoise, &all_strategies),
    ];
    if !fast {
        runs.extend([
            (BenchmarkScenario::ConstrainedMobile, &all_strategies[..]),
            (BenchmarkScenario::Heavy, &progressive_only[..]),
            (BenchmarkScenario::HeavyWhiteNoise, &progressive_only[..]),
        ]);
    }
    if !only.is_empty() {
        runs.retain(|(scenario, _)| only.contains(&scenario.spec().name));
        if runs.is_empty() {
            return Err(format!("no scenario matches --only {only:?}").into());
        }
    }

    let mut all_reports = Vec::new();
    for (scenario, strategies) in runs {
        let reports = benchmarks::compare(strategies, scenario)
            .await
            .map_err(|e| format!("benchmark failed: {e}"))?;
        for report in &reports {
            println!("\n{report}");
        }
        all_reports.extend(reports);
    }

    write_reports_json(&all_reports, &out_path)?;
    println!("Wrote {} reports to {out_path}", all_reports.len());
    Ok(())
}
