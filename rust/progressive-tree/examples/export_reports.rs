//! Runs standard benchmark scenarios across strategies and writes a structured JSON report
//! for interactive analysis in `benchmarks/analysis.ipynb`.
//!
//! Usage:
//!   cargo run --profile fast -p pink-lemonade-progressive-tree --example export_reports --manifest-path rust/Cargo.toml -- --fast
//!   cargo run --profile fast -p pink-lemonade-progressive-tree --example export_reports --manifest-path rust/Cargo.toml

use pink_lemonade_progressive_core::ProgressiveStrategy;
use pink_lemonade_progressive_harness::benchmarks::{self, BenchmarkScenario};
use pink_lemonade_progressive_harness::write_reports_json;
use pink_lemonade_progressive_naive::NaiveStrategy;
use pink_lemonade_progressive_tree::PagedTreeStrategy;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Run network simulation in deterministic virtual time.
    tokio::time::pause();

    let mut fast = false;
    let mut out_path = "benchmarks/results/latest.json".to_owned();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--fast" => fast = true,
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
    let all_strategies: [&dyn ProgressiveStrategy; 2] = [&NaiveStrategy, &tree];
    let progressive_only: [&dyn ProgressiveStrategy; 1] = [&tree];

    let runs: Vec<(BenchmarkScenario, &[&dyn ProgressiveStrategy])> = if fast {
        vec![
            (BenchmarkScenario::Light, &all_strategies),
            (BenchmarkScenario::Medium, &all_strategies),
        ]
    } else {
        vec![
            (BenchmarkScenario::Light, &all_strategies),
            (BenchmarkScenario::Medium, &all_strategies),
            (BenchmarkScenario::ConstrainedMobile, &all_strategies),
            (BenchmarkScenario::Heavy, &progressive_only),
        ]
    };

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
