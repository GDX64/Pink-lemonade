//! Structured benchmark reports and human-readable formatting.

use std::fmt;
use std::time::Duration;

use crate::transport::{NetworkProfile, TransportMetrics};

/// Metrics recorded for a single viewport step in a workload.
#[derive(Debug, Clone)]
pub struct StepReport {
    pub label: String,
    pub completed: bool,
    /// Simulated time until `query()` resolved (if it finished before any `max_dwell` timeout).
    pub latency: Option<Duration>,
    pub kernel_count: usize,
    pub client_memory_bytes: usize,
    pub transport: TransportMetrics,
    /// Optional density NMAE (%) when evaluated against ground-truth points.
    pub nmae_percent: Option<f64>,
}

/// Aggregated report for one [`crate::Workload`] executed against a client.
#[derive(Debug, Clone)]
pub struct WorkloadReport {
    pub name: String,
    /// Traffic incurred during `open_client` (e.g. manifest/header fetch).
    pub open_transport: TransportMetrics,
    pub steps: Vec<StepReport>,
    pub total_transport: TransportMetrics,
    pub peak_client_memory_bytes: usize,
}

impl WorkloadReport {
    pub fn mean_latency(&self) -> Option<Duration> {
        mean_duration(self.steps.iter().filter_map(|s| s.latency))
    }
}

/// Complete report for building and benchmarking one `ProgressiveStrategy`.
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    pub strategy_name: String,
    pub total_points: u64,
    pub build_duration: Duration,
    pub storage_bytes: u64,
    pub object_count: usize,
    pub network: NetworkProfile,
    pub memory_budget_bytes: usize,
    pub workloads: Vec<WorkloadReport>,
}

fn mean_duration(iter: impl Iterator<Item = Duration>) -> Option<Duration> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for d in iter {
        sum += d.as_secs_f64();
        count += 1;
    }
    if count == 0 {
        None
    } else {
        Some(Duration::from_secs_f64(sum / count as f64))
    }
}

fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.2} GiB", b / GIB)
    } else if b >= MIB {
        format!("{:.2} MiB", b / MIB)
    } else if b >= KIB {
        format!("{:.2} KiB", b / KIB)
    } else {
        format!("{bytes} B")
    }
}

fn format_opt_ms(d: Option<Duration>) -> String {
    match d {
        Some(v) => format!("{:.2} ms", v.as_secs_f64() * 1_000.0),
        None => "-".to_owned(),
    }
}

impl fmt::Display for BenchmarkReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "=== {} ({} points, network={}, mem_budget={}) ===",
            self.strategy_name,
            self.total_points,
            self.network.name,
            format_bytes(self.memory_budget_bytes as u64),
        )?;
        writeln!(
            f,
            "build: {:.2} ms | storage: {} across {} object(s)",
            self.build_duration.as_secs_f64() * 1_000.0,
            format_bytes(self.storage_bytes),
            self.object_count,
        )?;
        for w in &self.workloads {
            writeln!(
                f,
                "  workload {:<20} | wire={} (completed={}, wasted={}) | reqs={} (cancelled={}) | mean_latency={} | peak_mem={}",
                w.name,
                format_bytes(w.total_transport.total_wire_bytes()),
                format_bytes(w.total_transport.bytes_downloaded),
                format_bytes(w.total_transport.wasted_cancelled_bytes),
                w.total_transport.completed_requests,
                w.total_transport.cancelled_requests,
                format_opt_ms(w.mean_latency()),
                format_bytes(w.peak_client_memory_bytes as u64),
            )?;
        }
        Ok(())
    }
}
