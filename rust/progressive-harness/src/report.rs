//! Structured benchmark reports, human-readable formatting, and JSON export for notebook analysis.

use std::fmt::{self, Write as _};
use std::fs;
use std::io;
use std::path::Path;
use std::time::Duration;

use crate::transport::{NetworkProfile, TransportMetrics};

/// Metrics recorded for a single viewport step in a workload.
#[derive(Debug, Clone)]
pub struct StepReport {
    pub label: String,
    pub view_min_x: f64,
    pub view_max_x: f64,
    /// Explicit Y viewport bounds when the step zoomed vertically (`None` = auto-y).
    pub view_y_bounds: Option<[f64; 2]>,
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
    pub scenario_name: Option<String>,
    pub dataset_shape: Option<String>,
    pub total_points: u64,
    pub build_duration: Duration,
    pub storage_bytes: u64,
    pub object_count: usize,
    pub network: NetworkProfile,
    pub memory_budget_bytes: usize,
    pub workloads: Vec<WorkloadReport>,
}

impl BenchmarkReport {
    /// Serialize this report to pretty-printed JSON.
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        write_report_json(&mut out, self, 0);
        out.push('\n');
        out
    }

    /// Write this report as JSON to `path` (creating parent directories if needed).
    pub fn write_json(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, self.to_json())
    }
}

/// Serialize a slice of [`BenchmarkReport`]s as a JSON document `{ "reports": [...] }`.
pub fn reports_to_json(reports: &[BenchmarkReport]) -> String {
    let mut out = String::from("{\n  \"reports\": [");
    if !reports.is_empty() {
        out.push('\n');
        for (i, report) in reports.iter().enumerate() {
            if i > 0 {
                out.push_str(",\n");
            }
            write_report_json(&mut out, report, 4);
        }
        out.push_str("\n  ");
    }
    out.push_str("]\n}\n");
    out
}

/// Write a slice of [`BenchmarkReport`]s as JSON to `path` (creating parent directories if needed).
pub fn write_reports_json(reports: &[BenchmarkReport], path: impl AsRef<Path>) -> io::Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, reports_to_json(reports))
}

fn push_indent(out: &mut String, n: usize) {
    for _ in 0..n {
        out.push(' ');
    }
}

fn write_json_str(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn write_opt_f64(out: &mut String, val: Option<f64>) {
    match val {
        Some(v) if v.is_finite() => {
            let _ = write!(out, "{v}");
        }
        _ => out.push_str("null"),
    }
}

fn write_transport_json(out: &mut String, m: &TransportMetrics, indent: usize) {
    let pad = indent + 2;
    out.push_str("{\n");
    push_indent(out, pad);
    let _ = writeln!(out, "\"completed_requests\": {},", m.completed_requests);
    push_indent(out, pad);
    let _ = writeln!(out, "\"cancelled_requests\": {},", m.cancelled_requests);
    push_indent(out, pad);
    let _ = writeln!(out, "\"range_requests\": {},", m.range_requests);
    push_indent(out, pad);
    let _ = writeln!(out, "\"head_requests\": {},", m.head_requests);
    push_indent(out, pad);
    let _ = writeln!(out, "\"list_requests\": {},", m.list_requests);
    push_indent(out, pad);
    let _ = writeln!(out, "\"bytes_downloaded\": {},", m.bytes_downloaded);
    push_indent(out, pad);
    let _ = writeln!(
        out,
        "\"wasted_cancelled_bytes\": {},",
        m.wasted_cancelled_bytes
    );
    push_indent(out, pad);
    let _ = writeln!(out, "\"total_wire_bytes\": {},", m.total_wire_bytes());
    push_indent(out, pad);
    let _ = writeln!(out, "\"peak_in_flight\": {}", m.peak_in_flight);
    push_indent(out, indent);
    out.push('}');
}

fn write_step_json(out: &mut String, s: &StepReport, indent: usize) {
    let pad = indent + 2;
    push_indent(out, indent);
    out.push_str("{\n");
    push_indent(out, pad);
    out.push_str("\"label\": ");
    write_json_str(out, &s.label);
    out.push_str(",\n");
    push_indent(out, pad);
    let _ = writeln!(out, "\"view_min_x\": {},", s.view_min_x);
    push_indent(out, pad);
    let _ = writeln!(out, "\"view_max_x\": {},", s.view_max_x);
    push_indent(out, pad);
    let _ = writeln!(out, "\"view_width\": {},", s.view_max_x - s.view_min_x);
    push_indent(out, pad);
    out.push_str("\"view_min_y\": ");
    write_opt_f64(out, s.view_y_bounds.map(|b| b[0]));
    out.push_str(",
");
    push_indent(out, pad);
    out.push_str("\"view_max_y\": ");
    write_opt_f64(out, s.view_y_bounds.map(|b| b[1]));
    out.push_str(",
");
    push_indent(out, pad);
    let _ = writeln!(out, "\"completed\": {},", s.completed);
    push_indent(out, pad);
    out.push_str("\"latency_ms\": ");
    write_opt_f64(out, s.latency.map(|d| d.as_secs_f64() * 1_000.0));
    out.push_str(",\n");
    push_indent(out, pad);
    let _ = writeln!(out, "\"kernel_count\": {},", s.kernel_count);
    push_indent(out, pad);
    let _ = writeln!(out, "\"client_memory_bytes\": {},", s.client_memory_bytes);
    push_indent(out, pad);
    out.push_str("\"nmae_percent\": ");
    write_opt_f64(out, s.nmae_percent);
    out.push_str(",\n");
    push_indent(out, pad);
    out.push_str("\"transport\": ");
    write_transport_json(out, &s.transport, pad);
    out.push('\n');
    push_indent(out, indent);
    out.push('}');
}

fn write_workload_json(out: &mut String, w: &WorkloadReport, indent: usize) {
    let pad = indent + 2;
    push_indent(out, indent);
    out.push_str("{\n");
    push_indent(out, pad);
    out.push_str("\"name\": ");
    write_json_str(out, &w.name);
    out.push_str(",\n");
    push_indent(out, pad);
    out.push_str("\"mean_latency_ms\": ");
    write_opt_f64(out, w.mean_latency().map(|d| d.as_secs_f64() * 1_000.0));
    out.push_str(",\n");
    push_indent(out, pad);
    let _ = writeln!(
        out,
        "\"peak_client_memory_bytes\": {},",
        w.peak_client_memory_bytes
    );
    push_indent(out, pad);
    out.push_str("\"open_transport\": ");
    write_transport_json(out, &w.open_transport, pad);
    out.push_str(",\n");
    push_indent(out, pad);
    out.push_str("\"total_transport\": ");
    write_transport_json(out, &w.total_transport, pad);
    out.push_str(",\n");
    push_indent(out, pad);
    out.push_str("\"steps\": [");
    if !w.steps.is_empty() {
        out.push('\n');
        for (i, step) in w.steps.iter().enumerate() {
            if i > 0 {
                out.push_str(",\n");
            }
            write_step_json(out, step, pad + 2);
        }
        out.push('\n');
        push_indent(out, pad);
    }
    out.push_str("]\n");
    push_indent(out, indent);
    out.push('}');
}

fn write_report_json(out: &mut String, r: &BenchmarkReport, indent: usize) {
    let pad = indent + 2;
    push_indent(out, indent);
    out.push_str("{\n");
    push_indent(out, pad);
    out.push_str("\"strategy_name\": ");
    write_json_str(out, &r.strategy_name);
    out.push_str(",\n");
    push_indent(out, pad);
    out.push_str("\"scenario_name\": ");
    match &r.scenario_name {
        Some(name) => write_json_str(out, name),
        None => out.push_str("null"),
    }
    out.push_str(",\n");
    push_indent(out, pad);
    out.push_str("\"dataset_shape\": ");
    match &r.dataset_shape {
        Some(shape) => write_json_str(out, shape),
        None => out.push_str("null"),
    }
    out.push_str(",\n");
    push_indent(out, pad);
    let _ = writeln!(out, "\"total_points\": {},", r.total_points);
    push_indent(out, pad);
    let _ = writeln!(out, "\"raw_points_bytes\": {},", r.total_points * 24);
    push_indent(out, pad);
    let _ = writeln!(
        out,
        "\"build_duration_ms\": {},",
        r.build_duration.as_secs_f64() * 1_000.0
    );
    push_indent(out, pad);
    let _ = writeln!(out, "\"storage_bytes\": {},", r.storage_bytes);
    push_indent(out, pad);
    let _ = writeln!(out, "\"object_count\": {},", r.object_count);
    push_indent(out, pad);
    let _ = writeln!(out, "\"memory_budget_bytes\": {},", r.memory_budget_bytes);
    push_indent(out, pad);
    out.push_str("\"network\": {\n");
    push_indent(out, pad + 2);
    out.push_str("\"name\": ");
    write_json_str(out, r.network.name);
    out.push_str(",\n");
    push_indent(out, pad + 2);
    let _ = writeln!(
        out,
        "\"rtt_ms\": {},",
        r.network.rtt.as_secs_f64() * 1_000.0
    );
    push_indent(out, pad + 2);
    out.push_str("\"bandwidth_bytes_per_sec\": ");
    write_opt_f64(out, r.network.bandwidth_bytes_per_sec);
    out.push_str(",\n");
    push_indent(out, pad + 2);
    out.push_str("\"max_concurrency\": ");
    match r.network.max_concurrency {
        Some(c) => {
            let _ = write!(out, "{c}");
        }
        None => out.push_str("null"),
    }
    out.push('\n');
    push_indent(out, pad);
    out.push_str("},\n");
    push_indent(out, pad);
    out.push_str("\"workloads\": [");
    if !r.workloads.is_empty() {
        out.push('\n');
        for (i, workload) in r.workloads.iter().enumerate() {
            if i > 0 {
                out.push_str(",\n");
            }
            write_workload_json(out, workload, pad + 2);
        }
        out.push('\n');
        push_indent(out, pad);
    }
    out.push_str("]\n");
    push_indent(out, indent);
    out.push('}');
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
        match &self.scenario_name {
            Some(scenario) => writeln!(
                f,
                "=== {} [{}] ({} points, network={}, mem_budget={}) ===",
                self.strategy_name,
                scenario,
                self.total_points,
                self.network.name,
                format_bytes(self.memory_budget_bytes as u64),
            )?,
            None => writeln!(
                f,
                "=== {} ({} points, network={}, mem_budget={}) ===",
                self.strategy_name,
                self.total_points,
                self.network.name,
                format_bytes(self.memory_budget_bytes as u64),
            )?,
        }
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
