//! Simulation, conformance, and benchmarking harness for strategies implementing
//! `pink-lemonade-progressive-core`.
//!
//! Intended to be added under `[dev-dependencies]` of any strategy crate so both
//! integration tests (`tests/`) and benchmarks (`benches/`) can share:
//! - [`store`]: [`MemoryStore`] (fast tests) and [`DiskStore`] (GB-scale runs).
//! - [`transport`]: [`SimulatedTransport`] modeling RTT latency, finite bandwidth,
//!   bounded concurrency, and mid-flight cancellation byte waste under Tokio virtual time.
//! - [`memory`]: [`memory::TrackingAllocator`] for peak heap verification in benchmark binaries.
//! - [`fixture`], [`moments`], [`quality`], [`workload`], [`conformance`], [`report`], [`runner`].

pub mod conformance;
pub mod fixture;
pub mod memory;
pub mod moments;
pub mod quality;
pub mod report;
pub mod runner;
pub mod store;
pub mod transport;
pub mod workload;

pub use conformance::{ConformanceConfig, run_conformance_suite};
pub use fixture::{DatasetShape, DatasetStream, SplitMix64, stream_into_builder};
pub use memory::{MemoryStats, TrackingAllocator};
pub use report::{BenchmarkReport, StepReport, WorkloadReport};
pub use runner::{HarnessConfig, StepOutcome, run_benchmark, run_step, run_workload};
pub use store::{DiskStore, MemoryObjectSink, MemoryStore};
pub use transport::{NetworkProfile, SimulatedTransport, TransportMetrics};
pub use workload::{Workload, WorkloadStep};
