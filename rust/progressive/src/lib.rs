//! Interfaces and simulation harness for progressive, screen-resolution-driven
//! visualization of massive datasets over dumb static byte-range storage.
//!
//! # Architecture
//! - [`store`]: Static multi-object byte-range storage traits ([`ByteRangeStore`],
//!   [`ObjectWriter`]) with [`MemoryStore`] (fast tests) and [`DiskStore`] (GB-scale runs).
//! - [`strategy`]: Strategy contract pairing a streaming offline [`DatasetBuilder`] with
//!   an async, cancellation-safe [`ProgressiveClient`] that streams coarse-to-fine [`Frame`]s.
//! - [`transport`]: [`SimulatedTransport`] modeling RTT latency, finite bandwidth, bounded
//!   concurrency, and mid-flight cancellation byte waste under Tokio virtual time.
//! - [`memory`]: [`memory::TrackingAllocator`] for peak heap verification in standalone
//!   benchmark binaries.
//! - [`harness`]: Streaming fixtures, exact moment & density NMAE evaluators, realistic
//!   navigation workloads, conformance tests, and benchmark reports.

pub mod harness;
pub mod memory;
pub mod store;
pub mod strategy;
pub mod transport;
pub mod types;

pub use store::{ByteRangeStore, DiskStore, MemoryStore, ObjectMeta, ObjectWriter};
pub use strategy::{DatasetBuilder, FrameSink, ProgressiveClient, ProgressiveStrategy};
pub use transport::{NetworkProfile, SimulatedTransport, TransportMetrics};
pub use types::{
    BuildError, ClientConfig, ClientError, Frame, FrameStatus, Point, StoreError, ViewQuery,
};
