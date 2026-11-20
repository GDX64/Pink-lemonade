//! Core traits and data types for progressive, screen-resolution-driven visualization
//! of massive datasets over static byte-range storage.
//!
//! This crate intentionally contains **no** storage backends, network simulators, or
//! test/benchmark harnesses — only the lean contract that concrete strategies implement
//! (and that browser/WASM or native clients can depend on directly).
//!
//! For test stores, virtual-time network simulation, conformance tests, and benchmarks,
//! depend on `pink-lemonade-progressive-harness` under `[dev-dependencies]`.

pub mod store;
pub mod strategy;
pub mod types;

pub use store::{ByteRangeStore, ObjectMeta, ObjectWriter, validate_key};
pub use strategy::{DatasetBuilder, FrameSink, ProgressiveClient, ProgressiveStrategy};
pub use types::{
    BuildError, ClientConfig, ClientError, Frame, FrameStatus, Point, StoreError, ViewQuery,
};
