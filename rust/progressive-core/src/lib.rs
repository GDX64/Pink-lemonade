//! Core traits and data types for progressive, screen-resolution-driven visualization
//! of massive datasets over static byte-range storage.
//!
//! All async traits use [`async_trait`] so they remain `dyn`-compatible (`dyn ByteRangeStore`,
//! `dyn ProgressiveClient`, `dyn ProgressiveStrategy`) while keeping `async fn` syntax clean.

pub mod store;
pub mod strategy;
pub mod types;

pub use async_trait::async_trait;
pub use store::{ByteRangeStore, ObjectMeta, ObjectWriter, validate_key};
pub use strategy::{DatasetBuilder, ProgressiveClient, ProgressiveStrategy};
pub use types::{BuildError, ClientConfig, ClientError, Frame, Point, StoreError, ViewQuery};
