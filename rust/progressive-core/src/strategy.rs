//! Object-safe strategy traits: streaming offline builder and async viewport client.

use std::sync::Arc;

use async_trait::async_trait;

use crate::store::{ByteRangeStore, ObjectWriter};
use crate::types::{BuildError, ClientConfig, ClientError, Frame, Point, ViewQuery};

/// Offline streaming encoder that transforms an x-sorted dataset into static byte-range objects.
///
/// Points are pushed in bounded chunks via [`DatasetBuilder::push_chunk`] so multi-gigabyte
/// datasets can be indexed out-of-core without holding all raw points in memory.
pub trait DatasetBuilder {
    /// Ingest the next contiguous chunk of x-sorted points.
    fn push_chunk(&mut self, points: &[Point]) -> Result<(), BuildError>;

    /// Finalize any pending structures and write all static objects into `writer`.
    fn finish(self: Box<Self>, writer: &mut dyn ObjectWriter) -> Result<(), BuildError>;
}

/// Online interactive client that fetches only the byte ranges needed for the current
/// viewport and screen resolution.
#[async_trait]
pub trait ProgressiveClient: Send {
    /// Fetch any missing byte ranges needed for `query` and return the rendered [`Frame`].
    ///
    /// Implementations should be cancellation-safe: the caller may drop the future mid-flight
    /// when the user pans or zooms again, and the client must remain valid for later queries.
    async fn query(&mut self, query: ViewQuery) -> Result<Frame, ClientError>;

    /// Approximate heap bytes currently retained by the client (caches, index metadata, buffers).
    fn memory_bytes(&self) -> usize;
}

/// Pairs an offline [`DatasetBuilder`] and an online [`ProgressiveClient`] under one strategy name
/// so any strategy can be plugged into the conformance suite and benchmark harness via `dyn`.
#[async_trait]
pub trait ProgressiveStrategy: Send + Sync {
    /// Human-readable strategy identifier used in test output and benchmark reports.
    fn name(&self) -> &str;

    /// Construct a fresh streaming builder for a new dataset.
    fn create_builder(&self) -> Box<dyn DatasetBuilder>;

    /// Open a client against a static byte-range store with the given memory budget.
    ///
    /// Async so strategies may optionally fetch a small manifest/header on startup.
    async fn open_client(
        &self,
        store: Arc<dyn ByteRangeStore>,
        config: ClientConfig,
    ) -> Result<Box<dyn ProgressiveClient>, ClientError>;
}
