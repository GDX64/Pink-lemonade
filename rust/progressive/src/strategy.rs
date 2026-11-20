//! Strategy traits: streaming offline builder, progressive async client, and frame sink.

use std::future::Future;
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::store::{ByteRangeStore, ObjectWriter};
use crate::types::{BuildError, ClientConfig, ClientError, Frame, Point, ViewQuery};

/// Channel through which a [`ProgressiveClient`] emits progressive frames (coarse -> fine).
///
/// When the receiver is dropped (e.g. the user panned/zoomed and cancelled the query),
/// [`FrameSink::send`] returns `Err(ClientError::Cancelled)`.
#[derive(Debug, Clone)]
pub struct FrameSink {
    tx: mpsc::Sender<Frame>,
}

impl FrameSink {
    pub fn channel(buffer: usize) -> (Self, mpsc::Receiver<Frame>) {
        let (tx, rx) = mpsc::channel(buffer.max(1));
        (Self { tx }, rx)
    }

    /// Emit the next progressive frame to the caller.
    pub async fn send(&self, frame: Frame) -> Result<(), ClientError> {
        self.tx.send(frame).await.map_err(|_| ClientError::Cancelled)
    }

    /// Returns `true` if the caller has dropped the frame receiver (cancelling the query).
    pub fn is_closed(&self) -> bool {
        self.tx.is_closed()
    }
}

/// Offline streaming encoder that transforms an x-sorted dataset into static byte-range objects.
///
/// Points are pushed in bounded chunks via [`DatasetBuilder::push_chunk`] so multi-gigabyte
/// datasets can be indexed out-of-core without holding all raw points in memory.
pub trait DatasetBuilder {
    /// Ingest the next contiguous chunk of x-sorted points.
    fn push_chunk(&mut self, points: &[Point]) -> Result<(), BuildError>;

    /// Finalize any pending structures and write all static objects into `writer`.
    fn finish<W: ObjectWriter>(self, writer: &mut W) -> Result<(), BuildError>;
}

/// Online interactive client that fetches only the byte ranges needed for the current
/// viewport and screen resolution, streaming coarse-to-fine frames as data arrives.
pub trait ProgressiveClient: Send {
    /// Execute a progressive viewport query, streaming frames into `frames`.
    ///
    /// Guidelines for implementations:
    /// - Emit an immediate coarse frame (if cached/available) with `FrameStatus::Refining`.
    /// - Fetch missing byte ranges progressively (concurrently where helpful) and emit
    ///   refined frames as data arrives.
    /// - Emit a final frame with `FrameStatus::Complete` once the viewport has converged
    ///   to the requested screen resolution and `merge_threshold`.
    /// - Be cancellation-safe: the caller may drop the future mid-flight when the user
    ///   pans or zooms again, and the client must remain valid for subsequent queries.
    fn query<'a>(
        &'a mut self,
        query: &'a ViewQuery,
        frames: FrameSink,
    ) -> impl Future<Output = Result<(), ClientError>> + Send + 'a;

    /// Approximate heap bytes currently retained by the client (caches, index metadata, buffers).
    fn memory_bytes(&self) -> usize;
}

/// Pairs an offline [`DatasetBuilder`] and an online [`ProgressiveClient`] under one strategy name
/// so any strategy can be plugged into the conformance suite and benchmark harness.
pub trait ProgressiveStrategy: Send + Sync {
    type Builder: DatasetBuilder;
    type Client<S: ByteRangeStore + 'static>: ProgressiveClient;

    /// Human-readable strategy identifier used in test output and benchmark reports.
    fn name(&self) -> &str;

    /// Construct a fresh streaming builder for a new dataset.
    fn create_builder(&self) -> Self::Builder;

    /// Open a client against a static byte-range store with the given memory budget.
    ///
    /// Async so strategies may optionally fetch a small manifest/header on startup
    /// (which is automatically accounted for by [`crate::SimulatedTransport`]).
    fn open_client<S: ByteRangeStore + 'static>(
        &self,
        store: Arc<S>,
        config: ClientConfig,
    ) -> impl Future<Output = Result<Self::Client<S>, ClientError>> + Send;
}
