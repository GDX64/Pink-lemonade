//! Virtual-time network transport simulator wrapping any [`ByteRangeStore`].
//!
//! Models per-request RTT latency, finite download bandwidth, bounded concurrency, and
//! mid-flight request cancellation (including partial wasted bytes transferred before abort).
//! Designed to run under Tokio paused/virtual time so full user-session benchmarks run
//! deterministically and finish in milliseconds of wall-clock time.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;
use tokio::sync::Semaphore;
use tokio::time::Instant;

use crate::store::{ByteRangeStore, ObjectMeta};
use crate::types::StoreError;

/// Network characteristics simulated by [`SimulatedTransport`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NetworkProfile {
    pub name: &'static str,
    /// Base round-trip latency added to every request.
    pub rtt: Duration,
    /// Download throughput in bytes/second (`None` = infinite bandwidth).
    pub bandwidth_bytes_per_sec: Option<f64>,
    /// Maximum concurrent in-flight requests (`None` = unlimited).
    pub max_concurrency: Option<usize>,
}

impl NetworkProfile {
    /// Zero-latency, infinite-bandwidth profile for pure byte/request counting.
    pub const fn instant() -> Self {
        Self {
            name: "instant",
            rtt: Duration::ZERO,
            bandwidth_bytes_per_sec: None,
            max_concurrency: None,
        }
    }

    /// Local LAN / data-center profile: 2 ms RTT, 1 Gbps, 16 concurrent streams.
    pub const fn local_lan() -> Self {
        Self {
            name: "local-lan",
            rtt: Duration::from_millis(2),
            bandwidth_bytes_per_sec: Some(125_000_000.0),
            max_concurrency: Some(16),
        }
    }

    /// Typical consumer broadband: 30 ms RTT, 50 Mbps (~6.25 MB/s), 6 concurrent streams.
    pub const fn broadband() -> Self {
        Self {
            name: "broadband-50mbps",
            rtt: Duration::from_millis(30),
            bandwidth_bytes_per_sec: Some(6_250_000.0),
            max_concurrency: Some(6),
        }
    }

    /// Constrained mobile / fast 3G: 100 ms RTT, 5 Mbps (~625 KB/s), 6 concurrent streams.
    pub const fn mobile_5mbps() -> Self {
        Self {
            name: "mobile-5mbps",
            rtt: Duration::from_millis(100),
            bandwidth_bytes_per_sec: Some(625_000.0),
            max_concurrency: Some(6),
        }
    }

    pub fn transfer_duration(&self, bytes: usize) -> Duration {
        match self.bandwidth_bytes_per_sec {
            Some(bps) if bps > 0.0 && bytes > 0 => Duration::from_secs_f64(bytes as f64 / bps),
            _ => Duration::ZERO,
        }
    }
}

/// Snapshot of network traffic observed by a [`SimulatedTransport`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TransportMetrics {
    pub completed_requests: u64,
    pub cancelled_requests: u64,
    pub range_requests: u64,
    pub head_requests: u64,
    pub list_requests: u64,
    pub bytes_downloaded: u64,
    pub wasted_cancelled_bytes: u64,
    pub peak_in_flight: usize,
}

impl TransportMetrics {
    /// Total wire bytes consumed (both completed and partially transferred before cancellation).
    pub const fn total_wire_bytes(&self) -> u64 {
        self.bytes_downloaded + self.wasted_cancelled_bytes
    }

    /// Compute the delta between `self` (later snapshot) and `earlier`.
    pub const fn delta_since(&self, earlier: &Self) -> Self {
        Self {
            completed_requests: self.completed_requests - earlier.completed_requests,
            cancelled_requests: self.cancelled_requests - earlier.cancelled_requests,
            range_requests: self.range_requests - earlier.range_requests,
            head_requests: self.head_requests - earlier.head_requests,
            list_requests: self.list_requests - earlier.list_requests,
            bytes_downloaded: self.bytes_downloaded - earlier.bytes_downloaded,
            wasted_cancelled_bytes: self.wasted_cancelled_bytes - earlier.wasted_cancelled_bytes,
            peak_in_flight: self.peak_in_flight,
        }
    }
}

#[derive(Default)]
struct MetricsCounters {
    completed_requests: AtomicU64,
    cancelled_requests: AtomicU64,
    range_requests: AtomicU64,
    head_requests: AtomicU64,
    list_requests: AtomicU64,
    bytes_downloaded: AtomicU64,
    wasted_cancelled_bytes: AtomicU64,
    in_flight: AtomicUsize,
    peak_in_flight: AtomicUsize,
}

impl MetricsCounters {
    fn snapshot(&self) -> TransportMetrics {
        TransportMetrics {
            completed_requests: self.completed_requests.load(Ordering::Relaxed),
            cancelled_requests: self.cancelled_requests.load(Ordering::Relaxed),
            range_requests: self.range_requests.load(Ordering::Relaxed),
            head_requests: self.head_requests.load(Ordering::Relaxed),
            list_requests: self.list_requests.load(Ordering::Relaxed),
            bytes_downloaded: self.bytes_downloaded.load(Ordering::Relaxed),
            wasted_cancelled_bytes: self.wasted_cancelled_bytes.load(Ordering::Relaxed),
            peak_in_flight: self.peak_in_flight.load(Ordering::Relaxed),
        }
    }

    fn enter_in_flight(&self) {
        let now = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        self.peak_in_flight.fetch_max(now, Ordering::Relaxed);
    }

    fn exit_in_flight(&self) {
        self.in_flight.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Drop guard that records cancelled requests and estimates wasted bytes when a client
/// drops an in-flight `read_range` / `object_len` / `list_objects` future.
struct InFlightGuard {
    counters: Arc<MetricsCounters>,
    started_at: Instant,
    rtt: Duration,
    transfer_duration: Duration,
    payload_bytes: u64,
    completed: bool,
}

impl InFlightGuard {
    fn new(counters: Arc<MetricsCounters>, rtt: Duration) -> Self {
        counters.enter_in_flight();
        Self {
            counters,
            started_at: Instant::now(),
            rtt,
            transfer_duration: Duration::ZERO,
            payload_bytes: 0,
            completed: false,
        }
    }

    fn set_payload(&mut self, bytes: u64, transfer_duration: Duration) {
        self.payload_bytes = bytes;
        self.transfer_duration = transfer_duration;
    }

    fn mark_completed(&mut self) {
        self.completed = true;
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.counters.exit_in_flight();
        if !self.completed {
            self.counters
                .cancelled_requests
                .fetch_add(1, Ordering::Relaxed);
            if self.payload_bytes > 0 {
                let elapsed = self.started_at.elapsed();
                if elapsed > self.rtt {
                    let body_elapsed = elapsed - self.rtt;
                    let fraction = if self.transfer_duration.is_zero() {
                        1.0
                    } else {
                        (body_elapsed.as_secs_f64() / self.transfer_duration.as_secs_f64())
                            .clamp(0.0, 1.0)
                    };
                    let wasted = (self.payload_bytes as f64 * fraction).round() as u64;
                    self.counters
                        .wasted_cancelled_bytes
                        .fetch_add(wasted, Ordering::Relaxed);
                }
            }
        }
    }
}

/// Wraps any [`ByteRangeStore`] to simulate network latency, bandwidth, concurrency limits,
/// and exact traffic accounting.
pub struct SimulatedTransport<S> {
    inner: S,
    profile: NetworkProfile,
    semaphore: Option<Arc<Semaphore>>,
    counters: Arc<MetricsCounters>,
}

impl<S> SimulatedTransport<S> {
    pub fn new(inner: S, profile: NetworkProfile) -> Self {
        let semaphore = profile
            .max_concurrency
            .map(|limit| Arc::new(Semaphore::new(limit.max(1))));
        Self {
            inner,
            profile,
            semaphore,
            counters: Arc::new(MetricsCounters::default()),
        }
    }

    pub fn profile(&self) -> NetworkProfile {
        self.profile
    }

    pub fn inner(&self) -> &S {
        &self.inner
    }

    pub fn metrics(&self) -> TransportMetrics {
        self.counters.snapshot()
    }

    async fn acquire_slot(&self) -> Option<tokio::sync::OwnedSemaphorePermit> {
        match &self.semaphore {
            Some(sem) => Some(
                Arc::clone(sem)
                    .acquire_owned()
                    .await
                    .expect("transport semaphore never closed"),
            ),
            None => None,
        }
    }
}

impl<S: ByteRangeStore> ByteRangeStore for SimulatedTransport<S> {
    async fn read_range(&self, key: &str, offset: u64, len: usize) -> Result<Vec<u8>, StoreError> {
        let _permit = self.acquire_slot().await;
        let mut guard = InFlightGuard::new(Arc::clone(&self.counters), self.profile.rtt);
        let result = self.inner.read_range(key, offset, len).await;

        let payload_len = result.as_ref().map(|b| b.len()).unwrap_or(0);
        let transfer = self.profile.transfer_duration(payload_len);
        guard.set_payload(payload_len as u64, transfer);

        let total_delay = self.profile.rtt + transfer;
        if !total_delay.is_zero() {
            tokio::time::sleep(total_delay).await;
        }

        guard.mark_completed();
        self.counters
            .completed_requests
            .fetch_add(1, Ordering::Relaxed);
        self.counters.range_requests.fetch_add(1, Ordering::Relaxed);
        self.counters
            .bytes_downloaded
            .fetch_add(payload_len as u64, Ordering::Relaxed);
        result
    }

    async fn object_len(&self, key: &str) -> Result<u64, StoreError> {
        let _permit = self.acquire_slot().await;
        let mut guard = InFlightGuard::new(Arc::clone(&self.counters), self.profile.rtt);
        let result = self.inner.object_len(key).await;

        if !self.profile.rtt.is_zero() {
            tokio::time::sleep(self.profile.rtt).await;
        }

        guard.mark_completed();
        self.counters
            .completed_requests
            .fetch_add(1, Ordering::Relaxed);
        self.counters.head_requests.fetch_add(1, Ordering::Relaxed);
        result
    }

    async fn list_objects(&self) -> Result<Vec<ObjectMeta>, StoreError> {
        let _permit = self.acquire_slot().await;
        let mut guard = InFlightGuard::new(Arc::clone(&self.counters), self.profile.rtt);
        let result = self.inner.list_objects().await;

        // Approximate JSON/listing payload size for bandwidth accounting.
        let payload_len = result
            .as_ref()
            .map(|items| items.iter().map(|m| m.key.len() + 16).sum())
            .unwrap_or(0);
        let transfer = self.profile.transfer_duration(payload_len);
        guard.set_payload(payload_len as u64, transfer);

        let total_delay = self.profile.rtt + transfer;
        if !total_delay.is_zero() {
            tokio::time::sleep(total_delay).await;
        }

        guard.mark_completed();
        self.counters
            .completed_requests
            .fetch_add(1, Ordering::Relaxed);
        self.counters.list_requests.fetch_add(1, Ordering::Relaxed);
        self.counters
            .bytes_downloaded
            .fetch_add(payload_len as u64, Ordering::Relaxed);
        result
    }
}
