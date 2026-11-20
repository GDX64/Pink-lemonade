//! Core data, query, frame, and configuration types shared across progressive strategies.

use std::fmt;

/// A single weighted 2D sample `(x, y, weight)`.
///
/// Datasets ingested by a [`crate::DatasetBuilder`] must be ordered by non-decreasing `x`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
    pub weight: f64,
}

impl Point {
    pub const fn new(x: f64, y: f64, weight: f64) -> Self {
        Self { x, y, weight }
    }
}

/// Parameters describing the user's current viewport and screen resolution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewQuery {
    pub min_x: f64,
    pub max_x: f64,
    /// Optional explicit Y viewport bounds `[min_y, max_y]`.
    ///
    /// When `None`, the strategy uses the visible Y extrema of `[min_x, max_x]`
    /// (matching `ViewManager` auto-scaling in the chart).
    pub y_bounds: Option<[f64; 2]>,
    pub screen_w: f64,
    pub screen_h: f64,
    pub sigma_px: f64,
    pub merge_threshold: f64,
}

impl ViewQuery {
    pub const fn new(
        min_x: f64,
        max_x: f64,
        screen_w: f64,
        screen_h: f64,
        sigma_px: f64,
        merge_threshold: f64,
    ) -> Self {
        Self {
            min_x,
            max_x,
            y_bounds: None,
            screen_w,
            screen_h,
            sigma_px,
            merge_threshold,
        }
    }

    pub const fn with_y_bounds(mut self, min_y: f64, max_y: f64) -> Self {
        self.y_bounds = Some([min_y, max_y]);
        self
    }

    /// Data units -> sigma-normalised screen units, per axis.
    ///
    /// Matches `View::factors()` in `pink-lemonade-wasm`.
    pub fn factors(&self, y_range: [f64; 2]) -> (f64, f64) {
        let x_den = (self.max_x - self.min_x).max(1e-12);
        let y_den = (y_range[1] - y_range[0]).max(1e-12);
        let sigma = self.sigma_px.max(1e-12);
        (self.screen_w / x_den / sigma, self.screen_h / y_den / sigma)
    }

    /// Effective Y bounds to use for screen-space projection: explicit `y_bounds` if set,
    /// otherwise the provided visible data range.
    pub fn effective_y_bounds(&self, visible_y_range: [f64; 2]) -> [f64; 2] {
        self.y_bounds.unwrap_or(visible_y_range)
    }
}

/// Convergence state of a progressively rendered [`Frame`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameStatus {
    /// Intermediate coarse or partially fetched frame; more frames will follow unless cancelled.
    Refining,
    /// Final converged frame at the requested screen resolution.
    Complete,
}

/// A single rendered frame emitted by a [`crate::ProgressiveClient`].
///
/// Mirrors the output contract of `TreeDownsampler` + `visibleYRange` in `pink-lemonade-wasm`:
/// - `gpu_instances`: flat `[x, y, amplitude, p00, p01, p10, p11]` per Gaussian kernel (7 `f32`s).
/// - `line_points`: flat `[x, y, ...]` real samples sorted by `x` for the line overlay.
/// - `y_range`: exact (or progressively refined) visible `[min_y, max_y]` in `[min_x, max_x]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub gpu_instances: Vec<f32>,
    pub line_points: Vec<f32>,
    pub y_range: Option<[f64; 2]>,
    pub status: FrameStatus,
}

impl Frame {
    /// Number of packed 7-float Gaussian kernels in `gpu_instances`.
    pub fn kernel_count(&self) -> usize {
        self.gpu_instances.len() / 7
    }

    pub fn is_complete(&self) -> bool {
        self.status == FrameStatus::Complete
    }

    pub fn empty(status: FrameStatus) -> Self {
        Self {
            gpu_instances: Vec::new(),
            line_points: Vec::new(),
            y_range: None,
            status,
        }
    }

    pub fn empty_complete() -> Self {
        Self::empty(FrameStatus::Complete)
    }
}

/// Runtime limits and tuning passed when opening a [`crate::ProgressiveClient`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClientConfig {
    /// Maximum client-side memory budget in bytes for caches and index state.
    pub memory_budget_bytes: usize,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            // 64 MiB default client budget.
            memory_budget_bytes: 64 * 1024 * 1024,
        }
    }
}

/// Errors returned by static byte-range storage backends.
#[derive(Debug)]
pub enum StoreError {
    NotFound(String),
    InvalidKey(String),
    OutOfRange {
        key: String,
        offset: u64,
        object_len: u64,
    },
    Io(std::io::Error),
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(key) => write!(f, "object not found: {key}"),
            Self::InvalidKey(key) => write!(f, "invalid object key: {key}"),
            Self::OutOfRange {
                key,
                offset,
                object_len,
            } => write!(
                f,
                "byte range offset {offset} out of range for '{key}' (len {object_len})"
            ),
            Self::Io(err) => write!(f, "storage I/O error: {err}"),
        }
    }
}

impl std::error::Error for StoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for StoreError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

/// Errors returned during offline dataset building.
#[derive(Debug)]
pub enum BuildError {
    Store(StoreError),
    InvalidInput(String),
    Format(String),
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(err) => write!(f, "build storage error: {err}"),
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::Format(msg) => write!(f, "format error: {msg}"),
        }
    }
}

impl std::error::Error for BuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Store(err) => Some(err),
            _ => None,
        }
    }
}

impl From<StoreError> for BuildError {
    fn from(err: StoreError) -> Self {
        Self::Store(err)
    }
}

/// Errors returned during client initialization or progressive querying.
#[derive(Debug)]
pub enum ClientError {
    Store(StoreError),
    CorruptData(String),
    Cancelled,
}

impl fmt::Display for ClientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(err) => write!(f, "client storage error: {err}"),
            Self::CorruptData(msg) => write!(f, "corrupt artifact data: {msg}"),
            Self::Cancelled => write!(f, "query cancelled"),
        }
    }
}

impl std::error::Error for ClientError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Store(err) => Some(err),
            _ => None,
        }
    }
}

impl From<StoreError> for ClientError {
    fn from(err: StoreError) -> Self {
        Self::Store(err)
    }
}
