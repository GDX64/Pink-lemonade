//! Shared building blocks for progressive strategies over static byte-range storage.
//!
//! - [`node`]: moment- and extrema-preserving [`Node`] plus its compact 64-byte [`PackedNode`].
//! - [`cell`]: compact 24-byte [`MomentCell`] and a lean [`Moments`] accumulator.
//! - [`fetch`]: gap-bridging range coalescing and runtime-agnostic concurrent joins.
//! - [`cache`]: strictly byte-bounded LRU cache for tree pages, raw leaves, and moment cells.
//! - [`format`] / [`writer`] / [`reader`]: the paged multi-level moment hierarchy shared by the
//!   1D x-ordered tree and the 2D k-d tree (`manifest.bin`, `level_*.bin`, `points.bin`).
//!   The hierarchy is defined over an arbitrary permutation of the input points: leaf `i` covers
//!   permuted points `[i·L, (i+1)·L)` and node `j` at level `ℓ` covers children
//!   `[j·F, (j+1)·F)` at level `ℓ-1`.
//! - [`render`]: frontier resolution policies and frame assembly (kernels + line overlay).
//!
//! Like `progressive-core`, this crate avoids any async runtime so strategies built on it
//! compile for `wasm32-unknown-unknown`.

pub mod cache;
pub mod cell;
pub mod fetch;
pub mod format;
pub mod node;
pub mod reader;
pub mod render;
pub mod writer;

pub use cache::{CacheKey, CacheValue, LruPageCache};
pub use cell::{
    CELL_BYTES, MomentCell, Moments, QUANTIZED_MOMENTS_BYTES, QuantizedMoments, SPARSE_CELL_BYTES,
    SparseCell, decode_cells, decode_sparse_cells,
};
pub use fetch::{coalesce_runs, coalesce_spans, join2, try_join_all};
pub use format::{MANIFEST_KEY, Manifest, POINTS_KEY, level_key};
pub use node::{NODE_BYTES, Node, POINT_BYTES, PackedNode, decode_packed_nodes, decode_points};
pub use reader::{
    DEFAULT_MANIFEST_READ_BYTES, Decision, FrontierItem, HierarchyReader, read_manifest,
    read_small_object, read_whole_object,
};
pub use render::{
    assemble_frame, collect_points, offscreen_y, push_group, sort_line_points, standard_decision,
};
pub use writer::{PointCollector, level_counts, validate_points, write_hierarchy};
