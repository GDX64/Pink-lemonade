//! Downsampling strategies for resolution-bounded KDE, one module per method.
//!
//! - [`merge`]: the proposed O(N) screen-space one-pass weighted merge.
//! - [`tree_merge`]: reusable moment tree for fast changes to the view range.
//! - [`runnalls`]: KL-optimal greedy pairwise reduction (quality reference).
//! - [`salmond`]: clustering reduction, CAF (grouped, cheaper reference).
//!
//! All strategies share [`common::MergeResult`] and the same JS-facing setters, so a
//! benchmark can swap one for another without changing its call sites.

mod common;
pub mod downsampler;
mod merge;
mod runnalls;
mod salmond;
mod tree_merge;

pub use common::MergeResult;
pub use merge::Downsampler;
pub use runnalls::KlDownsampler;
pub use salmond::SalmondDownsampler;
pub use tree_merge::TreeDownsampler;
