//! Downsampling strategies for resolution-bounded KDE, one module per method.
//!
//! - [`merge`]: the proposed O(N) screen-space one-pass weighted merge.
//! - [`runnalls`]: KL-optimal greedy pairwise reduction (quality reference).
//! - [`salmond`]: clustering reduction, CAF (grouped, cheaper reference).
//!
//! All three share [`common::MergeResult`] and the same JS-facing setters, so a
//! benchmark can swap one for another without changing its call sites.

mod common;
mod merge;
mod runnalls;
mod salmond;

pub use common::MergeResult;
pub use merge::Downsampler;
pub use runnalls::KlDownsampler;
pub use salmond::SalmondDownsampler;
