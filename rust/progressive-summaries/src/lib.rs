//! Offline node-summary experiments (no network, no byte layout).
//!
//! - [`haar`]: Haar-coded density pyramid; the client keeps the coefficients whose blurred
//!   energy matters for the view, up to a relative L2 error `ε`.
//! - [`xtree`]: baseline binary moment x-tree with today's bbox resolve rule.
//! - [`blur`]: closed-form energy gains of Haar basis functions under the render kernel.

pub mod blur;
pub mod haar;
pub mod xtree;

pub use haar::{Budget, HaarIndex, HaarParams, HaarStats};
pub use xtree::{XTree, XTreeStats};
