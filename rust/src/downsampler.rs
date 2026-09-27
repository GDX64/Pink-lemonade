//! Shared Rust interface for downsampling strategies.

use crate::MergeResult;

/// Configures a downsampler and produces GPU instances from packed x/y/weight triplets.
/// The meaning of the merge threshold depends on the strategy.
pub trait Downsampler {
    fn set_view_min_x(&mut self, value: f64);
    fn set_view_max_x(&mut self, value: f64);
    fn set_view_min_y(&mut self, value: f64);
    fn set_view_max_y(&mut self, value: f64);
    fn set_screen_w(&mut self, value: f64);
    fn set_screen_h(&mut self, value: f64);
    fn set_merge_threshold(&mut self, value: f64);
    fn set_sigma_size_px(&mut self, value: f64);
    fn set_data_f64(&mut self, data: Box<[f64]>);
    fn merge_points(&self) -> MergeResult;
}
