use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct MergeResult {
    gpu_instances: Box<[f32]>,
    count: u32,
}

#[wasm_bindgen]
impl MergeResult {
    #[wasm_bindgen(getter, js_name = gpuInstances)]
    pub fn gpu_instances(&self) -> Box<[f32]> {
        self.gpu_instances.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn count(&self) -> u32 {
        self.count
    }
}

#[wasm_bindgen]
pub struct Downsampler {
    view_min_x: f64,
    view_max_x: f64,
    view_min_y: f64,
    view_max_y: f64,
    screen_w: f64,
    screen_h: f64,
    merge_threshold: f64,
    sigma_size_px: f64,
    data_f64: Vec<f64>,
}

impl Default for Downsampler {
    fn default() -> Self {
        Self {
            view_min_x: 0.0,
            view_max_x: 1.0,
            view_min_y: 0.0,
            view_max_y: 1.0,
            screen_w: 1.0,
            screen_h: 1.0,
            merge_threshold: 1.0,
            sigma_size_px: 1.0,
            data_f64: Vec::new(),
        }
    }
}

#[wasm_bindgen]
impl Downsampler {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Downsampler {
        Downsampler::default()
    }

    #[wasm_bindgen(js_name = setViewMinX)]
    pub fn set_view_min_x(&mut self, value: f64) {
        self.view_min_x = value;
    }

    #[wasm_bindgen(js_name = setViewMaxX)]
    pub fn set_view_max_x(&mut self, value: f64) {
        self.view_max_x = value;
    }

    #[wasm_bindgen(js_name = setViewMinY)]
    pub fn set_view_min_y(&mut self, value: f64) {
        self.view_min_y = value;
    }

    #[wasm_bindgen(js_name = setViewMaxY)]
    pub fn set_view_max_y(&mut self, value: f64) {
        self.view_max_y = value;
    }

    #[wasm_bindgen(js_name = setScreenW)]
    pub fn set_screen_w(&mut self, value: f64) {
        self.screen_w = value;
    }

    #[wasm_bindgen(js_name = setScreenH)]
    pub fn set_screen_h(&mut self, value: f64) {
        self.screen_h = value;
    }

    #[wasm_bindgen(js_name = setMergeThreshold)]
    pub fn set_merge_threshold(&mut self, value: f64) {
        self.merge_threshold = value;
    }

    #[wasm_bindgen(js_name = setSigmaSizePx)]
    pub fn set_sigma_size_px(&mut self, value: f64) {
        self.sigma_size_px = value;
    }

    #[wasm_bindgen(js_name = setDataF64)]
    pub fn set_data_f64(&mut self, data: Box<[f64]>) {
        self.data_f64 = data.into_vec();
    }

    #[wasm_bindgen(js_name = mergePoints)]
    pub fn merge_points(&self) -> MergeResult {
        if self.data_f64.len() < 3 {
            return MergeResult {
                gpu_instances: Vec::<f32>::new().into_boxed_slice(),
                count: 0,
            };
        }

        let x_den = (self.view_max_x - self.view_min_x).max(1e-12);
        let y_den = (self.view_max_y - self.view_min_y).max(1e-12);
        let sigma = self.sigma_size_px.max(1e-12);
        let x_factor = self.screen_w / x_den / sigma;
        let y_factor = self.screen_h / y_den / sigma;

        let to_sx = |x: f64| (x - self.view_min_x) * x_factor;
        let to_sy = |y: f64| (y - self.view_min_y) * y_factor;

        let start_idx = lower_bound_triplets(&self.data_f64, self.view_min_x);
        let end_idx = upper_bound_triplets(&self.data_f64, self.view_max_x);

        if end_idx <= start_idx {
            return MergeResult {
                gpu_instances: Vec::<f32>::new().into_boxed_slice(),
                count: 0,
            };
        }

        let merged = merge_pass(
            &self.data_f64,
            start_idx,
            end_idx,
            self.merge_threshold.max(1e-12),
            to_sx,
            to_sy,
        );

        let count = (merged.len() / 7) as u32;
        let mut gpu_instances = vec![0.0_f32; merged.len()];
        for (i, value) in merged.iter().enumerate() {
            gpu_instances[i] = *value as f32;
        }

        MergeResult {
            gpu_instances: gpu_instances.into_boxed_slice(),
            count,
        }
    }
}

fn lower_bound_triplets(data: &[f64], target_x: f64) -> usize {
    let n = data.len() / 3;
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let x = data[mid * 3];
        if x < target_x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

fn upper_bound_triplets(data: &[f64], target_x: f64) -> usize {
    let n = data.len() / 3;
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let x = data[mid * 3];
        if x <= target_x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

fn merge_pass<FSX, FSY>(
    pts: &[f64],
    start_idx: usize,
    end_idx: usize,
    threshold_px: f64,
    to_sx: FSX,
    to_sy: FSY,
) -> Vec<f64>
where
    FSX: Fn(f64) -> f64,
    FSY: Fn(f64) -> f64,
{
    let mut out = Vec::<f64>::with_capacity((end_idx - start_idx) * 7);
    let t2 = threshold_px * threshold_px;

    let mut has_cluster = false;
    let mut first_x = 0.0;
    let mut first_y = 0.0;
    let mut cluster_x = 0.0;
    let mut cluster_y = 0.0;
    let mut cluster_w = 0.0;
    let mut cluster_start = 0usize;
    let mut cluster_end = 0usize;

    let flush_cluster = |out: &mut Vec<f64>,
                         cluster_x: f64,
                         cluster_y: f64,
                         cluster_w: f64,
                         cluster_start: usize,
                         cluster_end: usize| {
        let cx = to_sx(cluster_x);
        let cy = to_sy(cluster_y);
        let mut weighted_xx = 0.0;
        let mut weighted_xy = 0.0;
        let mut weighted_yy = 0.0;

        for i in cluster_start..=cluster_end {
            let base = i * 3;
            let x = pts[base];
            let y = pts[base + 1];
            let w = pts[base + 2];
            let dx = to_sx(x) - cx;
            let dy = to_sy(y) - cy;
            weighted_xx += w * dx * dx;
            weighted_xy += w * dx * dy;
            weighted_yy += w * dy * dy;
        }

        let inv_w = 1.0 / cluster_w.max(1e-12);
        let p00 = 1.0 + weighted_xx * inv_w;
        let p01 = weighted_xy * inv_w;
        let p10 = p01;
        let p11 = 1.0 + weighted_yy * inv_w;
        let det_of_p = (p00 * p11 - p01 * p10).max(1e-12);
        let w = cluster_w / (2.0 * std::f64::consts::PI * det_of_p.sqrt());

        out.push(cluster_x);
        out.push(cluster_y);
        out.push(w);
        out.push(p00);
        out.push(p01);
        out.push(p10);
        out.push(p11);
    };

    for i in start_idx..end_idx {
        let base = i * 3;
        let xi = pts[base];
        let yi = pts[base + 1];
        let wi = pts[base + 2];

        if !has_cluster {
            has_cluster = true;
            first_x = xi;
            first_y = yi;
            cluster_x = xi;
            cluster_y = yi;
            cluster_w = wi;
            cluster_start = i;
            cluster_end = i;
            continue;
        }

        let new_w = cluster_w + wi;
        let new_x = (cluster_x * cluster_w + xi * wi) / new_w;
        let new_y = (cluster_y * cluster_w + yi * wi) / new_w;
        let new_sx = to_sx(new_x);
        let new_sy = to_sy(new_y);

        let dfx = to_sx(first_x) - new_sx;
        let dfy = to_sy(first_y) - new_sy;
        let dix = to_sx(xi) - new_sx;
        let diy = to_sy(yi) - new_sy;

        if dfx * dfx + dfy * dfy < t2 && dix * dix + diy * diy < t2 {
            cluster_x = new_x;
            cluster_y = new_y;
            cluster_w = new_w;
            cluster_end = i;
        } else {
            flush_cluster(
                &mut out,
                cluster_x,
                cluster_y,
                cluster_w,
                cluster_start,
                cluster_end,
            );

            has_cluster = true;
            first_x = xi;
            first_y = yi;
            cluster_x = xi;
            cluster_y = yi;
            cluster_w = wi;
            cluster_start = i;
            cluster_end = i;
        }
    }

    if has_cluster {
        flush_cluster(
            &mut out,
            cluster_x,
            cluster_y,
            cluster_w,
            cluster_start,
            cluster_end,
        );
    }

    out
}
