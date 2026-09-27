pub const POINTS: usize = 1_000_000;
pub const SCREEN_W: f64 = 1920.0;
pub const SCREEN_H: f64 = 1080.0;
pub const SIGMA_PX: f64 = 16.0;

pub fn input_data() -> Vec<f64> {
    let mut data = Vec::with_capacity(POINTS * 3);
    for i in 0..POINTS {
        let x = i as f64 / POINTS as f64;
        // Keep the same curve shape as the 100K fixture while varying density.
        let y = 0.5 + 0.25 * (i as f64 * 300.0 / POINTS as f64).sin();
        data.extend_from_slice(&[x, y, 1.0]);
    }
    data
}

/// Deterministic SplitMix64 generator for reproducible benchmark inputs.
pub struct Random(pub u64);

impl Random {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64
    }
}
