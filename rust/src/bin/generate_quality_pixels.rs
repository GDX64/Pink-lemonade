#[path = "../../benches/common/fixture.rs"]
mod fixture;

use std::collections::HashSet;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use fixture::{POINTS, Random, SCREEN_H, SCREEN_W, SIGMA_PX, input_data};

const SAMPLES: usize = 1_000;
const SEED: u64 = 0x510e_527f_ade6_82d1;

fn main() {
    let data = input_data();
    let mut random = Random(SEED);
    let mut seen = HashSet::new();
    let mut pixels = Vec::with_capacity(SAMPLES);

    while pixels.len() < SAMPLES {
        let point = (random.next_u64() as usize % (data.len() / 3)) * 3;
        let center_x = data[point] * SCREEN_W;
        let center_y = data[point + 1] * SCREEN_H;

        // Sample within two sigma of an input kernel. This guarantees that
        // every chosen pixel receives meaningful original density.
        let angle = random.unit() * std::f64::consts::TAU;
        let radius = 2.0 * SIGMA_PX * random.unit().sqrt();
        let x = (center_x + radius * angle.cos()).floor() as i32;
        let y = (center_y + radius * angle.sin()).floor() as i32;
        if x < 0 || x >= SCREEN_W as i32 || y < 0 || y >= SCREEN_H as i32 {
            continue;
        }
        if seen.insert((x, y)) {
            pixels.push((x, y));
        }
    }

    let mut csv = String::from("x,y,original_density\n");
    for (x, y) in pixels {
        let density = original_density(&data, x as f64 + 0.5, y as f64 + 0.5);
        assert!(density > 0.0);
        writeln!(csv, "{x},{y},{density:.17e}").unwrap();
    }

    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("benches/data/quality_pixels_{}m.csv", POINTS / 1_000_000));
    fs::write(&path, csv).unwrap();
    println!("wrote {SAMPLES} occupied pixels and original densities to {}", path.display());
}

fn original_density(data: &[f64], px: f64, py: f64) -> f64 {
    // The input is x-sorted. Outside ten horizontal sigmas, even a million
    // unit-weight kernels contribute less than 1e-16 in total density.
    let x_min = (px - 10.0 * SIGMA_PX) / SCREEN_W;
    let x_max = (px + 10.0 * SIGMA_PX) / SCREEN_W;
    let start = data.partition_point_triplets(x_min, false);
    let end = data.partition_point_triplets(x_max, true);
    data[start * 3..end * 3].chunks_exact(3)
        .map(|point| {
            let dx = (px - point[0] * SCREEN_W) / SIGMA_PX;
            let dy = (py - point[1] * SCREEN_H) / SIGMA_PX;
            point[2] * (-0.5 * (dx * dx + dy * dy)).exp() / (2.0 * std::f64::consts::PI)
        })
        .sum()
}

trait TripletSearch {
    fn partition_point_triplets(&self, x: f64, inclusive: bool) -> usize;
}

impl TripletSearch for [f64] {
    fn partition_point_triplets(&self, x: f64, inclusive: bool) -> usize {
        let (mut low, mut high) = (0, self.len() / 3);
        while low < high {
            let middle = (low + high) / 2;
            if self[middle * 3] < x || (inclusive && self[middle * 3] == x) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        low
    }
}
