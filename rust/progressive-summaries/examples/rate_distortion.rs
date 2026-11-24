//! Offline rate-distortion benchmark: Haar pyramid (ε sweep) vs moment x-tree (threshold sweep).
//!
//! Haar runs use the L1 budget over all coefficients by default (`--budget l2`,
//! `--keep-sub-emit`, `--detail D` and `--emit-cell S` reproduce the other variants).
//!
//! ```text
//! cargo run --profile fast -p pink-lemonade-progressive-summaries --example rate_distortion \
//!     --manifest-path rust/Cargo.toml -- [--points 1000000] [--samples 1000] [--sigma 16] [--out file.json]
//! ```
//!
//! Every view is measured cold. Bytes use the sizes in `haar` and `xtree`; error is the
//! harness's density NMAE at `--samples` reference pixels per view.

use std::fmt::Write as _;
use std::time::Instant;

use pink_lemonade_progressive_core::{Point, ViewQuery};
use pink_lemonade_progressive_harness::fixture::{DatasetShape, DatasetStream, SplitMix64};
use pink_lemonade_progressive_harness::quality::{PixelSample, evaluate_density_nmae};
use pink_lemonade_progressive_summaries::{Budget, HaarIndex, HaarParams, XTree};

const MAX_LEVEL: u32 = 22;
const TREE_THRESHOLDS: [f64; 3] = [0.5, 1.0, 2.0];
const EPSILONS: [f64; 7] = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5];

struct Args {
    points: u64,
    samples: usize,
    sigma: f64,
    detail: u32,
    emit_cell: f64,
    out: Option<String>,
    only: Option<String>,
    sub_emit: bool,
    budget: Budget,
}

fn parse_args() -> Args {
    let mut a = Args {
        points: 1_000_000,
        samples: 1000,
        sigma: 16.0,
        detail: 3,
        emit_cell: 0.5,
        out: None,
        only: None,
        sub_emit: true,
        budget: Budget::L1,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let v = argv.get(i + 1).cloned().unwrap_or_default();
        match argv[i].as_str() {
            "--points" => a.points = v.parse().expect("--points"),
            "--samples" => a.samples = v.parse().expect("--samples"),
            "--sigma" => a.sigma = v.parse().expect("--sigma"),
            "--detail" => a.detail = v.parse().expect("--detail"),
            "--emit-cell" => a.emit_cell = v.parse().expect("--emit-cell"),
            "--out" => a.out = Some(v),
            "--only" => a.only = Some(v),
            "--budget" => {
                a.budget = match v.as_str() {
                    "l1" => Budget::L1,
                    "l2" => Budget::L2,
                    _ => panic!("--budget l1|l2"),
                }
            }
            "--keep-sub-emit" => {
                a.sub_emit = false;
                i -= 1;
            }
            other => panic!("unknown argument {other}"),
        }
        i += 2;
    }
    a
}

struct View {
    label: String,
    query: ViewQuery,
    y_range: [f64; 2],
}

fn visible_y_range(points: &[Point], min_x: f64, max_x: f64) -> [f64; 2] {
    let a = points.partition_point(|p| p.x < min_x);
    let b = points.partition_point(|p| p.x <= max_x);
    let mut r = [f64::INFINITY, f64::NEG_INFINITY];
    for p in &points[a..b] {
        r[0] = r[0].min(p.y);
        r[1] = r[1].max(p.y);
    }
    r
}

fn views(points: &[Point], y_center: f64, sigma: f64) -> Vec<View> {
    let (w, h) = (1920.0, 1080.0);
    let q = |a: f64, b: f64| ViewQuery::new(a, b, w, h, sigma, 1.0);
    let mut out = Vec::new();
    let mut auto = |label: String, a: f64, b: f64| {
        out.push(View {
            label,
            query: q(a, b),
            y_range: visible_y_range(points, a, b),
        });
    };
    auto("overview".into(), 0.0, 1.0);
    for width in [0.2f64, 0.05, 0.01, 0.001] {
        let a = (0.37 - width * 0.5).clamp(0.0, 1.0 - width);
        auto(format!("zoom w={width}"), a, a + width);
    }
    auto("window [0.40,0.45]".into(), 0.40, 0.45);
    let anchor = out.last().expect("window").y_range;
    for factor in [8.0, 32.0] {
        let span = anchor[1] - anchor[0];
        let half = 0.5 / factor;
        let lo = anchor[0] + (y_center - half) * span;
        let hi = anchor[0] + (y_center + half) * span;
        out.push(View {
            label: format!("window y{factor}x"),
            query: q(0.40, 0.45).with_y_bounds(lo, hi),
            y_range: anchor,
        });
    }
    out
}

/// Same sampling as `harness::quality::sample_reference_pixels`, summing only points within
/// 6σ in x of each pixel (the kernel cutoff), which is exact and much faster on x-sorted data.
fn reference_samples(
    points: &[Point],
    q: &ViewQuery,
    y_range: [f64; 2],
    count: usize,
    seed: u64,
) -> Vec<PixelSample> {
    let a = points.partition_point(|p| p.x < q.min_x);
    let b = points.partition_point(|p| p.x <= q.max_x);
    let visible = &points[a..b];
    if visible.is_empty() {
        return Vec::new();
    }
    let bounds = q.effective_y_bounds(y_range);
    let x_span = (q.max_x - q.min_x).max(1e-12);
    let y_span = (bounds[1] - bounds[0]).max(1e-12);
    let (fx, fy) = q.factors(bounds);
    let inv_two_pi = 1.0 / (2.0 * std::f64::consts::PI);
    let mut rng = SplitMix64(seed);
    (0..count)
        .map(|_| {
            let anchor = visible[(rng.next_u64() as usize) % visible.len()];
            let base_px = (anchor.x - q.min_x) / x_span * q.screen_w;
            let base_py = (anchor.y - bounds[0]) / y_span * q.screen_h;
            let px = (base_px + (rng.unit() * 3.0 - 1.5) * q.sigma_px).clamp(0.0, q.screen_w);
            let py = (base_py + (rng.unit() * 3.0 - 1.5) * q.sigma_px).clamp(0.0, q.screen_h);
            let dx_ = q.min_x + px / q.screen_w * x_span;
            let dy_ = bounds[0] + py / q.screen_h * y_span;
            let reach = 6.0 / fx;
            let s = visible.partition_point(|p| p.x < dx_ - reach);
            let e = visible.partition_point(|p| p.x <= dx_ + reach);
            let reference_density = visible[s..e]
                .iter()
                .map(|p| {
                    let ddx = (dx_ - p.x) * fx;
                    let ddy = (dy_ - p.y) * fy;
                    let r2 = ddx * ddx + ddy * ddy;
                    if r2 > 36.0 {
                        0.0
                    } else {
                        p.weight * inv_two_pi * (-0.5 * r2).exp()
                    }
                })
                .sum();
            PixelSample {
                px,
                py,
                reference_density,
            }
        })
        .collect()
}

struct Row {
    dataset: String,
    view: String,
    method: String,
    param: f64,
    bytes: usize,
    kernels: usize,
    nmae: f64,
    est: Option<f64>,
    detail: String,
}

fn fmt_bytes(b: usize) -> String {
    let b = b as f64;
    if b >= 1024.0 * 1024.0 {
        format!("{:.2} MiB", b / 1024.0 / 1024.0)
    } else {
        format!("{:.1} KiB", b / 1024.0)
    }
}

fn main() {
    let args = parse_args();
    let datasets = [
        (
            "random_walk",
            DatasetShape::RandomWalk { seed: 578_211 },
            0.5,
        ),
        (
            "white_noise",
            DatasetShape::NoisySpikes { seed: 4_242 },
            0.1,
        ),
        ("bursty_x", DatasetShape::BurstyX { seed: 9_001 }, 0.5),
    ];
    let mut rows: Vec<Row> = Vec::new();
    for (name, shape, y_center) in datasets {
        if args.only.as_deref().is_some_and(|o| o != name) {
            continue;
        }
        let t0 = Instant::now();
        let points = DatasetStream::new(shape, args.points, 65_536).collect_all();
        let tree = XTree::build(points.clone());
        let haar = HaarIndex::build(points.clone(), MAX_LEVEL);
        eprintln!("[{name}] built in {:.1?}", t0.elapsed());

        for (vi, view) in views(&points, y_center, args.sigma).into_iter().enumerate() {
            let samples = reference_samples(
                &points,
                &view.query,
                view.y_range,
                args.samples,
                0xC0FFEE + vi as u64,
            );
            let visible = {
                let a = points.partition_point(|p| p.x < view.query.min_x);
                let b = points.partition_point(|p| p.x <= view.query.max_x);
                b - a
            };
            for t in TREE_THRESHOLDS {
                let q = ViewQuery {
                    merge_threshold: t,
                    ..view.query
                };
                let (frame, st) = tree.render(&q, view.y_range);
                let m = evaluate_density_nmae(visible, &frame, &q, view.y_range, &samples)
                    .expect("nmae");
                rows.push(Row {
                    dataset: name.into(),
                    view: view.label.clone(),
                    method: "xtree".into(),
                    param: t,
                    bytes: st.bytes,
                    kernels: frame.kernel_count(),
                    nmae: m.normalized_mean_absolute_error_percent,
                    est: None,
                    detail: format!("nodes {} raw {}", st.nodes_read, st.raw_points),
                });
            }
            for eps in EPSILONS {
                let params = HaarParams {
                    emit_cell_sigma: args.emit_cell,
                    detail_levels: args.detail,
                    epsilon: eps,
                    threshold_sub_emit: args.sub_emit,
                    budget: args.budget,
                };
                let (frame, st) = haar.render(&view.query, view.y_range, params);
                let m = evaluate_density_nmae(visible, &frame, &view.query, view.y_range, &samples)
                    .expect("nmae");
                rows.push(Row {
                    dataset: name.into(),
                    view: view.label.clone(),
                    method: "haar".into(),
                    param: eps,
                    bytes: st.bytes,
                    kernels: frame.kernel_count(),
                    nmae: m.normalized_mean_absolute_error_percent,
                    est: Some(100.0 * st.estimated_error),
                    detail: format!(
                        "emit {:?} fin {:?} coefs {}/{} roots {} raw {} neg {}",
                        st.emit_level,
                        st.finest_level,
                        st.kept_coefs,
                        st.total_coefs,
                        st.root_cells,
                        st.raw_points,
                        st.negative_kernels
                    ),
                });
            }
            eprintln!("[{name}] {} done ({:.1?})", view.label, t0.elapsed());
        }
    }

    // Markdown tables.
    let mut md = String::new();
    let mut last = (String::new(), String::new());
    for r in &rows {
        if r.dataset != last.0 {
            let _ = writeln!(md, "\n### {}\n", r.dataset);
        }
        if (r.dataset.clone(), r.view.clone()) != last {
            let _ = writeln!(
                md,
                "\n**{}**\n\n| method | param | bytes | kernels | NMAE % | est err % | detail |\n|---|---:|---:|---:|---:|---:|---|",
                r.view
            );
            last = (r.dataset.clone(), r.view.clone());
        }
        let _ = writeln!(
            md,
            "| {} | {} | {} | {} | {:.2} | {} | {} |",
            r.method,
            r.param,
            fmt_bytes(r.bytes),
            r.kernels,
            r.nmae,
            r.est.map_or("".into(), |e| format!("{e:.2}")),
            r.detail
        );
    }
    println!("{md}");

    if let Some(path) = args.out {
        let mut js = String::from("[\n");
        for (i, r) in rows.iter().enumerate() {
            let _ = writeln!(
                js,
                "  {{\"dataset\":\"{}\",\"view\":\"{}\",\"method\":\"{}\",\"param\":{},\"bytes\":{},\"kernels\":{},\"nmae_percent\":{:.4},\"est_error_percent\":{},\"detail\":\"{}\"}}{}",
                r.dataset,
                r.view,
                r.method,
                r.param,
                r.bytes,
                r.kernels,
                r.nmae,
                r.est.map_or("null".into(), |e| format!("{e:.4}")),
                r.detail,
                if i + 1 < rows.len() { "," } else { "" }
            );
        }
        js.push_str("]\n");
        std::fs::write(&path, js).expect("write json");
        eprintln!("wrote {path}");
    }
}
