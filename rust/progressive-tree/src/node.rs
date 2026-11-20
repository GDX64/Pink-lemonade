//! Moment-preserving tree node with bounding box and real-sample line extrema.
//!
//! Mirrors the mathematical definition in `pink-lemonade-wasm`'s `tree_merge::Node`
//! with a fixed 128-byte little-endian binary encoding for static byte-range storage.

use pink_lemonade_progressive_core::Point;

pub const NODE_BYTES: usize = 16 * 8; // 128 bytes
pub const POINT_BYTES: usize = 3 * 8; // 24 bytes

#[derive(Debug, Clone, Copy)]
pub struct Node {
    pub mass: f64,
    pub x: f64,
    pub y: f64,
    pub m2_xx: f64,
    pub m2_xy: f64,
    pub m2_yy: f64,
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
    pub first_x: f64,
    pub first_y: f64,
    pub last_x: f64,
    pub last_y: f64,
    pub min_y_x: f64,
    pub max_y_x: f64,
}

impl Default for Node {
    fn default() -> Self {
        Self {
            mass: 0.0,
            x: 0.0,
            y: 0.0,
            m2_xx: 0.0,
            m2_xy: 0.0,
            m2_yy: 0.0,
            min_x: f64::INFINITY,
            max_x: f64::NEG_INFINITY,
            min_y: f64::INFINITY,
            max_y: f64::NEG_INFINITY,
            first_x: 0.0,
            first_y: 0.0,
            last_x: 0.0,
            last_y: 0.0,
            min_y_x: 0.0,
            max_y_x: 0.0,
        }
    }
}

impl Node {
    pub fn from_point(p: Point) -> Self {
        Self {
            mass: p.weight,
            x: p.x,
            y: p.y,
            min_x: p.x,
            max_x: p.x,
            min_y: p.y,
            max_y: p.y,
            first_x: p.x,
            first_y: p.y,
            last_x: p.x,
            last_y: p.y,
            min_y_x: p.x,
            max_y_x: p.x,
            ..Self::default()
        }
    }

    pub fn from_points(points: &[Point]) -> Self {
        let mut node = Self::default();
        for &p in points {
            node = Self::combined(node, Self::from_point(p));
        }
        node
    }

    /// Associative moment- and extrema-preserving merge of two adjacent nodes (`a` precedes `b`).
    pub fn combined(a: Self, b: Self) -> Self {
        if a.min_x > a.max_x {
            return b;
        }
        if b.min_x > b.max_x {
            return a;
        }
        let mut merged = if a.mass == 0.0 {
            b
        } else if b.mass == 0.0 {
            a
        } else {
            let mass = a.mass + b.mass;
            let fraction = b.mass / mass;
            let dx = b.x - a.x;
            let dy = b.y - a.y;
            let between = a.mass * b.mass / mass;
            Self {
                mass,
                x: a.x + fraction * dx,
                y: a.y + fraction * dy,
                m2_xx: a.m2_xx + b.m2_xx + between * dx * dx,
                m2_xy: a.m2_xy + b.m2_xy + between * dx * dy,
                m2_yy: a.m2_yy + b.m2_yy + between * dy * dy,
                ..Self::default()
            }
        };
        merged.min_x = a.min_x.min(b.min_x);
        merged.max_x = a.max_x.max(b.max_x);
        (merged.min_y, merged.min_y_x) = if b.min_y < a.min_y {
            (b.min_y, b.min_y_x)
        } else {
            (a.min_y, a.min_y_x)
        };
        (merged.max_y, merged.max_y_x) = if b.max_y > a.max_y {
            (b.max_y, b.max_y_x)
        } else {
            (a.max_y, a.max_y_x)
        };
        merged.first_x = a.first_x;
        merged.first_y = a.first_y;
        merged.last_x = b.last_x;
        merged.last_y = b.last_y;
        merged
    }

    pub fn is_empty(&self) -> bool {
        self.min_x > self.max_x
    }

    pub fn fits_threshold(&self, x_factor: f64, y_factor: f64, merge_threshold: f64) -> bool {
        let dx = (self.max_x - self.min_x) * x_factor;
        let dy = (self.max_y - self.min_y) * y_factor;
        let threshold = merge_threshold.max(1e-12);
        dx * dx + dy * dy < threshold * threshold
    }

    pub fn fits_merged_group(&self, x_factor: f64, y_factor: f64, merge_threshold: f64) -> bool {
        let dx = (self.x - self.min_x).max(self.max_x - self.x) * x_factor;
        let dy = (self.y - self.min_y).max(self.max_y - self.y) * y_factor;
        let threshold = merge_threshold.max(1e-12);
        dx * dx + dy * dy < threshold * threshold
    }

    /// Emit 7 GPU floats `[x, y, amplitude, p00, p01, p10, p11]` if mass > 0.
    pub fn emit_gpu(&self, x_factor: f64, y_factor: f64, out: &mut Vec<f32>) {
        if self.mass <= 0.0 {
            return;
        }
        let p00 = 1.0 + self.m2_xx / self.mass * x_factor * x_factor;
        let p01 = self.m2_xy / self.mass * x_factor * y_factor;
        let p11 = 1.0 + self.m2_yy / self.mass * y_factor * y_factor;
        let det = (p00 * p11 - p01 * p01).max(1e-12);
        let amplitude = self.mass / (2.0 * std::f64::consts::PI * det.sqrt());
        out.extend_from_slice(&[
            self.x as f32,
            self.y as f32,
            amplitude as f32,
            p00 as f32,
            p01 as f32,
            p01 as f32,
            p11 as f32,
        ]);
    }

    /// Emit up to 4 real samples (first, min/max in x order, last) for the line plot.
    pub fn emit_line(&self, out: &mut Vec<f32>) {
        if self.is_empty() {
            return;
        }
        let (lo, hi) = if self.min_y_x <= self.max_y_x {
            ((self.min_y_x, self.min_y), (self.max_y_x, self.max_y))
        } else {
            ((self.max_y_x, self.max_y), (self.min_y_x, self.min_y))
        };
        for (x, y) in [
            (self.first_x, self.first_y),
            lo,
            hi,
            (self.last_x, self.last_y),
        ] {
            push_line_point(out, x, y);
        }
    }

    pub fn to_bytes(self) -> [u8; NODE_BYTES] {
        let fields = [
            self.mass,
            self.x,
            self.y,
            self.m2_xx,
            self.m2_xy,
            self.m2_yy,
            self.min_x,
            self.max_x,
            self.min_y,
            self.max_y,
            self.first_x,
            self.first_y,
            self.last_x,
            self.last_y,
            self.min_y_x,
            self.max_y_x,
        ];
        let mut buf = [0u8; NODE_BYTES];
        for (i, v) in fields.into_iter().enumerate() {
            buf[i * 8..(i + 1) * 8].copy_from_slice(&v.to_le_bytes());
        }
        buf
    }

    pub fn from_bytes(buf: &[u8; NODE_BYTES]) -> Self {
        let f = |i: usize| f64::from_le_bytes(buf[i * 8..(i + 1) * 8].try_into().unwrap());
        Self {
            mass: f(0),
            x: f(1),
            y: f(2),
            m2_xx: f(3),
            m2_xy: f(4),
            m2_yy: f(5),
            min_x: f(6),
            max_x: f(7),
            min_y: f(8),
            max_y: f(9),
            first_x: f(10),
            first_y: f(11),
            last_x: f(12),
            last_y: f(13),
            min_y_x: f(14),
            max_y_x: f(15),
        }
    }
}

pub fn push_line_point(out: &mut Vec<f32>, x: f64, y: f64) {
    let (x, y) = (x as f32, y as f32);
    if let [.., px, py] = out.as_slice()
        && *px == x
        && *py == y
    {
        return;
    }
    out.extend_from_slice(&[x, y]);
}

pub fn decode_nodes(bytes: &[u8]) -> Result<Vec<Node>, String> {
    if !bytes.len().is_multiple_of(NODE_BYTES) {
        return Err(format!(
            "node byte slice length {} is not a multiple of {NODE_BYTES}",
            bytes.len()
        ));
    }
    Ok(bytes
        .as_chunks::<NODE_BYTES>()
        .0
        .iter()
        .map(Node::from_bytes)
        .collect())
}

pub fn decode_points(bytes: &[u8]) -> Result<Vec<Point>, String> {
    if !bytes.len().is_multiple_of(POINT_BYTES) {
        return Err(format!(
            "point byte slice length {} is not a multiple of {POINT_BYTES}",
            bytes.len()
        ));
    }
    Ok(bytes
        .as_chunks::<POINT_BYTES>()
        .0
        .iter()
        .map(|c| {
            Point::new(
                f64::from_le_bytes(c[0..8].try_into().unwrap()),
                f64::from_le_bytes(c[8..16].try_into().unwrap()),
                f64::from_le_bytes(c[16..24].try_into().unwrap()),
            )
        })
        .collect())
}
