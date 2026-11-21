//! Moment-preserving tree node with bounding box and real-sample line extrema.
//!
//! On-wire and in-cache representation uses [`PackedNode`] (exactly 64 bytes — 2× smaller
//! than 16 × `f64`):
//! - Exact `f64` bounding box `[min_x, max_x, min_y, max_y]` (with `first_x == min_x`,
//!   `last_x == max_x` by construction of [`Node::combined`], and exact `y_range` bounds).
//! - `f32` mass, bounding-box-relative mean `(tx, ty)`, and normalized centered covariance
//!   `(var_x / dx², cov_xy / (dx·dy), var_y / dy²)`.
//! - `u16` box-relative positions for `first_y`, `last_y`, `min_y_x`, `max_y_x`.

use pink_lemonade_progressive_core::Point;

pub const NODE_BYTES: usize = 64;
pub const POINT_BYTES: usize = 3 * 8; // 24 bytes

/// Compact 64-byte cache/wire representation of a [`Node`].
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PackedNode {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    mass: f32,
    tx: f32,
    ty: f32,
    norm_var_x: f32,
    norm_cov_xy: f32,
    norm_var_y: f32,
    q_first_y: u16,
    q_last_y: u16,
    q_min_y_x: u16,
    q_max_y_x: u16,
}

const _: () = assert!(std::mem::size_of::<PackedNode>() == NODE_BYTES);

impl PackedNode {
    pub fn pack(n: Node) -> Self {
        if n.is_empty() {
            return Self {
                min_x: f64::INFINITY,
                max_x: f64::NEG_INFINITY,
                min_y: f64::INFINITY,
                max_y: f64::NEG_INFINITY,
                mass: 0.0,
                tx: 0.0,
                ty: 0.0,
                norm_var_x: 0.0,
                norm_cov_xy: 0.0,
                norm_var_y: 0.0,
                q_first_y: 0,
                q_last_y: 0,
                q_min_y_x: 0,
                q_max_y_x: 0,
            };
        }
        let dx = n.max_x - n.min_x;
        let dy = n.max_y - n.min_y;
        let inv_mass = if n.mass > 0.0 { 1.0 / n.mass } else { 0.0 };

        let tx = if dx > 0.0 {
            ((n.x - n.min_x) / dx).clamp(0.0, 1.0) as f32
        } else {
            0.0
        };
        let ty = if dy > 0.0 {
            ((n.y - n.min_y) / dy).clamp(0.0, 1.0) as f32
        } else {
            0.0
        };
        let norm_var_x = if dx > 0.0 {
            ((n.m2_xx * inv_mass) / (dx * dx)).max(0.0) as f32
        } else {
            0.0
        };
        let norm_var_y = if dy > 0.0 {
            ((n.m2_yy * inv_mass) / (dy * dy)).max(0.0) as f32
        } else {
            0.0
        };
        let norm_cov_xy = if dx > 0.0 && dy > 0.0 {
            ((n.m2_xy * inv_mass) / (dx * dy)) as f32
        } else {
            0.0
        };

        Self {
            min_x: n.min_x,
            max_x: n.max_x,
            min_y: n.min_y,
            max_y: n.max_y,
            mass: n.mass as f32,
            tx,
            ty,
            norm_var_x,
            norm_cov_xy,
            norm_var_y,
            q_first_y: quantize_u16(n.first_y, n.min_y, dy),
            q_last_y: quantize_u16(n.last_y, n.min_y, dy),
            q_min_y_x: quantize_u16(n.min_y_x, n.min_x, dx),
            q_max_y_x: quantize_u16(n.max_y_x, n.min_x, dx),
        }
    }

    pub fn unpack(self) -> Node {
        if self.min_x > self.max_x {
            return Node::default();
        }
        let dx = self.max_x - self.min_x;
        let dy = self.max_y - self.min_y;
        let mass = self.mass as f64;
        Node {
            mass,
            x: self.min_x + self.tx as f64 * dx,
            y: self.min_y + self.ty as f64 * dy,
            m2_xx: self.norm_var_x as f64 * mass * dx * dx,
            m2_xy: self.norm_cov_xy as f64 * mass * dx * dy,
            m2_yy: self.norm_var_y as f64 * mass * dy * dy,
            min_x: self.min_x,
            max_x: self.max_x,
            min_y: self.min_y,
            max_y: self.max_y,
            first_x: self.min_x,
            first_y: dequantize_u16(self.q_first_y, self.min_y, dy),
            last_x: self.max_x,
            last_y: dequantize_u16(self.q_last_y, self.min_y, dy),
            min_y_x: dequantize_u16(self.q_min_y_x, self.min_x, dx),
            max_y_x: dequantize_u16(self.q_max_y_x, self.min_x, dx),
        }
    }

    pub fn to_bytes(self) -> [u8; NODE_BYTES] {
        let mut buf = [0u8; NODE_BYTES];
        buf[0..8].copy_from_slice(&self.min_x.to_le_bytes());
        buf[8..16].copy_from_slice(&self.max_x.to_le_bytes());
        buf[16..24].copy_from_slice(&self.min_y.to_le_bytes());
        buf[24..32].copy_from_slice(&self.max_y.to_le_bytes());
        buf[32..36].copy_from_slice(&self.mass.to_le_bytes());
        buf[36..40].copy_from_slice(&self.tx.to_le_bytes());
        buf[40..44].copy_from_slice(&self.ty.to_le_bytes());
        buf[44..48].copy_from_slice(&self.norm_var_x.to_le_bytes());
        buf[48..52].copy_from_slice(&self.norm_cov_xy.to_le_bytes());
        buf[52..56].copy_from_slice(&self.norm_var_y.to_le_bytes());
        buf[56..58].copy_from_slice(&self.q_first_y.to_le_bytes());
        buf[58..60].copy_from_slice(&self.q_last_y.to_le_bytes());
        buf[60..62].copy_from_slice(&self.q_min_y_x.to_le_bytes());
        buf[62..64].copy_from_slice(&self.q_max_y_x.to_le_bytes());
        buf
    }

    pub fn from_bytes(buf: &[u8; NODE_BYTES]) -> Self {
        Self {
            min_x: f64::from_le_bytes(buf[0..8].try_into().unwrap()),
            max_x: f64::from_le_bytes(buf[8..16].try_into().unwrap()),
            min_y: f64::from_le_bytes(buf[16..24].try_into().unwrap()),
            max_y: f64::from_le_bytes(buf[24..32].try_into().unwrap()),
            mass: f32::from_le_bytes(buf[32..36].try_into().unwrap()),
            tx: f32::from_le_bytes(buf[36..40].try_into().unwrap()),
            ty: f32::from_le_bytes(buf[40..44].try_into().unwrap()),
            norm_var_x: f32::from_le_bytes(buf[44..48].try_into().unwrap()),
            norm_cov_xy: f32::from_le_bytes(buf[48..52].try_into().unwrap()),
            norm_var_y: f32::from_le_bytes(buf[52..56].try_into().unwrap()),
            q_first_y: u16::from_le_bytes(buf[56..58].try_into().unwrap()),
            q_last_y: u16::from_le_bytes(buf[58..60].try_into().unwrap()),
            q_min_y_x: u16::from_le_bytes(buf[60..62].try_into().unwrap()),
            q_max_y_x: u16::from_le_bytes(buf[62..64].try_into().unwrap()),
        }
    }
}

fn quantize_u16(val: f64, lo: f64, span: f64) -> u16 {
    if span > 0.0 {
        (((val - lo) / span).clamp(0.0, 1.0) * 65535.0).round() as u16
    } else {
        0
    }
}

fn dequantize_u16(q: u16, lo: f64, span: f64) -> f64 {
    lo + (q as f64 / 65535.0) * span
}

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

    /// Associative moment- and extrema-preserving merge of two nodes.
    ///
    /// `first`/`last` samples are chosen by x (ties keep `a`'s first and `b`'s last), so the
    /// merge is valid both for x-adjacent nodes (`a` precedes `b`) and for arbitrary 2D cells
    /// whose x-ranges overlap. In every case `first_x == min_x` and `last_x == max_x`.
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
        (merged.first_x, merged.first_y) = if b.first_x < a.first_x {
            (b.first_x, b.first_y)
        } else {
            (a.first_x, a.first_y)
        };
        (merged.last_x, merged.last_y) = if a.last_x > b.last_x {
            (a.last_x, a.last_y)
        } else {
            (b.last_x, b.last_y)
        };
        merged
    }

    /// Weighted variance of `y` (`m2_yy / mass`), or `0` for massless nodes.
    pub fn var_y(&self) -> f64 {
        if self.mass > 0.0 {
            (self.m2_yy / self.mass).max(0.0)
        } else {
            0.0
        }
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
        PackedNode::pack(self).to_bytes()
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

pub fn decode_packed_nodes(bytes: &[u8]) -> Result<Vec<PackedNode>, String> {
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
        .map(PackedNode::from_bytes)
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
