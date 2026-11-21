//! Compact moment cells for dense 2D moment planes (e.g. the moment ripmap).
//!
//! A [`MomentCell`] stores only mass, mean, and centered covariance (6 × `f32` = 24 bytes),
//! normalized against a *rectangle known to both builder and client* (an x-block's
//! `[min_x, max_x]` and one y-bin's `[bin_lo, bin_lo + bin_h]`). The rectangle itself is never
//! stored: it is derived from the enclosing x-node's exact `f64` bounds and the bin index.

use pink_lemonade_progressive_core::Point;

use crate::node::Node;

pub const CELL_BYTES: usize = 6 * 4;

/// Mass, mean, and centered second moments of a weighted point set.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Moments {
    pub mass: f64,
    pub x: f64,
    pub y: f64,
    pub m2_xx: f64,
    pub m2_xy: f64,
    pub m2_yy: f64,
}

impl Moments {
    /// Add one weighted point (Welford / Chan update).
    pub fn push(&mut self, p: Point) {
        self.merge(Self {
            mass: p.weight,
            x: p.x,
            y: p.y,
            ..Self::default()
        });
    }

    /// Associative moment-preserving merge.
    pub fn merge(&mut self, b: Self) {
        if b.mass == 0.0 {
            return;
        }
        if self.mass == 0.0 {
            *self = b;
            return;
        }
        let mass = self.mass + b.mass;
        let fraction = b.mass / mass;
        let dx = b.x - self.x;
        let dy = b.y - self.y;
        let between = self.mass * b.mass / mass;
        *self = Self {
            mass,
            x: self.x + fraction * dx,
            y: self.y + fraction * dy,
            m2_xx: self.m2_xx + b.m2_xx + between * dx * dx,
            m2_xy: self.m2_xy + b.m2_xy + between * dx * dy,
            m2_yy: self.m2_yy + b.m2_yy + between * dy * dy,
        };
    }
}

/// 24-byte normalized moment cell (all zeros = empty slot).
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct MomentCell {
    pub mass: f32,
    /// Mean x relative to the rectangle: `(x - x_lo) / dx`.
    pub tx: f32,
    /// Mean y relative to the rectangle: `(y - y_lo) / dy`.
    pub ty: f32,
    /// `var_x / dx²`.
    pub norm_var_x: f32,
    /// `cov_xy / (dx · dy)`.
    pub norm_cov_xy: f32,
    /// `var_y / dy²`.
    pub norm_var_y: f32,
}

const _: () = assert!(std::mem::size_of::<MomentCell>() == CELL_BYTES);

impl MomentCell {
    /// Normalize `m` against rectangle `[x_lo, x_lo + dx] × [y_lo, y_lo + dy]`.
    ///
    /// Degenerate spans (`dx == 0` / `dy == 0`) imply zero variance along that axis, so the
    /// normalized values are stored as `0`.
    pub fn pack(m: Moments, x_lo: f64, dx: f64, y_lo: f64, dy: f64) -> Self {
        if m.mass == 0.0 {
            return Self::default();
        }
        let inv_mass = 1.0 / m.mass;
        let rel = |v: f64, lo: f64, span: f64| {
            if span > 0.0 {
                ((v - lo) / span).clamp(0.0, 1.0) as f32
            } else {
                0.0
            }
        };
        let norm = |m2: f64, span_a: f64, span_b: f64| {
            if span_a > 0.0 && span_b > 0.0 {
                (m2 * inv_mass / (span_a * span_b)) as f32
            } else {
                0.0
            }
        };
        Self {
            mass: m.mass as f32,
            tx: rel(m.x, x_lo, dx),
            ty: rel(m.y, y_lo, dy),
            norm_var_x: norm(m.m2_xx, dx, dx).max(0.0),
            norm_cov_xy: norm(m.m2_xy, dx, dy),
            norm_var_y: norm(m.m2_yy, dy, dy).max(0.0),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.mass == 0.0
    }

    /// Reconstruct a renderable [`Node`] whose bounding box is the normalization rectangle.
    ///
    /// Line extrema fields are set to the rectangle corners; callers must not emit line
    /// samples from cell nodes (they come from the enclosing exact x-node instead).
    pub fn unpack(self, x_lo: f64, dx: f64, y_lo: f64, dy: f64) -> Node {
        let mass = self.mass as f64;
        let (x_hi, y_hi) = (x_lo + dx, y_lo + dy);
        Node {
            mass,
            x: x_lo + self.tx as f64 * dx,
            y: y_lo + self.ty as f64 * dy,
            m2_xx: self.norm_var_x as f64 * mass * dx * dx,
            m2_xy: self.norm_cov_xy as f64 * mass * dx * dy,
            m2_yy: self.norm_var_y as f64 * mass * dy * dy,
            min_x: x_lo,
            max_x: x_hi,
            min_y: y_lo,
            max_y: y_hi,
            first_x: x_lo,
            first_y: y_lo,
            last_x: x_hi,
            last_y: y_lo,
            min_y_x: x_lo,
            max_y_x: x_lo,
        }
    }

    pub fn to_bytes(self) -> [u8; CELL_BYTES] {
        let mut buf = [0u8; CELL_BYTES];
        for (i, v) in [
            self.mass,
            self.tx,
            self.ty,
            self.norm_var_x,
            self.norm_cov_xy,
            self.norm_var_y,
        ]
        .into_iter()
        .enumerate()
        {
            buf[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        buf
    }

    pub fn from_bytes(buf: &[u8; CELL_BYTES]) -> Self {
        let f = |i: usize| f32::from_le_bytes(buf[i * 4..i * 4 + 4].try_into().unwrap());
        Self {
            mass: f(0),
            tx: f(1),
            ty: f(2),
            norm_var_x: f(3),
            norm_cov_xy: f(4),
            norm_var_y: f(5),
        }
    }
}

pub fn decode_cells(bytes: &[u8]) -> Result<Vec<MomentCell>, String> {
    if !bytes.len().is_multiple_of(CELL_BYTES) {
        return Err(format!(
            "cell byte slice length {} is not a multiple of {CELL_BYTES}",
            bytes.len()
        ));
    }
    Ok(bytes
        .as_chunks::<CELL_BYTES>()
        .0
        .iter()
        .map(MomentCell::from_bytes)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_roundtrip_preserves_moments() {
        let pts = [
            Point::new(0.1, 2.0, 1.0),
            Point::new(0.2, 2.5, 3.0),
            Point::new(0.35, 2.2, 0.5),
        ];
        let mut m = Moments::default();
        for p in pts {
            m.push(p);
        }
        let cell = MomentCell::from_bytes(&MomentCell::pack(m, 0.0, 0.4, 2.0, 0.5).to_bytes());
        let n = cell.unpack(0.0, 0.4, 2.0, 0.5);
        for (got, want) in [
            (n.mass, m.mass),
            (n.x, m.x),
            (n.y, m.y),
            (n.m2_xx, m.m2_xx),
            (n.m2_xy, m.m2_xy),
            (n.m2_yy, m.m2_yy),
        ] {
            assert!((got - want).abs() <= 1e-6 * want.abs().max(1.0), "{got} vs {want}");
        }
    }
}
