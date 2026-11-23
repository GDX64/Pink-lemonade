//! Compact moment cells for dense 2D moment planes (e.g. the moment ripmap).
//!
//! A [`MomentCell`] stores only mass, mean, and centered covariance (6 × `f32` = 24 bytes),
//! normalized against a *rectangle known to both builder and client* (an x-block's
//! `[min_x, max_x]` and one y-bin's `[bin_lo, bin_lo + bin_h]`). The rectangle itself is never
//! stored: it is derived from the enclosing x-node's exact `f64` bounds and the bin index.

use pink_lemonade_progressive_core::Point;

use crate::node::Node;

pub const CELL_BYTES: usize = 6 * 4;
/// Byte size of [`QuantizedMoments`] (5 × 16-bit fixed point + `f32` mass).
pub const QUANTIZED_MOMENTS_BYTES: usize = 14;
/// Byte size of [`SparseCell`] (`u16` bin/row index + 14-byte [`QuantizedMoments`]).
pub const SPARSE_CELL_BYTES: usize = 16;

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

/// 14-byte quantized moment summary normalized against a known rectangle `[x_lo, x_lo + dx] × [y_lo, y_lo + dy]`.
///
/// Uses Popoviciu's `[0, 0.25]` bound on normalized variances (`[-0.25, 0.25]` for covariance):
/// - `q_tx, q_ty ∈ [0, 1]` quantized to `u16` (`0..=65535`)
/// - `q_var_x, q_var_y ∈ [0, 0.25]` quantized to `u16` (`0..=65535`)
/// - `q_cov_xy ∈ [-0.25, 0.25]` quantized to `i16` (`-32767..=32767`)
/// - `mass: f32`
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct QuantizedMoments {
    pub q_tx: u16,
    pub q_ty: u16,
    pub q_var_x: u16,
    pub q_cov_xy: i16,
    pub q_var_y: u16,
    pub mass: f32,
}

impl QuantizedMoments {
    /// Pack normalized moments `m` within `[x_lo, x_lo + dx] × [y_lo, y_lo + dy]`.
    pub fn pack(m: Moments, x_lo: f64, dx: f64, y_lo: f64, dy: f64) -> Self {
        if m.mass == 0.0 {
            return Self::default();
        }
        let inv_mass = 1.0 / m.mass;
        let tx = if dx > 0.0 {
            ((m.x - x_lo) / dx).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let ty = if dy > 0.0 {
            ((m.y - y_lo) / dy).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let nvx = if dx > 0.0 {
            (m.m2_xx * inv_mass / (dx * dx)).clamp(0.0, 0.25)
        } else {
            0.0
        };
        let nvy = if dy > 0.0 {
            (m.m2_yy * inv_mass / (dy * dy)).clamp(0.0, 0.25)
        } else {
            0.0
        };
        let ncxy = if dx > 0.0 && dy > 0.0 {
            (m.m2_xy * inv_mass / (dx * dy)).clamp(-0.25, 0.25)
        } else {
            0.0
        };
        Self {
            q_tx: (tx * 65535.0).round() as u16,
            q_ty: (ty * 65535.0).round() as u16,
            q_var_x: (nvx * 4.0 * 65535.0).round() as u16,
            q_cov_xy: (ncxy * 4.0 * 32767.0).round() as i16,
            q_var_y: (nvy * 4.0 * 65535.0).round() as u16,
            mass: m.mass as f32,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.mass <= 0.0
    }

    /// Reconstruct a renderable [`Node`] with PSD-clamped covariance over `[x_lo, x_lo + dx] × [y_lo, y_lo + dy]`.
    pub fn unpack(self, x_lo: f64, dx: f64, y_lo: f64, dy: f64) -> Node {
        let mass = self.mass as f64;
        let tx = self.q_tx as f64 / 65535.0;
        let ty = self.q_ty as f64 / 65535.0;
        let nvx = self.q_var_x as f64 / (4.0 * 65535.0);
        let nvy = self.q_var_y as f64 / (4.0 * 65535.0);
        let max_cov = (nvx * nvy).sqrt();
        let ncxy = (self.q_cov_xy as f64 / (4.0 * 32767.0)).clamp(-max_cov, max_cov);

        let (x_hi, y_hi) = (x_lo + dx, y_lo + dy);
        Node {
            mass,
            x: x_lo + tx * dx,
            y: y_lo + ty * dy,
            m2_xx: nvx * mass * dx * dx,
            m2_xy: ncxy * mass * dx * dy,
            m2_yy: nvy * mass * dy * dy,
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

    pub fn to_bytes(self) -> [u8; QUANTIZED_MOMENTS_BYTES] {
        let mut buf = [0u8; QUANTIZED_MOMENTS_BYTES];
        buf[0..2].copy_from_slice(&self.q_tx.to_le_bytes());
        buf[2..4].copy_from_slice(&self.q_ty.to_le_bytes());
        buf[4..6].copy_from_slice(&self.q_var_x.to_le_bytes());
        buf[6..8].copy_from_slice(&self.q_cov_xy.to_le_bytes());
        buf[8..10].copy_from_slice(&self.q_var_y.to_le_bytes());
        buf[10..14].copy_from_slice(&self.mass.to_le_bytes());
        buf
    }

    pub fn from_bytes(buf: &[u8; QUANTIZED_MOMENTS_BYTES]) -> Self {
        Self {
            q_tx: u16::from_le_bytes(buf[0..2].try_into().unwrap()),
            q_ty: u16::from_le_bytes(buf[2..4].try_into().unwrap()),
            q_var_x: u16::from_le_bytes(buf[4..6].try_into().unwrap()),
            q_cov_xy: i16::from_le_bytes(buf[6..8].try_into().unwrap()),
            q_var_y: u16::from_le_bytes(buf[8..10].try_into().unwrap()),
            mass: f32::from_le_bytes(buf[10..14].try_into().unwrap()),
        }
    }
}

/// 16-byte sparse quantized moment cell: `u16` bin/row index + 14-byte [`QuantizedMoments`].
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct SparseCell {
    pub bin: u16,
    pub moments: QuantizedMoments,
}

impl SparseCell {
    pub fn pack(bin: u16, m: Moments, x_lo: f64, dx: f64, y_lo: f64, dy: f64) -> Self {
        Self {
            bin,
            moments: QuantizedMoments::pack(m, x_lo, dx, y_lo, dy),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.moments.is_empty()
    }

    pub fn mass(&self) -> f32 {
        self.moments.mass
    }

    pub fn unpack(self, x_lo: f64, dx: f64, y_lo: f64, dy: f64) -> Node {
        self.moments.unpack(x_lo, dx, y_lo, dy)
    }

    pub fn to_bytes(self) -> [u8; SPARSE_CELL_BYTES] {
        let mut buf = [0u8; SPARSE_CELL_BYTES];
        buf[0..2].copy_from_slice(&self.bin.to_le_bytes());
        buf[2..16].copy_from_slice(&self.moments.to_bytes());
        buf
    }

    pub fn from_bytes(buf: &[u8; SPARSE_CELL_BYTES]) -> Self {
        Self {
            bin: u16::from_le_bytes(buf[0..2].try_into().unwrap()),
            moments: QuantizedMoments::from_bytes(buf[2..16].try_into().unwrap()),
        }
    }
}

pub fn decode_sparse_cells(bytes: &[u8]) -> Result<Vec<SparseCell>, String> {
    if !bytes.len().is_multiple_of(SPARSE_CELL_BYTES) {
        return Err(format!(
            "sparse cell slice length {} is not a multiple of {SPARSE_CELL_BYTES}",
            bytes.len()
        ));
    }
    Ok(bytes
        .as_chunks::<SPARSE_CELL_BYTES>()
        .0
        .iter()
        .map(SparseCell::from_bytes)
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

    #[test]
    fn sparse_cell_roundtrip_preserves_moments() {
        let pts = [
            Point::new(0.1, 2.0, 1.0),
            Point::new(0.2, 2.5, 3.0),
            Point::new(0.35, 2.2, 0.5),
        ];
        let mut m = Moments::default();
        for p in pts {
            m.push(p);
        }
        let cell = SparseCell::from_bytes(&SparseCell::pack(7, m, 0.0, 0.4, 2.0, 0.5).to_bytes());
        assert_eq!(cell.bin, 7);
        let n = cell.unpack(0.0, 0.4, 2.0, 0.5);
        for (got, want) in [
            (n.mass, m.mass),
            (n.x, m.x),
            (n.y, m.y),
            (n.m2_xx, m.m2_xx),
            (n.m2_xy, m.m2_xy),
            (n.m2_yy, m.m2_yy),
        ] {
            assert!(
                (got - want).abs() <= 1e-4 * want.abs().max(1.0),
                "{got} vs {want}"
            );
        }
    }
}
