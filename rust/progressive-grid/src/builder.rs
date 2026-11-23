//! Offline builder: column pyramid plus sparse moment planes at several aspect ratios.
//!
//! Points arrive x-sorted, so every column at every level is a contiguous run of
//! `points.bin`. For each aspect `d = i - j` the builder finds the finest level `(i, j)` whose
//! plane keeps at least `min_points_per_cell` points per non-empty cell on average (aborting a
//! pass as soon as it overflows that budget), then derives every coarser level `(i-1, j-1)` of
//! the same aspect by merging 2×2 cells.

use pink_lemonade_progressive_common::{Moments, MomentCell, Node, PointCollector};
use pink_lemonade_progressive_core::{BuildError, DatasetBuilder, ObjectWriter, Point, StoreError};

use crate::SparseGridStrategy;
use crate::format::{
    Bounds, CELLS_KEY, COLUMNS_KEY, GRID_CELL_BYTES, LevelInfo, MANIFEST_KEY, MAX_LEVEL, Manifest,
    POINTS_KEY, PlaneInfo, index_at, quantize, write_cell, write_record,
};

pub struct SparseGridBuilder {
    pub(crate) config: SparseGridStrategy,
    pub(crate) points: PointCollector,
}

impl DatasetBuilder for SparseGridBuilder {
    fn push_chunk(&mut self, points: &[Point]) -> Result<(), BuildError> {
        self.points.push_chunk(points)
    }

    fn finish(self: Box<Self>, writer: &mut dyn ObjectWriter) -> Result<(), BuildError> {
        let config = self.config;
        let points = self.points.points;
        if points.len() as u64 >= u32::MAX as u64 {
            return Err(BuildError::InvalidInput(format!(
                "sparse grid supports < 2^32 - 1 points (got {})",
                points.len()
            )));
        }
        let mut manifest = Manifest {
            total_points: points.len() as u64,
            col_leaf: config.col_leaf,
            min_points_per_cell: config.min_points_per_cell,
            bounds: Bounds::default(),
            levels: Vec::new(),
        };
        if !points.is_empty() {
            write_points(&points, writer)?;
            build(&points, &config, &mut manifest, writer)?;
        }
        writer.write_object(MANIFEST_KEY, &manifest.encode())?;
        Ok(())
    }
}

/// Deepest x-level: the first whose columns hold at most `col_leaf` points on average.
pub(crate) fn max_level_for(n: u64, col_leaf: u32) -> u32 {
    let mut level = 0;
    while level < MAX_LEVEL && n.div_ceil(1u64 << level) > col_leaf as u64 {
        level += 1;
    }
    level
}

/// Non-empty cell of a plane before packing.
#[derive(Debug, Clone, Copy)]
struct GridCell {
    col: u32,
    row: u32,
    m: Moments,
}

/// A plane ready to be indexed by the column records.
struct BuiltPlane {
    info: PlaneInfo,
    /// First cell of every column, plus the total (`2^i + 1` entries).
    col_starts: Vec<u32>,
}

fn write_points(points: &[Point], writer: &mut dyn ObjectWriter) -> Result<(), BuildError> {
    let mut sink = writer.create_object(POINTS_KEY)?;
    for p in points {
        sink.write_all(&p.x.to_le_bytes()).map_err(StoreError::from)?;
        sink.write_all(&p.y.to_le_bytes()).map_err(StoreError::from)?;
        sink.write_all(&p.weight.to_le_bytes()).map_err(StoreError::from)?;
    }
    sink.flush().map_err(StoreError::from)?;
    Ok(())
}

fn build(
    points: &[Point],
    config: &SparseGridStrategy,
    manifest: &mut Manifest,
    writer: &mut dyn ObjectWriter,
) -> Result<(), BuildError> {
    let n = points.len();
    let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
    for p in points {
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
    }
    let bounds = Bounds {
        x0: points[0].x,
        w: points[n - 1].x - points[0].x,
        y0: min_y,
        h: max_y - min_y,
    };
    manifest.bounds = bounds;
    let max_level = max_level_for(n as u64, config.col_leaf.max(1));

    // Column point starts at every level (from the finest), and exact column nodes.
    let fine_cols = 1usize << max_level;
    let mut counts = vec![0u32; fine_cols];
    for p in points {
        counts[index_at(quantize(p.x, bounds.x0, bounds.w), max_level) as usize] += 1;
    }
    let mut fine_starts = Vec::with_capacity(fine_cols + 1);
    let mut acc = 0u32;
    for c in counts {
        fine_starts.push(acc);
        acc += c;
    }
    fine_starts.push(acc);
    let starts: Vec<Vec<u32>> = (0..=max_level)
        .map(|l| {
            let shift = max_level - l;
            (0..=(1usize << l)).map(|c| fine_starts[c << shift]).collect()
        })
        .collect();
    let mut nodes: Vec<Vec<Node>> = vec![Vec::new(); max_level as usize + 1];
    nodes[max_level as usize] = fine_starts
        .windows(2)
        .map(|w| Node::from_points(&points[w[0] as usize..w[1] as usize]))
        .collect();
    for l in (0..max_level as usize).rev() {
        nodes[l] = nodes[l + 1]
            .chunks(2)
            .map(|pair| Node::combined(pair[0], pair[1]))
            .collect();
    }

    // Planes, aspect by aspect.
    let qy: Vec<u32> = points.iter().map(|p| quantize(p.y, bounds.y0, bounds.h)).collect();
    let limit = n / config.min_points_per_cell.max(1) as usize;
    let mut planes: Vec<Vec<BuiltPlane>> = (0..=max_level).map(|_| Vec::new()).collect();
    let mut sink = writer.create_object(CELLS_KEY)?;
    let mut written_cells = 0u64;
    let mut buf = Vec::new();
    for d in config.aspects() {
        let i_fine = (max_level as i64).min(d + MAX_LEVEL as i64);
        let j_fine = i_fine - d;
        if i_fine < 0 || j_fine < 0 {
            continue;
        }
        let (mut i, mut j) = (i_fine as u32, j_fine as u32);
        let mut cells = loop {
            if let Some(cells) = grid_cells(points, &qy, &starts[i as usize], j, limit) {
                break Some(cells);
            }
            if i == 0 || j == 0 {
                break None;
            }
            i -= 1;
            j -= 1;
        };
        while let Some(level_cells) = cells {
            buf.clear();
            let mut col_starts = vec![0u32; (1usize << i) + 1];
            for cell in &level_cells {
                col_starts[cell.col as usize + 1] += 1;
                let (x_lo, dx, y_lo, dy) = bounds.cell_rect(i, cell.col as u64, j, cell.row as u64);
                write_cell(cell.row, MomentCell::pack(cell.m, x_lo, dx, y_lo, dy), &mut buf);
            }
            for c in 1..col_starts.len() {
                col_starts[c] += col_starts[c - 1];
            }
            sink.write_all(&buf).map_err(StoreError::from)?;
            planes[i as usize].push(BuiltPlane {
                info: PlaneInfo {
                    j,
                    offset: written_cells * GRID_CELL_BYTES as u64,
                    cells: level_cells.len() as u64,
                    id: 0,
                },
                col_starts,
            });
            written_cells += level_cells.len() as u64;
            cells = (i > 0 && j > 0).then(|| coarsen(&level_cells));
            i = i.saturating_sub(1);
            j = j.saturating_sub(1);
        }
    }
    sink.flush().map_err(StoreError::from)?;
    drop(sink);

    // Column records, level-major.
    let mut sink = writer.create_object(COLUMNS_KEY)?;
    let mut offset = 0u64;
    for (l, level_planes) in planes.iter_mut().enumerate() {
        level_planes.sort_by_key(|p| p.info.j);
        let cols = 1usize << l;
        let mut out = Vec::new();
        let mut cell_starts = vec![0u32; level_planes.len()];
        for c in 0..=cols {
            for (s, p) in cell_starts.iter_mut().zip(level_planes.iter()) {
                *s = p.col_starts[c];
            }
            let node = nodes[l].get(c).copied().unwrap_or_default();
            write_record(node, starts[l][c], &cell_starts, &mut out);
        }
        sink.write_all(&out).map_err(StoreError::from)?;
        manifest.levels.push(LevelInfo {
            offset,
            planes: level_planes.iter().map(|p| p.info).collect(),
        });
        offset += out.len() as u64;
    }
    sink.flush().map_err(StoreError::from)?;
    Ok(())
}

/// Non-empty cells of plane `(i, j)` column-major (columns given by their point starts), or
/// `None` as soon as there are more than `limit`.
fn grid_cells(points: &[Point], qy: &[u32], starts: &[u32], j: u32, limit: usize) -> Option<Vec<GridCell>> {
    let mut out = Vec::new();
    let mut buf: Vec<(u32, u32)> = Vec::new();
    for (c, w) in starts.windows(2).enumerate() {
        let (s, e) = (w[0] as usize, w[1] as usize);
        if s == e {
            continue;
        }
        buf.clear();
        buf.extend((s..e).map(|k| (index_at(qy[k], j) as u32, k as u32)));
        buf.sort_unstable();
        let mut k = 0;
        while k < buf.len() {
            let row = buf[k].0;
            let mut m = Moments::default();
            while k < buf.len() && buf[k].0 == row {
                m.push(points[buf[k].1 as usize]);
                k += 1;
            }
            out.push(GridCell { col: c as u32, row, m });
            if out.len() > limit {
                return None;
            }
        }
    }
    Some(out)
}

/// Merge 2×2 cells: plane `(i, j)` → `(i - 1, j - 1)`, keeping column-major order.
fn coarsen(cells: &[GridCell]) -> Vec<GridCell> {
    let mut out = Vec::with_capacity(cells.len() / 2);
    let mut buf: Vec<(u32, Moments)> = Vec::new();
    let mut k = 0;
    while k < cells.len() {
        let col = cells[k].col >> 1;
        buf.clear();
        while k < cells.len() && cells[k].col >> 1 == col {
            buf.push((cells[k].row >> 1, cells[k].m));
            k += 1;
        }
        buf.sort_by_key(|e| e.0);
        let mut t = 0;
        while t < buf.len() {
            let row = buf[t].0;
            let mut m = Moments::default();
            while t < buf.len() && buf[t].0 == row {
                m.merge(buf[t].1);
                t += 1;
            }
            out.push(GridCell { col, row, m });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_level_targets_col_leaf() {
        assert_eq!(max_level_for(1, 256), 0);
        assert_eq!(max_level_for(256, 256), 0);
        assert_eq!(max_level_for(257, 256), 1);
        assert_eq!(max_level_for(10_000_000, 256), 16);
    }

    #[test]
    fn coarsening_matches_direct_build() {
        let points: Vec<Point> = (0..4_000)
            .map(|k| {
                let x = k as f64 * 0.25;
                Point::new(x, (x * 0.11).sin() * 5.0 + (k % 13) as f64 * 0.1, 1.0 + (k % 3) as f64)
            })
            .collect();
        let (x0, w) = (points[0].x, points[3_999].x - points[0].x);
        let qy: Vec<u32> = points.iter().map(|p| quantize(p.y, -6.0, 13.0)).collect();
        let starts = |l: u32| -> Vec<u32> {
            let mut s = vec![0u32; (1 << l) + 1];
            for p in &points {
                s[index_at(quantize(p.x, x0, w), l) as usize + 1] += 1;
            }
            for c in 1..s.len() {
                s[c] += s[c - 1];
            }
            s
        };
        let fine = grid_cells(&points, &qy, &starts(6), 5, usize::MAX).unwrap();
        let direct = grid_cells(&points, &qy, &starts(5), 4, usize::MAX).unwrap();
        let merged = coarsen(&fine);
        assert_eq!(merged.len(), direct.len());
        for (a, b) in merged.iter().zip(&direct) {
            assert_eq!((a.col, a.row), (b.col, b.row));
            assert!((a.m.mass - b.m.mass).abs() < 1e-9);
            assert!((a.m.m2_yy - b.m.m2_yy).abs() < 1e-6 * b.m.m2_yy.max(1.0));
        }
        assert!(grid_cells(&points, &qy, &starts(6), 5, 3).is_none());
    }
}
