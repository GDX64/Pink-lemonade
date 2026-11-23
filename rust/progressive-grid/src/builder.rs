//! Offline builder: column pyramid plus sparse moment planes at several aspect ratios.
//!
//! Points arrive x-sorted, so every column at every level is a contiguous run of
//! `points.bin`. For each aspect `d = i - j`, every level `(i, i - d)` with `i ≤ max_level` and
//! `j ≤ MAX_ROW_LEVEL` is built directly from the points under a per-plane cell budget that is
//! spent on the densest columns first ([`Gate`]); a plane where no column qualifies is not stored.

use pink_lemonade_progressive_common::{Moments, Node, PointCollector, SparseCell};
use pink_lemonade_progressive_core::{BuildError, DatasetBuilder, ObjectWriter, Point, StoreError};

use crate::SparseGridStrategy;
use crate::format::{
    Bounds, CELLS_KEY, COLUMNS_KEY, GRID_CELL_BYTES, LevelInfo, MANIFEST_KEY, MAX_LEVEL,
    MAX_ROW_LEVEL, Manifest, POINTS_KEY, PlaneInfo, index_at, quantize, write_cell, write_record,
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

    // Planes, aspect by aspect: every level `(i, i - d)` is computed straight from the points,
    // keeping only the columns that pass the per-column density gate.
    let qy: Vec<u32> = points.iter().map(|p| quantize(p.y, bounds.y0, bounds.h)).collect();
    let gate = Gate {
        budget: n as u64 / config.min_points_per_cell.max(1) as u64,
        min_column_points_per_cell: config.min_column_points_per_cell.max(1) as u64,
    };
    let mut planes: Vec<Vec<BuiltPlane>> = (0..=max_level).map(|_| Vec::new()).collect();
    let mut sink = writer.create_object(CELLS_KEY)?;
    let mut written_cells = 0u64;
    let mut scratch = RowScratch::default();
    let mut buf = Vec::new();
    for d in config.aspects() {
        for i in d.max(0)..=max_level as i64 {
            let j = i - d;
            if j > MAX_ROW_LEVEL as i64 {
                break;
            }
            let (i, j) = (i as u32, j as u32);
            let cells = gated_cells(points, &qy, &starts[i as usize], j, gate, &mut scratch);
            if cells.is_empty() {
                continue;
            }
            buf.clear();
            let mut col_starts = vec![0u32; (1usize << i) + 1];
            for cell in &cells {
                col_starts[cell.col as usize + 1] += 1;
                let (x_lo, dx, y_lo, dy) = bounds.cell_rect(i, cell.col as u64, j, cell.row as u64);
                write_cell(SparseCell::pack(cell.row as u16, cell.m, x_lo, dx, y_lo, dy), &mut buf);
            }
            for c in 1..col_starts.len() {
                col_starts[c] += col_starts[c - 1];
            }
            sink.write_all(&buf).map_err(StoreError::from)?;
            planes[i as usize].push(BuiltPlane {
                info: PlaneInfo {
                    j,
                    offset: written_cells * GRID_CELL_BYTES as u64,
                    cells: cells.len() as u64,
                    id: 0,
                },
                col_starts,
            });
            written_cells += cells.len() as u64;
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

/// Which columns a plane stores. A column of `p` points qualifies when it fills at most
/// `p / min_column_points_per_cell` rows; qualifying columns are then admitted densest first
/// (most points per cell) while the plane's total stays within `budget` cells.
///
/// On evenly dense data this keeps whole planes exactly when the old global rule did (average
/// ≥ `N / budget` points per cell); on uneven data the dense columns still get fine rows.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Gate {
    pub(crate) budget: u64,
    pub(crate) min_column_points_per_cell: u64,
}

/// Row bookkeeping reused across columns: `stamp[row] == generation` marks rows already seen in
/// the current column, `slot[row]` is their index in the column's accumulators.
#[derive(Default)]
struct RowScratch {
    stamp: Vec<u32>,
    slot: Vec<u32>,
    generation: u32,
}

/// Non-empty cells of plane `(i, j)` column-major, for the columns admitted by `gate`.
///
/// Two passes: count each column's non-empty rows (giving up once it cannot qualify), choose the
/// densest qualifying columns within the budget, then accumulate moments for those columns only.
fn gated_cells(points: &[Point], qy: &[u32], starts: &[u32], j: u32, gate: Gate, scratch: &mut RowScratch) -> Vec<GridCell> {
    let rows = 1usize << j;
    scratch.stamp.clear();
    scratch.stamp.resize(rows, 0);
    scratch.slot.resize(rows, 0);
    scratch.generation = 0;

    // Pass 1: qualifying columns and their non-empty row counts.
    let mut candidates: Vec<(usize, u64, u64)> = Vec::new(); // (column, points, rows)
    for (c, w) in starts.windows(2).enumerate() {
        let (s, e) = (w[0] as usize, w[1] as usize);
        let max_rows = (e - s) as u64 / gate.min_column_points_per_cell;
        if max_rows == 0 {
            continue;
        }
        scratch.generation += 1;
        let generation = scratch.generation;
        let mut count = 0u64;
        for &q in &qy[s..e] {
            let row = index_at(q, j) as usize;
            if scratch.stamp[row] != generation {
                scratch.stamp[row] = generation;
                count += 1;
                if count > max_rows {
                    break;
                }
            }
        }
        if count <= max_rows {
            candidates.push((c, (e - s) as u64, count));
        }
    }
    // Densest first (points per cell, compared exactly), ties by column.
    candidates.sort_unstable_by(|a, b| (b.1 * a.2).cmp(&(a.1 * b.2)).then(a.0.cmp(&b.0)));
    let mut admitted = vec![false; starts.len().saturating_sub(1)];
    let mut used = 0u64;
    for &(c, _, count) in &candidates {
        if used + count > gate.budget {
            break;
        }
        used += count;
        admitted[c] = true;
    }

    // Pass 2: moments of the admitted columns, column-major.
    let mut out = Vec::with_capacity(used as usize);
    let mut seen: Vec<u32> = Vec::new();
    let mut accs: Vec<Moments> = Vec::new();
    for (c, w) in starts.windows(2).enumerate() {
        if !admitted[c] {
            continue;
        }
        let (s, e) = (w[0] as usize, w[1] as usize);
        scratch.generation += 1;
        let generation = scratch.generation;
        seen.clear();
        accs.clear();
        for k in s..e {
            let row = index_at(qy[k], j) as usize;
            if scratch.stamp[row] != generation {
                scratch.stamp[row] = generation;
                scratch.slot[row] = seen.len() as u32;
                seen.push(row as u32);
                accs.push(Moments::default());
            }
            accs[scratch.slot[row] as usize].push(points[k]);
        }
        let mut order: Vec<usize> = (0..seen.len()).collect();
        order.sort_unstable_by_key(|&t| seen[t]);
        out.extend(order.into_iter().map(|t| GridCell {
            col: c as u32,
            row: seen[t],
            m: accs[t],
        }));
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

    /// Under a tight budget the dense band is admitted first and the spread column is left out;
    /// with room for both, both are stored; stored cells keep the column's exact mass.
    #[test]
    fn budget_goes_to_the_densest_columns() {
        let mut points: Vec<Point> = (0..4_000)
            .map(|k| Point::new(k as f64 / 8_000.0, 0.5 + (k % 7) as f64 * 1e-3, 1.0))
            .collect();
        points.extend(
            (0..4_000).map(|k| Point::new(0.5 + k as f64 / 8_000.0, (k * 7_919 % 4_000) as f64 / 4_000.0, 1.0)),
        );
        let qy: Vec<u32> = points.iter().map(|p| quantize(p.y, 0.0, 1.0)).collect();
        let starts = [0u32, 4_000, 8_000];
        let mut scratch = RowScratch::default();
        // 2^8 rows: the band fills ~2 rows, the spread column ~256 rows (both qualify at ≥ 4
        // points per cell). A 100-cell budget only fits the band.
        let tight = Gate { budget: 100, min_column_points_per_cell: 4 };
        let cells = gated_cells(&points, &qy, &starts, 8, tight, &mut scratch);
        assert!(!cells.is_empty());
        assert!(cells.iter().all(|c| c.col == 0), "the spread column must be left out");
        assert!(cells.windows(2).all(|w| w[0].row < w[1].row), "rows sorted");
        let mass: f64 = cells.iter().map(|c| c.m.mass).sum();
        assert!((mass - 4_000.0).abs() < 1e-9);
        let roomy = Gate { budget: 1_000, min_column_points_per_cell: 4 };
        let both = gated_cells(&points, &qy, &starts, 8, roomy, &mut scratch);
        assert!(both.iter().any(|c| c.col == 1));
        // 2^12 rows: the spread column fills ~4,000 rows (< 4 points per cell) and never qualifies.
        let fine = gated_cells(&points, &qy, &starts, 12, roomy, &mut scratch);
        assert!(fine.iter().all(|c| c.col == 0));
    }
}
