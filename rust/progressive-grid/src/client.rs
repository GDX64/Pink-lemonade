//! Online client: direct-addressed column records, then one round of cell and raw reads.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use pink_lemonade_progressive_common::{
    DEFAULT_MANIFEST_READ_BYTES, Decision, FrontierItem, LruPageCache, Node, POINT_BYTES,
    assemble_frame, coalesce_runs, coalesce_spans, decode_points, join2, offscreen_y, push_group,
    read_small_object, standard_decision, try_join_all,
};
use pink_lemonade_progressive_core::{
    ByteRangeStore, ClientError, Frame, Point, ProgressiveClient, ViewQuery, async_trait,
};

use crate::format::{
    CELLS_KEY, COLUMNS_KEY, GRID_CELL_BYTES, MANIFEST_KEY, Manifest, POINTS_KEY, read_cell,
    record_cell_start, record_node, record_point_start, records_at, slot,
};

/// Widest on-screen column (in merge-threshold units) the level choice starts from.
const MAX_COL_WIDTH: f64 = 0.9;
/// Column records are fetched and cached in pages of this many records.
const PAGE_RECORDS: u64 = 8;
/// Bridge gaps of up to 16 KiB; cap merged reads at 1 MiB.
const MAX_GAP_BYTES: u64 = 16 * 1024;
const MAX_RUN_BYTES: u64 = 1024 * 1024;
/// Bridge gaps of up to 256 raw points (6 KiB); cap merged raw reads at 32K points.
const MAX_GAP_POINTS: u64 = 256;
const MAX_RUN_POINTS: u64 = 32 * 1024;
/// Cache namespaces: column pages use the level; a column's cells in a plane use
/// `CELL_SPACE + plane id`.
const CELL_SPACE: u16 = 1024;
/// Guards the record-fetch loop (each pass only runs when rounding put a needed record
/// outside the prefetched pages).
const MAX_RECORD_PASSES: usize = 32;

/// Column record pages fetched during one query, keyed by `(level, page)`.
type Pages = HashMap<(u32, u64), Arc<[u8]>>;

pub struct SparseGridClient {
    store: Arc<dyn ByteRangeStore>,
    manifest: Manifest,
    cache: LruPageCache,
}

/// A decoded column: exact node, point run, and its cell run in every plane of its level.
#[derive(Debug, Clone)]
struct Column {
    level: u32,
    c: u64,
    node: Node,
    points: (u64, u64),
    cells: Vec<(u64, u64)>,
}

impl Column {
    fn point_count(&self) -> u64 {
        self.points.1 - self.points.0
    }
}

/// Exact decomposition of `[min_x, max_x]` into columns, in x order.
enum Piece {
    /// Fully inside: may be summarized.
    Inside(Column),
    /// Straddles the viewport at the deepest level: raw points, clipped.
    Edge(Column),
}

/// How one piece is rendered.
enum Render {
    Node(Node),
    Cells { col: Column, plane: usize },
    Raw { start: u64, count: u64 },
}

impl SparseGridClient {
    pub async fn open(store: Arc<dyn ByteRangeStore>, memory_budget_bytes: usize) -> Result<Self, ClientError> {
        let raw = read_small_object(store.as_ref(), MANIFEST_KEY, DEFAULT_MANIFEST_READ_BYTES).await?;
        let manifest = Manifest::decode(&raw)?;
        let cache = LruPageCache::new(memory_budget_bytes.saturating_sub(manifest.approx_memory_bytes()));
        Ok(Self { store, manifest, cache })
    }

    fn max_level(&self) -> u32 {
        self.manifest.levels.len() as u32 - 1
    }

    /// Page ids holding records `[first, last]` of `level` (clamped to the level).
    fn pages_for(level: u32, first: u64, last: u64, out: &mut Vec<(u32, u64)>) {
        let last = last.min(records_at(level) - 1);
        if first > last {
            return;
        }
        for page in first / PAGE_RECORDS..=last / PAGE_RECORDS {
            out.push((level, page));
        }
    }

    /// Make sure every wanted page is in `pages` (cache first, then coalesced range reads).
    async fn fetch_pages(&mut self, pages: &mut Pages, mut wanted: Vec<(u32, u64)>) -> Result<(), ClientError> {
        wanted.sort_unstable();
        wanted.dedup();
        let mut missing: BTreeMap<u32, Vec<u64>> = BTreeMap::new();
        for (level, page) in wanted {
            if pages.contains_key(&(level, page)) {
                continue;
            }
            match self.cache.get_blob(level as u16, page) {
                Some(bytes) => {
                    pages.insert((level, page), bytes);
                }
                None => missing.entry(level).or_default().push(page),
            }
        }
        if missing.is_empty() {
            return Ok(());
        }

        let mut futures = Vec::new();
        for (level, ids) in missing {
            let info = &self.manifest.levels[level as usize];
            let rb = info.record_bytes() as u64;
            let page_bytes = PAGE_RECORDS * rb;
            let total = records_at(level);
            for (start, count) in coalesce_runs(ids, MAX_GAP_BYTES / page_bytes, (MAX_RUN_BYTES / page_bytes).max(1) as usize) {
                let first = start * PAGE_RECORDS;
                let end = ((start + count as u64) * PAGE_RECORDS).min(total);
                let store = Arc::clone(&self.store);
                let offset = info.offset + first * rb;
                let len = ((end - first) * rb) as usize;
                futures.push(async move {
                    let bytes = store.read_range(COLUMNS_KEY, offset, len).await?;
                    Ok::<_, ClientError>((level, start, rb as usize, bytes))
                });
            }
        }
        for (level, start, rb, bytes) in try_join_all(futures).await? {
            let page_bytes = PAGE_RECORDS as usize * rb;
            if !bytes.len().is_multiple_of(rb) {
                return Err(ClientError::CorruptData("short column record read".to_owned()));
            }
            for (k, chunk) in bytes.chunks(page_bytes).enumerate() {
                let page: Arc<[u8]> = Arc::from(chunk);
                self.cache.insert_blob(level as u16, start + k as u64, Arc::clone(&page));
                pages.insert((level, start + k as u64), page);
            }
        }
        Ok(())
    }

    /// Record `r` of `level`, or note its page as missing.
    fn record<'a>(&self, pages: &'a Pages, level: u32, r: u64, missing: &mut Vec<(u32, u64)>) -> Option<&'a [u8]> {
        let rb = self.manifest.levels[level as usize].record_bytes();
        let page = r / PAGE_RECORDS;
        let Some(bytes) = pages.get(&(level, page)) else {
            missing.push((level, page));
            return None;
        };
        let off = (r % PAGE_RECORDS) as usize * rb;
        bytes.get(off..off + rb)
    }

    fn column(&self, pages: &Pages, level: u32, c: u64, missing: &mut Vec<(u32, u64)>) -> Option<Column> {
        let this = self.record(pages, level, c, missing);
        let next = self.record(pages, level, c + 1, missing);
        let (this, next) = (this?, next?);
        let planes = self.manifest.levels[level as usize].planes.len();
        Some(Column {
            level,
            c,
            node: record_node(this),
            points: (record_point_start(this), record_point_start(next)),
            cells: (0..planes)
                .map(|p| (record_cell_start(this, p), record_cell_start(next, p)))
                .collect(),
        })
    }

    /// Decompose `[a, b]` starting from the columns of `level`; `Err` lists missing pages.
    fn decompose(&self, pages: &Pages, level: u32, a: f64, b: f64) -> Result<Vec<Piece>, Vec<(u32, u64)>> {
        let bounds = self.manifest.bounds;
        let mut out = Vec::new();
        let mut missing = Vec::new();
        for c in bounds.col(a, level)..=bounds.col(b, level) {
            self.walk(pages, level, c, a, b, &mut out, &mut missing);
        }
        if missing.is_empty() { Ok(out) } else { Err(missing) }
    }

    #[allow(clippy::too_many_arguments)]
    fn walk(
        &self,
        pages: &Pages,
        level: u32,
        c: u64,
        a: f64,
        b: f64,
        out: &mut Vec<Piece>,
        missing: &mut Vec<(u32, u64)>,
    ) {
        let Some(col) = self.column(pages, level, c, missing) else {
            return;
        };
        let n = col.node;
        if n.is_empty() || n.max_x < a || n.min_x > b {
            return;
        }
        if a <= n.min_x && n.max_x <= b {
            out.push(Piece::Inside(col));
        } else if level == self.max_level() || col.point_count() <= self.manifest.col_leaf as u64 {
            out.push(Piece::Edge(col));
        } else {
            self.walk(pages, level + 1, 2 * c, a, b, out, missing);
            self.walk(pages, level + 1, 2 * c + 1, a, b, out, missing);
        }
    }

    /// Decompose from `level`, fetching any record the prefetch missed.
    async fn pieces(&mut self, pages: &mut Pages, level: u32, a: f64, b: f64) -> Result<Vec<Piece>, ClientError> {
        for _ in 0..MAX_RECORD_PASSES {
            match self.decompose(pages, level, a, b) {
                Ok(pieces) => return Ok(pieces),
                Err(missing) => self.fetch_pages(pages, missing).await?,
            }
        }
        Err(ClientError::CorruptData("column records did not resolve".to_owned()))
    }

    /// Coarsest plane at `level` whose cells fit the merge threshold on screen.
    fn pick_plane(&self, level: u32, x_factor: f64, y_factor: f64, threshold: f64) -> Option<usize> {
        let w = slot(self.manifest.bounds.w, level) * x_factor;
        if w >= threshold {
            return None;
        }
        let h_max = (threshold * threshold - w * w).sqrt();
        self.manifest.levels[level as usize]
            .planes
            .iter()
            .position(|p| slot(self.manifest.bounds.h, p.j) * y_factor < h_max)
    }

    /// Render plan for `pieces` and its estimated wire bytes.
    fn plan(&self, pieces: Vec<Piece>, query: &ViewQuery, x_factor: f64, y_factor: f64) -> (Vec<Render>, u64) {
        let threshold = query.merge_threshold.max(1e-12);
        let mut bytes = 0u64;
        let renders = pieces
            .into_iter()
            .map(|piece| {
                let col = match piece {
                    Piece::Edge(col) => {
                        bytes += col.point_count() * POINT_BYTES as u64;
                        return Render::Raw {
                            start: col.points.0,
                            count: col.point_count(),
                        };
                    }
                    Piece::Inside(col) => col,
                };
                if standard_decision(&col.node, query, x_factor, y_factor) == Decision::Resolve {
                    return Render::Node(col.node);
                }
                // The per-column gate may have left this column out of the plane (empty run): a
                // finer plane would fail the same gate, so the column goes raw.
                match self
                    .pick_plane(col.level, x_factor, y_factor, threshold)
                    .filter(|&plane| col.cells[plane].1 > col.cells[plane].0)
                {
                    Some(plane) => {
                        let (s, e) = col.cells[plane];
                        bytes += (e - s) * GRID_CELL_BYTES as u64;
                        Render::Cells { col, plane }
                    }
                    None => {
                        bytes += col.point_count() * POINT_BYTES as u64;
                        Render::Raw {
                            start: col.points.0,
                            count: col.point_count(),
                        }
                    }
                }
            })
            .collect();
        (renders, bytes)
    }
}

/// Emit one column's cells as kernels: cells far off-screen in y (explicit `y_bounds`) collapse
/// into one kernel above and one below, and on-screen cells merge greedily in row order while the
/// merged kernel still fits the threshold (`push_group`). Moments stay exact either way.
#[allow(clippy::too_many_arguments)]
fn emit_column_cells(
    bytes: &[u8],
    level: u32,
    c: u64,
    j: u32,
    bounds: &crate::format::Bounds,
    query: &ViewQuery,
    (x_factor, y_factor): (f64, f64),
    gpu: &mut Vec<f32>,
) {
    let threshold = query.merge_threshold.max(1e-12);
    let (mut below, mut above) = (Node::default(), Node::default());
    let mut groups: Vec<Node> = Vec::new();
    for chunk in bytes.as_chunks::<GRID_CELL_BYTES>().0 {
        let cell = read_cell(chunk);
        if cell.is_empty() {
            continue;
        }
        let (x_lo, dx, y_lo, dy) = bounds.cell_rect(level, c, j, cell.bin as u64);
        let node = cell.unpack(x_lo, dx, y_lo, dy);
        // `offscreen_y` implies the cell lies entirely above or entirely below the screen.
        if let Some([lo, _]) = query.y_bounds
            && offscreen_y(&node, query, y_factor)
        {
            let side = if node.max_y < lo { &mut below } else { &mut above };
            let merged = Node::combined(*side, node);
            if offscreen_y(&merged, query, y_factor) {
                *side = merged;
                continue;
            }
        }
        push_group(node, x_factor, y_factor, threshold, &mut groups);
    }
    for n in [below, above].iter().chain(groups.iter()) {
        if !n.is_empty() {
            n.emit_gpu(x_factor, y_factor, gpu);
        }
    }
}

/// Exact `[min_y, max_y]` of the inside pieces (`lower`) and an upper bound that also counts
/// the edge columns' whole y-extent.
fn y_bounds_of(pieces: &[Piece]) -> (Option<[f64; 2]>, Option<[f64; 2]>) {
    let fold = |acc: Option<[f64; 2]>, n: &Node| -> Option<[f64; 2]> {
        Some(match acc {
            Some([lo, hi]) => [lo.min(n.min_y), hi.max(n.max_y)],
            None => [n.min_y, n.max_y],
        })
    };
    let (mut lower, mut upper) = (None, None);
    for piece in pieces {
        match piece {
            Piece::Inside(col) => {
                lower = fold(lower, &col.node);
                upper = fold(upper, &col.node);
            }
            Piece::Edge(col) => upper = fold(upper, &col.node),
        }
    }
    (lower, upper)
}

/// Read cell runs `(plane id, column, absolute start, count)` from `cells.bin`.
async fn download_cells(
    store: Arc<dyn ByteRangeStore>,
    runs: Vec<(u32, u64, u64, u64)>,
) -> Result<Vec<((u32, u64), Arc<[u8]>)>, ClientError> {
    if runs.is_empty() {
        return Ok(Vec::new());
    }
    let spans: Vec<(u64, u64)> = runs.iter().map(|&(_, _, s, n)| (s, n)).collect();
    let cell = GRID_CELL_BYTES as u64;
    let reads = coalesce_spans(spans, MAX_GAP_BYTES / cell, MAX_RUN_BYTES / cell);
    let futures: Vec<_> = reads
        .iter()
        .map(|&(start, end)| {
            let store = Arc::clone(&store);
            async move {
                let bytes = store
                    .read_range(CELLS_KEY, start * cell, ((end - start) * cell) as usize)
                    .await?;
                Ok::<_, ClientError>((start, bytes))
            }
        })
        .collect();
    let fetched = try_join_all(futures).await?;
    runs.into_iter()
        .map(|(id, c, s, n)| {
            let (start, bytes) = fetched
                .iter()
                .find(|(start, bytes)| s >= *start && s + n <= start + bytes.len() as u64 / cell)
                .ok_or_else(|| ClientError::CorruptData("short cell read".to_owned()))?;
            let lo = ((s - start) * cell) as usize;
            Ok(((id, c), Arc::from(&bytes[lo..lo + (n * cell) as usize])))
        })
        .collect()
}

/// Read raw point runs `(start, count)` from `points.bin`.
async fn download_points(
    store: Arc<dyn ByteRangeStore>,
    runs: Vec<(u64, u64)>,
) -> Result<Vec<(u64, Arc<[Point]>)>, ClientError> {
    if runs.is_empty() {
        return Ok(Vec::new());
    }
    let reads = coalesce_spans(runs.clone(), MAX_GAP_POINTS, MAX_RUN_POINTS);
    let futures: Vec<_> = reads
        .iter()
        .map(|&(start, end)| {
            let store = Arc::clone(&store);
            async move {
                let bytes = store
                    .read_range(POINTS_KEY, start * POINT_BYTES as u64, (end - start) as usize * POINT_BYTES)
                    .await?;
                Ok::<_, ClientError>((start, bytes))
            }
        })
        .collect();
    let fetched = try_join_all(futures).await?;
    runs.into_iter()
        .map(|(s, n)| {
            let (start, bytes) = fetched
                .iter()
                .find(|(start, bytes)| s >= *start && s + n <= start + (bytes.len() / POINT_BYTES) as u64)
                .ok_or_else(|| ClientError::CorruptData("short point read".to_owned()))?;
            let lo = (s - start) as usize * POINT_BYTES;
            let points = decode_points(&bytes[lo..lo + n as usize * POINT_BYTES]).map_err(ClientError::CorruptData)?;
            Ok((s, Arc::from(points)))
        })
        .collect()
}

#[async_trait]
impl ProgressiveClient for SparseGridClient {
    async fn query(&mut self, query: ViewQuery) -> Result<Frame, ClientError> {
        if self.manifest.levels.is_empty() || query.min_x > query.max_x {
            return Ok(Frame::empty());
        }
        let (a, b) = (query.min_x, query.max_x);
        let bounds = self.manifest.bounds;
        let max_level = self.max_level();
        let x_factor = query.factors([0.0, 1.0]).0;

        // Candidate x-levels: the coarsest whose columns are at most 0.9σ wide, and the next.
        let base = (0..=max_level)
            .find(|&i| slot(bounds.w, i) * x_factor <= MAX_COL_WIDTH * query.merge_threshold)
            .unwrap_or(max_level);
        let candidates: Vec<u32> = (base..=(base + 1).min(max_level)).collect();

        // Round 1: column records of the visible range at both candidates, plus the edge chains
        // of every finer level (the only columns a straddling column can split into).
        let mut wanted = Vec::new();
        for &level in &candidates {
            Self::pages_for(level, bounds.col(a, level), bounds.col(b, level) + 1, &mut wanted);
        }
        let deepest_candidate = *candidates.last().unwrap();
        for level in deepest_candidate + 1..=max_level {
            for (edge, needed) in [(a, a > bounds.x0), (b, b < bounds.x0 + bounds.w)] {
                if needed {
                    let c = bounds.col(edge, level);
                    Self::pages_for(level, c & !1, (c | 1) + 1, &mut wanted);
                }
            }
        }
        let mut pages = Pages::new();
        self.fetch_pages(&mut pages, wanted).await?;

        let mut decompositions = Vec::with_capacity(candidates.len());
        for &level in &candidates {
            decompositions.push(self.pieces(&mut pages, level, a, b).await?);
        }
        drop(pages);

        // The exact y-range of the inside columns is a lower bound of the visible range, so the
        // y-factor estimated from it only over-refines.
        let (lower, upper) = y_bounds_of(&decompositions[0]);
        let Some(estimate) = lower.or(upper) else {
            return Ok(Frame::empty());
        };
        let y_factor_estimate = query.factors(query.effective_y_bounds(estimate)).1;
        let (renders, _) = decompositions
            .into_iter()
            .map(|pieces| self.plan(pieces, &query, x_factor, y_factor_estimate))
            .min_by_key(|(_, bytes)| *bytes)
            .expect("at least one candidate");

        // Round 2: cells and raw points, from cache or one concurrent batch of reads.
        let levels = &self.manifest.levels;
        let mut cells: HashMap<(u32, u64), Arc<[u8]>> = HashMap::new();
        let mut leaves: HashMap<u64, Arc<[Point]>> = HashMap::new();
        let mut missing_cells = Vec::new();
        let mut missing_points = Vec::new();
        for render in &renders {
            match render {
                Render::Cells { col, plane } => {
                    let info = levels[col.level as usize].planes[*plane];
                    let (s, e) = col.cells[*plane];
                    match self.cache.get_blob(CELL_SPACE + info.id as u16, col.c) {
                        Some(bytes) => {
                            cells.insert((info.id, col.c), bytes);
                        }
                        None => missing_cells.push((info.id, col.c, info.absolute(s), e - s)),
                    }
                }
                Render::Raw { start, count } => match self.cache.get_leaf_points(*start) {
                    Some(points) if points.len() as u64 == *count => {
                        leaves.insert(*start, points);
                    }
                    _ => missing_points.push((*start, *count)),
                },
                Render::Node(_) => {}
            }
        }
        let (got_cells, got_points) = join2(
            download_cells(Arc::clone(&self.store), missing_cells),
            download_points(Arc::clone(&self.store), missing_points),
        )
        .await?;
        for ((id, c), bytes) in got_cells {
            self.cache.insert_blob(CELL_SPACE + id as u16, c, Arc::clone(&bytes));
            cells.insert((id, c), bytes);
        }
        for (start, points) in got_points {
            self.cache.insert_leaf_points(start, Arc::clone(&points));
            leaves.insert(start, points);
        }

        // Exact visible y-range: inside columns by their exact nodes, raw runs by clipped points.
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        let mut items = Vec::with_capacity(renders.len());
        for render in renders {
            match render {
                Render::Node(node) => {
                    lo = lo.min(node.min_y);
                    hi = hi.max(node.max_y);
                    items.push(FrontierItem::Resolved(node));
                }
                Render::Cells { col, plane } => {
                    lo = lo.min(col.node.min_y);
                    hi = hi.max(col.node.max_y);
                    items.push(FrontierItem::Plane {
                        level: col.level as usize,
                        index: col.c,
                        node: col.node,
                        k: plane as u8,
                    });
                }
                Render::Raw { start, .. } => {
                    for p in leaves[&start].iter().filter(|p| p.x >= a && p.x <= b) {
                        lo = lo.min(p.y);
                        hi = hi.max(p.y);
                    }
                    items.push(FrontierItem::Leaf(start));
                }
            }
        }
        if !lo.is_finite() {
            return Ok(Frame::empty());
        }
        let y_range = [lo, hi];
        let factors = query.factors(query.effective_y_bounds(y_range));

        let mut emit_cells = |item: &FrontierItem, gpu: &mut Vec<f32>| {
            let FrontierItem::Plane { level, index, k, .. } = *item else {
                return;
            };
            let info = levels[level].planes[k as usize];
            let Some(bytes) = cells.get(&(info.id, index)) else {
                return;
            };
            emit_column_cells(bytes, level as u32, index, info.j, &bounds, &query, factors, gpu);
        };
        Ok(assemble_frame(items, &leaves, &query, factors, y_range, true, &mut emit_cells))
    }

    fn memory_bytes(&self) -> usize {
        self.manifest.approx_memory_bytes() + self.cache.used_bytes()
    }
}
