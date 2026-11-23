import {
  type ClientCache,
  type Dataset,
  type FrontierItem,
  type KernelGroup,
  type MomentNode,
  type RangeRequest,
  type StoredObject,
  type ViewQuery,
  NODE_BYTES,
  POINT_BYTES,
  assembleGroups,
  coalesceRuns,
  emptyCache,
  mergeNodes,
  offscreenY,
  summarize,
} from "./ripmap-model";
import { coalesceSpans } from "./quadtree-model";

/**
 * In-browser model of `rust/progressive-grid` (`sparse_grid`, `PLPGRID1`).
 *
 * Mirrors the Rust builder (30-bit quantized bbox grid, column pyramid down to `colLeaf`
 * points, one sparse plane per level of every aspect `d = i - j`, whose cell budget is spent on
 * the densest qualifying columns first) and the client (two candidate x-levels, record pages for
 * the visible columns plus the edge chains, exact decomposition, plane chosen per column by
 * exact bytes, one final round of cells + raw edge points, greedy cell merging).
 */

/** Shared 16-byte `SparseCell`: `u16` row + 14-byte quantized moments. */
export const GRID_CELL_BYTES = 16;
export const COLUMNS_KEY = "columns.bin";
export const CELLS_KEY = "cells.bin";
const Q_BITS = 30;
const Q_SCALE = 2 ** Q_BITS;
export const MAX_LEVEL = 24;
export const MAX_ROW_LEVEL = 16;
export const MAX_COL_WIDTH = 0.9;
export const PAGE_RECORDS = 8;
const MAX_GAP_BYTES = 16 * 1024;
const MAX_RUN_BYTES = 1024 * 1024;
const MAX_GAP_POINTS = 256;
const MAX_RUN_POINTS = 32 * 1024;
const HEADER_BYTES = 64;
const LEVEL_INFO_BYTES = 12;
const PLANE_INFO_BYTES = 20;

export interface GridParams {
  colLeaf: number;
  minPointsPerCell: number;
  minAspect: number;
  maxAspect: number;
  aspectStep: number;
  /** A column qualifies for a plane only if its cells average at least this many points. */
  minColumnPointsPerCell: number;
}

/** `SparseGridStrategy::default()`. */
export const GRID_PARAMS: GridParams = {
  colLeaf: 256,
  minPointsPerCell: 16,
  minAspect: -2,
  maxAspect: 12,
  aspectStep: 2,
  minColumnPointsPerCell: 4,
};

export interface Bounds {
  x0: number;
  w: number;
  y0: number;
  h: number;
}

export function quantize(v: number, lo: number, span: number): number {
  if (span <= 0) return 0;
  const s = Math.floor(((v - lo) / span) * Q_SCALE);
  return Math.min(Math.max(s, 0), Q_SCALE - 1);
}

export const indexAt = (q: number, level: number) => q >>> (Q_BITS - level);
export const slot = (span: number, level: number) => span / 2 ** level;
export const slotLo = (lo: number, span: number, idx: number, level: number) => lo + slot(span, level) * idx;
export const colOf = (b: Bounds, x: number, level: number) => indexAt(quantize(x, b.x0, b.w), level);

/** One stored non-empty cell, with its normalization rectangle. */
export interface GridCell {
  col: number;
  row: number;
  count: number;
  mass: number;
  x: number;
  y: number;
  cxx: number;
  cxy: number;
  cyy: number;
  rx0: number;
  rx1: number;
  ry0: number;
  ry1: number;
}

/** One stored plane `(i, j)`. */
export interface GridPlane {
  id: number;
  i: number;
  j: number;
  cells: GridCell[];
  /** First cell of every column of level `i`, plus the total (`2^i + 1` entries). */
  colStarts: Int32Array;
  /** Byte offset of the plane's first cell in `cells.bin`. */
  offset: number;
}

export interface GridModel {
  strategy: "grid";
  /** x-sorted points (= `points.bin`). */
  data: Dataset;
  params: GridParams;
  bounds: Bounds;
  maxLevel: number;
  /** `columns[i][c]`: exact column node (`start`/`end` = its run of `points.bin`). */
  columns: MomentNode[][];
  planes: GridPlane[];
  /** Planes of each x-level, sorted by `j` (the order of their CSC pointers in a record). */
  planesByLevel: GridPlane[][];
  /** Byte offset of each level's records in `columns.bin`, and the record size. */
  levelOffsets: number[];
  recordBytes: number[];
  /** Per aspect: the levels `(i, j)` stored (any column passed the gate). */
  aspects: Array<{ d: number; levels: Array<[number, number]> }>;
  objects: StoredObject[];
  totalBytes: number;
  rawBytes: number;
}

/** First x-level whose columns average ≤ `colLeaf` points (`max_level_for`). */
export function maxLevelFor(n: number, colLeaf: number): number {
  let level = 0;
  while (level < MAX_LEVEL && Math.ceil(n / 2 ** level) > colLeaf) level++;
  return level;
}

class Sums {
  count = 0;
  w = 0;
  wx = 0;
  wy = 0;
  wxx = 0;
  wxy = 0;
  wyy = 0;
  push(x: number, y: number, w: number) {
    this.count++;
    this.w += w;
    this.wx += w * x;
    this.wy += w * y;
    this.wxx += w * x * x;
    this.wxy += w * x * y;
    this.wyy += w * y * y;
  }
  add(o: Sums) {
    this.count += o.count;
    this.w += o.w;
    this.wx += o.wx;
    this.wy += o.wy;
    this.wxx += o.wxx;
    this.wxy += o.wxy;
    this.wyy += o.wyy;
  }
}

interface RawCell {
  col: number;
  row: number;
  s: Sums;
}

/**
 * `gated_cells` from `progressive-grid/src/builder.rs`: non-empty cells of plane `(i, j)`.
 * A column of `p` points qualifies when it fills ≤ `p / minColumnPointsPerCell` rows; qualifying
 * columns are admitted densest first while the plane stays within `budget` cells.
 */
function gatedCells(data: Dataset, qy: Uint32Array, starts: Int32Array, j: number, budget: number, minColumnPointsPerCell: number): RawCell[] {
  const candidates: Array<[number, number, number]> = []; // column, points, rows
  for (let c = 0; c + 1 < starts.length; c++) {
    const s = starts[c]!;
    const e = starts[c + 1]!;
    const maxRows = Math.floor((e - s) / minColumnPointsPerCell);
    if (maxRows === 0) continue;
    const seen = new Set<number>();
    for (let k = s; k < e && seen.size <= maxRows; k++) seen.add(indexAt(qy[k]!, j));
    if (seen.size <= maxRows) candidates.push([c, e - s, seen.size]);
  }
  candidates.sort((a, b) => b[1] * a[2] - a[1] * b[2] || a[0] - b[0]);
  const admitted = new Set<number>();
  let used = 0;
  for (const [c, , count] of candidates) {
    if (used + count > budget) break;
    used += count;
    admitted.add(c);
  }
  const out: RawCell[] = [];
  for (let c = 0; c + 1 < starts.length; c++) {
    if (!admitted.has(c)) continue;
    const rows = new Map<number, Sums>();
    for (let k = starts[c]!; k < starts[c + 1]!; k++) {
      const row = indexAt(qy[k]!, j);
      let acc = rows.get(row);
      if (!acc) rows.set(row, (acc = new Sums()));
      acc.push(data.xs[k]!, data.ys[k]!, data.ws[k]!);
    }
    for (const row of [...rows.keys()].sort((x, y) => x - y)) out.push({ col: c, row, s: rows.get(row)! });
  }
  return out;
}

function toCell(b: Bounds, i: number, j: number, raw: RawCell): GridCell {
  const { s } = raw;
  const m = s.w > 0 ? s.w : 1;
  const x = s.wx / m;
  const y = s.wy / m;
  const rx0 = slotLo(b.x0, b.w, raw.col, i);
  const ry0 = slotLo(b.y0, b.h, raw.row, j);
  return {
    col: raw.col,
    row: raw.row,
    count: s.count,
    mass: s.w,
    x,
    y,
    cxx: Math.max(s.wxx / m - x * x, 0),
    cxy: s.wxy / m - x * y,
    cyy: Math.max(s.wyy / m - y * y, 0),
    rx0,
    rx1: rx0 + slot(b.w, i),
    ry0,
    ry1: ry0 + slot(b.h, j),
  };
}

export function buildGrid(data: Dataset, inParams: GridParams): GridModel {
  const params = { ...inParams, colLeaf: Math.max(inParams.colLeaf, 1), minPointsPerCell: Math.max(inParams.minPointsPerCell, 1) };
  const n = data.xs.length;
  const { xs, ys } = data;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let k = 0; k < n; k++) {
    minY = Math.min(minY, ys[k]!);
    maxY = Math.max(maxY, ys[k]!);
  }
  const bounds: Bounds = n > 0 ? { x0: xs[0]!, w: xs[n - 1]! - xs[0]!, y0: minY, h: maxY - minY } : { x0: 0, w: 0, y0: 0, h: 0 };
  const maxLevel = n > 0 ? maxLevelFor(n, params.colLeaf) : -1;

  // Column point starts per level (from the finest), and exact column nodes.
  const starts: Int32Array[] = [];
  const columns: MomentNode[][] = [];
  if (n > 0) {
    const fine = new Int32Array(2 ** maxLevel + 1);
    for (let k = 0; k < n; k++) fine[colOf(bounds, xs[k]!, maxLevel) + 1]!++;
    for (let c = 1; c < fine.length; c++) fine[c] = fine[c]! + fine[c - 1]!;
    for (let l = 0; l <= maxLevel; l++) {
      const shift = 2 ** (maxLevel - l);
      const s = new Int32Array(2 ** l + 1);
      for (let c = 0; c <= 2 ** l; c++) s[c] = fine[c * shift]!;
      starts.push(s);
      const cols: MomentNode[] = [];
      for (let c = 0; c < 2 ** l; c++) cols.push(summarize(data, l, c, s[c]!, s[c + 1]!));
      columns.push(cols);
    }
  }

  // Planes, aspect by aspect: every level built from the points, gated per column.
  const qy = new Uint32Array(n);
  for (let k = 0; k < n; k++) qy[k] = quantize(ys[k]!, bounds.y0, bounds.h);
  const budget = Math.floor(n / params.minPointsPerCell);
  const planes: GridPlane[] = [];
  const aspects: GridModel["aspects"] = [];
  let writtenCells = 0;
  for (let d = params.minAspect; d <= params.maxAspect && n > 0; d += Math.max(params.aspectStep, 1)) {
    const record: GridModel["aspects"][number] = { d, levels: [] };
    aspects.push(record);
    for (let i = Math.max(d, 0); i <= maxLevel; i++) {
      const j = i - d;
      if (j > MAX_ROW_LEVEL) break;
      const cells = gatedCells(data, qy, starts[i]!, j, budget, params.minColumnPointsPerCell);
      if (cells.length === 0) continue;
      record.levels.push([i, j]);
      const colStarts = new Int32Array(2 ** i + 1);
      for (const cell of cells) colStarts[cell.col + 1]!++;
      for (let c = 1; c < colStarts.length; c++) colStarts[c] = colStarts[c]! + colStarts[c - 1]!;
      planes.push({
        id: 0,
        i,
        j,
        cells: cells.map((raw) => toCell(bounds, i, j, raw)),
        colStarts,
        offset: writtenCells * GRID_CELL_BYTES,
      });
      writtenCells += cells.length;
    }
  }

  const planesByLevel: GridPlane[][] = Array.from({ length: maxLevel + 1 }, () => []);
  for (const p of planes) planesByLevel[p.i]!.push(p);
  let id = 0;
  for (const level of planesByLevel) {
    level.sort((a, b) => a.j - b.j);
    for (const p of level) p.id = id++;
  }

  const levelOffsets: number[] = [];
  const recordBytes: number[] = [];
  let columnsBytes = 0;
  let manifestBytes = HEADER_BYTES;
  for (let l = 0; l <= maxLevel; l++) {
    const rb = NODE_BYTES + 4 + 4 * planesByLevel[l]!.length;
    levelOffsets.push(columnsBytes);
    recordBytes.push(rb);
    columnsBytes += (2 ** l + 1) * rb;
    manifestBytes += LEVEL_INFO_BYTES + PLANE_INFO_BYTES * planesByLevel[l]!.length;
  }
  const objects: StoredObject[] = [{ key: "manifest.bin", bytes: manifestBytes, kind: "manifest" }];
  if (n > 0) {
    objects.push({ key: COLUMNS_KEY, bytes: columnsBytes, kind: "level" });
    objects.push({ key: CELLS_KEY, bytes: writtenCells * GRID_CELL_BYTES, kind: "plane" });
    objects.push({ key: "points.bin", bytes: n * POINT_BYTES, kind: "points" });
  }
  const totalBytes = objects.reduce((s, o) => s + o.bytes, 0);
  return {
    strategy: "grid",
    data,
    params,
    bounds,
    maxLevel,
    columns,
    planes,
    planesByLevel,
    levelOffsets,
    recordBytes,
    aspects,
    objects,
    totalBytes,
    rawBytes: n * POINT_BYTES,
  };
}

// ---------------------------------------------------------------------------------------------
// Query.

/** A column of the exact decomposition: fully inside, or straddling at the bottom (raw). */
export interface GridPiece {
  kind: "inside" | "edge";
  col: MomentNode;
}

export type GridRender =
  | { kind: "node"; col: MomentNode; reason: "fits" | "offscreen" }
  | { kind: "cells"; col: MomentNode; plane: GridPlane; cells: GridCell[]; kernels: number }
  | { kind: "raw"; col: MomentNode; edge: boolean };

export interface CandidateEval {
  level: number;
  /** On-screen column width (σ units) and the row height budget it leaves. */
  w: number;
  hMax: number;
  /** Plane picked for the candidate's own level (null = none fits → raw). */
  plane: GridPlane | null;
  pieces: GridPiece[];
  renders: GridRender[];
  bytes: number;
}

export interface GridQueryResult {
  query: ViewQuery;
  yRange: [number, number] | null;
  effectiveY: [number, number];
  xFactor: number;
  yFactor: number;
  /** y-factor estimated from the inside columns' exact range (used to pick planes). */
  yFactorEstimate: number;
  base: number;
  candidates: CandidateEval[];
  chosen: CandidateEval | null;
  /** Record pages read (or found in cache) per level. */
  pages: Array<{ level: number; page: number; cached: boolean }>;
  items: FrontierItem[];
  groups: KernelGroup[];
  requests: RangeRequest[];
  rounds: number;
  wireBytes: number;
  kernelCount: number;
  cacheHits: number;
}

/**
 * `emit_column_cells` from `progressive-grid/src/client.rs`: kernels emitted for one column's
 * cells after collapsing far off-screen cells (above / below) and greedy merging in row order.
 */
export function mergedCellKernels(cells: GridCell[], query: ViewQuery, xFactor: number, yFactor: number, threshold: number) {
  const asNode = (c: GridCell): MomentNode => ({
    level: -1,
    index: -1,
    start: 0,
    end: 0,
    minX: c.rx0,
    maxX: c.rx1,
    minY: c.ry0,
    maxY: c.ry1,
    mass: c.mass,
    x: c.x,
    y: c.y,
    cxx: c.cxx,
    cxy: c.cxy,
    cyy: c.cyy,
  });
  const fitsMerged = (n: MomentNode) => {
    const dx = Math.max(n.x - n.minX, n.maxX - n.x) * xFactor;
    const dy = Math.max(n.y - n.minY, n.maxY - n.y) * yFactor;
    return dx * dx + dy * dy < threshold * threshold;
  };
  let below: MomentNode | null = null;
  let above: MomentNode | null = null;
  const groups: MomentNode[] = [];
  for (const cell of cells) {
    if (cell.mass <= 0) continue;
    const node = asNode(cell);
    if (query.yBounds && offscreenY(node, query, yFactor)) {
      const isBelow = node.maxY < query.yBounds[0];
      const side: MomentNode | null = isBelow ? below : above;
      const merged: MomentNode = side ? mergeNodes([side, node], -1, -1) : node;
      if (offscreenY(merged, query, yFactor)) {
        if (isBelow) below = merged;
        else above = merged;
        continue;
      }
    }
    const last = groups[groups.length - 1];
    if (last) {
      const merged = mergeNodes([last, node], -1, -1);
      if (fitsMerged(merged)) {
        groups[groups.length - 1] = merged;
        continue;
      }
    }
    groups.push(node);
  }
  return groups.length + (below ? 1 : 0) + (above ? 1 : 0);
}

function pagesFor(level: number, first: number, last: number, out: Array<[number, number]>) {
  const lastRec = Math.min(last, 2 ** level);
  for (let p = Math.floor(first / PAGE_RECORDS); p <= Math.floor(lastRec / PAGE_RECORDS); p++) out.push([level, p]);
}

function fitsThreshold(n: MomentNode, xFactor: number, yFactor: number, threshold: number) {
  const dx = (n.maxX - n.minX) * xFactor;
  const dy = (n.maxY - n.minY) * yFactor;
  return dx * dx + dy * dy < threshold * threshold;
}

export function runGridQuery(model: GridModel, query: ViewQuery, cache: ClientCache = emptyCache()): GridQueryResult {
  const { bounds, maxLevel } = model;
  const requests: RangeRequest[] = [];
  let round = 0;
  let cacheHits = 0;
  const sigma = Math.max(query.sigmaPx, 1e-12);
  const threshold = Math.max(query.mergeThreshold, 1e-12);
  const a = query.minX;
  const b = query.maxX;
  const xFactor = query.screenW / Math.max(b - a, 1e-12) / sigma;
  const yFactorOf = (range: [number, number]) => query.screenH / Math.max(range[1] - range[0], 1e-12) / sigma;
  const empty = (base: number): GridQueryResult => ({
    query,
    yRange: null,
    effectiveY: query.yBounds ?? [0, 1],
    xFactor,
    yFactor: 1,
    yFactorEstimate: 1,
    base,
    candidates: [],
    chosen: null,
    pages: [],
    items: [],
    groups: [],
    requests,
    rounds: round,
    wireBytes: requests.reduce((s, r) => s + r.len, 0),
    kernelCount: 0,
    cacheHits,
  });

  if (!cache.opened) {
    requests.push({ key: "manifest.bin", offset: 0, len: model.objects[0]!.bytes, round, phase: "open" });
    round++;
    cache.opened = true;
  }
  if (maxLevel < 0 || a > b) return empty(0);

  let base = maxLevel;
  for (let i = 0; i <= maxLevel; i++) {
    if (slot(bounds.w, i) * xFactor <= MAX_COL_WIDTH * threshold) {
      base = i;
      break;
    }
  }
  const levels = base + 1 <= maxLevel ? [base, base + 1] : [base];

  // Round 1: visible records at both candidates + the edge chains of every finer level.
  const wanted: Array<[number, number]> = [];
  for (const level of levels) pagesFor(level, colOf(bounds, a, level), colOf(bounds, b, level) + 1, wanted);
  for (let level = levels[levels.length - 1]! + 1; level <= maxLevel; level++) {
    for (const [edge, needed] of [
      [a, a > bounds.x0],
      [b, b < bounds.x0 + bounds.w],
    ] as const) {
      if (!needed) continue;
      const c = colOf(bounds, edge, level);
      pagesFor(level, c & ~1, (c | 1) + 1, wanted);
    }
  }
  const pageLog: GridQueryResult["pages"] = [];
  const have = new Set<string>();
  const fetchPages = (list: Array<[number, number]>) => {
    const missing = new Map<number, number[]>();
    for (const [level, page] of list) {
      const key = `${level}:${page}`;
      if (have.has(key)) continue;
      have.add(key);
      if (cache.treePages.has(key)) {
        cacheHits++;
        pageLog.push({ level, page, cached: true });
        continue;
      }
      cache.treePages.add(key);
      pageLog.push({ level, page, cached: false });
      const ids = missing.get(level) ?? [];
      ids.push(page);
      missing.set(level, ids);
    }
    if (missing.size === 0) return;
    for (const [level, ids] of [...missing.entries()].sort((x, y) => x[0] - y[0])) {
      const rb = model.recordBytes[level]!;
      const pageBytes = PAGE_RECORDS * rb;
      const total = 2 ** level + 1;
      for (const [start, count] of coalesceRuns(ids, Math.floor(MAX_GAP_BYTES / pageBytes), Math.max(Math.floor(MAX_RUN_BYTES / pageBytes), 1))) {
        const first = start * PAGE_RECORDS;
        const end = Math.min((start + count) * PAGE_RECORDS, total);
        requests.push({ key: COLUMNS_KEY, offset: model.levelOffsets[level]! + first * rb, len: (end - first) * rb, round, phase: "frontier" });
      }
    }
    round++;
  };
  fetchPages(wanted);

  // Exact decomposition from a candidate level (any record outside the prefetch costs a round).
  const decompose = (level0: number): GridPiece[] => {
    const pieces: GridPiece[] = [];
    const touched: Array<[number, number]> = [];
    const walk = (level: number, c: number) => {
      touched.push([level, Math.floor(c / PAGE_RECORDS)], [level, Math.floor((c + 1) / PAGE_RECORDS)]);
      const col = model.columns[level]![c]!;
      if (col.mass <= 0 && col.minX > col.maxX) return;
      if (col.end === col.start || col.maxX < a || col.minX > b) return;
      if (a <= col.minX && col.maxX <= b) pieces.push({ kind: "inside", col });
      else if (level === maxLevel || col.end - col.start <= model.params.colLeaf) pieces.push({ kind: "edge", col });
      else {
        walk(level + 1, 2 * c);
        walk(level + 1, 2 * c + 1);
      }
    };
    for (let c = colOf(bounds, a, level0); c <= colOf(bounds, b, level0); c++) walk(level0, c);
    fetchPages(touched);
    return pieces;
  };
  const decompositions = levels.map(decompose);

  // Lower bound of the visible y-range → conservative y-factor for the plane choice.
  let lower: [number, number] | null = null;
  let upper: [number, number] | null = null;
  const fold = (acc: [number, number] | null, n: MomentNode): [number, number] => (acc ? [Math.min(acc[0], n.minY), Math.max(acc[1], n.maxY)] : [n.minY, n.maxY]);
  for (const p of decompositions[0]!) {
    if (p.kind === "inside") lower = fold(lower, p.col);
    upper = fold(upper, p.col);
  }
  const estimate = lower ?? upper;
  if (!estimate) return empty(base);
  const yFactorEstimate = yFactorOf(query.yBounds ?? estimate);

  const pickPlane = (level: number) => {
    const w = slot(bounds.w, level) * xFactor;
    if (w >= threshold) return { w, hMax: 0, plane: null };
    const hMax = Math.sqrt(threshold * threshold - w * w);
    const plane = model.planesByLevel[level]!.find((p) => slot(bounds.h, p.j) * yFactorEstimate < hMax) ?? null;
    return { w, hMax, plane };
  };

  const candidates: CandidateEval[] = levels.map((level, idx) => {
    const pieces = decompositions[idx]!;
    let bytes = 0;
    const renders: GridRender[] = pieces.map((piece) => {
      const col = piece.col;
      const points = col.end - col.start;
      if (piece.kind === "edge") {
        bytes += points * POINT_BYTES;
        return { kind: "raw", col, edge: true };
      }
      if (offscreenY(col, query, yFactorEstimate)) return { kind: "node", col, reason: "offscreen" };
      if (fitsThreshold(col, xFactor, yFactorEstimate, threshold)) return { kind: "node", col, reason: "fits" };
      // A column the per-column gate left out of the plane (empty run) goes raw.
      const { plane } = pickPlane(col.level);
      const s = plane ? plane.colStarts[col.index]! : 0;
      const e = plane ? plane.colStarts[col.index + 1]! : 0;
      if (!plane || e === s) {
        bytes += points * POINT_BYTES;
        return { kind: "raw", col, edge: false };
      }
      bytes += (e - s) * GRID_CELL_BYTES;
      return { kind: "cells", col, plane, cells: plane.cells.slice(s, e), kernels: 0 };
    });
    const { w, hMax, plane } = pickPlane(level);
    return { level, w, hMax, plane, pieces, renders, bytes };
  });
  const chosen = candidates.reduce((best, c) => (c.bytes < best.bytes ? c : best));

  // Round 2: cells + raw points, concurrently.
  const cellSpans: Array<[number, number]> = [];
  const rawSpans: Array<[number, number]> = [];
  for (const r of chosen.renders) {
    if (r.kind === "cells") {
      const key = `${r.plane.id}:${r.col.index}`;
      if (cache.planeBlocks.has(key)) cacheHits++;
      else {
        cache.planeBlocks.add(key);
        cellSpans.push([r.plane.offset / GRID_CELL_BYTES + r.plane.colStarts[r.col.index]!, r.cells.length]);
      }
    } else if (r.kind === "raw") {
      const key = `raw:${r.col.start}:${r.col.end}`;
      if (cache.planeBlocks.has(key)) cacheHits++;
      else {
        cache.planeBlocks.add(key);
        rawSpans.push([r.col.start, r.col.end - r.col.start]);
      }
    }
  }
  const cellReads = coalesceSpans(cellSpans, Math.floor(MAX_GAP_BYTES / GRID_CELL_BYTES), Math.floor(MAX_RUN_BYTES / GRID_CELL_BYTES));
  const rawReads = coalesceSpans(rawSpans, MAX_GAP_POINTS, MAX_RUN_POINTS);
  for (const [s, e] of cellReads) requests.push({ key: CELLS_KEY, offset: s * GRID_CELL_BYTES, len: (e - s) * GRID_CELL_BYTES, round, phase: "planes+leaves" });
  for (const [s, e] of rawReads) requests.push({ key: "points.bin", offset: s * POINT_BYTES, len: (e - s) * POINT_BYTES, round, phase: "planes+leaves" });
  if (cellReads.length + rawReads.length > 0) round++;

  // Exact visible y-range: inside nodes + clipped raw points.
  let lo = Infinity;
  let hi = -Infinity;
  const items: FrontierItem[] = [];
  const { xs, ys } = model.data;
  for (const r of chosen.renders) {
    if (r.kind === "raw") {
      for (let k = r.col.start; k < r.col.end; k++) {
        const x = xs[k]!;
        if (x < a || x > b) continue;
        lo = Math.min(lo, ys[k]!);
        hi = Math.max(hi, ys[k]!);
      }
      items.push({ kind: "leaf", node: r.col });
    } else {
      lo = Math.min(lo, r.col.minY);
      hi = Math.max(hi, r.col.maxY);
      if (r.kind === "node") items.push({ kind: "resolved", node: r.col, reason: r.reason });
    }
  }
  if (!(lo <= hi)) return { ...empty(base), candidates, chosen, pages: pageLog };
  const yRange: [number, number] = [lo, hi];
  const effectiveY = query.yBounds ?? yRange;
  const yFactor = yFactorOf(effectiveY);
  const groups = assembleGroups(model, items, query, xFactor, yFactor, threshold);
  let cellKernels = 0;
  for (const r of chosen.renders) {
    if (r.kind !== "cells") continue;
    r.kernels = mergedCellKernels(r.cells, query, xFactor, yFactor, threshold);
    cellKernels += r.kernels;
  }
  return {
    query,
    yRange,
    effectiveY,
    xFactor,
    yFactor,
    yFactorEstimate,
    base,
    candidates,
    chosen,
    pages: pageLog,
    items,
    groups,
    requests,
    rounds: round,
    wireBytes: requests.reduce((s, r) => s + r.len, 0),
    kernelCount: groups.length + cellKernels,
    cacheHits,
  };
}
