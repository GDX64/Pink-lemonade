/**
 * In-browser model of the hierarchy-based progressive strategies: the moment ripmap
 * (`rust/progressive-ripmap`) and the paged x-tree (`rust/progressive-tree`).
 *
 * Mirrors, at visualization fidelity:
 * - the shared paged moment hierarchy (`level_counts`, `write_hierarchy`, root level inlined
 *   in `manifest.bin`, 64-byte nodes, 24-byte points),
 * - the ripmap's dense per-block y-bin planes (`max_depth`, `bin_geometry`, `plane_{ℓ}_{k}.bin`),
 * - the online query: pruned exact `visible_y_range`, the level-by-level frontier with the
 *   ripmap `decide` policy (`fits_threshold` / `offscreen_y` / `needed_depth`) or the x-tree's
 *   `standard_decision`, and the byte-range requests the client would issue (`coalesce_runs`),
 * - frame assembly (`assemble_frame`): greedy merging of resolved nodes and recursively halved
 *   raw leaf points into the emitted kernels (`push_group` / `collect_points`).
 *
 * The Rust code stays the source of truth; constants below are copied from it.
 */

export const NODE_BYTES = 64;
export const POINT_BYTES = 24;
export const CELL_BYTES = 24;
export const MAX_Y_DEPTH = 14;
export const MAX_PLANE_BLOCK_WIDTH = 0.9;
export const OFFSCREEN_MARGIN_SIGMA = 6;
const PLANE_MAX_GAP_BYTES = 16 * 1024;
const PLANE_MAX_RUN_BYTES = 1024 * 1024;
const TREE_MAX_GAP_BYTES = 16 * 1024;
const TREE_MAX_RUN_BYTES = 256 * 1024;
const MAX_GAP_LEAVES = 8;
const MAX_COALESCE_LEAVES = 256;

export type DatasetShape = "random_walk" | "noisy_spikes";

/** `moment_ripmap` (x-hierarchy + y-bin planes) or `paged_moment_tree` (x-hierarchy only). */
export type Strategy = "ripmap" | "tree";

export interface RipmapParams {
  leafPoints: number;
  fanout: number;
  rootCap: number;
  minPointsPerBin: number;
}

export const DEFAULT_PARAMS: RipmapParams = {
  leafPoints: 32,
  fanout: 4,
  rootCap: 1024,
  minPointsPerBin: 16,
};

/** `PagedTreeStrategy::default()`: fanout 32 (2 KiB pages) and `root_cap = fanout`. */
export const TREE_PARAMS: RipmapParams = {
  leafPoints: 32,
  fanout: 32,
  rootCap: 32,
  minPointsPerBin: 16,
};

export interface Dataset {
  shape: DatasetShape;
  xs: Float64Array;
  ys: Float64Array;
  ws: Float64Array;
}

/** Moment summary of a contiguous run of points (one hierarchy node / x-block). */
export interface MomentNode {
  level: number;
  index: number;
  /** Point index range `[start, end)` in x order. */
  start: number;
  end: number;
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  mass: number;
  x: number;
  y: number;
  cxx: number;
  cxy: number;
  cyy: number;
}

/** One y-bin cell of a plane block (a decoded `MomentCell`). */
export interface PlaneCell {
  bin: number;
  lo: number;
  h: number;
  count: number;
  mass: number;
  x: number;
  y: number;
  cxx: number;
  cxy: number;
  cyy: number;
}

export interface StoredObject {
  key: string;
  bytes: number;
  kind: "manifest" | "header" | "level" | "plane" | "points";
  level?: number;
  k?: number;
}

export interface RipmapModel {
  strategy: Strategy;
  data: Dataset;
  params: RipmapParams;
  levelCounts: number[];
  /** `levels[ℓ][j]`: node `j` at level `ℓ` (level 0 = leaves of `leafPoints` points). */
  levels: MomentNode[][];
  /** Deepest stored y-depth per level (0 = no planes). */
  kmax: number[];
  objects: StoredObject[];
  totalBytes: number;
  rawBytes: number;
}

export interface ViewQuery {
  minX: number;
  maxX: number;
  yBounds: [number, number] | null;
  screenW: number;
  screenH: number;
  sigmaPx: number;
  mergeThreshold: number;
}

export type Decision =
  | { kind: "resolve"; reason: "fits" | "offscreen" }
  | { kind: "plane"; k: number }
  | { kind: "expand"; reason: "boundary" | "too-big" | "too-wide" | "k-exceeds-kmax" };

export interface DecisionDetail {
  /** On-screen width / span in σ units. */
  w: number;
  span: number;
  hMax: number;
  kNeeded: number | null;
  kmax: number;
  fullyInside: boolean;
}

export interface TraceEntry {
  node: MomentNode;
  decision: Decision;
  detail: DecisionDetail;
}

export type FrontierItem =
  | { kind: "resolved"; node: MomentNode; reason: "fits" | "offscreen" }
  | { kind: "plane"; node: MomentNode; k: number; cells: PlaneCell[] }
  | { kind: "leaf"; node: MomentNode };

/**
 * One emitted Gaussian kernel built by greedy merging (`push_group`): consecutive resolved
 * nodes and halved raw-point runs keep merging into the previous kernel while the combined
 * moments still fit (`fits_merged_group`). Plane cells are emitted directly, never grouped.
 */
export interface KernelGroup {
  node: MomentNode;
  /** Frontier items (resolved nodes / leaves) that contributed to this kernel. */
  items: number[];
  /** Pieces merged: resolved nodes plus raw-point runs. */
  parts: number;
  /** Raw points merged in from leaves (0 if built only from resolved nodes). */
  rawPoints: number;
}

export interface RangeRequest {
  key: string;
  offset: number;
  len: number;
  round: number;
  phase: "open" | "y-range" | "frontier" | "planes+leaves";
}

export interface QueryResult {
  query: ViewQuery;
  yRange: [number, number] | null;
  effectiveY: [number, number];
  xFactor: number;
  yFactor: number;
  /** `trace[ℓ]`: every node classified at level `ℓ` during the frontier traversal. */
  trace: TraceEntry[][];
  items: FrontierItem[];
  groups: KernelGroup[];
  requests: RangeRequest[];
  rounds: number;
  wireBytes: number;
  kernelCount: number;
  cacheHits: number;
}

/** Fetch-state carried between queries when the cache is kept warm (unbounded here). */
export interface ClientCache {
  treePages: Set<string>;
  leaves: Set<number>;
  planeBlocks: Set<string>;
  opened: boolean;
}

export function emptyCache(): ClientCache {
  return { treePages: new Set(), leaves: new Set(), planeBlocks: new Set(), opened: false };
}

// ---------------------------------------------------------------------------------------------
// Dataset fixtures (same shapes as `progressive-harness` `DatasetShape`, different RNG).

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianSample(mean: number, std: number, x: number): number {
  const z = (x - mean) / std;
  return Math.exp(-0.5 * z * z);
}

export function generateDataset(shape: DatasetShape, n: number, seed = 42): Dataset {
  const xs = new Float64Array(n);
  const ys = new Float64Array(n);
  const ws = new Float64Array(n);
  const rand = mulberry32(seed);
  let spare: number | null = null;
  const normal = () => {
    if (spare !== null) {
      const s = spare;
      spare = null;
      return s;
    }
    const u = Math.max(rand(), 1e-12);
    const v = rand();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };
  let walk = n / 1000;
  for (let i = 0; i < n; i++) {
    const x = i / n;
    xs[i] = x;
    if (shape === "random_walk") {
      walk += normal();
      const density = gaussianSample(0.9, 0.05, x) + gaussianSample(0.3, 0.1, x) * 0.8;
      const s = density + normal() * 0.5;
      const weight = Math.abs(s ** 5 * 100);
      ys[i] = walk;
      ws[i] = weight > 0 ? weight : 1;
    } else {
      ys[i] = rand() * 20 - 10 + (i % 997 === 0 ? 80 : 0);
      ws[i] = 1;
    }
  }
  return { shape, xs, ys, ws };
}

// ---------------------------------------------------------------------------------------------
// Moments.

class Acc {
  count = 0;
  w = 0;
  wx = 0;
  wy = 0;
  wxx = 0;
  wxy = 0;
  wyy = 0;
  minX = Infinity;
  maxX = -Infinity;
  minY = Infinity;
  maxY = -Infinity;

  push(x: number, y: number, w: number) {
    this.count++;
    this.w += w;
    this.wx += w * x;
    this.wy += w * y;
    this.wxx += w * x * x;
    this.wxy += w * x * y;
    this.wyy += w * y * y;
    if (x < this.minX) this.minX = x;
    if (x > this.maxX) this.maxX = x;
    if (y < this.minY) this.minY = y;
    if (y > this.maxY) this.maxY = y;
  }

  moments() {
    const m = this.w > 0 ? this.w : 1;
    const x = this.wx / m;
    const y = this.wy / m;
    return {
      mass: this.w,
      x,
      y,
      cxx: Math.max(this.wxx / m - x * x, 0),
      cxy: this.wxy / m - x * y,
      cyy: Math.max(this.wyy / m - y * y, 0),
    };
  }
}

export function summarize(data: Dataset, level: number, index: number, start: number, end: number) {
  const acc = new Acc();
  for (let i = start; i < end; i++) acc.push(data.xs[i]!, data.ys[i]!, data.ws[i]!);
  return toNode(acc, level, index, start, end);
}

function toNode(acc: Acc, level: number, index: number, start: number, end: number): MomentNode {
  return {
    level,
    index,
    start,
    end,
    minX: acc.minX,
    maxX: acc.maxX,
    minY: acc.minY,
    maxY: acc.maxY,
    ...acc.moments(),
  };
}

// ---------------------------------------------------------------------------------------------
// Builder.

/** `level_counts` from `progressive-common/src/writer.rs`. */
export function levelCounts(n: number, leafPoints: number, fanout: number, rootCap: number) {
  if (n === 0) return [];
  const f = Math.max(fanout, 2);
  const cap = Math.max(rootCap, 1);
  const counts = [Math.ceil(n / Math.max(leafPoints, 1))];
  while (counts[counts.length - 1]! > cap) counts.push(Math.ceil(counts[counts.length - 1]! / f));
  return counts;
}

/** `max_depth` from `progressive-ripmap/src/format.rs`. */
export function maxDepth(blockPoints: number, minPointsPerBin: number): number {
  const ratio = Math.floor(blockPoints / Math.max(minPointsPerBin, 1));
  return ratio < 2 ? 0 : Math.min(Math.floor(Math.log2(ratio)), MAX_Y_DEPTH);
}

export function blockPointsAt(params: RipmapParams, level: number) {
  return params.leafPoints * params.fanout ** level;
}

export function planeKey(level: number, k: number) {
  return `plane_${level}_${k}.bin`;
}

export function buildRipmap(data: Dataset, inParams: RipmapParams, strategy: Strategy = "ripmap"): RipmapModel {
  // The x-tree always rolls up until at most `fanout` root nodes remain.
  const params = strategy === "tree" ? { ...inParams, rootCap: inParams.fanout } : inParams;
  const n = data.xs.length;
  const counts = levelCounts(n, params.leafPoints, params.fanout, params.rootCap);
  const levels: MomentNode[][] = [];
  for (let level = 0; level < counts.length; level++) {
    const bp = blockPointsAt(params, level);
    const nodes: MomentNode[] = [];
    if (level === 0) {
      for (let j = 0; j < counts[0]!; j++) {
        nodes.push(summarize(data, 0, j, j * bp, Math.min((j + 1) * bp, n)));
      }
    } else {
      const below = levels[level - 1]!;
      for (let j = 0; j < counts[level]!; j++) {
        nodes.push(mergeNodes(below.slice(j * params.fanout, (j + 1) * params.fanout), level, j));
      }
    }
    levels.push(nodes);
  }

  const kmax = counts.map((_, level) =>
    strategy === "tree" ? 0 : maxDepth(blockPointsAt(params, level), params.minPointsPerBin),
  );
  const rootLevel = counts.length - 1;
  const objects: StoredObject[] = [
    { key: "manifest.bin", bytes: 64 + (counts[rootLevel] ?? 0) * NODE_BYTES, kind: "manifest" },
  ];
  if (strategy === "ripmap") objects.push({ key: "ripmap.bin", bytes: 20 + counts.length, kind: "header" });
  for (let level = 0; level < rootLevel; level++) {
    objects.push({ key: `level_${level}.bin`, bytes: counts[level]! * NODE_BYTES, kind: "level", level });
  }
  for (let level = 0; level < counts.length; level++) {
    for (let k = 1; k <= kmax[level]!; k++) {
      objects.push({
        key: planeKey(level, k),
        bytes: counts[level]! * 2 ** k * CELL_BYTES,
        kind: "plane",
        level,
        k,
      });
    }
  }
  objects.push({ key: "points.bin", bytes: n * POINT_BYTES, kind: "points" });
  const totalBytes = objects.reduce((s, o) => s + o.bytes, 0);
  return { strategy, data, params, levelCounts: counts, levels, kmax, objects, totalBytes, rawBytes: n * POINT_BYTES };
}

export function mergeNodes(children: MomentNode[], level: number, index: number): MomentNode {
  let mass = 0;
  let sx = 0;
  let sy = 0;
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const c of children) {
    mass += c.mass;
    sx += c.mass * c.x;
    sy += c.mass * c.y;
    minX = Math.min(minX, c.minX);
    maxX = Math.max(maxX, c.maxX);
    minY = Math.min(minY, c.minY);
    maxY = Math.max(maxY, c.maxY);
  }
  const m = mass > 0 ? mass : 1;
  const x = sx / m;
  const y = sy / m;
  let cxx = 0;
  let cxy = 0;
  let cyy = 0;
  for (const c of children) {
    const dx = c.x - x;
    const dy = c.y - y;
    cxx += c.mass * (c.cxx + dx * dx);
    cxy += c.mass * (c.cxy + dx * dy);
    cyy += c.mass * (c.cyy + dy * dy);
  }
  return {
    level,
    index,
    start: children[0]!.start,
    end: children[children.length - 1]!.end,
    minX,
    maxX,
    minY,
    maxY,
    mass,
    x,
    y,
    cxx: cxx / m,
    cxy: cxy / m,
    cyy: cyy / m,
  };
}

/**
 * Decode block `node` of plane `(ℓ, k)`: bin at the finest stored depth, then merge sibling bins
 * down to `k`, exactly like `write_planes`.
 */
export function planeCells(model: RipmapModel, node: MomentNode, k: number): PlaneCell[] {
  const { xs, ys, ws } = model.data;
  const kmax = Math.max(model.kmax[node.level] ?? 0, k);
  const finest = 2 ** kmax;
  const shift = 2 ** (kmax - k);
  const dy = node.maxY - node.minY;
  const accs = Array.from({ length: 2 ** k }, () => new Acc());
  for (let i = node.start; i < node.end; i++) {
    const y = ys[i]!;
    const b = dy > 0 ? Math.min(Math.floor(((y - node.minY) / dy) * finest), finest - 1) : 0;
    accs[Math.floor(b / shift)]!.push(xs[i]!, y, ws[i]!);
  }
  const binH = dy / 2 ** k;
  return accs.map((acc, bin) => ({
    bin,
    lo: node.minY + bin * binH,
    h: binH,
    count: acc.count,
    ...acc.moments(),
  }));
}

// ---------------------------------------------------------------------------------------------
// Query.

/** `coalesce_runs` from `progressive-common/src/fetch.rs`. */
export function coalesceRuns(indices: number[], maxGap: number, maxRun: number): Array<[number, number]> {
  if (indices.length === 0) return [];
  const run = Math.max(maxRun, 1);
  const sorted = [...new Set(indices)].sort((a, b) => a - b);
  const runs: Array<[number, number]> = [];
  let start = sorted[0]!;
  let end = start + 1;
  for (const idx of sorted.slice(1)) {
    const gap = idx - end;
    const mergedLen = idx - start + 1;
    if (gap <= maxGap && mergedLen <= run) {
      end = idx + 1;
    } else {
      runs.push([start, end - start]);
      start = idx;
      end = idx + 1;
    }
  }
  runs.push([start, end - start]);
  return runs;
}

/** `needed_depth` from `progressive-ripmap/src/client.rs`. */
export function neededDepth(node: MomentNode, w: number, yFactor: number, threshold: number) {
  const hMax = Math.sqrt(Math.max(threshold * threshold - w * w, 0));
  const span = (node.maxY - node.minY) * yFactor;
  if (span <= hMax) return { k: 1, hMax, span };
  const bins = Math.ceil(span / Math.max(hMax, 1e-12));
  return { k: Math.min(Math.max(Math.ceil(Math.log2(bins)), 1), 63), hMax, span };
}

export function offscreenY(node: MomentNode, query: ViewQuery, yFactor: number): boolean {
  if (!query.yBounds) return false;
  const [lo, hi] = query.yBounds;
  const spread = Math.sqrt(1 + node.cyy * yFactor * yFactor);
  let edgeGap: number;
  let meanGap: number;
  if (node.minY > hi) {
    edgeGap = node.minY - hi;
    meanGap = node.y - hi;
  } else if (node.maxY < lo) {
    edgeGap = lo - node.maxY;
    meanGap = lo - node.y;
  } else {
    return false;
  }
  return edgeGap * yFactor >= OFFSCREEN_MARGIN_SIGMA && meanGap * yFactor >= OFFSCREEN_MARGIN_SIGMA * spread;
}

export function runQuery(model: RipmapModel, query: ViewQuery, cache: ClientCache = emptyCache()): QueryResult {
  const { levels, params, levelCounts: counts } = model;
  const rootLevel = counts.length - 1;
  const fanout = params.fanout;
  const requests: RangeRequest[] = [];
  let round = 0;
  let cacheHits = 0;

  if (!cache.opened) {
    // Manifest (+ `ripmap.bin` header for the ripmap) are read concurrently on open.
    for (const obj of model.objects) {
      if (obj.kind === "manifest" || obj.kind === "header") {
        requests.push({ key: obj.key, offset: 0, len: obj.bytes, round, phase: "open" });
      }
    }
    round++;
    cache.opened = true;
  }

  // Fetch tree pages (children of `parents` at `childLevel`) not yet cached: one round.
  const fetchPages = (childLevel: number, parents: number[], phase: RangeRequest["phase"]) => {
    const missing = parents.filter((p) => {
      const hit = cache.treePages.has(`${childLevel}:${p}`);
      if (hit) cacheHits++;
      return !hit;
    });
    if (missing.length === 0) return;
    const pageBytes = fanout * NODE_BYTES;
    const total = counts[childLevel]!;
    for (const [start, count] of coalesceRuns(
      missing,
      Math.floor(TREE_MAX_GAP_BYTES / pageBytes),
      Math.max(Math.floor(TREE_MAX_RUN_BYTES / pageBytes), 1),
    )) {
      const first = start * fanout;
      const last = Math.min((start + count) * fanout, total);
      requests.push({
        key: `level_${childLevel}.bin`,
        offset: first * NODE_BYTES,
        len: (last - first) * NODE_BYTES,
        round,
        phase,
      });
      for (let p = start; p < start + count; p++) cache.treePages.add(`${childLevel}:${p}`);
    }
    round++;
  };

  const leafRequests = (missing: number[], phase: RangeRequest["phase"], r: number) => {
    const n = model.data.xs.length;
    for (const [start, count] of coalesceRuns(missing, MAX_GAP_LEAVES, MAX_COALESCE_LEAVES)) {
      const startPt = start * params.leafPoints;
      const endPt = Math.min((start + count) * params.leafPoints, n);
      requests.push({
        key: "points.bin",
        offset: startPt * POINT_BYTES,
        len: (endPt - startPt) * POINT_BYTES,
        round: r,
        phase,
      });
      for (let l = start; l < start + count; l++) cache.leaves.add(l);
    }
  };

  const children = (level: number, parent: number) =>
    levels[level - 1]!.slice(parent * fanout, (parent + 1) * fanout);

  // Phase 1: pruned exact visible_y_range.
  const { minX, maxX } = query;
  let minY = Infinity;
  let maxY = -Infinity;
  let candidates: MomentNode[] = [];
  for (const node of levels[rootLevel] ?? []) {
    if (node.maxX < minX || node.minX > maxX) continue;
    if (minX <= node.minX && node.maxX <= maxX) {
      minY = Math.min(minY, node.minY);
      maxY = Math.max(maxY, node.maxY);
    } else candidates.push(node);
  }
  let straddling = candidates.filter((c) => c.minY < minY || c.maxY > maxY);
  for (let childLevel = rootLevel - 1; childLevel >= 0 && straddling.length > 0; childLevel--) {
    fetchPages(childLevel, straddling.map((s) => s.index), "y-range");
    candidates = [];
    for (const parent of straddling) {
      for (const node of children(childLevel + 1, parent.index)) {
        if (node.maxX < minX || node.minX > maxX) continue;
        if (minX <= node.minX && node.maxX <= maxX) {
          minY = Math.min(minY, node.minY);
          maxY = Math.max(maxY, node.maxY);
        } else candidates.push(node);
      }
    }
    straddling = candidates.filter((c) => c.minY < minY || c.maxY > maxY);
  }
  if (straddling.length > 0) {
    const missing = straddling.map((s) => s.index).filter((l) => !cache.leaves.has(l));
    if (missing.length > 0) {
      leafRequests(missing, "y-range", round);
      round++;
    }
    for (const leaf of straddling) {
      for (let i = leaf.start; i < leaf.end; i++) {
        const x = model.data.xs[i]!;
        if (x >= minX && x <= maxX) {
          minY = Math.min(minY, model.data.ys[i]!);
          maxY = Math.max(maxY, model.data.ys[i]!);
        }
      }
    }
  }
  const yRange: [number, number] | null = minY <= maxY ? [minY, maxY] : null;
  const effectiveY = query.yBounds ?? yRange ?? [0, 1];
  const sigma = Math.max(query.sigmaPx, 1e-12);
  const xFactor = query.screenW / Math.max(maxX - minX, 1e-12) / sigma;
  const yFactor = query.screenH / Math.max(effectiveY[1] - effectiveY[0], 1e-12) / sigma;
  const threshold = Math.max(query.mergeThreshold, 1e-12);

  const trace: TraceEntry[][] = counts.map(() => []);
  const items: FrontierItem[] = [];
  if (!yRange) {
    return { query, yRange, effectiveY, xFactor, yFactor, trace, items, groups: [], requests, rounds: round, wireBytes: sum(requests), kernelCount: 0, cacheHits };
  }

  // Phase 2: frontier with the ripmap decision policy.
  const decide = (node: MomentNode, level: number, fullyInside: boolean): TraceEntry => {
    const w = (node.maxX - node.minX) * xFactor;
    const span = (node.maxY - node.minY) * yFactor;
    const kmax = model.kmax[level] ?? 0;
    const detail: DecisionDetail = { w, span, hMax: NaN, kNeeded: null, kmax, fullyInside };
    if (!fullyInside) return { node, decision: { kind: "expand", reason: "boundary" }, detail };
    if (offscreenY(node, query, yFactor)) return { node, decision: { kind: "resolve", reason: "offscreen" }, detail };
    if (w * w + span * span < threshold * threshold) {
      return { node, decision: { kind: "resolve", reason: "fits" }, detail };
    }
    if (model.strategy === "tree") return { node, decision: { kind: "expand", reason: "too-big" }, detail };
    if (w <= MAX_PLANE_BLOCK_WIDTH * threshold) {
      const nd = neededDepth(node, w, yFactor, threshold);
      detail.hMax = nd.hMax;
      detail.kNeeded = nd.k;
      if (nd.k <= kmax) return { node, decision: { kind: "plane", k: nd.k }, detail };
      return { node, decision: { kind: "expand", reason: "k-exceeds-kmax" }, detail };
    }
    return { node, decision: { kind: "expand", reason: "too-wide" }, detail };
  };

  const done: Array<{ kind: "resolved" | "plane" | "leaf"; entry: TraceEntry }> = [];
  const classify = (node: MomentNode, level: number, expand: MomentNode[]) => {
    if (node.maxX < minX || node.minX > maxX) return;
    const fullyInside = minX <= node.minX && node.maxX <= maxX;
    const entry = decide(node, level, fullyInside);
    trace[level]!.push(entry);
    const d = entry.decision;
    if (d.kind === "resolve") done.push({ kind: "resolved", entry });
    else if (d.kind === "plane") done.push({ kind: "plane", entry });
    else if (level > 0) expand.push(node);
    else done.push({ kind: "leaf", entry });
  };

  let expand: MomentNode[] = [];
  for (const node of levels[rootLevel] ?? []) classify(node, rootLevel, expand);
  for (let childLevel = rootLevel - 1; childLevel >= 0 && expand.length > 0; childLevel--) {
    fetchPages(childLevel, expand.map((e) => e.index), "frontier");
    const next: MomentNode[] = [];
    for (const parent of expand) {
      for (const node of children(childLevel + 1, parent.index)) classify(node, childLevel, next);
    }
    expand = next;
  }

  // Phase 3: raw boundary leaves and plane blocks, fetched concurrently in one round.
  const missingLeaves: number[] = [];
  const missingPlanes = new Map<string, number[]>();
  let kernelCount = 0;
  for (const { kind, entry } of done.sort((a, b) => a.entry.node.minX - b.entry.node.minX)) {
    const node = entry.node;
    if (kind === "resolved") {
      const d = entry.decision as Extract<Decision, { kind: "resolve" }>;
      items.push({ kind: "resolved", node, reason: d.reason });
    } else if (kind === "leaf") {
      items.push({ kind: "leaf", node });
      if (cache.leaves.has(node.index)) cacheHits++;
      else missingLeaves.push(node.index);
    } else {
      const k = (entry.decision as Extract<Decision, { kind: "plane" }>).k;
      const cells = planeCells(model, node, k);
      items.push({ kind: "plane", node, k, cells });
      kernelCount += cells.filter((c) => c.count > 0).length;
      const blockKey = `${node.level}:${k}:${node.index}`;
      if (cache.planeBlocks.has(blockKey)) {
        cacheHits++;
      } else {
        const key = `${node.level}:${k}`;
        const list = missingPlanes.get(key) ?? [];
        list.push(node.index);
        missingPlanes.set(key, list);
        cache.planeBlocks.add(blockKey);
      }
    }
  }
  if (missingLeaves.length > 0 || missingPlanes.size > 0) {
    leafRequests(missingLeaves, "planes+leaves", round);
    for (const [key, blocks] of missingPlanes) {
      const [level, k] = key.split(":").map(Number) as [number, number];
      const bytesPerBlock = 2 ** k * CELL_BYTES;
      const total = counts[level]!;
      for (const [start, count] of coalesceRuns(
        blocks,
        Math.floor(PLANE_MAX_GAP_BYTES / bytesPerBlock),
        Math.max(Math.floor(PLANE_MAX_RUN_BYTES / bytesPerBlock), 1),
      )) {
        const end = Math.min(start + count, total);
        requests.push({
          key: planeKey(level, k),
          offset: start * bytesPerBlock,
          len: (end - start) * bytesPerBlock,
          round,
          phase: "planes+leaves",
        });
        // Bridged gap blocks arrive too but the Rust client only caches requested ones.
      }
    }
    round++;
  }

  const groups = assembleGroups(model, items, query, xFactor, yFactor, threshold);
  kernelCount += groups.length;

  return {
    query,
    yRange,
    effectiveY,
    xFactor,
    yFactor,
    trace,
    items,
    groups,
    requests,
    rounds: round,
    wireBytes: sum(requests),
    kernelCount,
    cacheHits,
  };
}

function sum(requests: RangeRequest[]) {
  return requests.reduce((s, r) => s + r.len, 0);
}

// ---------------------------------------------------------------------------------------------
// Frame assembly (`assemble_frame` in `progressive-common/src/render.rs`).

function fitsMergedGroup(node: MomentNode, xFactor: number, yFactor: number, threshold: number) {
  const dx = Math.max(node.x - node.minX, node.maxX - node.x) * xFactor;
  const dy = Math.max(node.y - node.minY, node.maxY - node.y) * yFactor;
  return dx * dx + dy * dy < threshold * threshold;
}

function fitsThreshold(node: MomentNode, xFactor: number, yFactor: number, threshold: number) {
  const dx = (node.maxX - node.minX) * xFactor;
  const dy = (node.maxY - node.minY) * yFactor;
  return dx * dx + dy * dy < threshold * threshold;
}

export function assembleGroups(
  model: { data: Dataset },
  items: FrontierItem[],
  query: ViewQuery,
  xFactor: number,
  yFactor: number,
  threshold: number,
): KernelGroup[] {
  const groups: KernelGroup[] = [];
  const { xs } = model.data;

  const pushGroup = (node: MomentNode, item: number, rawPoints: number) => {
    if (node.mass <= 0 && node.minX > node.maxX) return;
    const last = groups[groups.length - 1];
    if (last) {
      const combined = mergeNodes([last.node, node], -1, -1);
      combined.start = Math.min(last.node.start, node.start);
      combined.end = Math.max(last.node.end, node.end);
      if (fitsMergedGroup(combined, xFactor, yFactor, threshold)) {
        last.node = combined;
        if (last.items[last.items.length - 1] !== item) last.items.push(item);
        last.parts++;
        last.rawPoints += rawPoints;
        return;
      }
    }
    groups.push({ node, items: [item], parts: 1, rawPoints });
  };

  // Recursively halve an x-sorted run of raw points until each half fits (`collect_points`).
  const collect = (start: number, end: number, item: number) => {
    if (start >= end) return;
    const node = summarize(model.data, -1, -1, start, end);
    if (end - start === 1 || fitsThreshold(node, xFactor, yFactor, threshold)) {
      pushGroup(node, item, end - start);
    } else {
      const mid = start + ((end - start) >> 1);
      collect(start, mid, item);
      collect(mid, end, item);
    }
  };

  for (const [i, item] of items.entries()) {
    if (item.kind === "resolved") {
      pushGroup(item.node, i, 0);
    } else if (item.kind === "leaf") {
      let start = item.node.start;
      let end = item.node.end;
      while (start < end && xs[start]! < query.minX) start++;
      while (end > start && xs[end - 1]! > query.maxX) end--;
      collect(start, end, i);
    }
  }
  return groups;
}
