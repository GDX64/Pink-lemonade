/**
 * In-browser model of the hierarchy-based progressive strategies: Moment Ripmap v2
 * (`rust/progressive-ripmap`, `PLPRIPM2`) and the paged x-tree (`rust/progressive-tree`, `PLPTREE2`).
 *
 * Mirrors, at visualization fidelity:
 * - Ripmap v2 (`PLPRIPM2`): 4-object layout (`manifest.bin`, `levels.bin`, `planes.bin`,
 *   `points.bin`), 16-byte sparse `SparseCell`s gated per block (`nonEmpty <= blockPoints / minPointsPerBin`),
 *   inline 6-byte plane directory entries per node, data-derived `planeDepthsForLevel`,
 *   per-node `x_factor`-guided level jumping, explicit-`y_bounds` Phase-1 skip, cross-`k` cache
 *   reuse, and greedy merging + off-screen y collapsing of plane cells.
 * - Paged x-tree (`PLPTREE2`): unchanged shared paged moment hierarchy with `standard_decision`.
 *
 * The Rust code stays the source of truth; constants below are copied from it.
 */

export const NODE_BYTES = 64;
export const POINT_BYTES = 24;
/** 16-byte quantized `SparseCell` (`bin: u16` + 14-byte `QuantizedMoments`). */
export const CELL_BYTES = 16;
/** 6-byte inline plane directory slot `(cell_offset: u32, cell_count: u16)`. */
export const DIR_ENTRY_BYTES = 6;
export const MAX_Y_DEPTH = 14;
export const DEFAULT_SPARSE_EXTRA_DEPTH = 2;
export const MAX_PLANE_BLOCK_WIDTH = 0.9;
export const OFFSCREEN_MARGIN_SIGMA = 6;
const MAX_BOUNDARY_JUMP = 3;
const PLANE_MAX_GAP_BYTES = 16 * 1024;
const PLANE_MAX_RUN_BYTES = 1024 * 1024;
const TREE_MAX_GAP_BYTES = 16 * 1024;
const TREE_MAX_RUN_BYTES = 256 * 1024;
const MAX_GAP_LEAVES = 8;
const MAX_COALESCE_LEAVES = 256;

export type DatasetShape = "random_walk" | "noisy_spikes";

/** `moment_ripmap` (`PLPRIPM2` sparse planes) or `paged_moment_tree` (x-hierarchy only). */
export type Strategy = "ripmap" | "tree";

export interface RipmapParams {
  leafPoints: number;
  fanout: number;
  rootCap: number;
  minPointsPerBin: number;
  sparseExtraDepth?: number;
}

export const DEFAULT_PARAMS: RipmapParams = {
  leafPoints: 32,
  fanout: 4,
  rootCap: 1024,
  minPointsPerBin: 16,
  sparseExtraDepth: DEFAULT_SPARSE_EXTRA_DEPTH,
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

export interface PlaneDirEntry {
  cellOffset: number;
  cellCount: number;
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
  /** Stored y-depths `k` per level (`PLPRIPM2` data-derived schedule). */
  planeDepths: number[][];
  /** Byte offset of each non-root level inside `levels.bin` (`PLPRIPM2`). */
  levelByteOffsets: number[];
  /** Byte offset of each stored plane `(ℓ, slot)` inside `planes.bin` (`PLPRIPM2`). */
  planeByteOffsets: number[][];
  /** `blockDirs[ℓ][j][slot]`: inline directory slot `(cellOffset, cellCount)` for block `j`. */
  blockDirs: PlaneDirEntry[][][];
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

/** `plane_depths_for_level` from `progressive-ripmap/src/format.rs` (`PLPRIPM2`). */
export function planeDepthsForLevel(
  level: number,
  blockPoints: number,
  minPointsPerBin: number,
  sparseExtraDepth = DEFAULT_SPARSE_EXTRA_DEPTH,
): number[] {
  if (level === 0) return [];
  const ratio = Math.floor(blockPoints / Math.max(minPointsPerBin, 1));
  if (ratio < 2) return [];
  const kDense = Math.min(Math.floor(Math.log2(ratio)), MAX_Y_DEPTH);
  const kCap = Math.min(kDense + sparseExtraDepth, MAX_Y_DEPTH);
  const stride1End = Math.min(Math.max(kDense - 2, 0), kCap);

  const ks: number[] = [];
  for (let k = 1; k <= stride1End; k++) ks.push(k);
  let k = stride1End === 0 ? 1 : Math.min(stride1End + 2, kCap);
  while (k <= kCap) {
    if (ks[ks.length - 1] !== k) ks.push(k);
    if (k === kCap) break;
    k = Math.min(k + 2, kCap);
  }
  return ks;
}

/** `max_bins_per_block` from `progressive-ripmap/src/format.rs`. */
export function maxBinsPerBlock(blockPoints: number, minPointsPerBin: number): number {
  return Math.min(Math.floor(blockPoints / Math.max(minPointsPerBin, 1)), 65535);
}

/** `record_bytes` from `progressive-ripmap/src/format.rs`. */
export function recordBytesFor(planeCount: number): number {
  return NODE_BYTES + planeCount * DIR_ENTRY_BYTES;
}

/** Legacy helper kept for compatibility: deepest dense depth `floor(log2(blockPoints / minPointsPerBin))`. */
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
  const sparseExtra = params.sparseExtraDepth ?? DEFAULT_SPARSE_EXTRA_DEPTH;
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

  const rootLevel = counts.length - 1;

  if (strategy === "tree") {
    const kmax = counts.map(() => 0);
    const planeDepths = counts.map(() => [] as number[]);
    const levelByteOffsets = counts.map(() => 0);
    const planeByteOffsets = counts.map(() => [] as number[]);
    const blockDirs = counts.map((c) => Array.from({ length: c }, () => [] as PlaneDirEntry[]));
    const objects: StoredObject[] = [
      { key: "manifest.bin", bytes: 32 + counts.length * 8 + (counts[rootLevel] ?? 0) * NODE_BYTES, kind: "manifest" },
    ];
    for (let level = 0; level < rootLevel; level++) {
      objects.push({ key: `level_${level}.bin`, bytes: counts[level]! * NODE_BYTES, kind: "level", level });
    }
    objects.push({ key: "points.bin", bytes: n * POINT_BYTES, kind: "points" });
    const totalBytes = objects.reduce((s, o) => s + o.bytes, 0);
    return {
      strategy,
      data,
      params,
      levelCounts: counts,
      levels,
      kmax,
      planeDepths,
      levelByteOffsets,
      planeByteOffsets,
      blockDirs,
      objects,
      totalBytes,
      rawBytes: n * POINT_BYTES,
    };
  }

  // PLPRIPM2: 4 objects (`manifest.bin`, `levels.bin`, `planes.bin`, `points.bin`),
  // 16-byte sparse cells gated per block, inline per-node directories.
  const planeDepths = counts.map((_, level) =>
    planeDepthsForLevel(level, blockPointsAt(params, level), params.minPointsPerBin, sparseExtra),
  );
  const kmax = planeDepths.map((ks) => (ks.length > 0 ? ks[ks.length - 1]! : 0));
  const levelByteOffsets: number[] = counts.map(() => 0);
  const planeByteOffsets: number[][] = planeDepths.map((ks) => ks.map(() => 0));
  const blockDirs: PlaneDirEntry[][][] = counts.map((c, level) =>
    Array.from({ length: c }, () => planeDepths[level]!.map(() => ({ cellOffset: 0, cellCount: 0 }))),
  );

  let levelsBytes = 0;
  let planesBytes = 0;
  const { ys } = data;

  for (let level = 0; level < counts.length; level++) {
    const ks = planeDepths[level]!;
    const nodes = levels[level]!;
    const bp = blockPointsAt(params, level);
    const maxBins = maxBinsPerBlock(bp, params.minPointsPerBin);

    if (ks.length > 0) {
      const kHi = ks[ks.length - 1]!;
      const kLo = ks[0]!;
      const finest = 1 << kHi;
      const occupied = new Uint8Array(finest);
      const cellCountsPerPlane = ks.map(() => 0);

      for (let j = 0; j < nodes.length; j++) {
        const node = nodes[j]!;
        const dy = node.maxY - node.minY;
        const inv = dy > 0 ? finest / dy : 0;
        occupied.fill(0);
        for (let i = node.start; i < node.end; i++) {
          const b = dy > 0 ? Math.min(Math.floor((ys[i]! - node.minY) * inv), finest - 1) : 0;
          occupied[b] = 1;
        }

        let len = finest;
        let slot = ks.length - 1;
        for (let k = kHi; k >= kLo; k--) {
          if (ks[slot] === k) {
            let nonEmpty = 0;
            for (let b = 0; b < len; b++) nonEmpty += occupied[b]!;
            if (nonEmpty > 0 && nonEmpty <= maxBins) {
              blockDirs[level]![j]![slot] = {
                cellOffset: cellCountsPerPlane[slot]!,
                cellCount: nonEmpty,
              };
              cellCountsPerPlane[slot]! += nonEmpty;
            }
            if (slot === 0) break;
            slot--;
          }
          if (k > kLo) {
            len >>>= 1;
            for (let b = 0; b < len; b++) occupied[b] = occupied[2 * b]! | occupied[2 * b + 1]!;
          }
        }
      }

      for (let slot = 0; slot < ks.length; slot++) {
        planeByteOffsets[level]![slot] = planesBytes;
        planesBytes += cellCountsPerPlane[slot]! * CELL_BYTES;
      }
    }

    if (level < rootLevel) {
      levelByteOffsets[level] = levelsBytes;
      levelsBytes += nodes.length * recordBytesFor(ks.length);
    }
  }

  let manifestBytes = 40;
  for (let level = 0; level < counts.length; level++) {
    manifestBytes += 17 + planeDepths[level]!.length * 17;
  }
  if (rootLevel >= 0) {
    manifestBytes += counts[rootLevel]! * recordBytesFor(planeDepths[rootLevel]!.length);
  }

  const objects: StoredObject[] = [{ key: "manifest.bin", bytes: manifestBytes, kind: "manifest" }];
  if (levelsBytes > 0) objects.push({ key: "levels.bin", bytes: levelsBytes, kind: "level" });
  if (planesBytes > 0) objects.push({ key: "planes.bin", bytes: planesBytes, kind: "plane" });
  objects.push({ key: "points.bin", bytes: n * POINT_BYTES, kind: "points" });
  const totalBytes = objects.reduce((s, o) => s + o.bytes, 0);

  return {
    strategy,
    data,
    params,
    levelCounts: counts,
    levels,
    kmax,
    planeDepths,
    levelByteOffsets,
    planeByteOffsets,
    blockDirs,
    objects,
    totalBytes,
    rawBytes: n * POINT_BYTES,
  };
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

function coalesceByteSpans(spans: Array<[number, number]>, maxGap: number, maxRun: number): Array<[number, number]> {
  const clean = spans.filter(([, len]) => len > 0).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (clean.length === 0) return [];
  const reads: Array<[number, number]> = [];
  let [start, end] = [clean[0]![0], clean[0]![0] + clean[0]![1]];
  for (let i = 1; i < clean.length; i++) {
    const [s, len] = clean[i]!;
    const e = s + len;
    if (s <= end + maxGap && Math.max(e, end) - start <= maxRun) {
      end = Math.max(end, e);
    } else {
      reads.push([start, end]);
      [start, end] = [s, e];
    }
  }
  reads.push([start, end]);
  return reads;
}

/** Count emitted GPU kernels for a sparse plane block after greedy merging + safe off-screen y collapsing. */
function countMergedPlaneKernels(
  node: MomentNode,
  cells: PlaneCell[],
  query: ViewQuery,
  xFactor: number,
  yFactor: number,
  threshold: number,
): number {
  let below: MomentNode | null = null;
  let above: MomentNode | null = null;
  const onScreen: MomentNode[] = [];
  for (const cell of cells) {
    if (cell.count <= 0 || cell.mass <= 0) continue;
    const cellNode: MomentNode = {
      level: -1,
      index: cell.bin,
      start: node.start,
      end: node.end,
      minX: node.minX,
      maxX: node.maxX,
      minY: cell.lo,
      maxY: cell.lo + cell.h,
      mass: cell.mass,
      x: cell.x,
      y: cell.y,
      cxx: cell.cxx,
      cxy: cell.cxy,
      cyy: cell.cyy,
    };
    if (query.yBounds) {
      const [ylo, yhi] = query.yBounds;
      if (cellNode.maxY < ylo && offscreenY(cellNode, query, yFactor)) {
        const cand: MomentNode = below ? mergeNodes([below, cellNode], -1, -1) : cellNode;
        if (offscreenY(cand, query, yFactor)) {
          below = cand;
          continue;
        }
      } else if (cellNode.minY > yhi && offscreenY(cellNode, query, yFactor)) {
        const cand: MomentNode = above ? mergeNodes([above, cellNode], -1, -1) : cellNode;
        if (offscreenY(cand, query, yFactor)) {
          above = cand;
          continue;
        }
      }
    }
    const last = onScreen[onScreen.length - 1];
    if (last) {
      const combined = mergeNodes([last, cellNode], -1, -1);
      if (fitsMergedGroup(combined, xFactor, yFactor, threshold)) {
        onScreen[onScreen.length - 1] = combined;
        continue;
      }
    }
    onScreen.push(cellNode);
  }
  return onScreen.length + (below ? 1 : 0) + (above ? 1 : 0);
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
  const isV2 = model.strategy === "ripmap";
  const requests: RangeRequest[] = [];
  let round = 0;
  let cacheHits = 0;

  if (!cache.opened) {
    // Manifest is read in a single EOF-clamped request on open.
    for (const obj of model.objects) {
      if (obj.kind === "manifest" || obj.kind === "header") {
        requests.push({ key: obj.key, offset: 0, len: obj.bytes, round, phase: "open" });
      }
    }
    round++;
    cache.opened = true;
  }

  // Fetch tree pages at `targetLevel` (each page = `fanout` records) not yet cached: one round.
  const fetchPages = (targetLevel: number, pageIndices: number[], phase: RangeRequest["phase"], r = round): boolean => {
    const missing = [...new Set(pageIndices)].filter((p) => {
      const hit = cache.treePages.has(`${targetLevel}:${p}`);
      if (hit) cacheHits++;
      return !hit;
    });
    if (missing.length === 0) return false;
    const stride = isV2 ? recordBytesFor(model.planeDepths[targetLevel]!.length) : NODE_BYTES;
    const pageBytes = fanout * stride;
    const total = counts[targetLevel]!;
    const baseOff = isV2 ? model.levelByteOffsets[targetLevel]! : 0;
    const key = isV2 ? "levels.bin" : `level_${targetLevel}.bin`;
    for (const [start, count] of coalesceRuns(
      missing,
      Math.floor(TREE_MAX_GAP_BYTES / pageBytes),
      Math.max(Math.floor(TREE_MAX_RUN_BYTES / pageBytes), 1),
    )) {
      const first = start * fanout;
      const last = Math.min((start + count) * fanout, total);
      requests.push({
        key,
        offset: baseOff + first * stride,
        len: (last - first) * stride,
        round: r,
        phase,
      });
      for (let p = start; p < start + count; p++) cache.treePages.add(`${targetLevel}:${p}`);
    }
    return true;
  };

  const fetchDescendants = (parentLevel: number, targetLevel: number, parents: number[], phase: RangeRequest["phase"]) => {
    const step = parentLevel - targetLevel;
    const pagesPerParent = fanout ** (step - 1);
    const totalPages = Math.ceil(counts[targetLevel]! / fanout);
    const needed: number[] = [];
    for (const p of parents) {
      const lo = p * pagesPerParent;
      const hi = Math.min((p + 1) * pagesPerParent, totalPages);
      for (let page = lo; page < hi; page++) needed.push(page);
    }
    if (fetchPages(targetLevel, needed, phase)) round++;
    const out: MomentNode[] = [];
    for (const page of needed) {
      const base = page * fanout;
      const end = Math.min(base + fanout, counts[targetLevel]!);
      for (let idx = base; idx < end; idx++) out.push(levels[targetLevel]![idx]!);
    }
    return out;
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

  const { minX, maxX } = query;
  const sigma = Math.max(query.sigmaPx, 1e-12);
  const xFactor = query.screenW / Math.max(maxX - minX, 1e-12) / sigma;
  const threshold = Math.max(query.mergeThreshold, 1e-12);
  const trace: TraceEntry[][] = counts.map(() => []);
  const items: FrontierItem[] = [];

  if (!isV2) {
    // Unchanged `paged_moment_tree` (x-tree) traversal.
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
      if (fetchPages(childLevel, straddling.map((s) => s.index), "y-range")) round++;
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
    const yFactor = query.screenH / Math.max(effectiveY[1] - effectiveY[0], 1e-12) / sigma;
    if (!yRange) {
      return { query, yRange, effectiveY, xFactor, yFactor, trace, items, groups: [], requests, rounds: round, wireBytes: sum(requests), kernelCount: 0, cacheHits };
    }
    const done: Array<{ kind: "resolved" | "leaf"; entry: TraceEntry }> = [];
    const classify = (node: MomentNode, level: number, expand: MomentNode[]) => {
      if (node.maxX < minX || node.minX > maxX) return;
      const fullyInside = minX <= node.minX && node.maxX <= maxX;
      const w = (node.maxX - node.minX) * xFactor;
      const span = (node.maxY - node.minY) * yFactor;
      const detail: DecisionDetail = { w, span, hMax: NaN, kNeeded: null, kmax: 0, fullyInside };
      const decision: Decision = !fullyInside
        ? { kind: "expand", reason: "boundary" }
        : offscreenY(node, query, yFactor)
          ? { kind: "resolve", reason: "offscreen" }
          : w * w + span * span < threshold * threshold
            ? { kind: "resolve", reason: "fits" }
            : { kind: "expand", reason: "too-big" };
      const entry = { node, decision, detail };
      trace[level]!.push(entry);
      if (decision.kind === "resolve") done.push({ kind: "resolved", entry });
      else if (level > 0) expand.push(node);
      else done.push({ kind: "leaf", entry });
    };
    let expand: MomentNode[] = [];
    for (const node of levels[rootLevel] ?? []) classify(node, rootLevel, expand);
    for (let childLevel = rootLevel - 1; childLevel >= 0 && expand.length > 0; childLevel--) {
      if (fetchPages(childLevel, expand.map((e) => e.index), "frontier")) round++;
      const next: MomentNode[] = [];
      for (const parent of expand) {
        for (const node of children(childLevel + 1, parent.index)) classify(node, childLevel, next);
      }
      expand = next;
    }
    const missingLeaves: number[] = [];
    for (const { kind, entry } of done.sort((a, b) => a.entry.node.minX - b.entry.node.minX)) {
      if (kind === "resolved") {
        const d = entry.decision as Extract<Decision, { kind: "resolve" }>;
        items.push({ kind: "resolved", node: entry.node, reason: d.reason });
      } else {
        items.push({ kind: "leaf", node: entry.node });
        if (cache.leaves.has(entry.node.index)) cacheHits++;
        else missingLeaves.push(entry.node.index);
      }
    }
    if (missingLeaves.length > 0) {
      leafRequests(missingLeaves, "planes+leaves", round);
      round++;
    }
    const groups = assembleGroups(model, items, query, xFactor, yFactor, threshold);
    return { query, yRange, effectiveY, xFactor, yFactor, trace, items, groups, requests, rounds: round, wireBytes: sum(requests), kernelCount: groups.length, cacheHits };
  }

  // PLPRIPM2 Phase 0: per-node x_factor-guided jump to initial mixed-level frontier.
  const explicitYFactor = query.yBounds
    ? query.screenH / Math.max(query.yBounds[1] - query.yBounds[0], 1e-12) / sigma
    : null;
  const byLevel: MomentNode[][] = counts.map(() => []);
  byLevel[rootLevel] = (levels[rootLevel] ?? []).filter((n) => !(n.maxX < minX || n.minX > maxX));
  if (byLevel[rootLevel]!.length === 0) {
    return { query, yRange: null, effectiveY: query.yBounds ?? [0, 1], xFactor, yFactor: explicitYFactor ?? 1, trace, items, groups: [], requests, rounds: round, wireBytes: sum(requests), kernelCount: 0, cacheHits };
  }
  const targetW = MAX_PLANE_BLOCK_WIDTH * threshold;
  for (let curLevel = rootLevel; curLevel >= 1; curLevel--) {
    const curNodes = byLevel[curLevel]!;
    byLevel[curLevel] = [];
    if (curNodes.length === 0) continue;
    const byDrop = new Map<number, number[]>();
    for (const node of curNodes) {
      const fullyInside = minX <= node.minX && node.maxX <= maxX;
      const isOffscreen = fullyInside && explicitYFactor !== null && offscreenY(node, query, explicitYFactor);
      const w = (node.maxX - node.minX) * xFactor;
      if (w <= targetW || isOffscreen) {
        byLevel[curLevel]!.push(node);
      } else {
        const idealDrop = Math.max(Math.ceil(Math.log(w / targetW) / Math.log(fanout)), 1);
        const drop = Math.min(idealDrop, MAX_BOUNDARY_JUMP, curLevel);
        const list = byDrop.get(drop) ?? [];
        list.push(node.index);
        byDrop.set(drop, list);
      }
    }
    for (const [drop, parents] of byDrop) {
      const targetLevel = curLevel - drop;
      const desc = fetchDescendants(curLevel, targetLevel, parents, "frontier");
      for (const n of desc) {
        if (!(n.maxX < minX || n.minX > maxX)) byLevel[targetLevel]!.push(n);
      }
    }
  }
  const initNodes: MomentNode[] = [];
  for (let l = 0; l < counts.length; l++) initNodes.push(...byLevel[l]!);
  initNodes.sort((a, b) => a.index * fanout ** a.level - b.index * fanout ** b.level);

  // PLPRIPM2 Phase 1: pruned exact visible_y_range (skipped completely when explicit yBounds is set).
  let knownYRange: [number, number] | null = null;
  if (!query.yBounds) {
    let minY = Infinity;
    let maxY = -Infinity;
    const stradByLevel: MomentNode[][] = counts.map(() => []);
    for (const node of initNodes) {
      if (minX <= node.minX && node.maxX <= maxX) {
        minY = Math.min(minY, node.minY);
        maxY = Math.max(maxY, node.maxY);
      } else {
        stradByLevel[node.level]!.push(node);
      }
    }
    for (let curLevel = rootLevel; curLevel >= 1; curLevel--) {
      const active = stradByLevel[curLevel]!.filter((c) => c.minY < minY || c.maxY > maxY);
      if (active.length === 0) continue;
      const drop = Math.min(curLevel, MAX_BOUNDARY_JUMP);
      const targetLevel = curLevel - drop;
      const desc = fetchDescendants(curLevel, targetLevel, active.map((s) => s.index), "y-range");
      for (const node of desc) {
        if (node.maxX < minX || node.minX > maxX) continue;
        if (minX <= node.minX && node.maxX <= maxX) {
          minY = Math.min(minY, node.minY);
          maxY = Math.max(maxY, node.maxY);
        } else {
          stradByLevel[targetLevel]!.push(node);
        }
      }
    }
    const leafStrad = stradByLevel[0]!.filter((c) => c.minY < minY || c.maxY > maxY);
    if (leafStrad.length > 0) {
      const missing = leafStrad.map((s) => s.index).filter((l) => !cache.leaves.has(l));
      if (missing.length > 0) {
        leafRequests(missing, "y-range", round);
        round++;
      }
      for (const leaf of leafStrad) {
        for (let i = leaf.start; i < leaf.end; i++) {
          const x = model.data.xs[i]!;
          if (x >= minX && x <= maxX) {
            minY = Math.min(minY, model.data.ys[i]!);
            maxY = Math.max(maxY, model.data.ys[i]!);
          }
        }
      }
    }
    knownYRange = minY <= maxY ? [minY, maxY] : null;
    if (!knownYRange) {
      return { query, yRange: null, effectiveY: [0, 1], xFactor, yFactor: 1, trace, items, groups: [], requests, rounds: round, wireBytes: sum(requests), kernelCount: 0, cacheHits };
    }
  }

  const effectiveY = query.yBounds ?? knownYRange ?? [0, 1];
  const yFactor = query.screenH / Math.max(effectiveY[1] - effectiveY[0], 1e-12) / sigma;

  // PLPRIPM2 Phase 2: frontier with sparse per-block plane directory check.
  const decide = (node: MomentNode, level: number, fullyInside: boolean): { entry: TraceEntry; slot: number } => {
    const w = (node.maxX - node.minX) * xFactor;
    const span = (node.maxY - node.minY) * yFactor;
    const kmax = model.kmax[level] ?? 0;
    const detail: DecisionDetail = { w, span, hMax: NaN, kNeeded: null, kmax, fullyInside };
    if (!fullyInside) return { entry: { node, decision: { kind: "expand", reason: "boundary" }, detail }, slot: -1 };
    if (offscreenY(node, query, yFactor)) return { entry: { node, decision: { kind: "resolve", reason: "offscreen" }, detail }, slot: -1 };
    if (w * w + span * span < threshold * threshold) {
      return { entry: { node, decision: { kind: "resolve", reason: "fits" }, detail }, slot: -1 };
    }
    if (w <= MAX_PLANE_BLOCK_WIDTH * threshold) {
      const nd = neededDepth(node, w, yFactor, threshold);
      detail.hMax = nd.hMax;
      detail.kNeeded = nd.k;
      const ks = model.planeDepths[level]!;
      const dirs = model.blockDirs[level]![node.index]!;
      for (let slot = 0; slot < ks.length; slot++) {
        if (ks[slot]! >= nd.k && dirs[slot]!.cellCount > 0) {
          return { entry: { node, decision: { kind: "plane", k: ks[slot]! }, detail }, slot };
        }
      }
      return { entry: { node, decision: { kind: "expand", reason: "k-exceeds-kmax" }, detail }, slot: -1 };
    }
    return { entry: { node, decision: { kind: "expand", reason: "too-wide" }, detail }, slot: -1 };
  };

  const done: Array<{ orderKey: number; kind: "resolved" | "plane" | "leaf"; entry: TraceEntry; slot: number }> = [];
  const expandByLevel: MomentNode[][] = counts.map(() => []);
  const classify = (node: MomentNode, level: number) => {
    if (node.maxX < minX || node.minX > maxX) return;
    const fullyInside = minX <= node.minX && node.maxX <= maxX;
    const { entry, slot } = decide(node, level, fullyInside);
    trace[level]!.push(entry);
    const orderKey = node.index * fanout ** level;
    const d = entry.decision;
    if (d.kind === "resolve") done.push({ orderKey, kind: "resolved", entry, slot: -1 });
    else if (d.kind === "plane") done.push({ orderKey, kind: "plane", entry, slot });
    else if (level > 0) expandByLevel[level]!.push(node);
    else done.push({ orderKey, kind: "leaf", entry, slot: -1 });
  };

  for (const node of initNodes) classify(node, node.level);
  for (let curLevel = rootLevel; curLevel >= 1; curLevel--) {
    const expanding = expandByLevel[curLevel]!;
    expandByLevel[curLevel] = [];
    if (expanding.length === 0) continue;

    const onlyStraddling = expanding.every((n) => !(minX <= n.minX && n.maxX <= maxX));
    const lowerEmpty = expandByLevel.slice(1, curLevel).every((a) => a.length === 0);
    if (onlyStraddling && lowerEmpty && curLevel <= MAX_BOUNDARY_JUMP && curLevel > 1) {
      const maxSpanSigma = Math.max(...expanding.map((n) => (n.maxY - n.minY) * yFactor), 0);
      const leafSpanBound = 2 << curLevel;
      if (maxSpanSigma <= leafSpanBound) {
        const desc = fetchDescendants(curLevel, 0, expanding.map((e) => e.index), "frontier");
        for (const leaf of desc) classify(leaf, 0);
      } else {
        // Concurrent 1-RTT fetch of `0..curLevel` boundary subtree slices.
        let fetchedAny = false;
        for (let lvl = 0; lvl < curLevel; lvl++) {
          const step = curLevel - lvl;
          const pagesPerParent = fanout ** (step - 1);
          const totalPages = Math.ceil(counts[lvl]! / fanout);
          const needed: number[] = [];
          for (const e of expanding) {
            const lo = e.index * pagesPerParent;
            const hi = Math.min((e.index + 1) * pagesPerParent, totalPages);
            for (let p = lo; p < hi; p++) needed.push(p);
          }
          if (fetchPages(lvl, needed, "frontier", round)) fetchedAny = true;
        }
        if (fetchedAny) round++;
        const walkSub = (parent: MomentNode) => {
          for (const child of children(parent.level, parent.index)) {
            if (child.maxX < minX || child.minX > maxX) continue;
            const fullyInside = minX <= child.minX && child.maxX <= maxX;
            const { entry, slot } = decide(child, child.level, fullyInside);
            trace[child.level]!.push(entry);
            const orderKey = child.index * fanout ** child.level;
            const d = entry.decision;
            if (d.kind === "resolve") done.push({ orderKey, kind: "resolved", entry, slot: -1 });
            else if (d.kind === "plane") done.push({ orderKey, kind: "plane", entry, slot });
            else if (child.level === 0) done.push({ orderKey, kind: "leaf", entry, slot: -1 });
            else walkSub(child);
          }
        };
        for (const parent of expanding) walkSub(parent);
      }
      break;
    }

    const childLevel = curLevel - 1;
    if (fetchPages(childLevel, expanding.map((e) => e.index), "frontier")) round++;
    for (const parent of expanding) {
      for (const child of children(curLevel, parent.index)) classify(child, childLevel);
    }
  }

  // PLPRIPM2 Phase 3: concurrent raw leaves + coalesced sparse `planes.bin` spans (with cross-k cache reuse).
  const missingLeaves: number[] = [];
  const missingPlaneSpans: Array<[number, number]> = [];
  let kernelCount = 0;
  for (const { kind, entry, slot } of done.sort((a, b) => a.orderKey - b.orderKey)) {
    const node = entry.node;
    if (kind === "resolved") {
      const d = entry.decision as Extract<Decision, { kind: "resolve" }>;
      items.push({ kind: "resolved", node, reason: d.reason });
    } else if (kind === "leaf") {
      items.push({ kind: "leaf", node });
      if (cache.leaves.has(node.index)) cacheHits++;
      else missingLeaves.push(node.index);
    } else {
      const ks = model.planeDepths[node.level]!;
      let actualSlot = slot;
      let hit = false;
      for (let trySlot = slot; trySlot < ks.length; trySlot++) {
        if (cache.planeBlocks.has(`${node.level}:${trySlot}:${node.index}`)) {
          actualSlot = trySlot;
          hit = true;
          cacheHits++;
          break;
        }
      }
      const actualK = ks[actualSlot]!;
      const cells = planeCells(model, node, actualK);
      items.push({ kind: "plane", node, k: actualK, cells });
      kernelCount += countMergedPlaneKernels(node, cells, query, xFactor, yFactor, threshold);
      if (!hit) {
        const dir = model.blockDirs[node.level]![node.index]![slot]!;
        const byteOff = model.planeByteOffsets[node.level]![slot]! + dir.cellOffset * CELL_BYTES;
        const byteLen = dir.cellCount * CELL_BYTES;
        if (byteLen > 0) missingPlaneSpans.push([byteOff, byteLen]);
        cache.planeBlocks.add(`${node.level}:${slot}:${node.index}`);
      }
    }
  }
  if (missingLeaves.length > 0 || missingPlaneSpans.length > 0) {
    leafRequests(missingLeaves, "planes+leaves", round);
    for (const [startByte, endByte] of coalesceByteSpans(missingPlaneSpans, PLANE_MAX_GAP_BYTES, PLANE_MAX_RUN_BYTES)) {
      requests.push({
        key: "planes.bin",
        offset: startByte,
        len: endByte - startByte,
        round,
        phase: "planes+leaves",
      });
    }
    round++;
  }

  // Derive exact yRange from frontier + clipped leaves if Phase 1 was skipped (explicit yBounds).
  let yRange = knownYRange;
  if (!yRange) {
    let lo = Infinity;
    let hi = -Infinity;
    for (const item of items) {
      if (item.kind === "resolved" || item.kind === "plane") {
        lo = Math.min(lo, item.node.minY);
        hi = Math.max(hi, item.node.maxY);
      } else {
        for (let i = item.node.start; i < item.node.end; i++) {
          const x = model.data.xs[i]!;
          if (x >= minX && x <= maxX) {
            lo = Math.min(lo, model.data.ys[i]!);
            hi = Math.max(hi, model.data.ys[i]!);
          }
        }
      }
    }
    yRange = lo <= hi ? [lo, hi] : null;
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
