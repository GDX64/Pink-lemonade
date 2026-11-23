import {
  type ClientCache,
  type Dataset,
  type DecisionDetail,
  type FrontierItem,
  type MomentNode,
  type QueryResult,
  type RangeRequest,
  type StoredObject,
  type TraceEntry,
  type ViewQuery,
  POINT_BYTES,
  assembleGroups,
  emptyCache,
  mergeNodes,
  offscreenY,
  summarize,
} from "./ripmap-model";

/**
 * In-browser model of `rust/progressive-quadtree` (`moment_quadtree`, `PLPQUAD1`).
 *
 * Mirrors the Rust builder (midpoint quadrant splits over the data bbox, empty quadrants
 * dropped, leaves of ≤ `leafPoints` points sorted by x, points stored in Z order) and the
 * client (pruned exact y-range descent, depth-by-depth frontier with resolve-or-expand,
 * coalesced child-page and leaf reads), and returns the same `QueryResult` shape as the
 * ripmap / x-tree model so the explorer page can draw all three.
 */

/** 64-byte moment node + `first: u32` + `count: u32`. */
export const RECORD_BYTES = 72;
const HEADER_BYTES = 32;
const DEPTH_INFO_BYTES = 16;
const MAX_GAP_RECORDS = Math.floor((16 * 1024) / RECORD_BYTES);
const MAX_RUN_RECORDS = Math.floor((256 * 1024) / RECORD_BYTES);
const MAX_GAP_POINTS = 256;
const MAX_RUN_POINTS = 8 * 1024;
export const NODES_KEY = "nodes.bin";

export interface QuadParams {
  leafPoints: number;
  rootCap: number;
  maxDepth: number;
}

/** `QuadtreeStrategy::default()`. */
export const QUAD_PARAMS: QuadParams = {
  leafPoints: 32,
  rootCap: 1024,
  maxDepth: 40,
};

/** A quadtree node: `level` = depth, `index` = position in its depth, `[start, end)` in Z order. */
export interface QuadNode extends MomentNode {
  /** The node's quadrant (region) — the classic grid cell, not the tight bbox. */
  rx0: number;
  rx1: number;
  ry0: number;
  ry1: number;
  isLeaf: boolean;
  /** Children `[firstChild, firstChild + childCount)` in depth `level + 1`. */
  firstChild: number;
  childCount: number;
  /** Z-order path from the root (one char per depth), for frontier ordering. */
  path: string;
}

export interface DepthInfo {
  count: number;
  /** Inlined in `manifest.bin` (top depths) or stored in `nodes.bin` at `offset`. */
  inline: boolean;
  offset: number;
}

export interface QuadModel {
  strategy: "quadtree";
  /** Points permuted into quadtree (Z) order: every subtree is a contiguous run. */
  data: Dataset;
  /** The original x-sorted dataset (for drawing raw points and window statistics). */
  sorted: Dataset;
  params: QuadParams;
  /** Node count per depth (`levelCounts[0]` = 1 root). */
  levelCounts: number[];
  levels: QuadNode[][];
  depthInfo: DepthInfo[];
  inlineDepths: number;
  objects: StoredObject[];
  totalBytes: number;
  rawBytes: number;
}

interface Pending {
  start: number;
  end: number;
  rx0: number;
  rx1: number;
  ry0: number;
  ry1: number;
  path: string;
}

export function buildQuadtree(sorted: Dataset, inParams: QuadParams): QuadModel {
  const params = { ...inParams, leafPoints: Math.max(inParams.leafPoints, 1), maxDepth: Math.min(inParams.maxDepth, 62) };
  const n = sorted.xs.length;
  const idx = new Uint32Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  const tmp = new Uint32Array(n);
  const { xs, ys } = sorted;

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < n; i++) {
    minX = Math.min(minX, xs[i]!);
    maxX = Math.max(maxX, xs[i]!);
    minY = Math.min(minY, ys[i]!);
    maxY = Math.max(maxY, ys[i]!);
  }

  // Structure, breadth-first so the children of one node are adjacent in the next depth.
  type Proto = Pending & { firstChild: number; childCount: number };
  const protos: Proto[][] = [];
  let current: Pending[] = n > 0 ? [{ start: 0, end: n, rx0: minX, rx1: maxX, ry0: minY, ry1: maxY, path: "" }] : [];
  let depth = 0;
  while (current.length > 0) {
    const level: Proto[] = [];
    const next: Pending[] = [];
    for (const p of current) {
      const size = p.end - p.start;
      let coincide = true;
      const x0 = xs[idx[p.start]!]!;
      const y0 = ys[idx[p.start]!]!;
      for (let i = p.start + 1; i < p.end && coincide; i++) {
        if (xs[idx[i]!] !== x0 || ys[idx[i]!] !== y0) coincide = false;
      }
      if (size <= params.leafPoints || depth >= params.maxDepth || coincide) {
        // Counting-sort partitions are stable and the input is x-sorted: leaves stay x-sorted.
        level.push({ ...p, firstChild: 0, childCount: 0 });
        continue;
      }
      const mx = p.rx0 + (p.rx1 - p.rx0) * 0.5;
      const my = p.ry0 + (p.ry1 - p.ry0) * 0.5;
      const quad = (i: number) => (xs[i]! >= mx ? 1 : 0) | (ys[i]! >= my ? 2 : 0);
      const counts = [0, 0, 0, 0];
      for (let i = p.start; i < p.end; i++) counts[quad(idx[i]!)]!++;
      const offs = [p.start, 0, 0, 0];
      for (let q = 1; q < 4; q++) offs[q] = offs[q - 1]! + counts[q - 1]!;
      const cursor = [...offs];
      for (let i = p.start; i < p.end; i++) {
        const v = idx[i]!;
        tmp[cursor[quad(v)]!++] = v;
      }
      idx.set(tmp.subarray(p.start, p.end), p.start);
      const first = next.length;
      for (let q = 0; q < 4; q++) {
        if (counts[q] === 0) continue;
        next.push({
          start: offs[q]!,
          end: offs[q]! + counts[q]!,
          rx0: q & 1 ? mx : p.rx0,
          rx1: q & 1 ? p.rx1 : mx,
          ry0: q & 2 ? my : p.ry0,
          ry1: q & 2 ? p.ry1 : my,
          path: p.path + String.fromCharCode(48 + (next.length - first)),
        });
      }
      level.push({ ...p, firstChild: first, childCount: next.length - first });
    }
    protos.push(level);
    current = next;
    depth++;
  }

  // Points in Z order.
  const data: Dataset = {
    shape: sorted.shape,
    xs: new Float64Array(n),
    ys: new Float64Array(n),
    ws: new Float64Array(n),
  };
  for (let i = 0; i < n; i++) {
    const j = idx[i]!;
    data.xs[i] = sorted.xs[j]!;
    data.ys[i] = sorted.ys[j]!;
    data.ws[i] = sorted.ws[j]!;
  }

  // Moments rolled up from the deepest depth.
  const levels: QuadNode[][] = protos.map(() => []);
  for (let d = protos.length - 1; d >= 0; d--) {
    levels[d] = protos[d]!.map((p, i) => {
      const base =
        p.childCount === 0
          ? summarize(data, d, i, p.start, p.end)
          : mergeNodes(levels[d + 1]!.slice(p.firstChild, p.firstChild + p.childCount), d, i);
      return {
        ...base,
        start: p.start,
        end: p.end,
        rx0: p.rx0,
        rx1: p.rx1,
        ry0: p.ry0,
        ry1: p.ry1,
        isLeaf: p.childCount === 0,
        firstChild: p.firstChild,
        childCount: p.childCount,
        path: p.path,
      };
    });
  }

  const levelCounts = levels.map((l) => l.length);
  const inlineDepths = countInlineDepths(levelCounts, params.rootCap);
  const depthInfo: DepthInfo[] = [];
  let nodesBytes = 0;
  for (let d = 0; d < levelCounts.length; d++) {
    if (d < inlineDepths) {
      depthInfo.push({ count: levelCounts[d]!, inline: true, offset: 0 });
      continue;
    }
    depthInfo.push({ count: levelCounts[d]!, inline: false, offset: nodesBytes });
    nodesBytes += levelCounts[d]! * RECORD_BYTES;
  }
  const inlineNodes = levelCounts.slice(0, inlineDepths).reduce((s, c) => s + c, 0);
  const objects: StoredObject[] = [
    { key: "manifest.bin", bytes: HEADER_BYTES + levelCounts.length * DEPTH_INFO_BYTES + inlineNodes * RECORD_BYTES, kind: "manifest" },
  ];
  if (nodesBytes > 0) objects.push({ key: NODES_KEY, bytes: nodesBytes, kind: "level" });
  if (n > 0) objects.push({ key: "points.bin", bytes: n * POINT_BYTES, kind: "points" });
  const totalBytes = objects.reduce((s, o) => s + o.bytes, 0);
  return { strategy: "quadtree", data, sorted, params, levelCounts, levels, depthInfo, inlineDepths, objects, totalBytes, rawBytes: n * POINT_BYTES };
}

/** `inline_depths` from `progressive-quadtree/src/builder.rs`: top depths totalling ≤ `rootCap`. */
export function countInlineDepths(counts: number[], rootCap: number): number {
  let inline = 0;
  let inlined = 0;
  for (const c of counts) {
    if (inline > 0 && inlined + c > rootCap) break;
    inlined += c;
    inline++;
  }
  return inline;
}

/** `coalesce_spans` from `progressive-common/src/fetch.rs`: `(start, len)` → `[start, end)` reads. */
export function coalesceSpans(spans: Array<[number, number]>, maxGap: number, maxRun: number): Array<[number, number]> {
  const sorted = spans.filter(([, len]) => len > 0).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (sorted.length === 0) return [];
  const reads: Array<[number, number]> = [];
  let [start, end] = [sorted[0]![0], sorted[0]![0] + sorted[0]![1]];
  for (const [s, len] of sorted.slice(1)) {
    const e = s + len;
    if (s <= end + maxGap && Math.max(e, end) - start <= maxRun) end = Math.max(end, e);
    else {
      reads.push([start, end]);
      [start, end] = [s, e];
    }
  }
  reads.push([start, end]);
  return reads;
}

export function runQuadQuery(model: QuadModel, query: ViewQuery, cache: ClientCache = emptyCache()): QueryResult {
  const { levels, depthInfo } = model;
  const requests: RangeRequest[] = [];
  let round = 0;
  let cacheHits = 0;

  if (!cache.opened) {
    requests.push({ key: "manifest.bin", offset: 0, len: model.objects[0]!.bytes, round, phase: "open" });
    round++;
    cache.opened = true;
  }

  // Child pages of `parents` (all at depth `childDepth - 1`): free when inlined, else one round.
  const fetchChildren = (childDepth: number, parents: QuadNode[], phase: RangeRequest["phase"]) => {
    const info = depthInfo[childDepth];
    if (!info || info.inline) return;
    const missing: Array<[number, number]> = [];
    for (const p of parents) {
      if (cache.treePages.has(`${childDepth}:${p.firstChild}`)) cacheHits++;
      else missing.push([p.firstChild, p.childCount]);
    }
    if (missing.length === 0) return;
    for (const [start, end] of coalesceSpans(missing, MAX_GAP_RECORDS, MAX_RUN_RECORDS)) {
      requests.push({
        key: NODES_KEY,
        offset: info.offset + start * RECORD_BYTES,
        len: (end - start) * RECORD_BYTES,
        round,
        phase,
      });
    }
    for (const [first] of missing) cache.treePages.add(`${childDepth}:${first}`);
    round++;
  };
  const children = (p: QuadNode) => levels[p.level + 1]!.slice(p.firstChild, p.firstChild + p.childCount);

  const fetchLeaves = (leaves: QuadNode[], phase: RangeRequest["phase"]) => {
    const missing: Array<[number, number]> = [];
    for (const l of leaves) {
      if (cache.leaves.has(l.start)) cacheHits++;
      else missing.push([l.start, l.end - l.start]);
    }
    if (missing.length === 0) return;
    for (const [start, end] of coalesceSpans(missing, MAX_GAP_POINTS, MAX_RUN_POINTS)) {
      requests.push({ key: "points.bin", offset: start * POINT_BYTES, len: (end - start) * POINT_BYTES, round, phase });
    }
    for (const [start] of missing) cache.leaves.add(start);
    round++;
  };

  // Phase 1: pruned exact visible y-range.
  const { minX, maxX } = query;
  let minY = Infinity;
  let maxY = -Infinity;
  const root = levels[0]?.[0];
  let level: QuadNode[] = root ? [root] : [];
  const straddlingLeaves: QuadNode[] = [];
  while (level.length > 0) {
    let straddling: QuadNode[] = [];
    for (const node of level) {
      if (node.maxX < minX || node.minX > maxX) continue;
      if (minX <= node.minX && node.maxX <= maxX) {
        minY = Math.min(minY, node.minY);
        maxY = Math.max(maxY, node.maxY);
      } else straddling.push(node);
    }
    straddling = straddling.filter((s) => s.minY < minY || s.maxY > maxY);
    const inner = straddling.filter((s) => !s.isLeaf);
    straddlingLeaves.push(...straddling.filter((s) => s.isLeaf));
    if (inner.length === 0) break;
    fetchChildren(inner[0]!.level + 1, inner, "y-range");
    level = inner.flatMap(children);
  }
  const leavesForY = straddlingLeaves.filter((s) => s.minY < minY || s.maxY > maxY);
  fetchLeaves(leavesForY, "y-range");
  for (const leaf of leavesForY) {
    for (let i = leaf.start; i < leaf.end; i++) {
      const x = model.data.xs[i]!;
      if (x >= minX && x <= maxX) {
        minY = Math.min(minY, model.data.ys[i]!);
        maxY = Math.max(maxY, model.data.ys[i]!);
      }
    }
  }
  const yRange: [number, number] | null = minY <= maxY ? [minY, maxY] : null;
  const effectiveY = query.yBounds ?? yRange ?? [0, 1];
  const sigma = Math.max(query.sigmaPx, 1e-12);
  const xFactor = query.screenW / Math.max(maxX - minX, 1e-12) / sigma;
  const yFactor = query.screenH / Math.max(effectiveY[1] - effectiveY[0], 1e-12) / sigma;
  const threshold = Math.max(query.mergeThreshold, 1e-12);

  const trace: TraceEntry[][] = levels.map(() => []);
  const items: FrontierItem[] = [];
  if (!yRange) {
    return { query, yRange, effectiveY, xFactor, yFactor, trace, items, groups: [], requests, rounds: round, wireBytes: total(requests), kernelCount: 0, cacheHits };
  }

  // Phase 2: depth-by-depth frontier with the standard resolve-or-expand decision.
  const decide = (node: QuadNode, fullyInside: boolean): TraceEntry => {
    const w = (node.maxX - node.minX) * xFactor;
    const span = (node.maxY - node.minY) * yFactor;
    const detail: DecisionDetail = { w, span, hMax: NaN, kNeeded: null, kmax: 0, fullyInside };
    if (!fullyInside) return { node, decision: { kind: "expand", reason: "boundary" }, detail };
    if (offscreenY(node, query, yFactor)) return { node, decision: { kind: "resolve", reason: "offscreen" }, detail };
    if (w * w + span * span < threshold * threshold) return { node, decision: { kind: "resolve", reason: "fits" }, detail };
    return { node, decision: { kind: "expand", reason: "too-big" }, detail };
  };

  const done: Array<{ path: string; item: FrontierItem }> = [];
  const leafItems: QuadNode[] = [];
  level = root ? [root] : [];
  while (level.length > 0) {
    const expand: QuadNode[] = [];
    for (const node of level) {
      if (node.maxX < minX || node.minX > maxX) continue;
      const entry = decide(node, minX <= node.minX && node.maxX <= maxX);
      trace[node.level]!.push(entry);
      const d = entry.decision;
      if (d.kind === "resolve") done.push({ path: node.path, item: { kind: "resolved", node, reason: d.reason } });
      else if (node.isLeaf) {
        done.push({ path: node.path, item: { kind: "leaf", node } });
        leafItems.push(node);
      } else expand.push(node);
    }
    if (expand.length === 0) break;
    fetchChildren(expand[0]!.level + 1, expand, "frontier");
    level = expand.flatMap(children);
  }

  // Phase 3: raw leaves.
  fetchLeaves(leafItems, "planes+leaves");
  done.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0));
  items.push(...done.map((d) => d.item));
  const groups = assembleGroups(model, items, query, xFactor, yFactor, threshold);

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
    wireBytes: total(requests),
    kernelCount: groups.length,
    cacheHits,
  };
}

function total(requests: RangeRequest[]) {
  return requests.reduce((s, r) => s + r.len, 0);
}
