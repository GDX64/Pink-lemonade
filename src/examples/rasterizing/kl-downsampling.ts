import initWasm, { KlDownsampler } from "../../../pkg/pink_lemonade_wasm";
import { lowerBound, upperBound } from "./view-manager";

/**
 * Runnalls' Kullback-Leibler based Gaussian mixture reduction.
 *
 * A. R. Runnalls, "Kullback-Leibler Approach to Gaussian Mixture Reduction",
 * IEEE Trans. Aerospace and Electronic Systems, 43(3), 2007.
 *
 * Components are merged greedily in pairs. At each iteration the pair whose
 * merge minimises an upper bound on the KL discrimination between the mixture
 * before and after merging is selected:
 *
 *   B(i,j) = 1/2 [ (wi+wj) log det(Pij) - wi log det(Pi) - wj log det(Pj) ]
 *
 * where (wij, muij, Pij) is the moment-preserving merge of i and j. Unlike the
 * screen-space one-pass merge, this criterion depends on the component
 * covariances, is invariant to affine rescaling of the state, and is
 * independent of input ordering. It is intended as a quality reference, not as
 * a per-frame competitor: it costs O(N^2) distance evaluations to build and
 * O(N) per merge to maintain.
 */

/** Screen-space mixture in structure-of-arrays form. P is symmetric: p01 = p10. */
export interface Mixture {
  /** Number of components. */
  count: number;
  x: Float64Array;
  y: Float64Array;
  /** Total mass (not amplitude). */
  w: Float64Array;
  p00: Float64Array;
  p01: Float64Array;
  p11: Float64Array;
}

export function createMixture(capacity: number): Mixture {
  return {
    count: 0,
    x: new Float64Array(capacity),
    y: new Float64Array(capacity),
    w: new Float64Array(capacity),
    p00: new Float64Array(capacity),
    p01: new Float64Array(capacity),
    p11: new Float64Array(capacity),
  };
}

/**
 * Moment-preserving merge of components i and j, written into `out` at `oi`.
 * Preserves total mass, mean and covariance of the two-component sub-mixture.
 */
export function mergePair(
  m: Mixture,
  i: number,
  j: number,
  out: Mixture,
  oi: number,
): void {
  const wi = m.w[i]!;
  const wj = m.w[j]!;
  const w = wi + wj;
  const a = wi / w;
  const b = wj / w;
  const dx = m.x[i]! - m.x[j]!;
  const dy = m.y[i]! - m.y[j]!;
  // Spread term: (wi*wj/w^2) * delta delta^T
  const k = a * b;

  out.w[oi] = w;
  out.x[oi] = a * m.x[i]! + b * m.x[j]!;
  out.y[oi] = a * m.y[i]! + b * m.y[j]!;
  out.p00[oi] = a * m.p00[i]! + b * m.p00[j]! + k * dx * dx;
  out.p01[oi] = a * m.p01[i]! + b * m.p01[j]! + k * dx * dy;
  out.p11[oi] = a * m.p11[i]! + b * m.p11[j]! + k * dy * dy;
}

/** log det of a symmetric 2x2, guarded against non-positive determinants. */
function logDet(p00: number, p01: number, p11: number): number {
  const det = p00 * p11 - p01 * p01;
  return Math.log(Math.max(det, 1e-300));
}

/**
 * Runnalls' merging cost B(i,j): the increase in the KL upper bound caused by
 * replacing components i and j with their moment-preserving merge. Always >= 0,
 * and exactly 0 when the two components are identical.
 */
export function runnallsCost(m: Mixture, i: number, j: number): number {
  const wi = m.w[i]!;
  const wj = m.w[j]!;
  const w = wi + wj;
  if (w <= 0) return 0;
  const a = wi / w;
  const b = wj / w;
  const dx = m.x[i]! - m.x[j]!;
  const dy = m.y[i]! - m.y[j]!;
  const k = a * b;

  const p00 = a * m.p00[i]! + b * m.p00[j]! + k * dx * dx;
  const p01 = a * m.p01[i]! + b * m.p01[j]! + k * dx * dy;
  const p11 = a * m.p11[i]! + b * m.p11[j]! + k * dy * dy;

  const li = logDet(m.p00[i]!, m.p01[i]!, m.p11[i]!);
  const lj = logDet(m.p00[j]!, m.p01[j]!, m.p11[j]!);

  return 0.5 * (w * logDet(p00, p01, p11) - wi * li - wj * lj);
}

export interface ReduceOptions {
  /** Stop once the mixture has at most this many components. */
  targetCount?: number;
  /** Stop once the cheapest available merge costs more than this. */
  klThreshold?: number;
}

export interface ReduceResult {
  mixture: Mixture;
  /** Cost of the last merge performed, or 0 if none. */
  lastCost: number;
  /** Number of merges performed. */
  merges: number;
}

/** Copy of `src`, sized exactly to its component count. */
function cloneMixture(src: Mixture): Mixture {
  const out = createMixture(src.count);
  out.count = src.count;
  out.x.set(src.x.subarray(0, src.count));
  out.y.set(src.y.subarray(0, src.count));
  out.w.set(src.w.subarray(0, src.count));
  out.p00.set(src.p00.subarray(0, src.count));
  out.p01.set(src.p01.subarray(0, src.count));
  out.p11.set(src.p11.subarray(0, src.count));
  return out;
}

/**
 * Reduce a copy of `input` by greedy pairwise KL-optimal merging. Stops when
 * either criterion in `options` is met; with neither set the mixture is reduced
 * to a single component. `input` is left untouched.
 */
export function runnallsReduce(
  input: Mixture,
  options: ReduceOptions = {},
): ReduceResult {
  const targetCount = options.targetCount ?? 1;
  const klThreshold = options.klThreshold ?? Number.POSITIVE_INFINITY;

  const m = cloneMixture(input);
  const n = m.count;
  const alive = new Uint8Array(n).fill(1);
  const best = new Int32Array(n).fill(-1);
  const bestCost = new Float64Array(n).fill(Number.POSITIVE_INFINITY);
  let live = n;
  let merges = 0;
  let lastCost = 0;

  const recomputeBest = (i: number) => {
    let bi = -1;
    let bc = Number.POSITIVE_INFINITY;
    for (let j = 0; j < n; j++) {
      if (j === i || !alive[j]) continue;
      const c = runnallsCost(m, i, j);
      if (c < bc) {
        bc = c;
        bi = j;
      }
    }
    best[i] = bi;
    bestCost[i] = bc;
  };

  if (n > 1) {
    for (let i = 0; i < n; i++) recomputeBest(i);
  }

  while (live > targetCount && live > 1) {
    // Global minimum-cost pair.
    let a = -1;
    let ac = Number.POSITIVE_INFINITY;
    for (let i = 0; i < n; i++) {
      if (!alive[i]) continue;
      const cost = bestCost[i]!;
      if (cost < ac) {
        ac = cost;
        a = i;
      }
    }
    if (a < 0 || !Number.isFinite(ac)) break;
    if (ac > klThreshold) break;

    const b = best[a]!;
    mergePair(m, a, b, m, a);
    alive[b] = 0;
    live--;
    merges++;
    lastCost = ac;

    if (live <= 1) break;

    // `a` changed, so every cost involving `a` is stale.
    recomputeBest(a);
    for (let k = 0; k < n; k++) {
      if (!alive[k] || k === a) continue;
      const c = runnallsCost(m, k, a);
      if (c < bestCost[k]!) {
        bestCost[k] = c;
        best[k] = a;
      } else if (best[k] === a || best[k] === b) {
        // Previous partner is gone or got more expensive; rescan.
        recomputeBest(k);
      }
    }
  }

  const out = createMixture(live);
  let o = 0;
  for (let i = 0; i < n; i++) {
    if (!alive[i]) continue;
    out.x[o] = m.x[i]!;
    out.y[o] = m.y[i]!;
    out.w[o] = m.w[i]!;
    out.p00[o] = m.p00[i]!;
    out.p01[o] = m.p01[i]!;
    out.p11[o] = m.p11[i]!;
    o++;
  }
  out.count = live;

  return { mixture: out, lastCost, merges };
}

export interface KLMergeResult {
  gpuInstances: Float32Array;
  count: number;
}

/**
 * wasm-backed Runnalls reduction, mirroring `wasmMerge`. Use this rather than
 * the TypeScript class whenever the result is being timed against the wasm
 * merge, so both sides run under the same runtime.
 */
export async function wasmKlMerge() {
  await initWasm();
  return new KlDownsampler();
}

/**
 * Drop-in analogue of the wasm `Downsampler`, backed by Runnalls' reduction.
 *
 * `setMergeThreshold` sets the KL cost bound rather than a pixel radius --- the
 * two algorithms are controlled by different quantities --- and
 * `setTargetCount` caps the component budget, which is the knob used when
 * comparing both methods at an equal number of kernels.
 */
export class KLDownsampler {
  private viewMinX = 0;
  private viewMaxX = 1;
  private viewMinY = 0;
  private viewMaxY = 1;
  private screenW = 1;
  private screenH = 1;
  private sigmaSizePx = 1;
  private klThreshold = Number.POSITIVE_INFINITY;
  private targetCount = Number.POSITIVE_INFINITY;
  private dataF64: Float64Array = new Float64Array(0);

  setViewMinX(value: number): void {
    this.viewMinX = value;
  }
  setViewMaxX(value: number): void {
    this.viewMaxX = value;
  }
  setViewMinY(value: number): void {
    this.viewMinY = value;
  }
  setViewMaxY(value: number): void {
    this.viewMaxY = value;
  }
  setScreenW(value: number): void {
    this.screenW = value;
  }
  setScreenH(value: number): void {
    this.screenH = value;
  }
  setSigmaSizePx(value: number): void {
    this.sigmaSizePx = value;
  }
  /** KL cost bound; merging stops once the cheapest merge exceeds it. */
  setMergeThreshold(value: number): void {
    this.klThreshold = value;
  }
  /** Component budget; merging stops once the mixture is this small. */
  setTargetCount(value: number): void {
    this.targetCount = value;
  }
  setDataF64(data: Float64Array): void {
    this.dataF64 = data;
  }

  /** Present for parity with the wasm bindings; nothing to release. */
  free(): void {}

  mergePoints(): KLMergeResult {
    const empty = { gpuInstances: new Float32Array(0), count: 0 };
    if (this.dataF64.length < 3) return empty;

    const xDen = Math.max(this.viewMaxX - this.viewMinX, 1e-12);
    const yDen = Math.max(this.viewMaxY - this.viewMinY, 1e-12);
    const sigma = Math.max(this.sigmaSizePx, 1e-12);
    const xFactor = this.screenW / xDen / sigma;
    const yFactor = this.screenH / yDen / sigma;

    const startIdx = lowerBound(this.dataF64, this.viewMinX);
    const endIdx = upperBound(this.dataF64, this.viewMaxX);
    if (endIdx <= startIdx) return empty;

    // One unit-covariance kernel per visible sample, in screen space.
    const n = endIdx - startIdx;
    const m = createMixture(n);
    m.count = n;
    for (let i = 0; i < n; i++) {
      const base = (startIdx + i) * 3;
      m.x[i] = (this.dataF64[base]! - this.viewMinX) * xFactor;
      m.y[i] = (this.dataF64[base + 1]! - this.viewMinY) * yFactor;
      m.w[i] = this.dataF64[base + 2]!;
      m.p00[i] = 1;
      m.p01[i] = 0;
      m.p11[i] = 1;
    }

    const { mixture } = runnallsReduce(m, {
      targetCount: this.targetCount,
      klThreshold: this.klThreshold,
    });

    // Back to data space, with the amplitude convention used by the GPU path.
    const count = mixture.count;
    const gpuInstances = new Float32Array(count * 7);
    for (let i = 0; i < count; i++) {
      const p00 = mixture.p00[i]!;
      const p01 = mixture.p01[i]!;
      const p11 = mixture.p11[i]!;
      const det = Math.max(p00 * p11 - p01 * p01, 1e-12);
      const gi = i * 7;
      gpuInstances[gi] = this.viewMinX + mixture.x[i]! / xFactor;
      gpuInstances[gi + 1] = this.viewMinY + mixture.y[i]! / yFactor;
      gpuInstances[gi + 2] = mixture.w[i]! / (2 * Math.PI * Math.sqrt(det));
      gpuInstances[gi + 3] = p00;
      gpuInstances[gi + 4] = p01;
      gpuInstances[gi + 5] = p01;
      gpuInstances[gi + 6] = p11;
    }

    return { gpuInstances, count };
  }
}
