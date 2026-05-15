export type DownsampleStrategy = "merge" | "lttb";

export interface DownsampleArgs {
  /** Packed [x, y, weight] triples, already filtered to the visible range. */
  points: Float32Array;
  strategy: DownsampleStrategy;
  /** Maps a data-space x to screen pixels. */
  toSX: (x: number) => number;
  /** Maps a data-space y to screen pixels. */
  toSY: (y: number) => number;
  /**
   * merge: distance threshold in px below which adjacent points collapse.
   * lttb:  desired number of output points.
   */
  threshold?: number;
  mergeThreshold?: number; // only for merge strategy, overrides threshold
}

export function downsample(args: DownsampleArgs): Float32Array {
  switch (args.strategy) {
    case "merge":
      return mergeDownsample(args);
    case "lttb":
      return lttbDownsample(args);
  }
}

// ---------------------------------------------------------------------------
// Merge strategy
// ---------------------------------------------------------------------------

const MERGE_PASSES = 3;

function mergePass(
  pts: Float32Array,
  toSX: (x: number) => number,
  toSY: (y: number) => number,
  thresholdPx: number,
): Float32Array {
  const out = new Float32Array(pts.length);
  let count = 0;
  const n = pts.length / 3;
  const t2 = thresholdPx * thresholdPx;

  for (let i = 0; i < n; i++) {
    const xi = pts[i * 3]!;
    const yi = pts[i * 3 + 1]!;
    const wi = pts[i * 3 + 2]!;
    const sx = toSX(xi);
    const sy = toSY(yi);

    if (count > 0) {
      const j = count - 1;
      const dx = sx - toSX(out[j * 3]!);
      const dy = sy - toSY(out[j * 3 + 1]!);
      if (dx * dx + dy * dy < t2) {
        const wj = out[j * 3 + 2]!;
        const wSum = wj + wi;
        out[j * 3] = (out[j * 3]! * wj + xi * wi) / wSum;
        out[j * 3 + 1] = (out[j * 3 + 1]! * wj + yi * wi) / wSum;
        out[j * 3 + 2] = wSum;
        continue;
      }
    }

    out[count * 3] = xi;
    out[count * 3 + 1] = yi;
    out[count * 3 + 2] = wi;
    count++;
  }

  return out.slice(0, count * 3);
}

function mergeDownsample({
  points,
  toSX,
  toSY,
  mergeThreshold,
}: DownsampleArgs): Float32Array {
  let result = points;
  for (let pass = 0; pass < MERGE_PASSES; pass++) {
    result = mergePass(result, toSX, toSY, mergeThreshold ?? 1);
  }
  return result;
}

// ---------------------------------------------------------------------------
// LTTB (Largest-Triangle-Three-Buckets) strategy
// ---------------------------------------------------------------------------

function triangleArea(
  ax: number,
  ay: number,
  bx: number,
  by: number,
  cx: number,
  cy: number,
): number {
  return Math.abs((ax - cx) * (by - ay) - (ax - bx) * (cy - ay)) * 0.5;
}

function lttbDownsample({
  points,
  toSX,
  toSY,
  threshold = 10,
}: DownsampleArgs): Float32Array {
  const n = points.length / 3;
  const targetCount = Math.round(threshold);
  if (targetCount >= n) return points;

  const out = new Float32Array(targetCount * 3);
  let outIdx = 0;

  const write = (i: number) => {
    out[outIdx * 3] = points[i * 3]!;
    out[outIdx * 3 + 1] = points[i * 3 + 1]!;
    out[outIdx * 3 + 2] = points[i * 3 + 2]!;
    outIdx++;
  };

  // always keep first and last
  write(0);

  const bucketCount = targetCount - 2;
  const bucketSize = (n - 2) / bucketCount;

  let prevSelected = 0;

  for (let b = 0; b < bucketCount; b++) {
    const bucketStart = Math.floor((b + 1) * bucketSize) + 1;
    const bucketEnd = Math.min(Math.floor((b + 2) * bucketSize) + 1, n - 1);

    // anchor A: previously selected point in screen space
    const ax = toSX(points[prevSelected * 3]!);
    const ay = toSY(points[prevSelected * 3 + 1]!);

    // anchor C: average of the next bucket in screen space (approximate)
    const nextStart = Math.floor((b + 2) * bucketSize) + 1;
    const nextEnd = Math.min(Math.floor((b + 3) * bucketSize) + 1, n - 1);
    let cx = 0,
      cy = 0,
      cCount = 0;
    for (let k = nextStart; k < nextEnd; k++) {
      cx += toSX(points[k * 3]!);
      cy += toSY(points[k * 3 + 1]!);
      cCount++;
    }
    if (cCount === 0) {
      cx = ax;
      cy = ay;
    } else {
      cx /= cCount;
      cy /= cCount;
    }

    // pick point in current bucket with largest triangle area
    let bestIdx = bucketStart;
    let bestArea = -1;
    for (let k = bucketStart; k < bucketEnd; k++) {
      const bx = toSX(points[k * 3]!);
      const by = toSY(points[k * 3 + 1]!);
      const area = triangleArea(ax, ay, bx, by, cx, cy);
      if (area > bestArea) {
        bestArea = area;
        bestIdx = k;
      }
    }

    write(bestIdx);
    prevSelected = bestIdx;
  }

  write(n - 1);

  return out;
}

