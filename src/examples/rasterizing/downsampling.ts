export type DownsampleStrategy = "merge" | "lttb" | "rdp";

export interface DownsampleArgs {
  /** Packed [x, y, weight] triples, already filtered to the visible range. */
  points: Float64Array;
  strategy: DownsampleStrategy;
  /** Maps a data-space x to screen pixels. */
  toSX: (x: number) => number;
  /** Maps a data-space y to screen pixels. */
  toSY: (y: number) => number;
  /**
   * merge: distance threshold in px below which adjacent points collapse.
   * lttb:  desired number of output points.
   * rdp:   use `epsilon` instead.
   */
  threshold?: number;
  mergeThreshold?: number; // only for merge strategy, overrides threshold
  /** rdp: perpendicular deviation tolerance in screen pixels. */
  epsilon?: number;
}

export function downsample(args: DownsampleArgs): Float64Array {
  switch (args.strategy) {
    case "merge":
      return mergeDownsample(args);
    case "lttb":
      return lttbDownsample(args);
    case "rdp":
      return rdpDownsample(args);
  }
}

// ---------------------------------------------------------------------------
// Merge strategy
// ---------------------------------------------------------------------------

function mergePass(
  pts: Float64Array,
  toSX: (x: number) => number,
  toSY: (y: number) => number,
  thresholdPx: number,
): Float64Array {
  const out = new Float64Array(pts.length);
  let count = 0;
  const n = pts.length / 3;
  const t2 = thresholdPx * thresholdPx;

  // Indices of points currently accumulated in the open cluster.
  // We store them to re-check each against the evolving centroid.
  const clusterIdx: number[] = [];
  let clusterX = 0;
  let clusterY = 0;
  let clusterW = 0;

  const flushCluster = () => {
    out[count * 3] = clusterX;
    out[count * 3 + 1] = clusterY;
    out[count * 3 + 2] = clusterW;
    count++;
    clusterIdx.length = 0;
  };

  for (let i = 0; i < n; i++) {
    const xi = pts[i * 3]!;
    const yi = pts[i * 3 + 1]!;
    const wi = pts[i * 3 + 2]!;

    if (clusterIdx.length === 0) {
      // Start a new cluster with this point.
      clusterIdx.push(i);
      clusterX = xi;
      clusterY = yi;
      clusterW = wi;
      continue;
    }

    // Hypothetical centroid if we absorb point i.
    const newW = clusterW + wi;
    const newX = (clusterX * clusterW + xi * wi) / newW;
    const newY = (clusterY * clusterW + yi * wi) / newW;
    const newSX = toSX(newX);
    const newSY = toSY(newY);

    // Check that every point already in the cluster stays within threshold
    // of the new centroid.
    let fits = true;
    for (const k of clusterIdx) {
      const dx = toSX(pts[k * 3]!) - newSX;
      const dy = toSY(pts[k * 3 + 1]!) - newSY;
      if (dx * dx + dy * dy >= t2) {
        fits = false;
        break;
      }
    }
    // Also check the incoming point itself.
    if (fits) {
      const dx = toSX(xi) - newSX;
      const dy = toSY(yi) - newSY;
      if (dx * dx + dy * dy >= t2) fits = false;
    }

    if (fits) {
      clusterIdx.push(i);
      clusterX = newX;
      clusterY = newY;
      clusterW = newW;
    } else {
      flushCluster();
      clusterIdx.push(i);
      clusterX = xi;
      clusterY = yi;
      clusterW = wi;
    }
  }

  if (clusterIdx.length > 0) flushCluster();

  return out.slice(0, count * 3);
}

function mergeDownsample({
  points,
  toSX,
  toSY,
  mergeThreshold,
}: DownsampleArgs): Float64Array {
  return mergePass(points, toSX, toSY, mergeThreshold ?? 1);
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
}: DownsampleArgs): Float64Array {
  const n = points.length / 3;
  const targetCount = Math.round(threshold);
  if (targetCount >= n) return points;

  const out = new Float64Array(targetCount * 3);
  let outIdx = 0;

  const write = (i: number, extraWeight = 0) => {
    out[outIdx * 3] = points[i * 3]!;
    out[outIdx * 3 + 1] = points[i * 3 + 1]!;
    out[outIdx * 3 + 2] = points[i * 3 + 2]! + extraWeight;
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
    // and sum weights of all non-selected points in the bucket into the winner
    let bestIdx = bucketStart;
    let bestArea = -1;
    let bucketWeightSum = 0;
    for (let k = bucketStart; k < bucketEnd; k++) {
      const bx = toSX(points[k * 3]!);
      const by = toSY(points[k * 3 + 1]!);
      const area = triangleArea(ax, ay, bx, by, cx, cy);
      if (area > bestArea) {
        bestArea = area;
        bestIdx = k;
      }
      bucketWeightSum += points[k * 3 + 2]!;
    }

    // winner keeps the total weight of its bucket
    const winnerWeight = points[bestIdx * 3 + 2]!;
    write(bestIdx, bucketWeightSum - winnerWeight);
    prevSelected = bestIdx;
  }

  write(n - 1);

  return out;
}

// ---------------------------------------------------------------------------
// RDP (Ramer-Douglas-Peucker) strategy
// ---------------------------------------------------------------------------

function perpendicularDistanceSq(
  px: number,
  py: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
): number {
  const dx = bx - ax;
  const dy = by - ay;
  const lenSq = dx * dx + dy * dy;
  if (lenSq === 0) {
    const ex = px - ax,
      ey = py - ay;
    return ex * ex + ey * ey;
  }
  const t = ((px - ax) * dx + (py - ay) * dy) / lenSq;
  const nx = px - (ax + t * dx);
  const ny = py - (ay + t * dy);
  return nx * nx + ny * ny;
}

function rdpRecursive(
  pts: Float64Array,
  toSX: (x: number) => number,
  toSY: (y: number) => number,
  start: number,
  end: number,
  epsilonSq: number,
  keep: Uint8Array,
): void {
  if (end <= start + 1) return;

  const ax = toSX(pts[start * 3]!);
  const ay = toSY(pts[start * 3 + 1]!);
  const bx = toSX(pts[end * 3]!);
  const by = toSY(pts[end * 3 + 1]!);

  let maxDistSq = 0;
  let maxIdx = start;
  for (let i = start + 1; i < end; i++) {
    const d = perpendicularDistanceSq(
      toSX(pts[i * 3]!),
      toSY(pts[i * 3 + 1]!),
      ax,
      ay,
      bx,
      by,
    );
    if (d > maxDistSq) {
      maxDistSq = d;
      maxIdx = i;
    }
  }

  if (maxDistSq > epsilonSq) {
    keep[maxIdx] = 1;
    rdpRecursive(pts, toSX, toSY, start, maxIdx, epsilonSq, keep);
    rdpRecursive(pts, toSX, toSY, maxIdx, end, epsilonSq, keep);
  }
}

function rdpDownsample({
  points,
  toSX,
  toSY,
  epsilon = 1,
}: DownsampleArgs): Float64Array {
  const n = points.length / 3;
  if (n <= 2) return points;

  const keep = new Uint8Array(n);
  keep[0] = 1;
  keep[n - 1] = 1;

  rdpRecursive(points, toSX, toSY, 0, n - 1, epsilon * epsilon, keep);

  // accumulate dropped weights into the next kept point
  const weights = new Float64Array(n);
  for (let i = 0; i < n; i++) weights[i] = points[i * 3 + 2]!;
  let pending = 0;
  for (let i = 0; i < n; i++) {
    if (keep[i]) {
      weights[i]! += pending;
      pending = 0;
    } else {
      pending += weights[i]!;
    }
  }

  let count = 0;
  for (let i = 0; i < n; i++) count += keep[i]!;

  const out = new Float64Array(count * 3);
  let outIdx = 0;
  for (let i = 0; i < n; i++) {
    if (keep[i]) {
      out[outIdx * 3] = points[i * 3]!;
      out[outIdx * 3 + 1] = points[i * 3 + 1]!;
      out[outIdx * 3 + 2] = weights[i]!;
      outIdx++;
    }
  }
  return out;
}

