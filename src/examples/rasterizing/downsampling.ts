export type DownsampleStrategy = "merge" | "lttb" | "rdp";

export interface DownsampleArgs {
  /** Packed [x, y, weight] triples, already filtered to the visible range. */
  points: Float64Array;
  strategy: DownsampleStrategy;
  /** Maps a data-space x to screen pixels. */
  toSX: (x: number) => number;
  /** Maps a data-space y to screen pixels. */
  toSY: (y: number) => number;
  mergeThreshold?: number; // only for merge strategy, overrides threshold
}

export function downsample(args: DownsampleArgs): Float64Array {
  switch (args.strategy) {
    case "merge":
      return mergeDownsample(args);
    default:
      throw new Error(`Unsupported downsample strategy: ${args.strategy}`);
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
  // Merge output layout per point:
  // [x, y, weight, variance]
  const out = new Float64Array((pts.length / 3) * 4);
  let count = 0;
  const n = pts.length / 3;
  const t2 = thresholdPx * thresholdPx;

  // Track only the first point of the open cluster — it's most likely to be
  // left behind as the centroid drifts forward with each new addition.
  let hasCluster = false;
  let firstX = 0;
  let firstY = 0;
  let clusterX = 0;
  let clusterY = 0;
  let clusterW = 0;
  let clusterStartIdx = 0;
  let clusterEndIdx = 0;

  const flushCluster = () => {
    // Assume each Gaussian has base variance 1 and add weighted spread around
    // the merged centroid in screen space.
    const cx = toSX(clusterX);
    const cy = toSY(clusterY);
    let weightedSqDistSum = 0;
    for (let i = clusterStartIdx; i <= clusterEndIdx; i++) {
      const x = pts[i * 3]!;
      const y = pts[i * 3 + 1]!;
      const w = pts[i * 3 + 2]!;
      const dx = toSX(x) - cx;
      const dy = toSY(y) - cy;
      weightedSqDistSum += w * (dx * dx + dy * dy);
    }
    const variance = 1 + weightedSqDistSum / Math.max(clusterW, 1e-12);

    const w = clusterW / variance; // we must make this adjust to preserve the total mass of the cluster

    const o = count * 4;
    out[o] = clusterX;
    out[o + 1] = clusterY;
    out[o + 2] = w;
    out[o + 3] = variance;
    count++;
    hasCluster = false;
  };

  for (let i = 0; i < n; i++) {
    const xi = pts[i * 3]!;
    const yi = pts[i * 3 + 1]!;
    const wi = pts[i * 3 + 2]!;

    if (!hasCluster) {
      hasCluster = true;
      firstX = xi;
      firstY = yi;
      clusterX = xi;
      clusterY = yi;
      clusterW = wi;
      clusterStartIdx = i;
      clusterEndIdx = i;
      continue;
    }

    // Hypothetical centroid if we absorb point i.
    const newW = clusterW + wi;
    const newX = (clusterX * clusterW + xi * wi) / newW;
    const newY = (clusterY * clusterW + yi * wi) / newW;
    const newSX = toSX(newX);
    const newSY = toSY(newY);

    // Check the first cluster point (most likely to exceed threshold) and the incoming point.
    const dfx = toSX(firstX) - newSX;
    const dfy = toSY(firstY) - newSY;
    const dix = toSX(xi) - newSX;
    const diy = toSY(yi) - newSY;

    if (dfx * dfx + dfy * dfy < t2 && dix * dix + diy * diy < t2) {
      clusterX = newX;
      clusterY = newY;
      clusterW = newW;
      clusterEndIdx = i;
    } else {
      flushCluster();
      hasCluster = true;
      firstX = xi;
      firstY = yi;
      clusterX = xi;
      clusterY = yi;
      clusterW = wi;
      clusterStartIdx = i;
      clusterEndIdx = i;
    }
  }

  if (hasCluster) flushCluster();

  return out.slice(0, count * 4);
}

function mergeDownsample({
  points,
  toSX,
  toSY,
  mergeThreshold,
}: DownsampleArgs): Float64Array {
  return mergePass(points, toSX, toSY, mergeThreshold ?? 1);
}
