/**
 * Shared Monte Carlo machinery for measuring KDE approximation error, used by
 * both the screen-space merge table and the Runnalls comparison.
 */

export type GaussianComponent = {
  x: number;
  y: number;
  w: number;
  p00: number;
  p01: number;
  p10: number;
  p11: number;
};

export function decodeOriginalComponents(
  dataF64: Float64Array,
): GaussianComponent[] {
  const components: GaussianComponent[] = [];
  for (let i = 0; i < dataF64.length; i += 3) {
    const x = dataF64[i]!;
    const y = dataF64[i + 1]!;
    const weight = dataF64[i + 2]!;
    components.push({
      x,
      y,
      // Single-point kernels use identity covariance in merge implementation.
      w: weight / (2 * Math.PI),
      p00: 1,
      p01: 0,
      p10: 0,
      p11: 1,
    });
  }
  return components;
}

export function decodeComponents(instances: Float32Array): GaussianComponent[] {
  const components: GaussianComponent[] = [];
  for (let i = 0; i < instances.length; i += 7) {
    components.push({
      x: instances[i]!,
      y: instances[i + 1]!,
      w: instances[i + 2]!,
      p00: instances[i + 3]!,
      p01: instances[i + 4]!,
      p10: instances[i + 5]!,
      p11: instances[i + 6]!,
    });
  }
  return components;
}

export function buildToSX(
  viewMinX: number,
  viewMaxX: number,
  screenW: number,
  sigmaSizePx: number,
) {
  const xFactor = screenW / (viewMaxX - viewMinX) / sigmaSizePx;
  return (x: number) => (x - viewMinX) * xFactor;
}

export function buildToSY(
  viewMinY: number,
  viewMaxY: number,
  screenH: number,
  sigmaSizePx: number,
) {
  const yFactor = screenH / (viewMaxY - viewMinY) / sigmaSizePx;
  return (y: number) => (y - viewMinY) * yFactor;
}

export function evaluateKdeAt(
  components: GaussianComponent[],
  x: number,
  y: number,
  toSX: (x: number) => number,
  toSY: (y: number) => number,
): number {
  const sx = toSX(x);
  const sy = toSY(y);
  let sum = 0;

  for (const c of components) {
    const mux = toSX(c.x);
    const muy = toSY(c.y);
    const dx = sx - mux;
    const dy = sy - muy;

    const det = c.p00 * c.p11 - c.p01 * c.p10;
    if (det <= 1e-12) {
      continue;
    }

    const inv00 = c.p11 / det;
    const inv01 = -c.p01 / det;
    const inv10 = -c.p10 / det;
    const inv11 = c.p00 / det;
    const d2 = dx * (inv00 * dx + inv01 * dy) + dy * (inv10 * dx + inv11 * dy);

    sum += c.w * Math.exp(-0.5 * d2);
  }

  return sum;
}

export function createRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 0x1_0000_0000;
  };
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export type ErrorStats = {
  mse: number;
  nrmse: number;
  rme: number;
};

/**
 * MSE / NRMSE / 95% relative margin of error of an approximation against a
 * reference, evaluated at pre-drawn sample points.
 */
export function errorStats(
  approximation: number[],
  reference: number[],
): ErrorStats {
  const n = approximation.length;
  let sqErrorSum = 0;
  let sqErrorSumSq = 0;
  let minReference = Number.POSITIVE_INFINITY;
  let maxReference = Number.NEGATIVE_INFINITY;

  for (let i = 0; i < n; i++) {
    const diff = approximation[i]! - reference[i]!;
    const sqError = diff * diff;
    sqErrorSum += sqError;
    sqErrorSumSq += sqError * sqError;
    if (reference[i]! < minReference) minReference = reference[i]!;
    if (reference[i]! > maxReference) maxReference = reference[i]!;
  }

  const mse = sqErrorSum / Math.max(n, 1);
  const varianceSqError =
    n > 1 ? (sqErrorSumSq - (sqErrorSum * sqErrorSum) / n) / (n - 1) : 0;
  const stdErrMse = Math.sqrt(Math.max(varianceSqError, 0) / Math.max(n, 1));
  const rme = mse > 0 ? ((1.96 * stdErrMse) / mse) * 100 : 0;

  const referenceRange = maxReference - minReference;
  const nrmse = referenceRange > 0 ? Math.sqrt(mse) / referenceRange : 0;

  return { mse, nrmse, rme };
}
