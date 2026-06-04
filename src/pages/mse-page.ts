import { createNoiseFloatData } from "../chart/chart";
import { mergePoints } from "../examples/rasterizing/downsampling";

type GaussianComponent = {
  x: number;
  y: number;
  w: number;
  p00: number;
  p01: number;
  p10: number;
  p11: number;
};

const SAMPLE_COUNT = 10_000;

export function runMsePage() {
  const scenarios = [1_000, 10_000, 100_000] as const;
  return scenarios.map((n) => {
    const metrics = estimateMonteCarloMetrics(n);
    return {
      N: n.toLocaleString("en-US"),
      merged: String(metrics.mergedCount),
      nrmsePct: `${(metrics.nrmse * 100).toFixed(4)}%`,
      rme: `±${metrics.rme.toFixed(2)}%`,
      samples: String(metrics.acceptedSamples),
    };
  });
}

function estimateMonteCarloMetrics(n: number) {
  const { dataF64, yMin, yMax } = createNoiseFloatData(n);

  const viewMinX = 0;
  const viewMaxX = 1;
  const viewMinY = yMin;
  const viewMaxY = yMax;
  const screenW = 1920;
  const screenH = 1080;
  const mergeThreshold = 1;
  const sigmaSizePx = 16;

  const merged = mergePoints({
    viewMinX,
    viewMaxX,
    viewMinY,
    viewMaxY,
    screenW,
    screenH,
    mergeThreshold,
    sigmaSizePx,
    dataF64,
  });

  const mergedComponents = decodeComponents(merged.gpuInstances);
  const originalComponents = decodeOriginalComponents(dataF64);

  const toSX = buildToSX(viewMinX, viewMaxX, screenW, sigmaSizePx);
  const toSY = buildToSY(viewMinY, viewMaxY, screenH, sigmaSizePx);

  const rng = createRng(0xdecafbad ^ n);
  const minApproxKde = 0.01;
  let acceptedSamples = 0;
  let mse = 0;
  let sqErrorSum = 0;
  let sqErrorSumSq = 0;
  let minReference = Number.POSITIVE_INFINITY;
  let maxReference = Number.NEGATIVE_INFINITY;
  while (acceptedSamples < SAMPLE_COUNT) {
    const x = lerp(viewMinX, viewMaxX, rng());
    const y = lerp(viewMinY, viewMaxY, rng());
    const approximation = evaluateKdeAt(mergedComponents, x, y, toSX, toSY);

    if (approximation < minApproxKde) {
      continue;
    }

    const reference = evaluateKdeAt(originalComponents, x, y, toSX, toSY);
    const diff = approximation - reference;
    const sqError = diff * diff;
    mse += sqError;
    sqErrorSum += sqError;
    sqErrorSumSq += sqError * sqError;
    acceptedSamples++;
    if (reference < minReference) minReference = reference;
    if (reference > maxReference) maxReference = reference;
  }
  mse /= Math.max(acceptedSamples, 1);

  const varianceSqError =
    acceptedSamples > 1
      ? (sqErrorSumSq - (sqErrorSum * sqErrorSum) / acceptedSamples) /
        (acceptedSamples - 1)
      : 0;
  const stdErrMse = Math.sqrt(
    Math.max(varianceSqError, 0) / Math.max(acceptedSamples, 1),
  );
  const margin95 = 1.96 * stdErrMse;
  const rme = mse > 0 ? (margin95 / mse) * 100 : 0;

  const referenceRange = maxReference - minReference;
  const nrmse = referenceRange > 0 ? Math.sqrt(mse) / referenceRange : 0;

  return { mse, nrmse, rme, acceptedSamples, mergedCount: merged.count };
}

function decodeOriginalComponents(dataF64: Float64Array): GaussianComponent[] {
  const components: GaussianComponent[] = [];
  for (let i = 0; i < dataF64.length; i += 3) {
    const x = dataF64[i]!;
    const y = dataF64[i + 1]!;
    const weight = dataF64[i + 2]!;
    components.push({
      x,
      y,
      w: weight / (2 * Math.PI),
      p00: 1,
      p01: 0,
      p10: 0,
      p11: 1,
    });
  }
  return components;
}

function decodeComponents(instances: Float32Array): GaussianComponent[] {
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

function buildToSX(
  viewMinX: number,
  viewMaxX: number,
  screenW: number,
  sigmaSizePx: number,
) {
  const xFactor = screenW / (viewMaxX - viewMinX) / sigmaSizePx;
  return (x: number) => (x - viewMinX) * xFactor;
}

function buildToSY(
  viewMinY: number,
  viewMaxY: number,
  screenH: number,
  sigmaSizePx: number,
) {
  const yFactor = screenH / (viewMaxY - viewMinY) / sigmaSizePx;
  return (y: number) => (y - viewMinY) * yFactor;
}

function evaluateKdeAt(
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
    if (det <= 1e-12) continue;

    const inv00 = c.p11 / det;
    const inv01 = -c.p01 / det;
    const inv10 = -c.p10 / det;
    const inv11 = c.p00 / det;
    const d2 = dx * (inv00 * dx + inv01 * dy) + dy * (inv10 * dx + inv11 * dy);

    sum += c.w * Math.exp(-0.5 * d2);
  }

  return sum;
}

function createRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 0x1_0000_0000;
  };
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}
