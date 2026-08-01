import { describe, expect, test } from "vitest";
import { createNoiseFloatData } from "../src/chart/chart";
import { mergePoints } from "../src/examples/rasterizing/downsampling";
import {
  buildToSX,
  buildToSY,
  createRng,
  decodeComponents,
  decodeOriginalComponents,
  evaluateKdeAt,
  lerp,
} from "./kde-metrics";

const sampleCount = 10_000;

describe("Monte Carlo KDE error", () => {
  test("builds Monte Carlo table for N=1_000, 10_000, 100_000", () => {
    const scenarios = [1_000, 10_000, 100_000] as const;
    const table: Array<{
      N: string;
      merged: number;
      nrmsePct: string;
      rme: string;
    }> = [];

    for (const n of scenarios) {
      const metrics = estimateMonteCarloMetrics(n);
      table.push({
        N: n.toLocaleString("en-US"),
        merged: metrics.mergedCount,
        nrmsePct: `${(metrics.nrmse * 100).toFixed(4)}%`,
        rme: `±${metrics.rme.toFixed(2)}%`,
      });

      expect(Number.isFinite(metrics.mse)).toBe(true);
      expect(metrics.mse).toBeGreaterThanOrEqual(0);
      expect(metrics.acceptedSamples).toBeGreaterThan(0);
      expect(Number.isFinite(metrics.rme)).toBe(true);
      expect(metrics.rme).toBeGreaterThanOrEqual(0);
      expect(Number.isFinite(metrics.nrmse)).toBe(true);
      expect(metrics.nrmse).toBeGreaterThanOrEqual(0);
    }

    console.table(table);
  }, 60_000);
});

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
  while (acceptedSamples < sampleCount) {
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
