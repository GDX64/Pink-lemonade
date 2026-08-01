import { describe, expect, test } from "vitest";
import { createNoiseFloatData } from "../src/chart/chart";
import { mergePoints } from "../src/examples/rasterizing/downsampling";
import { KLDownsampler } from "../src/examples/rasterizing/kl-downsampling";
import {
  buildToSX,
  buildToSY,
  createRng,
  decodeComponents,
  decodeOriginalComponents,
  errorStats,
  evaluateKdeAt,
  lerp,
} from "./kde-metrics";

/**
 * Head-to-head against Runnalls' KL-optimal reduction at an equal kernel
 * budget. Runnalls is the quality ceiling, not a per-frame competitor: it is
 * O(N^2) to set up, so it only runs at the smaller dataset sizes.
 */

const sampleCount = 10_000;
const minApproxKde = 0.01;

const viewMinX = 0;
const viewMaxX = 1;
const screenW = 1920;
const screenH = 1080;
const mergeThreshold = 1;
const sigmaSizePx = 16;

describe("Runnalls vs screen-space merge", () => {
  test("compares NRMSE at matched kernel counts", () => {
    const scenarios = [1_000, 10_000] as const;
    const table: Array<Record<string, string | number>> = [];

    for (const n of scenarios) {
      const r = compare(n);
      table.push({
        N: n.toLocaleString("en-US"),
        kernels: r.kernels,
        "merge NRMSE": `${(r.merge.nrmse * 100).toFixed(4)}%`,
        "merge RME": `±${r.merge.rme.toFixed(2)}%`,
        "Runnalls NRMSE": `${(r.kl.nrmse * 100).toFixed(4)}%`,
        "Runnalls RME": `±${r.kl.rme.toFixed(2)}%`,
        "NRMSE ratio": `${(r.merge.nrmse / r.kl.nrmse).toFixed(1)}x`,
        "merge ms": r.mergeMs.toFixed(2),
        "Runnalls ms": r.klMs.toFixed(2),
      });

      // Both must be sane estimates of the same field.
      expect(r.samples).toBe(sampleCount);
      expect(Number.isFinite(r.merge.nrmse)).toBe(true);
      expect(Number.isFinite(r.kl.nrmse)).toBe(true);

      // The KL-optimal reduction is the ceiling: at the same budget it should
      // not be beaten by the one-pass screen-space merge.
      expect(r.kl.nrmse).toBeLessThanOrEqual(r.merge.nrmse);

      // ...and it must cost far more to obtain, which is what disqualifies it
      // as a per-frame method and makes it a reference instead.
      expect(r.klMs).toBeGreaterThan(r.mergeMs);
    }

    console.table(table);
  }, 600_000);

  test("Runnalls conserves total mass at the merge budget", () => {
    const n = 1_000;
    const { dataF64, yMin, yMax } = createNoiseFloatData(n);
    const merged = runMerge(dataF64, yMin, yMax);
    const kl = runKL(dataF64, yMin, yMax, merged.count).result;

    let expected = 0;
    for (let i = 2; i < dataF64.length; i += 3) expected += dataF64[i]!;

    // Recover mass from amplitude: w = A * 2*pi*sqrt(det P).
    let actual = 0;
    for (let i = 0; i < kl.count; i++) {
      const o = i * 7;
      const det =
        kl.gpuInstances[o + 3]! * kl.gpuInstances[o + 6]! -
        kl.gpuInstances[o + 4]! * kl.gpuInstances[o + 5]!;
      actual += kl.gpuInstances[o + 2]! * 2 * Math.PI * Math.sqrt(det);
    }

    expect(actual / expected).toBeCloseTo(1, 3);
  }, 120_000);
});

function runMerge(dataF64: Float64Array, yMin: number, yMax: number) {
  return mergePoints({
    viewMinX,
    viewMaxX,
    viewMinY: yMin,
    viewMaxY: yMax,
    screenW,
    screenH,
    mergeThreshold,
    sigmaSizePx,
    dataF64,
  });
}

function runKL(
  dataF64: Float64Array,
  yMin: number,
  yMax: number,
  targetCount: number,
) {
  const d = new KLDownsampler();
  d.setViewMinX(viewMinX);
  d.setViewMaxX(viewMaxX);
  d.setViewMinY(yMin);
  d.setViewMaxY(yMax);
  d.setScreenW(screenW);
  d.setScreenH(screenH);
  d.setSigmaSizePx(sigmaSizePx);
  d.setTargetCount(targetCount);
  d.setDataF64(dataF64);

  const start = performance.now();
  const result = d.mergePoints();
  return { result, ms: performance.now() - start };
}

function compare(n: number) {
  const { dataF64, yMin, yMax } = createNoiseFloatData(n);

  const mergeStart = performance.now();
  const merged = runMerge(dataF64, yMin, yMax);
  const mergeMs = performance.now() - mergeStart;

  const kernels = merged.count;
  const { result: klResult, ms: klMs } = runKL(dataF64, yMin, yMax, kernels);
  expect(klResult.count).toBe(kernels);

  const mergeComponents = decodeComponents(merged.gpuInstances);
  const klComponents = decodeComponents(klResult.gpuInstances);
  const originalComponents = decodeOriginalComponents(dataF64);

  const toSX = buildToSX(viewMinX, viewMaxX, screenW, sigmaSizePx);
  const toSY = buildToSY(yMin, yMax, screenH, sigmaSizePx);

  // Both approximations are scored on exactly the same accepted sample set.
  // Acceptance uses the merge field, matching the existing Monte Carlo table,
  // so background regions do not dilute the measured error.
  const rng = createRng(0xdecafbad ^ n);
  const mergeValues: number[] = [];
  const klValues: number[] = [];
  const referenceValues: number[] = [];

  while (referenceValues.length < sampleCount) {
    const x = lerp(viewMinX, viewMaxX, rng());
    const y = lerp(yMin, yMax, rng());
    const mergeValue = evaluateKdeAt(mergeComponents, x, y, toSX, toSY);
    if (mergeValue < minApproxKde) continue;

    mergeValues.push(mergeValue);
    klValues.push(evaluateKdeAt(klComponents, x, y, toSX, toSY));
    referenceValues.push(
      evaluateKdeAt(originalComponents, x, y, toSX, toSY),
    );
  }

  return {
    kernels,
    samples: referenceValues.length,
    merge: errorStats(mergeValues, referenceValues),
    kl: errorStats(klValues, referenceValues),
    mergeMs,
    klMs,
  };
}
