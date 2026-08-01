import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect, test } from "vitest";
import {
  initSync,
  KlDownsampler,
  SalmondDownsampler,
} from "../pkg/pink_lemonade_wasm";
import { createNoiseFloatData } from "../src/chart/chart";
import { mergePoints } from "../src/examples/rasterizing/downsampling";
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
 * Salmond's clustering reduction (CAF) against the screen-space merge and
 * Runnalls, at a matched kernel budget. Salmond sits between the two: it is
 * covariance-aware and order-independent like Runnalls, but reduces a whole
 * pass at a time rather than one optimal pair at a time.
 */

const sampleCount = 10_000;
const minApproxKde = 0.01;

const viewMinX = 0;
const viewMaxX = 1;
const screenW = 1920;
const screenH = 1080;
const mergeThreshold = 1;
const sigmaSizePx = 16;

beforeAll(() => {
  const wasmPath = fileURLToPath(
    new URL("../pkg/pink_lemonade_wasm_bg.wasm", import.meta.url),
  );
  initSync({ module: readFileSync(wasmPath) });
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

function configure<T extends KlDownsampler | SalmondDownsampler>(
  d: T,
  dataF64: Float64Array,
  yMin: number,
  yMax: number,
  targetCount: number,
): T {
  d.setViewMinX(viewMinX);
  d.setViewMaxX(viewMaxX);
  d.setViewMinY(yMin);
  d.setViewMaxY(yMax);
  d.setScreenW(screenW);
  d.setScreenH(screenH);
  d.setSigmaSizePx(sigmaSizePx);
  d.setTargetCount(targetCount);
  d.setDataF64(dataF64);
  return d;
}

function timed<T>(run: () => T) {
  const start = performance.now();
  const result = run();
  return { result, ms: performance.now() - start };
}

/** Total mass recovered from the packed instances: w = A * 2*pi*sqrt(det P). */
function totalMass(instances: Float32Array, count: number) {
  let mass = 0;
  for (let i = 0; i < count; i++) {
    const o = i * 7;
    const det =
      instances[o + 3]! * instances[o + 6]! - instances[o + 4]! * instances[o + 5]!;
    mass += instances[o + 2]! * 2 * Math.PI * Math.sqrt(Math.max(det, 0));
  }
  return mass;
}

describe("Salmond clustering reduction", () => {
  test("conserves total mass at the merge budget", () => {
    const { dataF64, yMin, yMax } = createNoiseFloatData(1_000);
    const target = runMerge(dataF64, yMin, yMax).count;
    const result = configure(
      new SalmondDownsampler(),
      dataF64,
      yMin,
      yMax,
      target,
    ).mergePoints();

    let expected = 0;
    for (let i = 2; i < dataF64.length; i += 3) expected += dataF64[i]!;

    expect(totalMass(result.gpuInstances, result.count) / expected).toBeCloseTo(
      1,
      3,
    );
  }, 120_000);

  test("meets the budget without ever exceeding it", () => {
    const { dataF64, yMin, yMax } = createNoiseFloatData(2_000);
    for (const target of [10, 100, 400, 1_000]) {
      const result = configure(
        new SalmondDownsampler(),
        dataF64,
        yMin,
        yMax,
        target,
      ).mergePoints();
      // Grouped merging can overshoot below the budget, never above it.
      expect(result.count).toBeLessThanOrEqual(target);
      expect(result.count).toBeGreaterThan(0);
    }
  }, 120_000);

  test("is independent of the order samples arrive in", () => {
    // The merge is a time-ordered scan; Salmond is not. Reversing the series
    // (which reverses x too, so the view is mirrored) must give the same set of
    // kernels up to that mirroring.
    const n = 500;
    const { dataF64, yMin, yMax } = createNoiseFloatData(n);
    const mirrored = new Float64Array(dataF64.length);
    for (let i = 0; i < n; i++) {
      const src = (n - 1 - i) * 3;
      mirrored[i * 3] = viewMaxX - dataF64[src]!;
      mirrored[i * 3 + 1] = dataF64[src + 1]!;
      mirrored[i * 3 + 2] = dataF64[src + 2]!;
    }

    const a = configure(
      new SalmondDownsampler(),
      dataF64,
      yMin,
      yMax,
      120,
    ).mergePoints();
    const b = configure(
      new SalmondDownsampler(),
      mirrored,
      yMin,
      yMax,
      120,
    ).mergePoints();

    expect(b.count).toBe(a.count);

    const key = (r: typeof a, mirror: boolean) => {
      const out: string[] = [];
      for (let i = 0; i < r.count; i++) {
        const o = i * 7;
        const x = mirror ? viewMaxX - r.gpuInstances[o]! : r.gpuInstances[o]!;
        out.push(`${x.toFixed(4)}|${r.gpuInstances[o + 1]!.toFixed(4)}`);
      }
      return out.sort();
    };
    expect(key(b, true)).toEqual(key(a, false));
  }, 120_000);

  test("compares NRMSE against the merge and Runnalls at matched budgets", () => {
    const table: Array<Record<string, string | number>> = [];

    for (const n of [1_000, 10_000] as const) {
      const { dataF64, yMin, yMax } = createNoiseFloatData(n);

      const merge = timed(() => runMerge(dataF64, yMin, yMax));
      const target = merge.result.count;

      const salmond = timed(() =>
        configure(
          new SalmondDownsampler(),
          dataF64,
          yMin,
          yMax,
          target,
        ).mergePoints(),
      );
      const kl = timed(() =>
        configure(
          new KlDownsampler(),
          dataF64,
          yMin,
          yMax,
          target,
        ).mergePoints(),
      );
      expect(kl.result.count).toBe(target);

      const mergeComponents = decodeComponents(merge.result.gpuInstances);
      const salmondComponents = decodeComponents(salmond.result.gpuInstances);
      const klComponents = decodeComponents(kl.result.gpuInstances);
      const originalComponents = decodeOriginalComponents(dataF64);

      const toSX = buildToSX(viewMinX, viewMaxX, screenW, sigmaSizePx);
      const toSY = buildToSY(yMin, yMax, screenH, sigmaSizePx);

      // All three approximations are scored on the same accepted sample set,
      // gated on the merge field exactly as the published Monte Carlo table is.
      const rng = createRng(0x5a10ffd ^ n);
      const mergeValues: number[] = [];
      const salmondValues: number[] = [];
      const klValues: number[] = [];
      const referenceValues: number[] = [];

      while (referenceValues.length < sampleCount) {
        const x = lerp(viewMinX, viewMaxX, rng());
        const y = lerp(yMin, yMax, rng());
        const mergeValue = evaluateKdeAt(mergeComponents, x, y, toSX, toSY);
        if (mergeValue < minApproxKde) continue;

        mergeValues.push(mergeValue);
        salmondValues.push(evaluateKdeAt(salmondComponents, x, y, toSX, toSY));
        klValues.push(evaluateKdeAt(klComponents, x, y, toSX, toSY));
        referenceValues.push(
          evaluateKdeAt(originalComponents, x, y, toSX, toSY),
        );
      }

      const mergeStats = errorStats(mergeValues, referenceValues);
      const salmondStats = errorStats(salmondValues, referenceValues);
      const klStats = errorStats(klValues, referenceValues);

      table.push({
        N: n.toLocaleString("en-US"),
        "merge kernels": target,
        "Salmond kernels": salmond.result.count,
        "merge NRMSE": `${(mergeStats.nrmse * 100).toFixed(4)}%`,
        "Salmond NRMSE": `${(salmondStats.nrmse * 100).toFixed(4)}%`,
        "Runnalls NRMSE": `${(klStats.nrmse * 100).toFixed(4)}%`,
        "merge ms": merge.ms.toFixed(2),
        "Salmond ms": salmond.ms.toFixed(2),
        "Runnalls ms": kl.ms.toFixed(2),
      });

      expect(Number.isFinite(salmondStats.nrmse)).toBe(true);
      expect(salmond.result.count).toBeLessThanOrEqual(target);

      // Runnalls is optimal in its own greedy sense and is the ceiling for both.
      expect(klStats.nrmse).toBeLessThanOrEqual(salmondStats.nrmse);

      // Clustering reduces a whole pass at a time instead of rescanning after
      // every single merge, so it must come out far cheaper than Runnalls. (No
      // claim is made against the merge here: these are single-shot timings,
      // and the benchmark page is where the two are measured properly.)
      expect(salmond.ms).toBeLessThan(kl.ms);
    }

    console.table(table);
  }, 600_000);
});
