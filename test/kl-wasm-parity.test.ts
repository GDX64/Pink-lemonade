import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect, test } from "vitest";
import { initSync, KlDownsampler } from "../pkg/pink_lemonade_wasm";
import { createNoiseFloatData } from "../src/chart/chart";
import { KLDownsampler } from "../src/examples/rasterizing/kl-downsampling";
import {
  buildToSX,
  buildToSY,
  createRng,
  decodeComponents,
  errorStats,
  evaluateKdeAt,
  lerp,
} from "./kde-metrics";

/** Mass, mean and covariance of a packed kernel buffer, in data space. */
function moments(instances: Float32Array, count: number) {
  let mass = 0;
  let mx = 0;
  let my = 0;
  const w: number[] = [];
  for (let i = 0; i < count; i++) {
    const o = i * 7;
    const det =
      instances[o + 3]! * instances[o + 6]! - instances[o + 4]! * instances[o + 5]!;
    const wi = instances[o + 2]! * 2 * Math.PI * Math.sqrt(Math.max(det, 1e-12));
    w.push(wi);
    mass += wi;
    mx += wi * instances[o]!;
    my += wi * instances[o + 1]!;
  }
  mx /= mass;
  my /= mass;

  let p00 = 0;
  let p11 = 0;
  for (let i = 0; i < count; i++) {
    const o = i * 7;
    const dx = instances[o]! - mx;
    const dy = instances[o + 1]! - my;
    p00 += w[i]! * dx * dx;
    p11 += w[i]! * dy * dy;
  }
  return { mass, mx, my, p00: p00 / mass, p11: p11 / mass };
}

/**
 * The wasm port must reproduce the TypeScript reference implementation, not
 * merely resemble it --- the benchmark compares wasm merge against wasm
 * Runnalls, so a divergence here would silently invalidate the comparison.
 */

beforeAll(() => {
  const wasmPath = fileURLToPath(
    new URL("../pkg/pink_lemonade_wasm_bg.wasm", import.meta.url),
  );
  initSync({ module: readFileSync(wasmPath) });
});

const view = {
  viewMinX: 0,
  viewMaxX: 1,
  screenW: 1920,
  screenH: 1080,
  sigmaSizePx: 16,
};

function configureWasm(
  data: Float64Array,
  yMin: number,
  yMax: number,
  targetCount: number,
) {
  const d = new KlDownsampler();
  d.setViewMinX(view.viewMinX);
  d.setViewMaxX(view.viewMaxX);
  d.setViewMinY(yMin);
  d.setViewMaxY(yMax);
  d.setScreenW(view.screenW);
  d.setScreenH(view.screenH);
  d.setSigmaSizePx(view.sigmaSizePx);
  d.setTargetCount(targetCount);
  d.setDataF64(data);
  return d;
}

function configureTs(
  data: Float64Array,
  yMin: number,
  yMax: number,
  targetCount: number,
) {
  const d = new KLDownsampler();
  d.setViewMinX(view.viewMinX);
  d.setViewMaxX(view.viewMaxX);
  d.setViewMinY(yMin);
  d.setViewMaxY(yMax);
  d.setScreenW(view.screenW);
  d.setScreenH(view.screenH);
  d.setSigmaSizePx(view.sigmaSizePx);
  d.setTargetCount(targetCount);
  d.setDataF64(data);
  return d;
}

describe("wasm / TypeScript Runnalls parity", () => {
  test.each([
    [500, 40],
    [500, 137],
    [2_000, 250],
  ])("N=%i reduced to %i kernels preserves the same moments", (n, target) => {
    const { dataF64, yMin, yMax } = createNoiseFloatData(n);

    const wasm = configureWasm(dataF64, yMin, yMax, target).mergePoints();
    const ts = configureTs(dataF64, yMin, yMax, target).mergePoints();

    expect(wasm.count).toBe(target);
    expect(ts.count).toBe(target);
    expect(wasm.gpuInstances.length).toBe(ts.gpuInstances.length);

    // Element-wise equality is NOT a valid expectation here. Rust's `ln` and
    // JS's `Math.log` can differ by one ULP, and the greedy selection amplifies
    // that: once two candidate pairs tie to within a ULP, the two runs pick
    // different pairs and the merge paths separate for good. What must agree is
    // everything the algorithm actually guarantees.
    const a = moments(wasm.gpuInstances, wasm.count);
    const b = moments(ts.gpuInstances, ts.count);

    // Total mass, mean and covariance are invariant under ANY sequence of
    // moment-preserving merges, so they must match regardless of path.
    expect(a.mass / b.mass).toBeCloseTo(1, 6);
    expect(a.mx).toBeCloseTo(b.mx, 4);
    expect(a.my).toBeCloseTo(b.my, 4);
    expect(a.p00 / b.p00).toBeCloseTo(1, 3);
    expect(a.p11 / b.p11).toBeCloseTo(1, 3);
  }, 60_000);

  test.each([
    [500, 40],
    [2_000, 250],
  ])(
    "N=%i reduced to %i kernels yields an indistinguishable density field",
    (n, target) => {
      const { dataF64, yMin, yMax } = createNoiseFloatData(n);

      const wasm = configureWasm(dataF64, yMin, yMax, target).mergePoints();
      const ts = configureTs(dataF64, yMin, yMax, target).mergePoints();

      const wasmComponents = decodeComponents(wasm.gpuInstances);
      const tsComponents = decodeComponents(ts.gpuInstances);
      const toSX = buildToSX(
        view.viewMinX,
        view.viewMaxX,
        view.screenW,
        view.sigmaSizePx,
      );
      const toSY = buildToSY(yMin, yMax, view.screenH, view.sigmaSizePx);

      const rng = createRng(0x5eed ^ n);
      const wasmValues: number[] = [];
      const tsValues: number[] = [];
      while (tsValues.length < 2_000) {
        const x = lerp(view.viewMinX, view.viewMaxX, rng());
        const y = lerp(yMin, yMax, rng());
        const value = evaluateKdeAt(tsComponents, x, y, toSX, toSY);
        if (value < 0.01) continue;
        tsValues.push(value);
        wasmValues.push(evaluateKdeAt(wasmComponents, x, y, toSX, toSY));
      }

      // Divergent merge paths must still produce the same density field to
      // well within the error either one has against the full KDE (~1e-4).
      const { nrmse } = errorStats(wasmValues, tsValues);
      expect(nrmse).toBeLessThan(1e-5);
    },
    60_000,
  );

  test("the KL threshold stop criterion agrees", () => {
    const { dataF64, yMin, yMax } = createNoiseFloatData(500);

    for (const threshold of [1e-4, 1e-2, 1]) {
      const wasm = configureWasm(dataF64, yMin, yMax, 1);
      wasm.setMergeThreshold(threshold);
      const ts = configureTs(dataF64, yMin, yMax, 1);
      ts.setMergeThreshold(threshold);

      const a = wasm.mergePoints();
      const b = ts.mergePoints();
      expect(a.count).toBe(b.count);
      expect(a.count).toBeGreaterThan(1);
    }
  }, 60_000);
});
