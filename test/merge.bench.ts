import { bench, describe } from "vitest";
import { createNoiseFloatData } from "../src/chart/chart";
import { GaussianChart } from "../src/examples/rasterizing/gaussian-chart";
import { mergePoints } from "../src/examples/rasterizing/downsampling";

describe("GaussianChart.mergePoints", () => {
  const n_1000_000 = createNoiseFloatData(1000_000);
  const merged_1000_000 = calc(n_1000_000);
  bench(
    `N = 1000_000 | merged=${merged_1000_000}`,
    () => {
      calc(n_1000_000);
    },
    { iterations: 50 },
  );

  const n_100_000 = createNoiseFloatData(100_000);
  const merged_100_000 = calc(n_100_000);
  bench(`N = 100_000 | merged=${merged_100_000}`, () => {
    calc(n_100_000);
  });

  const n_10_000 = createNoiseFloatData(10_000);
  const merged_10_000 = calc(n_10_000);
  bench(`N = 10_000 | merged=${merged_10_000}`, () => {
    calc(n_10_000);
  });

  const n_1_000 = createNoiseFloatData(1_000);
  const merged_1_000 = calc(n_1_000);
  bench(`N = 1_000 | merged=${merged_1_000}`, () => {
    calc(n_1_000);
  });
});

function calc({
  dataF64,
  yMin,
  yMax,
}: ReturnType<typeof createNoiseFloatData>) {
  const result = mergePoints({
    viewMinX: 0,
    viewMaxX: 1,
    viewMinY: yMin,
    viewMaxY: yMax,
    screenW: 1920,
    screenH: 1080,
    mergeThreshold: 1,
    sigmaSizePx: 16,
    dataF64,
  });

  if (result.count <= 0) {
    throw new Error("Expected merged points for full-range benchmark");
  }

  return result.count;
}
