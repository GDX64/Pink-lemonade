import { bench, describe } from "vitest";
import { readFile } from "node:fs/promises";
import { createNoiseFloatData } from "../src/chart/chart";
import { wasmMerge } from "../src/examples/rasterizing/downsampling";

const nativeFetch = globalThis.fetch.bind(globalThis);

globalThis.fetch = async (
  input: Parameters<typeof fetch>[0],
  init?: Parameters<typeof fetch>[1],
): Promise<Response> => {
  const url =
    typeof input === "string"
      ? input
      : input instanceof URL
        ? input.href
        : input.url;

  if (url.startsWith("file://")) {
    const filePath = new URL(url);
    const bytes = await readFile(filePath);
    return new Response(bytes, {
      status: 200,
      headers: { "content-type": "application/wasm" },
    });
  }

  return nativeFetch(input, init);
};

const downsampler = await wasmMerge();

describe("GaussianChart.mergePoints", () => {
  const samples = [
    ["1_000", createNoiseFloatData(1_000)],
    ["10_000", createNoiseFloatData(10_000)],
    ["100_000", createNoiseFloatData(100_000)],
    ["1_000_000", createNoiseFloatData(1_000_000)],
  ] as const;

  const merged_1_000 = calc(downsampler, samples[0][1]);
  const merged_10_000 = calc(downsampler, samples[1][1]);
  const merged_100_000 = calc(downsampler, samples[2][1]);
  const merged_1000_000 = calc(downsampler, samples[3][1]);

  bench(
    `N = 1000_000 | merged=${merged_1000_000}`,
    () => {
      calcWithoutData(downsampler, samples[3][1]);
    },
    {
      iterations: 50,
      setup: () => {
        downsampler.setDataF64(samples[3][1].dataF64);
      },
    },
  );

  bench(
    `N = 100_000 | merged=${merged_100_000}`,
    () => {
      calcWithoutData(downsampler, samples[2][1]);
    },
    {
      setup: () => {
        downsampler.setDataF64(samples[2][1].dataF64);
      },
    },
  );

  bench(
    `N = 10_000 | merged=${merged_10_000}`,
    () => {
      calcWithoutData(downsampler, samples[1][1]);
    },
    {
      setup: () => {
        downsampler.setDataF64(samples[1][1].dataF64);
      },
    },
  );

  bench(
    `N = 1_000 | merged=${merged_1_000}`,
    () => {
      calcWithoutData(downsampler, samples[0][1]);
    },
    {
      setup: () => {
        downsampler.setDataF64(samples[0][1].dataF64);
      },
    },
  );
});

function calc(
  downsampler: Awaited<ReturnType<typeof wasmMerge>>,
  {
    dataF64,
    yMin,
    yMax,
  }: ReturnType<typeof createNoiseFloatData>,
) {
  downsampler.setViewMinX(0);
  downsampler.setViewMaxX(1);
  downsampler.setViewMinY(yMin);
  downsampler.setViewMaxY(yMax);
  downsampler.setScreenW(1920);
  downsampler.setScreenH(1080);
  downsampler.setMergeThreshold(1);
  downsampler.setSigmaSizePx(16);
  downsampler.setDataF64(dataF64);

  const result = downsampler.mergePoints();

  if (result.count <= 0) {
    throw new Error("Expected merged points for full-range benchmark");
  }

  return result.count;
}

function calcWithoutData(
  downsampler: Awaited<ReturnType<typeof wasmMerge>>,
  { yMin, yMax }: ReturnType<typeof createNoiseFloatData>,
) {
  downsampler.setViewMinX(0);
  downsampler.setViewMaxX(1);
  downsampler.setViewMinY(yMin);
  downsampler.setViewMaxY(yMax);
  downsampler.setScreenW(1920);
  downsampler.setScreenH(1080);
  downsampler.setMergeThreshold(1);
  downsampler.setSigmaSizePx(16);

  const result = downsampler.mergePoints();

  if (result.count <= 0) {
    throw new Error("Expected merged points for full-range benchmark");
  }

  return result.count;
}
