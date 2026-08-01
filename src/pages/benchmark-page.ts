import { createNoiseFloatData } from "../chart/chart";
import { wasmMerge } from "../examples/rasterizing/downsampling";
import { wasmKlMerge } from "../examples/rasterizing/kl-downsampling";
import { wasmSalmondMerge } from "../examples/rasterizing/salmond-downsampling";

/**
 * Dataset sizes at which Runnalls' reduction is also benchmarked. It is
 * O(N^2) to set up, so it is capped well below the merge sizes: at N=10,000 a
 * single run already takes ~1 minute, and N=100,000 would take hours.
 */
const KL_SCENARIOS = new Map<string, number>([
  ["1_000", 5],
  ["100", 5],
]);

/**
 * Salmond's clustering reduction is far cheaper than Runnalls --- a whole pass
 * per iteration rather than a rescan per merge --- so it can be timed at 10,000
 * too. Still O(N^2) per pass, so 100,000 stays out of reach.
 */
const SALMOND_SCENARIOS = new Map<string, number>([
  ["1_000", 10],
  ["5_000", 10],
  ["10_000", 10],
]);

export type BenchmarkMethod = "merge" | "runnalls" | "salmond";

/**
 * One timed configuration. Held as numbers rather than formatted strings so the
 * chart can fit a scaling exponent and place a frame-budget line; formatting is
 * the renderer's job.
 */
export interface BenchmarkResult {
  method: BenchmarkMethod;
  /** Dataset size. */
  n: number;
  /** Kernels emitted. The merge's count is the budget the others are held to. */
  kernels: number;
  samples: number;
  meanMs: number;
  minMs: number;
  maxMs: number;
  p75Ms: number;
  p99Ms: number;
  p995Ms: number;
  p999Ms: number;
  /** 95% relative margin of error, or null when there is only one sample. */
  rmePct: number | null;
}

export async function runBenchmarkPage(
  onProgress?: (message: string) => void,
): Promise<BenchmarkResult[]> {
  const downsampler = await wasmMerge();
  const samples = [
    ["1_000", 1_000, createNoiseFloatData(1_000), 1_000],
    ["10_000", 10_000, createNoiseFloatData(10_000), 1_000],
    ["100_000", 100_000, createNoiseFloatData(100_000), 497],
    ["1000_000", 1_000_000, createNoiseFloatData(1_000_000), 50],
  ] as const;

  const mergedCount = new Map<string, number>();
  for (const [name, , sample] of samples) {
    mergedCount.set(name, calc(downsampler, sample));
  }

  const rows: BenchmarkResult[] = [];
  for (const [name, n, sample, iterations] of [...samples].reverse()) {
    onProgress?.(`merge, N = ${name} (${iterations} iterations)`);
    downsampler.setDataF64(sample.dataF64);
    const durations: number[] = [];
    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      calcWithoutData(downsampler, sample);
      durations.push(performance.now() - start);
    }
    rows.push(makeResult("merge", n, mergedCount.get(name) ?? 0, durations));
  }

  // Runnalls runs at the same kernel budget the merge produced, so the two
  // methods are timed for the same amount of output.
  const klDownsampler = await wasmKlMerge();
  for (const [name, n, sample] of samples) {
    const iterations = KL_SCENARIOS.get(name);
    if (iterations === undefined) continue;

    const target = mergedCount.get(name) ?? 0;
    // Uploaded once, outside the timed region, exactly as for the merge rows.
    klDownsampler.setDataF64(sample.dataF64);
    const durations: number[] = [];
    for (let i = 0; i < iterations; i++) {
      onProgress?.(
        `Runnalls, N = ${name} (${i + 1}/${iterations}, this is slow)`,
      );
      // Yield so the progress message paints before the blocking wasm call.
      await waitForNextFrame();
      const start = performance.now();
      const count = calcKl(klDownsampler, sample, target);
      durations.push(performance.now() - start);
      if (count !== target) {
        throw new Error(
          `Runnalls returned ${count} kernels, expected ${target}`,
        );
      }
    }
    rows.push(makeResult("runnalls", n, target, durations));
  }

  // Salmond clusters in groups, so it can only guarantee *at most* the merge's
  // kernel budget. The achieved count is read back rather than asserted.
  const salmondDownsampler = await wasmSalmondMerge();
  for (const [name, n, sample] of samples) {
    const iterations = SALMOND_SCENARIOS.get(name);
    if (iterations === undefined) continue;

    const target = mergedCount.get(name) ?? 0;
    salmondDownsampler.setDataF64(sample.dataF64);
    const durations: number[] = [];
    let achieved = 0;
    for (let i = 0; i < iterations; i++) {
      onProgress?.(`Salmond, N = ${name} (${i + 1}/${iterations})`);
      await waitForNextFrame();
      const start = performance.now();
      achieved = calcSalmond(salmondDownsampler, sample, target);
      durations.push(performance.now() - start);
    }
    rows.push(makeResult("salmond", n, achieved, durations));
  }

  return rows;
}

function waitForNextFrame(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => resolve());
  });
}

function makeResult(
  method: BenchmarkMethod,
  n: number,
  kernels: number,
  durationsMs: number[],
): BenchmarkResult {
  const sorted = [...durationsMs].sort((a, b) => a - b);
  const count = sorted.length;
  const mean = sorted.reduce((acc, v) => acc + v, 0) / Math.max(count, 1);
  const variance =
    count > 1
      ? sorted.reduce((acc, v) => acc + (v - mean) * (v - mean), 0) /
        (count - 1)
      : 0;
  const stdErr = Math.sqrt(variance / Math.max(count, 1));

  return {
    method,
    n,
    kernels,
    samples: count,
    meanMs: mean,
    minMs: sorted[0]!,
    maxMs: sorted[count - 1]!,
    p75Ms: quantile(sorted, 0.75),
    p99Ms: quantile(sorted, 0.99),
    p995Ms: quantile(sorted, 0.995),
    p999Ms: quantile(sorted, 0.999),
    rmePct: count > 1 && mean > 0 ? (1.96 * stdErr * 100) / mean : null,
  };
}

function quantile(sorted: number[], q: number) {
  const index = Math.min(sorted.length - 1, Math.floor(q * sorted.length));
  return sorted[Math.max(0, index)]!;
}

function calc(
  downsampler: Awaited<ReturnType<typeof wasmMerge>>,
  { dataF64, yMin, yMax }: ReturnType<typeof createNoiseFloatData>,
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
    throw new Error("Expected merged points for benchmark");
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
    throw new Error("Expected merged points for benchmark");
  }
  return result.count;
}

function calcSalmond(
  downsampler: Awaited<ReturnType<typeof wasmSalmondMerge>>,
  { yMin, yMax }: ReturnType<typeof createNoiseFloatData>,
  targetCount: number,
) {
  downsampler.setViewMinX(0);
  downsampler.setViewMaxX(1);
  downsampler.setViewMinY(yMin);
  downsampler.setViewMaxY(yMax);
  downsampler.setScreenW(1920);
  downsampler.setScreenH(1080);
  downsampler.setSigmaSizePx(16);
  downsampler.setTargetCount(targetCount);

  const result = downsampler.mergePoints();
  if (result.count <= 0) {
    throw new Error("Expected reduced kernels for benchmark");
  }
  return result.count;
}

function calcKl(
  downsampler: Awaited<ReturnType<typeof wasmKlMerge>>,
  { yMin, yMax }: ReturnType<typeof createNoiseFloatData>,
  targetCount: number,
) {
  downsampler.setViewMinX(0);
  downsampler.setViewMaxX(1);
  downsampler.setViewMinY(yMin);
  downsampler.setViewMaxY(yMax);
  downsampler.setScreenW(1920);
  downsampler.setScreenH(1080);
  downsampler.setSigmaSizePx(16);
  downsampler.setTargetCount(targetCount);

  const result = downsampler.mergePoints();
  if (result.count <= 0) {
    throw new Error("Expected reduced kernels for benchmark");
  }
  return result.count;
}

