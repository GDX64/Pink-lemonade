import { createNoiseFloatData } from "../chart/chart";
import { wasmMerge } from "../examples/rasterizing/downsampling";
import { wasmKlMerge } from "../examples/rasterizing/kl-downsampling";
import { wasmSalmondMerge } from "../examples/rasterizing/salmond-downsampling";
import {
  createBenchmarkProgressUi,
  waitForRepaint,
} from "./benchmark-progress";

/**
 * How long to stay inside the blocking timing loop before handing a frame back
 * to the browser. Yields happen *between* timed iterations, never inside one, so
 * the measurements are unaffected --- only wall-clock time grows, by roughly one
 * frame per interval.
 */
const YIELD_INTERVAL_MS = 100;

/**
 * Dataset sizes at which Runnalls' reduction is also benchmarked. It is
 * O(N^2) to set up, so it is capped well below the merge sizes: at N=10,000 a
 * single run already takes ~1 minute, and N=100,000 would take hours.
 */
const KL_SCENARIOS = new Map<string, number>([
  ["1_000", 5],
  ["3_000", 2],
]);

/**
 * Salmond's clustering reduction is far cheaper than Runnalls --- a whole pass
 * per iteration rather than a rescan per merge --- so it can be timed at 10,000
 * too. Still O(N^2) per pass, so 100,000 stays out of reach.
 */
const SALMOND_SCENARIOS = new Map<string, number>([
  ["1_000", 10],
  ["10_000", 5],
  ["100_000", 3],
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

/** Dataset sizes and how many times each is timed for the merge. */
const SCENARIOS = [
  ["1_000", 1_000, 1_000],
  ["3_000", 3_000, 1_000],
  ["10_000", 10_000, 1_000],
  ["100_000", 100_000, 500],
  ["1000_000", 1_000_000, 50],
] as const;

/**
 * One timed configuration, resolved before any measuring starts so the progress
 * bar knows how many there are. `run` performs exactly one reduction and returns
 * the kernel count; the driver times it and owns the loop.
 */
interface Phase {
  method: BenchmarkMethod;
  label: string;
  n: number;
  iterations: number;
  /** Uploads data. Runs once, outside the timed region. */
  prepare: () => void;
  run: () => number;
}

export async function runBenchmarkPage(): Promise<BenchmarkResult[]> {
  const ui = createBenchmarkProgressUi();

  try {
    ui.setPreparing("Loading wasm modules...");
    await waitForRepaint();
    const downsampler = await wasmMerge();
    const klDownsampler = await wasmKlMerge();
    const salmondDownsampler = await wasmSalmondMerge();

    // Generating a million points is itself slow enough to look like a hang, so
    // it reports too.
    const samples = new Map<
      string,
      { n: number; data: ReturnType<typeof createNoiseFloatData> }
    >();
    for (const [name, n] of SCENARIOS) {
      ui.setPreparing(`Generating dataset N = ${n.toLocaleString("en-US")}...`);
      await waitForRepaint();
      samples.set(name, { n, data: createNoiseFloatData(n) });
    }

    ui.setPreparing("Measuring kernel budgets...");
    await waitForRepaint();
    const mergedCount = new Map<string, number>();
    for (const [name, sample] of samples) {
      mergedCount.set(name, calc(downsampler, sample.data));
    }

    const phases: Phase[] = [];

    // Largest first, matching the original ordering.
    for (const [name, n, iterations] of [...SCENARIOS].reverse()) {
      const sample = samples.get(name)!;
      phases.push({
        method: "merge",
        label: `Screen-space merge, N = ${n.toLocaleString("en-US")}`,
        n,
        iterations,
        prepare: () => downsampler.setDataF64(sample.data.dataF64),
        run: () => calcWithoutData(downsampler, sample.data),
      });
    }

    // Runnalls runs at the same kernel budget the merge produced, so the two
    // methods are timed for the same amount of output.
    for (const [name, n] of SCENARIOS) {
      const iterations = KL_SCENARIOS.get(name);
      if (iterations === undefined) continue;

      const sample = samples.get(name)!;
      const target = mergedCount.get(name) ?? 0;
      phases.push({
        method: "runnalls",
        label: `Runnalls KL, N = ${n.toLocaleString("en-US")} (slow)`,
        n,
        iterations,
        prepare: () => klDownsampler.setDataF64(sample.data.dataF64),
        run: () => {
          const count = calcKl(klDownsampler, sample.data, target);
          if (count !== target) {
            throw new Error(
              `Runnalls returned ${count} kernels, expected ${target}`,
            );
          }
          return count;
        },
      });
    }

    // Salmond clusters in groups, so it can only guarantee *at most* the merge's
    // kernel budget. The achieved count is read back rather than asserted.
    for (const [name, n] of SCENARIOS) {
      const iterations = SALMOND_SCENARIOS.get(name);
      if (iterations === undefined) continue;

      const sample = samples.get(name)!;
      const target = mergedCount.get(name) ?? 0;
      phases.push({
        method: "salmond",
        label: `Salmond clustering, N = ${n.toLocaleString("en-US")}`,
        n,
        iterations,
        prepare: () => salmondDownsampler.setDataF64(sample.data.dataF64),
        run: () => calcSalmond(salmondDownsampler, sample.data, target),
      });
    }

    const rows: BenchmarkResult[] = [];
    const startedAt = performance.now();

    for (const [phaseIndex, phase] of phases.entries()) {
      phase.prepare();
      ui.update({
        phaseLabel: phase.label,
        phaseIndex,
        phaseCount: phases.length,
        phaseProgress: 0,
        phaseEtaMs: null,
        elapsedMs: performance.now() - startedAt,
      });
      await waitForRepaint();

      const durations: number[] = [];
      const phaseStart = performance.now();
      let lastYield = phaseStart;
      let kernels = 0;

      for (let i = 0; i < phase.iterations; i++) {
        const start = performance.now();
        kernels = phase.run();
        durations.push(performance.now() - start);

        const now = performance.now();
        const isLast = i === phase.iterations - 1;
        if (now - lastYield < YIELD_INTERVAL_MS && !isLast) continue;

        const completed = i + 1;
        // Wall-clock rate, so the estimate already accounts for yield overhead.
        const perIteration = (now - phaseStart) / completed;
        ui.update({
          phaseLabel: phase.label,
          phaseIndex,
          phaseCount: phases.length,
          phaseProgress: completed / phase.iterations,
          phaseEtaMs: perIteration * (phase.iterations - completed),
          elapsedMs: now - startedAt,
        });
        await waitForRepaint();
        lastYield = performance.now();
      }

      rows.push(makeResult(phase.method, phase.n, kernels, durations));
    }

    return rows;
  } finally {
    ui.done();
  }
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

