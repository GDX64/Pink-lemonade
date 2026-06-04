import { createNoiseFloatData } from "../chart/chart";
import { wasmMerge } from "../examples/rasterizing/downsampling";

export async function runBenchmarkPage() {
  const downsampler = await wasmMerge();
  const samples = [
    ["1_000", createNoiseFloatData(1_000), 16_570],
    ["10_000", createNoiseFloatData(10_000), 4_108],
    ["100_000", createNoiseFloatData(100_000), 497],
    ["1000_000", createNoiseFloatData(1_000_000), 50],
  ] as const;

  const mergedCount = new Map<string, number>();
  for (const [name, sample] of samples) {
    mergedCount.set(name, calc(downsampler, sample));
  }

  const rows: Array<Record<string, string>> = [];
  for (const [name, sample, iterations] of [...samples].reverse()) {
    downsampler.setDataF64(sample.dataF64);
    const durations: number[] = [];
    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      calcWithoutData(downsampler, sample);
      durations.push(performance.now() - start);
    }
    rows.push(makeBenchRow(name, mergedCount.get(name) ?? 0, durations));
  }

  return rows;
}

function makeBenchRow(name: string, merged: number, durationsMs: number[]) {
  const sorted = [...durationsMs].sort((a, b) => a - b);
  const n = sorted.length;
  const mean = sorted.reduce((acc, v) => acc + v, 0) / Math.max(n, 1);
  const variance =
    n > 1
      ? sorted.reduce((acc, v) => acc + (v - mean) * (v - mean), 0) / (n - 1)
      : 0;
  const stdErr = Math.sqrt(variance / Math.max(n, 1));
  const rme = mean > 0 ? (1.96 * stdErr * 100) / mean : 0;

  return {
    name: `N = ${name} | merged=${merged}`,
    hz: (1000 / Math.max(mean, 1e-9)).toFixed(2),
    min: sorted[0]!.toFixed(4),
    max: sorted[n - 1]!.toFixed(4),
    mean: mean.toFixed(4),
    p75: quantile(sorted, 0.75).toFixed(4),
    p99: quantile(sorted, 0.99).toFixed(4),
    p995: quantile(sorted, 0.995).toFixed(4),
    p999: quantile(sorted, 0.999).toFixed(4),
    rme: `±${rme.toFixed(2)}%`,
    samples: String(n),
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
