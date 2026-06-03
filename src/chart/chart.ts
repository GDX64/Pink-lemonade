export function gausianNoise(
  mean: number,
  stdDev: number,
  gen: () => number,
): number {
  let u1 = gen();
  let u2 = gen();
  let z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return z0 * stdDev + mean;
}

export interface DrawChartOptions {
  viewMinX: number;
  viewMaxX: number;
  drawBars?: boolean;
}

export type XYDataPoint = [number, number];

export type SplatKernel = "bilinear" | "quadratic" | "cubic" | "triangular";

export function createNoiseData(
  N: number,
  seed: number,
): [number, number, number][] {
  const mean = 0;
  const stdDev = 1;
  const gen = createGen(seed);
  let acc = N / 1000;
  let timeAcc = Date.now();
  const data: [number, number, number][] = [];
  for (let i = 0; i < N; i++) {
    acc += gausianNoise(mean, stdDev, gen);
    timeAcc += 1000;

    const x = i / N; //[0, 1] range for the splat kernel
    const density =
      gaussianSample(0.9, 0.05, x) + gaussianSample(0.3, 0.1, x) * 0.8;

    const s = density + gausianNoise(mean, stdDev, gen) * 0.5;
    const w = Math.abs(s ** 5 * 100);
    data.push([timeAcc, acc, w]);
  }
  return data;
}

function gaussianSample(mean: number, stdDev: number, x: number): number {
  return Math.exp(-0.5 * ((x - mean) / stdDev) ** 2);
}

function* pseudoRandomGen(seed: number): Generator<number> {
  let value = seed;
  while (true) {
    value = (value * 16807) % 2147483647;
    yield value / 2147483647;
  }
}

function createGen(seed = 12345): () => number {
  const localGen = pseudoRandomGen(seed);
  return () => localGen.next().value;
}
