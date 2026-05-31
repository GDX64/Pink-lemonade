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

function splatWeights(
  t: number,
  kernel: SplatKernel,
): { weights: number[]; offset: number } {
  if (kernel === "bilinear") {
    return { weights: [1 - t, t], offset: 0 };
  }
  if (kernel === "quadratic") {
    const u = t - 0.5;
    return {
      weights: [
        0.5 * (0.5 - u) * (0.5 - u),
        0.75 - u * u,
        0.5 * (0.5 + u) * (0.5 + u),
      ],
      offset: -1,
    };
  }
  if (kernel === "cubic") {
    return {
      weights: [
        ((1 - t) * (1 - t) * (1 - t)) / 6,
        (3 * t * t * t - 6 * t * t + 4) / 6,
        (-3 * t * t * t + 3 * t * t + 3 * t + 1) / 6,
        (t * t * t) / 6,
      ],
      offset: -1,
    };
  }
  // triangular: tent function spanning 64 bins, normalized so weights sum to 1
  const R = 4;
  const weights: number[] = [];
  for (let k = -(R - 1); k <= R; k++) {
    weights.push(Math.max(0, (R - Math.abs(t - k)) / (R * R)));
  }
  return { weights, offset: -(R - 1) };
}

export function drawSplatKernelSeries(
  data: Float64Array,
  args: {
    width: number;
    height: number;
    viewMinX?: number;
    viewMaxX?: number;
    kernel?: SplatKernel;
  },
) {
  if (!data.length) {
    throw new Error("Data must contain at least one point");
  }
  if (data.length % 2 !== 0) {
    throw new Error("Data must contain x, y pairs");
  }
  const width = args.width;
  const height = args.height;

  const binsX = width;
  const binsY = height;

  const unclampedViewMinX = args.viewMinX ?? -Infinity;
  const unclampedViewMaxX = args.viewMaxX ?? Infinity;
  const sortedViewMinX = Math.min(unclampedViewMinX, unclampedViewMaxX);
  const sortedViewMaxX = Math.max(unclampedViewMinX, unclampedViewMaxX);

  let dataMinX = Infinity;
  let dataMaxX = -Infinity;
  let minYValue = Infinity;
  let maxYValue = -Infinity;
  const len = data.length;
  for (let i = 0; i < len; i += 2) {
    const x = data[i]!;
    if (x < sortedViewMinX || x > sortedViewMaxX) {
      continue;
    }
    const y = data[i + 1]!;
    dataMinX = Math.min(dataMinX, x);
    dataMaxX = Math.max(dataMaxX, x);
    minYValue = Math.min(minYValue, y);
    maxYValue = Math.max(maxYValue, y);
  }

  const minXValue = Math.max(dataMinX, sortedViewMinX);
  const maxXValue = Math.min(dataMaxX, sortedViewMaxX);
  if (maxXValue < minXValue) {
    throw new Error("No data points are within the specified view range");
  }

  const xSpan = Math.max(maxXValue - minXValue, Number.EPSILON);
  const ySpan = Math.max(maxYValue - minYValue, Number.EPSILON);

  const scaleX = (value: number): number => {
    return ((value - minXValue) / xSpan) * binsX;
  };

  const scaleY = (value: number): number => {
    return binsY - ((value - minYValue) / ySpan) * binsY;
  };

  const kernel = args.kernel ?? "bilinear";
  const density = new Float32Array(binsX * binsY);
  let hasVisiblePoints = false;
  for (let i = 0; i < len; i += 2) {
    const xValue = data[i]!;
    if (xValue < minXValue || xValue > maxXValue) {
      continue;
    }
    hasVisiblePoints = true;
    const yValue = data[i + 1]!;
    const x = scaleX(xValue);
    const y = scaleY(yValue);
    const xBase = Math.floor(x - 0.5);
    const yBase = Math.floor(y - 0.5);
    const { weights: wx, offset: xOffset } = splatWeights(
      x - 0.5 - xBase,
      kernel,
    );
    const { weights: wy, offset: yOffset } = splatWeights(
      y - 0.5 - yBase,
      kernel,
    );
    for (let oi = 0; oi < wx.length; oi++) {
      const bx = xBase + xOffset + oi;
      if (bx < 0 || bx >= binsX) continue;
      for (let oj = 0; oj < wy.length; oj++) {
        const by = yBase + yOffset + oj;
        if (by < 0 || by >= binsY) continue;
        density[by * binsX + bx]! += wx[oi]! * wy[oj]!;
      }
    }
  }

  if (!hasVisiblePoints) {
    throw new Error("No data points are within the specified view range");
  }

  let maxDensity = 0;
  for (let i = 0; i < density.length; i++) {
    maxDensity = Math.max(maxDensity, density[i] ?? 0);
  }
  if (maxDensity <= 0) {
    throw new Error("Max density must be greater than 0");
  }

  for (let yBin = 0; yBin < binsY; yBin++) {
    for (let xBin = 0; xBin < binsX; xBin++) {
      const value = density[yBin * binsX + xBin] ?? 0;
      if (value <= 0) {
        continue;
      }

      const t = Math.max(0, Math.min(1, value / maxDensity));
      density[yBin * binsX + xBin] = t;
    }
  }

  return { density, scaleX, scaleY };
}

export function createNoiseData(
  N: number,
  seed: number,
): [number, number, number][] {
  const mean = 0;
  const stdDev = 1;
  const gen = createGen(seed);
  let acc = gausianNoise(mean, stdDev, gen);
  let timeAcc = Date.now();
  const data: [number, number, number][] = [];
  for (let i = 0; i < N; i++) {
    acc += gausianNoise(mean, stdDev, gen);
    timeAcc += 1000;
    const s =
      Math.sin((i / N) * Math.PI * 2) + gausianNoise(mean, stdDev, gen) / 10;
    const w = Math.abs(s * s * s * 100);
    data.push([timeAcc, acc, w]);
  }
  return data;
}

export function drawChart(
  data: Float64Array,
  canvas: HTMLCanvasElement,
  options: DrawChartOptions,
): void {
  if (!data.length) {
    return;
  }
  if (data.length % 2 !== 0) {
    throw new Error("Data must contain x, y pairs");
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Failed to get 2D context");
  }

  const width = canvas.width;
  const height = canvas.height;
  let maxData = -Infinity;
  let minData = Infinity;
  let minX = Infinity;
  let maxX = -Infinity;
  const len = data.length;
  for (let i = 0; i < len; i += 2) {
    const x = data[i]!;
    if (x > options.viewMaxX || x < options.viewMinX) {
      continue;
    }
    const y = data[i + 1]!;
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minData = Math.min(minData, y);
    maxData = Math.max(maxData, y);
  }

  if (maxX < minX) {
    return;
  }

  const ySpan = Math.max(maxData - minData, Number.EPSILON);
  const unclampedViewMinX = options.viewMinX;
  const unclampedViewMaxX = options.viewMaxX;
  const sortedViewMinX = Math.min(unclampedViewMinX, unclampedViewMaxX);
  const sortedViewMaxX = Math.max(unclampedViewMinX, unclampedViewMaxX);
  const viewMinX = Math.max(minX, sortedViewMinX);
  const viewMaxX = Math.min(maxX, sortedViewMaxX);
  const viewXSpan = Math.max(viewMaxX - viewMinX, Number.EPSILON);

  function scaleX(value: number): number {
    return ((value - viewMinX) / viewXSpan) * width;
  }
  const YPad = 0;
  const effectiveHeight = height - YPad * 2;
  function scaleY(value: number): number {
    return height - ((value - minData) / ySpan) * effectiveHeight - YPad;
  }

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#a4ff3d";

  const targetBinPx = 16;
  const binCount = Math.max(1, Math.floor(width / targetBinPx));
  const counts = new Uint32Array(binCount);
  for (let i = 0; i < len; i += 2) {
    const x = data[i]!;
    if (x < viewMinX || x > viewMaxX) {
      continue;
    }
    const normalized = (x - viewMinX) / viewXSpan;
    const clamped = Math.max(0, Math.min(1, normalized));
    const binIndex = Math.min(binCount - 1, Math.floor(clamped * binCount));
    counts[binIndex] = (counts[binIndex] ?? 0) + 1;
  }

  let maxCount = 0;
  for (let i = 0; i < counts.length; i++) {
    maxCount = Math.max(maxCount, counts[i] ?? 0);
  }

  if (maxCount > 0 && options.drawBars) {
    const barWidth = width / binCount;
    const histogramHeight = Math.max(24, height * 0.1);
    ctx.save();
    ctx.fillStyle = "rgb(255, 61, 61)";
    for (let i = 0; i < binCount; i++) {
      const count = counts[i] ?? 0;
      if (count <= 0) {
        continue;
      }

      const barHeight = (count / maxCount) * histogramHeight;
      const xStart = i * barWidth;
      const yStart = height - barHeight;
      ctx.fillRect(xStart, yStart, Math.ceil(barWidth), barHeight);
    }
    ctx.restore();
  }

  ctx.lineWidth = 1;
  ctx.beginPath();
  let hasVisiblePoint = false;
  for (let i = 0; i < len; i += 2) {
    const x = data[i]!;
    const y = data[i + 1]!;
    if (x < viewMinX || x > viewMaxX) {
      continue;
    }

    if (!hasVisiblePoint) {
      ctx.moveTo(scaleX(x), scaleY(y));
      hasVisiblePoint = true;
      continue;
    }

    ctx.lineTo(scaleX(x), scaleY(y));
  }

  if (!hasVisiblePoint) {
    return;
  }
  ctx.stroke();
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
