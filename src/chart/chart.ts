function gausianNoise(mean: number, stdDev: number): number {
  let u1 = nextRandom();
  let u2 = nextRandom();
  let z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return z0 * stdDev + mean;
}

export interface GaussianKernelSeriesPoint {
  x: number;
  y: number;
  sigmaX: number;
  normalization: number;
  supportRadius: number;
}

export interface DrawGaussianKernelSeriesOptions {
  mode?: "lines" | "heatmap";
  strokeStyle?: string;
  lineWidth?: number;
  amplitudePx?: number;
  samplesPerSigma?: number;
  backgroundFill?: string | null;
  heatmapBinsX?: number;
  heatmapBinsY?: number;
  heatmapOpacity?: number;
  heatmapSigmaYBins?: number;
  kernelSigmaXPx?: number;
  kernelSupportSigma?: number;
}

export interface DrawChartOptions {
  viewMinX?: number;
  viewMaxX?: number;
}

export type XYDataPoint = [number, number];

function toXYDataPoints(data: number[] | XYDataPoint[]): XYDataPoint[] {
  if (!data.length) {
    return [];
  }

  if (Array.isArray(data[0])) {
    return data as XYDataPoint[];
  }

  return (data as number[]).map((y, index) => [index, y]);
}

export function createGaussianKernel(
  x: number,
  y: number,
  hx: number,
): GaussianKernelSeriesPoint {
  if (!Number.isFinite(hx) || hx <= 0) {
    throw new RangeError("hx must be a finite number greater than 0");
  }

  const normalization = 1 / (hx * Math.sqrt(2 * Math.PI));
  const supportRadius = hx * 5;

  return {
    x,
    y,
    sigmaX: hx,
    normalization,
    supportRadius,
  };
}

export function createGaussianKernelSeries(
  data: number[] | XYDataPoint[],
  hx: number,
): GaussianKernelSeriesPoint[] {
  return toXYDataPoints(data).map(([x, y]) => createGaussianKernel(x, y, hx));
}

export function sampleGaussianKernelAtDistance(
  kernel: GaussianKernelSeriesPoint,
  px: number,
  py: number,
): number {
  const variance = kernel.sigmaX * kernel.sigmaX;
  const dx = px - kernel.x;
  const dy = py - kernel.y;
  const distanceFromCenterPx = Math.sqrt(dx * dx + dy * dy);
  return (
    kernel.normalization *
    Math.exp(-(distanceFromCenterPx * distanceFromCenterPx) / (2 * variance))
  );
}

export function drawGaussianKernelSeries(
  data: number[] | XYDataPoint[],
  canvas: HTMLCanvasElement | OffscreenCanvas,
  options: DrawGaussianKernelSeriesOptions = {},
): void {
  const points = toXYDataPoints(data);
  if (!points.length) {
    return;
  }

  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Failed to get 2D context");
  }

  const width = canvas.width;
  const height = canvas.height;
  const sigmaXPx = Math.max(0.5, options.kernelSigmaXPx ?? 6);
  const supportSigma = Math.max(2, options.kernelSupportSigma ?? 5);
  const binsX = width;
  const binsY = height;
  const opacity = Math.max(0, Math.min(1, options.heatmapOpacity ?? 1));

  let minXValue = Infinity;
  let maxXValue = -Infinity;
  let minYValue = Infinity;
  let maxYValue = -Infinity;
  for (const [x, y] of points) {
    minXValue = Math.min(minXValue, x);
    maxXValue = Math.max(maxXValue, x);
    minYValue = Math.min(minYValue, y);
    maxYValue = Math.max(maxYValue, y);
  }

  const xSpan = Math.max(maxXValue - minXValue, Number.EPSILON);
  const ySpan = Math.max(maxYValue - minYValue, Number.EPSILON);

  const scaleX = (value: number): number => {
    if (maxXValue === minXValue) {
      return width / 2;
    }
    return ((value - minXValue) / xSpan) * (width - 1);
  };

  const scaleY = (value: number): number => {
    if (maxYValue === minYValue) {
      return height / 2;
    }
    return height - 1 - ((value - minYValue) / ySpan) * (height - 1);
  };

  const kernels = points.map(([xValue, yValue]) => {
    const xPx = scaleX(xValue);
    const yPx = scaleY(yValue);
    const kernel = createGaussianKernel(xPx, yPx, sigmaXPx);
    return {
      ...kernel,
      supportRadius: kernel.sigmaX * supportSigma,
    };
  });

  const density = new Float32Array(binsX * binsY);
  const xScale = binsX > 1 ? (width - 1) / (binsX - 1) : 1;
  const yScale = binsY > 1 ? (height - 1) / (binsY - 1) : 1;

  for (const kernel of kernels) {
    const xCenterBin = Math.round(kernel.x / xScale);
    const yCenterBin = Math.round(kernel.y / yScale);
    const xRadiusBins = Math.max(1, Math.ceil(kernel.supportRadius / xScale));
    const yRadiusBins = xRadiusBins;

    const xStartBin = Math.max(0, xCenterBin - xRadiusBins);
    const xEndBin = Math.min(binsX - 1, xCenterBin + xRadiusBins);

    for (let xBin = xStartBin; xBin <= xEndBin; xBin++) {
      const pixelX = xBin * xScale;

      const yStart = Math.max(0, yCenterBin - yRadiusBins);
      const yEnd = Math.min(binsY - 1, yCenterBin + yRadiusBins);
      for (let yBin = yStart; yBin <= yEnd; yBin++) {
        const pixelY = yBin * yScale;
        const w = sampleGaussianKernelAtDistance(kernel, pixelX, pixelY);
        const densityIndex = yBin * binsX + xBin;
        density[densityIndex] = (density[densityIndex] ?? 0) + w;
      }
    }
  }

  let maxDensity = 0;
  for (let i = 0; i < density.length; i++) {
    maxDensity = Math.max(maxDensity, density[i] ?? 0);
  }
  if (maxDensity <= 0) {
    return;
  }

  const imageData = new ImageData(width, height);
  const cellWidth = width / binsX;
  const cellHeight = height / binsY;
  for (let yBin = 0; yBin < binsY; yBin++) {
    for (let xBin = 0; xBin < binsX; xBin++) {
      const value = density[yBin * binsX + xBin] ?? 0;
      if (value <= 0) {
        continue;
      }

      const t = Math.max(0, Math.min(1, value / maxDensity));
      const color = Math.floor(t * (2 ** 24 - 1));
      const r = (color >> 16) & 0xff;
      const g = (color >> 8) & 0xff;
      const b = color & 0xff;
      const a = Math.floor(opacity * 255);

      const px = xBin * cellWidth;
      const py = yBin * cellHeight;
      const xStart = Math.floor(px);
      const xEnd = Math.floor(px + cellWidth);
      const yStart = Math.floor(py);
      const yEnd = Math.floor(py + cellHeight);

      for (let yy = yStart; yy < yEnd; yy++) {
        for (let xx = xStart; xx < xEnd; xx++) {
          if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
            const index = (yy * width + xx) * 4;
            imageData.data[index] = r;
            imageData.data[index + 1] = g;
            imageData.data[index + 2] = b;
            imageData.data[index + 3] = a;
          }
        }
      }
    }
  }
  ctx.putImageData(imageData, 0, 0);
  ctx.restore();
}

export function drawSplatKernelSeries(
  data: XYDataPoint[],
  args: { width: number; height: number; viewMinX?: number; viewMaxX?: number },
): Float32Array {
  const points = toXYDataPoints(data);
  if (!points.length) {
    throw new Error("Data must contain at least one point");
  }
  const width = args.width;
  const height = args.height;

  const binsX = width;
  const binsY = height;

  let dataMinX = Infinity;
  let dataMaxX = -Infinity;
  let minYValue = Infinity;
  let maxYValue = -Infinity;
  for (const [x, y] of points) {
    dataMinX = Math.min(dataMinX, x);
    dataMaxX = Math.max(dataMaxX, x);
    minYValue = Math.min(minYValue, y);
    maxYValue = Math.max(maxYValue, y);
  }

  const unclampedViewMinX = args.viewMinX ?? dataMinX;
  const unclampedViewMaxX = args.viewMaxX ?? dataMaxX;
  const sortedViewMinX = Math.min(unclampedViewMinX, unclampedViewMaxX);
  const sortedViewMaxX = Math.max(unclampedViewMinX, unclampedViewMaxX);
  const minXValue = Math.max(dataMinX, sortedViewMinX);
  const maxXValue = Math.min(dataMaxX, sortedViewMaxX);
  const visiblePoints = points.filter(
    ([x]) => x >= minXValue && x <= maxXValue,
  );
  if (!visiblePoints.length) {
    return new Float32Array(binsX * binsY);
  }

  const xSpan = Math.max(maxXValue - minXValue, Number.EPSILON);
  const ySpan = Math.max(maxYValue - minYValue, Number.EPSILON);

  const scaleX = (value: number): number => {
    if (maxXValue === minXValue) {
      return width / 2;
    }
    return ((value - minXValue) / xSpan) * (width - 1);
  };

  const scaleY = (value: number): number => {
    if (maxYValue === minYValue) {
      return height / 2;
    }
    return height - 1 - ((value - minYValue) / ySpan) * (height - 1);
  };

  const density = new Float32Array(binsX * binsY);
  const splatRadius = 1.5;
  for (const [xValue, yValue] of visiblePoints) {
    const x = scaleX(xValue);
    const y = scaleY(yValue);
    const centerX = Math.round(x);
    const centerY = Math.round(y);

    const weightsX = [0, 0, 0];
    const weightsY = [0, 0, 0];
    for (let offset = -1; offset <= 1; offset++) {
      const index = offset + 1;
      weightsX[index] =
        Math.max(0, 1 - Math.abs(x - (centerX + offset)) / splatRadius) ?? 0;
      weightsY[index] =
        Math.max(0, 1 - Math.abs(y - (centerY + offset)) / splatRadius) ?? 0;
    }

    let totalWeight = 0;
    for (let oy = 0; oy < 3; oy++) {
      for (let ox = 0; ox < 3; ox++) {
        totalWeight += (weightsX[ox] ?? 0) * (weightsY[oy] ?? 0);
      }
    }
    if (totalWeight <= 0) {
      continue;
    }

    for (let offsetY = -1; offsetY <= 1; offsetY++) {
      const yBin = centerY + offsetY;
      if (yBin < 0 || yBin >= binsY) {
        continue;
      }

      for (let offsetX = -1; offsetX <= 1; offsetX++) {
        const xBin = centerX + offsetX;
        if (xBin < 0 || xBin >= binsX) {
          continue;
        }

        const weight =
          ((weightsX[offsetX + 1] ?? 0) * (weightsY[offsetY + 1] ?? 0)) /
          totalWeight;
        const index = yBin * binsX + xBin;
        density[index] = (density[index] ?? 0) + weight;
      }
    }
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

  return density;
}

export function createNoiseData(N: number): [number, number][] {
  const mean = 0;
  const stdDev = 1;
  let acc = gausianNoise(mean, stdDev);
  let timeAcc = 0;
  const data: [number, number][] = [[timeAcc, acc]];
  for (let i = 1; i < N; i++) {
    acc += gausianNoise(mean, stdDev);
    timeAcc += Math.abs(
      gausianNoise(mean, stdDev) *
        Math.sin((i / N) * Math.PI * 2 * 2 + Math.PI / 4),
    );
    data.push([timeAcc, acc]);
  }
  return data;
}

export function drawChart(
  data: [number, number][],
  canvas: HTMLCanvasElement,
  options: DrawChartOptions = {},
): void {
  if (!data.length) {
    return;
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
  for (const [x, y] of data) {
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minData = Math.min(minData, y);
    maxData = Math.max(maxData, y);
  }
  const ySpan = Math.max(maxData - minData, Number.EPSILON);
  const unclampedViewMinX = options.viewMinX ?? minX;
  const unclampedViewMaxX = options.viewMaxX ?? maxX;
  const sortedViewMinX = Math.min(unclampedViewMinX, unclampedViewMaxX);
  const sortedViewMaxX = Math.max(unclampedViewMinX, unclampedViewMaxX);
  const viewMinX = Math.max(minX, sortedViewMinX);
  const viewMaxX = Math.min(maxX, sortedViewMaxX);
  const viewXSpan = Math.max(viewMaxX - viewMinX, Number.EPSILON);

  function scaleX(value: number): number {
    if (viewMaxX === viewMinX) {
      return width / 2;
    }
    return ((value - viewMinX) / viewXSpan) * width;
  }
  function scaleY(value: number): number {
    if (maxData === minData) {
      return height / 2;
    }
    return height - ((value - minData) / ySpan) * height;
  }

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#a4ff3d";

  const targetBinPx = 16;
  const binCount = Math.max(1, Math.floor(width / targetBinPx));
  const counts = new Uint32Array(binCount);
  for (const [x] of data) {
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

  if (maxCount > 0) {
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
  for (let i = 0; i < data.length; i++) {
    const [x, y] = data[i]!;
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

export function drawSlidingHistogram(
  data: { xValue: number; hist: Map<number, number>; deltaX: number }[],
  canvas: OffscreenCanvas,
  binSize: number,
): void {
  if (!data.length) {
    return;
  }

  let maxX = -Infinity;
  let maxY = 0;
  let minX = Infinity;
  let minY = Infinity;
  let maxCount = 0;
  for (const { xValue, hist, deltaX } of data) {
    minX = Math.min(minX, xValue);
    maxX = Math.max(maxX, xValue + deltaX);
    for (const [bin, count] of hist.entries()) {
      maxCount = Math.max(maxCount, count);
      maxY = Math.max(maxY, bin);
      minY = Math.min(minY, bin);
    }
  }

  if (!Number.isFinite(minX) || !Number.isFinite(minY) || maxCount <= 0) {
    return;
  }

  const width = canvas.width;
  const height = canvas.height;
  const xSpan = Math.max(maxX - minX, Number.EPSILON);
  const ySpan = Math.max(maxY - minY, Number.EPSILON);

  function scaleX(value: number): number {
    return ((value - minX) / xSpan) * width;
  }
  function scaleY(value: number): number {
    return height - ((value - minY) / ySpan) * height;
  }
  function alphaScale(count: number): number {
    return maxCount > 0 ? count / maxCount : 0;
  }

  //draw as rectangles
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Failed to get 2D context");
  }
  const scaledBinsize = Math.abs(scaleY(binSize) - scaleY(0));
  for (const { xValue, hist, deltaX } of data) {
    const x = scaleX(xValue);
    const scaledDeltaX = scaleX(xValue + deltaX) - x;
    for (const [bin, count] of hist.entries()) {
      const y = scaleY(bin);
      drawDot(ctx, x, scaledDeltaX, y, scaledBinsize, alphaScale(count));
    }
  }
}

function drawDot(
  ctx: OffscreenCanvasRenderingContext2D,
  x: number,
  scaledDeltaX: number,
  y: number,
  scaledBinsize: number,
  intensity: number,
) {
  const r = Math.max(0, scaledBinsize) * 2;
  const POWER = 2;
  const cx = x + scaledDeltaX / 2;
  const cy = y + scaledBinsize / 2;
  const clampedIntensity = Math.max(0, Math.min(1, intensity));

  const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(r, 1));
  const stops = 10;
  for (let i = 0; i <= stops; i++) {
    const t = i / stops;
    const alpha = clampedIntensity * Math.pow(1 - t, POWER);
    gradient.addColorStop(t, `rgba(0, 0, 0, ${alpha})`);
  }

  ctx.save();
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, 2 * Math.PI);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

export function blurImage(canvas: OffscreenCanvas, radius: number): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Failed to get 2D context");
  }
  if (!Number.isFinite(radius) || radius <= 0) {
    return;
  }

  const sourceCanvas = document.createElement("canvas");
  sourceCanvas.width = canvas.width;
  sourceCanvas.height = canvas.height;

  const sourceCtx = sourceCanvas.getContext("2d");
  if (!sourceCtx) {
    throw new Error("Failed to get 2D context");
  }

  sourceCtx.drawImage(canvas, 0, 0);

  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.filter = `blur(${radius}px)`;
  ctx.drawImage(sourceCanvas, 0, 0);
  ctx.restore();
}

export function createSlidingHistogram(
  data: number[] | XYDataPoint[],
  each: number,
  binSize: number,
) {
  const points = toXYDataPoints(data);
  const windows = windowSlices(points, each);
  return windows.map((window, index) => {
    const values = window.map(([, value]) => value);
    const hist = createHistogram(values, binSize);
    const xStart = window[0]?.[0] ?? index * each;
    const xEnd = window[window.length - 1]?.[0] ?? xStart;
    const data: { xValue: number; hist: Map<number, number>; deltaX: number } =
      {
        xValue: xStart,
        hist: hist,
        deltaX: Math.max(xEnd - xStart, Number.EPSILON),
      };
    return data;
  });
}

function createHistogram(data: number[], binSize: number): Map<number, number> {
  const histogram = new Map<number, number>();
  for (const value of data) {
    const bin = Math.floor(value / binSize) * binSize;
    histogram.set(bin, (histogram.get(bin) || 0) + 1);
  }
  return histogram;
}

export function windowSlices<T>(data: T[], each: number): T[][] {
  const windows: T[][] = [];
  for (let i = 0; i < data.length; i += each) {
    windows.push(data.slice(i, i + each));
  }
  return windows;
}

function* pseudoRandomGen(seed: number): Generator<number> {
  let value = seed;
  while (true) {
    value = (value * 16807) % 2147483647;
    yield value / 2147483647;
  }
}

const gen = pseudoRandomGen(12345);
function nextRandom(): number {
  return gen.next().value;
}
