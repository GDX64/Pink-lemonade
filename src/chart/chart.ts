function gausianNoise(mean: number, stdDev: number): number {
  let u1 = nextRandom();
  let u2 = nextRandom();
  let z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return z0 * stdDev + mean;
}

export interface GaussianKernelSeriesPoint {
  centerX: number;
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
}

export function createGaussianKernelSeries(
  data: number[],
  hx: number,
): GaussianKernelSeriesPoint[] {
  if (!Number.isFinite(hx) || hx <= 0) {
    throw new RangeError("hx must be a finite number greater than 0");
  }

  const normalization = 1 / (hx * Math.sqrt(2 * Math.PI));
  const supportRadius = hx * 3;

  return data.map((y, centerX) => ({
    centerX,
    y,
    sigmaX: hx,
    normalization,
    supportRadius,
  }));
}

export function sampleGaussianKernelAtX(
  kernel: GaussianKernelSeriesPoint,
  x: number,
): number {
  const distance = x - kernel.centerX;
  const variance = kernel.sigmaX * kernel.sigmaX;
  return (
    kernel.normalization * Math.exp(-(distance * distance) / (2 * variance))
  );
}

export function drawGaussianKernelSeries(
  kernels: GaussianKernelSeriesPoint[],
  canvas: HTMLCanvasElement | OffscreenCanvas,
  options: DrawGaussianKernelSeriesOptions = {},
): void {
  if (!kernels.length) {
    return;
  }

  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Failed to get 2D context");
  }

  const width = canvas.width;
  const height = canvas.height;
  const backgroundFill = options.backgroundFill ?? "white";
  const lineWidth = options.lineWidth ?? 1.5;
  const strokeStyle = options.strokeStyle ?? "rgba(0, 0, 0, 0.45)";

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const kernel of kernels) {
    minX = Math.min(minX, kernel.centerX - kernel.supportRadius);
    maxX = Math.max(maxX, kernel.centerX + kernel.supportRadius);
    minY = Math.min(minY, kernel.y);
    maxY = Math.max(maxY, kernel.y);
  }

  ctx.save();
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, width, height);
  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = strokeStyle;

  drawGaussianKernelHeatmap(ctx, kernels, width, height, {
    minX,
    maxX,
    minY,
    maxY,
    binsX: Math.max(8, Math.floor(options.heatmapBinsX ?? width / 4)),
    binsY: Math.max(8, Math.floor(options.heatmapBinsY ?? height / 4)),
    opacity: Math.max(0, Math.min(1, options.heatmapOpacity ?? 0.9)),
    sigmaYBins: Math.max(0.25, options.heatmapSigmaYBins ?? 1.25),
  });
  ctx.restore();
}

interface DrawGaussianKernelHeatmapBounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  binsX: number;
  binsY: number;
  opacity: number;
  sigmaYBins: number;
}

function drawGaussianKernelHeatmap(
  ctx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D,
  kernels: GaussianKernelSeriesPoint[],
  width: number,
  height: number,
  bounds: DrawGaussianKernelHeatmapBounds,
): void {
  const { minX, maxX, minY, maxY, binsX, binsY, opacity, sigmaYBins } = bounds;

  const spanX = Math.max(maxX - minX, Number.EPSILON);
  const spanY = Math.max(maxY - minY, Number.EPSILON);
  const density = new Float32Array(binsX * binsY);
  const yRadiusBins = Math.max(1, Math.ceil(sigmaYBins * 3));

  function xBinToDataX(xBin: number): number {
    if (binsX === 1) {
      return minX;
    }
    return minX + (xBin / (binsX - 1)) * spanX;
  }

  function dataXToBin(x: number): number {
    if (binsX === 1) {
      return 0;
    }
    return Math.round(((x - minX) / spanX) * (binsX - 1));
  }

  function dataYToBin(y: number): number {
    if (binsY === 1) {
      return 0;
    }
    return Math.round(((y - minY) / spanY) * (binsY - 1));
  }

  for (const kernel of kernels) {
    const xStartBin = Math.max(
      0,
      dataXToBin(kernel.centerX - kernel.supportRadius),
    );
    const xEndBin = Math.min(
      binsX - 1,
      dataXToBin(kernel.centerX + kernel.supportRadius),
    );
    const yCenterBin = Math.max(0, Math.min(binsY - 1, dataYToBin(kernel.y)));

    for (let xBin = xStartBin; xBin <= xEndBin; xBin++) {
      const dataX = xBinToDataX(xBin);
      const wx = sampleGaussianKernelAtX(kernel, dataX) / kernel.normalization;
      if (wx <= 0) {
        continue;
      }

      const yStart = Math.max(0, yCenterBin - yRadiusBins);
      const yEnd = Math.min(binsY - 1, yCenterBin + yRadiusBins);
      for (let yBin = yStart; yBin <= yEnd; yBin++) {
        const dyBins = yBin - yCenterBin;
        const wy = Math.exp(-(dyBins * dyBins) / (2 * sigmaYBins * sigmaYBins));
        const densityIndex = yBin * binsX + xBin;
        density[densityIndex] = (density[densityIndex] ?? 0) + wx * wy;
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

  const cellWidth = width / binsX;
  const cellHeight = height / binsY;
  for (let yBin = 0; yBin < binsY; yBin++) {
    for (let xBin = 0; xBin < binsX; xBin++) {
      const value = density[yBin * binsX + xBin] ?? 0;
      if (value <= 0) {
        continue;
      }

      const t = Math.max(0, Math.min(1, value / maxDensity));
      ctx.fillStyle = heatmapColor(t);

      const px = xBin * cellWidth;
      const py = height - (yBin + 1) * cellHeight;
      ctx.fillRect(px, py, cellWidth + 0.5, cellHeight + 0.5);
    }
  }
}

function heatmapColor(t: number): string {
  const color = Math.round(t * 255);
  return `rgba(${color},${color},${color}, 1)`;
}

export function createNoiseData(N: number): number[] {
  const mean = 0;
  const stdDev = 1;
  let acc = gausianNoise(mean, stdDev);
  const data = [acc];
  for (let i = 1; i < N; i++) {
    acc += gausianNoise(mean, stdDev);
    data.push(acc);
  }
  return data;
}

export function drawChart(data: number[], canvas: HTMLCanvasElement): void {
  if (!data.length) {
    return;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Failed to get 2D context");
  }

  const width = canvas.width;
  const height = canvas.height;
  const maxData = Math.max(...data);
  const minData = Math.min(...data);

  function scaleX(value: number): number {
    return (value / (data.length - 1)) * width;
  }
  function scaleY(value: number): number {
    return height - ((value - minData) / (maxData - minData)) * height;
  }

  ctx.rect(0, 0, width, height);
  ctx.fillStyle = "white";
  ctx.fill();
  ctx.strokeStyle = "black";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(scaleX(0), scaleY(data[0]!));
  for (let i = 1; i < data.length; i++) {
    ctx.lineTo(scaleX(i), scaleY(data[i]!));
  }
  ctx.stroke();
}

export function drawSlidingHistogram(
  data: { xValue: number; hist: Map<number, number>; deltaX: number }[],
  canvas: OffscreenCanvas,
  binSize: number,
): void {
  let maxX = 0;
  let maxY = 0;
  let minX = 0;
  let minY = Infinity;
  let maxCount = 0;
  for (const { xValue, hist } of data) {
    maxX = Math.max(maxX, xValue);
    for (const [bin, count] of hist.entries()) {
      maxCount = Math.max(maxCount, count);
      maxY = Math.max(maxY, bin);
      minY = Math.min(minY, bin);
    }
  }
  const width = canvas.width;
  const height = canvas.height;
  function scaleX(value: number): number {
    return (value / maxX) * width;
  }
  function scaleY(value: number): number {
    return height - ((value - minY) / (maxY - minY)) * height;
  }
  function alphaScale(count: number): number {
    return count / maxCount;
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
  data: number[],
  each: number,
  binSize: number,
) {
  const windows = windowSlices(data, each);
  return windows.map((window, index) => {
    const hist = createHistogram(window, binSize);
    const data: { xValue: number; hist: Map<number, number>; deltaX: number } =
      {
        xValue: index * each,
        hist: hist,
        deltaX: each,
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

export function windowSlices(data: number[], each: number): number[][] {
  const windows: number[][] = [];
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
