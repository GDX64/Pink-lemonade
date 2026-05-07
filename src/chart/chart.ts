function gausianNoise(mean: number, stdDev: number): number {
  let u1 = nextRandom();
  let u2 = nextRandom();
  let z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return z0 * stdDev + mean;
}

export interface DrawChartOptions {
  viewMinX: number;
  viewMaxX: number;
  drawBars?: boolean;
}

export type XYDataPoint = [number, number];

export function drawSplatKernelSeries(
  data: Float64Array,
  args: { width: number; height: number; viewMinX?: number; viewMaxX?: number },
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
  let hasVisiblePoints = false;
  const weightsX = [0, 0, 0];
  const weightsY = [0, 0, 0];
  for (let i = 0; i < len; i += 2) {
    const xValue = data[i]!;
    if (xValue < minXValue || xValue > maxXValue) {
      continue;
    }
    hasVisiblePoints = true;
    const yValue = data[i + 1]!;
    const x = scaleX(xValue);
    const y = scaleY(yValue);
    const centerX = Math.round(x);
    const centerY = Math.round(y);

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
  const YPad = 20;
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

const gen = pseudoRandomGen(12345);
function nextRandom(): number {
  return gen.next().value;
}
