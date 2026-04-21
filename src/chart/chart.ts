function gausianNoise(mean: number, stdDev: number): number {
  let u1 = Math.random();
  let u2 = Math.random();
  let z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return z0 * stdDev + mean;
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
  blurRadius = 1,
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
  const scaledBinsize = scaleY(binSize) - scaleY(0);
  for (const { xValue, hist, deltaX } of data) {
    const x = scaleX(xValue);
    const scaledDeltaX = scaleX(xValue + deltaX) - x;
    for (const [bin, count] of hist.entries()) {
      const y = scaleY(bin);
      ctx.fillStyle = `rgba(0, 0, 0, ${alphaScale(count)})`;
      ctx.fillRect(x, y, scaledDeltaX, scaledBinsize);
    }
  }

  blurImage(canvas, blurRadius);
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
