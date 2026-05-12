import { gausianNoise } from "../../chart/chart";

export function example() {
  const canvas = createCanvas();
  const ctx = canvas.getContext("2d")!;

  const values = [...Array(1_000)].map(() => gausianNoise(0, 1));

  const params = { bins: 4, shifts: 2 };

  draw(canvas, ctx, values, params);

  window.addEventListener("resize", () => {
    resizeCanvas(canvas);
    draw(canvas, ctx, values, params);
  });
}

type ShiftedHistogram = {
  origin: number;
  binWidth: number;
  counts: Float64Array;
};

function computeShiftedHistograms(
  values: number[],
  bins: number,
  shifts: number,
): {
  min: number;
  max: number;
  range: number;
  histograms: ShiftedHistogram[];
  acc: Float64Array;
  delta: number;
} {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const h = range / bins;
  const delta = h / shifts;

  const histograms: ShiftedHistogram[] = [];
  const acc = new Float64Array(bins * shifts);

  for (let s = 0; s < shifts; s++) {
    const origin = min - s * delta;
    const counts = new Float64Array(bins);
    for (const v of values) {
      const idx = Math.floor((v - origin) / h);
      if (idx >= 0 && idx < bins) counts[idx]! += 1;
    }

    for (let bin = 0; bin < counts.length; bin++) {
      const count = counts[bin]!;
      for (let shift = 0; shift < shifts; shift++) {
        const accBin = bin * shifts + shift;
        const currentShift = (s + shift) % shifts;
        const weight = currentShift / shifts;
        acc[accBin]! += count * weight;
      }
    }

    histograms.push({ origin, binWidth: h, counts });
  }

  console.log({ acc });

  return { min, max, range, histograms, acc, delta };
}

function remap(
  value: number,
  fromMin: number,
  fromMax: number,
  toMin: number,
  toMax: number,
) {
  const t = (value - fromMin) / (fromMax - fromMin);
  return toMin + t * (toMax - toMin);
}

// Palette of distinct hues for the sub-histograms
const SHIFT_COLORS = [
  "#ff6b6b", // red
  "#ffd93d", // yellow
  "#6bcb77", // green
  "#4d96ff", // blue
  "#c77dff", // purple
  "#ff9f1c", // orange
  "#2ec4b6", // teal
  "#e71d36", // crimson
];

function draw(
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D,
  values: number[],
  params: { bins: number; shifts: number },
) {
  const W = canvas.width;
  const H = canvas.height;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#0d0f14";
  ctx.fillRect(0, 0, W, H);

  const { min, range, histograms, acc, delta } = computeShiftedHistograms(
    values,
    params.bins,
    params.shifts,
  );

  // Averaged histogram — align all shifts to the first origin for display
  const averaged = new Float64Array(params.bins);
  for (const h of histograms) {
    for (let b = 0; b < params.bins; b++) averaged[b]! += h.counts[b]!;
  }
  for (let b = 0; b < params.bins; b++) averaged[b]! /= params.shifts;

  const padL = 60 * devicePixelRatio;
  const padR = 20 * devicePixelRatio;
  const padT = 40 * devicePixelRatio;
  const padB = 50 * devicePixelRatio;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const maxCount = Math.max(...Array.from(averaged));

  const toCanvasX = (x: number) => padL + ((x - min) / range) * plotW;
  const toCanvasY = (d: number) => padT + plotH - (d / maxCount) * plotH;

  const drawBars = (
    { origin, binWidth, counts }: ShiftedHistogram,
    color: string,
    alpha: number,
  ) => {
    const barPx = (plotW / range) * binWidth;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;
    for (let b = 0; b < counts.length; b++) {
      const xLeft = origin + b * binWidth;
      const cx = toCanvasX(xLeft + binWidth * 0.5);
      const cy = toCanvasY(counts[b]!);
      const bh = padT + plotH - cy;
      ctx.fillRect(cx - barPx * 0.45, cy, barPx * 0.9, bh);
    }
    ctx.restore();
  };

  // Draw each shifted histogram at its true x-origin
  for (let s = 0; s < histograms.length; s++) {
    const color = SHIFT_COLORS[s % SHIFT_COLORS.length]!;
    drawBars(histograms[s]!, color, 0.2);
  }

  // Draw acc line — each sub-bin centred at min + (i + 0.5) * delta
  const maxAcc = Math.max(...Array.from(acc));
  ctx.save();
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2 * devicePixelRatio;
  ctx.globalAlpha = 0.9;
  ctx.beginPath();
  for (let i = 0; i < acc.length; i++) {
    const x = toCanvasX(min + (i + 0.5) * delta);
    const y = toCanvasY((acc[i]! / maxAcc) * maxCount);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.restore();

  // Axes
  ctx.strokeStyle = "#555";
  ctx.lineWidth = devicePixelRatio;
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, padT + plotH);
  ctx.lineTo(padL + plotW, padT + plotH);
  ctx.stroke();

  // X-axis labels
  ctx.fillStyle = "#aaa";
  ctx.font = `${11 * devicePixelRatio}px monospace`;
  ctx.textAlign = "center";
  for (let i = 0; i <= 6; i++) {
    const xVal = min + (i / 6) * range;
    const cx = toCanvasX(xVal);
    ctx.fillText(xVal.toFixed(1), cx, padT + plotH + 18 * devicePixelRatio);
  }

  // Y-axis labels
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    const d = (i / 4) * maxCount;
    const cy = toCanvasY(d);
    ctx.fillText(
      d.toFixed(0),
      padL - 6 * devicePixelRatio,
      cy + 4 * devicePixelRatio,
    );
  }

  // Legend
  const legendX = padL + plotW - 10 * devicePixelRatio;
  const legendY = padT + 10 * devicePixelRatio;
  const swatch = 10 * devicePixelRatio;
  const lineH = 16 * devicePixelRatio;
  ctx.textAlign = "right";
  ctx.font = `${10 * devicePixelRatio}px monospace`;
  for (let s = 0; s < histograms.length; s++) {
    const color = SHIFT_COLORS[s % SHIFT_COLORS.length]!;
    const y = legendY + s * lineH;
    ctx.globalAlpha = 0.7;
    ctx.fillStyle = color;
    ctx.fillRect(legendX - swatch, y, swatch, swatch);
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#aaa";
    ctx.fillText(
      `shift ${s + 1}`,
      legendX - swatch - 4 * devicePixelRatio,
      y + swatch,
    );
  }
  // Average legend entry
  const avgY = legendY + histograms.length * lineH;
  ctx.fillStyle = "#ffffff";
  ctx.globalAlpha = 0.9;
  ctx.fillRect(legendX - swatch, avgY, swatch, swatch);
  ctx.globalAlpha = 1;
  ctx.fillStyle = "#eee";
  ctx.fillText(
    "average",
    legendX - swatch - 4 * devicePixelRatio,
    avgY + swatch,
  );

  // Title
  ctx.fillStyle = "#eee";
  ctx.textAlign = "center";
  ctx.font = `bold ${13 * devicePixelRatio}px monospace`;
  ctx.fillText(
    `Averaged Shifted Histogram  (bins=${params.bins}, shifts=${params.shifts})`,
    W / 2,
    padT - 10 * devicePixelRatio,
  );
}

function resizeCanvas(canvas: HTMLCanvasElement) {
  canvas.width = Math.floor(canvas.clientWidth * devicePixelRatio);
  canvas.height = Math.floor(canvas.clientHeight * devicePixelRatio);
}

function createCanvas() {
  document.body.style.margin = "0";
  document.body.style.background = "#0d0f14";

  const canvas = document.createElement("canvas");
  canvas.style.position = "fixed";
  canvas.style.inset = "0";
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  canvas.style.display = "block";
  document.body.appendChild(canvas);
  resizeCanvas(canvas);
  return canvas;
}

