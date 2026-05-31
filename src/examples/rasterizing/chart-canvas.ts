const AXIS_Y_W = 70; // px reserved on the right for the Y axis
const AXIS_X_H = 30; // px reserved on the bottom for the X axis
const CHART_PAD_Y = 20; // px of top+bottom padding inside the heatmap canvas

export type ChartCanvasStyleAccessors = {
  getShowHeatmap: () => boolean;
  getPaletteLevel: () => number;
  getOpacityCut: () => number;
  getShowTimescale: () => boolean;
  getShowYAxis: () => boolean;
  getPixelRatio: () => number;
  getShowLine: () => boolean;
  getShowScatter: () => boolean;
  getLineOpacity: () => number;
  getLineWidth: () => number;
  getShowGaussianQuads: () => boolean;
  getShowGaussian3SigmaCircle: () => boolean;
  getGaussianSigmaSize: () => number;
  getGaussianTruncateNSigma: () => number;
  getHeatmapColor: (value: number) => [number, number, number];
  size(): { cssW: number; cssH: number };
};

function fmtCompact(v: number): string {
  if (v === 0) return "0";
  if (Math.abs(v) >= 1e9) return `${(v / 1e9).toPrecision(3)}B`;
  if (Math.abs(v) >= 1e6) return `${(v / 1e6).toPrecision(3)}M`;
  if (Math.abs(v) >= 1e3) return `${(v / 1e3).toPrecision(3)}K`;
  return v.toPrecision(3);
}

function formatTimestamp(ms: number): string {
  const d = new Date(ms);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

export class ChartCanvas {
  private readonly container: HTMLElement;
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly styleAccessors: ChartCanvasStyleAccessors;
  private heatmapSource: HTMLCanvasElement | null = null;
  private mergedPoints: Float32Array = new Float32Array(0);
  private maxVal = 1;
  private readonly xMin: number;
  private readonly xScale: number;
  private onLevelChange: ((level: number) => void) | null = null;
  private isDraggingLevel = false;

  private static readonly BAR_W = 16;
  private static readonly BAR_H = 160;
  private static readonly LABEL_W = 68;
  private static readonly PAD = 6;
  private static readonly TICKS_X = 8;
  private static readonly TICKS_Y = 6;
  private static readonly FONT = "11px monospace";
  private static readonly LEGEND_TITLE_H = 0;

  private legendBarTop = 0;
  private legendBarLeft = 0;
  private onWindowPointerDown: (e: PointerEvent) => void;
  private onCanvasPointerMove: (e: PointerEvent) => void;
  private onCanvasPointerUp: (e: PointerEvent) => void;
  private onCanvasPointerCancel: (e: PointerEvent) => void;
  private onWindowPointerMove: (e: PointerEvent) => void;

  constructor(
    container: HTMLElement,
    xMin: number,
    xScale: number,
    styleAccessors: ChartCanvasStyleAccessors,
  ) {
    this.container = container;
    this.xMin = xMin;
    this.xScale = xScale;
    this.styleAccessors = styleAccessors;
    this.canvas = document.createElement("canvas");
    this.canvas.style.cssText =
      "position:absolute;inset:0;width:100%;height:100%;pointer-events:auto;z-index:1;";
    this.container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext("2d")!;

    this.onWindowPointerDown = () => {};
    this.onCanvasPointerMove = () => {};
    this.onCanvasPointerUp = () => {};
    this.onCanvasPointerCancel = () => {};
    this.onWindowPointerMove = () => {};
    this.bindLevelDrag();
  }

  destroy() {
    window.removeEventListener("pointerdown", this.onWindowPointerDown);
    this.canvas.removeEventListener("pointermove", this.onCanvasPointerMove);
    this.canvas.removeEventListener("pointerup", this.onCanvasPointerUp);
    this.canvas.removeEventListener(
      "pointercancel",
      this.onCanvasPointerCancel,
    );
    window.removeEventListener("pointermove", this.onWindowPointerMove);
    this.container.style.cursor = "";
    if (this.canvas.parentElement === this.container) {
      this.container.removeChild(this.canvas);
    }
  }

  setOnLevelChange(cb: (level: number) => void) {
    this.onLevelChange = cb;
  }

  setHeatmapSource(canvas: HTMLCanvasElement) {
    this.heatmapSource = canvas;
  }

  getCanvasElement(): HTMLCanvasElement {
    return this.canvas;
  }

  toDataURL(type?: string, quality?: number): string {
    return this.canvas.toDataURL(type, quality);
  }

  private bindLevelDrag() {
    const { BAR_H } = ChartCanvas;

    const hitTest = (cssX: number, cssY: number) => {
      const relY = cssY - this.legendBarTop;
      const relX = cssX - this.legendBarLeft;
      const triY = (1 - this.styleAccessors.getPaletteLevel()) * (BAR_H - 1);
      return (
        relX >= 0 &&
        relX <= ChartCanvas.BAR_W &&
        relY >= triY - 4 &&
        relY <= triY + 4
      );
    };

    const levelFromY = (cssY: number) => {
      const relY = cssY - this.legendBarTop;
      return Math.min(1, Math.max(0, 1 - relY / (BAR_H - 1)));
    };

    this.onWindowPointerDown = (e: PointerEvent) => {
      if (!hitTest(e.clientX, e.clientY)) return;
      this.isDraggingLevel = true;
      this.canvas.style.pointerEvents = "auto";
      this.canvas.setPointerCapture(e.pointerId);
      e.stopPropagation();
    };
    window.addEventListener("pointerdown", this.onWindowPointerDown);

    this.onCanvasPointerMove = (e: PointerEvent) => {
      if (!this.isDraggingLevel) return;
      this.onLevelChange?.(levelFromY(e.clientY));
    };
    this.canvas.addEventListener("pointermove", this.onCanvasPointerMove);

    const stopDrag = (e: PointerEvent) => {
      if (!this.isDraggingLevel) return;
      this.isDraggingLevel = false;
      this.canvas.style.pointerEvents = "auto";
      this.canvas.releasePointerCapture(e.pointerId);
    };
    this.onCanvasPointerUp = stopDrag;
    this.onCanvasPointerCancel = stopDrag;
    this.canvas.addEventListener("pointerup", this.onCanvasPointerUp);
    this.canvas.addEventListener("pointercancel", this.onCanvasPointerCancel);

    this.onWindowPointerMove = (e: PointerEvent) => {
      if (this.isDraggingLevel) return;
      this.container.style.cursor = hitTest(e.clientX, e.clientY)
        ? "ns-resize"
        : "";
    };
    window.addEventListener("pointermove", this.onWindowPointerMove);
  }

  setMergedPoints(points: Float32Array) {
    this.mergedPoints = points;
  }

  setMaxVal(valFromGPU: number) {
    if (valFromGPU > 0 && valFromGPU !== this.maxVal) {
      this.maxVal = valFromGPU;
    }
  }

  render(
    viewMinX: number,
    viewMaxX: number,
    viewMinY: number,
    viewMaxY: number,
  ) {
    const dpr = Math.max(0.25, this.styleAccessors.getPixelRatio());
    const { cssW, cssH } = this.styleAccessors.size();
    const w = Math.round(cssW * dpr);
    const h = Math.round(cssH * dpr);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }

    const axisYWidth = this.styleAccessors.getShowYAxis() ? AXIS_Y_W : 0;
    const hmW = cssW - axisYWidth;
    const hmH = cssH - AXIS_X_H - CHART_PAD_Y * 2;

    const { ctx } = this;
    ctx.clearRect(0, 0, w, h);
    ctx.save();
    ctx.scale(dpr, dpr);

    if (this.styleAccessors.getShowHeatmap() && this.heatmapSource) {
      ctx.drawImage(this.heatmapSource, 0, 0, hmW, cssH - AXIS_X_H);
    }

    ctx.save();

    ctx.translate(0, CHART_PAD_Y);
    this.drawLinePlot(hmW, hmH, viewMinX, viewMaxX, viewMinY, viewMaxY);
    if (this.styleAccessors.getShowTimescale()) {
      this.drawXAxis(hmW, hmH, cssH - CHART_PAD_Y * 2, viewMinX, viewMaxX);
    }
    if (this.styleAccessors.getShowYAxis()) {
      this.drawYAxis(hmW, hmH, cssW, viewMinY, viewMaxY);
    }
    ctx.restore();

    this.drawLegend(cssW, cssH);

    ctx.restore();
  }

  private drawLinePlot(
    hmW: number,
    hmH: number,
    viewMinX: number,
    viewMaxX: number,
    viewMinY: number,
    viewMaxY: number,
  ) {
    const { ctx } = this;
    const toScreenX = (x: number) =>
      ((x - viewMinX) / (viewMaxX - viewMinX)) * hmW;
    const toScreenY = (y: number) =>
      (1 - (y - viewMinY) / (viewMaxY - viewMinY)) * hmH;

    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, hmW, hmH);
    ctx.clip();

    const pts = this.mergedPoints;
    const n = pts.length / 7;

    if (this.styleAccessors.getShowGaussianQuads()) {
      const sigmaPx = Math.max(0, this.styleAccessors.getGaussianSigmaSize());
      const nSigma = Math.max(
        0,
        this.styleAccessors.getGaussianTruncateNSigma(),
      );
      if (sigmaPx > 0) {
        ctx.strokeStyle = "rgba(255, 0, 0, 0.35)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
          const p = i * 7;
          const sx = toScreenX(pts[p]!);
          const sy = toScreenY(pts[p + 1]!);
          const p00 = pts[p + 3]!;
          const p01 = pts[p + 4]!;
          const p11 = pts[p + 6]!;
          const trace = p00 + p11;
          const detTerm = Math.sqrt(
            Math.max((p00 - p11) * (p00 - p11) + 4 * p01 * p01, 0),
          );
          const lambdaMax = Math.max(0.5 * (trace + detTerm), 1e-6);
          const quadHalfSizePx = sigmaPx * Math.sqrt(lambdaMax) * nSigma;
          const quadSizePx = quadHalfSizePx * 2;
          ctx.rect(
            sx - quadHalfSizePx + 0.5,
            sy - quadHalfSizePx + 0.5,
            quadSizePx,
            quadSizePx,
          );
        }
        ctx.stroke();
      }
    }

    if (this.styleAccessors.getShowGaussian3SigmaCircle()) {
      const sigmaPx = Math.max(0, this.styleAccessors.getGaussianSigmaSize());
      const nSigma = Math.max(
        0,
        this.styleAccessors.getGaussianTruncateNSigma(),
      );
      if (sigmaPx > 0) {
        ctx.strokeStyle = "rgba(0, 89, 255, 0.35)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
          const p = i * 7;
          const sx = toScreenX(pts[p]!);
          const sy = toScreenY(pts[p + 1]!);
          const p00 = pts[p + 3]!;
          const p01 = pts[p + 4]!;
          const p11 = pts[p + 6]!;
          const trace = p00 + p11;
          const detTerm = Math.sqrt(
            Math.max((p00 - p11) * (p00 - p11) + 4 * p01 * p01, 0),
          );
          const lambdaMax = Math.max(0.5 * (trace + detTerm), 1e-6);
          const circleRadiusPx = sigmaPx * Math.sqrt(lambdaMax) * nSigma;
          ctx.moveTo(sx + circleRadiusPx, sy);
          ctx.arc(sx, sy, circleRadiusPx, 0, Math.PI * 2);
        }
        ctx.stroke();
      }
    }

    if (
      !this.styleAccessors.getShowLine() &&
      !this.styleAccessors.getShowScatter()
    ) {
      ctx.restore();
      return;
    }

    if (this.styleAccessors.getShowLine()) {
      ctx.strokeStyle = `rgba(0,0,0,${this.styleAccessors.getLineOpacity()})`;
      ctx.lineWidth = this.styleAccessors.getLineWidth();
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const p = i * 7;
        const sx = toScreenX(pts[p]!);
        const sy = toScreenY(pts[p + 1]!);
        if (i === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      }
      ctx.stroke();
    }

    if (this.styleAccessors.getShowScatter()) {
      const alpha = this.styleAccessors.getLineOpacity();
      const radius = Math.max(1, this.styleAccessors.getLineWidth() * 1.5);
      ctx.fillStyle = `rgba(0,0,0,${alpha})`;
      for (let i = 0; i < n; i++) {
        const p = i * 7;
        const sx = toScreenX(pts[p]!);
        const sy = toScreenY(pts[p + 1]!);
        ctx.beginPath();
        ctx.arc(sx, sy, radius, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    ctx.restore();
  }

  private drawXAxis(
    hmW: number,
    hmH: number,
    cssH: number,
    viewMinX: number,
    viewMaxX: number,
  ) {
    const { ctx } = this;
    const { TICKS_X, FONT, PAD } = ChartCanvas;
    const stripY = hmH + CHART_PAD_Y;

    ctx.save();
    ctx.fillStyle = "#f5f5f5";
    ctx.fillRect(0, stripY, hmW, cssH);

    ctx.strokeStyle = "#999";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, stripY);
    ctx.lineTo(hmW, stripY);
    ctx.stroke();

    ctx.fillStyle = "#333";
    ctx.font = FONT;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";

    const xPad = 20;
    for (let i = 0; i <= TICKS_X; i++) {
      const t = i / TICKS_X;
      const x = xPad + t * (hmW - xPad * 2);
      const val = viewMinX + t * (viewMaxX - viewMinX);

      ctx.strokeStyle = "#bbb";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, stripY);
      ctx.lineTo(x, stripY + 5);
      ctx.stroke();

      ctx.fillStyle = "#333";
      ctx.fillText(
        formatTimestamp(val * this.xScale + this.xMin),
        x,
        stripY + PAD + 5,
      );
    }
    ctx.restore();
  }

  private drawYAxis(
    hmW: number,
    hmH: number,
    cssW: number,
    viewMinY: number,
    viewMaxY: number,
  ) {
    const { ctx } = this;
    const { TICKS_Y, FONT, PAD } = ChartCanvas;
    const stripX = hmW;

    ctx.save();
    ctx.fillStyle = "#f5f5f5";
    ctx.fillRect(stripX, 0, cssW - stripX, hmH);

    ctx.strokeStyle = "#999";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(stripX, 0);
    ctx.lineTo(stripX, hmH);
    ctx.stroke();

    ctx.fillStyle = "#333";
    ctx.font = FONT;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";

    for (let i = 0; i <= TICKS_Y; i++) {
      const t = i / TICKS_Y;
      const y = (1 - t) * hmH;
      const val = viewMinY + t * (viewMaxY - viewMinY);

      ctx.strokeStyle = "#bbb";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(stripX, y);
      ctx.lineTo(stripX + 5, y);
      ctx.stroke();

      ctx.fillStyle = "#333";
      ctx.fillText(fmtCompact(val), stripX + PAD + 5, y);
    }
    ctx.restore();
  }

  private drawLegend(_cssW: number, _cssH: number) {
    const { ctx } = this;
    const { BAR_W, BAR_H, PAD, FONT, LEGEND_TITLE_H } = ChartCanvas;

    const ox = 0;
    const oy = 10;

    ctx.save();
    ctx.translate(ox, oy);

    ctx.fillStyle = "#000";
    ctx.font = FONT;
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    // ctx.fillText("Satoshi", PAD, 0);

    ctx.translate(PAD, LEGEND_TITLE_H);

    this.legendBarTop = oy + LEGEND_TITLE_H;
    this.legendBarLeft = ox + PAD;

    for (let py = 0; py < BAR_H; py++) {
      const t = 1 - py / (BAR_H - 1);
      const [r, g, b] = this.styleAccessors.getHeatmapColor(t);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(0, py, BAR_W, 1);
    }

    ctx.textBaseline = "middle";
    for (let k = 0; k <= 5; k++) {
      const t = k / 5;
      const py = (1 - t) * (BAR_H - 1);
      let val = t * this.maxVal;
      if (t === 0) {
        val = this.styleAccessors.getOpacityCut() * this.maxVal;
      }
      ctx.fillStyle = "#000";
      ctx.fillText(fmtCompact(val), BAR_W + PAD, py);
    }

    const lineY = (1 - this.styleAccessors.getPaletteLevel()) * (BAR_H - 1);
    ctx.strokeStyle = this.isDraggingLevel
      ? "rgba(255,255,255,0.9)"
      : "rgba(0,0,0,0.7)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, lineY);
    ctx.lineTo(BAR_W, lineY);
    ctx.stroke();

    ctx.restore();
  }
}
