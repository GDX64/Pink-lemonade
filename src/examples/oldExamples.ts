import {
  createCanvas2DContext,
  FragmentShader,
  Rect,
  Scene,
} from "../canvas2d/context";

import {
  createNoiseData,
  drawChart,
  drawSplatKernelSeries,
} from "../chart/chart";
import noiseShader from "./noise.fragment.wgsl?raw";
import fragmentShaderSource from "./warping.fragment.wgsl?raw";

export async function cpuExample() {
  const canvas = createCanvas();
  const overlayCanvas = createCanvas();
  overlayCanvas.style.opacity = "0.2";
  const data = createNoiseData(100_000);
  const viewManager = new ViewManager(data);

  const ctx = await createCanvas2DContext(canvas);
  const donwScaling = 32;
  const width = Math.floor(canvas.width / donwScaling);
  const height = Math.floor(canvas.height / donwScaling);

  const postProcessFragmentShader = FragmentShader.new({
    source: noiseShader,
  });

  // ctx.addPostProcess(postProcessFragmentShader);

  const fragmentShader = FragmentShader.new({
    source: fragmentShaderSource,
  });

  const avgPoints: [number, number][] = [];
  for (let i = 0; i < data.length; i += 10) {
    let accY = 0;
    let accX = 0;
    for (let j = 0; j < 10 && i + j < data.length; j++) {
      accX += data[i + j]![0];
      accY += data[i + j]![1];
    }
    avgPoints.push([accX / 10, accY / 10]);
  }

  async function render() {
    drawChart(avgPoints, overlayCanvas, {
      viewMinX: viewManager.getViewMinX(),
      viewMaxX: viewManager.getViewMaxX(),
    });
    const density = drawSplatKernelSeries(data, {
      width,
      height,
      viewMinX: viewManager.getViewMinX(),
      viewMaxX: viewManager.getViewMaxX(),
    });

    ctx.updateCanvasTexture(texture, density);
    await ctx.draw(scene);
  }

  viewManager.bindCanvas(overlayCanvas);

  const density = drawSplatKernelSeries(data, { width, height });

  const texture = await ctx.createCanvasTexture({
    data: density,
    width,
    height,
  });

  fragmentShader.setTexture("canvasTexture", texture);

  const scene = new Scene();
  const rect = new Rect({
    x: 0,
    y: 0,
    width: canvas.width,
    height: canvas.height,
    fragmentShader,
    fragmentShaderEntryPoint: "main",
  });
  scene.add(rect);

  let lastTime = performance.now();
  ctx.loop(Infinity, async () => {
    const now = performance.now();
    const dtSeconds = Math.max(0, (now - lastTime) / 1000);
    lastTime = now;
    viewManager.tick(dtSeconds);
    await render();
  });

  return { ctx };
}

function createCanvas() {
  const canvas = document.createElement("canvas");
  canvas.style.position = "absolute";
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  document.body.appendChild(canvas);
  canvas.width = canvas.getBoundingClientRect().width * devicePixelRatio;
  canvas.height = canvas.getBoundingClientRect().height * devicePixelRatio;
  return canvas;
}

class ViewManager {
  private readonly dataMinX: number;
  private readonly dataMaxX: number;
  private readonly fullRangeX: number;
  private readonly minViewRangeX: number;
  private currentViewMinX: number;
  private currentViewMaxX: number;
  private targetViewMinX: number;
  private targetViewMaxX: number;
  private isPanning = false;
  private lastPointerX = 0;
  private readonly interpolationRate = 12;

  constructor(data: [number, number][]) {
    let minX = Infinity;
    let maxX = -Infinity;

    for (const [x] of data) {
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
    }

    this.dataMinX = minX;
    this.dataMaxX = maxX;
    this.fullRangeX = Math.max(this.dataMaxX - this.dataMinX, 1e-6);
    this.minViewRangeX = Math.max(this.fullRangeX / 512, 1e-6);
    this.currentViewMinX = this.dataMinX;
    this.currentViewMaxX = this.dataMaxX;
    this.targetViewMinX = this.dataMinX;
    this.targetViewMaxX = this.dataMaxX;
  }

  tick(dtSeconds: number): void {
    const alpha = 1 - Math.exp(-this.interpolationRate * dtSeconds);
    this.currentViewMinX +=
      (this.targetViewMinX - this.currentViewMinX) * alpha;
    this.currentViewMaxX +=
      (this.targetViewMaxX - this.currentViewMaxX) * alpha;

    if (Math.abs(this.targetViewMinX - this.currentViewMinX) < 1e-8) {
      this.currentViewMinX = this.targetViewMinX;
    }
    if (Math.abs(this.targetViewMaxX - this.currentViewMaxX) < 1e-8) {
      this.currentViewMaxX = this.targetViewMaxX;
    }
  }

  getViewMinX(): number {
    return this.currentViewMinX;
  }

  getViewMaxX(): number {
    return this.currentViewMaxX;
  }

  bindCanvas(canvas: HTMLCanvasElement): void {
    canvas.style.touchAction = "none";

    canvas.addEventListener("pointerdown", (event) => {
      this.isPanning = true;
      this.lastPointerX = event.clientX;
      canvas.setPointerCapture(event.pointerId);
    });

    canvas.addEventListener("pointermove", (event) => {
      if (!this.isPanning) {
        return;
      }

      const rect = canvas.getBoundingClientRect();
      const deltaXRatio = (event.clientX - this.lastPointerX) / rect.width;
      this.lastPointerX = event.clientX;

      const span = this.targetViewMaxX - this.targetViewMinX;
      const deltaX = deltaXRatio * span;
      this.targetViewMinX -= deltaX;
      this.targetViewMaxX -= deltaX;

      this.clampTargetViewport();
    });

    const stopPanning = (event: PointerEvent) => {
      if (!this.isPanning) {
        return;
      }

      this.isPanning = false;
      canvas.releasePointerCapture(event.pointerId);
    };

    canvas.addEventListener("pointerup", stopPanning);
    canvas.addEventListener("pointercancel", stopPanning);

    canvas.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();

        const rect = canvas.getBoundingClientRect();
        const anchorRatioX = Math.max(
          0,
          Math.min(1, (event.clientX - rect.left) / rect.width),
        );
        const currentSpan = this.targetViewMaxX - this.targetViewMinX;
        const zoomFactor = Math.exp(event.deltaY * 0.0015);
        const nextSpan = Math.max(
          this.minViewRangeX,
          Math.min(this.fullRangeX, currentSpan * zoomFactor),
        );

        if (!Number.isFinite(nextSpan) || nextSpan === currentSpan) {
          return;
        }

        const anchorX = this.targetViewMinX + anchorRatioX * currentSpan;
        this.targetViewMinX = anchorX - anchorRatioX * nextSpan;
        this.targetViewMaxX = this.targetViewMinX + nextSpan;
        this.clampTargetViewport();
      },
      { passive: false },
    );
  }

  private clampTargetViewport(): void {
    const span = Math.max(
      this.targetViewMaxX - this.targetViewMinX,
      this.minViewRangeX,
    );
    this.targetViewMinX = Math.max(
      this.dataMinX,
      Math.min(this.targetViewMinX, this.dataMaxX - span),
    );
    this.targetViewMaxX = this.targetViewMinX + span;
  }
}
