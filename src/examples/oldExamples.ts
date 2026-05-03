import {
  createCanvas2DContext,
  FragmentShader,
  Rect,
  Scene,
} from "../canvas2d/context";

import { createNoiseData, drawSplatKernelSeries } from "../chart/chart";
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

  const fragmentShader = FragmentShader.new({
    source: fragmentShaderSource,
  });

  async function render() {
    const density = drawSplatKernelSeries(data, {
      width,
      height,
      viewMinX: viewManager.getViewMinX(),
      viewMaxX: viewManager.getViewMaxX(),
    });

    ctx.updateCanvasTexture(texture, density);
    await ctx.draw(scene);
  }

  viewManager.setOnChange(() => {
    void render();
  });
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

  ctx.loop(Infinity, async () => {
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
  private viewMinX: number;
  private viewMaxX: number;
  private isPanning = false;
  private lastPointerX = 0;
  private onChange: (() => void) | undefined;

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
    this.viewMinX = this.dataMinX;
    this.viewMaxX = this.dataMaxX;
  }

  setOnChange(onChange: () => void): void {
    this.onChange = onChange;
  }

  getViewMinX(): number {
    return this.viewMinX;
  }

  getViewMaxX(): number {
    return this.viewMaxX;
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

      const span = this.viewMaxX - this.viewMinX;
      const deltaX = deltaXRatio * span;
      this.viewMinX -= deltaX;
      this.viewMaxX -= deltaX;

      this.clampViewport();
      this.notifyChange();
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
        const currentSpan = this.viewMaxX - this.viewMinX;
        const zoomFactor = Math.exp(event.deltaY * 0.0015);
        const nextSpan = Math.max(
          this.minViewRangeX,
          Math.min(this.fullRangeX, currentSpan * zoomFactor),
        );

        if (!Number.isFinite(nextSpan) || nextSpan === currentSpan) {
          return;
        }

        const anchorX = this.viewMinX + anchorRatioX * currentSpan;
        this.viewMinX = anchorX - anchorRatioX * nextSpan;
        this.viewMaxX = this.viewMinX + nextSpan;
        this.clampViewport();
        this.notifyChange();
      },
      { passive: false },
    );
  }

  private clampViewport(): void {
    const span = Math.max(this.viewMaxX - this.viewMinX, this.minViewRangeX);
    this.viewMinX = Math.max(
      this.dataMinX,
      Math.min(this.viewMinX, this.dataMaxX - span),
    );
    this.viewMaxX = this.viewMinX + span;
  }

  private notifyChange(): void {
    this.onChange?.();
  }
}
