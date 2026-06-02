import GUI from "lil-gui";
import { ChartCanvas } from "./chart-canvas";
import { downsample } from "./downsampling";
import { lowerBound, upperBound, ViewManager } from "./view-manager";

const AXIS_Y_W = 70; // px reserved on the right for the Y axis
const AXIS_X_H = 46; // px reserved on the bottom for the X axis
const CHART_PAD_Y = 20; // px of top+bottom padding inside the heatmap canvas
const N_SIGMA_TRUNCATE = 3; // number of sigmas to cover with the quad

const DEFAULT_PALETTE_COLORS = {
  c0: "#d1edff",
  c1: "#feffb8",
  c2: "#f28787",
};

function parseCssPixels(value: string): number {
  if (!value) return 0;
  const n = Number.parseFloat(value);
  return Number.isFinite(n) ? n : 0;
}

export type LoadedData = {
  n: number;
  dataF64: Float64Array;
  xMin: number;
  xScale: number;
};

export type GaussianChartOptions = {
  data: LoadedData;
};

type GaussianChartState = {
  showHeatmap: boolean;
  heatmapRenderMode: "composited" | "overlay";
  showTimescale: boolean;
  showYAxis: boolean;
  pixelRatio: number;
  sigmaSize: number;
  quantSteps: number;
  opacityCut: number;
  mergeThresholdSigmas: number;
  paletteLevel: number;
  showLine: boolean;
  showScatter: boolean;
  lineOpacity: number;
  lineWidth: number;
  showGaussianQuads: boolean;
  showGaussian3SigmaCircle: boolean;
  paletteColors: {
    c0: string;
    c1: string;
    c2: string;
  };
};

export class GaussianChart {
  readonly container: HTMLElement;
  private readonly n: number;
  private readonly dataF64: Float64Array;
  private readonly xMin: number;
  private readonly xScale: number;
  readonly state: GaussianChartState;

  private canvas!: HTMLCanvasElement;
  private fullChartSize: { cssW: number; cssH: number } = { cssW: 0, cssH: 0 };
  private gpu!: Awaited<ReturnType<typeof initWebGPU>>;
  private accumulationPipeline!: GPURenderPipeline;
  private reductionPipeline!: GPUComputePipeline;
  private tonemapPipeline!: GPURenderPipeline;
  private quadBuffer!: GPUBuffer;
  private instanceBuffer!: GPUBuffer;
  private uniformBuffer!: GPUBuffer;
  private statsBuffer!: GPUBuffer;
  private statsReadbackBuffer!: GPUBuffer;
  private colorBuffer!: GPUBuffer;
  private accBindGroup!: GPUBindGroup;
  private hdrTexture!: GPUTexture;
  private reductionBindGroup!: GPUBindGroup;
  private tonemapBindGroup!: GPUBindGroup;
  private viewManager!: ViewManager;
  private chartCanvas!: ChartCanvas;
  private destroyed = false;

  private hdrW = 0;
  private hdrH = 0;
  private lastViewMinX = NaN;
  private lastViewMaxX = NaN;
  private lastViewMinY = NaN;
  private lastViewMaxY = NaN;
  private readbackPending = false;
  private lastMergedCount = 0;
  private lastTime = 0;
  private fpsAccTime = 0;
  private frameCount = 0;

  private readonly controls: {
    sigmaSize: number;
    fps: string;
    totalPoints: number;
    renderedPoints: number;
  };
  private fpsController: any;
  private renderedPointsController: any;
  private gui: GUI | null = null;
  private integrateTimeoutId: number | null = null;
  private selectionIntegrateToken = 0;

  constructor(options: GaussianChartOptions) {
    const { data } = options;
    this.container = document.createElement("div");
    this.container.style.position = "relative";
    this.n = data.n;
    this.dataF64 = data.dataF64;
    this.xMin = data.xMin;
    this.xScale = data.xScale;
    this.state = {
      showHeatmap: true,
      heatmapRenderMode: "overlay",
      showTimescale: true,
      showYAxis: true,
      pixelRatio: window.devicePixelRatio,
      sigmaSize: 16,
      quantSteps: 0,
      opacityCut: 0.03,
      mergeThresholdSigmas: 1,
      paletteLevel: 0.5,
      showLine: true,
      showScatter: false,
      lineOpacity: 1,
      lineWidth: 1,
      showGaussianQuads: false,
      showGaussian3SigmaCircle: false,
      paletteColors: { ...DEFAULT_PALETTE_COLORS },
    };
    this.controls = {
      sigmaSize: this.state.sigmaSize,
      fps: "0.0",
      totalPoints: this.n,
      renderedPoints: 0,
    };
    this.fpsController = null;
    this.renderedPointsController = null;
  }

  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;

    if (this.integrateTimeoutId !== null) {
      window.clearTimeout(this.integrateTimeoutId);
      this.integrateTimeoutId = null;
    }

    if (this.gui) {
      this.gui.destroy();
      this.gui = null;
    }

    if (this.chartCanvas) {
      this.chartCanvas.destroy();
    }

    if (this.canvas && this.canvas.parentElement === this.container) {
      this.container.removeChild(this.canvas);
    }

    try {
      this.statsReadbackBuffer?.unmap();
    } catch {
      // Ignore: buffer may not be currently mapped.
    }

    this.quadBuffer?.destroy();
    this.instanceBuffer?.destroy();
    this.uniformBuffer?.destroy();
    this.statsBuffer?.destroy();
    this.statsReadbackBuffer?.destroy();
    this.colorBuffer?.destroy();
    this.hdrTexture?.destroy();
  }

  async start(): Promise<void> {
    this.canvas = this.createCanvas();
    this.gpu = await initWebGPU(this.canvas);
    this.accumulationPipeline = createAccumulationPipeline(
      this.gpu.device,
      this.getHeatmapCssH(),
    );
    this.reductionPipeline = createReductionPipeline(this.gpu.device);
    this.tonemapPipeline = createTonemapPipeline(
      this.gpu.device,
      this.gpu.format,
    );

    const { quadBuffer, instanceBuffer, uniformBuffer } = uploadData(
      this.gpu.device,
      this.n,
    );
    this.quadBuffer = quadBuffer;
    this.instanceBuffer = instanceBuffer;
    this.uniformBuffer = uniformBuffer;

    this.statsBuffer = this.gpu.device.createBuffer({
      size: 4,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });
    this.statsReadbackBuffer = this.gpu.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.colorBuffer = this.gpu.device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.writePaletteToBuffer();

    this.accBindGroup = this.gpu.device.createBindGroup({
      layout: this.accumulationPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }],
    });

    this.hdrW = this.canvas.width;
    this.hdrH = this.canvas.height;
    this.hdrTexture = createHDRTexture(this.gpu.device, this.hdrW, this.hdrH);
    this.reductionBindGroup = createReductionBindGroup(
      this.gpu.device,
      this.reductionPipeline,
      this.hdrTexture,
      this.statsBuffer,
    );
    this.tonemapBindGroup = createTonemapBindGroup(
      this.gpu.device,
      this.tonemapPipeline,
      this.hdrTexture,
      this.statsBuffer,
      this.colorBuffer,
    );

    this.viewManager = new ViewManager(this.dataF64);

    this.chartCanvas = new ChartCanvas(this.container, this.xMin, this.xScale, {
      getShowHeatmap: () =>
        this.state.showHeatmap && this.state.heatmapRenderMode === "composited",
      getPaletteLevel: () => this.state.paletteLevel,
      getOpacityCut: () => this.state.opacityCut,
      getShowTimescale: () => this.state.showTimescale,
      getShowYAxis: () => this.state.showYAxis,
      getPixelRatio: () => this.state.pixelRatio,
      getShowLine: () => this.state.showLine,
      getShowScatter: () => this.state.showScatter,
      getLineOpacity: () => this.state.lineOpacity,
      getLineWidth: () => this.state.lineWidth,
      getShowGaussianQuads: () => this.state.showGaussianQuads,
      getShowGaussian3SigmaCircle: () => this.state.showGaussian3SigmaCircle,
      getGaussianSigmaSize: () => this.state.sigmaSize,
      getGaussianTruncateNSigma: () => N_SIGMA_TRUNCATE,
      getHeatmapColor: (v) => this.heatmapColor(v),
      size: () => {
        return this.fullChartSize;
      },
    });
    this.chartCanvas.setHeatmapSource(this.canvas);
    this.viewManager.bindCanvas(this.chartCanvas.getCanvasElement());
    this.chartCanvas.setOnLevelChange((v) => {
      this.state.paletteLevel = v;
      this.writePaletteToBuffer();
    });
    this.chartCanvas.setOnSelectionChange((region) => {
      this.selectionIntegrateToken++;
      const token = this.selectionIntegrateToken;

      if (this.integrateTimeoutId !== null) {
        window.clearTimeout(this.integrateTimeoutId);
        this.integrateTimeoutId = null;
      }

      if (!region) {
        this.chartCanvas.setSelectionIntegral(null);
        return;
      }

      this.chartCanvas.setSelectionIntegral(null);
      this.integrateTimeoutId = window.setTimeout(() => {
        void this.computeSelectionIntegral(region, token);
      }, 150);
    });
    this.syncHeatmapPresentation();

    this.setupGui();
    this.lastTime = performance.now();
    window.addEventListener("resize", () =>
      this.handleResize(this.getContainerCssSize()),
    );
  }

  private setupGui() {
    const gui = new GUI({ title: "Render Controls" });
    const exportActions = {
      downloadImage: () => {
        const { cssW, cssH } = this.getContainerCssSize();
        void this.downloadImage({
          width: cssW,
          height: cssH,
          name: `gaussian-chart-${Date.now()}`,
          devicePixelRatio: this.getEffectivePixelRatio(),
        });
      },
    };
    this.gui = gui;
    this.container.appendChild(gui.domElement);
    gui.domElement.style.position = "absolute";
    gui.domElement.style.top = "0px";
    gui.domElement.style.right = `${AXIS_Y_W}px`;
    gui.domElement.style.zIndex = "3";
    gui
      .add(this.state, "showHeatmap")
      .name("Show heatmap")
      .onChange(() => this.syncHeatmapPresentation());
    gui
      .add(this.state, "heatmapRenderMode", ["composited", "overlay"])
      .name("Heatmap mode")
      .onChange(() => this.syncHeatmapPresentation());
    gui.add(this.state, "showTimescale").name("Show timescale");
    gui
      .add(this.state, "showYAxis")
      .name("Show Y axis")
      .onChange(() => this.handleResize(this.getContainerCssSize()));
    gui
      .add(this.state, "pixelRatio", 0.5, 4, 0.25)
      .name("Pixel ratio")
      .onChange(() => {
        this.handleResize(this.getContainerCssSize());
      });
    this.fpsController = gui.add(this.controls, "fps").name("FPS").disable();
    gui.add(this.controls, "totalPoints").name("Total points").disable();
    this.renderedPointsController = gui
      .add(this.controls, "renderedPoints")
      .name("Rendered points")
      .disable();

    gui
      .add(this.controls, "sigmaSize", 1, 50, 1)
      .name("Kernel sigma (px)")
      .onChange((v: number) => {
        this.state.sigmaSize = v;
        this.lastViewMinX = NaN; // force re-merge of points
      });
    gui
      .add(this.state, "quantSteps", 0, 32, 1)
      .name("Quantize steps")
      .onChange((v: number) => {
        this.state.quantSteps = v;
        this.writePaletteToBuffer();
      });
    gui
      .add(this.state, "opacityCut", 0.001, 0.15, 0.001)
      .name("Opacity cut")
      .onChange((v: number) => {
        this.state.opacityCut = v;
        this.writePaletteToBuffer();
      });
    gui
      .add(this.state, "mergeThresholdSigmas", 0, 3, 0.05)
      .name("Merge threshold (sigma)")
      .onChange((v: number) => {
        this.state.mergeThresholdSigmas = v;
        this.lastViewMinX = NaN;
      });

    const colorFolder = gui.addFolder("Color map");
    colorFolder
      .add(this.state, "paletteLevel", 0.01, 0.99, 0.01)
      .name("Mid level")
      .onChange((v: number) => {
        this.state.paletteLevel = v;
        this.writePaletteToBuffer();
      });
    colorFolder
      .addColor(this.state.paletteColors, "c0")
      .name("Low")
      .onChange(() => this.writePaletteToBuffer());
    colorFolder
      .addColor(this.state.paletteColors, "c1")
      .name("Mid")
      .onChange(() => this.writePaletteToBuffer());
    colorFolder
      .addColor(this.state.paletteColors, "c2")
      .name("High")
      .onChange(() => this.writePaletteToBuffer());

    const lineFolder = gui.addFolder("Line chart");
    lineFolder.add(this.state, "showLine").name("Show line");
    lineFolder.add(this.state, "showScatter").name("Show scatter");
    lineFolder.add(this.state, "lineOpacity", 0.01, 1, 0.01).name("Opacity");
    lineFolder.add(this.state, "lineWidth", 0.25, 5, 0.25).name("Line width");
    lineFolder.add(this.state, "showGaussianQuads").name("Show gaussian quads");
    lineFolder
      .add(this.state, "showGaussian3SigmaCircle")
      .name(`Show gaussian ${N_SIGMA_TRUNCATE} sigma circle`);

    gui.add(exportActions, "downloadImage").name("Download image");
  }

  private scheduleReadback() {
    if (this.readbackPending || this.destroyed) return;
    this.readbackPending = true;
    const encoder = this.gpu.device.createCommandEncoder();
    encoder.copyBufferToBuffer(
      this.statsBuffer,
      0,
      this.statsReadbackBuffer,
      0,
      4,
    );
    this.gpu.device.queue.submit([encoder.finish()]);
    this.statsReadbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
      if (this.destroyed) {
        this.readbackPending = false;
        try {
          this.statsReadbackBuffer.unmap();
        } catch {
          // Ignore race with destroy.
        }
        return;
      }
      const val =
        new Uint32Array(this.statsReadbackBuffer.getMappedRange())[0] ?? 1;
      this.statsReadbackBuffer.unmap();
      this.readbackPending = false;
      this.chartCanvas.setMaxVal(val);
    });
  }

  private async render() {
    if (this.destroyed) return;
    const now = performance.now();
    const dt = (now - this.lastTime) / 1000;
    this.lastTime = now;
    this.frameCount++;
    this.fpsAccTime += dt;
    if (this.fpsAccTime >= 0.5) {
      this.controls.fps = `${(this.frameCount / this.fpsAccTime).toFixed(1)} fps`;
      this.fpsController.updateDisplay();
      this.frameCount = 0;
      this.fpsAccTime = 0;
    }

    this.viewManager.tick(dt);

    const pixelRatio = this.getEffectivePixelRatio();
    const padFrac = CHART_PAD_Y / Math.max(1, this.getHeatmapCssH());

    updateUniform(
      this.gpu.device,
      this.uniformBuffer,
      this.viewManager.getViewMinX(),
      this.viewManager.getViewMaxX(),
      this.viewManager.getViewMinY(),
      this.viewManager.getViewMaxY(),
      this.hdrW,
      this.hdrH,
      this.state.sigmaSize * pixelRatio,
      padFrac,
    );

    this.gpu.device.queue.writeBuffer(
      this.statsBuffer,
      0,
      new Uint32Array([0]),
    );
    const encoder = this.gpu.device.createCommandEncoder();

    const accPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.hdrTexture.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });

    const viewMinX = this.viewManager.getViewMinX();
    const viewMaxX = this.viewManager.getViewMaxX();
    const viewMinY = this.viewManager.getViewMinY();
    const viewMaxY = this.viewManager.getViewMaxY();

    if (
      viewMinX !== this.lastViewMinX ||
      viewMaxX !== this.lastViewMaxX ||
      viewMinY !== this.lastViewMinY ||
      viewMaxY !== this.lastViewMaxY
    ) {
      const merged = this.mergePoints(
        viewMinX,
        viewMaxX,
        viewMinY,
        viewMaxY,
        this.hdrW,
        this.hdrH,
        this.state.mergeThresholdSigmas,
        this.state.sigmaSize * pixelRatio,
      );
      this.gpu.device.queue.writeBuffer(
        this.instanceBuffer,
        0,
        merged.gpuInstances,
      );
      this.lastMergedCount = merged.count;
      this.controls.renderedPoints = this.lastMergedCount;
      this.renderedPointsController.updateDisplay();
      this.chartCanvas.setMergedPoints(merged.gpuInstances);
      this.lastViewMinX = viewMinX;
      this.lastViewMaxX = viewMaxX;
      this.lastViewMinY = viewMinY;
      this.lastViewMaxY = viewMaxY;
    }

    accPass.setPipeline(this.accumulationPipeline);
    accPass.setBindGroup(0, this.accBindGroup);
    accPass.setVertexBuffer(0, this.quadBuffer);
    accPass.setVertexBuffer(1, this.instanceBuffer);
    accPass.draw(6, this.lastMergedCount);
    accPass.end();

    const reductionPass = encoder.beginComputePass();
    reductionPass.setPipeline(this.reductionPipeline);
    reductionPass.setBindGroup(0, this.reductionBindGroup);
    reductionPass.dispatchWorkgroups(
      Math.ceil(this.hdrW / 8),
      Math.ceil(this.hdrH / 8),
    );
    reductionPass.end();

    const tonemapPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.gpu.context.getCurrentTexture().createView(),
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    tonemapPass.setPipeline(this.tonemapPipeline);
    tonemapPass.setBindGroup(0, this.tonemapBindGroup);
    tonemapPass.draw(3);
    tonemapPass.end();

    this.gpu.device.queue.submit([encoder.finish()]);
    this.scheduleReadback();
    await this.gpu.device.queue.onSubmittedWorkDone();
    this.chartCanvas.render(viewMinX, viewMaxX, viewMinY, viewMaxY);
  }

  async setupRenderLoop() {
    while (true) {
      if (this.destroyed) {
        break;
      }
      await new Promise(requestAnimationFrame);
      await this.render();
    }
  }

  private async computeSelectionIntegral(
    region: { x: number; y: number; width: number; height: number },
    token: number,
  ) {
    if (this.destroyed || token !== this.selectionIntegrateToken) return;
    const result = await this.integrate(region);
    if (this.destroyed || token !== this.selectionIntegrateToken) return;
    this.chartCanvas.setSelectionIntegral(result?.integral ?? null);
  }

  async integrate(region?: {
    x: number;
    y: number;
    width: number;
    height: number;
  }): Promise<{
    integral: number;
    actualSum: number;
    errorPercent: number;
  } | null> {
    console.log(region);
    const width = this.hdrW;
    const height = this.hdrH;
    if (width <= 0 || height <= 0) return null;

    let x0 = 0;
    let y0 = 0;
    let x1 = width;
    let y1 = height;

    if (region) {
      const axisYWidth = this.state.showYAxis ? AXIS_Y_W : 0;
      const heatmapCssW = Math.max(1, this.fullChartSize.cssW - axisYWidth);
      const heatmapCssH = Math.max(1, this.fullChartSize.cssH - AXIS_X_H);
      const scaleX = width / heatmapCssW;
      const scaleY = height / heatmapCssH;

      const rx0 = Math.min(region.x, region.x + region.width);
      const ry0 = Math.min(region.y, region.y + region.height);
      const rx1 = Math.max(region.x, region.x + region.width);
      const ry1 = Math.max(region.y, region.y + region.height);

      x0 = Math.max(0, Math.min(width, Math.floor(rx0 * scaleX)));
      y0 = Math.max(0, Math.min(height, Math.floor(ry0 * scaleY)));
      x1 = Math.max(0, Math.min(width, Math.ceil(rx1 * scaleX)));
      y1 = Math.max(0, Math.min(height, Math.ceil(ry1 * scaleY)));

      if (x1 <= x0 || y1 <= y0) {
        return {
          integral: 0,
          actualSum: 0,
          errorPercent: 0,
        };
      }
    }

    const bytesPerPixel = 4; // r32float
    const unalignedBytesPerRow = width * bytesPerPixel;
    const bytesPerRow = Math.ceil(unalignedBytesPerRow / 256) * 256;
    const copySize = bytesPerRow * height;

    const readbackBuffer = this.gpu.device.createBuffer({
      size: copySize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = this.gpu.device.createCommandEncoder();
    encoder.copyTextureToBuffer(
      { texture: this.hdrTexture },
      {
        buffer: readbackBuffer,
        bytesPerRow,
        rowsPerImage: height,
      },
      {
        width,
        height,
        depthOrArrayLayers: 1,
      },
    );
    this.gpu.device.queue.submit([encoder.finish()]);

    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const mapped = readbackBuffer.getMappedRange();
    const data = new Float32Array(mapped);
    const strideFloats = bytesPerRow / bytesPerPixel;

    let sum = 0;
    for (let y = y0; y < y1; y++) {
      const rowStart = y * strideFloats;
      for (let x = x0; x < x1; x++) {
        sum += data[rowStart + x] ?? 0;
      }
    }

    readbackBuffer.unmap();
    readbackBuffer.destroy();
    const sigmaSize = this.state.sigmaSize * this.getEffectivePixelRatio();
    const scaleFactor = sigmaSize ** 2;
    const integral = sum / scaleFactor;

    let actualSum = 0;
    for (let i = 0; i < this.n; i++) {
      const w = this.dataF64[i * 3 + 2]!;
      actualSum += w;
    }

    const errorPercent =
      (Math.abs(integral - actualSum) / (actualSum || 1)) * 100;

    return {
      integral,
      actualSum,
      errorPercent,
    };
  }

  async downloadImage(options: {
    width: number;
    height: number;
    name: string;
    devicePixelRatio: number;
  }) {
    if (this.destroyed) return;

    const width = Math.max(1, Math.floor(options.width));
    const height = Math.max(1, Math.floor(options.height));
    const exportDpr = Math.max(0.25, options.devicePixelRatio);

    const prevContainerWidth = this.container.style.width;
    const prevContainerHeight = this.container.style.height;
    const prevPixelRatio = this.state.pixelRatio;
    const prevRenderMode = this.state.heatmapRenderMode;

    try {
      this.container.style.width = `${width}px`;
      this.container.style.height = `${height}px`;

      this.state.pixelRatio = exportDpr;
      this.state.heatmapRenderMode = "composited";

      this.syncHeatmapPresentation();
      this.handleResize({
        cssW: width,
        cssH: height,
      });

      await this.render();

      const link = document.createElement("a");
      link.download = options.name.endsWith(".png")
        ? options.name
        : `${options.name}.png`;

      link.href = this.exportImage("image/png");
      link.click();
    } finally {
      this.state.pixelRatio = prevPixelRatio;
      this.state.heatmapRenderMode = prevRenderMode;
      this.container.style.width = prevContainerWidth;
      this.container.style.height = prevContainerHeight;
      this.syncHeatmapPresentation();
      this.handleResize(this.getContainerCssSize());
    }
  }

  private handleResize(size: { cssH: number; cssW: number }) {
    if (this.destroyed) return;
    this.resizeHeatmapCanvas(size);
    this.hdrW = this.canvas.width;
    this.hdrH = this.canvas.height;
    this.lastViewMinX = NaN;
    this.hdrTexture.destroy();
    this.hdrTexture = createHDRTexture(this.gpu.device, this.hdrW, this.hdrH);
    this.reductionBindGroup = createReductionBindGroup(
      this.gpu.device,
      this.reductionPipeline,
      this.hdrTexture,
      this.statsBuffer,
    );
    this.tonemapBindGroup = createTonemapBindGroup(
      this.gpu.device,
      this.tonemapPipeline,
      this.hdrTexture,
      this.statsBuffer,
      this.colorBuffer,
    );
  }

  private writePaletteToBuffer() {
    const data = new Float32Array(16);
    const [r0, g0, b0] = hexToLinear(this.state.paletteColors.c0);
    const [r1, g1, b1] = hexToLinear(this.state.paletteColors.c1);
    const [r2, g2, b2] = hexToLinear(this.state.paletteColors.c2);
    data.set([
      r0,
      g0,
      b0,
      0,
      r1,
      g1,
      b1,
      0,
      r2,
      g2,
      b2,
      0,
      this.state.quantSteps,
      this.state.opacityCut,
      this.state.paletteLevel,
      0,
    ]);
    this.gpu.device.queue.writeBuffer(this.colorBuffer, 0, data);
  }

  private mergePoints(
    viewMinX: number,
    viewMaxX: number,
    viewMinY: number,
    viewMaxY: number,
    screenW: number,
    screenH: number,
    mergeThreshold: number,
    sigmaSizePx: number,
  ): {
    gpuInstances: Float32Array;
    count: number;
  } {
    const toSX = (x: number) =>
      (((x - viewMinX) / (viewMaxX - viewMinX)) * screenW) / sigmaSizePx;
    const toSY = (y: number) =>
      (((y - viewMinY) / (viewMaxY - viewMinY)) * screenH) / sigmaSizePx;

    const startIdx = lowerBound(this.dataF64, viewMinX);
    const endIdx = upperBound(this.dataF64, viewMaxX);

    const merged = downsample({
      points: this.dataF64.slice(startIdx * 3, endIdx * 3),
      strategy: "merge",
      toSX,
      toSY,
      mergeThreshold,
    });

    // merge strategy returns [x, y, w, p00, p01, p10, p11].
    const count = merged.length / 7;
    const gpuInstances = new Float32Array(count * 7);
    for (let i = 0; i < count; i++) {
      const mi = i * 7;
      const gi = i * 7;
      const x = merged[mi]!;
      const y = merged[mi + 1]!;
      const w = merged[mi + 2]!;
      const p00 = merged[mi + 3]!;
      const p01 = merged[mi + 4]!;
      const p10 = merged[mi + 5]!;
      const p11 = merged[mi + 6]!;
      gpuInstances[gi] = x;
      gpuInstances[gi + 1] = y;
      gpuInstances[gi + 2] = w;
      gpuInstances[gi + 3] = p00;
      gpuInstances[gi + 4] = p01;
      gpuInstances[gi + 5] = p10;
      gpuInstances[gi + 6] = p11;
    }

    return { gpuInstances, count };
  }

  private heatmapColor(value: number): [number, number, number] {
    const x = Math.min(Math.max(value, 0), 1);
    const [r0, g0, b0] = hexToLinear(this.state.paletteColors.c0);
    const [r1, g1, b1] = hexToLinear(this.state.paletteColors.c1);
    const [r2, g2, b2] = hexToLinear(this.state.paletteColors.c2);
    const c0 = [r0, g0, b0];
    const c1 = [r1, g1, b1];
    const c2 = [r2, g2, b2];
    const t1 = smoothstep(0.0, this.state.paletteLevel, x);
    const t2 = smoothstep(this.state.paletteLevel, 1.0, x);
    const mid = c0.map((v, i) => v + (c1[i]! - v) * t1);
    const rgb = mid.map((v, i) => v + (c2[i]! - v) * t2);
    return [
      Math.round(rgb[0]! * 255),
      Math.round(rgb[1]! * 255),
      Math.round(rgb[2]! * 255),
    ];
  }

  private createCanvas() {
    const canvas = document.createElement("canvas");
    canvas.style.cssText =
      "position:absolute;top:0;left:0;pointer-events:none;z-index:0;";
    this.canvas = canvas;
    this.resizeHeatmapCanvas(this.getContainerCssSize());
    return canvas;
  }

  private syncHeatmapPresentation() {
    const shouldOverlay =
      this.state.heatmapRenderMode === "overlay" && this.state.showHeatmap;
    if (shouldOverlay) {
      if (!this.canvas.parentElement) {
        this.container.appendChild(this.canvas);
      }
      this.canvas.style.opacity = "1";
      return;
    }

    this.canvas.style.opacity = "0";
    if (this.canvas.parentElement === this.container) {
      this.container.removeChild(this.canvas);
    }
  }

  private getContainerCssSize() {
    const cssW = Math.max(
      1,
      this.container.clientWidth || this.fullChartSize.cssW,
    );
    const cssH = Math.max(
      1,
      this.container.clientHeight || this.fullChartSize.cssH,
    );
    return { cssW, cssH };
  }

  private getHeatmapCssH() {
    const { cssH } = this.getContainerCssSize();
    return Math.max(1, cssH - AXIS_X_H);
  }

  private resizeHeatmapCanvas(size: { cssH: number; cssW: number }) {
    this.fullChartSize = size;
    const { cssH, cssW } = size;
    const axisYWidth = this.state.showYAxis ? AXIS_Y_W : 0;
    const heatmapCssW = Math.max(1, cssW - axisYWidth);
    const heatmapCssH = Math.max(1, cssH - AXIS_X_H);
    const pixelRatio = this.getEffectivePixelRatio();
    this.canvas.style.width = `${heatmapCssW}px`;
    this.canvas.style.height = `${heatmapCssH}px`;
    this.canvas.width = Math.round(heatmapCssW * pixelRatio);
    this.canvas.height = Math.round(heatmapCssH * pixelRatio);
  }

  private getEffectivePixelRatio(): number {
    return Math.max(0.25, this.state.pixelRatio);
  }

  setSelection(
    region: {
      x: number;
      y: number;
      width: number;
      height: number;
    } | null,
  ) {
    if (this.destroyed || !this.chartCanvas) return;
    this.chartCanvas.setSelection(region);
  }

  exportImage(type?: string, quality?: number): string {
    return this.chartCanvas.toDataURL(type, quality);
  }
}

async function initWebGPU(canvas: HTMLCanvasElement) {
  if (!navigator.gpu) throw new Error("WebGPU not supported");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter found");
  const device = await adapter.requestDevice({
    requiredFeatures: ["float32-blendable"],
  });
  const context = canvas.getContext("webgpu")!;
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "premultiplied" });
  return { device, context, format };
}

function createHDRTexture(device: GPUDevice, width: number, height: number) {
  return device.createTexture({
    size: [width, height],
    format: "r32float",
    usage:
      GPUTextureUsage.RENDER_ATTACHMENT |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_SRC,
  });
}

function createAccumulationPipeline(device: GPUDevice, heatmapcssH: number) {
  const nSigma = N_SIGMA_TRUNCATE.toFixed(6);
  const nSigma2 = (N_SIGMA_TRUNCATE * N_SIGMA_TRUNCATE).toFixed(6);
  const module = device.createShaderModule({
    code: /* wgsl */ `
      struct Uniforms {
        viewMinX: f32, viewMaxX: f32,
        minY: f32, maxY: f32,
        screenWidth: f32, screenHeight: f32,
        sigmaSize: f32, padFrac: f32,
      };

      @group(0) @binding(0) var<uniform> u: Uniforms;

      struct VertexOut {
        @builtin(position) pos: vec4f,
        @location(0) z: vec2f,
        @location(1) weight: f32,
      };

      @vertex
      fn vs_main(
        @location(0) quadOffset: vec2f,
        @location(1) point: vec2f,
        @location(2) weight: f32,
        @location(3) pRow0: vec2f,
        @location(4) pRow1: vec2f,
      ) -> VertexOut {
        if (point.x < u.viewMinX || point.x > u.viewMaxX) {
          return VertexOut(vec4f(10.0, 10.0, 10.0, 1.0), vec2f(0.0), 0.0);
        }

        let scale = 1.0 - 2.0 * u.padFrac;
        let nx = (point.x - u.viewMinX) / (u.viewMaxX - u.viewMinX) * 2.0 - 1.0;
        let ny = ((point.y - u.minY) / (u.maxY - u.minY) * 2.0 - 1.0) * scale;

        // z is in standardized Gaussian space; clip is at N_SIGMA_TRUNCATE bounds.
        let z = quadOffset * ${nSigma}f;

        // Build Cholesky factor L such that P = L * transpose(L).
        let p00 = max(pRow0.x, 1e-6);
        let p10 = pRow1.x;
        let p11 = max(pRow1.y, 1e-6);
        let l00 = sqrt(p00);
        let l10 = p10 / max(l00, 1e-6);
        let l11 = sqrt(max(p11 - l10 * l10, 1e-6));

        // Lower-triangular Cholesky factor: L = [[l00, 0], [l10, l11]].
        let L = mat2x2f(
          vec2f(l00, l10),
          vec2f(0.0, l11),
        );
        let local = L * z;
        let px = local * u.sigmaSize;
        let pxToNdc = mat2x2f(
          vec2f(2.0 / u.screenWidth, 0.0),
          vec2f(0.0, 2.0 / u.screenHeight),
        );
        let ndcDelta = pxToNdc * px;
        return VertexOut(
          vec4f(nx + ndcDelta.x, ny + ndcDelta.y, 0.0, 1.0),
          z,
          weight,
        );
      }

      @fragment
      fn fs_main(@location(0) z: vec2f, @location(1) weight: f32) -> @location(0) f32 {
        let d2 = dot(z, z);
        if (d2 > ${nSigma2}f) { discard; }
        let result = exp(-0.5 * d2);
        return result * weight;
      }
    `,
  });

  return device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module,
      entryPoint: "vs_main",
      buffers: [
        {
          arrayStride: 8,
          stepMode: "vertex",
          attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
        },
        {
          arrayStride: 28,
          stepMode: "instance",
          attributes: [
            { shaderLocation: 1, offset: 0, format: "float32x2" },
            { shaderLocation: 2, offset: 8, format: "float32" },
            { shaderLocation: 3, offset: 12, format: "float32x2" },
            { shaderLocation: 4, offset: 20, format: "float32x2" },
          ],
        },
      ],
    },
    fragment: {
      module,
      entryPoint: "fs_main",
      targets: [
        {
          format: "r32float",
          blend: {
            color: { srcFactor: "one", dstFactor: "one", operation: "add" },
            alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
          },
        },
      ],
    },
    primitive: { topology: "triangle-list" },
  });
}

function createReductionPipeline(device: GPUDevice) {
  const module = device.createShaderModule({
    code: /* wgsl */ `
      @group(0) @binding(0) var hdr: texture_2d<f32>;
      @group(0) @binding(1) var<storage, read_write> stats: array<atomic<u32>, 1>;

      @compute @workgroup_size(8, 8)
      fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
        let size = textureDimensions(hdr);
        if (gid.x >= size.x || gid.y >= size.y) { return; }

        let val = textureLoad(hdr, gid.xy, 0).r;
        if (val <= 0.0) { return; }

        atomicMax(&stats[0], u32(ceil(val)));
      }
    `,
  });

  return device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "cs_main" },
  });
}

function createTonemapPipeline(device: GPUDevice, format: GPUTextureFormat) {
  const module = device.createShaderModule({
    code: /* wgsl */ `
      struct Palette { c0: vec4f, c1: vec4f, c2: vec4f, steps: f32, opacityCut: f32, level: f32, _p3: f32 };

      @group(0) @binding(0) var hdr: texture_2d<f32>;
      @group(0) @binding(1) var<storage, read> stats: array<u32, 1>;
      @group(0) @binding(2) var<uniform> palette: Palette;

      @vertex
      fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
        var positions = array<vec2f, 3>(
          vec2f(-1.0, -1.0),
          vec2f( 3.0, -1.0),
          vec2f(-1.0,  3.0),
        );
        return vec4f(positions[vi], 0.0, 1.0);
      }

      fn mapColor(value: f32) -> vec3f {
        let x = clamp(value, 0.0, 1.0);
        let t1 = smoothstep(0.0, palette.level, x);
        let t2 = smoothstep(palette.level, 1.0, x);
        return mix(mix(palette.c0.rgb, palette.c1.rgb, t1), palette.c2.rgb, t2);
      }

      @fragment
      fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
        let accum = textureLoad(hdr, vec2i(pos.xy), 0).r;

        let maxVal = f32(stats[0]);
        var t = clamp(accum / maxVal, 0.0, 1.0);
        let opacity = smoothstep(0.0, palette.opacityCut, t);
        if palette.steps > 1.0 {
          t = floor(t * palette.steps) / (palette.steps - 1.0);
        }

        let rgb = mapColor(t) * opacity;
        return vec4f(rgb, opacity);
      }
    `,
  });

  return device.createRenderPipeline({
    layout: "auto",
    vertex: { module, entryPoint: "vs_main" },
    fragment: { module, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });
}

function createReductionBindGroup(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  hdrTexture: GPUTexture,
  statsBuffer: GPUBuffer,
) {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: hdrTexture.createView() },
      { binding: 1, resource: { buffer: statsBuffer } },
    ],
  });
}

function createTonemapBindGroup(
  device: GPUDevice,
  pipeline: GPURenderPipeline,
  hdrTexture: GPUTexture,
  statsBuffer: GPUBuffer,
  colorBuffer: GPUBuffer,
) {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: hdrTexture.createView() },
      { binding: 1, resource: { buffer: statsBuffer } },
      { binding: 2, resource: { buffer: colorBuffer } },
    ],
  });
}

function hexToLinear(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [((n >> 16) & 0xff) / 255, ((n >> 8) & 0xff) / 255, (n & 0xff) / 255];
}

// prettier-ignore
const QUAD_CORNERS = new Float32Array([
  -1, -1,   1, -1,  -1,  1,
  -1,  1,   1, -1,   1,  1,
]);

function uploadData(device: GPUDevice, pointCount: number) {
  const quadBuffer = device.createBuffer({
    size: QUAD_CORNERS.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(quadBuffer, 0, QUAD_CORNERS);

  const instanceBuffer = device.createBuffer({
    size: pointCount * 7 * 4,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });

  const uniformBuffer = device.createBuffer({
    size: 8 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(
    uniformBuffer,
    0,
    new Float32Array([0, 1, 0, 1, 1, 1, 0, 0]),
  );

  return { quadBuffer, instanceBuffer, uniformBuffer };
}

function updateUniform(
  device: GPUDevice,
  uniformBuffer: GPUBuffer,
  viewMinX: number,
  viewMaxX: number,
  minY: number,
  maxY: number,
  width: number,
  height: number,
  kernelSz: number,
  padFrac: number,
) {
  device.queue.writeBuffer(
    uniformBuffer,
    0,
    new Float32Array([
      viewMinX,
      viewMaxX,
      minY,
      maxY,
      width,
      height,
      kernelSz,
      padFrac,
    ]),
  );
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(Math.max((x - edge0) / (edge1 - edge0), 0), 1);
  return t * t * (3 - 2 * t);
}
