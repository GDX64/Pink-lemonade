import GUI from "lil-gui";
import { createNoiseData } from "../../chart/chart";
import { downsample } from "./downsampling";
import jsonData from "./data.json";

const AXIS_Y_W = 70; // px reserved on the right for the Y axis
const AXIS_X_H = 30; // px reserved on the bottom for the X axis
const CHART_PAD_Y = 20; // px of top+bottom padding inside the heatmap canvas

// RGB triplets for the low/mid/high stops of the colormap
const paletteColors = {
  c0: "#d1edff",
  c1: "#feffb8",
  c2: "#f28787",
};

let kernelSize = 150;
let quantSteps = 5;
let opacityCut = 0.06;
let mergeThreshold = 10;

export async function rasterizingExample() {
  const canvas = createCanvas();
  //   canvas.style.opacity = "0.75";
  // const data = createNoiseData(100_000);
  const data = jsonData.data as [number, number, number][];

  // normalize x to [0, 1] to avoid float precision loss in GPU/math
  const xMin = data[0]![0];
  const xMax = data[data.length - 1]![0];
  const xScale = xMax - xMin || 1; // original span in ms; used to denormalize for display
  debugger;

  // pre-pack into Float64Array once — x is sorted and normalized
  const dataF64 = new Float64Array(data.length * 3);
  for (let i = 0; i < data.length; i++) {
    const [x, y, w] = data[i]!;
    dataF64[i * 3] = (x - xMin) / xScale;
    dataF64[i * 3 + 1] = y;
    dataF64[i * 3 + 2] = w ?? 1.0;
  }

  const gpu = await initWebGPU(canvas);
  const accumulationPipeline = createAccumulationPipeline(gpu.device);
  const reductionPipeline = createReductionPipeline(gpu.device);
  const tonemapPipeline = createTonemapPipeline(gpu.device, gpu.format);
  const { quadBuffer, instanceBuffer, uniformBuffer } = uploadData(
    gpu.device,
    data,
  );

  const statsBuffer = gpu.device.createBuffer({
    size: 4,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });

  const statsReadbackBuffer = gpu.device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const colorBuffer = gpu.device.createBuffer({
    size: 64, // 3 colors × vec4f + vec4f(steps, pad, pad, pad)
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  writePaletteToBuffer(gpu.device, colorBuffer);

  const accBindGroup = gpu.device.createBindGroup({
    layout: accumulationPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  let hdrW = canvas.width;
  let hdrH = canvas.height;
  let hdrTexture = createHDRTexture(gpu.device, hdrW, hdrH);
  let reductionBindGroup = createReductionBindGroup(
    gpu.device,
    reductionPipeline,
    hdrTexture,
    statsBuffer,
  );
  let tonemapBindGroup = createTonemapBindGroup(
    gpu.device,
    tonemapPipeline,
    hdrTexture,
    statsBuffer,
    colorBuffer,
  );

  const viewManager = new ViewManager(dataF64);
  viewManager.bindCanvas(canvas);

  let lastViewMinX = NaN;
  let lastViewMaxX = NaN;
  let lastViewMinY = NaN;
  let lastViewMaxY = NaN;

  const controls = {
    kernelSize,
    fps: "0.0",
    totalPoints: data.length,
    renderedPoints: 0,
  };
  const gui = new GUI({ title: "Render Controls" });
  gui.domElement.style.right = `${AXIS_Y_W}px`;
  const fpsController = gui.add(controls, "fps").name("FPS").disable();
  const totalPointsController = gui
    .add(controls, "totalPoints")
    .name("Total points")
    .disable();
  const renderedPointsController = gui
    .add(controls, "renderedPoints")
    .name("Rendered points")
    .disable();
  gui
    .add(controls, "kernelSize", 1, 300, 1)
    .name("Kernel size (R)")
    .onChange((v: number) => {
      kernelSize = v;
    });
  gui
    .add({ quantSteps }, "quantSteps", 0, 32, 1)
    .name("Quantize steps")
    .onChange((v: number) => {
      quantSteps = v;
      writePaletteToBuffer(gpu.device, colorBuffer);
    });
  gui
    .add({ opacityCut }, "opacityCut", 0.001, 0.15, 0.001)
    .name("Opacity cut")
    .onChange((v: number) => {
      opacityCut = v;
      writePaletteToBuffer(gpu.device, colorBuffer);
    });
  gui
    .add({ mergeThreshold }, "mergeThreshold", 2, 50, 1)
    .name("Merge threshold (px)")
    .onChange((v: number) => {
      mergeThreshold = v;
      lastViewMinX = NaN; // force re-merge
    });
  const colorFolder = gui.addFolder("Color map");
  colorFolder
    .addColor(paletteColors, "c0")
    .name("Low")
    .onChange(() => writePaletteToBuffer(gpu.device, colorBuffer));
  colorFolder
    .addColor(paletteColors, "c1")
    .name("Mid")
    .onChange(() => writePaletteToBuffer(gpu.device, colorBuffer));
  colorFolder
    .addColor(paletteColors, "c2")
    .name("High")
    .onChange(() => writePaletteToBuffer(gpu.device, colorBuffer));
  const chartCanvas = new ChartCanvas(data, xMin, xScale);
  let lastTime = performance.now();
  let fpsAccTime = 0;
  let frameCount = 0;
  let readbackPending = false;
  let lastMergedCount = 0;

  function scheduleReadback() {
    if (readbackPending) return;
    readbackPending = true;
    const encoder = gpu.device.createCommandEncoder();
    encoder.copyBufferToBuffer(statsBuffer, 0, statsReadbackBuffer, 0, 4);
    gpu.device.queue.submit([encoder.finish()]);
    statsReadbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
      const val = new Uint32Array(statsReadbackBuffer.getMappedRange())[0] ?? 1;
      statsReadbackBuffer.unmap();
      readbackPending = false;
      chartCanvas.setMaxVal(val);
    });
  }

  function render() {
    const now = performance.now();
    const dt = (now - lastTime) / 1000;
    lastTime = now;
    frameCount++;
    fpsAccTime += dt;
    if (fpsAccTime >= 0.5) {
      controls.fps = `${(frameCount / fpsAccTime).toFixed(1)} fps`;
      fpsController.updateDisplay();
      frameCount = 0;
      fpsAccTime = 0;
    }

    viewManager.tick(dt);

    updateUniform(
      gpu.device,
      uniformBuffer,
      viewManager.getViewMinX(),
      viewManager.getViewMaxX(),
      viewManager.getViewMinY(),
      viewManager.getViewMaxY(),
      hdrW,
      hdrH,
      kernelSize,
    );

    gpu.device.queue.writeBuffer(statsBuffer, 0, new Uint32Array([0]));

    const encoder = gpu.device.createCommandEncoder();

    const accPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: hdrTexture.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    const viewMinX = viewManager.getViewMinX();
    const viewMaxX = viewManager.getViewMaxX();
    const viewMinY = viewManager.getViewMinY();
    const viewMaxY = viewManager.getViewMaxY();

    if (
      viewMinX !== lastViewMinX ||
      viewMaxX !== lastViewMaxX ||
      viewMinY !== lastViewMinY ||
      viewMaxY !== lastViewMaxY
    ) {
      const merged = mergePoints(
        dataF64,
        viewMinX,
        viewMaxX,
        viewMinY,
        viewMaxY,
        hdrW,
        hdrH,
      );
      gpu.device.queue.writeBuffer(instanceBuffer, 0, merged);
      lastMergedCount = merged.length / 3;
      controls.renderedPoints = lastMergedCount;
      renderedPointsController.updateDisplay();
      chartCanvas.setMergedPoints(merged);
      lastViewMinX = viewMinX;
      lastViewMaxX = viewMaxX;
      lastViewMinY = viewMinY;
      lastViewMaxY = viewMaxY;
    }

    accPass.setPipeline(accumulationPipeline);
    accPass.setBindGroup(0, accBindGroup);
    accPass.setVertexBuffer(0, quadBuffer);
    accPass.setVertexBuffer(1, instanceBuffer);
    accPass.draw(6, lastMergedCount);
    accPass.end();

    const reductionPass = encoder.beginComputePass();
    reductionPass.setPipeline(reductionPipeline);
    reductionPass.setBindGroup(0, reductionBindGroup);
    reductionPass.dispatchWorkgroups(Math.ceil(hdrW / 8), Math.ceil(hdrH / 8));
    reductionPass.end();

    const tonemapPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: gpu.context.getCurrentTexture().createView(),
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    tonemapPass.setPipeline(tonemapPipeline);
    tonemapPass.setBindGroup(0, tonemapBindGroup);
    tonemapPass.draw(3);
    tonemapPass.end();

    gpu.device.queue.submit([encoder.finish()]);
    scheduleReadback();
    chartCanvas.render(
      viewManager.getViewMinX(),
      viewManager.getViewMaxX(),
      viewManager.getViewMinY(),
      viewManager.getViewMaxY(),
    );
    requestAnimationFrame(render);
  }

  render();

  window.addEventListener("resize", () => {
    resizeHeatmapCanvas(canvas);
    hdrW = canvas.width;
    hdrH = canvas.height;
    lastViewMinX = NaN; // force re-merge with new screen dimensions
    hdrTexture.destroy();
    hdrTexture = createHDRTexture(gpu.device, hdrW, hdrH);
    reductionBindGroup = createReductionBindGroup(
      gpu.device,
      reductionPipeline,
      hdrTexture,
      statsBuffer,
    );
    tonemapBindGroup = createTonemapBindGroup(
      gpu.device,
      tonemapPipeline,
      hdrTexture,
      statsBuffer,
      colorBuffer,
    );
  });
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
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
}

function createAccumulationPipeline(device: GPUDevice) {
  // Fraction of canvas height reserved for top+bottom padding
  const padFrac = CHART_PAD_Y / (window.innerHeight - AXIS_X_H);
  const module = device.createShaderModule({
    code: /* wgsl */ `
      struct Uniforms {
        viewMinX: f32, viewMaxX: f32,
        minY: f32, maxY: f32,
        screenWidth: f32, screenHeight: f32,
        kernelSize: f32, _pad: f32,
      };

      @group(0) @binding(0) var<uniform> u: Uniforms;

      struct VertexOut {
        @builtin(position) pos: vec4f,
        @location(0) offset: vec2f,
        @location(1) weight: f32,
      };

      @vertex
      fn vs_main(
        @location(0) quadOffset: vec2f,
        @location(1) point: vec2f,
        @location(2) weight: f32,
      ) -> VertexOut {
        // Cull points outside the X view range (move to degenerate clip position)
        if (point.x < u.viewMinX || point.x > u.viewMaxX) {
          return VertexOut(vec4f(10.0, 10.0, 10.0, 1.0), vec2f(0.0), 0.0);
        }

        let padFrac = ${padFrac.toFixed(6)}f;
        let scale = 1.0 - 2.0 * padFrac;
        let nx = (point.x - u.viewMinX) / (u.viewMaxX - u.viewMinX) * 2.0 - 1.0;
        let ny = ((point.y - u.minY) / (u.maxY - u.minY) * 2.0 - 1.0) * scale;
        let r = vec2f(u.kernelSize / u.screenWidth, u.kernelSize / u.screenHeight);
        return VertexOut(
          vec4f(nx + quadOffset.x * r.x, ny + quadOffset.y * r.y, 0.0, 1.0),
          quadOffset,
          weight,
        );
      }

      @fragment
      fn fs_main(@location(0) offset: vec2f, @location(1) weight: f32) -> @location(0) f32 {
        let d2 = dot(offset, offset);
        if d2 > 0.81 { discard; } // 0.9² — drop fragments outside the radius
        // 4/π normalizes the Gaussian so it integrates to 1 over ℝ²
        let result = (4.0 / 3.14159265) * exp(-4.0 * d2);
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
          arrayStride: 12,
          stepMode: "instance",
          attributes: [
            { shaderLocation: 1, offset: 0, format: "float32x2" },
            { shaderLocation: 2, offset: 8, format: "float32" },
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
      struct Palette { c0: vec4f, c1: vec4f, c2: vec4f, steps: f32, opacityCut: f32, _p2: f32, _p3: f32 };

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
        let t1 = smoothstep(0.0, 0.5, x);
        let t2 = smoothstep(0.5, 1.0, x);
        return mix(mix(palette.c0.rgb, palette.c1.rgb, t1), palette.c2.rgb, t2);
      }

      @fragment
      fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
        let accum = textureLoad(hdr, vec2i(pos.xy), 0).r;

        let maxVal = f32(stats[0]);
        var t = clamp(accum / maxVal, 0.0, 1.0);
        let opacity = step(palette.opacityCut, t);
        if palette.steps > 1.0 { 
          t = floor(t * palette.steps) / (palette.steps - 1.0); 
        }
        // let opacity = ;

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

function writePaletteToBuffer(device: GPUDevice, buffer: GPUBuffer) {
  const data = new Float32Array(16);
  const [r0, g0, b0] = hexToLinear(paletteColors.c0);
  const [r1, g1, b1] = hexToLinear(paletteColors.c1);
  const [r2, g2, b2] = hexToLinear(paletteColors.c2);
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
    quantSteps,
    opacityCut,
    0,
    0,
  ]);
  device.queue.writeBuffer(buffer, 0, data);
}
// prettier-ignore
const QUAD_CORNERS = new Float32Array([
  -1, -1,   1, -1,  -1,  1,
  -1,  1,   1, -1,   1,  1,
]);

function lowerBound(pts: Float64Array, x: number): number {
  let lo = 0,
    hi = pts.length / 3;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (pts[mid * 3]! < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function upperBound(pts: Float64Array, x: number): number {
  let lo = 0,
    hi = pts.length / 3;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (pts[mid * 3]! <= x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function mergePoints(
  pts: Float64Array,
  viewMinX: number,
  viewMaxX: number,
  viewMinY: number,
  viewMaxY: number,
  screenW: number,
  screenH: number,
): Float32Array {
  const toSX = (x: number) =>
    ((x - viewMinX) / (viewMaxX - viewMinX)) * screenW;
  const toSY = (y: number) =>
    ((y - viewMinY) / (viewMaxY - viewMinY)) * screenH;

  const startIdx = lowerBound(pts, viewMinX);
  const endIdx = upperBound(pts, viewMaxX);

  const merged = downsample({
    points: pts.slice(startIdx * 3, endIdx * 3),
    strategy: "merge",
    toSX,
    toSY,
    mergeThreshold,
  });

  return new Float32Array(merged);
}

function uploadData(device: GPUDevice, data: [number, number, number][]) {
  const quadBuffer = device.createBuffer({
    size: QUAD_CORNERS.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(quadBuffer, 0, QUAD_CORNERS);

  const instanceBuffer = device.createBuffer({
    size: data.length * 3 * 4, // worst case: no merging
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });

  // 8 floats: viewMinX, viewMaxX, minY, maxY, screenWidth, screenHeight, (2 padding)
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
      0,
    ]),
  );
}

function formatTimestamp(ms: number): string {
  const d = new Date(ms);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(Math.max((x - edge0) / (edge1 - edge0), 0), 1);
  return t * t * (3 - 2 * t);
}

function heatmapColor(value: number): [number, number, number] {
  const x = Math.min(Math.max(value, 0), 1);
  const [r0, g0, b0] = hexToLinear(paletteColors.c0);
  const [r1, g1, b1] = hexToLinear(paletteColors.c1);
  const [r2, g2, b2] = hexToLinear(paletteColors.c2);
  const c0 = [r0, g0, b0];
  const c1 = [r1, g1, b1];
  const c2 = [r2, g2, b2];
  const t1 = smoothstep(0.0, 0.5, x);
  const t2 = smoothstep(0.5, 1.0, x);
  const mid = c0.map((v, i) => v + (c1[i]! - v) * t1);
  const rgb = mid.map((v, i) => v + (c2[i]! - v) * t2);
  return [
    Math.round(rgb[0]! * 255),
    Math.round(rgb[1]! * 255),
    Math.round(rgb[2]! * 255),
  ];
}

class ChartCanvas {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private mergedPoints: Float32Array = new Float32Array(0);
  private maxVal = 1;
  private readonly xMin: number;
  private readonly xScale: number;

  private static readonly BAR_W = 16;
  private static readonly BAR_H = 160;
  private static readonly LABEL_W = 68;
  private static readonly PAD = 6;
  private static readonly TICKS_X = 8;
  private static readonly TICKS_Y = 6;
  private static readonly FONT = "11px monospace";
  private static readonly LEGEND_TITLE_H = 25;
  private static readonly LEGEND_W =
    ChartCanvas.BAR_W + ChartCanvas.PAD + ChartCanvas.LABEL_W;
  private static readonly LEGEND_H =
    ChartCanvas.BAR_H + ChartCanvas.PAD * 2 + ChartCanvas.LEGEND_TITLE_H;

  constructor(_data: [number, number, number][], xMin: number, xScale: number) {
    this.xMin = xMin;
    this.xScale = xScale;
    this.canvas = document.createElement("canvas");
    this.canvas.style.cssText =
      "position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:1;";
    document.body.appendChild(this.canvas);
    this.ctx = this.canvas.getContext("2d")!;
  }

  setMergedPoints(points: Float32Array) {
    this.mergedPoints = points;
  }

  setMaxVal(val: number) {
    if (val > 0 && val !== this.maxVal) {
      this.maxVal = val;
    }
  }

  render(
    viewMinX: number,
    viewMaxX: number,
    viewMinY: number,
    viewMaxY: number,
  ) {
    const dpr = devicePixelRatio;
    const cssW = window.innerWidth;
    const cssH = window.innerHeight;
    const w = Math.round(cssW * dpr);
    const h = Math.round(cssH * dpr);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }

    // heatmap occupies [0, cssW - AXIS_Y_W] × [CHART_PAD_Y, cssH - AXIS_X_H - CHART_PAD_Y]
    const hmW = cssW - AXIS_Y_W;
    const hmH = cssH - AXIS_X_H - CHART_PAD_Y * 2;

    const { ctx } = this;
    ctx.clearRect(0, 0, w, h);
    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.save();

    ctx.translate(0, CHART_PAD_Y);
    this.drawLinePlot(hmW, hmH, viewMinX, viewMaxX, viewMinY, viewMaxY);
    this.drawXAxis(hmW, hmH, cssH - CHART_PAD_Y * 2, viewMinX, viewMaxX);
    this.drawYAxis(hmW, hmH, cssW, viewMinY, viewMaxY);
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

    ctx.strokeStyle = "rgba(0,0,0,0.55)";
    ctx.lineWidth = 0.75;
    ctx.beginPath();
    const pts = this.mergedPoints;
    const n = pts.length / 3;
    for (let i = 0; i < n; i++) {
      const sx = toScreenX(pts[i * 3]!);
      const sy = toScreenY(pts[i * 3 + 1]!);
      if (i === 0) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
    }
    ctx.stroke();
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
    const stripY = hmH + CHART_PAD_Y; // top of the X-axis strip, after bottom padding gap

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
    const stripX = hmW; // left edge of the Y-axis strip

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
      ctx.fillText(val.toFixed(2), stripX + PAD + 5, y);
    }
    ctx.restore();
  }

  private drawLegend(_cssW: number, _cssH: number) {
    const { ctx } = this;
    const { BAR_W, BAR_H, PAD, FONT, LEGEND_TITLE_H, LEGEND_W, LEGEND_H } =
      ChartCanvas;

    // place legend at top-left corner of the heatmap area
    const ox = 0;
    const oy = 10;

    ctx.save();
    ctx.translate(ox, oy);

    // ctx.fillStyle = "rgba(245,245,245,0.95)";
    // ctx.beginPath();
    // ctx.roundRect(0, 0, LEGEND_W, LEGEND_H, 4);
    // ctx.fill();

    ctx.fillStyle = "#000";
    ctx.font = FONT;
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText("Pts/Kernel", PAD, 0);

    ctx.translate(PAD, LEGEND_TITLE_H);

    for (let py = 0; py < BAR_H; py++) {
      const t = 1 - py / (BAR_H - 1);
      const [r, g, b] = heatmapColor(t);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(0, py, BAR_W, 1);
    }

    ctx.textBaseline = "middle";
    for (let k = 0; k <= 5; k++) {
      const t = k / 5;
      const py = (1 - t) * (BAR_H - 1);
      const val = t * this.maxVal;
      ctx.fillStyle = "#000";
      ctx.fillText(val.toFixed(2), BAR_W + PAD, py);
    }

    ctx.restore();
  }
}

function createCanvas() {
  const canvas = document.createElement("canvas");
  canvas.style.cssText = "position:absolute;top:0;left:0;background:#ffffff;";
  document.body.appendChild(canvas);
  resizeHeatmapCanvas(canvas);
  return canvas;
}

function resizeHeatmapCanvas(canvas: HTMLCanvasElement) {
  const cssW = window.innerWidth - AXIS_Y_W;
  const cssH = window.innerHeight - AXIS_X_H;
  canvas.style.width = `${cssW}px`;
  canvas.style.height = `${cssH}px`;
  canvas.style.top = "0px";
  canvas.width = Math.round(cssW * devicePixelRatio);
  canvas.height = Math.round(cssH * devicePixelRatio);
}

class ViewManager {
  private readonly pts: Float64Array;
  private readonly dataMinY: number;
  private readonly dataMaxY: number;
  private readonly fullRangeX: number;
  private readonly minViewRangeX: number;
  private currentViewMinX: number;
  private currentViewMaxX: number;
  private targetViewMinX: number;
  private targetViewMaxX: number;
  private currentViewMinY: number;
  private currentViewMaxY: number;
  private isPanning = false;
  private lastPointerX = 0;
  private readonly interpolationRate = 12;

  constructor(pts: Float64Array) {
    this.pts = pts;
    const n = pts.length / 3;
    let minY = Infinity,
      maxY = -Infinity;
    for (let i = 0; i < n; i++) {
      const y = pts[i * 3 + 1]!;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    // x is normalized to [0, 1] and sorted
    this.dataMinY = minY;
    this.dataMaxY = maxY;
    this.fullRangeX = 1; // x span is always 1 after normalization
    this.minViewRangeX = Math.max(1 / 2 ** 16, 1e-9);
    this.currentViewMinX = 0;
    this.currentViewMaxX = 1;
    this.targetViewMinX = 0;
    this.targetViewMaxX = 1;
    this.currentViewMinY = minY;
    this.currentViewMaxY = maxY;
  }

  private computeVisibleYRange(): [number, number] {
    const { pts } = this;
    const startIdx = lowerBound(pts, this.currentViewMinX);
    const endIdx = upperBound(pts, this.currentViewMaxX);
    let minY = Infinity,
      maxY = -Infinity;
    for (let i = startIdx; i < endIdx; i++) {
      const y = pts[i * 3 + 1]!;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    if (!isFinite(minY)) return [this.dataMinY, this.dataMaxY];
    return [minY, maxY];
  }

  tick(dtSeconds: number): void {
    const alpha = 1 - Math.exp(-this.interpolationRate * dtSeconds);
    this.currentViewMinX +=
      (this.targetViewMinX - this.currentViewMinX) * alpha;
    this.currentViewMaxX +=
      (this.targetViewMaxX - this.currentViewMaxX) * alpha;
    if (Math.abs(this.targetViewMinX - this.currentViewMinX) < 1e-8)
      this.currentViewMinX = this.targetViewMinX;
    if (Math.abs(this.targetViewMaxX - this.currentViewMaxX) < 1e-8)
      this.currentViewMaxX = this.targetViewMaxX;

    const [targetMinY, targetMaxY] = this.computeVisibleYRange();
    this.currentViewMinY += (targetMinY - this.currentViewMinY) * alpha;
    this.currentViewMaxY += (targetMaxY - this.currentViewMaxY) * alpha;
  }

  getViewMinX(): number {
    return this.currentViewMinX;
  }
  getViewMaxX(): number {
    return this.currentViewMaxX;
  }
  getViewMinY(): number {
    return this.currentViewMinY;
  }
  getViewMaxY(): number {
    return this.currentViewMaxY;
  }

  bindCanvas(canvas: HTMLCanvasElement): void {
    canvas.style.touchAction = "none";

    canvas.addEventListener("pointerdown", (e) => {
      this.isPanning = true;
      this.lastPointerX = e.clientX;
      canvas.setPointerCapture(e.pointerId);
    });

    canvas.addEventListener("pointermove", (e) => {
      if (!this.isPanning) return;
      const rect = canvas.getBoundingClientRect();
      const deltaXRatio = (e.clientX - this.lastPointerX) / rect.width;
      this.lastPointerX = e.clientX;
      const span = this.targetViewMaxX - this.targetViewMinX;
      const deltaX = deltaXRatio * span;
      this.targetViewMinX -= deltaX;
      this.targetViewMaxX -= deltaX;
      this.clampTarget();
    });

    const stopPan = (e: PointerEvent) => {
      if (!this.isPanning) return;
      this.isPanning = false;
      canvas.releasePointerCapture(e.pointerId);
    };
    canvas.addEventListener("pointerup", stopPan);
    canvas.addEventListener("pointercancel", stopPan);

    canvas.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const currentSpan = this.targetViewMaxX - this.targetViewMinX;

        const isLateral = Math.abs(e.deltaX) > Math.abs(e.deltaY);

        if (isLateral && e.deltaX !== 0) {
          const deltaX = (e.deltaX / rect.width) * currentSpan;
          this.targetViewMinX += deltaX;
          this.targetViewMaxX += deltaX;
          this.clampTarget();
        } else if (e.deltaY !== 0) {
          const anchorRatio = Math.max(
            0,
            Math.min(1, (e.clientX - rect.left) / rect.width),
          );
          const zoomFactor = Math.exp(e.deltaY * 0.0015);
          const nextSpan = Math.max(
            this.minViewRangeX,
            Math.min(this.fullRangeX, currentSpan * zoomFactor),
          );
          if (!Number.isFinite(nextSpan) || nextSpan === currentSpan) return;
          const anchorX = this.targetViewMinX + anchorRatio * currentSpan;
          this.targetViewMinX = anchorX - anchorRatio * nextSpan;
          this.targetViewMaxX = this.targetViewMinX + nextSpan;
          this.clampTarget();
        }
      },
      { passive: false },
    );
  }

  private clampTarget(): void {
    const span = Math.max(
      this.targetViewMaxX - this.targetViewMinX,
      this.minViewRangeX,
    );
    this.targetViewMinX = Math.max(0, Math.min(this.targetViewMinX, 1 - span));
    this.targetViewMaxX = this.targetViewMinX + span;
  }
}
