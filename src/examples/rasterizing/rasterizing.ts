import GUI from "lil-gui";
import { createNoiseData } from "../../chart/chart";
import { downsample } from "./downsampling";
import { lowerBound, upperBound, ViewManager } from "./view-manager";

const AXIS_Y_W = 70; // px reserved on the right for the Y axis
const AXIS_X_H = 30; // px reserved on the bottom for the X axis
const CHART_PAD_Y = 20; // px of top+bottom padding inside the heatmap canvas

// RGB triplets for the low/mid/high stops of the colormap
const paletteColors = {
  c0: "#d1edff",
  c1: "#feffb8",
  c2: "#f28787",
};

let sigmaSize = 17; // in pixels; the radius of the Gaussian "splat" for each point
let quantSteps = 0;
let opacityCut = 0.03;
let mergeThreshold = 20;
let paletteLevel = 0.5;
let showLine = true;
let lineOpacity = 0.55;
let lineWidth = 0.75;

const DATA_KIND = "random";

export async function rasterizingExample() {
  const canvas = createCanvas();
  //   canvas.style.opacity = "0.75";
  // const data = createNoiseData(100_000);
  const { n, dataF64, xMin, xScale } = await loadData();

  const gpu = await initWebGPU(canvas);
  const accumulationPipeline = createAccumulationPipeline(gpu.device);
  const reductionPipeline = createReductionPipeline(gpu.device);
  const tonemapPipeline = createTonemapPipeline(gpu.device, gpu.format);
  const { quadBuffer, instanceBuffer, uniformBuffer } = uploadData(
    gpu.device,
    n,
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
    sigmaSize,
    fps: "0.0",
    totalPoints: n,
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
    .add(controls, "sigmaSize", 1, 50, 1)
    .name("Kernel sigma (px)")
    .onChange((v: number) => {
      sigmaSize = v;
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
    .add({ paletteLevel }, "paletteLevel", 0.01, 0.99, 0.01)
    .name("Mid level")
    .onChange((v: number) => {
      paletteLevel = v;
      writePaletteToBuffer(gpu.device, colorBuffer);
    });
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
  const lineFolder = gui.addFolder("Line chart");
  lineFolder
    .add({ showLine }, "showLine")
    .name("Show line")
    .onChange((v: boolean) => {
      showLine = v;
    });
  lineFolder
    .add({ lineOpacity }, "lineOpacity", 0.01, 1, 0.01)
    .name("Opacity")
    .onChange((v: number) => {
      lineOpacity = v;
    });
  lineFolder
    .add({ lineWidth }, "lineWidth", 0.25, 5, 0.25)
    .name("Line width")
    .onChange((v: number) => {
      lineWidth = v;
    });

  const chartCanvas = new ChartCanvas(xMin, xScale);
  chartCanvas.setOnLevelChange((v) => {
    paletteLevel = v;
    writePaletteToBuffer(gpu.device, colorBuffer);
  });
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
      sigmaSize,
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

async function loadData() {
  if (DATA_KIND === "random") {
    const points = createNoiseData(100_000);
    const nFiltered = points.length;
    if (nFiltered === 0) {
      return { n: 0, dataF64: new Float64Array(0), xMin: 0, xScale: 1 };
    }

    // createNoiseData returns [time, price] points; normalize x and keep unit weight
    const xMin = points[0]![0];
    const xMax = points[nFiltered - 1]![0];
    const xScale = xMax - xMin || 1;
    const dataF64 = new Float64Array(nFiltered * 3);
    for (let i = 0; i < nFiltered; i++) {
      const [time, price] = points[i]!;
      dataF64[i * 3] = (time - xMin) / xScale;
      dataF64[i * 3 + 1] = price;
      dataF64[i * 3 + 2] = 1;
    }
    return { n: nFiltered, dataF64, xMin, xScale };
  }

  const jsonData = (await import("./data.json?url")).default;
  const { base64: base64Data }: { base64: string } = await (
    await fetch(jsonData)
  ).json();
  const other = base64Data.replaceAll("'", "");
  const decoded = atob(other);
  const buffer = new Uint8Array(decoded.length);
  for (let i = 0; i < decoded.length; i++) {
    buffer[i] = decoded.charCodeAt(i);
  }
  const raw = new Float64Array(buffer.buffer);

  // data is flat: [time, price, buyQty, sellQty, ...] — sort by time ascending
  const n = raw.length / 4;
  const indices = Array.from({ length: n }, (_, i) => i).sort(
    (a, b) => raw[a * 4]! - raw[b * 4]!,
  );
  const data = new Float64Array(n * 4);
  for (let i = 0; i < n; i++) {
    const src = indices[i]! * 4;
    data.set(raw.subarray(src, src + 4), i * 4);
  }

  const nFiltered = data.length / 4;

  // normalize x to [0, 1] to avoid float precision loss in GPU/math
  const xMin = data[0]!;
  const xMax = data[(nFiltered - 1) * 4]!;
  const xScale = xMax - xMin || 1; // original span in ms; used to denormalize for display

  // pre-pack into Float64Array once — x is sorted and normalized, weight = buyQty + sellQty
  const dataF64 = new Float64Array(nFiltered * 3);
  for (let i = 0; i < nFiltered; i++) {
    const time = data[i * 4]!;
    const price = data[i * 4 + 1]!;
    const buyQty = data[i * 4 + 2]!;
    const sellQty = data[i * 4 + 3]!;
    dataF64[i * 3] = (time - xMin) / xScale;
    dataF64[i * 3 + 1] = price;
    dataF64[i * 3 + 2] = buyQty + sellQty;
  }
  return { n: nFiltered, dataF64, xMin, xScale };
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
        sigmaSize: f32, _pad: f32,
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
        let kernelSize = u.sigmaSize * 3.0 * 2.0;
        let r = vec2f(kernelSize / u.screenWidth, kernelSize / u.screenHeight);
        return VertexOut(
          vec4f(nx + quadOffset.x * r.x, ny + quadOffset.y * r.y, 0.0, 1.0),
          quadOffset,
          weight,
        );
      }

      @fragment
      fn fs_main(@location(0) offset: vec2f, @location(1) weight: f32) -> @location(0) f32 {
        let d2 = dot(offset, offset);
        // we are doing a change of variables here
        // the ideia is that d = 1 represents 3*sigma
        // if d2 > 1.0 { discard; } // 0.9² — drop fragments outside the radius
        // 4/π normalizes the Gaussian so it integrates to 1 over ℝ²
        //d2 is in [0, 1]
        //so exp(- dˆ2 / (2 sigma²)) when d is 1 should be exp(- (3 sigma)ˆ2 / (2 sigmaˆ2)) = exp(-4.5)
        let result = exp(- d2 * 4.5);
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
    paletteLevel,
    0,
  ]);
  device.queue.writeBuffer(buffer, 0, data);
}
// prettier-ignore
const QUAD_CORNERS = new Float32Array([
  -1, -1,   1, -1,  -1,  1,
  -1,  1,   1, -1,   1,  1,
]);

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
    threshold: 1000,
  });

  return new Float32Array(merged);
}

function uploadData(device: GPUDevice, pointCount: number) {
  const quadBuffer = device.createBuffer({
    size: QUAD_CORNERS.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(quadBuffer, 0, QUAD_CORNERS);

  const instanceBuffer = device.createBuffer({
    size: pointCount * 3 * 4, // worst case: no merging
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
  const t1 = smoothstep(0.0, paletteLevel, x);
  const t2 = smoothstep(paletteLevel, 1.0, x);
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
  private onLevelChange: ((level: number) => void) | null = null;
  private isDraggingLevel = false;

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

  // legend origin in CSS pixels (set during render, used for hit-testing)
  private legendBarTop = 0;
  private legendBarLeft = 0;

  constructor(xMin: number, xScale: number) {
    this.xMin = xMin;
    this.xScale = xScale;
    this.canvas = document.createElement("canvas");
    this.canvas.style.cssText =
      "position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:1;";
    document.body.appendChild(this.canvas);
    this.ctx = this.canvas.getContext("2d")!;
    this.bindLevelDrag();
  }

  setOnLevelChange(cb: (level: number) => void) {
    this.onLevelChange = cb;
  }

  private bindLevelDrag() {
    const { BAR_H } = ChartCanvas;

    const hitTest = (cssX: number, cssY: number) => {
      const relY = cssY - this.legendBarTop;
      const relX = cssX - this.legendBarLeft;
      const triY = (1 - paletteLevel) * (BAR_H - 1);
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

    window.addEventListener("pointerdown", (e) => {
      if (!hitTest(e.clientX, e.clientY)) return;
      this.isDraggingLevel = true;
      this.canvas.style.pointerEvents = "auto";
      this.canvas.setPointerCapture(e.pointerId);
      e.stopPropagation();
    });

    this.canvas.addEventListener("pointermove", (e) => {
      if (!this.isDraggingLevel) return;
      this.onLevelChange?.(levelFromY(e.clientY));
    });

    const stopDrag = (e: PointerEvent) => {
      if (!this.isDraggingLevel) return;
      this.isDraggingLevel = false;
      this.canvas.style.pointerEvents = "none";
      this.canvas.releasePointerCapture(e.pointerId);
    };
    this.canvas.addEventListener("pointerup", stopDrag);
    this.canvas.addEventListener("pointercancel", stopDrag);

    // change cursor when hovering the triangle
    window.addEventListener("pointermove", (e) => {
      if (this.isDraggingLevel) return;
      document.body.style.cursor = hitTest(e.clientX, e.clientY)
        ? "ns-resize"
        : "";
    });
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

    if (!showLine) {
      ctx.restore();
      return;
    }
    ctx.strokeStyle = `rgba(0,0,0,${lineOpacity})`;
    ctx.lineWidth = lineWidth;
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
    ctx.fillText("Satoshi", PAD, 0);

    ctx.translate(PAD, LEGEND_TITLE_H);

    // record bar origin in CSS px for hit-testing
    this.legendBarTop = oy + LEGEND_TITLE_H;
    this.legendBarLeft = ox + PAD;

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
      let val = t * this.maxVal;
      if (t === 0) {
        val = opacityCut * this.maxVal;
      }
      ctx.fillStyle = "#000";
      ctx.fillText(fmtCompact(val), BAR_W + PAD, py);
    }

    // draw level bar across the gradient
    const lineY = (1 - paletteLevel) * (BAR_H - 1);
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
