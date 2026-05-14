import { createNoiseData } from "../../chart/chart";

export async function rasterizingExample() {
  const canvas = createCanvas();
  const data = createNoiseData(10_000);

  let dataMinX = data[0]![0];
  let dataMaxX = data[0]![0];
  for (const [x] of data) {
    dataMinX = Math.min(dataMinX, x);
    dataMaxX = Math.max(dataMaxX, x);
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
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

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
  );

  const viewManager = new ViewManager(data);
  viewManager.bindCanvas(canvas);

  const fpsEl = createFpsDisplay();
  let lastTime = performance.now();
  let fpsAccTime = 0;
  let frameCount = 0;

  function render() {
    const now = performance.now();
    const dt = (now - lastTime) / 1000;
    lastTime = now;
    frameCount++;
    fpsAccTime += dt;
    if (fpsAccTime >= 0.5) {
      fpsEl.textContent = `${(frameCount / fpsAccTime).toFixed(1)} fps`;
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
    accPass.setPipeline(accumulationPipeline);
    accPass.setBindGroup(0, accBindGroup);
    accPass.setVertexBuffer(0, quadBuffer);
    accPass.setVertexBuffer(1, instanceBuffer);
    accPass.draw(6, data.length);
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
    requestAnimationFrame(render);
  }

  render();

  window.addEventListener("resize", () => {
    canvas.width = canvas.getBoundingClientRect().width * devicePixelRatio;
    canvas.height = canvas.getBoundingClientRect().height * devicePixelRatio;
    hdrW = canvas.width;
    hdrH = canvas.height;
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
  const module = device.createShaderModule({
    code: /* wgsl */ `
      struct Uniforms {
        viewMinX: f32, viewMaxX: f32,
        minY: f32, maxY: f32,
        screenWidth: f32, screenHeight: f32,
      };

      @group(0) @binding(0) var<uniform> u: Uniforms;

      struct VertexOut {
        @builtin(position) pos: vec4f,
        @location(0) offset: vec2f,
      };

      @vertex
      fn vs_main(
        @location(0) quadOffset: vec2f,
        @location(1) point: vec2f,
      ) -> VertexOut {
        // Cull points outside the X view range (move to degenerate clip position)
        if (point.x < u.viewMinX || point.x > u.viewMaxX) {
          return VertexOut(vec4f(10.0, 10.0, 10.0, 1.0), vec2f(0.0));
        }

        let nx = (point.x - u.viewMinX) / (u.viewMaxX - u.viewMinX) * 2.0 - 1.0;
        let ny = (point.y - u.minY) / (u.maxY - u.minY) * 2.0 - 1.0;
        const R = 120.0;
        let r = vec2f(R / u.screenWidth, R / u.screenHeight);
        return VertexOut(
          vec4f(nx + quadOffset.x * r.x, ny + quadOffset.y * r.y, 0.0, 1.0),
          quadOffset,
        );
      }

      @fragment
      fn fs_main(@location(0) offset: vec2f) -> @location(0) f32 {
        let d2 = dot(offset, offset);
        return exp(-4.0 * d2);
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
          arrayStride: 8,
          stepMode: "instance",
          attributes: [{ shaderLocation: 1, offset: 0, format: "float32x2" }],
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
      @group(0) @binding(0) var hdr: texture_2d<f32>;
      @group(0) @binding(1) var<storage, read> stats: array<u32, 1>;

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
        // Classic heatmap: black -> blue -> cyan -> green -> yellow -> red -> white
        let c0 = vec3f(0.0, 0.0, 0.0);
        let c1 = vec3f(0.0, 0.0, 1.0);
        let c2 = vec3f(0.0, 1.0, 1.0);
        let c3 = vec3f(0.0, 1.0, 0.0);
        let c4 = vec3f(1.0, 1.0, 0.0);
        let c5 = vec3f(1.0, 0.0, 0.0);
        let c6 = vec3f(1.0, 1.0, 1.0);
        let s = x * 6.0;
        let i = floor(s);
        let f = s - i;
        var a: vec3f;
        var b: vec3f;
        if (i < 1.0)      { a = c0; b = c1; }
        else if (i < 2.0) { a = c1; b = c2; }
        else if (i < 3.0) { a = c2; b = c3; }
        else if (i < 4.0) { a = c3; b = c4; }
        else if (i < 5.0) { a = c4; b = c5; }
        else               { a = c5; b = c6; }
        return mix(a, b, f);
      }

      @fragment
      fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
        let accum = textureLoad(hdr, vec2i(pos.xy), 0).r;

        let maxVal = f32(stats[0]);
        let t = clamp(accum / maxVal, 0.0, 1.0);

        return vec4f(mapColor(t), 1.0);
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
) {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: hdrTexture.createView() },
      { binding: 1, resource: { buffer: statsBuffer } },
    ],
  });
}

// prettier-ignore
const QUAD_CORNERS = new Float32Array([
  -1, -1,   1, -1,  -1,  1,
  -1,  1,   1, -1,   1,  1,
]);

function uploadData(device: GPUDevice, data: [number, number][]) {
  const quadBuffer = device.createBuffer({
    size: QUAD_CORNERS.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(quadBuffer, 0, QUAD_CORNERS);

  const points = new Float32Array(data.length * 2);
  for (let i = 0; i < data.length; i++) {
    points[i * 2] = data[i]![0];
    points[i * 2 + 1] = data[i]![1];
  }
  const instanceBuffer = device.createBuffer({
    size: points.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(instanceBuffer, 0, points);

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
) {
  device.queue.writeBuffer(
    uniformBuffer,
    0,
    new Float32Array([viewMinX, viewMaxX, minY, maxY, width, height, 0, 0]),
  );
}

function createFpsDisplay() {
  const el = document.createElement("div");
  el.style.cssText =
    "position:fixed;top:8px;left:8px;color:#fff;font:12px monospace;background:rgba(0,0,0,.5);padding:2px 6px;border-radius:4px;pointer-events:none;";
  document.body.appendChild(el);
  return el;
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
  private readonly data: [number, number][];
  private readonly dataMinX: number;
  private readonly dataMaxX: number;
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

  constructor(data: [number, number][]) {
    this.data = data;
    let minX = Infinity,
      maxX = -Infinity;
    let minY = Infinity,
      maxY = -Infinity;
    for (const [x, y] of data) {
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
    this.dataMinX = minX;
    this.dataMaxX = maxX;
    this.dataMinY = minY;
    this.dataMaxY = maxY;
    this.fullRangeX = Math.max(this.dataMaxX - this.dataMinX, 1e-6);
    this.minViewRangeX = Math.max(this.fullRangeX / 512, 1e-6);
    this.currentViewMinX = this.dataMinX;
    this.currentViewMaxX = this.dataMaxX;
    this.targetViewMinX = this.dataMinX;
    this.targetViewMaxX = this.dataMaxX;
    this.currentViewMinY = minY;
    this.currentViewMaxY = maxY;
  }

  private computeVisibleYRange(): [number, number] {
    let minY = Infinity,
      maxY = -Infinity;
    for (const [x, y] of this.data) {
      if (x < this.currentViewMinX || x > this.currentViewMaxX) continue;
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
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
        const anchorRatio = Math.max(
          0,
          Math.min(1, (e.clientX - rect.left) / rect.width),
        );
        const currentSpan = this.targetViewMaxX - this.targetViewMinX;
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
      },
      { passive: false },
    );
  }

  private clampTarget(): void {
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
