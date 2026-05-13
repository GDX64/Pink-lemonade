import { createNoiseData } from "../../chart/chart";

const DOWNSCALE = 16;

export async function rasterizingExample() {
  const canvas = createCanvas();
  const data = createNoiseData(100_000);

  let minX = data[0]![0];
  let maxX = data[0]![0];
  let minY = data[0]![1];
  let maxY = data[0]![1];
  for (const [x, y] of data) {
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }

  const gpu = await initWebGPU(canvas);
  const accumulationPipeline = createAccumulationPipeline(gpu.device);
  const reductionPipeline = createReductionPipeline(gpu.device);
  const tonemapPipeline = createTonemapPipeline(gpu.device, gpu.format);
  const { quadBuffer, instanceBuffer, uniformBuffer } = uploadData(
    gpu.device,
    data,
  );

  // Holds a single u32: the max accumulation count across all pixels
  const statsBuffer = gpu.device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const accBindGroup = gpu.device.createBindGroup({
    layout: accumulationPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  let hdrW = Math.max(1, Math.ceil(canvas.width / DOWNSCALE));
  let hdrH = Math.max(1, Math.ceil(canvas.height / DOWNSCALE));
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

  const fpsEl = createFpsDisplay();
  let lastTime = performance.now();
  let frameCount = 0;

  function render() {
    const now = performance.now();
    frameCount++;
    const elapsed = now - lastTime;
    if (elapsed >= 500) {
      fpsEl.textContent = `${((frameCount / elapsed) * 1000).toFixed(1)} fps`;
      frameCount = 0;
      lastTime = now;
    }

    updateUniform(
      gpu.device,
      uniformBuffer,
      minX,
      maxX,
      minY,
      maxY,
      hdrW,
      hdrH,
    );

    // Reset max to 0 before reduction
    gpu.device.queue.writeBuffer(statsBuffer, 0, new Uint32Array([0]));

    const encoder = gpu.device.createCommandEncoder();

    // Pass 1: accumulate points into the low-res r32float HDR texture
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

    // Pass 2: reduce low-res HDR texture to find max accumulation value
    const reductionPass = encoder.beginComputePass();
    reductionPass.setPipeline(reductionPipeline);
    reductionPass.setBindGroup(0, reductionBindGroup);
    reductionPass.dispatchWorkgroups(Math.ceil(hdrW / 8), Math.ceil(hdrH / 8));
    reductionPass.end();

    // Pass 3: bicubic upsample + tonemap to full canvas resolution
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
    hdrW = Math.max(1, Math.ceil(canvas.width / DOWNSCALE));
    hdrH = Math.max(1, Math.ceil(canvas.height / DOWNSCALE));
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
    render();
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
        minX: f32, maxX: f32,
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
        let nx = (point.x - u.minX) / (u.maxX - u.minX) * 2.0 - 1.0;
        let ny = (point.y - u.minY) / (u.maxY - u.minY) * 2.0 - 1.0;
        const R = 6.0;
        let r = vec2f(R / u.screenWidth, R / u.screenHeight);
        return VertexOut(
          vec4f(nx + quadOffset.x * r.x, ny + quadOffset.y * r.y, 0.0, 1.0),
          quadOffset,
        );
      }

      @fragment
      fn fs_main(@location(0) offset: vec2f) -> @location(0) f32 {
        let d2 = dot(offset, offset);
        return max(0.0, 1.0 - d2);
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

      // Mitchell-Netravali cubic kernel (B=1/3, C=1/3)
      fn cubic_weight(x: f32) -> f32 {
        let ax = abs(x);
        let ax2 = ax * ax;
        let ax3 = ax2 * ax;
        const B = 1.0 / 3.0;
        const C = 1.0 / 3.0;
        if (ax < 1.0) {
          return ((12.0 - 9.0*B - 6.0*C) * ax3
                + (-18.0 + 12.0*B + 6.0*C) * ax2
                + (6.0 - 2.0*B)) / 6.0;
        } else if (ax < 2.0) {
          return ((-B - 6.0*C) * ax3
                + (6.0*B + 30.0*C) * ax2
                + (-12.0*B - 48.0*C) * ax
                + (8.0*B + 24.0*C)) / 6.0;
        }
        return 0.0;
      }

      fn bicubic_sample(uv: vec2f) -> f32 {
        let size = vec2f(textureDimensions(hdr));
        // pixel-space coordinate of the sample center
        let p = uv * size - 0.5;
        let p0 = floor(p);
        let frac = p - p0;

        var result = 0.0;
        for (var j = -1; j <= 2; j++) {
          let wy = cubic_weight(frac.y - f32(j));
          for (var i = -1; i <= 2; i++) {
            let wx = cubic_weight(frac.x - f32(i));
            let coord = vec2i(p0) + vec2i(i, j);
            let clamped = clamp(coord, vec2i(0), vec2i(size) - vec2i(1));
            result += wx * wy * textureLoad(hdr, clamped, 0).r;
          }
        }
        return max(result, 0.0);
      }

      @fragment
      fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
        let outSize = vec2f(textureDimensions(hdr) * ${DOWNSCALE}u);
        let uv = pos.xy / outSize;
        let accum = bicubic_sample(uv);

        let maxVal = f32(stats[0]);
        let t = clamp(accum / maxVal, 0.0, 1.0);

        // Classic heatmap: black -> blue -> cyan -> green -> yellow -> red -> white
        var color: vec3f;
        if (t < 1.0 / 6.0) {
          color = mix(vec3f(0.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), t * 6.0);
        } else if (t < 2.0 / 6.0) {
          color = mix(vec3f(0.0, 0.0, 1.0), vec3f(0.0, 1.0, 1.0), (t - 1.0 / 6.0) * 6.0);
        } else if (t < 3.0 / 6.0) {
          color = mix(vec3f(0.0, 1.0, 1.0), vec3f(0.0, 1.0, 0.0), (t - 2.0 / 6.0) * 6.0);
        } else if (t < 4.0 / 6.0) {
          color = mix(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 1.0, 0.0), (t - 3.0 / 6.0) * 6.0);
        } else if (t < 5.0 / 6.0) {
          color = mix(vec3f(1.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), (t - 4.0 / 6.0) * 6.0);
        } else {
          color = mix(vec3f(1.0, 0.0, 0.0), vec3f(1.0, 1.0, 1.0), (t - 5.0 / 6.0) * 6.0);
        }

        // Gamma correction (linear -> sRGB)
        // let gamma = pow(color, vec3f(1.0 / 2.2));
        let gamma = color;
        return vec4f(gamma, 1.0);
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

  const uniformBuffer = device.createBuffer({
    size: 6 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(
    uniformBuffer,
    0,
    new Float32Array([0, 1, 0, 1, 1, 1]),
  );

  return { quadBuffer, instanceBuffer, uniformBuffer };
}

function updateUniform(
  device: GPUDevice,
  uniformBuffer: GPUBuffer,
  minX: number,
  maxX: number,
  minY: number,
  maxY: number,
  width: number,
  height: number,
) {
  device.queue.writeBuffer(
    uniformBuffer,
    0,
    new Float32Array([minX, maxX, minY, maxY, width, height]),
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

