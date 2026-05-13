import { createNoiseData } from "../../chart/chart";

export async function rasterizingExample() {
  const canvas = createCanvas();
  const data = createNoiseData(1000);

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
  const pipeline = createGPUPipeline(gpu.device, gpu.format);
  const { quadBuffer, instanceBuffer, uniformBuffer } = uploadData(
    gpu.device,
    data,
    minX,
    maxX,
    minY,
    maxY,
  );

  const bindGroup = gpu.device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  function render() {
    updateUniform(
      gpu.device,
      uniformBuffer,
      minX,
      maxX,
      minY,
      maxY,
      canvas.width,
      canvas.height,
    );
    const encoder = gpu.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: gpu.context.getCurrentTexture().createView(),
          clearValue: { r: 0.05, g: 0.05, b: 0.08, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, quadBuffer);
    pass.setVertexBuffer(1, instanceBuffer);
    pass.draw(6, data.length);
    pass.end();
    gpu.device.queue.submit([encoder.finish()]);
  }

  render();

  window.addEventListener("resize", () => {
    canvas.width = canvas.getBoundingClientRect().width * devicePixelRatio;
    canvas.height = canvas.getBoundingClientRect().height * devicePixelRatio;
    render();
  });
}

async function initWebGPU(canvas: HTMLCanvasElement) {
  if (!navigator.gpu) throw new Error("WebGPU not supported");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter found");
  const device = await adapter.requestDevice();
  const context = canvas.getContext("webgpu")!;
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "premultiplied" });
  return { device, context, format };
}

function createGPUPipeline(device: GPUDevice, format: GPUTextureFormat) {
  const shaderModule = device.createShaderModule({
    code: /* wgsl */ `
      struct Uniforms {
        minX: f32,
        maxX: f32,
        minY: f32,
        maxY: f32,
        screenWidth: f32,
        screenHeight: f32,
      };

      @group(0) @binding(0) var<uniform> u: Uniforms;

      @vertex
      fn vs_main(
        @location(0) quadOffset: vec2f,  // per-vertex: corner of the unit quad
        @location(1) point: vec2f,        // per-instance: data point in data space
      ) -> @builtin(position) vec4f {
        let nx = (point.x - u.minX) / (u.maxX - u.minX) * 2.0 - 1.0;
        let ny = (point.y - u.minY) / (u.maxY - u.minY) * 2.0 - 1.0;
        const R = 10.0;
        let pointRadius = vec2f(R / u.screenWidth, R / u.screenHeight);
        return vec4f(nx + quadOffset.x * pointRadius.x, ny + quadOffset.y * pointRadius.y, 0.0, 1.0);
      }

      @fragment
      fn fs_main() -> @location(0) vec4f {
        return vec4f(0.2, 0.7, 1.0, 1.0);
      }
    `,
  });

  return device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
      buffers: [
        {
          // slot 0: quad geometry, one corner per vertex
          arrayStride: 8,
          stepMode: "vertex",
          attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
        },
        {
          // slot 1: data points, one per instance
          arrayStride: 8,
          stepMode: "instance",
          attributes: [{ shaderLocation: 1, offset: 0, format: "float32x2" }],
        },
      ],
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_main",
      targets: [{ format }],
    },
    primitive: { topology: "triangle-list" },
  });
}

// prettier-ignore
const QUAD_CORNERS = new Float32Array([
  -1, -1,   1, -1,  -1,  1,
  -1,  1,   1, -1,   1,  1,
]);

function uploadData(
  device: GPUDevice,
  data: [number, number][],
  minX: number,
  maxX: number,
  minY: number,
  maxY: number,
) {
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
    new Float32Array([minX, maxX, minY, maxY, 1, 1]),
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

