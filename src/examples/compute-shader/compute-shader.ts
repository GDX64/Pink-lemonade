import { createNoiseData } from "../../chart/chart";
import computeShaderSource from "./heatmap.compute.wgsl?raw";
import renderShaderSource from "./heatmap.render.wgsl?raw";

export async function example() {
  const canvas = createCanvas();

  if (!("gpu" in navigator)) {
    throw new Error("WebGPU is not supported in this browser.");
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("Failed to get GPU adapter.");
  }

  const device = await adapter.requestDevice();
  const context = canvas.getContext("webgpu");
  if (!context) {
    throw new Error("Failed to get WebGPU context.");
  }

  const format = navigator.gpu.getPreferredCanvasFormat();
  const configureContext = () => {
    resizeCanvasToDisplaySize(canvas);
    context.configure({
      device,
      format,
      alphaMode: "opaque",
    });
  };

  configureContext();

  const vertexStride = 2 * Float32Array.BYTES_PER_ELEMENT;
  const quadVertices = new Float32Array([
    -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0,
  ]);
  const vertexCount = quadVertices.length / 2;
  const vertexBuffer = device.createBuffer({
    size: quadVertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, quadVertices);

  const textureWidth = Math.floor(canvas.width / 32);
  const textureHeight = Math.floor(canvas.height / 32);
  const texelCount = textureWidth * textureHeight;
  const noiseTexture = device.createTexture({
    size: { width: textureWidth, height: textureHeight },
    format: "r32float",
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });

  const noiseView = noiseTexture.createView();

  const data = createNoiseData(10_000);
  data.sort((a, b) => a[0]! - b[0]!);
  const f32Data = new Float32Array(data.flat());
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < f32Data.length; i += 2) {
    const x = f32Data[i]!;
    const y = f32Data[i + 1]!;
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  const dataMinX = minX;
  const dataMaxX = maxX;
  const dataMinY = minY;
  const dataMaxY = maxY;

  const pointCount = f32Data.length / 2;
  const pointBuffer = device.createBuffer({
    size: f32Data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(pointBuffer, 0, f32Data);

  const xRange = Math.max(1e-6, maxX - minX);
  const yRange = Math.max(1e-6, maxY - minY);
  const scaleX = (textureWidth - 1) / xRange;
  const scaleY = (textureHeight - 1) / yRange;

  const cpuCounts = new Uint32Array(texelCount);
  for (let i = 0; i < f32Data.length; i += 2) {
    const xFloat = (f32Data[i]! - minX) * scaleX;
    const yFloat = (f32Data[i + 1]! - minY) * scaleY;
    const x = Math.min(textureWidth - 1, Math.max(0, Math.trunc(xFloat)));
    const y = Math.min(textureHeight - 1, Math.max(0, Math.trunc(yFloat)));
    cpuCounts[y * textureWidth + x]! += 1;
  }

  let minCount = Number.POSITIVE_INFINITY;
  let maxCount = 0;
  for (let i = 0; i < cpuCounts.length; i++) {
    const count = cpuCounts[i]!;
    if (count < minCount) {
      minCount = count;
    }
    if (count > maxCount) {
      maxCount = count;
    }
  }

  const countRange = Math.max(1, maxCount - minCount);
  const invCountRange = 1 / countRange;

  const paramsBytes = 12 * Uint32Array.BYTES_PER_ELEMENT;
  const paramsData = new ArrayBuffer(paramsBytes);
  const paramsView = new DataView(paramsData);
  paramsView.setUint32(16, pointCount, true);
  paramsView.setUint32(20, textureWidth, true);
  paramsView.setUint32(24, textureHeight, true);
  paramsView.setFloat32(28, minCount, true);
  paramsView.setFloat32(32, invCountRange, true);
  paramsView.setUint32(36, 0, true);
  paramsView.setUint32(40, 0, true);
  paramsView.setUint32(44, 0, true);
  const paramsBuffer = device.createBuffer({
    size: paramsBytes,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  let zoom = 1;
  let centerX = (dataMinX + dataMaxX) * 0.5;
  let centerY = (dataMinY + dataMaxY) * 0.5;

  const fullRangeX = Math.max(1e-6, dataMaxX - dataMinX);
  const fullRangeY = Math.max(1e-6, dataMaxY - dataMinY);

  const clampCenterToData = () => {
    const viewRangeX = fullRangeX / zoom;
    const viewRangeY = fullRangeY / zoom;
    const halfX = viewRangeX * 0.5;
    const halfY = viewRangeY * 0.5;

    centerX = clamp(centerX, dataMinX + halfX, dataMaxX - halfX);
    centerY = clamp(centerY, dataMinY + halfY, dataMaxY - halfY);
  };

  const writeComputeParams = () => {
    const viewRangeX = fullRangeX / zoom;
    const viewRangeY = fullRangeY / zoom;
    const viewMinX = centerX - viewRangeX * 0.5;
    const viewMinY = centerY - viewRangeY * 0.5;
    const viewScaleX = (textureWidth - 1) / Math.max(1e-6, viewRangeX);
    const viewScaleY = (textureHeight - 1) / Math.max(1e-6, viewRangeY);

    paramsView.setFloat32(0, viewMinX, true);
    paramsView.setFloat32(4, viewMinY, true);
    paramsView.setFloat32(8, viewScaleX, true);
    paramsView.setFloat32(12, viewScaleY, true);
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);
  };

  writeComputeParams();

  const computeShader = device.createShaderModule({
    code: computeShaderSource,
  });

  const computePipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: computeShader,
      entryPoint: "buildHeatmap",
    },
  });

  const computeBindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: {
          buffer: pointBuffer,
        },
      },
      {
        binding: 1,
        resource: noiseView,
      },
      {
        binding: 2,
        resource: {
          buffer: paramsBuffer,
        },
      },
    ],
  });

  const shader = device.createShaderModule({
    code: renderShaderSource,
  });

  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: shader,
      entryPoint: "vsMain",
      buffers: [
        {
          arrayStride: vertexStride,
          attributes: [
            {
              shaderLocation: 0,
              offset: 0,
              format: "float32x2",
            },
          ],
        },
      ],
    },
    fragment: {
      module: shader,
      entryPoint: "fsMain",
      targets: [{ format }],
    },
    primitive: {
      topology: "triangle-list",
    },
  });

  const renderBindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: noiseView,
      },
    ],
  });

  const eventUv = (event: PointerEvent | WheelEvent) => {
    const rect = canvas.getBoundingClientRect();
    const u = (event.clientX - rect.left) / rect.width;
    const v = 1 - (event.clientY - rect.top) / rect.height;
    return { u, v };
  };

  canvas.style.touchAction = "none";
  let isPanning = false;
  let lastPointerX = 0;
  let lastPointerY = 0;

  canvas.addEventListener("pointerdown", (event) => {
    isPanning = true;
    lastPointerX = event.clientX;
    lastPointerY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!isPanning) {
      return;
    }

    const rect = canvas.getBoundingClientRect();
    const dxUv = (event.clientX - lastPointerX) / rect.width;
    const dyUv = -(event.clientY - lastPointerY) / rect.height;
    lastPointerX = event.clientX;
    lastPointerY = event.clientY;

    const viewRangeX = fullRangeX / zoom;
    const viewRangeY = fullRangeY / zoom;
    centerX -= dxUv * viewRangeX;
    // centerY -= dyUv * viewRangeY;
    clampCenterToData();
    writeComputeParams();
    heatmapDirty = true;
    render();
  });

  canvas.addEventListener("pointerup", (event) => {
    isPanning = false;
    canvas.releasePointerCapture(event.pointerId);
  });

  canvas.addEventListener("pointercancel", (event) => {
    isPanning = false;
    canvas.releasePointerCapture(event.pointerId);
  });

  canvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();

      const { u, v } = eventUv(event);
      const zoomFactor = Math.exp(-event.deltaY * 0.0015);
      const newZoom = clamp(zoom * zoomFactor, 1, 64);

      if (newZoom === zoom) {
        return;
      }

      const currentRangeX = fullRangeX / zoom;
      const currentRangeY = fullRangeY / zoom;
      const worldX = centerX + (u - 0.5) * currentRangeX;
      const worldY = centerY + (v - 0.5) * currentRangeY;

      zoom = newZoom;

      const nextRangeX = fullRangeX / zoom;
      const nextRangeY = fullRangeY / zoom;
      centerX = worldX - (u - 0.5) * nextRangeX;
      centerY = worldY - (v - 0.5) * nextRangeY;
      clampCenterToData();
      writeComputeParams();
      heatmapDirty = true;
      render();
    },
    { passive: false },
  );

  let heatmapDirty = true;

  const render = () => {
    const encoder = device.createCommandEncoder();

    if (heatmapDirty) {
      const computePass = encoder.beginComputePass();
      computePass.setPipeline(computePipeline);
      computePass.setBindGroup(0, computeBindGroup);
      computePass.dispatchWorkgroups(
        Math.ceil(textureWidth / 8),
        Math.ceil(textureHeight / 8),
      );
      computePass.end();
      heatmapDirty = false;
    }

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0.05, g: 0.06, b: 0.09, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, renderBindGroup);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.draw(vertexCount);
    pass.end();

    device.queue.submit([encoder.finish()]);
  };

  render();

  const resizeObserver = new ResizeObserver(() => {
    configureContext();
    render();
  });
  resizeObserver.observe(canvas);

  void device.lost.then((info) => {
    resizeObserver.disconnect();
  });
}

function createCanvas() {
  const existing = document.getElementById("webgpu-triangle-canvas");
  if (existing instanceof HTMLCanvasElement) {
    return existing;
  }

  document.body.style.margin = "0";
  document.body.style.background = "#111";

  const canvas = document.createElement("canvas");
  canvas.id = "webgpu-triangle-canvas";
  canvas.style.position = "fixed";
  canvas.style.inset = "0";
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  canvas.style.display = "block";
  document.body.appendChild(canvas);
  resizeCanvasToDisplaySize(canvas);
  return canvas;
}

function resizeCanvasToDisplaySize(canvas: HTMLCanvasElement) {
  const width = Math.max(
    1,
    Math.floor(canvas.clientWidth * window.devicePixelRatio),
  );
  const height = Math.max(
    1,
    Math.floor(canvas.clientHeight * window.devicePixelRatio),
  );

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
