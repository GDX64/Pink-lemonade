import { createNoiseData } from "../chart/chart";

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

  const textureWidth = 128;
  const textureHeight = 128;
  const texelCount = textureWidth * textureHeight;
  const noiseTexture = device.createTexture({
    size: { width: textureWidth, height: textureHeight },
    format: "r32float",
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });

  const noiseView = noiseTexture.createView();

  const data = createNoiseData(1000_000);
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

  const pointCount = f32Data.length / 2;
  const pointBuffer = device.createBuffer({
    size: f32Data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(pointBuffer, 0, f32Data);

  const xRange = Math.max(1e-6, maxX - minX);
  const yRange = Math.max(1e-6, maxY - minY);
  const paramsBytes = 8 * Uint32Array.BYTES_PER_ELEMENT;
  const paramsData = new ArrayBuffer(paramsBytes);
  const paramsView = new DataView(paramsData);
  paramsView.setFloat32(0, minX, true);
  paramsView.setFloat32(4, minY, true);
  paramsView.setFloat32(8, (textureWidth - 1) / xRange, true);
  paramsView.setFloat32(12, (textureHeight - 1) / yRange, true);
  paramsView.setUint32(16, pointCount, true);
  paramsView.setUint32(20, textureWidth, true);
  paramsView.setUint32(24, textureHeight, true);
  paramsView.setUint32(28, 0, true);
  const paramsBuffer = device.createBuffer({
    size: paramsBytes,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, paramsData);

  const computeShader = device.createShaderModule({
    code: `
      struct Params {
        minX: f32,
        minY: f32,
        scaleX: f32,
        scaleY: f32,
        pointCount: u32,
        texWidth: u32,
        texHeight: u32,
        _pad0: u32,
      };

      @group(0) @binding(0)
      var<storage, read> points: array<vec2f>;

      @group(0) @binding(1)
      var heatOut: texture_storage_2d<r32float, write>;

      @group(0) @binding(2)
      var<uniform> params: Params;

      @compute @workgroup_size(8, 8)
      fn buildHeatmap(@builtin(global_invocation_id) gid: vec3u) {
        if (gid.x >= params.texWidth || gid.y >= params.texHeight) {
          return;
        }

        var count = 0u;
        for (var i = 0u; i < params.pointCount; i = i + 1u) {
          let p = points[i];
          let x = u32(clamp((p.x - params.minX) * params.scaleX, 0.0, f32(params.texWidth - 1u)));
          let y = u32(clamp((p.y - params.minY) * params.scaleY, 0.0, f32(params.texHeight - 1u)));
          if (x == gid.x && y == gid.y) {
            count = count + 1u;
          }
        }

        let intensity = clamp(log2(1.0 + f32(count)) / 8.0, 0.0, 1.0);
        textureStore(heatOut, vec2i(gid.xy), vec4f(intensity, 0.0, 0.0, 1.0));
      }
    `,
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
    code: `
      struct VertexOut {
        @builtin(position) position: vec4f,
        @location(0) uv: vec2f,
      };

      @group(0) @binding(0)
      var noiseTex: texture_2d<f32>;

      fn heatmap(tRaw: f32) -> vec3f {
        let t = clamp(tRaw, 0.0, 1.0);
        if (t < 0.33) {
          return mix(vec3f(0.02, 0.02, 0.08), vec3f(0.0, 0.65, 1.0), t / 0.33);
        }
        if (t < 0.66) {
          return mix(vec3f(0.0, 0.65, 1.0), vec3f(1.0, 0.9, 0.0), (t - 0.33) / 0.33);
        }
        return mix(vec3f(1.0, 0.9, 0.0), vec3f(1.0, 0.1, 0.02), (t - 0.66) / 0.34);
      }

      @vertex
      fn vsMain(@location(0) position: vec2f) -> VertexOut {
        var out: VertexOut;
        out.position = vec4f(position, 0.0, 1.0);
        out.uv = position * 0.5 + vec2f(0.5, 0.5);
        return out;
      }

      @fragment
      fn fsMain(@location(0) uv: vec2f) -> @location(0) vec4f {
        let size = vec2f(textureDimensions(noiseTex));
        let texel = vec2i(clamp(uv * size, vec2f(0.0, 0.0), size - vec2f(1.0, 1.0)));
        let n = textureLoad(noiseTex, texel, 0).x;
        return vec4f(heatmap(n), 1.0);
      }
    `,
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

  const render = () => {
    const encoder = device.createCommandEncoder();

    const computePass = encoder.beginComputePass();
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, computeBindGroup);
    computePass.dispatchWorkgroups(
      Math.ceil(textureWidth / 8),
      Math.ceil(textureHeight / 8),
    );
    computePass.end();

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
