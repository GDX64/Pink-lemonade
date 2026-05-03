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

  const noiseSize = 256;
  const noiseTexture = device.createTexture({
    size: { width: noiseSize, height: noiseSize },
    format: "r32float",
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });
  const noiseView = noiseTexture.createView();

  const computeShader = device.createShaderModule({
    code: `
      @group(0) @binding(0)
      var noiseOut: texture_storage_2d<r32float, write>;

      fn fade(t: f32) -> f32 {
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
      }

      fn hash(p: vec2i) -> f32 {
        let n = dot(vec2f(p), vec2f(127.1, 311.7));
        return fract(sin(n) * 43758.5453123);
      }

      fn grad(h: f32) -> vec2f {
        let angle = h * 6.28318530718;
        return vec2f(cos(angle), sin(angle));
      }

      fn perlin(p: vec2f) -> f32 {
        let i0 = vec2i(floor(p));
        let f = fract(p);

        let g00 = grad(hash(i0 + vec2i(0, 0)));
        let g10 = grad(hash(i0 + vec2i(1, 0)));
        let g01 = grad(hash(i0 + vec2i(0, 1)));
        let g11 = grad(hash(i0 + vec2i(1, 1)));

        let n00 = dot(g00, f - vec2f(0.0, 0.0));
        let n10 = dot(g10, f - vec2f(1.0, 0.0));
        let n01 = dot(g01, f - vec2f(0.0, 1.0));
        let n11 = dot(g11, f - vec2f(1.0, 1.0));

        let u = vec2f(fade(f.x), fade(f.y));
        return mix(mix(n00, n10, u.x), mix(n01, n11, u.x), u.y);
      }

      fn fbm(p: vec2f) -> f32 {
        var value = 0.0;
        var amplitude = 0.5;
        var frequency = 1.0;
        for (var octave = 0; octave < 5; octave = octave + 1) {
          value = value + amplitude * perlin(p * frequency);
          frequency = frequency * 2.0;
          amplitude = amplitude * 0.5;
        }
        return value;
      }

      @compute @workgroup_size(8, 8)
      fn csMain(@builtin(global_invocation_id) gid: vec3u) {
        let dims = textureDimensions(noiseOut);
        if (gid.x >= dims.x || gid.y >= dims.y) {
          return;
        }

        let uv = (vec2f(gid.xy) + vec2f(0.5, 0.5)) / vec2f(dims);
        let n = fbm(uv * 8.0);
        let grayscale = clamp(n * 0.5 + 0.5, 0.0, 1.0);

        textureStore(noiseOut, vec2i(gid.xy), vec4f(grayscale, 0.0, 0.0, 1.0));
      }
    `,
  });

  const computePipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: computeShader,
      entryPoint: "csMain",
    },
  });

  const computeBindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: noiseView,
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
        return vec4f(vec3f(n), 1.0);
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
    computePass.dispatchWorkgroups(noiseSize / 8, noiseSize / 8);
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
