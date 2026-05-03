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

  const shader = device.createShaderModule({
    code: `
      @vertex
      fn vsMain(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4f {
        let positions = array<vec2f, 3>(
          vec2f(0.0, 0.7),
          vec2f(-0.7, -0.7),
          vec2f(0.7, -0.7)
        );

        let p = positions[vertexIndex];
        return vec4f(p, 0.0, 1.0);
      }

      @fragment
      fn fsMain() -> @location(0) vec4f {
        return vec4f(0.97, 0.42, 0.18, 1.0);
      }
    `,
  });

  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: shader,
      entryPoint: "vsMain",
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

  const render = () => {
    const encoder = device.createCommandEncoder();
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
    pass.draw(3);
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
