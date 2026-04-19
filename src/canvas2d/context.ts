import {
  initializeWebGPU,
  type InitializedWebGPU,
  type InitializeWebGPUOptions,
} from "../core/webgpu";
import { WebGPUInitializationError } from "../errors";

export interface ClearColor {
  r: number;
  g: number;
  b: number;
  a: number;
}

export interface Canvas2DContextOptions extends InitializeWebGPUOptions {}

export interface RectOptions {
  x: number;
  y: number;
  width: number;
  height: number;
  fill?: string;
  fragmentShader?: string | undefined;
  fragmentShaderEntryPoint?: string | undefined;
}

interface RectGeometry {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface FillColor {
  r: number;
  g: number;
  b: number;
  a: number;
}

const DEFAULT_CLEAR: ClearColor = {
  r: 0,
  g: 0,
  b: 0,
  a: 1,
};

export class Rect {
  x: number;
  y: number;
  width: number;
  height: number;
  fill: string;
  fragmentShader: string | undefined;
  fragmentShaderEntryPoint: string | undefined;

  constructor(options: RectOptions) {
    assertFiniteRect(options, "Rect");

    this.x = options.x;
    this.y = options.y;
    this.width = options.width;
    this.height = options.height;
    this.fill = normalizeHexColor(options.fill ?? "#000000");
    this.fragmentShader = options.fragmentShader;
    this.fragmentShaderEntryPoint = options.fragmentShaderEntryPoint;
  }
}

export class Scene {
  private readonly rectangles: Rect[] = [];

  add(rect: Rect): void {
    this.rectangles.push(rect);
  }

  remove(rect: Rect): boolean {
    const index = this.rectangles.indexOf(rect);

    if (index < 0) {
      return false;
    }

    this.rectangles.splice(index, 1);
    return true;
  }

  clear(): void {
    this.rectangles.length = 0;
  }

  getRectangles(): ReadonlyArray<Rect> {
    return this.rectangles;
  }
}

export class WebGPUCanvas2DContext {
  private destroyed = false;
  private readonly state: InitializedWebGPU;
  private pendingClearValue: GPUColor | null = null;
  private readonly pipelineCache = new Map<string, GPURenderPipeline>();

  private constructor(state: InitializedWebGPU) {
    this.state = state;
    this.pipelineCache.set(
      this.getPipelineCacheKey(undefined, undefined),
      this.createPipeline(
        "triangle-list",
        DEFAULT_FRAGMENT_SHADER_SOURCE,
        "fs_main",
      ),
    );
  }

  static async create(
    canvas: HTMLCanvasElement,
    options: Canvas2DContextOptions = {},
  ): Promise<WebGPUCanvas2DContext> {
    const state = await initializeWebGPU(canvas, options);
    return new WebGPUCanvas2DContext(state);
  }

  get canvas(): HTMLCanvasElement {
    return this.state.canvas;
  }

  clear(color: Partial<ClearColor> = {}): void {
    this.assertActive();

    this.pendingClearValue = {
      r: color.r ?? DEFAULT_CLEAR.r,
      g: color.g ?? DEFAULT_CLEAR.g,
      b: color.b ?? DEFAULT_CLEAR.b,
      a: color.a ?? DEFAULT_CLEAR.a,
    };
  }

  async draw(scene: Scene): Promise<void> {
    this.assertActive();

    const groups = this.groupRectanglesByColor(scene);

    if (this.pendingClearValue == null && groups.length === 0) {
      return this.state.device.queue.onSubmittedWorkDone();
    }

    const transientBuffers: GPUBuffer[] = [];
    const encoder = this.state.device.createCommandEncoder();
    const currentTextureView = this.state.context
      .getCurrentTexture()
      .createView();

    if (this.pendingClearValue != null) {
      const clearPass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: currentTextureView,
            clearValue: this.pendingClearValue,
            loadOp: "clear",
            storeOp: "store",
          },
        ],
      });

      clearPass.end();
      this.pendingClearValue = null;
    }

    if (groups.length > 0) {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: currentTextureView,
            clearValue: DEFAULT_CLEAR,
            loadOp: "load",
            storeOp: "store",
          },
        ],
      });

      for (const group of groups) {
        this.encodeFilledRects(
          pass,
          group.rectangles,
          group.color,
          group.fragmentShader,
          group.fragmentShaderEntryPoint,
          transientBuffers,
        );
      }

      pass.end();
    }

    this.state.device.queue.submit([encoder.finish()]);
    await this.state.device.queue.onSubmittedWorkDone();

    for (const buffer of transientBuffers) {
      buffer.destroy();
    }
  }

  destroy(): void {
    if (this.destroyed) {
      return;
    }

    this.pendingClearValue = null;
    this.destroyed = true;
  }

  private groupRectanglesByColor(scene: Scene): ReadonlyArray<{
    rectangles: ReadonlyArray<RectGeometry>;
    color: FillColor;
    fragmentShader: string | undefined;
    fragmentShaderEntryPoint: string | undefined;
  }> {
    const groups = new Map<
      string,
      {
        rectangles: RectGeometry[];
        colorHex: string;
        fragmentShader: string | undefined;
        fragmentShaderEntryPoint: string | undefined;
      }
    >();

    for (const rect of scene.getRectangles()) {
      assertFiniteRect(rect, "Rect");

      const colorKey = normalizeHexColor(rect.fill);
      const pipelineKey = this.getPipelineCacheKey(
        rect.fragmentShader,
        rect.fragmentShaderEntryPoint,
      );
      const groupKey = `${pipelineKey}::${colorKey}`;
      const group = groups.get(groupKey);
      const geometry: RectGeometry = {
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
      };

      if (group == null) {
        groups.set(groupKey, {
          rectangles: [geometry],
          colorHex: colorKey,
          fragmentShader: rect.fragmentShader,
          fragmentShaderEntryPoint: rect.fragmentShaderEntryPoint,
        });
      } else {
        group.rectangles.push(geometry);
      }
    }

    return Array.from(groups.values(), (group) => ({
      rectangles: group.rectangles,
      color: parseHexColor(group.colorHex),
      fragmentShader: group.fragmentShader,
      fragmentShaderEntryPoint: group.fragmentShaderEntryPoint,
    }));
  }

  private createPipeline(
    topology: GPUPrimitiveTopology,
    fragmentShaderSource: string,
    fragmentEntryPoint: string,
  ): GPURenderPipeline {
    const shaderModule = this.state.device.createShaderModule({
      code: `
        struct VertexInput {
          @location(0) position: vec2<f32>,
          @location(1) color: vec4<f32>,
        };

        struct VertexOutput {
          @builtin(position) position: vec4<f32>,
          @location(0) color: vec4<f32>,
        };

        @vertex
        fn vs_main(input: VertexInput) -> VertexOutput {
          var output: VertexOutput;
          output.position = vec4<f32>(input.position, 0.0, 1.0);
          output.color = input.color;
          return output;
        }

        ${fragmentShaderSource}
      `,
    });

    return this.state.device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: shaderModule,
        entryPoint: "vs_main",
        buffers: [
          {
            arrayStride: Float32Array.BYTES_PER_ELEMENT * 6,
            attributes: [
              {
                shaderLocation: 0,
                offset: 0,
                format: "float32x2",
              },
              {
                shaderLocation: 1,
                offset: Float32Array.BYTES_PER_ELEMENT * 2,
                format: "float32x4",
              },
            ],
          },
        ],
      },
      fragment: {
        module: shaderModule,
        entryPoint: fragmentEntryPoint,
        targets: [{ format: this.state.format }],
      },
      primitive: {
        topology,
      },
    });
  }

  private encodeFilledRects(
    pass: GPURenderPassEncoder,
    rectangles: ReadonlyArray<RectGeometry>,
    color: FillColor,
    fragmentShader: string | undefined,
    fragmentShaderEntryPoint: string | undefined,
    transientBuffers: GPUBuffer[],
  ): void {
    const vertices = this.buildRectVertices(rectangles, color);

    if (vertices.length === 0) {
      return;
    }

    const vertexBuffer = this.state.device.createBuffer({
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    this.state.device.queue.writeBuffer(vertexBuffer, 0, vertices);
    transientBuffers.push(vertexBuffer);

    pass.setPipeline(
      this.getOrCreatePipeline(fragmentShader, fragmentShaderEntryPoint),
    );
    pass.setVertexBuffer(0, vertexBuffer);
    pass.draw(vertices.length / 6);
  }

  private getOrCreatePipeline(
    fragmentShader: string | undefined,
    fragmentShaderEntryPoint: string | undefined,
  ): GPURenderPipeline {
    const key = this.getPipelineCacheKey(
      fragmentShader,
      fragmentShaderEntryPoint,
    );
    const cached = this.pipelineCache.get(key);

    if (cached != null) {
      return cached;
    }

    const source = fragmentShader ?? DEFAULT_FRAGMENT_SHADER_SOURCE;
    const entryPoint =
      fragmentShader == null ? "fs_main" : (fragmentShaderEntryPoint ?? "main");
    const pipeline = this.createPipeline("triangle-list", source, entryPoint);
    this.pipelineCache.set(key, pipeline);
    return pipeline;
  }

  private getPipelineCacheKey(
    fragmentShader: string | undefined,
    fragmentShaderEntryPoint: string | undefined,
  ): string {
    if (fragmentShader == null) {
      return "default::fs_main";
    }

    const entryPoint = fragmentShaderEntryPoint ?? "main";
    return `${entryPoint}::${fragmentShader}`;
  }

  private buildRectVertices(
    rects: ReadonlyArray<RectGeometry>,
    color: FillColor,
  ): Float32Array {
    const canvasWidth = this.state.canvas.width;
    const canvasHeight = this.state.canvas.height;

    if (canvasWidth === 0 || canvasHeight === 0) {
      return new Float32Array();
    }

    const floatsPerVertex = 6;
    const verticesPerRect = 6;
    const data = new Float32Array(
      rects.length * verticesPerRect * floatsPerVertex,
    );

    let offset = 0;
    const writeVertex = (x: number, y: number): void => {
      data[offset] = x;
      data[offset + 1] = y;
      data[offset + 2] = color.r;
      data[offset + 3] = color.g;
      data[offset + 4] = color.b;
      data[offset + 5] = color.a;
      offset += floatsPerVertex;
    };

    for (const rect of rects) {
      const left = Math.min(rect.x, rect.x + rect.width);
      const right = Math.max(rect.x, rect.x + rect.width);
      const top = Math.min(rect.y, rect.y + rect.height);
      const bottom = Math.max(rect.y, rect.y + rect.height);

      const x0 = (left / canvasWidth) * 2 - 1;
      const x1 = (right / canvasWidth) * 2 - 1;
      const y0 = 1 - (top / canvasHeight) * 2;
      const y1 = 1 - (bottom / canvasHeight) * 2;

      writeVertex(x0, y0);
      writeVertex(x1, y0);
      writeVertex(x0, y1);
      writeVertex(x0, y1);
      writeVertex(x1, y0);
      writeVertex(x1, y1);
    }

    return data;
  }

  private assertActive(): void {
    if (this.destroyed) {
      throw new WebGPUInitializationError(
        "Cannot use a destroyed WebGPU canvas context.",
      );
    }
  }
}

function assertFiniteRect(
  rect: { x: number; y: number; width: number; height: number },
  source: string,
): void {
  if (
    !Number.isFinite(rect.x) ||
    !Number.isFinite(rect.y) ||
    !Number.isFinite(rect.width) ||
    !Number.isFinite(rect.height)
  ) {
    throw new TypeError(
      `${source} requires finite numeric coordinates and dimensions.`,
    );
  }
}

function normalizeHexColor(value: string): string {
  if (!HEX_COLOR_REGEX.test(value)) {
    throw new TypeError(
      "Only hex colors are supported. Use #RGB, #RGBA, #RRGGBB, or #RRGGBBAA.",
    );
  }

  return value.toLowerCase();
}

const HEX_COLOR_REGEX =
  /^#(?:[\da-fA-F]{3}|[\da-fA-F]{4}|[\da-fA-F]{6}|[\da-fA-F]{8})$/;

const DEFAULT_FRAGMENT_SHADER_SOURCE = `
  @fragment
  fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
  }
`;

function parseHexColor(value: string): FillColor {
  const hex = value.slice(1);

  if (hex.length === 3 || hex.length === 4) {
    const rDigit = hex.charAt(0);
    const gDigit = hex.charAt(1);
    const bDigit = hex.charAt(2);
    const aDigit = hex.charAt(3);

    const r = Number.parseInt(rDigit + rDigit, 16) / 255;
    const g = Number.parseInt(gDigit + gDigit, 16) / 255;
    const b = Number.parseInt(bDigit + bDigit, 16) / 255;
    const a = hex.length === 4 ? Number.parseInt(aDigit + aDigit, 16) / 255 : 1;

    return {
      r: clamp01(r),
      g: clamp01(g),
      b: clamp01(b),
      a: clamp01(a),
    };
  }

  if (hex.length === 6 || hex.length === 8) {
    const r = Number.parseInt(hex.slice(0, 2), 16) / 255;
    const g = Number.parseInt(hex.slice(2, 4), 16) / 255;
    const b = Number.parseInt(hex.slice(4, 6), 16) / 255;
    const a = hex.length === 8 ? Number.parseInt(hex.slice(6, 8), 16) / 255 : 1;

    return {
      r: clamp01(r),
      g: clamp01(g),
      b: clamp01(b),
      a: clamp01(a),
    };
  }

  return { r: 0, g: 0, b: 0, a: 1 };
}

function clamp01(value: number): number {
  if (Number.isNaN(value)) {
    return 0;
  }

  if (value < 0) {
    return 0;
  }

  if (value > 1) {
    return 1;
  }

  return value;
}

export async function createCanvas2DContext(
  canvas: HTMLCanvasElement,
  options: Canvas2DContextOptions = {},
): Promise<WebGPUCanvas2DContext> {
  return WebGPUCanvas2DContext.create(canvas, options);
}
