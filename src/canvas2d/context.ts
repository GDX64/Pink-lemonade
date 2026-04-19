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

interface RectPathSegment {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface Point {
  x: number;
  y: number;
}

interface LineSegment {
  from: Point;
  to: Point;
}

interface FillColor {
  r: number;
  g: number;
  b: number;
  a: number;
}

type PendingCommand =
  | {
      kind: "clear";
      clearValue: GPUColor;
    }
  | {
      kind: "fill-rects";
      rects: ReadonlyArray<RectPathSegment>;
      color: FillColor;
    }
  | {
      kind: "stroke-lines";
      lines: ReadonlyArray<LineSegment>;
      color: FillColor;
    };

const DEFAULT_CLEAR: ClearColor = {
  r: 0,
  g: 0,
  b: 0,
  a: 1,
};

export class WebGPUCanvas2DContext {
  private destroyed = false;
  private readonly state: InitializedWebGPU;
  private readonly commandQueue: PendingCommand[] = [];
  private readonly pathRectangles: RectPathSegment[] = [];
  private readonly pathLines: LineSegment[] = [];
  private currentPathCursor: Point | null = null;
  private readonly fillPipeline: GPURenderPipeline;
  private readonly strokePipeline: GPURenderPipeline;
  private readonly colorResolver: CanvasRenderingContext2D;
  private currentFillStyle = "#000000";
  private currentStrokeStyle = "#000000";

  private constructor(state: InitializedWebGPU) {
    this.state = state;
    this.fillPipeline = this.createColorPipeline("triangle-list");
    this.strokePipeline = this.createColorPipeline("line-list");

    const colorResolverCanvas = document.createElement("canvas");
    colorResolverCanvas.width = 1;
    colorResolverCanvas.height = 1;

    const colorResolver = colorResolverCanvas.getContext("2d");

    if (colorResolver == null) {
      throw new WebGPUInitializationError(
        "Unable to create an internal color parser for fill operations.",
      );
    }

    colorResolver.fillStyle = this.currentFillStyle;
    this.colorResolver = colorResolver;
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

  get fillStyle(): string {
    return this.currentFillStyle;
  }

  set fillStyle(value: string) {
    this.assertActive();
    this.colorResolver.fillStyle = this.currentFillStyle;
    this.colorResolver.fillStyle = value;

    if (typeof this.colorResolver.fillStyle === "string") {
      this.currentFillStyle = this.colorResolver.fillStyle;
    }
  }

  get strokeStyle(): string {
    return this.currentStrokeStyle;
  }

  set strokeStyle(value: string) {
    this.assertActive();
    this.colorResolver.fillStyle = this.currentStrokeStyle;
    this.colorResolver.fillStyle = value;

    if (typeof this.colorResolver.fillStyle === "string") {
      this.currentStrokeStyle = this.colorResolver.fillStyle;
    }
  }

  clear(color: Partial<ClearColor> = {}): void {
    this.assertActive();

    const clearValue: GPUColor = {
      r: color.r ?? DEFAULT_CLEAR.r,
      g: color.g ?? DEFAULT_CLEAR.g,
      b: color.b ?? DEFAULT_CLEAR.b,
      a: color.a ?? DEFAULT_CLEAR.a,
    };

    this.commandQueue.push({
      kind: "clear",
      clearValue,
    });
  }

  rect(x: number, y: number, width: number, height: number): void {
    this.assertActive();

    if (
      !Number.isFinite(x) ||
      !Number.isFinite(y) ||
      !Number.isFinite(width) ||
      !Number.isFinite(height)
    ) {
      throw new TypeError("rect() requires finite numeric arguments.");
    }

    this.pathRectangles.push({ x, y, width, height });
  }

  moveTo(x: number, y: number): void {
    this.assertActive();

    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      throw new TypeError("moveTo() requires finite numeric arguments.");
    }

    this.currentPathCursor = { x, y };
  }

  lineTo(x: number, y: number): void {
    this.assertActive();

    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      throw new TypeError("lineTo() requires finite numeric arguments.");
    }

    const nextPoint: Point = { x, y };

    if (this.currentPathCursor == null) {
      this.currentPathCursor = nextPoint;
      return;
    }

    this.pathLines.push({
      from: { ...this.currentPathCursor },
      to: nextPoint,
    });
    this.currentPathCursor = nextPoint;
  }

  fill(): void {
    this.assertActive();

    if (this.pathRectangles.length === 0) {
      return;
    }

    this.commandQueue.push({
      kind: "fill-rects",
      rects: this.pathRectangles.map((segment) => ({ ...segment })),
      color: this.parseCssColor(this.currentFillStyle),
    });
  }

  stroke(): void {
    this.assertActive();

    if (this.pathLines.length === 0) {
      return;
    }

    this.commandQueue.push({
      kind: "stroke-lines",
      lines: this.pathLines.map((segment) => ({
        from: { ...segment.from },
        to: { ...segment.to },
      })),
      color: this.parseCssColor(this.currentStrokeStyle),
    });
  }

  async flush(): Promise<void> {
    this.assertActive();

    if (this.commandQueue.length === 0) {
      return this.state.device.queue.onSubmittedWorkDone();
    }

    const encoder = this.state.device.createCommandEncoder();
    const queuedCommands = this.commandQueue.splice(
      0,
      this.commandQueue.length,
    );
    const currentTextureView = this.state.context
      .getCurrentTexture()
      .createView();
    const transientBuffers: GPUBuffer[] = [];

    for (const command of queuedCommands) {
      if (command.kind === "clear") {
        const pass = encoder.beginRenderPass({
          colorAttachments: [
            {
              view: currentTextureView,
              clearValue: command.clearValue,
              loadOp: "clear",
              storeOp: "store",
            },
          ],
        });

        pass.end();
        continue;
      }

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

      if (command.kind === "fill-rects") {
        this.encodeFilledRects(pass, command, transientBuffers);
      }

      if (command.kind === "stroke-lines") {
        this.encodeStrokedLines(pass, command, transientBuffers);
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

    this.commandQueue.length = 0;
    this.pathRectangles.length = 0;
    this.pathLines.length = 0;
    this.currentPathCursor = null;
    this.destroyed = true;
  }

  private createColorPipeline(
    topology: GPUPrimitiveTopology,
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

        @fragment
        fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
          return input.color;
        }
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
        entryPoint: "fs_main",
        targets: [{ format: this.state.format }],
      },
      primitive: {
        topology,
      },
    });
  }

  private encodeFilledRects(
    pass: GPURenderPassEncoder,
    command: Extract<PendingCommand, { kind: "fill-rects" }>,
    transientBuffers: GPUBuffer[],
  ): void {
    const vertices = this.buildRectVertices(command.rects, command.color);

    if (vertices.length === 0) {
      return;
    }

    const vertexBuffer = this.state.device.createBuffer({
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    this.state.device.queue.writeBuffer(vertexBuffer, 0, vertices);
    transientBuffers.push(vertexBuffer);

    pass.setPipeline(this.fillPipeline);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.draw(vertices.length / 6);
  }

  private encodeStrokedLines(
    pass: GPURenderPassEncoder,
    command: Extract<PendingCommand, { kind: "stroke-lines" }>,
    transientBuffers: GPUBuffer[],
  ): void {
    const vertices = this.buildLineVertices(command.lines, command.color);

    if (vertices.length === 0) {
      return;
    }

    const vertexBuffer = this.state.device.createBuffer({
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    this.state.device.queue.writeBuffer(vertexBuffer, 0, vertices);
    transientBuffers.push(vertexBuffer);

    pass.setPipeline(this.strokePipeline);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.draw(vertices.length / 6);
  }

  private buildRectVertices(
    rects: ReadonlyArray<RectPathSegment>,
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

  private buildLineVertices(
    lines: ReadonlyArray<LineSegment>,
    color: FillColor,
  ): Float32Array {
    const canvasWidth = this.state.canvas.width;
    const canvasHeight = this.state.canvas.height;

    if (canvasWidth === 0 || canvasHeight === 0) {
      return new Float32Array();
    }

    const floatsPerVertex = 6;
    const verticesPerLine = 2;
    const data = new Float32Array(
      lines.length * verticesPerLine * floatsPerVertex,
    );

    let offset = 0;
    const writeVertex = (x: number, y: number): void => {
      const normalizedX = (x / canvasWidth) * 2 - 1;
      const normalizedY = 1 - (y / canvasHeight) * 2;

      data[offset] = normalizedX;
      data[offset + 1] = normalizedY;
      data[offset + 2] = color.r;
      data[offset + 3] = color.g;
      data[offset + 4] = color.b;
      data[offset + 5] = color.a;
      offset += floatsPerVertex;
    };

    for (const line of lines) {
      writeVertex(line.from.x, line.from.y);
      writeVertex(line.to.x, line.to.y);
    }

    return data;
  }

  private parseCssColor(value: string): FillColor {
    this.colorResolver.fillStyle = "#000000";
    this.colorResolver.fillStyle = value;
    const normalized = this.colorResolver.fillStyle;

    if (typeof normalized !== "string") {
      return { r: 0, g: 0, b: 0, a: 1 };
    }

    return parseNormalizedColor(normalized);
  }

  private assertActive(): void {
    if (this.destroyed) {
      throw new WebGPUInitializationError(
        "Cannot use a destroyed WebGPU canvas context.",
      );
    }
  }
}

function parseNormalizedColor(value: string): FillColor {
  if (value.startsWith("#")) {
    return parseHexColor(value);
  }

  const rgbaMatch = value.match(
    /^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+)\s*)?\)$/,
  );

  if (rgbaMatch == null) {
    return { r: 0, g: 0, b: 0, a: 1 };
  }

  const [, rText, gText, bText, aText] = rgbaMatch;
  const r = Number(rText) / 255;
  const g = Number(gText) / 255;
  const b = Number(bText) / 255;
  const a = aText == null ? 1 : Number(aText);

  return {
    r: clamp01(r),
    g: clamp01(g),
    b: clamp01(b),
    a: clamp01(a),
  };
}

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
