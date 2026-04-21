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
  fragmentShader?: RectFragmentShader | undefined;
  fragmentShaderEntryPoint?: string | undefined;
}

export interface FragmentShaderOptions {
  source: string;
  overrides?: Record<string, number> | undefined;
  uniforms?: FragmentShaderUniforms | undefined;
}

export interface CanvasTexture {
  texture: GPUTexture;
  view: GPUTextureView;
  sampler: GPUSampler;
  width: number;
  height: number;
}

export type FragmentShaderUniformValue =
  | number
  | readonly [number, number, number, number];

export type FragmentShaderUniforms = Record<string, FragmentShaderUniformValue>;

export type RectFragmentShader = string | FragmentShader;

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

export class FragmentShader {
  private static nextId = 1;

  readonly id: number;
  readonly source: string;
  readonly overrides: Record<string, number> | undefined;
  private uniforms: FragmentShaderUniforms | undefined;
  private textures: Record<string, CanvasTexture> | undefined;

  private constructor(options: FragmentShaderOptions) {
    this.id = FragmentShader.nextId;
    FragmentShader.nextId += 1;
    this.source = options.source;
    this.overrides = normalizeOverrides(options.overrides);
    this.uniforms = normalizeUniforms(options.uniforms);
  }

  static new(options: FragmentShaderOptions): FragmentShader {
    return new FragmentShader(options);
  }

  setUniforms(uniforms: FragmentShaderUniforms): void {
    this.uniforms = normalizeUniforms(uniforms);
  }

  getUniforms(): FragmentShaderUniforms | undefined {
    return this.uniforms;
  }

  setTexture(name: string, texture: CanvasTexture): void {
    if (name.trim().length === 0) {
      throw new TypeError("FragmentShader texture name must be non-empty.");
    }

    if (this.textures == null) {
      this.textures = {};
    }

    this.textures[name] = texture;
  }

  getTextures(): Readonly<Record<string, CanvasTexture>> | undefined {
    return this.textures;
  }
}

export class Rect {
  x: number;
  y: number;
  width: number;
  height: number;
  fill: string;
  fragmentShader: RectFragmentShader | undefined;
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
  private readonly globalUniformBuffer: GPUBuffer;
  private readonly globalUniformData = new Float32Array(8);
  private readonly globalBindGroupLayout: GPUBindGroupLayout;
  private readonly globalBindGroup: GPUBindGroup;
  private readonly customUniformBindGroupLayout: GPUBindGroupLayout;
  private readonly fallbackCanvasTexture: CanvasTexture;
  private readonly ownedCanvasTextures = new Set<GPUTexture>();
  private readonly customUniformStates = new Map<
    FragmentShader,
    {
      buffer: GPUBuffer;
      bindGroup: GPUBindGroup;
      floatCount: number;
      texture: CanvasTexture;
    }
  >();

  private constructor(state: InitializedWebGPU) {
    this.state = state;
    this.globalBindGroupLayout = this.state.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: {
            type: "uniform",
          },
        },
      ],
    });
    this.globalUniformBuffer = this.state.device.createBuffer({
      size: alignUniformBufferByteSize(
        this.globalUniformData.byteLength,
        this.state.device.limits.minUniformBufferOffsetAlignment,
      ),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.globalBindGroup = this.state.device.createBindGroup({
      layout: this.globalBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.globalUniformBuffer,
          },
        },
      ],
    });
    this.customUniformBindGroupLayout = this.state.device.createBindGroupLayout(
      {
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.FRAGMENT,
            buffer: {
              type: "uniform",
            },
          },
          {
            binding: 1,
            visibility: GPUShaderStage.FRAGMENT,
            sampler: {
              type: "filtering",
            },
          },
          {
            binding: 2,
            visibility: GPUShaderStage.FRAGMENT,
            texture: {
              sampleType: "float",
              viewDimension: "2d",
              multisampled: false,
            },
          },
        ],
      },
    );
    this.fallbackCanvasTexture = this.createFallbackCanvasTexture();
    this.pipelineCache.set(
      this.getPipelineCacheKey("default", "fs_main"),
      this.createPipeline(
        "triangle-list",
        DEFAULT_FRAGMENT_SHADER_SOURCE,
        "fs_main",
        undefined,
        false,
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

  async createCanvasTexture(
    canvas: HTMLCanvasElement | OffscreenCanvas,
  ): Promise<CanvasTexture> {
    this.assertActive();

    if (canvas.width === 0 || canvas.height === 0) {
      throw new TypeError(
        "createCanvasTexture requires a canvas with non-zero width and height.",
      );
    }

    const texture = this.state.device.createTexture({
      size: {
        width: canvas.width,
        height: canvas.height,
        depthOrArrayLayers: 1,
      },
      format: "rgba8unorm",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.state.device.queue.copyExternalImageToTexture(
      { source: canvas },
      { texture },
      {
        width: canvas.width,
        height: canvas.height,
        depthOrArrayLayers: 1,
      },
    );

    const sampler = this.state.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      mipmapFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });

    const canvasTexture: CanvasTexture = {
      texture,
      view: texture.createView(),
      sampler,
      width: canvas.width,
      height: canvas.height,
    };

    this.ownedCanvasTextures.add(texture);
    return canvasTexture;
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

  async loop(time: number, cb: () => Promise<void>): Promise<void> {
    const start = Date.now();

    while (true) {
      const now = Date.now();
      const elapsed = now - start;
      if (elapsed >= time) {
        break;
      }
      await this.raf();
      await cb();
    }
  }

  raf(): Promise<void> {
    return new Promise((resolve) => {
      requestAnimationFrame(() => resolve());
    });
  }

  async draw(scene: Scene): Promise<void> {
    this.assertActive();
    this.updateGlobalTimestamp(performance.now(), [
      this.canvas.width,
      this.canvas.height,
    ]);

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
    this.globalUniformBuffer.destroy();
    this.fallbackCanvasTexture.texture.destroy();
    for (const texture of this.ownedCanvasTextures) {
      texture.destroy();
    }
    this.ownedCanvasTextures.clear();
    for (const state of this.customUniformStates.values()) {
      state.buffer.destroy();
    }
    this.customUniformStates.clear();
    this.destroyed = true;
  }

  private groupRectanglesByColor(scene: Scene): ReadonlyArray<{
    rectangles: ReadonlyArray<RectGeometry>;
    color: FillColor;
    fragmentShader: RectFragmentShader | undefined;
    fragmentShaderEntryPoint: string | undefined;
  }> {
    const groups = new Map<
      string,
      {
        rectangles: RectGeometry[];
        colorHex: string;
        fragmentShader: RectFragmentShader | undefined;
        fragmentShaderEntryPoint: string | undefined;
      }
    >();

    for (const rect of scene.getRectangles()) {
      assertFiniteRect(rect, "Rect");

      const colorKey = normalizeHexColor(rect.fill);
      const resolvedShader = resolveRectFragmentShader(rect.fragmentShader);
      const entryPoint =
        rect.fragmentShader == null
          ? "fs_main"
          : (rect.fragmentShaderEntryPoint ?? "main");
      const pipelineKey = this.getPipelineCacheKey(
        resolvedShader.cacheKey,
        entryPoint,
      );
      const groupKey = `${pipelineKey}::${resolvedShader.bindingKey}::${colorKey}`;
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
    fragmentConstants: Record<string, number> | undefined,
    useCustomUniformGroup: boolean,
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

        ${FRAGMENT_GLOBALS_SHADER_SOURCE}

        ${fragmentShaderSource}
      `,
    });

    const pipelineLayout = this.state.device.createPipelineLayout({
      bindGroupLayouts: useCustomUniformGroup
        ? [this.globalBindGroupLayout, this.customUniformBindGroupLayout]
        : [this.globalBindGroupLayout],
    });

    const fragmentStage: GPUFragmentState = {
      module: shaderModule,
      entryPoint: fragmentEntryPoint,
      targets: [{ format: this.state.format }],
    };

    if (fragmentConstants != null) {
      fragmentStage.constants = fragmentConstants;
    }

    return this.state.device.createRenderPipeline({
      layout: pipelineLayout,
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
      fragment: fragmentStage,
      primitive: {
        topology,
      },
    });
  }

  private encodeFilledRects(
    pass: GPURenderPassEncoder,
    rectangles: ReadonlyArray<RectGeometry>,
    color: FillColor,
    fragmentShader: RectFragmentShader | undefined,
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
    pass.setBindGroup(0, this.globalBindGroup);
    if (fragmentShader instanceof FragmentShader) {
      pass.setBindGroup(
        1,
        this.getOrCreateCustomUniformBindGroup(fragmentShader),
      );
    }
    pass.setVertexBuffer(0, vertexBuffer);
    pass.draw(vertices.length / 6);
  }

  private getOrCreateCustomUniformBindGroup(
    fragmentShader: FragmentShader,
  ): GPUBindGroup {
    const uniforms = fragmentShader.getUniforms();
    const data = packCustomUniforms(uniforms);
    const floatCount = data.length;
    const texture = this.resolveTextureBinding(fragmentShader);

    const state = this.customUniformStates.get(fragmentShader);

    if (
      state != null &&
      state.floatCount === floatCount &&
      state.texture === texture
    ) {
      this.state.device.queue.writeBuffer(state.buffer, 0, data);
      return state.bindGroup;
    }

    if (state != null) {
      state.buffer.destroy();
    }

    const buffer = this.state.device.createBuffer({
      size: alignUniformBufferByteSize(
        data.byteLength,
        this.state.device.limits.minUniformBufferOffsetAlignment,
      ),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const bindGroup = this.state.device.createBindGroup({
      layout: this.customUniformBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer,
          },
        },
        {
          binding: 1,
          resource: texture.sampler,
        },
        {
          binding: 2,
          resource: texture.view,
        },
      ],
    });

    this.state.device.queue.writeBuffer(buffer, 0, data);
    this.customUniformStates.set(fragmentShader, {
      buffer,
      bindGroup,
      floatCount,
      texture,
    });

    return bindGroup;
  }

  private updateGlobalTimestamp(
    timestampMs: number,
    resolution: [number, number],
  ): void {
    this.globalUniformData[0] = timestampMs;
    // Keep std140-like alignment for WGSL uniforms: vec2 starts at 8-byte offset.
    this.globalUniformData[1] = 0;
    this.globalUniformData[2] = resolution[0];
    this.globalUniformData[3] = resolution[1];
    this.globalUniformData[4] = 0;
    this.globalUniformData[5] = 0;
    this.globalUniformData[6] = 0;
    this.globalUniformData[7] = 0;
    this.state.device.queue.writeBuffer(
      this.globalUniformBuffer,
      0,
      this.globalUniformData,
    );
  }

  private getOrCreatePipeline(
    fragmentShader: RectFragmentShader | undefined,
    fragmentShaderEntryPoint: string | undefined,
  ): GPURenderPipeline {
    const resolvedShader = resolveRectFragmentShader(fragmentShader);
    const entryPoint =
      fragmentShader == null ? "fs_main" : (fragmentShaderEntryPoint ?? "main");
    const key = this.getPipelineCacheKey(resolvedShader.cacheKey, entryPoint);
    const cached = this.pipelineCache.get(key);

    if (cached != null) {
      return cached;
    }

    const pipeline = this.createPipeline(
      "triangle-list",
      resolvedShader.source,
      entryPoint,
      resolvedShader.constants,
      resolvedShader.usesCustomUniformGroup,
    );
    this.pipelineCache.set(key, pipeline);
    return pipeline;
  }

  private getPipelineCacheKey(
    shaderKey: string,
    fragmentShaderEntryPoint: string,
  ): string {
    return `${fragmentShaderEntryPoint}::${shaderKey}`;
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

  private resolveTextureBinding(fragmentShader: FragmentShader): CanvasTexture {
    const textures = fragmentShader.getTextures();

    if (textures == null) {
      return this.fallbackCanvasTexture;
    }

    const names = Object.keys(textures).sort();

    if (names.length === 0) {
      return this.fallbackCanvasTexture;
    }

    const firstName = names[0]!;
    return textures[firstName] ?? this.fallbackCanvasTexture;
  }

  private createFallbackCanvasTexture(): CanvasTexture {
    const texture = this.state.device.createTexture({
      size: {
        width: 1,
        height: 1,
        depthOrArrayLayers: 1,
      },
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    this.state.device.queue.writeTexture(
      {
        texture,
      },
      new Uint8Array([255, 255, 255, 255]),
      {
        bytesPerRow: 4,
        rowsPerImage: 1,
      },
      {
        width: 1,
        height: 1,
        depthOrArrayLayers: 1,
      },
    );

    const sampler = this.state.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      mipmapFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });

    return {
      texture,
      view: texture.createView(),
      sampler,
      width: 1,
      height: 1,
    };
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

const FRAGMENT_GLOBALS_SHADER_SOURCE = `
  struct GlobalUniforms {
    timestamp: f32,
    resolution: vec2<f32>,
    _pad0: vec3<f32>,
  };

  @group(0) @binding(0)
  var<uniform> globalUniforms: GlobalUniforms;
`;

function resolveRectFragmentShader(
  fragmentShader: RectFragmentShader | undefined,
): {
  source: string;
  constants: Record<string, number> | undefined;
  cacheKey: string;
  bindingKey: string;
  usesCustomUniformGroup: boolean;
} {
  if (fragmentShader == null) {
    return {
      source: DEFAULT_FRAGMENT_SHADER_SOURCE,
      constants: undefined,
      cacheKey: "default",
      bindingKey: "none",
      usesCustomUniformGroup: false,
    };
  }

  if (typeof fragmentShader === "string") {
    return {
      source: fragmentShader,
      constants: undefined,
      cacheKey: `source:${fragmentShader}`,
      bindingKey: "none",
      usesCustomUniformGroup: false,
    };
  }

  if (fragmentShader.overrides == null) {
    return {
      source: fragmentShader.source,
      constants: undefined,
      cacheKey: `source:${fragmentShader.source}`,
      bindingKey: `shader:${fragmentShader.id}`,
      usesCustomUniformGroup: true,
    };
  }

  return {
    source: fragmentShader.source,
    constants: fragmentShader.overrides,
    cacheKey: `source:${fragmentShader.source}::overrides:${serializeOverrides(fragmentShader.overrides)}`,
    bindingKey: `shader:${fragmentShader.id}`,
    usesCustomUniformGroup: true,
  };
}

function normalizeOverrides(
  overrides: Record<string, number> | undefined,
): Record<string, number> | undefined {
  if (overrides == null) {
    return undefined;
  }

  const normalized: Record<string, number> = {};

  for (const [key, value] of Object.entries(overrides)) {
    if (!Number.isFinite(value)) {
      throw new TypeError(
        `FragmentShader override '${key}' must be a finite number.`,
      );
    }

    normalized[key] = value;
  }

  return normalized;
}

function normalizeUniforms(
  uniforms: FragmentShaderUniforms | undefined,
): FragmentShaderUniforms | undefined {
  if (uniforms == null) {
    return undefined;
  }

  const normalized: FragmentShaderUniforms = {};

  for (const [key, value] of Object.entries(uniforms)) {
    if (typeof value === "number") {
      if (!Number.isFinite(value)) {
        throw new TypeError(
          `FragmentShader uniform '${key}' must be a finite number.`,
        );
      }
      normalized[key] = value;
      continue;
    }

    if (value.length !== 4) {
      throw new TypeError(
        `FragmentShader uniform '${key}' vec4 value must have exactly 4 items.`,
      );
    }

    for (const component of value) {
      if (!Number.isFinite(component)) {
        throw new TypeError(
          `FragmentShader uniform '${key}' vec4 components must be finite numbers.`,
        );
      }
    }

    normalized[key] = [value[0], value[1], value[2], value[3]] as const;
  }

  return normalized;
}

function packCustomUniforms(
  uniforms: FragmentShaderUniforms | undefined,
): Float32Array {
  if (uniforms == null) {
    return new Float32Array(16);
  }

  const packed: number[] = [];

  const padToVec4Boundary = (): void => {
    while (packed.length % 4 !== 0) {
      packed.push(0);
    }
  };

  for (const value of Object.values(uniforms)) {
    if (typeof value === "number") {
      packed.push(value);
      continue;
    }

    padToVec4Boundary();
    packed.push(value[0], value[1], value[2], value[3]);
  }

  padToVec4Boundary();

  if (packed.length < 16) {
    while (packed.length < 16) {
      packed.push(0);
    }
  }

  return new Float32Array(packed);
}

function alignUniformBufferByteSize(
  byteSize: number,
  minUniformBufferOffsetAlignment: number,
): number {
  const alignment = Math.max(16, minUniformBufferOffsetAlignment);
  const remainder = byteSize % alignment;

  if (remainder === 0) {
    return byteSize;
  }

  return byteSize + alignment - remainder;
}

function serializeOverrides(overrides: Record<string, number>): string {
  const keys = Object.keys(overrides).sort();
  return keys.map((key) => `${key}=${overrides[key]}`).join(";");
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

export interface InitializeWebGPUOptions {
  alphaMode?: GPUCanvasAlphaMode;
  format?: GPUTextureFormat;
  powerPreference?: GPUPowerPreference;
}

export interface InitializedWebGPU {
  adapter: GPUAdapter;
  device: GPUDevice;
  canvas: HTMLCanvasElement;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
}

export async function initializeWebGPU(
  canvas: HTMLCanvasElement,
  options: InitializeWebGPUOptions = {},
): Promise<InitializedWebGPU> {
  if (!("gpu" in navigator) || navigator.gpu == null) {
    throw new WebGPUNotSupportedError();
  }

  const context = canvas.getContext("webgpu") as GPUCanvasContext | null;

  if (context == null) {
    throw new WebGPUInitializationError(
      "Unable to acquire a WebGPU canvas context.",
    );
  }

  try {
    const adapterRequest: GPURequestAdapterOptions = {};

    if (options.powerPreference != null) {
      adapterRequest.powerPreference = options.powerPreference;
    }

    const adapter = await navigator.gpu.requestAdapter(adapterRequest);

    if (adapter == null) {
      throw new WebGPUInitializationError(
        "No compatible WebGPU adapter was found.",
      );
    }

    const device = await adapter.requestDevice();
    const format = options.format ?? navigator.gpu.getPreferredCanvasFormat();

    context.configure({
      device,
      format,
      alphaMode: options.alphaMode ?? "premultiplied",
    });

    return {
      adapter,
      device,
      canvas,
      context,
      format,
    };
  } catch (error) {
    if (error instanceof WebGPUInitializationError) {
      throw error;
    }

    throw new WebGPUInitializationError(
      "Failed to initialize WebGPU for canvas.",
      error,
    );
  }
}

export class WebGPUNotSupportedError extends Error {
  constructor(message = "WebGPU is not supported in this browser or runtime.") {
    super(message);
    this.name = "WebGPUNotSupportedError";
  }
}

export class WebGPUInitializationError extends Error {
  readonly cause?: unknown;

  constructor(message: string, cause?: unknown) {
    super(message);
    this.name = "WebGPUInitializationError";
    this.cause = cause;
  }
}
