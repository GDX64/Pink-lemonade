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

type PendingCommand = {
  kind: "clear";
  clearValue: GPUColor;
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

  private constructor(state: InitializedWebGPU) {
    this.state = state;
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

    for (const command of queuedCommands) {
      if (command.kind === "clear") {
        const view = this.state.context.getCurrentTexture().createView();
        const pass = encoder.beginRenderPass({
          colorAttachments: [
            {
              view,
              clearValue: command.clearValue,
              loadOp: "clear",
              storeOp: "store",
            },
          ],
        });

        pass.end();
      }
    }

    this.state.device.queue.submit([encoder.finish()]);
    await this.state.device.queue.onSubmittedWorkDone();
  }

  destroy(): void {
    if (this.destroyed) {
      return;
    }

    this.commandQueue.length = 0;
    this.destroyed = true;
  }

  private assertActive(): void {
    if (this.destroyed) {
      throw new WebGPUInitializationError(
        "Cannot use a destroyed WebGPU canvas context.",
      );
    }
  }
}

export async function createCanvas2DContext(
  canvas: HTMLCanvasElement,
  options: Canvas2DContextOptions = {},
): Promise<WebGPUCanvas2DContext> {
  return WebGPUCanvas2DContext.create(canvas, options);
}
