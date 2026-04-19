export {
  createCanvas2DContext,
  Rect,
  Scene,
  WebGPUCanvas2DContext,
  type Canvas2DContextOptions,
  type ClearColor,
  type RectOptions,
} from "./canvas2d/context";

export {
  initializeWebGPU,
  type InitializedWebGPU,
  type InitializeWebGPUOptions,
} from "./core/webgpu";

export { WebGPUInitializationError, WebGPUNotSupportedError } from "./errors";
