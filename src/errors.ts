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
