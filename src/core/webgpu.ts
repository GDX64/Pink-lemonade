import { WebGPUInitializationError, WebGPUNotSupportedError } from "../errors";

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
