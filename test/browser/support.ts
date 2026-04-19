export type WebGPUSupportState = "none" | "unusable" | "usable";

export async function detectWebGPUSupport(): Promise<WebGPUSupportState> {
  if (!("gpu" in navigator) || navigator.gpu == null) {
    return "none";
  }

  const adapter = await navigator.gpu.requestAdapter();
  return adapter == null ? "unusable" : "usable";
}
