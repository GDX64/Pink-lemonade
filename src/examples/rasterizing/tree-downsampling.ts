import initWasm, { TreeDownsampler } from "../../../pkg/pink_lemonade_wasm";

export async function wasmTreeMerge(): Promise<TreeDownsampler> {
  await initWasm();
  return new TreeDownsampler();
}
