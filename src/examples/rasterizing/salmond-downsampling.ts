import initWasm, { SalmondDownsampler } from "../../../pkg/pink_lemonade_wasm";

/**
 * wasm-backed Salmond clustering reduction (CAF), mirroring `wasmMerge` and
 * `wasmKlMerge`. The implementation lives in `rust/src/salmond.rs`; there is no
 * TypeScript reference for it, unlike Runnalls.
 *
 * `setMergeThreshold` sets the initial clustering threshold T1 --- a squared,
 * mass-weighted Mahalanobis radius, not a pixel radius --- and `setTargetCount`
 * caps the component budget. Note that clustering combines components in groups,
 * so it can only guarantee *at most* `targetCount` kernels: callers comparing at
 * an equal kernel budget must read back the achieved count rather than assume it.
 */
export async function wasmSalmondMerge() {
  await initWasm();
  return new SalmondDownsampler();
}

export type { SalmondDownsampler };
