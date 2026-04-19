import { describe, expect, test } from "vitest";
import {
  createCanvas2DContext,
  WebGPUInitializationError,
  WebGPUNotSupportedError,
} from "../../src";
import { detectWebGPUSupport } from "./support";

describe("createCanvas2DContext", () => {
  test("reports unsupported WebGPU environments with typed errors", async () => {
    const canvas = document.createElement("canvas");
    const support = await detectWebGPUSupport();

    if (support === "none") {
      await expect(createCanvas2DContext(canvas)).rejects.toBeInstanceOf(
        WebGPUNotSupportedError,
      );
      return;
    }

    if (support === "unusable") {
      await expect(createCanvas2DContext(canvas)).rejects.toBeInstanceOf(
        WebGPUInitializationError,
      );
      return;
    }

    const context = await createCanvas2DContext(canvas);
    expect(context.canvas).toBe(canvas);
    context.destroy();
  });
});
