import { describe, expect, test } from "vitest";
import { createCanvas2DContext, WebGPUInitializationError } from "../../src";
import { detectWebGPUSupport } from "./support";

describe("WebGPUCanvas2DContext", () => {
  test("clear executes on supported systems and destroy guards subsequent calls", async () => {
    const canvas = document.createElement("canvas");
    canvas.width = 64;
    canvas.height = 64;

    const support = await detectWebGPUSupport();

    if (support !== "usable") {
      return;
    }

    const context = await createCanvas2DContext(canvas);
    expect(() => context.clear({ r: 1, g: 0.5, b: 0, a: 1 })).not.toThrow();

    context.destroy();
    expect(() => context.clear()).toThrowError(WebGPUInitializationError);
  });
});
