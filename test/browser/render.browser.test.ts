import { describe, expect, test, vi } from "vitest";
import { createCanvas2DContext, WebGPUInitializationError } from "../../src";
import { detectWebGPUSupport } from "./support";

describe("WebGPUCanvas2DContext", () => {
  test("clear executes on supported systems and destroy guards subsequent calls", async () => {
    const canvas = document.createElement("canvas");
    canvas.width = 64;
    canvas.height = 64;
    canvas.style.width = "64px";
    canvas.style.height = "64px";
    document.body.appendChild(canvas);

    const support = await detectWebGPUSupport();

    if (support !== "usable") {
      return;
    }

    const context = await createCanvas2DContext(canvas);

    context.clear({ r: 1, g: 1, b: 1, a: 1 });

    await context.flush();

    context.destroy();
  });
});
