import { describe, test } from "vitest";
import { createCanvas2DContext, Rect, Scene } from "../../src";
import pulseFragmentShader from "./pulse.fragment.wgsl?raw";
import { detectWebGPUSupport } from "./support";

describe("WebGPUCanvas2DContext", () => {
  test("basic commands", async () => {
    const canvas = document.createElement("canvas");
    canvas.width = 400;
    canvas.height = 400;
    canvas.style.width = "400px";
    canvas.style.height = "400px";
    document.body.appendChild(canvas);

    const support = await detectWebGPUSupport();

    if (support !== "usable") {
      return;
    }

    const context = await createCanvas2DContext(canvas);

    context.clear({ r: 1, g: 1, b: 1, a: 1 });

    const rect = new Rect({
      x: 0,
      y: 0,
      width: 400,
      height: 400,
      fragmentShader: pulseFragmentShader,
      fragmentShaderEntryPoint: "main",
    });

    const scene = new Scene();
    scene.add(rect);

    await context.draw(scene);
    context.loop(30_000, async () => {
      await context.draw(scene);
    });

    // context.destroy();
  });
});
