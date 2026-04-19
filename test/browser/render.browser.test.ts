import { describe, test } from "vitest";
import { createCanvas2DContext, Rect, Scene } from "../../src";
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

    const fragmentShader = `
    @fragment
    fn main() -> @location(0) vec4<f32> {
      return vec4<f32>(1.0, 1.0, 0.0, 1.0);
    }
    `;

    const rect = new Rect({
      x: 50,
      y: 50,
      width: 100,
      height: 100,
      fragmentShader,
      fragmentShaderEntryPoint: "main",
    });

    const scene = new Scene();
    scene.add(rect);

    await context.draw(scene);

    context.destroy();
  });
});
