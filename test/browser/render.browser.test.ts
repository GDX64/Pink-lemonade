import { describe, test } from "vitest";
import { createCanvas2DContext, Rect, Scene, FragmentShader } from "../../src";
import pulseFragmentShader from "./pulse.fragment.wgsl?raw";
import singularityFragmentShader from "./singularity.fragment.wgsl?raw";
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

    const fragmentShader = FragmentShader.new({
      source: pulseFragmentShader,
    });
    fragmentShader.setUniforms({
      speed: 0.1,
      colorShift: 1,
      intensity: 1,
      height: 1,
      turbulence: 1,
      baseColor: [7.0, 2.0, 3.0, 0.0],
      u_resolution_x: 400,
      u_resolution_y: 400,
    });

    context.clear({ r: 1, g: 1, b: 1, a: 1 });

    const rect = new Rect({
      x: 0,
      y: 0,
      width: 400,
      height: 400,
      fragmentShader,
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

  test("singularity shader example", async () => {
    const canvas = document.createElement("canvas");
    canvas.width = 400;
    canvas.height = 400;
    canvas.style.width = "400px";
    canvas.style.height = "400px";
    canvas.style.background = "black";
    document.body.appendChild(canvas);

    const support = await detectWebGPUSupport();

    if (support !== "usable") {
      return;
    }

    const context = await createCanvas2DContext(canvas);
    const fragmentShader = FragmentShader.new({
      source: singularityFragmentShader,
    });

    fragmentShader.setUniforms({
      speed: 1,
      intensity: 1,
      size: 1,
      waveStrength: 1,
      colorShift: 1,
      u_resolution_x: 400,
      u_resolution_y: 400,
    });

    context.clear({ r: 0, g: 0, b: 0, a: 1 });

    const rect = new Rect({
      x: 0,
      y: 0,
      width: 400,
      height: 400,
      fragmentShader,
      fragmentShaderEntryPoint: "main",
    });

    const scene = new Scene();
    scene.add(rect);

    await context.draw(scene);
    context.loop(30_000, async () => {
      await context.draw(scene);
    });
  });
});
