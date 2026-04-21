import { createCanvas2DContext, FragmentShader, Rect, Scene } from "../../src";
import {
  createNoiseData,
  createSlidingHistogram,
  drawSlidingHistogram,
} from "../../src/chart/chart";
import { describe, test } from "vitest";
import { detectWebGPUSupport } from "./support";

describe("WebGPUCanvas2DContext - Chart Example", async () => {
  test("chart example", async () => {
    const support = await detectWebGPUSupport();

    if (support !== "usable") {
      return;
    }

    const canvas = document.createElement("canvas");
    canvas.style.position = "absolute";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    document.body.appendChild(canvas);
    canvas.width = canvas.getBoundingClientRect().width;
    canvas.height = canvas.getBoundingClientRect().height;

    const offCanvas = new OffscreenCanvas(canvas.width, canvas.height);
    const data = createNoiseData(10_000);
    const binSize = 1;
    const histData = createSlidingHistogram(data, 500, binSize);
    drawSlidingHistogram(histData, offCanvas, binSize);

    const ctx = await createCanvas2DContext(canvas);
    const fragmentShader = FragmentShader.new({
      source: `
        @group(1) @binding(1) var canvasSampler: sampler;
        @group(1) @binding(2) var canvasTexture: texture_2d<f32>;

        @fragment
        fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
            let uv = vec2<f32>(pos.x / ${canvas.width}.0, pos.y / ${canvas.height}.0);
            return textureSample(canvasTexture, canvasSampler, uv);
        }
        `,
    });

    const texture = await ctx.createCanvasTexture(offCanvas);
    fragmentShader.setTexture("canvasTexture", texture);

    const scene = new Scene();
    const rect = new Rect({
      x: 0,
      y: 0,
      width: canvas.width,
      height: canvas.height,
      fragmentShader,
      fragmentShaderEntryPoint: "main",
    });
    scene.add(rect);

    await ctx.draw(scene);
  });
});
