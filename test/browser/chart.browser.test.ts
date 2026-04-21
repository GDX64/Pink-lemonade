import { createCanvas2DContext, FragmentShader, Rect, Scene } from "../../src";
import {
  drawChart,
  createNoiseData,
  createSlidingHistogram,
  drawSlidingHistogram,
} from "../../src/chart/chart";
import { describe, test } from "vitest";

describe("WebGPUCanvas2DContext - Chart Example", async () => {
  test("chart example", async () => {
    const canvas = document.createElement("canvas");
    canvas.style.position = "absolute";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    document.body.appendChild(canvas);
    canvas.width = canvas.getBoundingClientRect().width;
    canvas.height = canvas.getBoundingClientRect().height;

    const offCanvas = new OffscreenCanvas(canvas.width, canvas.height);
    const data = createNoiseData(10_000);
    // drawChart(data, canvas);
    const binSize = 1;
    const histData = createSlidingHistogram(data, 500, binSize);
    drawSlidingHistogram(histData, offCanvas, binSize);

    const ctx = await createCanvas2DContext(canvas);
    const fragmentShader = FragmentShader.new({
      source: `
        @fragment
        fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
            return vec4<f32>(pos.x / ${canvas.width}.0, pos.y / ${canvas.height}.0, 0.5, 1.0);
        }
        `,
    });

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

    ctx.loop(Infinity, async () => {
      await ctx.draw(scene);
    });
  });
});
