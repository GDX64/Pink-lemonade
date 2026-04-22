export {
  type CanvasTexture,
  createCanvas2DContext,
  FragmentShader,
  Rect,
  Scene,
  WebGPUCanvas2DContext,
  type Canvas2DContextOptions,
  type ClearColor,
  type FragmentShaderOptions,
  type FragmentShaderUniforms,
  type FragmentShaderUniformValue,
  type RectOptions,
} from "./canvas2d/context";

import {
  createCanvas2DContext,
  FragmentShader,
  Rect,
  Scene,
} from "./canvas2d/context";
import {
  createNoiseData,
  createSlidingHistogram,
  drawSlidingHistogram,
} from "./chart/chart";
import fragmentShaderSource from "./warping.fragment.wgsl?raw";

export async function example() {
  const canvas = document.createElement("canvas");
  canvas.style.position = "absolute";
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  document.body.appendChild(canvas);
  canvas.width = canvas.getBoundingClientRect().width * devicePixelRatio;
  canvas.height = canvas.getBoundingClientRect().height * devicePixelRatio;

  const offCanvas = new OffscreenCanvas(canvas.width, canvas.height);
  const N = 10_000;
  const data = createNoiseData(N);
  const binSize = 10;
  const bins = canvas.width / 50;
  const each = Math.round(N / bins);
  const histData = createSlidingHistogram(data, each, binSize);
  drawSlidingHistogram(histData, offCanvas, binSize);

  const ctx = await createCanvas2DContext(canvas);
  const fragmentShader = FragmentShader.new({
    source: fragmentShaderSource,
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
  ctx.loop(Infinity, async () => {
    await ctx.draw(scene);
  });

  return { ctx };
}
