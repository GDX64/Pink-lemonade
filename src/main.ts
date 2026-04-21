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
