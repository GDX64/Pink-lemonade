import {
  createCanvas2DContext,
  FragmentShader,
  Rect,
  Scene,
} from "../canvas2d/context";

import {
  createNoiseData,
  createSlidingHistogram,
  drawChart,
  drawSplatKernelSeries,
  drawSlidingHistogram,
} from "../chart/chart";
import fragmentShaderSource from "./warping.fragment.wgsl?raw";

export async function cpuExample() {
  const canvas = createCanvas();
  const overlayCanvas = createCanvas();
  overlayCanvas.style.opacity = "0.2";
  const data = createNoiseData(1000_000);
  drawChart(data, overlayCanvas);
  const donwScaling = 16;
  const width = Math.floor(canvas.width / donwScaling);
  const height = Math.floor(canvas.height / donwScaling);
  const density = drawSplatKernelSeries(data, { width, height });
  const ctx = await createCanvas2DContext(canvas);
  const fragmentShader = FragmentShader.new({
    source: fragmentShaderSource,
  });

  window.addEventListener("click", () => {
    drawSplatKernelSeries(data, { width, height });
  });

  const texture = await ctx.createCanvasTexture({
    data: density,
    width,
    height,
  });
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
  // ctx.loop(Infinity, async () => {
  //   await ctx.draw(scene);
  // });

  return { ctx };
}

function createCanvas() {
  const canvas = document.createElement("canvas");
  canvas.style.position = "absolute";
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  document.body.appendChild(canvas);
  canvas.width = canvas.getBoundingClientRect().width * devicePixelRatio;
  canvas.height = canvas.getBoundingClientRect().height * devicePixelRatio;
  return canvas;
}
