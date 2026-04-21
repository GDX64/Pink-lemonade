import {
  drawChart,
  createNoiseData,
  createSlidingHistogram,
  drawSlidingHistogram,
} from "../../src/chart/chart";
import { describe, test } from "vitest";

describe("WebGPUCanvas2DContext - Chart Example", () => {
  test("chart example", async () => {
    const canvas = document.createElement("canvas");
    canvas.style.position = "absolute";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    document.body.appendChild(canvas);
    canvas.width = canvas.getBoundingClientRect().width;
    canvas.height = canvas.getBoundingClientRect().height;

    const data = createNoiseData(10_000);
    // drawChart(data, canvas);
    const binSize = 1;
    const histData = createSlidingHistogram(data, 500, binSize);
    drawSlidingHistogram(histData, canvas, binSize);
  });
});
