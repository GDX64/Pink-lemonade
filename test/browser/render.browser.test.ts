import { describe, expect, test, vi } from "vitest";
import { createCanvas2DContext } from "../../src";
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
    context.fillStyle = "red";
    context.rect(200, 200, 100, 100);
    context.fill();

    context.moveTo(1, 1);
    context.lineTo(200, 200);
    context.lineTo(200, 1);
    context.lineTo(1, 1);
    context.strokeStyle = "blue";
    context.stroke();

    await context.flush();

    context.destroy();
  });
});
