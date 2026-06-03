import { createNoiseData, createNoiseFloatData } from "../../chart/chart";
import { GaussianChart, type LoadedData } from "./gaussian-chart";

const BASE_WIDTH = 800;
const BASE_HEIGHT = 600;

export async function rasterizingExample() {
  await createChart1();
  // await randomImageFull();
  // await randomApproxImage();
  // await areaSelection();
}

async function createChart1() {
  const data = await loadData("random");
  const chart = new GaussianChart({
    data,
  });
  const container = chart.container;
  document.body.appendChild(container);
  // container.style.width = "100vw";
  // container.style.height = "100vh";
  container.style.width = `${BASE_WIDTH}px`;
  container.style.height = `${BASE_HEIGHT}px`;
  container.style.position = "relative";
  chart.state.showTimescale = true;
  chart.state.showYAxis = true;
  chart.state.quantSteps = 6;
  await chart.start();
  chart.setupRenderLoop();
  setTimeout(() => {
    chart
      .integrate({
        x: 0,
        y: 0,
        width: BASE_WIDTH,
        height: BASE_HEIGHT,
      })
      .then((result) => console.log(result));
  }, 100);
}

async function randomApproxImage() {
  const data = await loadData("random");
  const chart = new GaussianChart({
    data,
  });
  chart.state.showTimescale = true;
  chart.state.showYAxis = true;
  chart.state.quantSteps = 6;
  await chart.start();
  chart.downloadImage({
    width: BASE_WIDTH,
    height: BASE_HEIGHT,
    name: "approx_image",
    devicePixelRatio: 3,
  });
}

async function randomImageFull() {
  const data = await loadData("random");
  const chart = new GaussianChart({
    data,
  });
  chart.state.showTimescale = true;
  chart.state.showYAxis = true;
  chart.state.mergeThresholdSigmas = 0;
  chart.state.quantSteps = 6;
  await chart.start();
  chart.downloadImage({
    width: BASE_WIDTH,
    height: BASE_HEIGHT,
    name: "full_image",
    devicePixelRatio: 3,
  });
}

async function loadData(kind: "random" | "data"): Promise<LoadedData> {
  if (kind === "random") {
    return createNoiseFloatData();
  }

  const jsonData = (await import("./data.json?url")).default;
  const { base64: base64Data }: { base64: string } = await (
    await fetch(jsonData)
  ).json();
  const other = base64Data.replaceAll("'", "");
  const decoded = atob(other);
  const buffer = new Uint8Array(decoded.length);
  for (let i = 0; i < decoded.length; i++) {
    buffer[i] = decoded.charCodeAt(i);
  }
  const raw = new Float64Array(buffer.buffer);

  // data is flat: [time, price, buyQty, sellQty, ...] — sort by time ascending
  const n = raw.length / 4;
  const indices = Array.from({ length: n }, (_, i) => i).sort(
    (a, b) => raw[a * 4]! - raw[b * 4]!,
  );
  const data = new Float64Array(n * 4);
  for (let i = 0; i < n; i++) {
    const src = indices[i]! * 4;
    data.set(raw.subarray(src, src + 4), i * 4);
  }

  const nFiltered = data.length / 4;

  // normalize x to [0, 1] to avoid float precision loss in GPU/math
  const xMin = data[0]!;
  const xMax = data[(nFiltered - 1) * 4]!;
  const xScale = xMax - xMin || 1; // original span in ms; used to denormalize for display

  // pre-pack into Float64Array once — x is sorted and normalized, weight = buyQty + sellQty
  const dataF64 = new Float64Array(nFiltered * 3);
  for (let i = 0; i < nFiltered; i++) {
    const time = data[i * 4]!;
    const price = data[i * 4 + 1]!;
    const buyQty = data[i * 4 + 2]!;
    const sellQty = data[i * 4 + 3]!;
    dataF64[i * 3] = (time - xMin) / xScale;
    dataF64[i * 3 + 1] = price;
    dataF64[i * 3 + 2] = buyQty + sellQty;
  }
  return { n: nFiltered, dataF64, xMin, xScale };
}
