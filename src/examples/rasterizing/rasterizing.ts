import { createNoiseFloatData } from "../../chart/chart";
import { distribution } from "../../chart/chart";
import { GaussianChart, type LoadedData } from "./gaussian-chart";
import Plotly from "plotly.js-dist-min";

const BASE_WIDTH = 800;
const BASE_HEIGHT = 600;

export async function rasterizingExample() {
  await createChart1();
}

export async function rasterizingApplication(mount?: HTMLElement) {
  await createChart1(mount);
}

export async function renderFigureChart(
  kind: "figure-1" | "figure-2" | "figure-3" | "figure-4" | "figure-5",
  mount: HTMLElement,
) {
  if (kind === "figure-5") {
    await renderDistributionFigure(mount);
    return null;
  }

  const dataKind = kind === "figure-4" ? "data" : "random";
  const data = await loadData(dataKind);
  const chart = new GaussianChart({ data });
  mount.innerHTML = "";
  const container = chart.container;
  mount.appendChild(container);
  container.style.width = `${BASE_WIDTH}px`;
  container.style.height = `${BASE_HEIGHT}px`;
  container.style.position = "relative";
  chart.state.showTimescale = true;
  chart.state.showYAxis = true;
  chart.state.quantSteps = 6;
  chart.state.mergeThresholdSigmas = 1;
  if (kind === "figure-1") {
    chart.state.mergeThresholdSigmas = 0;
  }
  await chart.start();
  if (kind === "figure-3") {
    chart.setViewRangeX(0.6500622508888406, 1);
    chart.setSelection({
      x: 388,
      y: 277.5,
      width: 246,
      height: 182,
    });
  }
  chart.setupRenderLoop();
  return chart;
}

async function renderDistributionFigure(mount: HTMLElement) {
  mount.innerHTML = "";

  const plotHost = document.createElement("div");
  plotHost.style.width = `${BASE_WIDTH}px`;
  plotHost.style.height = `${BASE_HEIGHT}px`;
  plotHost.style.position = "relative";
  mount.appendChild(plotHost);

  const sampleCount = 1000;
  const xs = Array.from(
    { length: sampleCount },
    (_, i) => i / (sampleCount - 1),
  );
  const ys = xs.map((x) => distribution(x));

  await Plotly.newPlot(
    plotHost,
    [
      {
        x: xs,
        y: ys,
        type: "scatter",
        mode: "lines",
        line: {
          color: "#1f6fb2",
          width: 3,
        },
        // name: "amplitude(x)",
      },
    ],
    {
      width: BASE_WIDTH,
      height: BASE_HEIGHT,
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      margin: { l: 60, r: 20, t: 40, b: 50 },
      // title: {
      //   text: "Probability Distribution Used in Examples",
      //   font: { color: "#16324f", size: 18 },
      // },
      xaxis: {
        title: { text: "x" },
        range: [0, 1],
        // gridcolor: "#dbe7f3",
        zerolinecolor: "#b8cde2",
      },
      yaxis: {
        // title: { text: "amplitude(x)" },
        // gridcolor: "#dbe7f3",
        zerolinecolor: "#b8cde2",
      },
      showlegend: false,
    },
    {
      responsive: false,
      displaylogo: false,
      toImageButtonOptions: {
        format: "svg",
        filename: "amplitude",
        width: 800,
        height: 400,
        scale: 1,
      },
    },
  );
}

export async function generateApproxFigure() {
  await randomApproxImage();
}

export async function generateFullFigure() {
  await randomImageFull();
}

async function createChart1(mount?: HTMLElement) {
  const data = await loadData("random");
  const chart = new GaussianChart({
    data,
  });
  const container = chart.container;
  const target = mount ?? document.body;
  target.appendChild(container);
  const useFullscreen = !!mount;
  const width = useFullscreen
    ? Math.max(window.innerWidth, target.clientWidth)
    : BASE_WIDTH;
  const height = useFullscreen
    ? Math.max(window.innerHeight, target.clientHeight)
    : BASE_HEIGHT;
  container.style.width = `${width}px`;
  container.style.height = `${height}px`;
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
        width,
        height,
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
