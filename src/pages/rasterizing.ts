import { createNoiseFloatData } from "../data/noise";
import { distribution } from "../data/noise";
import {
  createGaussianChart,
  type ChartData,
  type CreateChartOptions,
  type GaussianChartHandle,
} from "../chart/api";
import Plotly from "plotly.js-dist-min";

const BASE_WIDTH = 800;
const BASE_HEIGHT = 400;

/** Settings shared by every chart in the app and the paper figures. */
const BASE_SETTINGS = {
  axes: { x: { show: true }, y: { show: true } },
  style: { quantSteps: 6 },
} satisfies Omit<CreateChartOptions, "data">;

export async function rasterizingExample() {
  await createApplicationChart();
}

export async function rasterizingApplication(mount?: HTMLElement) {
  await createApplicationChart(mount);
}

export async function renderFigureChart(
  kind: "figure-1" | "figure-2" | "figure-3" | "figure-4" | "figure-5",
  mount: HTMLElement,
): Promise<GaussianChartHandle | null> {
  if (kind === "figure-5") {
    await renderDistributionFigure(mount);
    return null;
  }

  mount.innerHTML = "";
  const host = sizedHost(
    BASE_WIDTH,
    kind === "figure-4" ? BASE_HEIGHT * 1.5 : BASE_HEIGHT,
  );
  mount.appendChild(host);

  return createGaussianChart(host, {
    ...BASE_SETTINGS,
    data: await loadData(kind === "figure-4" ? "data" : "random"),
    style: {
      ...BASE_SETTINGS.style,
      ...(kind === "figure-4" ? { opacityCut: 0.042 } : {}),
    },
    merge: { thresholdSigmas: kind === "figure-1" ? 0 : 1 },
  });
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
  await exportFigure("approx_image", 1);
}

export async function generateFullFigure() {
  await exportFigure("full_image", 0);
}

async function createApplicationChart(mount?: HTMLElement) {
  // With a mount the chart fills it and follows its size; otherwise it is
  // appended to the page at the base size.
  const host = sizedHost(BASE_WIDTH, BASE_HEIGHT);
  if (mount) {
    host.style.width = "100%";
    host.style.height = "100%";
  }
  (mount ?? document.body).appendChild(host);
  return createGaussianChart(host, {
    ...BASE_SETTINGS,
    data: await loadData("random"),
    controls: true,
  });
}

async function exportFigure(name: string, thresholdSigmas: number) {
  // Rendered off-screen at a fixed size, then downloaded.
  const host = sizedHost(BASE_WIDTH, BASE_HEIGHT);
  host.style.position = "fixed";
  host.style.left = "-10000px";
  document.body.appendChild(host);
  const chart = await createGaussianChart(host, {
    ...BASE_SETTINGS,
    data: await loadData("random"),
    merge: { thresholdSigmas },
  });
  try {
    const blob = await chart.toImage({
      width: BASE_WIDTH,
      height: BASE_HEIGHT,
      pixelRatio: 3,
    });
    if (blob) downloadBlob(blob, `${name}.png`);
  } finally {
    chart.destroy();
    host.remove();
  }
}

function sizedHost(width: number, height: number) {
  const host = document.createElement("div");
  host.style.width = `${width}px`;
  host.style.height = `${height}px`;
  host.style.position = "relative";
  return host;
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

async function loadJson(): Promise<string> {
  const jsonData = (await import("../data/data.json?url")).default;
  return jsonData;
}

async function loadData(kind: "random" | "data"): Promise<ChartData> {
  if (kind === "random") {
    // The generator returns x normalized to [0, 1]; restore timestamps in
    // place so the chart gets data in real units without another copy.
    const { dataF64, xMin, xScale } = createNoiseFloatData(10_000_000);
    for (let i = 0; i < dataF64.length; i += 3) {
      dataF64[i] = dataF64[i]! * xScale + xMin;
    }
    return { points: dataF64 };
  }

  const { base64: base64Data }: { base64: string } = await (
    await fetch(await loadJson())
  ).json();
  const decoded = atob(base64Data.replaceAll("'", ""));
  const buffer = new Uint8Array(decoded.length);
  for (let i = 0; i < decoded.length; i++) {
    buffer[i] = decoded.charCodeAt(i);
  }
  // Flat records: [time, price, buyQty, sellQty, ...]. The chart sorts by
  // time and normalizes it; weight is the traded volume.
  const raw = new Float64Array(buffer.buffer);
  const n = raw.length / 4;
  const points = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    points[i * 3] = raw[i * 4]!;
    points[i * 3 + 1] = raw[i * 4 + 1]!;
    points[i * 3 + 2] = raw[i * 4 + 2]! + raw[i * 4 + 3]!;
  }
  return { points };
}
