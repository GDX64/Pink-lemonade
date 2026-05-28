import { createNoiseData } from "../../chart/chart";
import { GaussianChart, type LoadedData } from "./gaussian-chart";

const DATA_KIND = "random";

export async function rasterizingExample() {
  const chart = new GaussianChart(await loadData());
  await chart.start();
}

async function loadData(): Promise<LoadedData> {
  if (DATA_KIND === "random") {
    const points = createNoiseData(100_000);
    const nFiltered = points.length;
    if (nFiltered === 0) {
      return { n: 0, dataF64: new Float64Array(0), xMin: 0, xScale: 1 };
    }

    // createNoiseData returns [time, price] points; normalize x and keep unit weight
    const xMin = points[0]![0];
    const xMax = points[nFiltered - 1]![0];
    const xScale = xMax - xMin || 1;
    const dataF64 = new Float64Array(nFiltered * 3);
    for (let i = 0; i < nFiltered; i++) {
      const [time, price] = points[i]!;
      dataF64[i * 3] = (time - xMin) / xScale;
      dataF64[i * 3 + 1] = price;
      dataF64[i * 3 + 2] = 1;
    }
    return { n: nFiltered, dataF64, xMin, xScale };
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
