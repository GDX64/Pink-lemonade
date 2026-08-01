import Plotly from "plotly.js-dist-min";
import type { BenchmarkMethod, BenchmarkResult } from "./benchmark-page";

/**
 * Latency against dataset size, log-log, one series per reduction method.
 *
 * Both axes span four to five decades, so log-log is the only readable
 * geometry --- and it makes the complexity claim legible as a slope rather than
 * as arithmetic the reader has to do: a least-squares fit of log(latency)
 * against log(N) recovers the exponent, which is annotated in the legend.
 *
 * Error bars are the 95% relative margin of error on the mean --- how precisely
 * the mean itself is pinned down by the number of runs taken, not how much
 * individual runs scatter. They say "this curve is where I claim it is"; they do
 * not describe the latency distribution, so they stay small once a scenario has
 * enough iterations.
 */

/** 60 fps. The line each method has to stay under to be a per-frame method. */
const FRAME_BUDGET_MS = 1000 / 60;

/** Fixed floor for the latency axis: 1 microsecond. */
const Y_MIN_MS = 0.001;

const SERIES: Array<{
  method: BenchmarkMethod;
  label: string;
  color: string;
}> = [
  { method: "merge", label: "screen-space merge", color: "#1f6fb2" },
  { method: "salmond", label: "Salmond clustering", color: "#e08a1e" },
  { method: "runnalls", label: "Runnalls KL", color: "#c0392b" },
];

/**
 * Least-squares slope of log10(y) against log10(x): the empirical scaling
 * exponent. Returns null for a single point, where no slope is defined.
 */
function fitExponent(xs: number[], ys: number[]): number | null {
  const n = xs.length;
  if (n < 2) return null;

  const lx = xs.map((v) => Math.log10(v));
  const ly = ys.map((v) => Math.log10(v));
  const mx = lx.reduce((a, b) => a + b, 0) / n;
  const my = ly.reduce((a, b) => a + b, 0) / n;

  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    num += (lx[i]! - mx) * (ly[i]! - my);
    den += (lx[i]! - mx) * (lx[i]! - mx);
  }
  return den === 0 ? null : num / den;
}

function formatMs(ms: number): string {
  if (ms >= 1_000) return `${(ms / 1_000).toFixed(2)} s`;
  if (ms >= 1) return `${ms.toFixed(2)} ms`;
  return `${(ms * 1_000).toFixed(1)} µs`;
}

export async function renderBenchmarkChart(
  results: BenchmarkResult[],
  mount: HTMLElement,
) {
  mount.innerHTML = "";

  const plotHost = document.createElement("div");
  plotHost.style.width = "100%";
  mount.appendChild(plotHost);

  const traces: unknown[] = [];

  for (const { method, label, color } of SERIES) {
    const points = results
      .filter((r) => r.method === method)
      .sort((a, b) => a.n - b.n);
    if (points.length === 0) continue;

    const xs = points.map((p) => p.n);
    const ys = points.map((p) => p.meanMs);
    const exponent = fitExponent(xs, ys);
    const name =
      exponent === null
        ? `${label} (single point)`
        : `${label} — slope ${exponent.toFixed(2)}`;

    traces.push({
      x: xs,
      y: ys,
      name,
      type: "scatter",
      mode: points.length > 1 ? "lines+markers" : "markers",
      line: { color, width: 3 },
      marker: { color, size: 9 },
      error_y: {
        type: "data",
        // 95% margin of error on the mean, as an absolute duration. A single
        // run has no margin to report, so it gets a bare marker.
        array: points.map((p) =>
          p.rmePct === null ? 0 : (p.meanMs * p.rmePct) / 100,
        ),
        color,
        thickness: 1.5,
        width: 6,
      },
      customdata: points.map((p) => [
        p.kernels,
        formatMs(p.p99Ms),
        formatMs(p.maxMs),
        p.samples,
        p.rmePct === null ? "n/a" : `±${p.rmePct.toFixed(2)}%`,
      ]),
      hovertemplate:
        `<b>${label}</b><br>` +
        "N = %{x:,}<br>" +
        "mean %{y:.4f} ms<br>" +
        "p99 %{customdata[1]} · max %{customdata[2]}<br>" +
        "kernels %{customdata[0]}<br>" +
        "%{customdata[3]} runs, RME %{customdata[4]}" +
        "<extra></extra>",
    });
  }

  const allN = results.map((r) => r.n);
  const xMin = Math.min(...allN);
  const xMax = Math.max(...allN);

  // Pinning the floor at 1 microsecond keeps the decades comparable between
  // runs; the top follows the slowest mean, which the error bars hug closely.
  const yTopMs = Math.max(...results.map((r) => r.meanMs), Y_MIN_MS);

  await Plotly.newPlot(
    plotHost,
    traces,
    {
      height: 620,
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      margin: { l: 78, r: 28, t: 64, b: 66 },
      title: {
        text: "Reduction latency vs dataset size",
        font: { color: "#16324f", size: 18 },
      },
      xaxis: {
        title: { text: "N (points)", font: { color: "#16324f", size: 14 } },
        type: "log",
        dtick: 1,
        gridcolor: "#e7eef6",
        zeroline: false,
        color: "#2d4d6c",
      },
      yaxis: {
        title: {
          text: "latency per reduction (ms)",
          font: { color: "#16324f", size: 14 },
        },
        type: "log",
        // Log axes take their range in log10 units, not data units.
        range: [Math.log10(Y_MIN_MS), Math.log10(yTopMs * 2)],
        gridcolor: "#e7eef6",
        zeroline: false,
        color: "#2d4d6c",
      },
      shapes: [
        {
          type: "line",
          xref: "x",
          yref: "y",
          x0: Math.log10(xMin),
          x1: Math.log10(xMax),
          y0: Math.log10(FRAME_BUDGET_MS),
          y1: Math.log10(FRAME_BUDGET_MS),
          line: { color: "#7a8ba0", width: 2, dash: "dash" },
        },
      ],
      annotations: [
        {
          xref: "x",
          yref: "y",
          x: Math.log10(xMax),
          y: Math.log10(FRAME_BUDGET_MS),
          xanchor: "right",
          yanchor: "bottom",
          text: "60 fps frame budget (16.7 ms)",
          showarrow: false,
          font: { color: "#5c708a", size: 12 },
        },
      ],
      legend: {
        x: 0.02,
        y: 0.98,
        bgcolor: "rgba(255,255,255,0.85)",
        bordercolor: "#d8e2ee",
        borderwidth: 1,
        font: { color: "#16324f", size: 13 },
      },
      hovermode: "closest",
    },
    { responsive: true, displaylogo: false },
  );

  return plotHost;
}
