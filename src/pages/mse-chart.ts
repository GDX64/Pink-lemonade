import Plotly from "plotly.js-dist-min";
import type { MseMethod, MseResult } from "./mse-page";

/**
 * NRMSE against dataset size, log-log, one series per reduction method.
 *
 * Kernel budget grows with N (it is set by screen resolution, not by a fixed
 * count), so NRMSE is not expected to fall monotonically --- the useful
 * reading is the vertical gap between series at a given N, which is the
 * quality cost of each method relative to the merge at an equal kernel budget.
 *
 * Error bars are the 95% relative margin of error on the MSE estimate, carried
 * onto the NRMSE scale. They describe how precisely the Monte Carlo estimate is
 * pinned down by its sample count, not how the error itself is distributed.
 */

const SERIES: Array<{ method: MseMethod; label: string; color: string }> = [
  { method: "merge", label: "screen-space merge", color: "#1f6fb2" },
  { method: "salmond", label: "Salmond clustering", color: "#e08a1e" },
  { method: "runnalls", label: "Runnalls KL", color: "#c0392b" },
];

/** Fixed floor for the NRMSE axis, in percent. */
const Y_MIN_PCT = 1e-4;

export async function renderMseChart(results: MseResult[], mount: HTMLElement) {
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
    const ys = points.map((p) => p.nrmse * 100);

    traces.push({
      x: xs,
      y: ys,
      name: points.length > 1 ? label : `${label} (single point)`,
      type: "scatter",
      mode: points.length > 1 ? "lines+markers" : "markers",
      line: { color, width: 3 },
      marker: { color, size: 9 },
      error_y: {
        type: "data",
        // 95% margin of error on the NRMSE, as an absolute percentage.
        array: points.map((p) => (ys[points.indexOf(p)]! * p.rmePct) / 100),
        color,
        thickness: 1.5,
        width: 6,
      },
      customdata: points.map((p) => [p.kernels, p.samples, p.rmePct]),
      hovertemplate:
        `<b>${label}</b><br>` +
        "N = %{x:,}<br>" +
        "NRMSE %{y:.4f}%<br>" +
        "kernels %{customdata[0]}<br>" +
        "%{customdata[1]} samples, RME ±%{customdata[2]:.2f}%" +
        "<extra></extra>",
    });
  }

  const yTopPct = Math.max(...results.map((r) => r.nrmse * 100), Y_MIN_PCT);

  await Plotly.newPlot(
    plotHost,
    traces,
    {
      width: 800,
      height: 400,
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      // Wide right margin reserves the gutter the legend sits in.
      margin: { l: 80, r: 20, t: 20, b: 60 },
      // title: {
      //   text: "Reduction accuracy vs dataset size",
      //   font: { color: "#16324f", size: 18 },
      // },
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
          text: "NRMSE (%)",
          font: { color: "#16324f", size: 14 },
        },
        type: "log",
        // Log axes take their range in log10 units, not data units.
        range: [Math.log10(Y_MIN_PCT), Math.log10(yTopPct * 2)],
        gridcolor: "#e7eef6",
        zeroline: false,
        color: "#2d4d6c",
      },
      legend: {
        x: 1,
        y: 0.2,
        xanchor: "right",
        yanchor: "bottom",
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
