# Supplementary Material

This repository contains the interactive artifact accompanying the paper. The reviewer-facing entry point is the built page at `./index.html`.

## How to use it

1. Open `index.html` in a recent Chromium-based browser.
2. On the home screen, choose the artifact you want to inspect:
   - `application`
   - `benchmark`
   - `Mean Squared Error`
   - `full KDE figure`
   - `merge points figure`
   - `Integral selection + zoom figure`
   - `Bitcoin figure`
   - `Distribution figure (Plotly)`
3. The selected page runs directly in the browser and displays the resulting visualization or table.

## Notes for the reviewer

- The `benchmark` and `Mean Squared Error` pages perform computation in the browser and may take a short time to finish.
- Some visualizations rely on WebGPU support. If WebGPU is unavailable on the machine/browser, affected views may not render correctly.
- The source-level `index.html` at the repository root is for development. For review, use `/index.html`.


## Project layout

```
src/
  main.ts            app entry: page router for the reviewer-facing pages
  index.ts           library entry
  chart/             WebGPU/canvas chart rendering (ChartCanvas, GaussianChart, ViewManager)
  downsampling/      TS wrappers around the WASM reducers (merge, kl, salmond, tree)
  data/              synthetic noise generators + sample datasets (data.json is local, gitignored)
  pages/             one module per page; benchmark/ and mse/ hold their page + chart
  examples/          standalone, unused-by-app experiments
  types/             ambient module declarations
rust/                WASM crate (built into pkg/ by `npm run wasm`)
test/                vitest tests; shared helpers in test/support/
paper/               LaTeX sources, bibliography, revision letter; references/ holds cited-paper text
```

## Chart API

```ts
import { createGaussianChart } from "./src/chart/api";

const chart = await createGaussianChart(mountEl, {   // mountEl must have a size
  data: { x: timestamps, y: prices, weight: volumes }, // or { points: Float64Array [x,y,w,...] }
  style: { sigma: 16, quantSteps: 6, opacityCut: 0.03 },
  axes: { x: { show: true }, y: { show: true } },
  merge: { thresholdSigmas: 1 },                      // 0 = no merging
  controls: false,                                    // lil-gui debug panel
});

chart.update({ style: { opacityCut: 0.05 } });
chart.view.set({ x: [t0, t1] });                      // data units
chart.selection.set({ x: [t0, t1], y: [p0, p1] });
const off = chart.on("selectionchange", (sel) => {});  // also "viewchange", "frame"
const { integral } = (await chart.integrate(chart.selection.get() ?? undefined))!;
const png = await chart.toImage({ width: 800, height: 400, pixelRatio: 3 });
chart.destroy();
```
