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

