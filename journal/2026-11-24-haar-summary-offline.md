# 2026-11-24 — Haar-Coded Density Summary: Offline Rate-Distortion Against the Moment x-Tree

## Context & Goal

First step of the plan in [From Batched Mixture Reduction to Mergeable Summaries](./2026-11-23-mergeable-summaries.md): an **offline summary benchmark**, in memory, with no network and no byte layout. That entry proposed starting with moments up to order 4, a small mixture, and Haar. The user asked to try **Haar** first.

- **Question:** how many bytes does a Haar-coded summary need to reach a given rendered-density error, compared with today's moment x-tree?
- **Question:** does a per-view error estimate built from the coefficients bound the true error?

The code is a new crate, `rust/progressive-summaries/` (`pink-lemonade-progressive-summaries`). It is an experiment crate, not a strategy, so it has no `DatasetBuilder` or `ProgressiveClient`. It contains:
- `haar.rs`: the summary;
- `blur.rs`: closed-form blur gains;
- `xtree.rs`: the baseline;
- `examples/rate_distortion.rs`: the benchmark.

## Design

### Haar pyramid (`HaarIndex`)

- **Grid.** Dyadic cells over the data bbox. Column level `a` gives `2^a` columns and row level `b` gives `2^b` rows.
- **Index.** The offline index is a cell-mass oracle standing in for stored coefficients: for each column level up to 22, the points of every column sorted by y, with prefix weights. Any dyadic cell's mass is then two binary searches.
- **Levels per view.**
  - **Emit level `(Ex, Ey)`:** the coarsest level whose cells are at most `emit_cell_sigma = 0.5` kernel widths on screen.
  - **Finest level:** `(Fx, Fy) = (Ex, Ey) + D`, with `D = 3`, so sub-cells are about σ/16.
  - **Root level:** `(Fx − n, Fy − n)` with `n = min(Fx, Fy)`, so one axis is a single column or row.
- **Transform.** The non-standard 2D Haar transform runs from the root level to the finest level. Each step halves both axes and stores 3 details per parent cell (x, y and xy differences of the four child masses).
  - Every basis function at a given step has the view's emit-cell aspect ratio, so the coefficients a view needs are local to it.
  - A tensor-product (standard) decomposition would not be local: its coarse-x × fine-y basis functions span the whole dataset.
  - A stored version needs one pyramid per aspect ratio, as the sparse grid does.
- **Exactness rule.** As in every strategy, only finest-level columns fully inside `[min_x, max_x]` are coded. The points in the partial edge columns are drawn raw. With `y_bounds`, only rows within 6σ of the screen are coded; points beyond are not drawn, rather than collapsed.
- **Emission.** One Gaussian per emit cell, moment-matched to its reconstructed sub-cells, each treated as a box with variance `h²/12`. Cells whose reconstructed mass is negative are drawn with **negative amplitude**; see Finding 3 for why.

### Error budgets (the cut policy)

Dropping a detail `d` moves `|d|/2` of mass across half a parent cell. In σ units, the blur damps that change by a known factor that depends only on the parent cell's width `P` on screen. `blur.rs` gives closed forms, each checked against numerical integration in its tests:
- `box_gain`, `wavelet_gain`: L2 energy surviving the blur.
- `wavelet_l1_gain(P) = (2/P)·(h·erf(h/√2) + 2φ(h) − 2φ(0))` with `h = P/2`: the fraction of the moved mass still visible after the blur.

The client drops the smallest coefficients until a budget is used up:
- **L2:** `Σ c²·gain ≤ ε²·total`. This assumes the blurred basis functions are orthogonal.
- **L1:** `Σ |d|·gain₁ ≤ ε·M`, where `M` is the coded region's mass. By the triangle inequality this is an **upper bound on the global relative L1 error of the rendered density**, with no assumption.

### Baseline: x-tree (`XTree`)

- A binary moment tree over x-runs with 32-point leaves.
- The same `standard_decision`, `push_group` and `collect_points` as the paged strategies.
- Cost is every node the client reads (both children of each expanded node) plus the raw points of every leaf it opens.
- The merge threshold `t ∈ {0.5, 1, 2}` traces its rate-distortion curve.

### Byte accounting

These sizes are assumptions; no layout exists yet.

| item | bytes |
|---|---:|
| x-tree node | 32 (`f32` bbox + 16-byte moments; the real `PackedNode` is 64) |
| raw point | 24 |
| Haar detail coefficient | 4 (`f32`) |
| parent cell with at least one kept detail | +2 (position) |
| Haar root cell | 6 |

The Haar count assumes the client can fetch **exactly the kept coefficients**. That needs storage sorted by magnitude within spatial blocks, which has not been designed. **This favours Haar.**

### Setup

- 1M points: `RandomWalk` (seed 578 211), `NoisySpikes` white noise (seed 4 242, y-zoom centre 0.1), `BurstyX` (seed 9 001).
- Screen 1920 × 1080; σ = 16 px and σ = 4 px.
- Views, each measured cold, one at a time:
  - the overview;
  - zoom widths 0.2, 0.05, 0.01 and 0.001 around 0.37;
  - window `[0.40, 0.45]`, plus 8× and 32× y-zooms of it.
- **Error:** the harness's `evaluate_density_nmae` at 1,000 reference pixels per view. The reference sampler matches the harness exactly but sums only the points within 6σ in x.

## How the design got here (100K points, σ = 16)

Each step was a failure that changed the design.

1. **L2 budget: the estimate is far too optimistic.**
   - Random walk `zoom w=0.01`: ε = 1% gives 7.4% NMAE and 13,472 kernels, against 850 at ε = 0.
   - Window y32x: ε = 1% gives 86.9%.
   - **Cause:** below about σ, blurred Haar wavelets are almost identical (derivative-of-Gaussian shapes), so dropped details add coherently, not in quadrature.
   - Keeping every detail below the emit level stops the worst of it, but those are 85% of all coefficients, so little is saved. The error was still 9.4% on `zoom w=0.01`.
   - L2-optimal truncation also spreads low ringing over empty space, which costs little in L2 but a lot in L1.
2. **L1 budget measured against the wrong mass.** The first L1 runs measured ε against the root cells' mass. Root cells extend far outside zoomed views, which inflated the budget. The budget now uses the coded region's mass.
3. **Negative cells must be drawn.**
   - Even with the right budget, white-noise `zoom w=0.05` at ε = 1% gave 10.7% NMAE.
   - Instrumenting the reconstruction showed 543 units of negative mass (11% of the view's 4,997) in skipped cells.
   - Dropping details just above the emit scale creates sub-σ ringing. It cancels after the blur, but only if the negative half is drawn.
   - With signed kernels the same view gives 0.16%, and the bound holds everywhere.

The variants stay selectable for reproduction: `--budget l2`, `--keep-sub-emit`.

## Results (1M points, L1 budget, D = 3, emit cell 0.5σ)

### Does the L1 estimate bound the true error?

True NMAE divided by ε, over every Haar row with ε > 0 (24 views per σ):

| ε | σ = 16: max (median) | σ = 4: max (median) |
|---:|---:|---:|
| 0.01 | 0.93 (0.14) | 2.93 (0.41) |
| 0.05 | 0.32 (0.06) | 0.70 (0.13) |
| 0.1 | 0.25 (0.06) | 0.46 (0.10) |
| 0.2 | 0.24 (0.08) | 0.33 (0.12) |
| 0.5 | 0.34 (0.12) | 0.28 (0.13) |

- The bound covers truncation only. The only row above 1 is at σ = 4, ε = 1%, where the ε = 0 floor dominates.
- That floor comes from quantizing sub-cells to σ/16. It is worst on sparse deep zooms: random walk `zoom w=0.001` gives 0.90% at σ = 16 and 2.84% at σ = 4.
- Otherwise the bound holds with a 3–15× margin, so it is safe but loose.

### Cheapest configuration of each method with NMAE ≤ 1%

Cells are `bytes / NMAE (parameter)`. Each method picks its own cheapest point; the x-tree's bytes barely move with `t` (raw points dominate), so it often reaches far below 1%.

#### σ = 16 px

| dataset | view | x-tree (t) | Haar (ε) | x-tree ÷ Haar | kernels x-tree / Haar |
|---|---|---|---|---:|---|
| random_walk | overview | 288.4 KiB / 0.10% (1) | 19.8 KiB / 0.62% (0.1) | 14.6 | 857 / 8031 |
| random_walk | zoom w=0.2 | 238.9 KiB / 0.12% (1) | 20.1 KiB / 0.61% (0.1) | 11.9 | 587 / 5536 |
| random_walk | zoom w=0.05 | 1007.3 KiB / 0.13% (1) | 35.8 KiB / 0.47% (0.1) | 28.1 | 1073 / 8097 |
| random_walk | zoom w=0.01 | 257.2 KiB / 0.02% (0.5) | 40.4 KiB / 0.72% (0.1) | 6.4 | 3769 / 11439 |
| random_walk | zoom w=0.001 | 27.3 KiB / 0.05% (0.5) | 22.9 KiB / 0.94% (0.02) | 1.2 | 800 / 3760 |
| random_walk | window [0.40,0.45] | 1.18 MiB / 0.15% (1) | 41.7 KiB / 0.54% (0.1) | 28.9 | 1785 / 9154 |
| random_walk | window y8x | 374.0 KiB / 0.02% (0.5) | 60.4 KiB / 0.72% (0.05) | 6.2 | 11060 / 11483 |
| random_walk | window y32x | 217.7 KiB / 0.01% (0.5) | 34.7 KiB / 0.89% (0.05) | 6.3 | 7705 / 10288 |
| white_noise | overview | 24.89 MiB / 0.00% (0.5) | 0.5 KiB / 0.84% (0.1) | 52404 | 866260 / 32768 |
| white_noise | zoom w=0.2 | 4.98 MiB / 0.01% (0.5) | 34.0 KiB / 0.24% (0.1) | 149.8 | 173255 / 38838 |
| white_noise | zoom w=0.05 | 1.25 MiB / 0.01% (0.5) | 16.5 KiB / 0.66% (0.2) | 77.5 | 43293 / 43678 |
| white_noise | zoom w=0.01 | 257.2 KiB / 0.02% (0.5) | 49.4 KiB / 0.80% (0.2) | 5.2 | 8664 / 13419 |
| white_noise | zoom w=0.001 | 27.3 KiB / 0.06% (0.5) | 41.1 KiB / 0.78% (0.05) | 0.7 | 874 / 3270 |
| white_noise | window [0.40,0.45] | 1.25 MiB / 0.01% (0.5) | 16.8 KiB / 0.68% (0.2) | 76.1 | 43347 / 38455 |
| white_noise | window y8x | 1.25 MiB / 0.01% (0.5) | 287.5 KiB / 0.94% (0.2) | 4.4 | 49092 / 100823 |
| white_noise | window y32x | 1.25 MiB / 0.00% (0.5) | 336.0 KiB / 0.64% (0.05) | 3.8 | 49766 / 30986 |
| bursty_x | overview | 193.5 KiB / 0.54% (2) | 35.7 KiB / 0.52% (0.05) | 5.4 | 575 / 8596 |
| bursty_x | zoom w=0.2 | 128.5 KiB / 0.64% (2) | 10.6 KiB / 0.55% (0.1) | 12.2 | 377 / 18426 |
| bursty_x | zoom w=0.05 | 62.0 KiB / 0.99% (2) | 12.9 KiB / 0.60% (0.1) | 4.8 | 192 / 32302 |
| bursty_x | zoom w=0.01 | 38.2 KiB / 0.07% (0.5) | 32.2 KiB / 0.77% (0.1) | 1.2 | 747 / 2781 |
| bursty_x | zoom w=0.001 | 5.5 KiB / 0.24% (0.5) | 11.1 KiB / 0.76% (0.01) | 0.5 | 127 / 145 |
| bursty_x | window [0.40,0.45] | 1022.2 KiB / 0.49% (2) | 22.6 KiB / 0.58% (0.1) | 45.2 | 942 / 9090 |
| bursty_x | window y8x | 549.2 KiB / 0.02% (0.5) | 39.7 KiB / 0.62% (0.1) | 13.8 | 16040 / 5399 |
| bursty_x | window y32x | 364.5 KiB / 0.02% (0.5) | 37.8 KiB / 0.81% (0.1) | 9.6 | 12865 / 3900 |

#### σ = 4 px

| dataset | view | x-tree (t) | Haar (ε) | x-tree ÷ Haar | kernels x-tree / Haar |
|---|---|---|---|---:|---|
| random_walk | overview | 1.49 MiB / 0.98% (2) | 187.3 KiB / 0.61% (0.1) | 8.1 | 3452 / 57955 |
| random_walk | zoom w=0.2 | 4.87 MiB / 0.15% (1) | 162.6 KiB / 0.55% (0.1) | 30.7 | 8785 / 32993 |
| random_walk | zoom w=0.05 | 1.25 MiB / 0.02% (0.5) | 193.1 KiB / 0.72% (0.1) | 6.6 | 23403 / 61121 |
| random_walk | zoom w=0.01 | 257.2 KiB / 0.03% (0.5) | 159.4 KiB / 0.91% (0.05) | 1.6 | 8075 / 70622 |
| random_walk | zoom w=0.001 | 27.3 KiB / 0.25% (0.5) | none ≤ 1% | – | 980 / – |
| random_walk | window [0.40,0.45] | 1.25 MiB / 0.02% (0.5) | 186.3 KiB / 0.88% (0.1) | 6.8 | 28436 / 67159 |
| random_walk | window y8x | 368.0 KiB / 0.01% (0.5) | 182.9 KiB / 0.86% (0.02) | 2.0 | 13284 / 44991 |
| random_walk | window y32x | 216.9 KiB / 0.01% (0.5) | none ≤ 1% | – | 8027 / – |
| white_noise | overview | 24.89 MiB / 0.00% (0.5) | 161.1 KiB / 0.75% (0.2) | 158.2 | 964204 / 412886 |
| white_noise | zoom w=0.2 | 4.98 MiB / 0.01% (0.5) | 751.9 KiB / 0.74% (0.2) | 6.8 | 192805 / 300107 |
| white_noise | zoom w=0.05 | 1.25 MiB / 0.02% (0.5) | 1.04 MiB / 0.68% (0.1) | 1.2 | 48189 / 196887 |
| white_noise | zoom w=0.01 | 257.2 KiB / 0.03% (0.5) | 438.0 KiB / 0.87% (0.05) | 0.6 | 9638 / 35695 |
| white_noise | zoom w=0.001 | 27.3 KiB / 0.22% (0.5) | none ≤ 1% | – | 971 / – |
| white_noise | window [0.40,0.45] | 1.25 MiB / 0.01% (0.5) | 1.04 MiB / 0.67% (0.1) | 1.2 | 48193 / 198139 |
| white_noise | window y8x | 1.25 MiB / 0.00% (0.5) | 1.57 MiB / 0.97% (0.05) | 0.8 | 49766 / 124481 |
| white_noise | window y32x | 1.25 MiB / 0.00% (0.5) | 620.1 KiB / 0.84% (0.01) | 2.1 | 49939 / 20212 |
| bursty_x | overview | 8.01 MiB / 0.39% (2) | 122.9 KiB / 0.96% (0.1) | 66.8 | 8094 / 77560 |
| bursty_x | zoom w=0.2 | 4.48 MiB / 0.50% (2) | 93.6 KiB / 0.48% (0.1) | 49.0 | 4770 / 37330 |
| bursty_x | zoom w=0.05 | 2.11 MiB / 0.81% (2) | 88.6 KiB / 0.51% (0.1) | 24.4 | 2283 / 22405 |
| bursty_x | zoom w=0.01 | 38.2 KiB / 0.09% (0.5) | 92.6 KiB / 0.85% (0.02) | 0.4 | 1212 / 1446 |
| bursty_x | zoom w=0.001 | 5.5 KiB / 0.23% (0.5) | none ≤ 1% | – | 145 / – |
| bursty_x | window [0.40,0.45] | 1.28 MiB / 0.02% (0.5) | 271.8 KiB / 0.47% (0.1) | 4.8 | 29681 / 44201 |
| bursty_x | window y8x | 546.1 KiB / 0.02% (0.5) | 204.6 KiB / 0.97% (0.1) | 2.7 | 19540 / 21046 |
| bursty_x | window y32x | 364.5 KiB / 0.04% (0.5) | 120.1 KiB / 0.89% (0.05) | 3.0 | 13526 / 8624 |

### At the x-tree's own error (t = 1)

Requiring Haar to match the x-tree's error at `t = 1` (typically 0.05–0.3%) is a much stricter target. At σ = 16:
- **Haar still wins 2.9–73×** on the dense views: random-walk overview and zooms 0.2 and 0.05; white-noise overview, zooms 0.2 and 0.05 and the window; `BurstyX` zooms 0.2 and 0.05, the window, and both y-zooms.
- **It cannot reach that error** on 6 of 24 views, because of the σ/16 quantization floor: random walk `zoom w=0.001`, the window and both y-zooms; white-noise y32x; the `BurstyX` overview.

## Findings

1. **A Haar summary with an L1 budget is a large win wherever there are many points per kernel.**
   - At NMAE ≤ 1%, σ = 16, against the x-tree:
     - random walk: 6–29× fewer bytes, except `zoom w=0.001` (1.2×);
     - white noise: 3.8–150× fewer, except `zoom w=0.001` (0.7×), plus 52,000× on the overview;
     - `BurstyX`: 5–45× fewer on every view down to width 0.05.
   - White noise is the extreme. A uniform band needs almost no details once their blurred effect is under budget: **the overview is 0.5 KiB at 0.84%**. The x-tree reads all 24.9 MiB there; it is the known worst case, which the ripmap and grid already fix.
2. **It loses on sparse deep zooms.**
   - At width 0.001 (≈100 points on screen) Haar costs 0.5–1.5× more than raw points at σ = 16 and cannot reach 1% at σ = 4.
   - Coding positions to σ/16 through 3 levels of details costs more than 24-byte raw points. It also leaves a 1–3% quantization floor that the bound doesn't see.
   - Raw points, or moments, are the better code when cells hold few points.
3. **Only the L1 budget is usable, and only with signed emission.** The L1 estimate is a true upper bound because it never assumes cancellation: true/ε ≤ 0.93 for every ε at σ = 16, and ≤ 0.70 for ε ≥ 0.05 at σ = 4. The L2 estimate assumes cancellation and fails by 7–90×. Signed kernels are required, and the renderer must accumulate density with sign.
4. **The bound is loose by 3–15×.** For the same true error, ε can be set several times higher. An ε-to-NMAE calibration, like `c₃`/`c₄` for Edgeworth, would let the budget aim at the true error.
5. **Kernel counts go up on line-like data.** At σ = 16, the ≤ 1% configurations draw up to 9× more kernels than the x-tree on random walk, and 49× and 168× more on `BurstyX` zooms 0.2 and 0.05. The emit grid is σ/2, while tree nodes span up to σ. Signed ringing kernels add to the count. White noise draws 4.5–26× fewer on the overview and zoom 0.2, and about as many or up to 2× more elsewhere. Emitting coarser cells where no detail was kept would reduce the line-data counts.
6. **It breaks the project's exactness invariant.** After truncation, mass, mean and covariance over the view are bounded in L1 but no longer exact. Any strategy built on this needs either an exception to the "moments always exact" rule or forced retention of the coefficients that carry the region's total moments.

## Caveats

- Everything is offline and cold per view, with no caching across steps, no layout, and assumed byte sizes. The Haar side assumes ideal, magnitude-ordered fetches. The x-tree is the only baseline, so the ripmap and sparse grid, the real competitors, are not in these tables.
- The x-tree is charged 32 B per node, not the real 64 B, and is a binary tree, not paged.
- The budget uses the exact region mass; a real client would use its reconstruction.
- With `y_bounds`, rows beyond 6σ are skipped rather than collapsed into off-screen kernels. NMAE is unaffected.

## Reproduce

```bash
cargo run --profile fast -p pink-lemonade-progressive-summaries --example rate_distortion --manifest-path rust/Cargo.toml -- --only random_walk --out rw.json
```

- Also run with `--only white_noise` and `--only bursty_x`, and add `--sigma 4` for the small-kernel tables.
- Each run takes under 1.5 min at 1M points.
- Defaults: `--points 1000000 --samples 1000 --detail 3 --emit-cell 0.5 --budget l1`.

Verification:
- `cargo test -p pink-lemonade-progressive-summaries`: closed-form blur gains against numerical integration, cell masses summing to the total, and lossless reconstruction preserving the view's mass.
- `cargo clippy --all-targets -D warnings` and the `wasm32-unknown-unknown` check pass.

## Next Hypotheses

1. **Hybrid code: Haar where dense, raw where sparse.** Stop the pyramid at cells with ≤ k points (k ≈ 8–32) and fetch their raw points instead. This targets Finding 2 and should keep the dense-view wins.
2. **Calibrate the L1 bound** against true NMAE (Finding 4) and re-plot rate-distortion at matched true error.
3. **Put the real competitors in the offline benchmark.** Add ripmap- and grid-style planes (moment cells, no thresholding) on the same views, so Haar is compared against the designs that already fix white noise.
4. **Design the stored layout:** per-aspect pyramids, with coefficients ordered by blurred magnitude inside spatial blocks so a client can read prefixes. Then measure what that ordering costs against the ideal fetches assumed here.
5. **Coarser emission where no detail was kept**, to bring random-walk kernel counts down (Finding 5).
6. **Moments up to order 4 and the small mixture** remain to be run through this benchmark, as the mergeable-summaries entry planned.
