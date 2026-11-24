# 2026-11-23 — Salmond Tree: Sibling Groups Chosen by Gaussian Mixture Clustering

> **Status: code discarded.** The `progressive-salmond` crate and its `vs_xtree` runner described below were built, measured and then deleted without being committed: the follow-up discussion ([mergeable summaries](./2026-11-23-mergeable-summaries.md)) reframed the problem more generally, and this crate was only a first probe of it. The design and the numbers are kept here as the record; to reproduce them, rebuild from this description.

## Context & Goal

Every hierarchy so far picks a node's children geometrically: x order (x-tree), axis splits (k-d tree, quadtree) or fixed cells (ripmap, grid). The in-memory reducers in `rust/src/` (Runnalls, Salmond) give much better mixtures, but they cost `O(N²)` or more per view, so they can't run per frame. The question here: **if those reducers choose the groups offline, in bounded batches, do the nodes fit on screen earlier, so the client descends less?**

Salmond goes first because it is the cheaper of the two (one pass costs `O(n · clusters)`, versus Runnalls' rescan after every merge). During development the comparison is **x-tree vs Salmond tree only**, on the 1M scenarios.

## Strategy Design (`progressive-salmond`, `PLPSALM1`)

- **Bottom-up build in bounded windows:**
  - **Level 0:** the x-sorted points are cut into windows of `window = 1024` consecutive points. Each window is reduced by Salmond to about `window / ratio = 32` clusters, and each cluster becomes one leaf, its points sorted by x.
  - **Level ℓ+1:** windows of 1024 consecutive level-ℓ nodes are clustered the same way, now with each node's own covariance.
  - **Stop:** once a level has ≤ `top_nodes = 32` nodes.
  - **Cost:** linear in N, about 0.3–0.9 s for 1M points.
- **Exactness is untouched.** Salmond only decides the partition. Every node is the exact `Node::combined` fold of its children (or `Node::from_points` of its points), so mass, moments, extrema and line samples are as exact as in the x-tree.
- **Clustering space:**
  - **Units:** overview screen units, x scaled by `(16/9) / x_span` and y by `1 / y_span`, as in the k-d tree.
  - **Kernel:** every component gets `s²·I` added to its covariance, since raw points have none and Salmond needs a non-singular principal covariance. `s` is half the spacing at which the window's bbox would hold the target cluster count, falling back to the longest side for collinear windows.
- **Salmond port:**
  - **Same as `rust/src/salmond.rs`:** distance `D² = (b_i b_c / (b_i + b_c)) Δμᵀ P_c⁻¹ Δμ`, heaviest component first, the paper's `T1` / `ΔT`, and a 0.95 budget tolerance.
  - **New:** a union-find merge forest, so the result is a partition, not just a reduced mixture.
  - **Groups:** sorted by mean x inside each window. Internal members are also sorted by x, which keeps the next level's windows x-coherent.
- **Layout (three objects):**
  - **Variable fanout:** clusters are uneven, so each record is a 72-byte `(PackedNode, first: u32, count: u32)` with the quadtree's `LEAF_BIT` scheme instead of implicit `j·F` indexing.
  - **`manifest.bin`:** depth table plus inlined top depths, `root_cap = 32` records like the x-tree.
  - **`nodes.bin`:** the remaining depths back to back, children of one parent adjacent.
  - **`points.bin`:** tree order, so every subtree is one contiguous run.
  - **Depths at 1M:** 31 / ~977 / ~31K, the same as the x-tree.
- **Client:** the quadtree's depth-by-depth descent (one round trip per remote depth, coalesced range reads, LRU blob cache), with an in-order frontier: children replace their parent in place, so greedy kernel merging sees siblings together. It uses the standard resolve rule: fully inside the x-range and `fits_threshold`, or off-screen in y. Line samples are re-sorted by x.
- **Verification:**
  - **Tests:** `run_conformance_suite` at 4K, at 200K, and with `window = 64, ratio = 4` (a deep tree on 4K points). Unit tests cover the Salmond partition (separated blobs split, budget respected, every input labelled), layout contiguity, band-pure leaves on a two-band signal, and the manifest round trip.
  - **Lint and target:** clippy `-D warnings` and the `wasm32-unknown-unknown` check pass.
  - **`./dev clippy`:** the crate was added to it while it existed. Side fix that *was* kept: `dev.py` was missing a `-p` before `pink-lemonade-progressive-grid`, so `./dev clippy` passed the grid crate as a stray argument.
- **Dev runner (deleted with the crate):** `cargo run --profile fast -p pink-lemonade-progressive-salmond --example vs_xtree [-- --only <scenario>] [--nmae] [--out <json>]` runs only the x-tree and the Salmond tree. `--nmae` turns on density NMAE, which the presets leave off.

---

## Experiment 1 — The budget bracketing bug (v0)

The first build ported the in-memory bracketing as is. The bisection's lower bound was the last *accepted* threshold. After a pass, the merged components carry their spread in their covariance (`P_c` grows), so the next pass at that same threshold reaches much farther and the whole bracket overshoots. Bisection converged onto an overshooting threshold and accepted it.

Storage reveals the problem: 23.5 MiB on white noise means only about 8.7K records, so windows held about 9 clusters instead of 32.

v0 numbers, 1M points, `vs_xtree` without NMAE. The cells are wire KiB, x-tree → Salmond:

| Workload | `medium` | `medium_white_noise` | `medium_bursty` |
| --- | ---: | ---: | ---: |
| cold_overview | 2013 → **1613** | 25454 → **23823** | 2091 → **1986** |
| progressive_zoom_in | 2045 → **1672** | 43383 → **40315** | 5194 → **2758** |
| lateral_pan | **364** → 369 | 4397 → **4297** | 5211 → **775** |
| jump_and_revisit | **428** → 546 | 6413 → **6197** | 6179 → **1203** |
| y_zoom | **257** → 341 | **1278** → 1303 | **1316** → 1487 |

Because the clusters were too coarse, white noise barely improved: about 9 y-groups per window are still too tall to fit the threshold.

## Experiment 2 — Fixed bracketing (v1, current)

After an overshoot, the pass is now bisected over `(0, threshold)` from the *current* state, where a zero threshold merges nothing. If no single pass lands in `[0.95·target, target]` because the count jumps across it, the largest still-above-budget pass is accepted and the passes continue with finer steps. Storage is now 25.1 MiB (+0.8% vs the x-tree's 24.9 MiB, and 3 objects instead of 4).

1M points, broadband, 4 MiB budget, `vs_xtree --nmae`. Cells read `wire KiB / mean latency ms / mean density NMAE`; the lower wire in each row is bold.

### `medium` (random walk)

| Workload | x-tree | Salmond tree |
| --- | ---: | ---: |
| cold_overview | **2013** / 186.0 / 0.24% | 2233 / 185.0 / 0.31% |
| progressive_zoom_in | **2045** / 72.8 / 0.21% | 2273 / 67.5 / 0.28% |
| lateral_pan | **364** / 91.6 / 0.19% | 460 / 98.8 / 0.26% |
| jump_and_revisit | **428** / 88.3 / 0.18% | 631 / 87.2 / 0.26% |
| rapid_interrupt_pan | **107** / 140.0 / 0.16% | 161 / 132.0 / 0.30% |
| y_zoom | **257** / 29.3 / 0.11% | 305 / 43.8 / 0.19% |

### `medium_white_noise` (NoisySpikes)

| Workload | x-tree | Salmond tree |
| --- | ---: | ---: |
| cold_overview | 25454 / 1489.0 / 0.08% | **2243** / 218.0 / 0.12% |
| progressive_zoom_in | 43383 / 464.5 / 0.30% | **3586** / 127.7 / 0.28% |
| lateral_pan | **4397** / 126.9 / 0.18% | 4687 / 167.5 / 0.18% |
| jump_and_revisit | **6413** / 182.0 / 0.17% | 7433 / 257.3 / 0.17% |
| rapid_interrupt_pan | **1279** / 175.0 / 0.15% | 1406 / 262.0 / 0.16% |
| y_zoom | **1278** / 29.0 / 0.14% | 1411 / 58.7 / 0.38% |

### `medium_bursty` (BurstyX)

| Workload | x-tree | Salmond tree |
| --- | ---: | ---: |
| cold_overview | **2091** / 683.0 / 0.06% | 2145 / 179.0 / 0.09% |
| progressive_zoom_in | 5194 / 339.5 / 0.41% | **2728** / 84.7 / 0.54% |
| lateral_pan | 5211 / 120.5 / 0.15% | **792** / 88.4 / 0.23% |
| jump_and_revisit | 6179 / 168.2 / 0.07% | **1207** / 136.5 / 0.18% |
| rapid_interrupt_pan | 1311 / 242.0 / 0.07% | **350** / 245.0 / 0.13% |
| y_zoom | **1316** / 30.0 / 0.16% | 1495 / 131.7 / 0.18% |

Build time is 0.33 s (random walk and bursty) and 0.92 s (white noise), against 0.08–0.11 s for the x-tree.

## Findings

1. **The idea works where x order is the wrong grouping.**
   - **White noise:** 11× less wire on cold overview (25.4 → 2.2 MiB) and 12× on zoom-in, with 4–7× lower latency. The clusters are short in y, so they fit the threshold where x-runs of 32 noise points never do.
   - **Bursty:** 4–7× less wire on pans and revisits. Windows of a fixed point count adapt to the density, and clusters don't straddle the empty stretches.
   - **Quality:** NMAE stays ≤ 0.54% everywhere.
2. **Smooth signals pay a small tax.** On the random walk, x-neighbours are already y-coherent, so clustering can't beat x-runs. Uneven, overlapping clusters straddle viewport edges more often: +11% wire on cold overview and +47% on `jump_and_revisit`.
3. **Windowed white-noise views are slightly worse (+7–16%).** This is the expected boundary cost: a cluster spans its whole 1024-point window in x, so at a viewport edge every cluster of that window straddles and descends.
4. **`y_zoom` latency is worse on every dataset (29 → 44–132 ms)** while wire is only 12–19% higher. Off-screen culling resolves whole x-runs in the x-tree, but Salmond clusters mix on- and off-screen y in one node, so more depths are walked. This needs a closer look.
5. **Coarser clusters helped the smooth signal.** v0, with about 9 clusters per window, beat the x-tree on random-walk cold overview and zoom-in (−20%). The cluster budget is a real knob, and the best value probably depends on the data.
6. **NMAE is 0.03–0.15 points higher than the x-tree.** Resolved nodes hold more points, since a node that fits sooner is drawn as one kernel. It's well within budget, but not free.

## Next Hypotheses

Superseded in part by the [mergeable summaries](./2026-11-23-mergeable-summaries.md) entry, which generalizes what a node stores. The hypotheses as written at the time:


- **Compare with the real competitors.** The k-d tree, ripmap v2 and sparse grid already fix white noise. Run `heavy_white_noise`, `heavy_bursty` and the σ = 4 px variants against them before drawing conclusions about the Salmond tree's standing.
- **Ratio and window sweep:** `ratio ∈ {8, 16, 32, 64}` × `window ∈ {256, 1024, 4096}`. Smaller windows cut the boundary cost (finding 3); a larger ratio may recover the random walk (finding 5).
- **Clustering scale per level:** cluster in the units of the zoom where each level is actually resolved, instead of overview units for every level.
- **Runnalls instead of Salmond:** order-independent, KL-optimal pair merges, at `O(n²)` per window with a heap.
- **Investigate `y_zoom`** (finding 4).
