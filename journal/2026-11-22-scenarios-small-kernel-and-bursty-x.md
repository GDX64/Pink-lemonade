# 2026-11-22 — New Benchmark Scenarios: σ = 4 px Kernels and `BurstyX` (Phase A)

## Context & Goal

This is Phase A of the plan agreed between Claude and Antigravity on the agent board and approved by the user. See [ripmap v2 vs sparse grid](./2026-11-22-ripmap-v2-vs-sparse-grid.md) for the motivation. The standard scenarios only render with a 16 px kernel on uniformly spaced x, so constants tuned to that setting were invisible:
- ripmap v2's `k ≤ 8 / 9` caps;
- whether count-based x-blocks (ripmap, x-tree) or geometric columns (sparse grid) cope better with uneven x.

Phase A adds both stresses to the standard report before either strategy is changed (Phases B and C).

## Infrastructure Changes

- **`DatasetShape::BurstyX { seed }`** (`progressive-harness/src/fixture.rs`):
  - x: 90% of the points sit in ten bursts, each 1% of x wide and centred on `0.05 + 0.1k`. Three 7%-wide stretches (`[0.165, 0.235]`, `[0.465, 0.535]`, `[0.765, 0.835]`) have no points. The remaining 10% of points are spread uniformly over the rest.
  - y: a Brownian random walk with unit weights.
  - x comes from a monotone piecewise-linear inverse CDF, so it is sorted by construction. A unit test checks sortedness, the 90% burst fraction (±0.5%), and the empty gaps.
  - The existing `MultiRegime` shape was not usable for this: its x density varies only by ±20% (`dx/dt ∈ [0.8, 1.2]`).
- **New `BenchmarkScenario`s** (`progressive-harness/src/benchmarks.rs`):

  | Scenario | Data | Kernel | Budget | Run in |
  | --- | --- | --- | --- | --- |
  | `HeavySmallKernel` (`heavy_sigma4`) | 10M random walk | σ = 4 px | 16 MiB | full |
  | `HeavyWhiteNoiseSmallKernel` (`heavy_white_noise_sigma4`) | 10M noisy spikes | σ = 4 px | 16 MiB | full |
  | `HeavyBursty` (`heavy_bursty`) | 10M `BurstyX` | σ = 16 px | 16 MiB | full |
  | `MediumBursty` (`medium_bursty`) | 1M `BurstyX` | σ = 16 px | 4 MiB | `--fast` |

  A σ = 4 px kernel on 1920 × 1080 makes the screen 480σ × 270σ, so a dense view needs ~16× more kernels.
- **`export_reports`:** `MediumBursty` runs in `--fast`; the three 10M scenarios run in the full set (non-naive strategies).
- **Regression check:** the regenerated `benchmarks/results/latest.json` has 55 reports. Its 96 workload rows from `heavy`, `heavy_white_noise` and `medium_white_noise` match the previous run on `96af706` exactly in wire bytes and request counts.

## Results — new scenarios (broadband 30 ms / 50 Mbps, cold cache)

Cells are `wire / requests / mean latency`. The lowest wire in each row is bold; a dash means not run.

### `heavy_sigma4` (10M random walk, σ = 4 px)

| Workload | x-tree | k-d tree | ripmap v2 | quadtree | sparse grid |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cold_overview` | 19.60 MiB / 83 / 1,053 ms | 5.01 MiB / 714 / 3,917 ms | **380.7 KiB / 9 / 114 ms** | 1.66 MiB / 17 / 273 ms | 670.3 KiB / 5 / 155 ms |
| `progressive_zoom_in` | 47.32 MiB / 2,721 / 2,588 ms | 21.05 MiB / 2,543 / 2,314 ms | **4.16 MiB / 68 / 162 ms** | 12.83 MiB / 506 / 639 ms | 4.53 MiB / 43 / 129 ms |
| `lateral_pan` | 56.79 MiB / 756 / 791 ms | 28.36 MiB / 2,157 / 1,591 ms | **2.09 MiB / 61 / 135 ms** | 16.81 MiB / 1,387 / 1,199 ms | 2.74 MiB / 66 / 119 ms |
| `jump_and_revisit` | 74.54 MiB / 415 / 846 ms | 43.22 MiB / 1,705 / 1,791 ms | **2.35 MiB / 55 / 142 ms** | 17.17 MiB / 1,565 / 1,596 ms | 13.38 MiB / 46 / 163 ms |
| `rapid_interrupt_pan` | 12.43 MiB / 71 / 853 ms | 8.68 MiB / 355 / 2,126 ms | 640.1 KiB / 7 / 188 ms | 4.14 MiB / 383 / 2,310 ms | **623.7 KiB / 11 / 148 ms** |
| `y_zoom` | 19.37 MiB / 221 / 317 ms | 12.90 MiB / 836 / 809 ms | **4.77 MiB / 126 / 188 ms** | 9.11 MiB / 583 / 686 ms | 5.12 MiB / 115 / 147 ms |

### `heavy_white_noise_sigma4` (10M noisy spikes, σ = 4 px)

| Workload | x-tree | k-d tree | ripmap v2 | quadtree | sparse grid |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cold_overview` | 248.57 MiB / 1,304 / 13,704 ms | 22.46 MiB / 98 / 1,254 ms | 9.38 MiB / 41 / 745 ms | 19.56 MiB / 83 / 1,202 ms | **6.13 MiB / 10 / 257 ms** |
| `progressive_zoom_in` | 435.02 MiB / 2,288 / 4,052 ms | 86.40 MiB / 811 / 1,153 ms | **22.29 MiB / 126 / 302 ms** | 101.28 MiB / 963 / 1,412 ms | 69.26 MiB / 119 / 494 ms |
| `lateral_pan` | 42.89 MiB / 247 / 420 ms | 45.37 MiB / 796 / 758 ms | 44.17 MiB / 252 / 464 ms | 86.32 MiB / 1,629 / 1,508 ms | **40.24 MiB / 100 / 241 ms** |
| `jump_and_revisit` | 74.60 MiB / 412 / 832 ms | 78.83 MiB / 823 / 1,215 ms | 76.81 MiB / 415 / 874 ms | 98.91 MiB / 1,045 / 1,529 ms | **69.79 MiB / 123 / 521 ms** |
| `rapid_interrupt_pan` | 12.44 MiB / 70 / 822 ms | 13.23 MiB / 129 / 1,152 ms | 12.84 MiB / 70 / 875 ms | 16.78 MiB / 172 / 1,543 ms | **11.63 MiB / 22 / 516 ms** |
| `y_zoom` | 12.44 MiB / 70 / 137 ms | 13.09 MiB / 158 / 220 ms | 12.83 MiB / 70 / 146 ms | 30.36 MiB / 514 / 728 ms | **11.63 MiB / 22 / 86 ms** |

### `heavy_bursty` (10M `BurstyX`, σ = 16 px)

| Workload | x-tree | k-d tree | ripmap v2 | quadtree | sparse grid |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cold_overview` | 717.7 KiB / 47 / 355 ms | 483.5 KiB / 73 / 447 ms | 25.71 MiB / 107 / 1,394 ms | 207.2 KiB / 5 / 86 ms | **114.2 KiB / 5 / 76 ms** |
| `progressive_zoom_in` | 4.90 MiB / 106 / 158 ms | 3.13 MiB / 459 / 450 ms | 31.85 MiB / 141 / 322 ms | 1.88 MiB / 97 / 229 ms | **599.0 KiB / 55 / 92 ms** |
| `lateral_pan` | 4.42 MiB / 413 / 348 ms | 4.22 MiB / 623 / 495 ms | 4.27 MiB / 84 / 132 ms | 2.32 MiB / 177 / 305 ms | **515.1 KiB / 87 / 93 ms** |
| `jump_and_revisit` | 5.32 MiB / 742 / 738 ms | 5.41 MiB / 800 / 826 ms | 1.81 MiB / 195 / 234 ms | 2.91 MiB / 204 / 382 ms | **549.3 KiB / 55 / 71 ms** |
| `rapid_interrupt_pan` | 1.10 MiB / 98 / 607 ms | 1.47 MiB / 203 / 1,166 ms | 220.9 KiB / 13 / 146 ms | 998.8 KiB / 50 / 299 ms | **144.8 KiB / 15 / 99 ms** |
| `y_zoom` | 11.14 MiB / 455 / 467 ms | 2.79 MiB / 354 / 366 ms | 2.03 MiB / 101 / 133 ms | **1.09 MiB / 81 / 186 ms** | 5.94 MiB / 35 / 81 ms |

Storage (10M `BurstyX`): x-tree 248.57, k-d tree 250.67, ripmap v2 307.86, quadtree 296.03, sparse grid 329.91 MiB.

### `medium_bursty` (1M `BurstyX`, σ = 16 px, 4 MiB budget)

| Workload | naive | x-tree | k-d tree | ripmap v2 | quadtree | sparse grid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cold_overview` | 22.89 MiB | 2.04 MiB | 855.6 KiB | 2.64 MiB | 310.6 KiB | **140.9 KiB** |
| `progressive_zoom_in` | 137.33 MiB | 5.07 MiB | 2.47 MiB | 5.67 MiB | 1.83 MiB | **436.1 KiB** |
| `lateral_pan` | 183.11 MiB | 5.09 MiB | 2.35 MiB | 5.13 MiB | 2.10 MiB | **294.7 KiB** |
| `jump_and_revisit` | 137.33 MiB | 6.03 MiB | 3.71 MiB | 6.21 MiB | 2.81 MiB | **548.4 KiB** |
| `rapid_interrupt_pan` | 22.89 MiB | 1.28 MiB | 989.6 KiB | 1.32 MiB | 699.6 KiB | **96.5 KiB** |
| `y_zoom` | 137.33 MiB | 1.29 MiB | 1.49 MiB | 1.33 MiB | 1.47 MiB | **1.30 MiB** |

---

## Findings

1. **`BurstyX` is the strongest separator so far, and it breaks ripmap v2.**
   - The v2 overview reads 25.71 MiB, 37× the x-tree's 717.7 KiB and 230× the grid's 114.2 KiB. That is about the background's 1M points (10% of 10M × 24 B ≈ 22.9 MiB), so v2 appears to read the sparse background raw. The step still renders only 652 kernels, and the client fills its 16 MiB budget.
   - Likely cause (not yet isolated): count-based blocks in the sparse background are ~9× wider than average. They exceed the 0.9σ plane width at every level that stores planes, so they descend to level 0, which stores none.
   - The x-tree, with the same resolve-or-expand rule but no level jump, reads far less, so the level jump and the boundary fast paths are suspects as well.
   - This is the opposite of the expectation in board Turn 2 (count-based blocks favoured on uneven x).
2. **The sparse grid wins 5 of 6 `BurstyX` views.** Against v2 it reads 1.5–54× less (`rapid_interrupt_pan` 1.5×, `jump_and_revisit` 3.4×, `lateral_pan` 8.5×, `progressive_zoom_in` 54×). Its geometric columns cost nothing in empty stretches and resolve sparse background columns whole. Its weak spot is `y_zoom` (5.94 MiB vs the quadtree's 1.09 MiB): the deep rows needed there are dropped by the global 16-points-per-cell cutoff (the same failure as on white noise).
3. **At σ = 4 px the result depends on the signal.**
   - **Random walk:** v2 wins 5 of 6 views. The grid's `jump_and_revisit` (13.38 MiB vs 2.35) is its global density cutoff again.
   - **White noise:** every strategy degrades to near-raw windowed views (40–99 MiB). The grid is lowest on 5 of 6 and fastest throughout.
   - The ripmap's `k` caps (Phase C) and the grid's global cutoff (Phase C) are the two fixes these numbers are meant to test.

## Next Hypotheses

1. **Phase C, ripmap (Antigravity):** besides the data-derived `k` caps, explain and fix the `BurstyX` overview. Candidate causes are the level jump estimating block width from the average density, no planes at level 0, and the boundary fast path.
2. **Phase C, grid (Claude):** a per-column density cutoff, expected to fix `BurstyX` `y_zoom`, random-walk σ = 4 px `jump_and_revisit`, and white-noise zoom-in.
3. Re-run all four new scenarios after Phase C and apply the agreed decision rule.
