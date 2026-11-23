# 2026-11-22 — Head-to-Head: Ripmap v2 (`PLPRIPM2`) vs Sparse Grid, plus a σ = 4 px Probe

## Context & Goal

Two entries from the same day each compared themselves only against ripmap **v1**: the [sparse grid](./2026-11-22-sparse-grid.md) and [ripmap v2](./2026-11-22-ripmap-v2-sparse-planes.md). The ripmap v2 entry lists a sparse grid head-to-head as its first next hypothesis. Both strategies are now on `master` (merge `96af706`), so this entry runs them side by side.

A note on numbering: both entries call their benchmark section "Experiment 6". They are left unedited. Here, "the grid entry" and "the ripmap v2 entry" refer to them by name.

A second question came from reviewing the v2 code. Several of its storage and traversal caps are fixed constants that look sized for the benchmark's 16 px kernel on a 1920 × 1080 screen (about 120σ × 67σ):
- `MAX_STORED_K_COARSE = 8` and `MAX_STORED_K_WINDOWED = 9`;
- no planes at level 0;
- the 16σ span threshold of the boundary fast path.

So both strategies were also run at **σ = 4 px**, where a view spans 4× more kernel widths on each axis.

## Results — standard scenarios (10M, σ = 16 px)

`export_reports --only heavy --only heavy_white_noise --only medium_white_noise` on `master` at `96af706`. Network: broadband (30 ms, 50 Mbps, 6 concurrent requests), 16 MiB budget. The v2 numbers reproduce the v2 entry exactly. Cells are `wire / requests / mean latency`; the lower wire in each row is bold.

### Storage & build (10M)

| Strategy | Storage (random walk / white noise) | Objects | Build |
| --- | ---: | ---: | ---: |
| `moment_ripmap` v2 | **`307.86 / 296.95 MiB`** (1.35× / 1.30× raw) | 4 | **2.0 / 1.5 s** |
| `sparse_grid` | `335.69 / 308.89 MiB` (1.47× / 1.35× raw) | 4 | 6.9 / 11.9 s |

### `Heavy` (random walk)

| Workload | ripmap v2 | sparse grid |
| --- | ---: | ---: |
| `cold_overview` | **68.2 KiB / 3 / 36 ms** | 126.4 KiB / 5 / 78 ms |
| `progressive_zoom_in` | **650.9 KiB / 44 / 101 ms** | 835.3 KiB / 64 / 102 ms |
| `lateral_pan` | **406.3 KiB / 52 / 101 ms** | 576.6 KiB / 104 / 98 ms |
| `jump_and_revisit` | **422.8 KiB / 37 / 84 ms** | 574.6 KiB / 55 / 77 ms |
| `rapid_interrupt_pan` | 131.8 KiB / 10 / 106 ms | **127.6 KiB / 16 / 101 ms** |
| `y_zoom` | 587.5 KiB / 39 / 80 ms | **476.1 KiB / 21 / 51 ms** |

### `HeavyWhiteNoise`

| Workload | ripmap v2 | sparse grid |
| --- | ---: | ---: |
| `cold_overview` | **247.1 KiB / 2 / 65 ms** | 629.2 KiB / 5 / 160 ms |
| `progressive_zoom_in` | **1.10 MiB / 68 / 126 ms** | 4.07 MiB / 63 / 145 ms |
| `lateral_pan` | **749.0 KiB / 101 / 106 ms** | 1.48 MiB / 94 / 114 ms |
| `jump_and_revisit` | **712.4 KiB / 52 / 77 ms** | 1.52 MiB / 48 / 95 ms |
| `rapid_interrupt_pan` | **231.2 KiB / 13 / 115 ms** | 384.6 KiB / 14 / 142 ms |
| `y_zoom` | 13.38 MiB / 84 / 181 ms | **11.80 MiB / 32 / 111 ms** |

`MediumWhiteNoise` (1M): ripmap v2 reads less on the overview (279 vs 459 KiB) and on the zoom-in (2.14 vs 2.21 MiB). The grid reads less on the windowed views (`lateral_pan` 4.01 vs 4.43 MiB, `jump_and_revisit` 5.40 vs 6.77 MiB). At this size both fall back to raw points on 5% windows.

## Results — σ = 4 px probe (throwaway test, not committed)

Setup: the standard workloads rebuilt with `sigma_px = 4` (`Workload::standard_suite(1920, 1080, 4.0, 1.0, y_center)`), same network and budgets. The test was deleted after the run.

- **1M (`Medium` sine / random walk, `MediumWhiteNoise`):** neither strategy can serve this resolution from planes. Both read near-raw amounts: white-noise overview 26.2 MiB (ripmap v2) and 23.7 MiB (grid) against 22.9 MiB raw. Windowed views read the same raw x-windows (≈ 4.1–4.4 MiB `lateral_pan`) for both strategies and both signal families. The 1M probe does not discriminate between them.
- **10M `HeavyWhiteNoise`:**

| Workload | ripmap v2 | sparse grid |
| --- | ---: | ---: |
| `cold_overview` | 9.38 MiB / 41 / 744 ms | **6.13 MiB / 10 / 257 ms** |
| `progressive_zoom_in` | **22.29 MiB / 126 / 302 ms** | 69.26 MiB / 119 / 494 ms |
| `lateral_pan` | 44.17 MiB / 252 / 462 ms | **40.24 MiB / 100 / 241 ms** |
| `jump_and_revisit` | 76.81 MiB / 415 / 872 ms | **69.79 MiB / 123 / 521 ms** |
| `rapid_interrupt_pan` | 12.84 MiB / 70 / 873 ms | **11.63 MiB / 22 / 516 ms** |
| `y_zoom` | 12.83 MiB / 70 / 145 ms | **11.63 MiB / 22 / 86 ms** |

A 4× smaller σ needs ~16× more kernels for a dense view.
- **Overview:** ripmap v2 grows 39× (247 KiB → 9.38 MiB), the grid 10× (629 KiB → 6.13 MiB). This is consistent with v2's `k ≤ 8` cap on coarse blocks forcing expansion once views need finer bins, but the cap's effect was not isolated.
- **Windowed views:** both strategies collapse to near-raw reads (40–77 MiB).
- **Zoom-in:** the grid is 3× worse (69 vs 22 MiB). Its global 16-points-per-cell cut-off drops the fine planes this zoom needs.

## Findings

1. **Ripmap v2 now leads at the benchmark's resolution, replacing the sparse grid's claim.** The grid entry claimed 1.9–2.1× less wire than the ripmap on white-noise windowed views; that was against v1. Against v2 the grid is behind on 9 of the 12 10M workloads. On white noise it reads 1.66–3.7× more on every view but `y_zoom`. On random walk it reads 1.28–1.85× more on the overview, zoom-in, pan and jump. Latency is close on random walk and favours v2 on white noise. The grid still wins `y_zoom` on both families (0.81× / 0.88× the wire, ~0.62× the latency) and narrowly wins random-walk `rapid_interrupt_pan` (127.6 vs 131.8 KiB).
2. **Most of v2's lead comes from parts the grid can adopt:**
   - 16-byte quantized cells against the grid's 28 (24-byte moments + `u32` row), 1.75× less per cell;
   - greedy merging of plane cells into fewer kernels;
   - a manifest read in one EOF-clamped request.

   The structural difference is where columns come from: v2's x-hierarchy builds blocks from point counts, while the grid uses fixed geometric columns. That difference was not isolated.
3. **Neither design is robust to a smaller kernel yet.** At σ = 4 px both degrade to raw on windowed views. v2's overview grows faster than kernel count (39× against the 16× expected), plausibly from its fixed `k` caps, and the grid's zoom-in suffers from its global density cut-off. The benchmark only runs σ = 16 px, so constants tuned on it are not tested elsewhere.

## Next Hypotheses

1. Add a second kernel size (e.g. σ = 4 px at 10M) to the standard scenarios, so caps tuned to σ = 16 show up in the report.
2. Make v2's `k` caps a function of `block_points` and data density only (or a builder parameter) instead of the benchmark's screen, and re-measure both σ.
3. Port 16-byte quantized cells, cell merging and the one-request manifest to the sparse grid, to see what remains of the gap once cell format is equal.
