# 2026-11-22 — Sparse Grid Phase C (16-Byte Cells, Budgeted Column Gate, Cell Merging) and the Final Head-to-Head

## Context & Goal

This is the grid half of Phase C in the plan agreed between Claude and Antigravity on the agent board and approved by the user. The ripmap half and Phase B are in [Ripmap v2 Hardening](./2026-11-22-ripmap-v2-data-driven-caps-and-bursty-x.md); the scenarios come from [Phase A](./2026-11-22-scenarios-small-kernel-and-bursty-x.md).

- **Goal:** port the sparse grid to the shared 16-byte cells, merge cells into fewer kernels, load the manifest in one request, and replace its global density cutoff with a per-column one.
- **Then:** run the agreed final comparison on every scenario and apply the decision rule. One design is retired to a baseline only if the other is within ~5% wire and latency everywhere and no worse on storage or build time.

## Changes (`progressive-grid`, format version 2)

- **Cells:** the shared `SparseCell` (16 B: `u16` row + 14-byte `QuantizedMoments`) replaces `row: u32` + 24-byte `MomentCell` (28 B). Rows are capped at `2^16` (`MAX_ROW_LEVEL = 16`); the deepest row level stored at 10M is 14.
- **Manifest:** loaded with `read_small_object` (1 request) instead of `object_len` + read.
- **Client:**
  - Plane choice is per column: a column that is not in its plane (empty run in the CSC pointers) goes raw.
  - Cells merge greedily in row order (`push_group`).
  - With explicit `y_bounds`, far off-screen cells collapse into one kernel above and one below, as in the ripmap.
- **Builder:** every level `(i, i - d)` of every aspect is built directly from the points; 2×2 coarsening and the global search are gone. Which columns a plane keeps is decided by a **budgeted gate**:
  - **Floor:** a column qualifies when its cells average ≥ `min_column_points_per_cell = 4` points.
  - **Budget:** qualifying columns are admitted densest first while the plane stays within `N / min_points_per_cell = N / 16` cells.
  - On evenly dense data this keeps whole planes exactly when the old global rule did. On uneven data the densest columns still get fine rows.

### How the gate was chosen (1M scenarios, wire per workload)

The first Phase C attempt mirrored the ripmap's per-block gate: a column needed `≤ p/16` non-empty rows and an own-extent span `≤ 4·p/16` rows. It cut 10M storage to 291 / 279 MiB, but regressed random-walk views. On `constrained_mobile` they rose from 407 KiB to 1.30 MiB (`lateral_pan`), 421 KiB to 1.73 MiB (`jump_and_revisit`), and 111 to 561 KiB (`rapid_interrupt_pan`). The old global average had let moderately sparse columns ride along with dense ones.

A throwaway probe (deleted after use) then measured two knobs:
- **Span limit (`2^extra` × the dense row count, extra ∈ {2, 4, 6, 16}):** no effect on `constrained_mobile`, `medium_bursty` or `medium` (the sine wave). On `medium_white_noise`, extra ≥ 4 cut `progressive_zoom_in` from 2.74 to 1.88 MiB. The span limit was dropped.
- **Budgeted gate, per-column floor ∈ {2, 4, 6, 16}:** floors 2–6 recover and beat the old grid. With floor 4 vs 16:
  - `constrained_mobile`: `lateral_pan` 300 vs 1,303 KiB, `jump_and_revisit` 294 vs 1,733 KiB.
  - `medium_white_noise`: `lateral_pan` 2.99 vs 4.11 MiB.
  - Storage is 2–6 MiB higher at 1M.

  Floor 2 is better again on `medium_bursty` `jump_and_revisit` (443 vs 832 KiB) and stores ~1–2 MiB more. The default is 4.

## Verification

- `cargo test -p pink-lemonade-progressive-grid`: unit tests including the budget gate, plus conformance at 4K, at 200K, and with 8-point columns on a single aspect. Density NMAE ≤ 5% on both signal families. Clippy `-D warnings` and the `wasm32-unknown-unknown` check pass.
- The explorer model (`grid-model.ts`) mirrors the gate and the merging; `vitest` passes 14/14. Checked in the browser: on 300K white noise the overview is 34.8 KiB (was 47.5), and Y zoom 4× now uses plane (11, 7) for dense columns with 59 raw ones (was all raw).
- **Full `export_reports` on `master`** (with `ripmap-phase-c` merged):
  - Every `paged_moment_tree`, `moment_kdtree`, `moment_quadtree` and naive row matches the Phase A baseline exactly (210 rows, 0 diffs in wire and requests).
  - Every ripmap row matches a run before this gate change (60 / 60).
  - The run is the new `benchmarks/results/latest.json`.

---

## Final Head-to-Head — ripmap v2 (Phase C) vs sparse grid (Phase C)

Cells are `wire / requests / mean latency`. Ratios are grid ÷ ripmap, so below 1 means the grid is cheaper or faster.

#### heavy  storage ripmap 324.01 / grid 348.13 MiB; build 2.0 / 10.5 s
| Workload | ripmap v2 | sparse grid | grid/ripmap wire | latency |
| --- | ---: | ---: | ---: | ---: |
| `cold_overview` | 73.7 KiB / 3 / 36 ms | 102.8 KiB / 4 / 74 ms | 1.40 | 2.06 |
| `progressive_zoom_in` | 657.8 KiB / 44 / 101 ms | 728.8 KiB / 63 / 100 ms | 1.11 | 0.99 |
| `lateral_pan` | 411.8 KiB / 52 / 101 ms | 467.0 KiB / 103 / 96 ms | 1.13 | 0.95 |
| `jump_and_revisit` | 428.3 KiB / 37 / 84 ms | 459.7 KiB / 54 / 75 ms | 1.07 | 0.90 |
| `rapid_interrupt_pan` | 137.2 KiB / 10 / 106 ms | 107.2 KiB / 15 / 97 ms | 0.78 | 0.92 |
| `y_zoom` | 592.9 KiB / 39 / 80 ms | 330.7 KiB / 18 / 48 ms | 0.56 | 0.60 |

#### heavy_white_noise  storage ripmap 316.44 / grid 345.90 MiB; build 1.4 / 8.1 s
| Workload | ripmap v2 | sparse grid | grid/ripmap wire | latency |
| --- | ---: | ---: | ---: | ---: |
| `cold_overview` | 252.6 KiB / 2 / 65 ms | 390.0 KiB / 4 / 121 ms | 1.54 | 1.86 |
| `progressive_zoom_in` | 1.11 MiB / 68 / 126 ms | 1.95 MiB / 149 / 190 ms | 1.76 | 1.51 |
| `lateral_pan` | 754.5 KiB / 101 / 106 ms | 935.3 KiB / 106 / 105 ms | 1.24 | 1.00 |
| `jump_and_revisit` | 717.8 KiB / 52 / 77 ms | 1005.2 KiB / 52 / 81 ms | 1.40 | 1.05 |
| `rapid_interrupt_pan` | 236.6 KiB / 13 / 115 ms | 257.1 KiB / 13 / 121 ms | 1.09 | 1.05 |
| `y_zoom` | 13.39 MiB / 84 / 180 ms | 12.15 MiB / 130 / 190 ms | 0.91 | 1.05 |

#### heavy_sigma4  storage ripmap 324.01 / grid 348.13 MiB; build 2.0 / 10.3 s
| Workload | ripmap v2 | sparse grid | grid/ripmap wire | latency |
| --- | ---: | ---: | ---: | ---: |
| `cold_overview` | 393.2 KiB / 10 / 116 ms | 507.5 KiB / 4 / 128 ms | 1.29 | 1.10 |
| `progressive_zoom_in` | 4.18 MiB / 69 / 162 ms | 3.60 MiB / 114 / 166 ms | 0.86 | 1.02 |
| `lateral_pan` | 2.07 MiB / 71 / 135 ms | 1.95 MiB / 69 / 102 ms | 0.94 | 0.76 |
| `jump_and_revisit` | 2.35 MiB / 55 / 142 ms | 3.45 MiB / 209 / 237 ms | 1.47 | 1.67 |
| `rapid_interrupt_pan` | 628.9 KiB / 11 / 188 ms | 467.2 KiB / 11 / 123 ms | 0.74 | 0.65 |
| `y_zoom` | 4.78 MiB / 126 / 188 ms | 2.94 MiB / 124 / 145 ms | 0.61 | 0.77 |

#### heavy_white_noise_sigma4  storage ripmap 316.44 / grid 345.90 MiB; build 1.4 / 8.8 s
| Workload | ripmap v2 | sparse grid | grid/ripmap wire | latency |
| --- | ---: | ---: | ---: | ---: |
| `cold_overview` | 2.28 MiB / 6 / 252 ms | 3.63 MiB / 7 / 259 ms | 1.59 | 1.03 |
| `progressive_zoom_in` | 16.98 MiB / 98 / 250 ms | 31.71 MiB / 415 / 577 ms | 1.87 | 2.31 |
| `lateral_pan` | 44.18 MiB / 252 / 464 ms | 40.27 MiB / 99 / 241 ms | 0.91 | 0.52 |
| `jump_and_revisit` | 76.82 MiB / 415 / 874 ms | 69.85 MiB / 122 / 522 ms | 0.91 | 0.60 |
| `rapid_interrupt_pan` | 12.84 MiB / 70 / 875 ms | 11.64 MiB / 21 / 517 ms | 0.91 | 0.59 |
| `y_zoom` | 12.84 MiB / 70 / 146 ms | 11.64 MiB / 21 / 86 ms | 0.91 | 0.59 |

#### heavy_bursty  storage ripmap 323.98 / grid 340.35 MiB; build 1.7 / 10.0 s
| Workload | ripmap v2 | sparse grid | grid/ripmap wire | latency |
| --- | ---: | ---: | ---: | ---: |
| `cold_overview` | 229.3 KiB / 22 / 194 ms | 95.9 KiB / 4 / 73 ms | 0.42 | 0.38 |
| `progressive_zoom_in` | 769.0 KiB / 67 / 133 ms | 787.6 KiB / 57 / 90 ms | 1.02 | 0.68 |
| `lateral_pan` | 314.4 KiB / 51 / 116 ms | 454.6 KiB / 86 / 92 ms | 1.45 | 0.79 |
| `jump_and_revisit` | 273.4 KiB / 59 / 158 ms | 494.9 KiB / 54 / 70 ms | 1.81 | 0.45 |
| `rapid_interrupt_pan` | 93.1 KiB / 14 / 221 ms | 129.5 KiB / 14 / 98 ms | 1.39 | 0.44 |
| `y_zoom` | 586.1 KiB / 27 / 66 ms | 586.9 KiB / 23 / 50 ms | 1.00 | 0.77 |

#### medium  storage ripmap 29.61 / grid 35.20 MiB; build 0.1 / 0.8 s
| Workload | ripmap v2 | sparse grid | grid/ripmap wire | latency |
| --- | ---: | ---: | ---: | ---: |
| `cold_overview` | 295.0 KiB / 2 / 71 ms | 267.1 KiB / 4 / 101 ms | 0.91 | 1.42 |
| `progressive_zoom_in` | 491.1 KiB / 29 / 84 ms | 721.4 KiB / 25 / 64 ms | 1.47 | 0.77 |
| `lateral_pan` | 210.4 KiB / 47 / 95 ms | 233.2 KiB / 32 / 64 ms | 1.11 | 0.68 |
| `jump_and_revisit` | 233.9 KiB / 25 / 66 ms | 211.5 KiB / 16 / 45 ms | 0.90 | 0.69 |
| `rapid_interrupt_pan` | 96.4 KiB / 7 / 99 ms | 56.5 KiB / 5 / 68 ms | 0.59 | 0.69 |
| `y_zoom` | 310.7 KiB / 26 / 45 ms | 273.6 KiB / 16 / 23 ms | 0.88 | 0.51 |

#### medium_white_noise  storage ripmap 29.34 / grid 33.72 MiB; build 0.1 / 0.7 s
| Workload | ripmap v2 | sparse grid | grid/ripmap wire | latency |
| --- | ---: | ---: | ---: | ---: |
| `cold_overview` | 279.3 KiB / 2 / 68 ms | 293.6 KiB / 4 / 105 ms | 1.05 | 1.54 |
| `progressive_zoom_in` | 2.14 MiB / 55 / 98 ms | 1.81 MiB / 42 / 92 ms | 0.85 | 0.93 |
| `lateral_pan` | 4.43 MiB / 45 / 136 ms | 2.92 MiB / 113 / 126 ms | 0.66 | 0.92 |
| `jump_and_revisit` | 6.77 MiB / 50 / 173 ms | 3.93 MiB / 80 / 122 ms | 0.58 | 0.70 |
| `rapid_interrupt_pan` | 1.32 MiB / 10 / 179 ms | 1.10 MiB / 13 / 155 ms | 0.83 | 0.87 |
| `y_zoom` | 1.32 MiB / 10 / 30 ms | 1.42 MiB / 34 / 50 ms | 1.08 | 1.66 |

#### medium_bursty  storage ripmap 29.59 / grid 30.55 MiB; build 0.1 / 0.7 s
| Workload | ripmap v2 | sparse grid | grid/ripmap wire | latency |
| --- | ---: | ---: | ---: | ---: |
| `cold_overview` | 309.9 KiB / 30 / 206 ms | 110.8 KiB / 4 / 75 ms | 0.36 | 0.36 |
| `progressive_zoom_in` | 608.9 KiB / 68 / 122 ms | 370.1 KiB / 23 / 54 ms | 0.61 | 0.44 |
| `lateral_pan` | 567.1 KiB / 27 / 66 ms | 284.6 KiB / 24 / 62 ms | 0.50 | 0.94 |
| `jump_and_revisit` | 744.2 KiB / 38 / 115 ms | 831.6 KiB / 24 / 51 ms | 1.12 | 0.45 |
| `rapid_interrupt_pan` | 240.9 KiB / 11 / 212 ms | 88.3 KiB / 5 / 73 ms | 0.37 | 0.34 |
| `y_zoom` | 1.12 MiB / 45 / 85 ms | 1.43 MiB / 18 / 40 ms | 1.28 | 0.47 |

#### constrained_mobile  storage ripmap 29.59 / grid 31.35 MiB; build 0.2 / 0.8 s
| Workload | ripmap v2 | sparse grid | grid/ripmap wire | latency |
| --- | ---: | ---: | ---: | ---: |
| `cold_overview` | 111.8 KiB / 3 / 200 ms | 115.8 KiB / 4 / 351 ms | 1.04 | 1.75 |
| `progressive_zoom_in` | 545.2 KiB / 24 / 300 ms | 570.4 KiB / 27 / 290 ms | 1.05 | 0.97 |
| `lateral_pan` | 520.6 KiB / 68 / 408 ms | 300.3 KiB / 32 / 245 ms | 0.58 | 0.60 |
| `jump_and_revisit` | 907.3 KiB / 74 / 466 ms | 294.3 KiB / 16 / 203 ms | 0.32 | 0.44 |
| `rapid_interrupt_pan` | 155.7 KiB / 10 / 459 ms | 78.2 KiB / 5 / 308 ms | 0.50 | 0.67 |
| `y_zoom` | 1.19 MiB / 71 / 371 ms | 1.05 MiB / 49 / 245 ms | 0.88 | 0.66 |

#### light  storage ripmap 0.25 / grid 0.29 MiB; build 0.0 / 0.0 s
| Workload | ripmap v2 | sparse grid | grid/ripmap wire | latency |
| --- | ---: | ---: | ---: | ---: |
| `cold_overview` | 254.0 KiB / 3 / 62 ms | 240.0 KiB / 3 / 100 ms | 0.94 | 1.61 |
| `progressive_zoom_in` | 254.0 KiB / 3 / 10 ms | 240.0 KiB / 3 / 17 ms | 0.94 | 1.61 |
| `lateral_pan` | 60.9 KiB / 11 / 39 ms | 46.4 KiB / 11 / 39 ms | 0.76 | 1.01 |
| `jump_and_revisit` | 69.9 KiB / 5 / 22 ms | 66.5 KiB / 9 / 43 ms | 0.95 | 1.95 |
| `rapid_interrupt_pan` | 32.4 KiB / 3 / 64 ms | 15.8 KiB / 3 / 64 ms | 0.49 | 1.00 |
| `y_zoom` | 31.6 KiB / 2 / 6 ms | 15.8 KiB / 3 / 11 ms | 0.50 | 1.94 |

---

## Findings

1. **Neither design dominates, so the agreed rule keeps both.**
   - The grid is more than 5% worse than the ripmap on wire or latency in 28 workload rows; the ripmap is more than 5% worse than the grid in 45.
   - Wire wins over the 60 rows: ripmap 26, grid 34. Over the 30 rows at 10M: ripmap 18, grid 12.
   - The ripmap is smaller and builds 5–7× faster: 10M storage 316–324 vs 340–348 MiB, build 1.4–2.0 vs 8.1–10.5 s.
2. **Where the ripmap wins.**
   - 10M white noise at σ = 16, 5 of 6 views: grid wire 1.09–1.76×, e.g. `progressive_zoom_in` 1.95 vs 1.11 MiB.
   - 10M `BurstyX` windowed views: grid wire 1.39–1.81× (`jump_and_revisit` 495 vs 273 KiB).
   - Overviews at 10M: 1.29–1.59× more for the grid on random walk, white noise, and both σ = 4 px scenarios.
   - Count-based blocks with per-block bins fit the dense white-noise band more tightly than global power-of-two rows.
3. **Where the grid wins.**
   - Seven of ten `y_zoom` views: 0.56× on 10M random walk, 0.61× at σ = 4 px. The exceptions are `heavy_bursty` (a tie), `medium_white_noise` (1.08×) and `medium_bursty` (1.28×).
   - The random-walk views at σ = 4 px other than the overview and jump (0.61–0.94× wire).
   - The near-raw white-noise σ = 4 px windowed views: 0.91× wire, 0.52–0.60× latency.
   - Most 1M views: `constrained_mobile` `jump_and_revisit` 0.32×, `medium_white_noise` `jump_and_revisit` 0.58×.
   - The `BurstyX` overview: 0.42× at 10M, 0.36× at 1M.
   - Latency is usually lower: two direct-addressed rounds and fewer requests.
4. **The grid's remaining weak spots:**
   - Build time.
   - 10M white-noise zoom-in.
   - `heavy_sigma4` `jump_and_revisit` (1.47× wire, 1.67× latency).
   - The white-noise σ = 4 px zoom-in (1.87×), where global rows are much finer than needed in sparse columns.
5. **This entry corrects two earlier claims:**
   - The first Phase C attempt (above) shows that mirroring the ripmap's per-block gate is not enough for global rows. This qualifies the board's expectation that a per-column cutoff would simply fix the grid's random-walk views.
   - The [ripmap v2 vs sparse grid](./2026-11-22-ripmap-v2-vs-sparse-grid.md) entry reported v2 ahead on 9 of 12 10M workloads at σ = 16. After both Phase C changes, the 10M split across all five 10M scenarios is ripmap 18, grid 12.

## Next Hypotheses

1. **Grid:** rows relative to each column's own y-extent (like ripmap bins), while keeping direct addressing. This targets the white-noise and overview gaps.
2. **Grid:** a faster build from one sort per x-level shared by all aspects (today each aspect re-scans the points per level).
3. **Ripmap:** y-subrange reads for deep `y_zoom`, its consistent loss here.
4. **Both:** keep the σ = 4 px and `BurstyX` scenarios in every comparison. Both caught design flaws that the 16 px uniform-x scenarios hid.
