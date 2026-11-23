# 2026-11-22 — Sparse Moment Grid Pyramid (shape-matched cells, two round trips)

## Context & Goal

The [classic moment quadtree](./2026-11-22-moment-quadtree.md) confirmed two weaknesses of tree descents: one round trip per depth, and square cells that over-refine in y when dense data is zoomed in x only (white-noise `lateral_pan` 61.82 MiB vs the ripmap's 2.87 MiB). This entry tests the idea saved before that experiment: a **sparse grid pyramid** where every cell is *direct-addressed* (no tree walk) and grids are stored at **several cell aspect ratios**, so the client picks cells whose on-screen shape fits the merge threshold.

Targets: white-noise windowed views near or below the ripmap, about two round trips per view at any zoom, and less storage than the ripmap's 1.78× raw.

## Infrastructure Changes

- **`pink-lemonade-progressive-grid`** (new crate, `PLPGRID1`), added to the workspace, `./dev clippy`, and `export_reports` (now 6 strategies; 5 on the 10M scenarios).
- No changes to `progressive-common`: the crate reuses `MomentCell`, `PackedNode`, `LruPageCache` blobs, `coalesce_runs` / `coalesce_spans`, and `assemble_frame` (cells are emitted through its `Plane` hook, with line extrema from the exact column node).

## Strategy Design (`progressive-grid`)

- **Geometry:** the data bbox is cut into `2^i` columns (x-level `i`) × `2^j` rows (y-level `j`). Coordinates are quantized once to 30-bit fixed point, so a point's column at level `i` is `qx >> (30 - i)`. Levels therefore nest exactly in builder and client, and cell normalization rectangles are recomputed bit for bit by the client.
- **Column pyramid:** x-levels `0..=I`, where `I` is the first level whose columns average ≤ `col_leaf = 256` points (`I = 16` at 10M). Every level stores `2^i + 1` column records (the last is a sentinel with totals). A record is the column's exact 64-byte `PackedNode` (tight bbox, moments, line extrema), its first point in `points.bin` (`u32`), and its first cell in **each plane of that level** (`u32` each). Those are the CSC column pointers of every aspect at once, so one read serves whichever plane the client picks.
- **Planes:** aspects `d = i - j ∈ {-2, 0, 2, …, 12}` (rows 4× finer than columns, up to columns 4096× finer than rows). For each aspect the builder finds the finest level `(i, j)` whose non-empty cells average ≥ `min_points_per_cell = 16` points. A pass aborts as soon as it exceeds `N / 16` cells, so failed passes are cheap. It then derives every coarser `(i-1, j-1)` of the aspect by merging 2×2 cells. Only non-empty cells are stored, column-major then by row: `row: u32` + 24-byte `MomentCell` = 28 B.
- **Layout (four objects):** `manifest.bin` (bbox, per-level record offset, plane table), `columns.bin` (all levels back to back), `cells.bin` (all planes back to back), `points.bin` (x-sorted, so any column at any level is one contiguous run).
- **Client (two rounds of reads):**
  1. **Candidate levels:** the coarsest level whose columns are ≤ 0.9σ wide on screen (`base`), and `base + 1`.
  2. **Round 1:** column records for the visible columns at both candidates, plus the *edge chains* of every finer level. Only the columns containing `min_x` / `max_x` and their siblings can straddle, so these are computed from the query alone. Records are cached in 8-record pages, coalesced with 16 KiB gaps.
  3. **Exact decomposition:** a column is *inside* when its exact node lies within `[min_x, max_x]`. A straddling column splits into its two children, down to the deepest level or ≤ `col_leaf` points, and those points are read raw and clipped.
  4. **Plane choice:** the inside columns give a lower bound of the visible y-range, so the y-factor estimate can only over-refine. For each candidate level, each inside column either resolves as one kernel (`standard_decision`) or uses the coarsest plane at its level with `w² + h² < t²`, otherwise raw. The candidate with the fewest **exact bytes** wins; the cell counts are known from the CSC pointers.
  5. **Round 2:** cells (one coalesced range per run of columns per plane) and raw edge points, concurrently. The exact `y_range` comes from the inside nodes plus the clipped raw points.
- **Verification:** `run_conformance_suite` at 4K, at 200K, and with `col_leaf = 8` and a single aspect (deep edge chains, frequent raw fallback). Density NMAE ≤ 5% on every standard workload for both signal families at 100K. Unit tests cover level nesting, manifest round trip, and 2×2 coarsening matching a direct build. Clippy `-D warnings` passes, and so does the `wasm32-unknown-unknown` check.

---

## Experiment 6 — Sparse grid vs x-tree, ripmap, and quadtree

The scenarios, network and budget match Experiment 5: 10M points (228.88 MiB raw), broadband with a 30 ms round trip, 50 Mbps and 6 concurrent requests, and a 16 MiB client budget. Numbers come from `export_reports --only heavy --only heavy_white_noise` and are deterministic under virtual time. Each cell reads `wire / requests / mean latency`; the lowest wire in each row is bold. The k-d tree ran too and is unchanged from Experiment 5.

### Storage & build (10M)

| Strategy | Storage (random walk / white noise) | Overhead vs raw | Objects | Build |
| --- | ---: | ---: | ---: | ---: |
| `paged_moment_tree` | `248.57 MiB` | `+8.6%` | 5 | 0.8–1.6 s |
| `moment_ripmap` | `406.99 MiB` | `+77.8%` | 44 | 2.4–2.9 s |
| `moment_quadtree` | `295.53 / 305.20 MiB` | `+29.1% / +33.3%` | **3** | 4.4–5.0 s |
| `sparse_grid` | `335.69 / 308.89 MiB` | `+46.7% / +35.0%` | 4 | 8.1 / 14.0 s |

Grid breakdown (from the ignored diagnostic test `storage_breakdown_at_10m`): `columns.bin` 11.20 / 9.48 MiB, `cells.bin` 95.60 / 70.53 MiB. On the random walk, cells stay sparse up to fine levels, so aspects keep levels down to `(12, 14)`. White noise stops earlier: the densest stored plane per aspect has ~220–270K cells.

### `Heavy` — integrated Gaussian noise (`random_walk`)

| Workload | `paged_moment_tree` | `moment_ripmap` | `moment_quadtree` | `sparse_grid` |
| --- | ---: | ---: | ---: | ---: |
| `cold_overview` | 629.81 KiB / 6 / 107 ms | **79.88 KiB / 8 / 35 ms** | 264.30 KiB / 4 / 98 ms | 126.42 KiB / 5 / 78 ms |
| `progressive_zoom_in` | 7.10 MiB / 122 / 179 ms | **768.26 KiB / 70 / 151 ms** | 3.46 MiB / 199 / 360 ms | 835.30 KiB / 64 / 102 ms |
| `lateral_pan` | 3.76 MiB / 314 / 316 ms | 630.63 KiB / 130 / 222 ms | 3.46 MiB / 218 / 380 ms | **576.58 KiB / 104 / 98 ms** |
| `jump_and_revisit` | 10.53 MiB / 1,676 / 1,562 ms | 646.42 KiB / 72 / 175 ms | 3.03 MiB / 121 / 283 ms | **574.56 KiB / 55 / 77 ms** |
| `rapid_interrupt_pan` | 1.17 MiB / 157 / 948 ms | 175.90 KiB / 22 / 258 ms | 886.27 KiB / 38 / 452 ms | **127.59 KiB / 16 / 101 ms** |
| `y_zoom` | 6.41 MiB / 491 / 483 ms | 762.12 KiB / 52 / 101 ms | 1.63 MiB / 108 / 257 ms | **476.14 KiB / 21 / 51 ms** |

### `HeavyWhiteNoise` — uncorrelated noise + spikes (`noisy_spikes`)

| Workload | `paged_moment_tree` | `moment_ripmap` | `moment_quadtree` | `sparse_grid` |
| --- | ---: | ---: | ---: | ---: |
| `cold_overview` | 248.57 MiB / 1,304 / 13,704 ms | 937.23 KiB / 5 / 181 ms | 1.52 MiB / 10 / 253 ms | **629.17 KiB / 5 / 160 ms** |
| `progressive_zoom_in` | 435.02 MiB / 2,288 / 4,052 ms | 4.21 MiB / 164 / 299 ms | 26.29 MiB / 709 / 786 ms | **4.07 MiB / 63 / 145 ms** |
| `lateral_pan` | 42.89 MiB / 247 / 420 ms | 2.87 MiB / 118 / 257 ms | 61.82 MiB / 1,741 / 1,477 ms | **1.48 MiB / 94 / 114 ms** |
| `jump_and_revisit` | 74.60 MiB / 412 / 832 ms | 3.13 MiB / 72 / 231 ms | 98.57 MiB / 1,067 / 1,547 ms | **1.52 MiB / 48 / 95 ms** |
| `rapid_interrupt_pan` | 12.44 MiB / 70 / 822 ms | 820.10 KiB / 18 / 307 ms | 16.67 MiB / 180 / 1,574 ms | **384.64 KiB / 14 / 142 ms** |
| `y_zoom` | 12.44 MiB / 70 / 137 ms | 14.16 MiB / 979 / 992 ms | 23.43 MiB / 983 / 985 ms | **11.80 MiB / 32 / 111 ms** |

### 1M (`Medium` sine / `MediumWhiteNoise`, 4 MiB budget, from `compare_with_other_strategies`)

| Workload | ripmap (sine) | grid (sine) | ripmap (white noise) | grid (white noise) |
| --- | ---: | ---: | ---: | ---: |
| `cold_overview` | 728.52 KiB / 9 / 90 ms | **411.30 KiB / 5 / 125 ms** | 2.12 MiB / 9 / 198 ms | **458.93 KiB / 5 / 132 ms** |
| `progressive_zoom_in` | 1.11 MiB / 53 / 123 ms | **904.88 KiB / 26 / 69 ms** | 6.22 MiB / 271 / 307 ms | **2.21 MiB / 25 / 89 ms** |
| `lateral_pan` | **203.32 KiB / 60 / 134 ms** | 282.09 KiB / 33 / 66 ms | 4.41 MiB / 58 / 175 ms | **4.01 MiB / 19 / 136 ms** |
| `jump_and_revisit` | **230.74 KiB / 32 / 87 ms** | 266.87 KiB / 17 / 47 ms | 6.71 MiB / 63 / 224 ms | **5.40 MiB / 16 / 157 ms** |
| `rapid_interrupt_pan` | 80.51 KiB / 12 / 161 ms | **70.65 KiB / 6 / 71 ms** | 1.30 MiB / 14 / 210 ms | **1.17 MiB / 5 / 189 ms** |
| `y_zoom` | **282.99 KiB / 30 / 49 ms** | 292.24 KiB / 16 / 23 ms | 1.30 MiB / 14 / 35 ms | **1.17 MiB / 5 / 32 ms** |

---

## Findings

1. **Direct addressing delivers the round-trip target.** Every view costs two rounds of reads (records, then cells and raw edges). Mean latency is the lowest of all strategies on every 10M workload except the random-walk overview. There it is 78 ms vs the ripmap's 35 ms, because the ripmap resolves that view from its inlined manifest top while the grid always reads column records first. Speedups vs the ripmap on the other workloads range from 1.1× (white-noise overview, 160 vs 181 ms) to 2.6× (random-walk `rapid_interrupt_pan`, 101 vs 258 ms), and reach 8.9× on white-noise `y_zoom` (111 vs 992 ms: both read raw, but the grid reads it in 32 requests instead of 979).
2. **Shape-matched cells fix the quadtree's failure mode and beat the ripmap on white noise.** Windowed white-noise views take **1.9–2.1× less wire than the ripmap**: `lateral_pan` 1.48 vs 2.87 MiB, `jump_and_revisit` 1.52 vs 3.13 MiB, `rapid_interrupt_pan` 385 vs 820 KiB. Against the quadtree they take 42× less on `lateral_pan`. The cells come from tall-thin aspects (`d = 6…12`), and the grid's rows are global, so a spike only occupies the cells it touches instead of stretching every block's bins.
3. **Random walk is on par with the ripmap in wire, faster in latency.** Windowed views take 0.73–1.09× the ripmap's wire (`rapid_interrupt_pan` 0.73×, `progressive_zoom_in` 1.09×), and `y_zoom` takes 0.62× (476 vs 762 KiB). The likely cause (not isolated) is that with explicit y-bounds, columns whose data is far off-screen resolve as one kernel through `offscreen_y`, so their cells are never read. The overview reads 1.58× more (126 vs 80 KiB): 512 columns × two candidate levels of 100-byte records, plus their cells.
4. **Deep y-zoom on white noise still falls back to raw.** `y_zoom` (5% window, 32× y-zoom) needs rows ~2¹² finer than the spike-stretched data height. Planes that fine average fewer than 16 points per cell *globally* (most rows of the bbox are empty above the noise band), so they are not stored and the views read raw: 11.80 MiB vs the ripmap's 14.16. The same global cut-off is why at 1M the white-noise 5% windows (`lateral_pan`, `jump_and_revisit`) also go raw: 4.01 / 5.40 MiB.
5. **Storage 1.35–1.47× raw in 4 objects** (ripmap 1.78× in 44). Most of it is cells (70–96 MiB); the column index is 4%. **Build is the slowest of all: 8.1 / 14.0 s at 10M.** That is one per-column sort pass per aspect, plus aborted passes on levels that turn out too fine.
6. **Interrupted pans cancel more requests** (`rapid_interrupt_pan`: 30 cancelled vs 5 for the others) because each view fires its whole second round at once. Completed wire is still the lowest.

## Deviations from the saved idea

- The grid is anchored to the **data bbox**, not to fixed global coordinates (appends would need a rebuild). Appends are not benchmarked, so this is deferred.
- One `columns.bin` and one `cells.bin` instead of a pair of files per level. Reads are byte ranges, so per-level files would only add objects.
- The CSC pointers of every plane at a level live **inside the column records**, so the plane can be chosen after round 1 without a third round.
- Aspects run in steps of 2: the client compensates by choosing between two adjacent x-levels by exact byte cost.

## Next Hypotheses

1. **Local instead of global density cut-off.** Store a fine plane only for the columns where it averages ≥ 16 points per cell (an empty CSC range elsewhere, falling back to raw there). This targets the white-noise `y_zoom` and the 1M windowed views, which today go fully raw although the noise band itself is dense.
2. **Read only the visible rows** when `y_bounds` is set. A column's cells are sorted by row, so a per-column row index (or a coarse-plane prefix) would let a y-zoomed view skip off-screen cells.
3. **Inline the top levels' records and cells in `manifest.bin`** to make the overview one round trip (the one workload where the ripmap is faster).
4. **Faster build:** one global sort by `(column at I, qy)` and k-way merges instead of a sort per aspect.
5. **Ablations:** aspect step 1 vs 2, `min_points_per_cell` 8 / 32, `col_leaf` 128 / 512, and `MAX_COL_WIDTH`.

---

## Addendum (same day) — In-browser explorer (`?page=grid`)

- **`src/pages/ripmap/grid-model.ts`** mirrors the Rust builder and client in TypeScript, and **`grid-page.ts`** renders it. The page shows:
  - the screen with cells, column kernels, raw edge points and the KDE;
  - the column decomposition per x-level, with the record pages read in round 1;
  - a map of every stored plane `(i, j)`, marking the candidate levels, the chosen plane, the planes used by edge chains, and the grids a view needed that were not stored;
  - the candidate comparison table, byte ranges per object, and round trips.
- Model tests (`test/ripmap-model.test.ts`) check level nesting, CSC consistency, and exact mass and `y_range` on three views per signal family.
- **Shape ablation observed in the explorer.** Setup: 300K white noise + spikes, `Zoom 20×` view `[0.45, 0.50]`, cold cache, `σ = 14 px`, `t = 1`, in a ~700 px wide plot. Wire is in-browser model bytes, not the Rust benchmark.
  - With the default shapes, the client picks plane `(10, 6)` (`d = 4`, 1024 × 64 cells) and reads **40.5 KiB in 3 rounds**.
  - With square cells only (`d = 0`, quadtree-like), no plane at levels 10 or 11 has rows fine enough, so every column goes raw: **369.1 KiB (105% of the raw window)**. That is 9.1× more, which reproduces the quadtree's failure mode in isolation.
  - Adding `Y zoom 4×` on the default shapes also goes raw (372.1 KiB). The needed grids `(10, 8)` and `(11, 7)` were dropped by the global 16-points-per-cell cutoff (finding 4), and the plane map shows them as holes next to stored neighbours.
