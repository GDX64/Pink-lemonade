# 2026-11-20 — Progressive Harness & Paged Moment Tree (v1 → v2)

## Context & Goal

The existing in-memory `TreeDownsampler` (`rust/src/tree_merge.rs`) handles millions of samples interactively in WASM, but requires holding the full raw dataset and complete binary moment tree in client memory. Our goal is to support **gigabyte-scale datasets** under a strict client memory budget by fetching only the data required for the current viewport and screen resolution over **dumb static byte-range storage** (mirroring HTTP `Range` requests on static hosting / object storage).

## Infrastructure Built Today

- **`pink-lemonade-progressive-core`** (`rust/progressive-core/`): lean, WASM-compatible, `dyn`-compatible async traits (`ByteRangeStore`, `ObjectWriter`, `DatasetBuilder`, `ProgressiveClient`, `ProgressiveStrategy`) and pass-by-value core types (`Point`, `ViewQuery`, `Frame`, `ClientConfig`).
- **`pink-lemonade-progressive-harness`** (`rust/progressive-harness/`): `[dev-dependencies]` test & benchmark harness with `MemoryStore`, `DiskStore`, Tokio virtual-time `SimulatedTransport` (`NetworkProfile`), `TrackingAllocator`, streaming fixtures (`DatasetStream`), exact moment & density NMAE verifiers, `run_conformance_suite`, standard scenarios (`Light`, `Medium`, `ConstrainedMobile`, `Heavy`, `Gigabyte`), and JSON export.
- **`pink-lemonade-progressive-naive`** (`rust/progressive-naive/`): single-file full-download control baseline (`points.bin`, no downsampling).
- **`pink-lemonade-progressive-tree`** (`rust/progressive-tree/`): paged multi-level moment tree (`PagedTreeStrategy`) + `examples/export_reports.rs`.
- **Analysis workflow**: root `.venv`, `./dev benchmark --fast` (~0.8 s warm via `[profile.fast]`), and `benchmarks/analysis.ipynb`.

---

## Experiment 1 — Initial Paged Moment Tree (`v1`)

**Configuration:** `128 B` nodes (16 × `f64`), `LEAF_POINTS = 64`, `FANOUT = 32` (4 KiB pages), strict contiguous coalescing (`max_gap = 0`, max 16 tree pages / 32 leaves), multi-object layout (`manifest.bin` with inlined root + `level_*.bin` + `points.bin`), bounded LRU cache.

### Results vs `naive_full_download`

- **`Medium` (`sine_wave`, 1M points, 22.89 MiB raw, broadband 30 ms / 50 Mbps, 4 MiB budget):**
  - `naive_full_download` exceeded the 4 MiB budget on every step, transferring `22.89–183.11 MiB` at `3,871 ms` mean latency.
  - `paged_moment_tree` (`v1`) cut `cold_overview` to **`1.97 MiB`** (`282 ms`, 0 bytes of `points.bin`) and `lateral_pan` to **`386 KiB`** (`110 ms` — **485× less wire**, **35× faster**).
- **`ConstrainedMobile` (`noisy_spikes`, 1M points) & `Heavy` (`multi_regime`, 10M points, 240 MiB raw):**
  - **Failure mode discovered:** On IID/white vertical noise, consecutive `x`-samples jump across a wide `y` band (`Δy >> 1σ`). No internal node or 64-point leaf ever satisfied `fits_threshold`, so the 1D x-ordered tree descended all the way into `points.bin` and downloaded **100% of the dataset** (`24.86 MiB` on 1M, `248.57 MiB` on 10M).

### Takeaway 1
A 1D x-ordered moment tree cannot compress uncorrelated vertical white noise — only signals with local vertical continuity (or a 2D spatial / y-split index).

---

## Experiment 2 — Realistic Target Signal: Integrated Gaussian Noise (`RandomWalk`)

Our target datasets (financial/telemetry time series, matching `src/data/noise.ts`) behave like **integrated Gaussian noise (Brownian random walk)** rather than IID white noise.

- **Why the scaling flips mathematically:** For a Brownian walk of $N$ steps, global vertical span grows as $\Theta(\sqrt{N})$ and any $k$-step slice spans $\Theta(\sqrt{k})$. Reaching $\Delta y < 1\sigma$ on a screen of height $H/\sigma \approx 67.5$ requires slice fraction $f = k/N \lesssim (1 / 67.5)^2 \approx 1 / 4556$ — **independent of $N$**.
- Ported `src/data/noise.ts` (`Box-Muller N(0, 1)` cumulative sum + bimodal weight distribution) as `DatasetShape::RandomWalk { seed: 578_211 }` and switched `ConstrainedMobile`, `Heavy`, and `Gigabyte` to it.

### `Heavy` (`random_walk`, 10M points, 240 MiB raw, broadband, 16 MiB budget) — `v1` Baseline

| Workload | Wire | Requests | Mean Latency | Peak Mem |
| --- | ---: | ---: | ---: | ---: |
| `cold_overview` | `710 KiB` (0.29% of raw) | `33` | `240 ms` | `720 KiB` |
| `progressive_zoom_in` | `7.71 MiB` | `1,042` | `965 ms` | `7.87 MiB` |
| `lateral_pan` | `6.63 MiB` | `2,045` | `1,455 ms` | `6.80 MiB` |
| `jump_and_revisit` | `18.67 MiB` | `5,973` | `5,312 ms` | `16.00 MiB` |
| `rapid_interrupt_pan` | `2.53 MiB` | `908` | `4,840 ms` | `2.60 MiB` |

### Takeaway 2
`cold_overview` dropped **346×** vs raw (`240 MiB → 710 KiB`), confirming the $\sqrt{k}/\sqrt{N}$ scaling on 10M points. However, zoomed workloads showed severe **request fragmentation**: `jump_and_revisit` downloaded only `18.7 MiB` yet fired **5,973 tiny requests** (`5.3 s` latency) because `max_gap = 0` couldn't merge alternating calm/steep pages ("swiss-cheese" access pattern).

---

## Experiment 3 — Compact Nodes, Pruned `y_range`, Gap-Bridging & Tree Geometry (`v2`)

### Optimizations Implemented
1. **64-byte `PackedNode` (`PLPTREE2`, 2× smaller wire & cache):** exact `f64` `[min_x, max_x, min_y, max_y]` (dropping redundant `first_x`/`last_x`), `f32` `mass`, box-relative mean `(tx, ty)`, and box-normalized centered covariance `(var_x/dx², cov_xy/(dx·dy), var_y/dy²)`, plus `u16` box-relative line extrema offsets. Passes full conformance suite (`2e-4` moments, `1e-6` `y_range`, exact line extrema).
2. **Pruned `visible_y_range` boundary descents:** folds fully-contained interior nodes first and skips descending any straddling boundary node whose `[min_y, max_y]` already lies within the known interior range (exact, zero approximation).
3. **Gap-bridging range coalescing:** merges nearby missing pages/leaves across small gaps to avoid extra RTTs.
4. **Selective leaf caching:** when bridging gaps in `points.bin`, only actually requested leaves are inserted into the LRU cache so unused bridged leaves don't evict hot regions.

### Parameter Sweep on `Heavy` (`random_walk`, 10M points, 16 MiB budget)

| Variant | Config (`leaf`, `fanout`, `gap_tree`, `gap_leaf`) | `cold_overview` (wire / reqs / ms) | `progressive_zoom_in` (wire / reqs / ms) | `lateral_pan` (wire / reqs / ms) | `jump_and_revisit` (wire / reqs / ms / mem) | `rapid_interrupt_pan` (wire / reqs / ms) |
| --- | --- | --- | --- | --- | --- | --- |
| **Baseline (`v1`)** | 128 B, `64 / 32 / 0 / 0` | `710 KiB / 33 / 240 ms` | `7.71 MiB / 1042 / 965 ms` | `6.63 MiB / 2045 / 1455 ms` | `18.67 MiB / 5973 / 5312 ms / 16.0 MiB` | `2.53 MiB / 908 / 4840 ms` |
| **Exp A** | 64 B, `64 / 32 / 8 / 12` | `365 KiB / 24 / 229 ms` | `7.51 MiB / 360 / 382 ms` | `14.48 MiB / 651 / 566 ms` | `57.46 MiB / 378 / 747 ms / 16.0 MiB` | `7.36 MiB / 169 / 1209 ms` |
| **Exp B** | 64 B, `32 / 16 / 4 / 2` | `1.06 MiB / 38 / 283 ms` | `5.06 MiB / 481 / 469 ms` | `3.76 MiB / 349 / 368 ms` | `6.94 MiB / 2646 / 2401 ms / 7.38 MiB` | `1.15 MiB / 169 / 1013 ms` |
| **Exp C** | 64 B, `64 / 8 / 16 / 4` | `757 KiB / 51 / 385 ms` | `5.67 MiB / 421 / 430 ms` | `7.52 MiB / 1299 / 1004 ms` | `44.63 MiB / 1623 / 1783 ms / 16.0 MiB` | `3.39 MiB / 522 / 2950 ms` |
| **Exp D (`v2`, kept)** | 64 B, `32 / 32 / 8 / 8` + selective leaf cache | **`630 KiB / 6 / 107 ms`** | **`7.10 MiB / 122 / 179 ms`** | **`3.76 MiB / 314 / 316 ms`** | **`10.53 MiB / 1676 / 1562 ms / 6.66 MiB`** | **`1.17 MiB / 157 / 948 ms`** |

### Key Findings from the Sweep
- **Tree height dominates RTT (`FANOUT = 32` wins):** `FANOUT = 8` (Exp C) and `FANOUT = 16` (Exp B) added 1–2 extra remote levels, increasing sequential RTTs on overview and pan. Keeping `FANOUT = 32` keeps 10M points at only 3 remote levels (`cold_overview = 6 reqs / 107 ms`).
- **Smaller leaves dramatically reduce `points.bin` spills (`LEAF_POINTS = 32` wins):** At `LEAF_POINTS = 64`, ~10–15% of Brownian leaves exceeded `1σ` on 5% zooms; halving leaf size to `32` cut vertical span by $\sqrt{2}$ while keeping storage overhead at **+3.6%** (`248.57 MiB` vs `240 MiB` raw, thanks to the 64 B node).
- **Aggressive raw-leaf gap-bridging backfires on cache:** Bridging 12–16 leaves in `points.bin` (Exp A and a `gap_leaf=16` check) grabbed too many unneeded raw leaves, overflowing the 16 MiB cache and causing `revisit_A` to miss. Pairing `gap_leaf = 8` (6 KiB) with **selective leaf caching** dropped `jump_and_revisit` peak client RAM from **`16.00 MiB → 6.66 MiB`** (100% cache hits on both revisits!).

### Summary of `v1 → v2` Gains
- **`Heavy` (10M `random_walk`):** 2.2×–5.4× lower mean latency across all 5 workloads, 3.6×–8.5× fewer requests, 1.8×–2.2× less wire on pan/jump/interrupt, and 2.4× lower peak RAM on `jump_and_revisit` (`6.66 MiB` for a 240 MiB dataset).
- **`ConstrainedMobile` (1M `random_walk`, 5 Mbps / 100 ms RTT):** `progressive_zoom_in` dropped `2,006 ms → 595 ms` (**3.4×**), `cold_overview` `1,241 ms` vs naive's `38,501 ms` (**31×**).
- **`Medium` (1M `sine_wave`):** `cold_overview` `34 → 11 reqs` (`282 ms → 186 ms`), `lateral_pan` `364 KiB` / `92 ms`.

---

## Next Hypotheses to Explore

1. **Subtree-packed pages (van Emde Boas / mini binary subtrees):** Pack a height-5 binary subtree (31 × 64 B ≈ 2 KiB) into each page to get intermediate 2× resolution steps (`512 → 256 → 128 → 64 → 32`) with zero extra RTTs.
2. **Reducing `jump_and_revisit` remaining leaf requests (`1,676`):** Inspect whether `region_A`/`region_D` steep 32-point leaves can be handled by either `LEAF_POINTS = 16` (stored as a single `level_0` only where needed, or adaptive leaf splitting) or batched multi-range / smarter gap threshold tuned by bandwidth-delay product.
3. **Single-file container format:** Pack `manifest`, `level_*`, and `points` into one static file (1 URL on CDN/S3) with level byte offsets in the header.
4. **2D spatial / y-split index (optional track):** If white-noise datasets (`NoisySpikes` / `MultiRegime`) become a priority, test a paged 2D quadtree/KD-tree alongside `paged_moment_tree`.
