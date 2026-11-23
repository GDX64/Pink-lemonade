# 2026-11-21 — 2D Space Partitioning: Moment k-d Tree vs Moment Ripmap

## Context & Goal

The 1D x-ordered `paged_moment_tree` (v2, [previous entry](./2026-11-20-progressive-moment-tree.md)) is excellent on vertically coherent signals but downloads **100% of the dataset** on uncorrelated white noise: every x-node spans the full y-band, so nothing ever fits `1σ`. We implemented two 2D strategies and compared them head-to-head against v2 on both signal families:

1. **`moment_kdtree`** — a classic fixed-shape 2D partition (low storage, baseline for fixed-shape over-refinement).
2. **`moment_ripmap`** — decoupled x/y resolution: σ-wide x-blocks rendered through dense per-block y-bin moment planes.

## Infrastructure Changes

- **`pink-lemonade-progressive-common`** (new, WASM-ready): extracted `Node`/`PackedNode`, LRU cache, fetch helpers, and the whole paged hierarchy (`write_hierarchy` + `HierarchyReader`) from `progressive-tree`. The hierarchy is defined over an *arbitrary permutation* of the points (leaf `i` = permuted points `[i·L, (i+1)·L)`), with a pluggable per-node `Decision` (`Resolve` / `Plane(k)` / `Expand`) consulted only for nodes fully inside `[min_x, max_x]`, so exactness never depends on point order.
  - **Regression check:** after the extraction, v2's Light + Medium results were **bit-identical** in wire bytes, request counts, and latency (only `+8 B` reported memory for the manifest's stored magic). `PLPTREE2` artifacts are unchanged.
  - `Node::combined` now picks `first`/`last` samples by x (identical for x-adjacent merges, required for overlapping 2D cells).
  - **Byte-based page coalescing:** gap/run limits are now `16 KiB` / `256 KiB` instead of `8` / `128` pages (identical at fanout 32). At fanout 8 the page-count limits produced 64 KiB requests: k-d `cold_overview` on 1M white noise went **129 → 26 requests, 723 → 288 ms** (wire `1.21 → 1.97 MiB`, from wider gap bridging).
- **Off-screen y culling** (`offscreen_y`, used by all hierarchy strategies): with explicit `y_bounds`, a fully-inside node whose bbox is ≥ 6σ off-screen *and* whose merged Gaussian mean is ≥ 6 marginal std-devs away is emitted as one kernel. Moments stay exact (nodes are collapsed, never dropped).
- **Harness:** `MediumWhiteNoise` (1M) and `HeavyWhiteNoise` (10M) `NoisySpikes` scenarios; a new `y_zoom` workload (5% x-window: auto-y → 2× → 8× → 32× box-zoom → 8× vertical pan → reset) using `WorkloadStep::relative_y_bounds` resolved against the preceding auto-y frame; explicit-`y_bounds` views in `run_conformance_suite`; `view_min_y`/`view_max_y` in the JSON report; `export_reports --only <scenario>`.

## Strategy Designs

### Moment k-d tree (`progressive-kdtree`, `PLPKDTR1`)
- Builder permutes the points so every `L·Fᵏ`-point chunk is a 2D cell: binary splits along the longest side in *overview screen units* (x scaled by 16:9), split positions snapped to multiples of the child size; each 32-point leaf is then sorted by x. The **unchanged** shared hierarchy writer/reader serve it (`L = 32`, `F = 8`, root level ≤ 1024 nodes inlined in the manifest).
- Line samples are re-sorted by x (k-d order is not x-monotonic).

### Moment ripmap (`progressive-ripmap`, `PLPRIPM1` + `ripmap.bin` + `plane_{ℓ}_{k}.bin`)
- x-ordered hierarchy with **fanout 4** (block widths within 4× of each other) for exact `y_range`, line extrema, and boundary clipping.
- For each level `ℓ` and depth `k = 1..kmax(ℓ)`: every x-block stores `2^k` dense **24-byte `MomentCell`s** (mass, mean, centered covariance as `f32`, normalized to the block's x-range and the bin's y-range — the rectangle is re-derived from the block's exact `f64` bounds, never stored). Planes exist only while `block_points / 2^k ≥ 16`.
- Query: a fully-inside node that does not already fit is resolved through plane `(ℓ, k)` once its on-screen width `w ≤ 0.9σ`, with the smallest `k` such that `w² + h² ≤ 1`. Plane blocks and raw boundary leaves are fetched concurrently (`join2`) after the frontier.

---

## Experiment 4 — Heavy Head-to-Head (10M points, 228.88 MiB raw, broadband 30 ms / 50 Mbps, 16 MiB budget)

> **Correction (2026-11-22):** 10M points × 24 B = 240,000,000 B = **228.88 MiB**, not 240 MiB; storage overheads and ratios against raw below have been recomputed (raw wire/latency numbers are unchanged).

Numbers below are from `benchmarks/results/latest.json`. Wire bytes, request counts, and kernel counts are fully deterministic under virtual time; mean latencies jitter by ≤ 0.1% between runs (Tokio paused-clock auto-advance).

### Storage & build

| Strategy | Storage | Overhead vs raw | Objects | Build |
| --- | ---: | ---: | ---: | ---: |
| `paged_moment_tree` (v2) | `248.57 MiB` | `+8.6%` | 5 | ~0.7–1.3 s |
| `moment_kdtree` | `250.67 MiB` | `+9.5%` | 5 | ~2.3–2.8 s |
| `moment_ripmap` | `406.99 MiB` | `+77.8%` (1.78×) | 44 | ~1.9–2.4 s |

Ripmap breakdown: `228.9 MiB` points + `25.4 MiB` fanout-4 x-levels + `152.7 MiB` planes.

### `Heavy` — integrated Gaussian noise (`random_walk`): wire / requests / mean latency

| Workload | `paged_moment_tree` | `moment_kdtree` | `moment_ripmap` |
| --- | --- | --- | --- |
| `cold_overview` | `630 KiB / 6 / 106 ms` | `406 KiB / 82 / 506 ms` | **`80 KiB / 8 / 35 ms`** |
| `progressive_zoom_in` | `7.10 MiB / 122 / 179 ms` | `4.90 MiB / 652 / 627 ms` | **`768 KiB / 70 / 151 ms`** |
| `lateral_pan` | `3.76 MiB / 314 / 315 ms` | `5.96 MiB / 908 / 713 ms` | **`631 KiB / 130 / 222 ms`** |
| `jump_and_revisit` | `10.53 MiB / 1,676 / 1,562 ms` | `6.89 MiB / 993 / 992 ms` | **`646 KiB / 72 / 175 ms`** |
| `rapid_interrupt_pan` | `1.17 MiB / 157 / 947 ms` | `1.66 MiB / 235 / 1,331 ms` | **`176 KiB / 22 / 258 ms`** |
| `y_zoom` | `6.41 MiB / 491 / 483 ms` | `2.83 MiB / 430 / 427 ms` | **`762 KiB / 52 / 101 ms`** |

### `HeavyWhiteNoise` — uncorrelated `noisy_spikes`: wire / requests / mean latency

| Workload | `paged_moment_tree` | `moment_kdtree` | `moment_ripmap` |
| --- | --- | --- | --- |
| `cold_overview` | `248.57 MiB / 1,304 / 13,690 ms` | **`875 KiB / 18 / 188 ms`** | `937 KiB / 5 / 181 ms` |
| `progressive_zoom_in` | `435.02 MiB / 2,288 / 4,047 ms` | `19.00 MiB / 509 / 553 ms` | **`4.21 MiB / 164 / 299 ms`** |
| `lateral_pan` | `42.89 MiB / 247 / 419 ms` | `45.30 MiB / 867 / 803 ms` | **`2.87 MiB / 118 / 257 ms`** |
| `jump_and_revisit` | `74.60 MiB / 412 / 831 ms` | `78.62 MiB / 843 / 1,230 ms` | **`3.13 MiB / 72 / 231 ms`** |
| `rapid_interrupt_pan` | `12.44 MiB / 70 / 821 ms` | `13.20 MiB / 130 / 1,159 ms` | **`820 KiB / 18 / 307 ms`** |
| `y_zoom` | **`12.44 MiB / 70 / 137 ms`** | `15.57 MiB / 609 / 622 ms` | `14.16 MiB / 979 / 990 ms` |

Rendered kernels on the white-noise overview: x-tree **7,548,936** (every raw point), k-d tree 4,879, ripmap 13,632.

### At 1M points the picture is narrower

- **`MediumWhiteNoise`:** both 2D strategies fix the overview (`cold_overview`: x-tree `24.86 MiB / 1,487 ms`, k-d `1.97 MiB / 288 ms`, ripmap `2.12 MiB / 198 ms`), but a 5% window is only ~1.2 MiB raw, so windowed workloads land near the raw window for every strategy (`lateral_pan` 4.3–5.0 MiB, `y_zoom` 1.25–1.44 MiB). The ripmap's planes only pay off once σ-wide blocks hold enough points per bin (`≥ 16`), which at 1M happens for the overview but not inside 5% windows.
- **`ConstrainedMobile`** (1M `random_walk`, 5 Mbps / 100 ms): ripmap `cold_overview` `0.18 MiB / 246 ms` vs x-tree `1.97 MiB / 1,241 ms` and naive `22.89 MiB / 38,501 ms`; `lateral_pan` `0.95` vs `4.38 MiB`; `y_zoom` is a wash (`1.41` vs `1.35 MiB`).

### Density quality (worst-step NMAE per workload, ground truth, 48 pixel samples/step)

All three strategies stay far inside the 5% tolerance at 100K, 1M, and 10M points (`density_nmae_within_tolerance_on_all_workloads`, `PLP_NMAE_POINTS` override). At 10M:

| | x-tree | k-d tree | ripmap |
| --- | --- | --- | --- |
| `noisy_spikes` (worst over workloads) | `0.323%` | `0.421%` | `0.323%` (≤ `0.053%` on every plane-served workload) |
| `random_walk` (worst over workloads) | `0.224%` | `0.317%` | `0.224%` (≤ `0.026%` on every plane-served workload) |

On plane-served workloads the ripmap's per-bin covariances give **~7–80× lower NMAE** than v2's merged x-columns (e.g. white-noise `lateral_pan` `0.004%` vs `0.115%`; the one exception is white-noise `y_zoom` at 2.6×, which is mostly raw-served); its worst case is the deep `progressive_zoom_in` step served from raw points, identical to v2.

---

## Findings

1. **The ripmap dominates on both signal families**, not just white noise. On the target `random_walk` signal it cuts wire **6–17×** vs v2 (`jump_and_revisit`: `10.53 MiB / 1,562 ms → 646 KiB / 175 ms`; `cold_overview` `80 KiB` = 0.03% of raw) while improving density accuracy. Rendering σ-wide x-blocks as a handful of y-bins is far cheaper than descending an x-tree until each node is also `< 1σ` tall. On white noise it reduces the x-tree's 100% download by **15–270×** (`cold_overview` `248.6 MiB → 937 KiB`, 13.7 s → 181 ms).
2. **The fixed-shape k-d tree only helps the overview.** It nails the white-noise overview (`875 KiB`), but its cells are shaped for the overview aspect: at a 20× x-zoom they are ~20× too wide, so windowed views fall through to raw leaves *plus* `O(√cells)` boundary leaves per viewport edge, ending up at or above the x-tree (`lateral_pan` `45.3` vs `42.9 MiB`). On random walk, its extra levels/boundary leaves cost 5–8× more requests than v2. It is dominated by the ripmap everywhere except storage (+9.5% vs +78%).
3. **Deep vertical zoom on white noise remains unsolved for all three** (`y_zoom` ≈ the 12.4 MiB raw window). The ripmap is *worse* in request count (979): `+80` spikes stretch each block's `[min_y, max_y]`, so equal-width dense bins spend their depth on the empty noise–spike gap, `k_needed > kmax` on spike-containing blocks, and those fall back to fragmented raw leaves (the 2× step alone costs 8 MiB). On random walk, `y_zoom` works well for the ripmap (`762 KiB`) and k-d tree (`2.83 MiB`, aided by off-screen culling).
4. **Storage is the ripmap's price:** 1.78× raw (planes ≈ 0.67×, fanout-4 x-levels ≈ 0.11×), tunable via `min_points_per_bin` and by skipping planes at levels/depths the query policy rarely selects.

## Methodology Notes

- **Static object count:** at 10M points the ripmap writes **44 objects**: 36 `plane_{ℓ}_{k}.bin` files (levels 0–5 hold `kmax` = 1, 3, 5, 7, 9, 11 depths), `points.bin`, `level_0..4.bin`, `manifest.bin`, and `ripmap.bin`. The tree and k-d tree write 5 objects each. A query touches only the few planes it selects, but on static hosting this is 44 URLs per dataset.
- **Quality checks must run at scale:** at 100K–1M points the ripmap's windowed views are mostly served from raw leaves, so their NMAE matches the x-tree's exactly and says nothing about the planes. Plane quality only shows up at 10M (`PLP_NMAE_POINTS=10000000`, ~75 s with `--profile fast`).
- **Regression checks compare wire, requests, and kernel counts, not latency**: those are exactly deterministic under virtual time, while latency jitters ≤ 0.1%.

## Next Hypotheses

1. **Sparse ripmap planes** (store only non-empty bins as `(bin u16, MomentCell)` with a per-block offset table): removes the outlier-stretch waste and lets `kmax` go deeper where data concentrates, which should fix the white-noise `y_zoom` fallback. Key question: whether the offset table can ride along with the level pages already fetched (no extra RTT).
2. **Robust bin ranges + outlier cells:** bin within per-block quantile bounds and keep extreme points in a separate exact outlier list (line extrema / `y_range` already come from the exact x-nodes).
3. **Plane pruning sweep:** measure which `(ℓ, k)` planes the policy actually selects across workloads and drop the rest; sweep `min_points_per_bin ∈ {8, 16, 32}` and x-fanout `{4, 8}` to trade storage vs wire.
4. **Single-file container:** pack the ripmap's 44 objects into one file with a byte-offset table, so each dataset is one URL.
5. **Retire or rework the k-d tree:** a view-adaptive variant would need x/y-decoupled cells — which is what the ripmap already is. Keep it only as a fixed-shape baseline.
