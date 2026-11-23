# 2026-11-22 — Moment Ripmap v2 (`PLPRIPM2`): Sparse 16-Byte Planes, Inline Directories, and Level-Jumping Client

## Context & Goal

In [Experiment 4](./2026-11-21-2d-strategies-kdtree-ripmap.md) and [Experiment 5](./2026-11-22-moment-quadtree.md), `moment_ripmap` (v1, `PLPRIPM1`) led on 10 of 12 `Heavy` (10M-point) workloads, but carried four structural weaknesses:

1. **Outlier / spike stretch on dense planes:** `+80` spikes in `noisy_spikes` stretched each block's `[min_y, max_y]` 5×, wasting ~80% of dense 24-byte slots on empty zeros, pushing `needed_depth` past `kmax(ℓ)`, and cascading alternating blocks to fragmented raw leaves (`HeavyWhiteNoise` `y_zoom`: `14.16 MiB / 979 req / 990 ms`; `MediumWhiteNoise` `progressive_zoom_in`: `6.22 MiB / 271 req / 314 ms`).
2. **Fanout-4 RTT tax & fragmented plane requests:** Walking `level_4 → 3 → 2 → 1 → 0` one RTT per level and splitting adjacent blocks with different `k` into separate `plane_{ℓ}_{k}.bin` reads cost `222–257 ms` mean latency and `118–130` requests on `lateral_pan`.
3. **Storage & object count:** `406.99 MiB (+77.8% vs 228.88 MiB raw)` spread across **44 objects** (36 plane files + 5 levels + `manifest.bin` + `ripmap.bin` + `points.bin`), plus 4 requests on `open_client`.
4. **High kernel counts on planes:** Plane bins were emitted directly without greedy merging (`13,632` kernels on `HeavyWhiteNoise` `cold_overview` vs `4,879` for the k-d tree).

The goal was to eliminate all four without touching `progressive-common` or any other strategy.

---

## Design

We evaluated the improvements in two measured stages:

### Stage 1 — Client-only optimizations (unchanged `PLPRIPM1` layout)
- **`x_factor`-guided level jumping:** Since `x_factor` is known before `y_range`, jump directly from `root_nodes` to the target level `ℓ*` where `w ≤ 0.9σ` (skipping `level_4.bin` on 5% windows and immediately providing a 99%+ interior `y_range`).
- **Skip `visible_y_range` on explicit `y_bounds`:** Fold exact `Frame.y_range` directly from the frontier and clipped boundary leaves when `query.y_bounds.is_some()`.
- **Boundary skip-descent:** Jump straddling boundary nodes up to 3 levels (`≤ 64` nodes = 4 KiB) per RTT instead of 1 level per RTT.
- **Cross-`k` cache reuse:** Look up `k..=kmax` in LRU so coarser queries reuse already-cached finer blocks at zero network cost.
- **Cross-`k` run promotion & free absorption:** Absorb coarser blocks already inside a finer run's coalesced byte span, and promote small coarse runs (`≤ 4 KiB` extra) into an adjacent `k + 1` plane to save whole requests.
- **Cache bridged gap blocks** (inserted before requested blocks so active entries stay MRU) and **greedy cell merging + safe `offscreen_y` bin collapsing** in `emit_plane`.

### Stage 2 — `PLPRIPM2` format (sparse 16-byte cells, inline directories, ≤ 4 objects)
- **4-object container** (`manifest.bin`, `levels.bin`, `planes.bin`, `points.bin`; ≤ 2 objects on tiny datasets):
  - `manifest.bin` (`PLPRIPM2`) folds in all level offsets, plane offsets, and inlined root records, loaded in **1 EOF-clamped `read_range(0, 128 KiB)`** (1 request, 1 RTT, no `HEAD`, no `ripmap.bin`).
  - `levels.bin` packs non-root levels back-to-back. Each record at level `ℓ` is a 64-byte `PackedNode` plus `P_ℓ` inline 6-byte `(cell_offset: u32, cell_count: u16)` directory entries locating that block's sparse cells in `planes.bin` — **zero extra RTTs for sparse offset lookup**.
  - `planes.bin` packs all `(ℓ, k)` sparse cell streams back-to-back.
- **16-byte quantized `SparseCell`:** Stores `(bin: u16, q_tx: u16, q_ty: u16, q_var_x: u16, q_cov_xy: i16, q_var_y: u16, mass: f32)` using Popoviciu's `[0, 0.25]` bound on normalized variances (`[-0.25, 0.25]` for covariance), clamping reconstructed covariance to guarantee positive semi-definiteness.
- **Adaptive per-block sparsity & plane pruning:**
  - Level 0 (32-pt leaves) stores no planes (saves `14.3 MiB`).
  - Levels `ℓ ≥ 1` store stride-1 `k` for cheap depths (`k ≤ k_dense - 2`, ≤ 1% raw each) and stride-2 for expensive depths up to `min(k_dense + 2, 8)` for coarse blocks (`≥ 8,192` pts) and `min(k_dense + 2, 9)` for windowed blocks.
  - A block is emitted into plane `(ℓ, k)` only when its **non-empty** bin count is `≤ block_points / min_points_per_bin` (else `cell_count = 0`, 0 bytes stored). Spiky blocks automatically gain +2 extra depth (`k = 7` at Level 2, `k = 9` at Level 3) without wasting storage on dense continuous blocks.
- **Hybrid boundary fast path:** When only straddling boundary nodes remain within 3 levels of level 0, low-span boundaries (`≤ 16σ`) jump directly to `level_0` (2 requests, 1 RTT), while tall-span boundaries (`> 16σ`, e.g., white noise) fetch `0..cur_level` concurrently in 1 RTT so interior siblings still hit their sparse planes.
- **Faster builder:** Shifted raw-moment accumulation (`u = x - min_x, v = y - min_y`) eliminates divisions from the inner point loop and bin-halving pass, reusing hierarchy bounds computed once and writing buffered chunks sequentially.

---

## Experiment 6 — Ablation & Head-to-Head Results

Scenarios and network match Experiments 4–5 (10M points = `228.88 MiB` raw, broadband 30 ms / 50 Mbps, 16 MiB budget; 1M points = `22.89 MiB` raw). Cells are `wire / completed requests / mean latency`.

### Storage, objects & build time (10M)

| Version | Storage (`random_walk` / `noisy_spikes`) | Overhead vs raw | Objects | Build (`random_walk` / `noisy_spikes`) |
| --- | ---: | ---: | ---: | ---: |
| Ripmap v1 (`PLPRIPM1`) | `406.99 / 406.99 MiB` | `+77.8% / +77.8%` | 44 | `~2.65 s / ~1.97 s` |
| **Ripmap v2 (`PLPRIPM2`)** | **`307.86 / 296.95 MiB`** | **`+34.5% / +29.7%`** | **4** | **`~1.88 s / ~1.22 s`** |

### `Heavy` — 10M `random_walk`

| Workload | Ripmap v1 (baseline) | Stage 1 (client-only) | **Ripmap v2 (`PLPRIPM2`)** |
| --- | --- | --- | --- |
| `cold_overview` | `79.9 KiB / 8 / 35.0 ms` | `65.7 KiB / 7 / 35.0 ms` | **`68.2 KiB / 3 / 36.0 ms`** |
| `progressive_zoom_in` | `768.3 KiB / 70 / 151.3 ms` | `629.5 KiB / 55 / 114.3 ms` | **`650.9 KiB / 44 / 101.0 ms`** |
| `lateral_pan` | `630.6 KiB / 130 / 222.1 ms` | `435.4 KiB / 78 / 111.2 ms` | **`406.3 KiB / 52 / 101.1 ms`** |
| `jump_and_revisit` | `646.4 KiB / 72 / 174.8 ms` | `436.2 KiB / 43 / 85.2 ms` | **`422.8 KiB / 37 / 83.8 ms`** |
| `rapid_interrupt_pan` | `175.9 KiB / 22 / 258.0 ms` | `130.4 KiB / 16 / 134.0 ms` | **`131.8 KiB / 10 / 106.0 ms`** |
| `y_zoom` | `762.1 KiB / 52 / 100.5 ms` | `644.4 KiB / 44 / 75.5 ms` | **`587.5 KiB / 39 / 80.0 ms`** |

### `HeavyWhiteNoise` — 10M `noisy_spikes`

| Workload | Ripmap v1 (baseline) | Stage 1 (client-only) | **Ripmap v2 (`PLPRIPM2`)** |
| --- | --- | --- | --- |
| `cold_overview` | `937.2 KiB / 5 / 181.0 ms` | `937.2 KiB / 4 / 181.0 ms` | **`247.1 KiB / 2 / 65.0 ms`** |
| `progressive_zoom_in` | `4.21 MiB / 164 / 299.0 ms` | `4.21 MiB / 146 / 263.2 ms` | **`1.10 MiB / 68 / 126.0 ms`** |
| `lateral_pan` | `2.87 MiB / 118 / 256.8 ms` | `2.86 MiB / 66 / 144.8 ms` | **`749.0 KiB / 101 / 105.8 ms`** |
| `jump_and_revisit` | `3.13 MiB / 72 / 230.7 ms` | `3.10 MiB / 36 / 142.7 ms` | **`712.4 KiB / 52 / 76.7 ms`** |
| `rapid_interrupt_pan` | `820.1 KiB / 18 / 307.0 ms` | `817.1 KiB / 11 / 214.0 ms` | **`231.2 KiB / 13 / 115.0 ms`** |
| `y_zoom` | `14.16 MiB / 979 / 989.8 ms` | `14.16 MiB / 970 / 969.3 ms` | **`13.38 MiB / 84 / 180.5 ms`** |

### 1M scenarios (`Medium`, `MediumWhiteNoise`, `ConstrainedMobile`)

- **`MediumWhiteNoise` (`1M` `noisy_spikes`):** storage `34.97 → 29.34 MiB` (`22 → 4` objects). `cold_overview` dropped **`2.12 MiB / 9 req / 198 ms → 279.3 KiB / 2 req / 68 ms`** (**7.8× less wire**, beating the quadtree's previous `1.25 MiB` best by 4.6×). `progressive_zoom_in` dropped **`6.22 MiB / 271 req / 314 ms → 2.14 MiB / 55 req / 98 ms`** (**2.9× less wire, 4.9× fewer requests, 3.2× faster**), eliminating the `width = 0.200` spike-fallback cliff completely.
- **`Medium` (`1M` `sine_wave`):** `cold_overview` `728.5 → 295.0 KiB` (`9 → 2` req, `90 → 71 ms`); `progressive_zoom_in` `1,134 → 491 KiB` (`53 → 29` req, `123 → 83 ms`).
- **`ConstrainedMobile` (`1M` `random_walk`, 5 Mbps / 100 ms):** `cold_overview` `188.3 → 111.8 KiB` (`246 → 200 ms`); `lateral_pan` `977.2 → 520.6 KiB` (`112 → 68` req, `673 → 408 ms`); `rapid_interrupt_pan` `218.9 → 155.7 KiB` (`697 → 459 ms`).

### Kernel counts & 10M density quality (`PLP_NMAE_POINTS=10000000`, 48 pixel samples/step)

- Greedy merging + off-screen bin collapsing cut `HeavyWhiteNoise` `cold_overview` kernels from **`13,632 → 5,343` (-61%)** and `Heavy` `cold_overview` from **`965 → 508` (-47%)**.
- All workloads stay far inside the 5% NMAE tolerance with 16-byte quantized `SparseCell`s:
  - `noisy_spikes`: `cold_overview` `0.024%`, `lateral_pan` `0.074%`, `jump_and_revisit` `0.070%`, `rapid_interrupt_pan` `0.058%`, `y_zoom` `0.083%`, `progressive_zoom_in` `0.323%` (deep raw step, identical to x-tree).
  - `random_walk`: `cold_overview` `0.114%`, `lateral_pan` `0.168%`, `jump_and_revisit` `0.184%`, `rapid_interrupt_pan` `0.138%`, `y_zoom` `0.186%`, `progressive_zoom_in` `0.224%`.

### Verification

- `cargo test --workspace` (including 4K, 200K, and `root_cap = 16` conformance suites + `SparseCell` roundtrip test): all passed.
- `cargo clippy ... --all-targets -- -D warnings` and `cargo check ... --target wasm32-unknown-unknown`: zero warnings/errors.
- Report diff against baseline `benchmarks/results/latest.json`: **0 diffs** across `naive_full_download`, `paged_moment_tree`, `moment_kdtree`, and `moment_quadtree`.

---

## Findings

1. **Sparse 16-byte planes with inline per-node directories dominate dense planes on every metric.** Cutting cell size by 33% (`24 → 16 B`), omitting empty bins, pruning Level 0 and `k > 8` near-overview planes, and packing into 4 objects cut storage overhead by more than half (`+77.8% → +29.7–34.5%`) while slashing `HeavyWhiteNoise` wire **3.5–4.5×** (`cold_overview` `247 KiB`, beating k-d tree's `875 KiB` by 3.5×) and `MediumWhiteNoise` `cold_overview` **7.8×** (`279 KiB`).
2. **Client-side traversal wins are orthogonal and cut windowed latency ~2–3×.** Jumping directly to `ℓ*`, skipping `visible_y_range` on explicit `y_bounds`, concurrent boundary subtree reads, cross-`k` cache reuse, and cross-`k` promotion halved latency on both signal families (`Heavy` `lateral_pan`: `222 → 101 ms`; `HeavyWhiteNoise` `jump_and_revisit`: `231 → 77 ms`).
3. **`HeavyWhiteNoise` `y_zoom` latency dropped 5.5× (`990 → 180.5 ms`) and requests 11.7× (`979 → 84`)**, because `auto_y` and `zoom y2x` stay in sparse Level-3 planes (`k = 7, 9`), avoiding the alternating raw-leaf cascade. Deep `y8x`/`y32x` zoom on uncorrelated white noise still falls back to the `11.4 MiB` raw x-window because `points.bin` is x-ordered and `k = 10..12` exceeds the 16 pts/bin cap.

## Next Hypotheses

1. **Direct head-to-head with the shape-matched sparse grid pyramid** once ready on `harness`, especially on `HeavyWhiteNoise` `y_zoom` (`y8x`/`y32x` where y-indexed slicing can avoid reading off-screen y bands) vs `random_walk` and `lateral_pan`.
2. **Y-subrange reads on sparse planes:** Because each block's `SparseCell`s are already sorted by `bin`, adding a small per-block y-checkpoint (or binary-searchable band index for deep `k ≥ 10`) could let the ripmap serve `y8x`/`y32x` without downloading off-screen bins.
