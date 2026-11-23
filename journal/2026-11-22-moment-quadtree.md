# 2026-11-22 — Classic Moment Quadtree (and in-browser strategy explorers)

## Context & Goal

The moment ripmap ([previous entry](./2026-11-21-2d-strategies-kdtree-ripmap.md)) leads on both signal families, but its dense per-block y-bins waste depth when outliers stretch a block's `[min_y, max_y]`, and it costs 1.78× raw in 44 objects. Before designing a sparse, shape-matched grid, we measured the textbook alternative: a **classic point-region quadtree with exact moments per node**. It is sparse by construction (empty quadrants are not stored, an outlier gets its own small cell) and has no per-block y-range to stretch. The hypothesis under test was that its square cells would fail when time series are zoomed in x only.

## Infrastructure Changes

- **`pink-lemonade-progressive-quadtree`** (new crate, `PLPQUAD1`), added to the workspace, `./dev clippy`, and `export_reports`.
- **`progressive-common`:** `CacheKey::Blob` / `CacheValue::Bytes` (strategy-defined byte pages, decoded on use) and `coalesce_spans`, a gap-bridging coalescer for *variable-length* `(start, len)` ranges (quadtree child pages and leaves have no fixed size).
- **In-browser explorers** (`src/pages/ripmap/`): TypeScript models of the ripmap, x-tree, and quadtree builders and clients drive `?page=ripmap`, `?page=xtree`, and `?page=quadtree`. Each pan or zoom re-runs the client policy and shows per-node decisions, greedy kernel merging (`push_group` / `collect_points`, now modelled exactly), the byte ranges read per object, and the round trips. The models are unit-tested against the Rust layout helpers and for exact mass and `y_range`.

## Strategy Design (`progressive-quadtree`)

- **Builder:** starting from the data bbox, every node with more than `leaf_points = 32` points splits its region at the midpoint into 4 quadrants: squares in normalized data coordinates. Empty quadrants are dropped. Splitting stops at `max_depth = 40` or when all points in a node coincide. Points are permuted into quadtree (Z) order, so every subtree is a contiguous run of `points.bin`; each leaf is sorted by x. Moments are rolled up bottom-up with `Node::combined`.
- **Record:** a 72-byte record per node: the shared 64-byte `PackedNode` (tight bbox, moments, line extrema) plus `first: u32, count: u32`. For an internal node these are its children's index range in the next depth; for a leaf (high bit of `count` set) they are its point range. Each depth is an array ordered by parent, so a node's ≤ 4 children are adjacent.
- **Layout (three objects):** the top depths totalling ≤ `root_cap = 1024` records are inlined in `manifest.bin`; every deeper depth follows the previous one in a single `nodes.bin`, located through a per-depth `(count, byte offset)` table; raw points are in `points.bin`.
- **Client:** the same exactness rules as every other strategy. Pruned exact `y_range` descent, then a depth-by-depth frontier with `standard_decision`: nodes straddling the viewport edge descend, fully-inside nodes resolve when their *tight bbox* fits `w² + h² < t²` (or are off-screen in y), anything else expands down to raw leaves. Child pages are coalesced with 16 KiB gaps / 256 KiB runs, leaves with 256-point gaps / 8K-point runs. Frontier items are ordered by a packed `u128` Z-key before greedy merging. **One round trip per depth** below the inlined top.
- **Verification:** `run_conformance_suite` at 4K, at 200K, and with only the root inlined (exercises per-depth offsets from the first descent); density NMAE ≤ 5% on every standard workload for both signal families at 100K; clippy `-D warnings`; `wasm32-unknown-unknown` check.

---

## Experiment 5 — Quadtree vs x-tree, k-d tree, and ripmap

Scenarios and network are identical to Experiment 4 (10M points, 228.88 MiB raw, broadband 30 ms / 50 Mbps, 16 MiB budget). Cells are `wire / requests / mean latency`; the lowest wire per row is bold.

### Storage & build (10M)

| Strategy | Storage (random walk / white noise) | Overhead vs raw | Objects | Build |
| --- | ---: | ---: | ---: | ---: |
| `paged_moment_tree` | `248.57 MiB` | `+8.6%` | 5 | ~0.7–1.3 s |
| `moment_kdtree` | `250.67 MiB` | `+9.5%` | 5 | ~2.4–2.9 s |
| `moment_ripmap` | `406.99 MiB` | `+77.8%` | 44 | ~1.9–2.7 s |
| `moment_quadtree` | `295.53 / 305.20 MiB` | `+29.1% / +33.3%` | **3** | ~2.6–3.4 s |

The quadtree pays for the 8-byte link in every record and for leaves that are often far from full: a split creates up to four children, whatever their counts.

### `Heavy` — integrated Gaussian noise (`random_walk`)

| Workload | `paged_moment_tree` | `moment_kdtree` | `moment_ripmap` | `moment_quadtree` |
| --- | --- | --- | --- | --- |
| `cold_overview` | `630 KiB / 5 / 107 ms` | `406 KiB / 81 / 506 ms` | **`80 KiB / 6 / 35 ms`** | `264 KiB / 3 / 98 ms` |
| `progressive_zoom_in` | `7.10 MiB / 121 / 179 ms` | `4.90 MiB / 651 / 627 ms` | **`768 KiB / 68 / 151 ms`** | `3.46 MiB / 198 / 360 ms` |
| `lateral_pan` | `3.76 MiB / 313 / 316 ms` | `5.96 MiB / 907 / 713 ms` | **`631 KiB / 128 / 222 ms`** | `3.46 MiB / 217 / 380 ms` |
| `jump_and_revisit` | `10.53 MiB / 1,675 / 1,562 ms` | `6.89 MiB / 992 / 993 ms` | **`646 KiB / 70 / 175 ms`** | `3.03 MiB / 120 / 282 ms` |
| `rapid_interrupt_pan` | `1.17 MiB / 156 / 948 ms` | `1.66 MiB / 234 / 1,332 ms` | **`176 KiB / 20 / 258 ms`** | `886 KiB / 37 / 452 ms` |
| `y_zoom` | `6.41 MiB / 490 / 483 ms` | `2.83 MiB / 429 / 427 ms` | **`762 KiB / 50 / 101 ms`** | `1.63 MiB / 107 / 257 ms` |

### `HeavyWhiteNoise` — uniform noise + `+80` spikes (`noisy_spikes`)

| Workload | `paged_moment_tree` | `moment_kdtree` | `moment_ripmap` | `moment_quadtree` |
| --- | --- | --- | --- | --- |
| `cold_overview` | `248.57 MiB / 1,303 / 13,704 ms` | **`875 KiB / 17 / 189 ms`** | `937 KiB / 3 / 181 ms` | `1.52 MiB / 9 / 253 ms` |
| `progressive_zoom_in` | `435.02 MiB / 2,287 / 4,052 ms` | `19.00 MiB / 508 / 554 ms` | **`4.21 MiB / 162 / 299 ms`** | `26.29 MiB / 708 / 785 ms` |
| `lateral_pan` | `42.89 MiB / 246 / 420 ms` | `45.30 MiB / 866 / 805 ms` | **`2.87 MiB / 116 / 257 ms`** | `61.82 MiB / 1,740 / 1,477 ms` |
| `jump_and_revisit` | `74.60 MiB / 411 / 832 ms` | `78.62 MiB / 842 / 1,232 ms` | **`3.13 MiB / 70 / 231 ms`** | `98.57 MiB / 1,066 / 1,547 ms` |
| `rapid_interrupt_pan` | `12.44 MiB / 69 / 822 ms` | `13.20 MiB / 129 / 1,160 ms` | **`820 KiB / 16 / 307 ms`** | `16.67 MiB / 179 / 1,574 ms` |
| `y_zoom` | **`12.44 MiB / 69 / 137 ms`** | `15.57 MiB / 608 / 622 ms` | `14.16 MiB / 977 / 992 ms` | `23.43 MiB / 982 / 985 ms` |

### Medium (1M) highlights

- **`MediumWhiteNoise` `cold_overview` is the quadtree's only win:** `1.25 MiB / 8 / 216 ms` vs ripmap `2.12 MiB` and k-d `1.97 MiB`. Sparse square cells with exact moments are very efficient while the whole x-range is on screen. Every windowed white-noise workload is the worst of the four (`lateral_pan` `6.71 MiB / 362 req` vs x-tree `4.29 MiB / 43 req`).
- **`Medium` (random walk):** between the x-tree and the k-d tree on the overview (`1.16 MiB`), but last on `lateral_pan` (`1.47 MiB` vs ripmap `203 KiB`), because each depth costs a round trip and leaves are small and scattered (`179` requests vs `58`).

### Ablation — object layout

The first version packed consecutive depths into `nodes_{g}.bin` objects of ≥ 64K records (6 objects at 10M random walk, 5 for white noise). Reads are byte ranges, so object boundaries cannot change what a query reads. Moving to a single `nodes.bin` (manifest `VERSION 2`, 16-byte depth entries instead of 24) left **requests, kernel counts, and latency identical on every workload of every scenario**. Wire dropped by exactly the manifest saving (`56–128 B` per cold open), and every dataset is now 3 objects. A file per depth would have been ~15–20 objects with the same reads.

---

## Findings

1. **The square-cell hypothesis is confirmed.** On white noise, windowed views are the worst of all four strategies: `lateral_pan` downloads `61.8 MiB`, which is **1.4× the x-tree's** full-leaf fallback and **21.5× the ripmap's**. To be narrow enough on screen at a 20× x-zoom, a square cell must also be ~20× shorter than the kernel needs, so a dense band splits into far more cells than pixels. The explorer shows it directly: at 300K white noise and 20× zoom it reads **161.6% of the raw window** in 11 round trips. One leaf there is `2.08σ` wide but only `0.065σ` tall.
2. **On thin curves it is a solid second.** On the 10M random walk it uses **1.1–3.5× less wire than the x-tree** (`jump_and_revisit` `10.53 → 3.03 MiB`) and fewer requests than the k-d tree, but **2.2–5.6× more than the ripmap**. A random walk occupies few cells per column, so sparsity helps; but the cells still cannot be wider than tall, and the ripmap's σ-wide x-blocks with y-bins remain cheaper.
3. **Depth costs round trips.** One RTT per depth below the inlined top, and fanout ≤ 4, add up: on white noise `lateral_pan` issues `1,740` requests vs `116` for the ripmap; on the random walk `cold_overview` is cheap in requests (3) but not in latency (`98 ms` vs `35 ms`).
4. **Explicit `y_bounds` do not rescue it.** `y_zoom` runs inside a 5% x-window, where cells are already too short. Off-screen culling only collapses what is outside the band; on white noise `y_zoom` is the worst result (`23.43 MiB / 982 req`).
5. **Storage lands between the trees and the ripmap** (1.29–1.33× raw) with the fewest objects of any strategy (3).

## Methodology Notes

- **Layout changes are verified by diffing reports**, not by eye: snapshot `latest.json` and the 10M report, rerun, and compare wire, requests, and kernel counts per workload (exactly deterministic) with latency within 1%. All four strategies other than the one changed must diff to zero.
- **The explorer models are executable documentation, not benchmarks:** they reproduce the decision policy, merging, and byte ranges exactly, but use their own RNG and default to small `root_cap` values so the descent is visible. Wire numbers there are not comparable to the Rust harness.
- **Correction to earlier entries:** 10M × 24 B is 228.88 MiB, not 240 MiB. Storage overheads in the 2026-11-20 and 2026-11-21 entries were recomputed: x-tree `+8.6%`, k-d tree `+9.5%`, ripmap `1.78×` with planes `152.7 MiB` (0.67×).

## Next Hypotheses

1. **Shape-matched sparse grid pyramid:** keep sparsity, but store a few cell aspect ratios (1:1, 1:4, 1:16, …) and pick per view, so a 20× x-zoom uses tall-thin cells. Use a level-per-object CSC layout (cells column-major, a `u32` column-offset array per level) so the client computes the level from the zoom and reads one contiguous range of columns: **~2 round trips regardless of depth**, instead of one per depth. Target: white-noise `lateral_pan` near the ripmap's `2.87 MiB`.
2. **Retire the classic quadtree as a candidate** and keep it as the isotropic-cell baseline in `export_reports`, like the k-d tree is for fixed-shape cells.
3. **Sparse ripmap planes** (from the previous entry) remain open; they address the same outlier-stretch weakness from the ripmap side and should be compared against (1).
