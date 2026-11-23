# 2026-11-22 — Ripmap v2 Hardening: Data-Driven `k` Caps, Per-Node Level Jumping for `BurstyX`, and Explorer v2 (Phase B + C)

## Context & Goal

Two real weaknesses in `moment_ripmap` v2 (`PLPRIPM2`, [previous entry](./2026-11-22-ripmap-v2-sparse-planes.md)) surfaced when tested outside the default 16 px / uniform-x regime ([head-to-head](./2026-11-22-ripmap-v2-vs-sparse-grid.md) and [Phase A](./2026-11-22-scenarios-small-kernel-and-bursty-x.md)):

1. **Screen-tuned `k` caps failed at σ = 4 px:** `MAX_STORED_K_COARSE = 8` capped Level 4 (`8,192` pts/block) at `k = 8`. On `heavy_white_noise_sigma4`, the 1,221 Level-4 overview blocks needed `k = 9` (294 bins), missed by one level, and expanded to 4,883 Level-3 blocks (`9.38 MiB / 41 req / 744 ms`).
2. **Global `max_w` level jump broke `BurstyX`:** `jump_to_initial_level` took `max_w` across *all* overlapping nodes and jumped the entire set together. On `BurstyX`, a single node straddling a 7% empty gap had `w = 8.4σ` all the way down to level 0, dragging all 306 root nodes to level 0 and causing `heavy_bursty` `cold_overview` to read **`25.71 MiB / 107 req / 1,394 ms`** (and `progressive_zoom_in` **`31.85 MiB`**).

In addition, Phase B extracted the shared 16-byte sparse cell and 1-RTT manifest helpers into `progressive-common`, and Phase C updated `AGENTS.md` and the in-browser ripmap explorer (`src/pages/ripmap/ripmap-model.ts`) to model `PLPRIPM2`.

---

## Changes

### Phase B — Shared `progressive-common` refactor (`4f492f7`, 0 diffs)
- Added `QuantizedMoments` (14 B), `SparseCell { bin: u16, moments: QuantizedMoments }` (16 B), `decode_sparse_cells`, `DEFAULT_MANIFEST_READ_BYTES` (128 KiB), `read_manifest`, and `read_small_object` to `pink-lemonade-progressive-common`.
- Switched `progressive-ripmap` onto these helpers and verified **0 diffs** across all 55 reports (330 workload rows) against `benchmarks/results/latest.json`.

### Phase C — Ripmap v2 hardening (`ripmap-phase-c`)
- **Removed screen-tuned `k` caps (`format.rs`, `lib.rs`):** Dropped `MAX_STORED_K_COARSE` / `MAX_STORED_K_WINDOWED`. Every level `ℓ ≥ 1` now uses purely data-derived `k_cap = (k_dense + sparse_extra_depth).min(MAX_Y_DEPTH)` (with `sparse_extra_depth: u8` exposed as a builder parameter, default `2`), bounded by the per-block gate (`non_empty ≤ block_points / min_points_per_bin` ⇒ `≤ N` bytes per plane).
- **Per-node level jumping (`client.rs`):** Replaced uniform `jump_to_initial_level` with `jump_to_initial_frontier`, partitioning nodes at each level into `ready` (`w ≤ 0.9·t`, or `offscreen_y` when explicit `y_bounds` is set) vs `wide` (`w > 0.9·t`) and jumping only `wide` nodes by their own `drop`. Narrow burst blocks stay at the root while only sparse background / gap-straddling nodes descend.
- **Level-scaled boundary bound (`client.rs`):** Replaced the fixed `16σ` boundary fast-path threshold with `(2u64 << cur_level) as f64` (`2 · 2^cur_level`, the Brownian leaf-span bound for `4^cur_level` leaves).
- **Explorer & docs:** Updated `src/pages/ripmap/ripmap-model.ts`, `ripmap-page.ts`, `test/ripmap-model.test.ts` (14/14 passing), and `AGENTS.md` to match `PLPRIPM2`. Added `bursty_x_overview_stays_compact` regression test.

---

## Measured Results — Before vs After Phase C & `sparse_extra_depth` Ablation

All runs use the exact Phase A harness scenarios (10M points = `228.88 MiB` raw, broadband 30 ms / 50 Mbps / 6 streams, 16 MiB budget). Cells are `wire / completed requests / mean latency`.

### Storage & build (10M)

| Scenario (10M) | Ripmap v2 (Phase A) | Phase C (`extra = 1`) | **Phase C (`extra = 2`, default)** |
| --- | ---: | ---: | ---: |
| `heavy` / `heavy_sigma4` (`random_walk`) | `307.86 MiB` (`+34.5%`) | `324.01 MiB` (`+41.6%`) | **`324.01 MiB` (`+41.6%`, ~2.0 s)** |
| `heavy_white_noise` / `..._sigma4` (`noisy_spikes`) | `296.95 MiB` (`+29.7%`) | `303.15 MiB` (`+32.4%`) | **`316.44 MiB` (`+38.3%`, ~1.4 s)** |
| `heavy_bursty` (`bursty_x`) | `307.86 MiB` (`+34.5%`) | `323.98 MiB` (`+41.6%`) | **`323.98 MiB` (`+41.6%`, ~1.8 s)** |

### `heavy_white_noise_sigma4` (10M `noisy_spikes`, σ = 4 px)

| Workload | Ripmap v2 (Phase A) | Phase C (`extra = 1`) | **Phase C (`extra = 2`)** |
| --- | --- | --- | --- |
| `cold_overview` | `9.38 MiB / 41 / 744 ms` | `2.28 MiB / 6 / 252 ms` | **`2.28 MiB / 6 / 252 ms`** |
| `progressive_zoom_in` | `22.29 MiB / 126 / 302 ms` | `66.10 MiB / 366 / 711 ms` | **`16.98 MiB / 98 / 250 ms`** |
| `lateral_pan` | `44.17 MiB / 252 / 462 ms` | `44.18 MiB / 252 / 464 ms` | **`44.18 MiB / 252 / 464 ms`** |
| `jump_and_revisit` | `76.81 MiB / 415 / 872 ms` | `76.82 MiB / 415 / 874 ms` | **`76.82 MiB / 415 / 874 ms`** |
| `rapid_interrupt_pan` | `12.84 MiB / 70 / 873 ms` | `12.84 MiB / 70 / 875 ms` | **`12.84 MiB / 70 / 875 ms`** |
| `y_zoom` | `12.83 MiB / 70 / 145 ms` | `12.84 MiB / 70 / 146 ms` | **`12.84 MiB / 70 / 146 ms`** |

### `heavy_bursty` (10M `bursty_x`, σ = 16 px)

| Workload | Ripmap v2 (Phase A) | **Phase C (`extra = 2`)** | Phase A `sparse_grid` (ref) |
| --- | --- | --- | --- |
| `cold_overview` | `25.71 MiB / 107 / 1,394 ms` | **`229.3 KiB / 22 / 194 ms`** | `114.2 KiB / 5 / 76 ms` |
| `progressive_zoom_in` | `31.85 MiB / 141 / 322 ms` | **`769.0 KiB / 67 / 133 ms`** | `599.0 KiB / 55 / 92 ms` |
| `lateral_pan` | `4.27 MiB / 84 / 132 ms` | **`314.4 KiB / 51 / 116 ms`** | `515.1 KiB / 87 / 93 ms` |
| `jump_and_revisit` | `1.81 MiB / 195 / 234 ms` | **`273.4 KiB / 59 / 158 ms`** | `549.3 KiB / 55 / 71 ms` |
| `rapid_interrupt_pan` | `220.9 KiB / 13 / 146 ms` | **`93.1 KiB / 14 / 221 ms`** | `144.8 KiB / 15 / 99 ms` |
| `y_zoom` | `2.03 MiB / 101 / 133 ms` | **`586.1 KiB / 27 / 66 ms`** | `5.94 MiB / 35 / 81 ms` |

### `heavy_white_noise` (10M `noisy_spikes`, σ = 16 px) & `medium_white_noise` (`extra = 1` vs `2`)

- `heavy_white_noise` `progressive_zoom_in`: `extra = 1` regresses to **`2,315.9 KiB / 158 req / 209 ms`**, vs **`1,131.8 KiB / 68 req / 126 ms`** at `extra = 2`.
- `medium_white_noise` `progressive_zoom_in`: `extra = 1` regresses to **`3,170.8 KiB / 149 req / 180 ms`**, vs **`2,190.3 KiB / 55 req / 98 ms`** at `extra = 2`.
- `heavy` (`random_walk`) and `heavy_sigma4` (`random_walk`) are identical between `extra = 1` and `extra = 2` (continuous walks exceed `max_bins` at both `+1` and `+2` extra depths; only the `+5.5 KiB` manifest growth from extra root plane slots shows up in wire).

---

## Findings

1. **Data-driven `k` caps fix σ = 4 px overview completely.** `heavy_white_noise_sigma4` `cold_overview` drops **4.1× in wire (`9.38 → 2.28 MiB`)**, **6.8× in requests (`41 → 6`)**, and **3.0× in latency (`744 → 252 ms`)**—matching our ~2.1 MiB prediction and beating Phase A's `sparse_grid` (`6.13 MiB / 257 ms`).
2. **Per-node level jumping fixes `BurstyX` by 10–115×.** `heavy_bursty` `cold_overview` drops from **`25.71 MiB → 229.3 KiB` (115×)**, `progressive_zoom_in` from **`31.85 MiB → 769.0 KiB` (42×)**, and all four windowed workloads (`lateral_pan`, `jump_and_revisit`, `rapid_interrupt_pan`, `y_zoom`) drop **2.4–13.9×**—now beating Phase A's `sparse_grid` in wire on all four windowed workloads (including 10× less wire on `y_zoom`: `586 KiB` vs `5.94 MiB`).
3. **`SPARSE_EXTRA_DEPTH = 2` beats `1` decisively.** On `random_walk`/`bursty_x` it costs **0 B** of extra storage (`324.01 MiB` vs `324.01 MiB`). On `noisy_spikes` it costs `+13.3 MiB` (`+5.8%` of raw) and cuts `progressive_zoom_in` wire **2.0× at σ = 16 px** and **4.0× at σ = 4 px** (`66.1 → 17.0 MiB`).

## Verification

- `cargo test -p pink-lemonade-progressive-ripmap`: 6/6 tests passed (including `bursty_x_overview_stays_compact` and `root_cap = 16` conformance).
- `cargo clippy -p pink-lemonade-progressive-ripmap --all-targets -- -D warnings` & `cargo check -p pink-lemonade-progressive-ripmap --target wasm32-unknown-unknown`: clean.
- `vitest run test/ripmap-model.test.ts`: 14/14 tests passed; zero TypeScript errors in `src/pages/ripmap/*`.

## Next Step

Merge with Claude's Phase C `progressive-grid` branch and run the full head-to-head evaluation across σ = 16 px, σ = 4 px, and `BurstyX` against the Turn 4 decision rule.
