# AGENTS.md

Guide for coding agents working in `pink-lemonade`.

## Project Overview

`pink-lemonade` is a WebGPU/WASM visualization library and research artifact for massive 2D time-series KDE / line-plot rendering, plus a Rust workspace for developing progressive, screen-resolution-driven loading over static byte-range storage.

### Repository Layout

- `src/` — TypeScript WebGPU chart (`GaussianChart`, `ViewManager`, `createGaussianChart`), WASM wrappers (`src/downsampling/`), and demo/paper pages (`src/pages/`).
- `test/` — Vitest unit & browser tests.
- `rust/` — Cargo workspace (`rust/Cargo.toml`, nightly toolchain pinned in `rust/rust-toolchain.toml`):
  - `rust/` (`pink-lemonade-wasm`) — Existing in-memory WASM downsamplers (`merge`, `tree_merge`, `runnalls`, `salmond`), built into `pkg/` by `npm run wasm`.
  - `rust/progressive-core/` (`pink-lemonade-progressive-core`) — Lean, WASM-compatible core types (`Point`, `ViewQuery`, `Frame`, `ClientConfig`) and `dyn`-compatible async traits (`ByteRangeStore`, `ObjectWriter`, `DatasetBuilder`, `ProgressiveClient`, `ProgressiveStrategy`).
  - `rust/progressive-harness/` (`pink-lemonade-progressive-harness`) — Test/benchmark harness crate meant strictly for `[dev-dependencies]`: `MemoryStore`, `DiskStore`, `SimulatedTransport` (`NetworkProfile`), `TrackingAllocator`, streaming fixtures (`DatasetStream`), moment & density NMAE evaluators, conformance suite (`run_conformance_suite`), standard scenarios (`benchmarks::run_on`, `benchmarks::compare`), and JSON report export.
  - `rust/progressive-naive/` (`pink-lemonade-progressive-naive`) — Simplest full-download baseline strategy.
  - `rust/progressive-tree/` (`pink-lemonade-progressive-tree`) — Paged multi-level moment tree strategy (`PagedTreeStrategy`) and the benchmark exporter (`examples/export_reports.rs`).
- `benchmarks/` — Exported reports (`benchmarks/results/latest.json`) and Jupyter notebook (`benchmarks/analysis.ipynb`).
- `requirements.txt` & `.venv/` — Root Python dependencies and virtual environment (`.venv/` gitignored).
- `dev.py` (+ `dev.ps1`, `dev.cmd`, `dev`) — Cross-shell developer CLI (`./dev ...`).

## Common Commands

### Progressive Rust Workflow (via `./dev`)
```powershell
./dev benchmark --fast            # ~1s warm: [profile.fast], Light (10K) + Medium (1M) -> benchmarks/results/latest.json
./dev benchmark --fast --render   # Same, plus headless re-execution of benchmarks/analysis.ipynb
./dev benchmark                   # Full scenarios (Light, Medium, ConstrainedMobile, Heavy [non-naive]) + re-render notebook
./dev notebook                    # Launch JupyterLab on benchmarks/analysis.ipynb using .venv
./dev test                        # Run all Rust workspace tests
./dev clippy                      # Run clippy (-D warnings) on all progressive crates
```

### Rust / WASM Direct Commands
```powershell
cargo test --workspace --manifest-path rust/Cargo.toml
cargo check -p pink-lemonade-wasm -p pink-lemonade-progressive-core -p pink-lemonade-progressive-tree --target wasm32-unknown-unknown --manifest-path rust/Cargo.toml
```

### Frontend / TypeScript Commands
```powershell
npm run wasm        # Build pink-lemonade-wasm into pkg/ (keep rust/Cargo.toml root package intact)
npm run typecheck   # tsc --noEmit
npm test            # Vitest suite
npm run build       # Full WASM + Vite bundle
```

### Python Environment Setup (if `.venv` is missing)
```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Architecture & Design Conventions

- **Dumb static byte-range server model:** Progressive strategies must encode datasets via `ObjectWriter` into static objects and read them online solely through `ByteRangeStore::read_range(key, offset, len)` (mirroring HTTP `Range` requests on static hosting/object storage).
- **Prefer `dyn` trait objects over generics:** Core traits use `#[async_trait]` (re-exported from `pink-lemonade-progressive-core`) so everything composes via `dyn ByteRangeStore`, `dyn ObjectWriter`, `dyn DatasetBuilder`, `dyn ProgressiveClient`, and `dyn ProgressiveStrategy`. Only introduce generics when dynamic dispatch cannot be used.
- **Pass cheap structs by value:** `ViewQuery`, `Point`, and `ClientConfig` are `Copy`; pass them by value rather than adding lifetime parameters.
- **Keep harness code out of library crates:** Never add test stores, network simulators, or benchmark harnesses to `progressive-core` or strategy crate `src/` directories — put them in `pink-lemonade-progressive-harness` and import under `[dev-dependencies]`.
- **Keep core & strategy crates WASM-ready:** Avoid depending on `tokio`'s runtime (`rt`/`time`) inside `progressive-core` or strategy crates so they compile cleanly for `wasm32-unknown-unknown`.
- **Virtual-time benchmarking & disk safety:** Standard tests and benchmarks run under Tokio virtual time (`#[tokio::test(start_paused = true)]` or `tokio::time::pause()`) with `NetworkProfile` simulating RTT, bandwidth, and concurrency. Do not run multi-hundred-MB / GB benchmarks on the user's physical disk (`DiskStore`) unless explicitly requested.
- **Keep `[profile.release]` intact for WASM:** `rust/Cargo.toml` enables LTO + `codegen-units = 1` for release WASM builds; use `--profile fast` (`opt-level = 2`, incremental) for fast native benchmark iteration.

## Adding a New Progressive Strategy

1. Create a new crate under `rust/progressive-<name>/` and add it to `[workspace] members` in `rust/Cargo.toml`.
2. Depend only on `pink-lemonade-progressive-core` in `[dependencies]`, and add `pink-lemonade-progressive-harness` + `tokio` under `[dev-dependencies]`.
3. Implement `DatasetBuilder`, `ProgressiveClient`, and `ProgressiveStrategy`, ensuring:
   - Exact moment preservation (mass, 1st, 2nd moments) on full and clipped views
   - Exact `y_range` and valid `line_points` (sorted `x`, preserving visible extrema)
   - Strict adherence to `ClientConfig::memory_budget_bytes` via `memory_bytes()`
   - Cancellation safety when a `query()` future is dropped mid-flight
4. Verify with `run_conformance_suite`, `cargo clippy ... -- -D warnings`, `cargo check --target wasm32-unknown-unknown`, and add the strategy to `rust/progressive-tree/examples/export_reports.rs` to compare it in `benchmarks/analysis.ipynb`.
