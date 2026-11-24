#!/usr/bin/env python3
"""Project developer CLI for pink-lemonade.

Examples:
  ./dev benchmark --fast        # Fast incremental build + Light & Medium JSON export (~1-2s warm)
  ./dev benchmark               # Full scenarios + re-execute benchmarks/analysis.ipynb
  ./dev benchmark --fast --render
  ./dev notebook                # Launch JupyterLab on benchmarks/analysis.ipynb
  ./dev test                    # Run all Rust workspace tests
  ./dev clippy                  # Run clippy on progressive crates
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUST_MANIFEST = ROOT / "rust" / "Cargo.toml"
NOTEBOOK = ROOT / "benchmarks" / "analysis.ipynb"
DEFAULT_OUT = ROOT / "benchmarks" / "results" / "latest.json"


def venv_python() -> Path | None:
    for candidate in (
        ROOT / ".venv" / "Scripts" / "python.exe",
        ROOT / ".venv" / "bin" / "python",
    ):
        if candidate.exists():
            return candidate
    return None


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def render_notebook() -> None:
    py = venv_python()
    if py is None:
        print(
            ".venv not found; create it with:\n"
            "  py -m venv .venv\n"
            "  .venv\\Scripts\\python.exe -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)
    run(
        [
            str(py),
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            str(NOTEBOOK),
        ]
    )


def cmd_benchmark(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    out_path = Path(args.out)

    cargo_cmd = [
        "cargo",
        "run",
        "--profile",
        "fast",
        "-p",
        "pink-lemonade-progressive-tree",
        "--example",
        "export_reports",
        "--manifest-path",
        str(RUST_MANIFEST),
        "--",
        "--out",
        str(out_path),
    ]
    if args.fast:
        cargo_cmd.append("--fast")

    run(cargo_cmd)

    should_render = args.render if args.fast else (not args.no_notebook)
    if should_render:
        render_notebook()

    elapsed = time.perf_counter() - t0
    rel_out = os.path.relpath(out_path, ROOT)
    print(f"\nDone in {elapsed:.2f}s -> {rel_out}")
    if not should_render and not args.open:
        print("Tip: re-run cells in benchmarks/analysis.ipynb or pass --render / --open.")

    if args.open:
        cmd_notebook(args)


def cmd_notebook(_args: argparse.Namespace) -> None:
    py = venv_python()
    if py is None:
        print(
            ".venv not found. Install requirements into .venv first.",
            file=sys.stderr,
        )
        sys.exit(1)
    run([str(py), "-m", "jupyterlab", str(NOTEBOOK)])


def cmd_test(_args: argparse.Namespace) -> None:
    run(["cargo", "test", "--workspace", "--manifest-path", str(RUST_MANIFEST)])


def cmd_clippy(_args: argparse.Namespace) -> None:
    run(
        [
            "cargo",
            "clippy",
            "-p",
            "pink-lemonade-progressive-core",
            "-p",
            "pink-lemonade-progressive-harness",
            "-p",
            "pink-lemonade-progressive-naive",
            "-p",
            "pink-lemonade-progressive-common",
            "-p",
            "pink-lemonade-progressive-tree",
            "-p",
            "pink-lemonade-progressive-kdtree",
            "-p",
            "pink-lemonade-progressive-ripmap",
            "-p",
            "pink-lemonade-progressive-quadtree",
            "-p",
            "pink-lemonade-progressive-grid",
            "-p",
            "pink-lemonade-progressive-summaries",
            "--all-targets",
            "--manifest-path",
            str(RUST_MANIFEST),
            "--",
            "-D",
            "warnings",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="./dev",
        description="Developer CLI for pink-lemonade progressive benchmarks & tooling.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_bench = sub.add_parser("benchmark", help="Run progressive benchmarks and export JSON")
    p_bench.add_argument(
        "--fast",
        action="store_true",
        help="Fast iteration mode (Light + Medium scenarios, skip headless notebook render)",
    )
    p_bench.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Output JSON report path (default: benchmarks/results/latest.json)",
    )
    p_bench.add_argument(
        "--render",
        action="store_true",
        help="Re-execute benchmarks/analysis.ipynb even with --fast",
    )
    p_bench.add_argument(
        "--no-notebook",
        action="store_true",
        help="Skip re-executing benchmarks/analysis.ipynb on full benchmark runs",
    )
    p_bench.add_argument(
        "--open",
        action="store_true",
        help="Open benchmarks/analysis.ipynb in JupyterLab after exporting",
    )
    p_bench.set_defaults(func=cmd_benchmark)

    p_nb = sub.add_parser("notebook", help="Launch JupyterLab on benchmarks/analysis.ipynb")
    p_nb.set_defaults(func=cmd_notebook)

    p_test = sub.add_parser("test", help="Run all Rust workspace tests")
    p_test.set_defaults(func=cmd_test)

    p_clippy = sub.add_parser("clippy", help="Run clippy (-D warnings) on progressive crates")
    p_clippy.set_defaults(func=cmd_clippy)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
