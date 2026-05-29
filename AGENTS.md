# Repository Guidelines

## Project Structure & Module Organization

This repository is now a hybrid Python, Rust, and Tauri desktop project. The
Python package lives in `src/mcseedfinder/`: `cli.py` owns the command line,
`finder.py` and `engine.py` coordinate searches, `criteria.py` compiles filters,
`app_contract.py` defines product-facing request/event/report types, and
`java_random.py`, `structures.py`, `biome_gen.py`, `bedrock.py`, and
`rust_backend.py` cover accuracy-critical worldgen behavior and native fallback
boundaries. The Rust core and PyO3 extension live in
`crates/mcseedfinder-core/`; key modules include RNG and structure placement,
condition trees, biome providers, Bedrock seed handling, GPU prefilters, GPU
biome/noise code, and the vendored cubiomes bridge. The Tauri 2 + React desktop
app lives in `desktop/`, with Rust commands under `desktop/src-tauri/` and the
React/TypeScript UI under `desktop/src/`.

Tests live in `tests/` and cover Python contracts, CLI behavior, Bedrock text
seed handling, biome behavior, condition trees, native Rust interop, Java RNG,
and structure placement. Example search specs live in `examples/`; deeper
accuracy and platform notes live in `docs/`. CI and release automation live in
`.github/workflows/`. Treat `src/mcseedfinder.egg-info/`, `__pycache__/`,
`target/`, `desktop/node_modules/`, `desktop/dist/`, generated native libraries,
and cache directories as generated artifacts.

## Build, Test, and Development Commands

- `git submodule update --init --recursive`: initialize the vendored cubiomes
  submodule required for exact Java biome support.
- `pip install -e ".[dev]"`: install the package in editable mode through
  maturin, building the PyO3 native extension and installing pytest, coverage,
  mypy, and ruff.
- `PYTHONPATH=src python -m mcseedfinder --help`: inspect the CLI without
  relying on installation state.
- `PYTHONPATH=src python -m mcseedfinder --show-seed 1`: quick CLI smoke check.
- `PYTHONPATH=src python -m unittest` or `pytest`: run the Python test suite.
- `ruff check`: run the configured Python lints.
- `python -m mcseedfinder --config examples/village_outpost.json --count 100000 --workers 4 --max-matches 1`:
  exercise a representative configured search.
- `cargo fmt --all -- --check`: verify Rust formatting.
- `cargo clippy --features pyo3 --manifest-path crates/mcseedfinder-core/Cargo.toml --all-targets -- -D warnings`:
  lint the default native core with PyO3 enabled.
- `cargo clippy --no-default-features --features pyo3 --manifest-path crates/mcseedfinder-core/Cargo.toml --all-targets -- -D warnings`:
  verify the no-cubiomes fallback still compiles cleanly.
- `cargo test --features pyo3 --manifest-path crates/mcseedfinder-core/Cargo.toml`:
  run Rust core tests.
- `RUSTC_WRAPPER= cargo check --manifest-path desktop/src-tauri/Cargo.toml`:
  check the Tauri backend without relying on a global rustc wrapper.
- `npm run install:desktop`: install desktop frontend dependencies.
- `npm run build`: build the desktop frontend through the root proxy script.
- `npm run dev`: start the Tauri desktop app for interactive local testing.

For performance checks, use `PYTHONPATH=src python -m mcseedfinder.benchmark
--count 100000` for the Python/native bridge path, and
`RUSTC_WRAPPER= cargo run --release --example bench_structure_search --manifest-path crates/mcseedfinder-core/Cargo.toml -- 1000000`
for the Rust structure-search path.

## Coding Style & Naming Conventions

Use Python 3.10+ features and keep runtime dependencies out of core Python code
unless `pyproject.toml` changes intentionally. Follow the existing Python style:
4-space indentation, `from __future__ import annotations`, type hints on public
functions, `snake_case` functions and modules, `PascalCase` classes, and
`UPPER_SNAKE_CASE` constants. Prefer small pure functions for deterministic
math, and keep comments focused on provenance, reference vectors, exactness
boundaries, or non-obvious algorithm details. Keep matching seeds on stdout and
diagnostics, progress, and reports on stderr so scripts can pipe results safely.

Rust code should stay rustfmt-clean and clippy-clean under both default and
`--no-default-features` builds. Do not add casual `unsafe`; if FFI or GPU buffer
code requires it, document the invariant being upheld. Keep Java RNG, structure,
stronghold, biome, and GPU parity logic backed by canonical vectors or explicit
cross-checks. `.cargo/config.toml` intentionally clears `rustc-wrapper`; preserve
that unless the whole workspace build strategy changes.

The desktop app is the usable tool, not a marketing landing page. Keep the map,
search controls, result stream, analyzer, and status feedback immediately
available. Tauri command and event payloads should stay aligned with
`SearchSpec`, `SearchEvent`, `SeedReport`, and the app contract tests. Check UI
changes for nonblank map tiles, readable controls, and no overlapping desktop
or narrow-viewport text.

## Accuracy Guidelines

Never present approximate or candidate worldgen as exact. Java RNG and
random-spread structure candidate placement are exact and tested. Stronghold
placement is exact candidate placement. CPU biome lookup is exact for Java
1.18+ when the native cubiomes extension is built; no-cubiomes builds must label
biome results as approximate fallback behavior. GPU structure and biome paths
are correctness features as well as performance features: parity tests against
the CPU evaluator and cubiomes-backed tile output are required when touching
WGSL, GPU buffer layouts, condition-tree evaluation, noise, or biome lookup.

Bedrock support currently includes exact text seed hashing to the signed 32-bit
Bedrock seed value. Bedrock worldgen criteria are not implemented and must fail
with clear edition-aware errors rather than producing Java-shaped guesses.
Structure placement identifies candidate chunks where Minecraft attempts a
structure; biome or terrain validity can still reject a candidate in game, so UI
and exports must keep exactness and validation-stage metadata visible.

## Testing Guidelines

Tests use `unittest` classes and `test_*` methods and are pytest-compatible.
Place new tests in `tests/test_<module>.py`. For `java_random.py`,
`structures.py`, Rust RNG/structure modules, cubiomes integration, and GPU code,
add or update canonical vectors whenever behavior changes. For criteria,
condition trees, CLI behavior, and app contracts, include positive and negative
cases and keep seed counts small. For persistence, export, cancellation, or
desktop command work, cover event order and failure modes, not just successful
happy paths.

## Vendored Submodule

`crates/mcseedfinder-core/vendor/cubiomes` is a git submodule used as the exact
Java biome backend. Inspect it separately with
`git -C crates/mcseedfinder-core/vendor/cubiomes status --short --branch` before
touching it. Do not reset, clean, or rewrite local submodule edits unless the
user explicitly asks; local changes inside the submodule are easy to hide from
top-level diffs.

## Commit & Pull Request Guidelines

Use Conventional Commit style, such as `feat(engine): add survivor telemetry` or
`fix(structures): correct monument spacing`. Pull requests should describe the
behavioral change, list commands run, link related issues, and call out any
accuracy impact, biome fallback behavior, Bedrock limitation, desktop command
contract change, or CLI compatibility change. Include example commands or
output when changing user-facing search behavior.

## Public Repository Status

The project is published as the public GitHub repository
`https://github.com/doublegate/MC-SeedFinder`, with local `main` tracking
`origin/main`. As of the Phase 8 baseline, the repository includes performance
foundations, biome-aware native search, batched IPC, additional structures,
version guards, GPU structure prefilters, GPU biome parity work, desktop map and
result UX, CI, and release workflows.
