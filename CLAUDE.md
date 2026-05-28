# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`mc-seed-finder` is a local-first Minecraft Java seed search toolkit: a Python CLI (`src/mcseedfinder/`), an accuracy-critical Rust core with PyO3 bindings (`crates/mcseedfinder-core/`), and a Tauri 2 + React desktop MVP (`desktop/`). Python conventions are in `@AGENTS.md`.

## The accuracy rule (most important)

Never present approximate worldgen as exact. Structure/stronghold placement and `java.util.Random` math are exact and verified against golden vectors. Biome filtering is exact via the vendored cubiomes C library (compiled into the native extension); it falls back to the approximate climate-noise generator only on builds without cubiomes (`--no-default-features`, or `HAS_CUBIOMES == False`). `SeedReport.exactness` and `CriteriaSet.uses_exact_biomes` track which is active — any output, label, or docstring must preserve this distinction. "Structure placement" means the chunk where Minecraft *attempts* placement — it does not prove in-world validity.

`java_random.py`/`java_random.rs` and `structures.py`/`structures.rs` are accuracy-critical. Their golden test vectors must not drift: only update a canonical vector when behavior changes *intentionally*, because accidental drift silently invalidates every search result. The Python and Rust paths must stay at parity (a test enforces this when the native extension is built).

## Build / test commands

- The native extension compiles the vendored cubiomes C sources, so a C compiler is required and the submodule must be present: `git submodule update --init --recursive`. No bindgen/libclang dependency (hand-written FFI in `crates/mcseedfinder-core/src/biomes.rs` + `csrc/shim.c`).
- Install editable (builds the native extension via maturin): `pip install -e ".[dev]"`.
- After changing Rust, rebuild the extension for Python tests. Without maturin: `cargo build --features "pyo3/extension-module"` then copy `target/debug/lib_native.so` → `src/mcseedfinder/_native.abi3.so`.
- Run without installing: prefix source-tree commands with `PYTHONPATH=src`, e.g. `PYTHONPATH=src python -m mcseedfinder --show-seed 1`.
- Python tests: `PYTHONPATH=src python -m unittest` (or `pytest`). Tests are `unittest` classes, `test_*` methods, pytest-compatible.
- Rust tests **require the feature flag**: `cargo test --features pyo3`. Plain `cargo test` will not exercise the bindings.
- Rust benchmark: `RUSTC_WRAPPER= cargo run --release --example bench_structure_search -- 100000`.
- Desktop checks: `cd desktop && npm run build && RUSTC_WRAPPER= cargo check --manifest-path src-tauri/Cargo.toml`.

Prefix Rust commands with `RUSTC_WRAPPER=` to disable `sccache` when it is configured but cannot run in the sandbox/CI. `/verify-all` runs the full three-language gate.

## Conventions and gotchas

- Keep matching seeds on **stdout** and all diagnostics/progress on **stderr** — results are meant to be piped into scripts.
- Core runtime is **stdlib-only** (zero runtime deps). Keep new dependencies out of core; `cubiomes-py` stays behind the `accurate` extra.
- Conventional Commits: `type(scope): imperative summary` (e.g. `fix(structures): correct monument spacing`). Call out any accuracy, biome-fallback, or CLI-compatibility impact in PRs.
- Java Edition 1.18+ is the only implemented target. Bedrock exists in the API but fails explicitly — do not stub it as working.
