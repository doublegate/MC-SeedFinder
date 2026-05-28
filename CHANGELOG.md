# Changelog

All notable changes to this project will be documented in this file.

This project follows the spirit of [Keep a Changelog](https://keepachangelog.com/)
and uses semantic versioning while the public API settles.

## [0.1.0] - 2026-05-27

### Added

- Initial Python package using a `src/` layout.
- Command-line seed finder entry point via `mcseedfinder` and
  `python -m mcseedfinder`.
- Java Edition seed search for:
  - nearby random-spread structures,
  - strongholds,
  - approximate spawn biome candidates,
  - approximate nearby biome candidates.
- Accuracy-critical Java RNG implementation matching `java.util.Random`.
- Accuracy-critical Java structure placement implementation for Minecraft
  Java Edition 1.18+ candidate positions.
- Stronghold concentric-ring generation.
- Criteria compiler for JSON and CLI-driven search specifications.
- Multiprocessing Python search engine with chunked seed enumeration.
- Versioned product-facing contracts:
  - `SearchSpec`,
  - `SearchEvent`,
  - `SeedReport`,
  - Java and Bedrock provider interfaces.
- Staged search events with candidate, verified match, rejected-stage summary,
  completion, and cancellation event types.
- SQLite-backed `JobStore` for local-first persisted jobs, events, settings,
  and results.
- `LocalCommandHost` synchronous command contract.
- `AsyncCommandHost` background job runner with:
  - threaded search execution,
  - pause/resume flags,
  - cooperative cancellation,
  - persisted progress and results,
  - export support.
- Rust workspace with `mcseedfinder-core`.
- Rust `JavaRandom` implementation and golden tests.
- Rust random-spread structure placement implementation and golden tests.
- Rust stronghold generation and count tests.
- PyO3 extension module exposed as `mcseedfinder._native`.
- Python `rust_backend` bridge with fallback behavior when the extension is not
  present.
- Native batch filtering for structure-only searches, including strongholds.
- Python benchmark command comparing Python and Rust-backed structure search.
- Rust release benchmark example for core structure filtering.
- Tauri 2 + React + TypeScript desktop MVP scaffold.
- Desktop UI with:
  - left search controls,
  - central map preview,
  - right results and analyzer inspector,
  - bottom status strip.
- Tauri command scaffold for:
  - `start_search`,
  - `pause_search`,
  - `resume_search`,
  - `cancel_search`,
  - `analyze_seed`,
  - `render_tile`,
  - `import_level_dat`,
  - `export_results`.
- Example criteria files in `examples/`.
- Unit tests for RNG, structures, criteria, engine events, CLI helper behavior,
  Rust backend parity, and app command persistence.

### Changed

- Build backend moved to `maturin` so the Python package can build the PyO3
  extension while preserving the Python `src/` layout.
- Structure-only sequential search chunks now route through the Rust backend
  when the native extension is available.
- CLI expanded with `--spec`, `--edition`, `--version`, `--export`, and
  `--export-format`.

### Known Limitations

- Bedrock Edition is not implemented.
- Biome filtering is approximate unless a compatible exact backend is installed
  and used.
- Structure searches report exact candidate placements, not guaranteed in-world
  generation after biome and terrain checks.
- Desktop commands are scaffolded in Rust; the Python async command host has the
  fuller local behavior.
- Map tile rendering and level.dat import are placeholders.
- GPU acceleration is not implemented.
