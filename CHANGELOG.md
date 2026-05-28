# Changelog

All notable changes to this project will be documented in this file.

This project follows the spirit of [Keep a Changelog](https://keepachangelog.com/)
and uses semantic versioning while the public API settles.

## [Unreleased]

### Changed

- **Design polish pass (Phase 4b-7).** Complete styles refresh moving the
  desktop to a refined dark editorial theme: high-contrast charcoal-green
  surfaces so the cubiomes biome map carries the visual weight, a single
  accent colour (cubiomes-leaf green) used sparingly for primary actions
  and active state, a consistent 6/10/14/20-px spacing scale, tabular
  numerals in the status bar so digit changes don't jitter the layout, and
  pin/spawn glyphs that read clearly on the dark map. Removed the three
  non-functional inspector tabs (Finder/Results/Analyzer) and added an
  inline body background to index.html so the dark surface paints before
  the bundle loads — no white flash on app launch.

### Added

- **Share links + tile screenshot export (Phase 4b-6).** New "Share" panel
  in the inspector with three actions:
  - *Copy share link* — encodes the full current setup (edition, version,
    count, max_matches, the entire conditions tree, plus the selected
    seed + map view if any) as a base64 JSON blob and copies it to the
    clipboard. Paste it back later (the "Paste a share link" textarea
    auto-imports on blur) to recreate the exact session.
  - *Download tile PNG* — saves the currently-rendered biome tile to disk,
    named after seed/center/scale.
- `wireToNode` reverse-serializer in `conditions.tsx` rebuilds the UI tree
  from the JSON shape with fresh IDs (no collisions on import).

- **Visual condition builder (Phase 4b-5).** The sidebar's hardcoded
  structure+distance form is replaced by a recursive tree editor that builds
  the `conditions` spec without hand-editing JSON. Every node has a type
  picker spanning all three logic gates (`all_of` / `any_of` / `none_of`) and
  every leaf (`nearby_structure`, `cluster`, `spawn_biome`, `nearby_biomes`,
  `biome_area`); type changes morph the node, preserving compatible fields.
  Biome leaves get a chip-style picker backed by a curated TS biome catalog
  + the same group quick-picks as the Python `BIOME_GROUPS`.
- New `desktop/src/biomes.ts` mirroring the relevant subset of
  `src/mcseedfinder/biomes.py` (~40 biomes + 7 quick-pick groups).
- New `desktop/src/conditions.tsx` — typed `TreeNode` model, `nodeToWire`
  serializer, `ConditionBuilder` component, recursive `NodeView`, biome
  multi-select with chips.
- `Max matches` is now a sidebar field (was hardcoded at 25).

- **Biome criteria in native desktop search (Phase 4b-4).** The Rust
  `conditions` evaluator gains biome leaves (`spawn_biome`, `nearby_biomes`,
  `biome_area`) that mirror the Python `SpawnBiome` / `NearbyBiomes` /
  `BiomeArea` semantics 1:1 (same grid sampling, same `all_required` /
  `min_samples` logic). Biomes are passed as numeric cubiomes IDs on the
  wire to avoid duplicating the Python biome catalog in Rust.
- New `evaluate_with_biomes` and `find_matches_range_with_biomes` that thread
  a `BiomeBackend` through the recursive evaluator; one backend is allocated
  per search, re-applying the seed only when it changes.
- `has_biome_leaves(node)` helper so callers route correctly: pure-structure
  trees keep the cheap structure-only `evaluate` path; biome-touching trees
  use the biome-aware variant.
- Tauri `start_search` now accepts conditions trees containing biome leaves;
  `run_search` picks the evaluator based on `has_biome_leaves` and reports
  both `structures` and `biomes` as `exact` in the `SeedReport.exactness`.
- 5 new Rust tests in `conditions::biome_eval_tests` (parity with the existing
  Python `test_biome_area_finds_ocean` / `test_provider_marks_biomes_exact`).

- **Import seed from `level.dat` (Phase 4b-3).** The Tauri `import_level_dat`
  command parses a Minecraft world's gzipped NBT save header (via `fastnbt` +
  `flate2`) and returns its seed, version name, and level name. Handles both
  modern (`Data.WorldGenSettings.seed`) and legacy (`Data.RandomSeed`)
  locations. The sidebar gains an "Import world" file picker that ships the
  bytes through with no file-dialog plugin needed; the imported seed appears
  in the results list and the biome map renders immediately.

- **Pan/zoom map + structure pins (Phase 4b-2).** The biome tile is now an
  interactive map: drag-to-pan (pointer events with capture, CSS transform
  during drag, refetch on release), zoom in/out across all five cubiomes
  scales (1, 4, 16, 64, 256), and a recenter button. Coloured pins for
  villages, outposts, ocean monuments, woodland mansions, and ring-1
  strongholds are queried in batch from the new `list_structures_in_view`
  Tauri command and overlaid at their exact block coordinates. Hovering a
  pin shows its structure name + coordinates.
- `list_structures_in_view` Tauri command returning every placement of the
  requested structure types whose block coordinate falls inside a given
  rectangle.

- **Desktop biome tile rendering (Phase 4b-1).** The Tauri `render_tile`
  command now returns a real base64-encoded PNG of the cubiomes biome map for
  a given seed. Implemented as `BiomeBackend::render_tile_png` /
  `render_tile_base64` in the core, using cubiomes `genBiomes` (batched) and
  the cubiomes RGB colormap, encoded with the `png` crate. The React side
  fetches the tile when a result seed is selected and displays the actual
  biome map in the centre pane (pixelated rendering for crispness).
- C shim helpers `mcsf_gen_biomes` (batched biome fill) and
  `mcsf_init_biome_colors` (256-entry RGB colormap).
- Root proxy `package.json` so `npm run dev` / `npm run build` work from the
  repo root.

- **Desktop app: real engine wiring with streamed events (Phase 4a).** The
  Tauri Rust process now depends on `mcseedfinder-core` (path dep, `biomes`
  feature, no PyO3) and runs an actual search in a worker thread instead of
  fabricating dummy results. Matches stream to the React UI via Tauri events
  (`search-started` / `search-match` / `search-progress` / `search-completed`),
  resolving the documented sync-vs-event-stream mismatch.
- Cooperative cancellation via per-job `AtomicBool` shared with the worker
  thread; `cancel_search` flips it and the next chunk boundary emits a
  `cancelled` completion event.
- `analyze_seed` returns a real report: exact origin biome (cubiomes), the
  nearest village's block coords, and the ring-1 stronghold positions.
- New `structures::iter_structures_in_radius` (matches the canonical region
  walk used elsewhere) so structure iteration in Rust mirrors Python.
- Frontend (`main.tsx`) consumes the new event stream — results stream in,
  the run/cancel buttons reflect live state, and "Analyze" calls into Rust
  for the exact origin biome of the selected seed.

### Deferred (Phase 4a-2 / 4b)

- Biome criteria in desktop *search* specs are rejected with a clear typed
  error for now; the Python engine retains full biome support. Biome
  evaluation in Rust is a separate piece of work.
- `render_tile` (cubiomes colormap → base64 PNG) and `import_level_dat`
  (NBT seed reader via fastnbt) remain "not_yet_implemented" responses.
- `pause`/`resume` are no-ops; cancellation works.

### Added

- **Native structure-only conditions-tree fast-path.** The Rust core gains a
  `conditions` module that evaluates the structure-only subset of the Python
  `conditions` tree (NearbyStructure + StructureCluster + all_of/any_of/none_of)
  natively. Spec is serialised once per worker as JSON over PyO3; per-seed
  evaluation never allocates or runs string comparisons. The worker pool prefers
  this richer path over the legacy flat-structure path, and falls back to Python
  evaluation only when a biome criterion is anywhere in the spec.
- `structures::count_structures_in_radius` helper (mirrors the canonical region
  walk used by `has_structure_in_radius`) so cluster predicates and "any in
  radius" predicates cannot disagree about a single placement.
- Rust/Python parity tests for flat structure, logic gates, cluster, none_of,
  biome-mixed fallback routing, and unknown-structure rejection.

### Architecture

- Evaluator splits cleanly into a **prefilter** (structure RNG, native; in
  scope for the Phase 6 wgpu compute shader) and a **confirmation** layer
  (biome conditions, Python + cubiomes for now). Adding biome-native evaluation
  in a later phase only extends the `conditions::Node` enum without touching
  the existing layers.

- **Advanced condition system.** New recursive ``conditions`` tree in the JSON
  schema with three combinators (``all_of`` / ``any_of`` / ``none_of``) and new
  leaf criteria: ``cluster`` (multi-structure / quad-hut style, generalized from
  Cubiomes Viewer's specialized quad-hut generator), ``biome_area`` (minimum
  sample count of a biome set within a region — meaningful now that biomes are
  exact). Backward compatible: flat keys (``spawn_biome``, ``nearby_structures``,
  ``nearby_biomes``) continue to work and AND with the tree.
- ``StructureCluster``, ``BiomeArea``, and ``GroupCriterion`` classes in
  ``criteria.py`` with cost-ordered short-circuit evaluation inside groups.
- ``examples/quad_hut.json`` and ``examples/village_or_outpost_with_plains.json``.
- **Exact biome generation via cubiomes.** Cubitect's `cubiomes` C library is
  vendored as a git submodule (`crates/mcseedfinder-core/vendor/cubiomes`) and
  compiled with the `cc` crate plus a small hand-written FFI shim (`csrc/shim.c`)
  — no bindgen, so the build is reproducible across clang/libclang versions.
- `CubiomesBiomeBackend` exposed through the `_native` extension, implementing
  the existing `BiomeGenerator` protocol (`get_biome(world_seed, x, z)`), plus a
  `HAS_CUBIOMES` flag and `rust_backend.make_biome_backend()` helper.
- Biome backend is threaded from `SearchSpec` version/dimension down through the
  criteria, finder (per-worker), and engine layers; `CriteriaSet.uses_exact_biomes`
  reports whether exact biomes are active.
- `--dimension {overworld,nether,end}` CLI flag; `--show-seed` now reports the
  origin biome exactly when cubiomes is available.
- Integration-regression biome vectors (Rust and Python) for MC 1.21.
- `cargo` feature `biomes` (default-on) gating the cubiomes build; build with
  `--no-default-features` for the pure-Rust structure/RNG core with no C toolchain.

### Changed

- Biome filtering is now **exact** by default (cubiomes), not approximate. The
  Perlin climate-noise generator remains a fallback when the extension is built
  without cubiomes. `SeedReport.exactness` marks biome features `exact` accordingly.
- The Java provider now accepts the overworld, nether, and end dimensions.
- Building the native extension requires a C compiler and the cubiomes submodule
  (`git submodule update --init`).

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
