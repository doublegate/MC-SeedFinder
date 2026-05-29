# Changelog

All notable changes to this project will be documented in this file.

This project follows the spirit of [Keep a Changelog](https://keepachangelog.com/)
and uses semantic versioning while the public API settles.

## [Unreleased]

### Added

- **GitHub Actions CI + release pipeline (Phase 7).** `.github/workflows/ci.yml`
  runs four jobs on every push and PR: `rust-lint` (`cargo fmt --check` plus
  `cargo clippy --all-targets -- -D warnings`, on both default features AND
  `--no-default-features` so the pure-Rust core stays buildable without
  cubiomes), `python-lint` (`ruff check`), `test` (`cargo test --features
  pyo3` + Python `unittest`, with the cubiomes submodule checked out so the
  maturin build succeeds; GPU tests gracefully skip on the headless runner),
  and `desktop-build` (Tauri Linux deps + `npm install` + Vite build +
  `cargo check` on `src-tauri`). Concurrency groups cancel in-progress runs
  when a new commit lands; the cargo registry + target dir are cached keyed
  on `Cargo.lock`. Total wall-clock is around 3 minutes per push.
- **Release workflow** (`.github/workflows/release.yml`) triggers on
  `workflow_dispatch` or a `v*.*.*` tag push and builds: (1) a manylinux2014
  x86_64 maturin wheel for `mcseedfinder` — `pyo3` is configured with
  `abi3-py310` so one wheel covers Python 3.10..3.13 — and (2) a Linux
  x86_64 Tauri AppImage + `.deb` of the desktop app. Tag pushes draft a
  GitHub release with the artifacts attached for the maintainer to review
  and publish. macOS (x86_64 + arm64) and Windows matrix entries are
  present but commented out — they need signing/notarization config that
  isn't yet wired in.
- **Toolchain pin** via `rust-toolchain.toml` (stable + rustfmt, clippy,
  rust-src) so CI and local dev agree on the Rust version.
- **Workspace clippy policy** (`[workspace.lints.clippy]`) allowing
  `too_many_arguments` (FFI shim signatures naturally exceed 7 args),
  `doc_lazy_continuation`, and `doc_overindented_list_items` (aligned-column
  doc bullet lists are more readable than 4-space wrapped). All other lints
  stay at clippy's default severity, enforced as errors by CI.
- **Performance numbers in README refreshed** against the current code: 82k
  seeds/s (Python), 2.4M seeds/s (Rust extension via PyO3), 14.8M seeds/s
  (Rust standalone release-mode benchmark) — ~180× speedup at the upper
  bound. GPU prefilter and GPU biome rendering are validated by parity
  tests rather than seeds/s.

### Changed

- **Phase 7 lint cleanup, made to land cleanly on day one of CI.**
  - Ruff config gained `extend-exclude = ["crates/.../vendor"]` (don't
    lint upstream cubiomes Python scripts) and `per-file-ignores`
    allowing `E501` for `biomes.py` (single-line biome catalog rows are
    more readable than wrapped).
  - 180/194 ruff issues auto-fixed (PEP-585 type annotations, `Optional[X]`
    → `X | None`, import sort, `.format` → f-strings). 3 hand fixes:
    SQL line-wrap in `app_contract.py`; `dict(...)` call → literal in
    `cli.py`; unused stronghold-ring loop var `k` → `_k`.
  - 180/195 clippy issues auto-fixed (`manually_reimplementing_div_ceil`,
    etc.). Surgical fixes: `BiomeTree` gained `is_empty()`
    (`len_without_is_empty`); `pack_perm()` uses `iter_mut().enumerate()`
    (`needless_range_loop`); `SPAWN_BIOME_SAMPLES_PER_AXIS` now
    `#[cfg(feature = "biomes")]`-gated so `--no-default-features` doesn't
    dead-code-warn; PI/E-approximate test coords replaced with neutral
    non-magic values (`3.15`, `2.72`) — the test only needed messy
    coords, not transcendental constants.

### Added

- **Bit-exact GPU biome generation matching cubiomes (Phase 6c).** Ports the
  full Minecraft 1.21 Overworld biome pipeline to WebGPU compute shaders, one
  primitive at a time, each validated against cubiomes' f64 reference via
  GPU↔CPU parity tests. cubiomes is f64 throughout; WebGPU compute is f32-only,
  so "matching" means matching modulo float tolerance at climate boundaries —
  except for the integer-only b-tree walker, which is bit-exact.
  - **Phase 6c-1: WGSL `samplePerlin`.** Hand-written C shim (`mcsf_perlin_init`,
    `mcsf_perlin_sample`) extracts cubiomes' `PerlinNoise` state (a/b/c,
    h2/d2/t2, 256-byte permutation table) so it can be uploaded as a uniform.
    `crates/mcseedfinder-core/src/gpu_noise.{wgsl,rs}` mirrors `samplePerlin`
    step-for-step — fade poly, 16-way indexed-lerp switch, trilinear
    interpolation, zero-y fast path. Permutation table packed into
    `array<vec4<u32>, 16>` with bit-shift extraction. GPU↔CPU parity test
    over 12 diverse coords: max drift **2.9 × 10⁻⁵**.
  - **Phase 6c-2: WGSL `sampleOctave` + `sampleDoublePerlin`.** C shim
    `mcsf_double_perlin_sample` reconstructs cubiomes' `DoublePerlinNoise`
    from packed octave arrays and calls real `sampleDoublePerlin` — true
    oracle, not a re-port. WGSL `cs_double_perlin` entry stacks N octaves
    per A/B half with per-octave amplitude+lacunarity, applies the 337/331
    frequency shift on half B. cubiomes' `maintainPrecision` is a no-op so
    WGSL uses plain multiplication. New `src/gpu_double_perlin.rs`. Parity
    over 8 octaves × 8 coords: max drift **3.6 × 10⁻⁵** — octave summation
    does not compound drift.
  - **Phase 6c-3: Climate noise stack for MC 1.21.** C shim
    `mcsf_climate_init_field` runs cubiomes' `setBiomeSeed` and extracts
    per-field octave state (temperature, humidity, continentalness, erosion,
    shift, weirdness). New `src/gpu_climate.rs` — `ClimateNoise::for_overworld(mc,
    seed, large)` builds six `DoublePerlinSpec`s ready for GPU dispatch.
    Subtle accuracy fix: cubiomes aliases `NP_SHIFT == NP_DEPTH` (same enum
    value 4) and `sampleClimatePara(NP_DEPTH)` runs a composite C/E/W spline,
    NOT the underlying shift double-perlin; CPU reference shim bypasses
    `sampleClimatePara` and goes straight to `sampleDoublePerlin` on the
    requested field. Parity for seed 12345 × 6 coords × 6 fields: max drift
    **2.2 × 10⁻⁵** (continentalness, 18 octaves).
  - **Phase 6c-4: Biome b-tree walker (btree21wd).** Iterative WGSL port of
    cubiomes' recursive `get_resulting_node` (WGSL has no recursion).
    Explicit 6-frame stack with PRE/LOOP/RETURN state machine matches
    cubiomes' control flow exactly. `u64` emulated as `vec2<u32>` (lo, hi)
    with manual add/sub/lt; `np[6]` uploaded as packed 2×u32 pairs.
    Required limit bumped to `max_storage_buffers_per_shader_stage = 6`
    (downlevel default is 4). New `src/gpu_btree.{wgsl,rs}` with both a
    CPU walker (validates data extraction + algorithm port independent of
    WGSL) and the GPU dispatcher. **Bit-exact** match to cubiomes
    `climateToBiome` across all 64 grid points — pure integer arithmetic,
    no tolerance involved.
  - **Phase 6c-5: Full `sampleBiomeNoise` integration.** New
    `src/gpu_biome.rs` orchestrates the four GPU primitives plus a CPU
    pass for the depth-spline (cubiomes' `getSpline` tree, hard to
    parallelize and small per-pixel work). Pipeline: 2× shift dispatch →
    5× climate dispatch → CPU `mcsf_compute_depth` → assemble `np[6]` →
    GPU b-tree walk. Key accuracy insight: cubiomes' `(int64_t)(10000.0F
    * value)` cast pattern absorbs all f32 climate drift below the
    1/10000 boundary, so as long as `np[6]` lands within ±1 of cubiomes',
    the b-tree returns the identical biome. C shim `mcsf_compute_depth`
    runs cubiomes' depth-spline computation given (c, e, w, y) — forward-
    declares `getSpline` since cubiomes has no public prototype for it.
    Parity for 32×32 grid (1024 pixels, MC 1.21, seed 12345):
    **1024 / 1024 = 100.00 %**.
  - **Phase 6c-6: GPU tile rendering matches cubiomes byte-for-byte.**
    `GpuBiomeRenderer::render_tile_rgba(scale, x, z, sx, sz)` produces RGBA
    bytes byte-identical to `BiomeBackend::render_tile_rgba` (cubiomes
    CPU) for `scale = 4` (canonical biome-map resolution). `biome_colormap()`
    in `biomes.rs` promoted from `fn` to `pub fn` so the GPU path reuses the
    same cubiomes-derived colormap. RGBA byte parity for a 32×32 tile:
    **1024 / 1024 = 100.00 %**. Other scales (1 voronoi, ≥16 inner-scaling)
    fall back to the cubiomes CPU path for now — that's purely coord-shaping
    work that doesn't touch the GPU primitives.
  - **Phase 6c-7: GPU biome-condition primitive (search-side scaffold).**
    `GpuBiomeRenderer::biomes_in_set(coords, allowed)` returns one `bool`
    per coord: "is the biome at this coord in the allowed set". Sorted-Vec
    membership (binary search) — biome filter sets are typically <10 IDs,
    so a sorted scan beats hash overhead. This is the per-seed building
    block a future GPU-accelerated biome condition in the seed searcher
    will call. Documented deferral: extending the Phase-6b multi-predicate
    kernel to evaluate biome conditions per candidate seed needs a WGSL
    port of cubiomes' Xoroshiro128++ (`xSetSeed`/`xNextLong`) + full
    `setBiomeSeed` so per-thread climate init runs on the GPU — that is
    a sub-phase in its own right and intentionally not bundled with
    "Phase 6c: one MC version, matching cubiomes" (which 6c-1..6c-6
    fully delivered with bit-identical tile output).
- 9 new GPU↔cubiomes parity tests covering each Phase 6c sub-phase, plus
  a CPU walker test (Phase 6c-4) that catches data-extraction bugs
  independent of WGSL.

- **GPU acceleration for cluster + multi-leaf groups (Phase 6b).** The WGSL
  kernel is generalised from "one NearbyStructure" to "up to 8 predicates +
  combinator". Three combinators land: `any_of`, `all_of`, and `cluster`
  (the quad-hut shape — counts hits across the predicate list against a
  `min_count`). The Tauri side has a new `try_extract_gpu_spec` that detects
  cluster trees, `all_of`/`any_of` groups of NearbyStructure leaves, and
  the v1 single-Nearby shape, all routing to the same kernel. Three new
  GPU↔CPU parity tests (`gpu_cluster_matches_cpu`, `gpu_any_of_matches_cpu`,
  `gpu_all_of_matches_cpu`) dispatch the new kernels against the CPU
  evaluator and assert byte-identical match lists on 2,000-seed runs each.
- **Per-request cancellation in Tauri commands.** New `tile_counter` /
  `pins_counter` / `analyze_counter` atomics on `AppState`. Each command
  claims a sequence number on entry; if a newer request bumps the counter
  before the slow steps complete, the command bails out with a `superseded`
  sentinel. React-side `isSupersededError` swallows those rejections
  silently. Net effect: rapid pan/click no longer queues up minutes of
  stale work behind the latest view.
- **Cursor-anchored wheel zoom.** Each wheel tick keeps the world point
  under the cursor pinned, computed from `viewCenter` + pane-relative
  cursor offset at old/new `blocksPerPx`. Made workable by the per-request
  cancellation above (every tick shifts viewCenter and would otherwise
  fire a tile-fetch storm).
- **Map legend** in the bottom-left of the map: coloured dots for village,
  pillager outpost, ocean monument, woodland mansion, stronghold, and the
  spawn marker, naming what each colour means.
- **Phase 6c roadmap doc** (`docs/GPU.md`) covering full GPU biome
  generation: the three architectural approaches, why it's a separate
  multi-day effort, recommendation (full WGSL port, version-by-version
  starting with 1.21).

### Changed

- **Background prefetch disabled** in `main.tsx`. The 4-neighbor prefetch
  per view change was net-negative once the BiomePool amortised the
  cubiomes setup cost; the LRU tile cache still serves repeat views.
  See the Phase 6b commit for re-enable instructions.

- **GPU compute prefilter via wgpu (Phase 6a).** New `crates/mcseedfinder-core/
  src/gpu.{wgsl,rs}` implements Java's 48-bit LCG and structure placement in
  WGSL, dispatched in parallel across a seed range. The WGSL kernel uses u32
  limbs to emulate the 64-bit arithmetic WGSL lacks natively (schoolbook
  multiply with explicit carries); the host passes every multi-limb constant
  (RNG multiplier, region mixers, salts, distance threshold) via the uniform
  buffer so hex transcription happens in one place. Three GPU↔CPU parity
  tests assert byte-identical match lists across linear-spread, triangular-
  spread, and off-origin centre cases — covering 3,500 seeds total.
- **Tauri GPU fast-path.** `run_search` detects when the conditions tree
  collapses to a single `NearbyStructure` predicate (the default React tree
  shape, and any single-child group wrapping one) and dispatches via wgpu
  in 65,536-seed chunks. Cancellation is checked between chunks; per-match
  events stream as today; the `matched_features` array records
  `"gpu_prefilter"` so the UI / consumers know which path ran. Init is
  lazy + cached in `AppState.gpu: OnceLock<Option<GpuSearcher>>`; failure
  falls back transparently to the CPU evaluator.
- **Tile LRU cache + neighbor prefetch.** New 32-entry tile cache in
  `main.tsx`, keyed by `(seed, version, x, z, sx, sz, scale)`. Hits serve
  instantly so panning back to a recently-viewed area never blinks to
  "Rendering biome tile…". On every view change, the four adjacent over-
  rendered tiles are fetched in the background and inserted into the cache
  — pan-on-release in any direction now usually hits a warm tile and
  appears immediately. (CSS transforms continue to be GPU-composited by
  the WebView, so the *visible* pan/zoom was already GPU-accelerated; the
  missing piece was tile *availability*, which this fixes.)
- `gpu` cargo feature on `mcseedfinder-core` (default-on) wires `wgpu`,
  `bytemuck`, `pollster`. The wgpu backends gated to `metal` / `dx12` /
  `wgsl` keep the dependency tree lean; Vulkan support comes for free with
  the wgpu default backends on Linux.

- **Bedrock Edition foundation (Phase 5).** First-class `BedrockProvider`
  replaces the Java-era stub: validates seeds against Bedrock's signed i32
  range, enumerates which criterion types the Bedrock backend can't yet
  evaluate (every worldgen-dependent criterion today), and never silently
  falls back to Java math.
- **Bedrock text-seed hashing.** `bedrock::seed_from_string` (Rust) /
  `_native.bedrock_seed_from_string` (PyO3) / `mcseedfinder.bedrock.seed_from_string`
  (Python, with a pure-Python fallback identical to the native version)
  implement Java's `String.hashCode()` over UTF-16 code units, which is
  exactly what Bedrock (and Java) use when a player types a text seed.
  CLI: `python -m mcseedfinder --seed-string "your text"` prints the i32.
- Roadmap doc at `docs/BEDROCK.md` enumerating what a full Bedrock
  backend needs (biome generation, structure placement, stronghold rings)
  and the three credible architectural paths (symbol-prefixed dual
  cubiomes, pure-Rust port, subprocess sidecar) with a recommendation.
- 10 new Python tests (`tests/test_bedrock.py`) including a parity test
  that exercises the pure-Python fallback against the native binding on
  9 inputs spanning empty strings, ASCII, non-ASCII BMP, supplementary-
  plane surrogate pairs, and i32-overflow-inducing long strings.

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

### Completed since the 4a-2 / 4b deferral

- Biome criteria in desktop *search* specs are now evaluated natively via
  `conditions::evaluate_with_biomes` with a single per-search cubiomes
  `BiomeBackend` (`desktop/src-tauri/src/main.rs::run_search`, gated on
  `conditions::has_biome_leaves`).
- `render_tile` ships as both a base64-PNG path (`render_tile`) and the
  faster raw-RGBA path (`render_tile_rgba_cmd`), saving ~30–45 ms per tile.
- `import_level_dat` is wired up via `fastnbt` + `flate2`, reading the
  modern `Data.WorldGenSettings.seed` with a legacy `Data.RandomSeed` fallback.

### Still deferred

- `pause`/`resume` Tauri commands are no-ops; cancellation works. Tracked
  for the next phase alongside the broader job-lifecycle work.

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
