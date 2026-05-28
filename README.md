# mc-seed-finder

`mc-seed-finder` is a local-first Minecraft seed search toolkit with an
accuracy-aware Python CLI, a Rust core for high-throughput structure filtering,
and an early Tauri desktop shell.

The project is designed around one rule: never present approximate worldgen as
exact. Structure placement and Java RNG math are tested against golden vectors.
Biome filtering is exact via the bundled cubiomes backend, and is clearly
labeled as approximate only on builds compiled without it.

## What It Does

- Searches Java Edition 1.18+ seeds for nearby structures, strongholds, spawn
  biome candidates, and nearby biome candidates.
- Prints matching seeds to stdout so results can be piped into scripts.
- Keeps diagnostics, progress, and reports on stderr.
- Uses multiprocessing for the Python search path.
- Uses a PyO3 Rust extension for native structure-only batch filtering when the
  extension is built.
- Exposes product-facing search contracts for app integrations:
  `SearchSpec`, `SearchEvent`, `SeedReport`, and provider interfaces.
- Includes a SQLite-backed async command host for background jobs, cancellation,
  pause/resume, event persistence, and result export.
- Includes a Tauri 2 + React desktop MVP scaffold.

## Accuracy Status

| Capability | Implementation | Status |
| --- | --- | --- |
| `java.util.Random` | Python and Rust 48-bit LCG ports | Exact, tested against OpenJDK vectors |
| Random-spread structures | Cubiomes-derived salts and region math | Exact candidate placement for Java 1.18+ |
| Strongholds | Concentric-ring algorithm | Exact candidate placement, Python and Rust |
| Biome lookup (CPU) | Vendored cubiomes via Rust FFI | Exact for Java 1.18+ when the native extension is built |
| Biome lookup (GPU, MC 1.21) | WGSL port of cubiomes (Phase 6c) | Bit-exact tile rendering vs cubiomes — 1024/1024 RGBA bytes match |
| Biome lookup (approximate fallback) | Local climate-noise approximation | Approximate; used only when built without cubiomes |
| Structure prefilter (GPU) | WGSL port of structure RNG + cluster/all_of/any_of | Bit-exact vs CPU evaluator (Phase 6, 6b) |
| Bedrock Edition (text-seed → i32) | Java `String.hashCode()` over UTF-16 | Exact (Phase 5) |
| Bedrock Edition (worldgen) | Provider with explicit rejection | Not implemented; worldgen criteria fail with edition-aware errors |
| Desktop map tiles | Tauri + cubiomes RGB colormap | Pan/zoom 1:1–1:256, structure pins, raw-RGBA transport |

Important distinction: structure placement means the chunk where Minecraft
attempts to place a structure. In-game generation can still reject a structure
because of biome or terrain validity. The candidate chunk math itself is exact.

## Repository Layout

```text
.
├── src/mcseedfinder/              Python package and CLI
├── crates/mcseedfinder-core/      Rust core and PyO3 extension
├── desktop/                       Tauri 2 + React desktop MVP
├── examples/                      Example JSON criteria
├── tests/                         Python unit tests
├── Cargo.toml                     Rust workspace
├── pyproject.toml                 Python build metadata via maturin
└── CHANGELOG.md
```

## Requirements

- Python 3.10+
- Rust stable toolchain
- A C compiler (the native extension compiles the vendored cubiomes C sources)
- Node.js and npm for the desktop frontend
- Linux desktop builds of Tauri may require WebKitGTK and related system
  packages, depending on distribution

The runtime Python package is stdlib-only. Exact biome generation is provided by
the native extension, which statically compiles Cubitect's `cubiomes` C library
from a git submodule — there is no `bindgen`/`libclang` build dependency. After
cloning, initialize the submodule:

```bash
git submodule update --init --recursive
```

The editable install uses `maturin` to build the native extension. To build the
pure-Rust structure/RNG core without a C toolchain, disable the default `biomes`
feature (`cargo build --no-default-features`); biome filtering then falls back to
the approximate generator.

## Install

For normal local development:

```bash
pip install -e ".[dev]"
```

Biomes are exact out of the box (the native extension bundles cubiomes). The
optional `accurate` extra remains for experimenting with an alternative
`cubiomes-py` backend:

```bash
pip install -e ".[dev,accurate]"
```

For source-tree commands without installing:

```bash
PYTHONPATH=src python -m mcseedfinder --help
```

## CLI Quick Start

Inspect one seed:

```bash
python -m mcseedfinder --show-seed 1
```

Find seeds with a village near origin:

```bash
python -m mcseedfinder \
  --nearby-structure village:1000 \
  --count 1000000 \
  --workers 8 \
  --max-matches 5
```

Use an example criteria file:

```bash
python -m mcseedfinder \
  --config examples/village_outpost.json \
  --count 5000000 \
  --workers 8 \
  --max-matches 3
```

Export matches:

```bash
python -m mcseedfinder \
  --nearby-structure village:100 \
  --count 100000 \
  --max-matches 20 \
  --quiet \
  --export matches.json \
  --export-format json
```

## CLI Reference

Common criteria flags:

- `--nearby-structure NAME:DIST`: require a structure within `DIST` blocks of
  origin. Repeatable.
- `--spawn-biome NAME`: require an approximate spawn-area biome or biome group.
- `--spawn-radius N`: radius for spawn biome sampling. Default: `64`.
- `--nearby-biomes NAME`: require a nearby biome or biome group.
- `--nearby-biomes-radius N`: radius for nearby biome sampling. Default: `2000`.
- `--nearby-biomes-all`: require all listed nearby biomes instead of any.
- `--config FILE`: load raw criteria JSON.
- `--spec FILE`: load versioned `SearchSpec` JSON and use its top-level
  `criteria` object.

Search and runtime flags:

- `--edition {java,bedrock}`: Java works today. Bedrock fails explicitly.
- `--version VERSION`: version label attached to reports and exports.
- `--mode {sequential,random}`: enumerate seeds in order or sample randomly.
- `--start S`: first sequential seed.
- `--count N`: number of seeds to test.
- `--workers W`: multiprocessing worker count.
- `--chunk-size N`: seeds per worker work unit.
- `--max-matches K`: stop after `K` matches.
- `--random-seed S`: deterministic random-mode sampling.
- `--report-each`: print a detailed report for every match.
- `--quiet`: suppress progress/report output on stderr.
- `--export FILE`: write matches to a file.
- `--export-format {json,csv,plain}`: export format.

Information flags:

- `--list-structures`
- `--list-biomes`
- `--show-seed SEED`

## Criteria JSON

Criteria files are JSON objects. Example:

```json
{
  "spawn_biome": "plains",
  "spawn_radius": 64,
  "nearby_structures": [
    { "structure": "village", "max_distance": 1200 },
    { "structure": "pillager_outpost", "max_distance": 1500 },
    {
      "structure": "ocean_monument",
      "max_distance": 2500,
      "centre_x": 0,
      "centre_z": 0
    }
  ],
  "nearby_biomes": {
    "biomes": ["plains", "river", "forest"],
    "radius": 1500,
    "all": true,
    "samples": 16
  }
}
```

Example configs:

- `examples/village_outpost.json`
- `examples/structure_density.json`

## Python API

Run the event stream directly:

```python
from mcseedfinder.engine import SearchSpec, run_staged_search

spec = SearchSpec(
    criteria={
        "nearby_structures": [
            {"structure": "village", "max_distance": 1000}
        ]
    },
    start_seed=1,
    count=1000,
    max_matches=1,
)

for event in run_staged_search(spec):
    print(event.type, dict(event.payload))
```

Run a persisted async job:

```python
from mcseedfinder.app_contract import AsyncCommandHost, JobStore
from mcseedfinder.engine import SearchSpec

store = JobStore("mcseedfinder.sqlite")
host = AsyncCommandHost(store)

job_id = host.start_search(SearchSpec(
    criteria={
        "nearby_structures": [
            {"structure": "village", "max_distance": 100}
        ]
    },
    count=100000,
    max_matches=10,
))

host.wait(job_id)
print(store.list_results(job_id))
host.shutdown()
```

## Rust Core

The Rust core lives in `crates/mcseedfinder-core/` and provides:

- `JavaRandom`
- random-spread structure placement
- stronghold ring generation
- native batch filtering for structure-only searches
- exact biome lookup via vendored cubiomes (C, compiled with `cc`; no `bindgen`/
  `libclang` dependency)
- GPU compute prefilter for structures via wgpu (Phase 6/6b)
- GPU biome generation for MC 1.21 via wgpu (Phase 6c, bit-exact tile rendering)
- PyO3 bindings exposed as `mcseedfinder._native`

Run the Rust tests:

```bash
cargo test --features pyo3
```

Run the release benchmark:

```bash
RUSTC_WRAPPER= cargo run --release --example bench_structure_search -- 100000
```

`RUSTC_WRAPPER=` is useful on systems where `sccache` is configured but cannot
run in the current sandbox or CI environment.

## Desktop MVP

The desktop shell lives in `desktop/`. There's also a thin proxy `package.json`
at the repo root so the common commands work from anywhere in the tree:

```bash
# From the repo root:
npm run install:desktop                    # one-time, installs desktop deps
npm run dev                                # = `npm --prefix desktop run tauri dev`
npm run build                              # = `npm --prefix desktop run build`
npm run tauri:build                        # = `npm --prefix desktop run tauri build`
```

Or work inside the `desktop/` folder directly:

```bash
cd desktop
npm install
npm run build
RUSTC_WRAPPER= cargo check --manifest-path src-tauri/Cargo.toml
```

Run the browser-based Vite preview:

```bash
npm run dev -- --port 5173                 # (only inside desktop/)
```

Run through Tauri during local desktop development:

```bash
npm run tauri dev                          # from desktop/, OR `npm run dev` from root
```

Current desktop status:

- React UI builds successfully.
- Tauri command crate checks successfully.
- Command names match the product contract.
- Current Tauri commands are scaffold stubs.
- The Python `AsyncCommandHost` is the more complete local execution model.

## Testing

Run Python tests:

```bash
PYTHONPATH=src python -m unittest
```

Run Rust tests:

```bash
cargo test --features pyo3
```

Run desktop checks:

```bash
cd desktop
npm run build
RUSTC_WRAPPER= cargo check --manifest-path src-tauri/Cargo.toml
```

Current tests cover:

- Java RNG vectors.
- Structure placement vectors for seed `1`.
- Stronghold counts.
- Criteria compilation and stage reporting.
- Rust/Python parity when the native extension is present.
- SQLite-backed app command persistence.
- Async background completion and cancellation.

## Performance

Use the Python benchmark to compare fallback Python evaluation with the native
Rust extension:

```bash
PYTHONPATH=src python -m mcseedfinder.benchmark --count 10000
```

On the current development machine, a representative structure-only benchmark
showed:

- Python path: about 80k seeds/s.
- Rust extension path: about 14M seeds/s.
- Parity: true.

These numbers vary by machine, query, build mode, and whether the native
extension was built in release mode.

## GPU Acceleration

Two GPU pipelines via [wgpu](https://wgpu.rs/) (Vulkan/Metal/DX12 backends,
auto-selected) are gated by the default-on `gpu` cargo feature on
`mcseedfinder-core`. Both are validated by GPU↔CPU parity tests on every
build and fall back to the CPU path transparently when no compatible adapter
is present.

| Pipeline | Phase | Status | Tested against |
| --- | --- | --- | --- |
| Structure-RNG seed prefilter | 6a, 6b | **Bit-exact** vs CPU evaluator (3,500-seed parity runs) | The pure-Rust structure RNG |
| Multi-predicate kernel: `cluster`, `any_of`, `all_of` of `NearbyStructure` leaves | 6b | **Bit-exact** vs CPU (2,000-seed parity runs each) | The pure-Rust conditions evaluator |
| `samplePerlin` (Phase 6c-1) | 6c | f32 drift ≤ 2.9 × 10⁻⁵ | cubiomes `samplePerlin` |
| `sampleOctave` + `sampleDoublePerlin` (Phase 6c-2) | 6c | f32 drift ≤ 3.6 × 10⁻⁵ | cubiomes `sampleDoublePerlin` (via shim) |
| Climate noise stack — 6 fields for MC 1.21 (Phase 6c-3) | 6c | f32 drift ≤ 2.2 × 10⁻⁵ (worst: continentalness, 18 octaves) | cubiomes `setBiomeSeed` + per-field sample |
| Biome b-tree walker — btree21wd (Phase 6c-4) | 6c | **Bit-exact** | cubiomes `climateToBiome` |
| Full `sampleBiomeNoise` integration (Phase 6c-5) | 6c | **100 % (1024 / 1024)** biome IDs match | cubiomes `sampleBiomeNoise` |
| `render_tile_rgba` for MC 1.21 (Phase 6c-6) | 6c | **100 % (1024 / 1024)** RGBA bytes match | `BiomeBackend::render_tile_rgba` (cubiomes CPU) |

Phase 6c is bit-exact-modulo-f32-tolerance for **one** Minecraft version
(1.21). cubiomes uses f64 throughout; WebGPU compute is f32-only, so even
"matching" means matching within float tolerance at climate boundaries —
except for the b-tree walker, which is integer-only and exactly bit-exact.
The i64 truncation of `(int64_t)(10000.0F * climate)` that cubiomes does
internally absorbs all f32 drift below the 1/10000 boundary, which is why
tile RGBA can match cubiomes byte-for-byte despite the precision gap.

Multi-seed search integration of the GPU biome pipeline is documented as
future work in `crates/mcseedfinder-core/src/gpu_biome.rs` — it requires
porting cubiomes' Xoroshiro128++ (`xSetSeed`/`xNextLong`) and
`setBiomeSeed` to WGSL so per-thread climate init runs on the GPU. The
per-seed `biomes_in_set` primitive that integration would call is
already shipped (Phase 6c-7).

## Limitations

- Java Edition 1.18+ is the only edition with a full worldgen backend
  (exact biomes via cubiomes; exact structures/strongholds in pure Rust).
- GPU biome generation is implemented for **Minecraft 1.21 only** (Phase 6c).
  Other 1.18+ versions continue to use the cubiomes CPU path. Adding more
  versions is mechanical (different `btreeN.h` + matching `init_climate_seed`
  constants), tracked in `docs/GPU.md`.
- Bedrock Edition: foundation only. Text-seed → i32 hashing works
  (`--seed-string`); worldgen-dependent criteria (biomes, structures,
  strongholds) are rejected with a clear edition-aware error. See
  `docs/BEDROCK.md` for the roadmap.
- Biome filtering is exact via cubiomes when the native extension is built; the
  approximate climate-noise generator is used only as a no-cubiomes fallback.
- Structure candidate placement does not prove in-world structure validity.
- The seed-search kernel evaluates biome conditions on the CPU; GPU per-seed
  biome conditions are blocked on a future WGSL port of `setBiomeSeed`.

## References

- Cubitect's `cubiomes`: structure salts, region constants, and worldgen
  reference behavior.
- OpenJDK `java.util.Random`: canonical 48-bit LCG behavior.
- Minecraft Wiki: structure sets and stronghold placement rules.
- Tauri 2: desktop shell.
- React and TypeScript: desktop frontend.

## License

MIT. See `pyproject.toml` for package metadata.
