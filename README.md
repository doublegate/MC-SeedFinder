# mc-seed-finder

`mc-seed-finder` is a local-first Minecraft seed search toolkit with an
accuracy-aware Python CLI, a Rust core for high-throughput structure filtering,
and an early Tauri desktop shell.

The project is designed around one rule: never present approximate worldgen as
exact. Structure placement and Java RNG math are tested against golden vectors.
Biome filtering is clearly labeled as approximate unless an exact backend is
installed.

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
| Biome lookup | Local climate-noise fallback | Approximate candidate filtering |
| Exact biome lookup | Optional `cubiomes-py` integration point | Exact when installed and wired |
| Bedrock Edition | Provider placeholder | Not implemented; fails explicitly |
| Desktop map tiles | UI/backend placeholders | Not implemented yet |

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
- Node.js and npm for the desktop frontend
- Linux desktop builds of Tauri may require WebKitGTK and related system
  packages, depending on distribution

The runtime Python package is stdlib-only unless optional extras are installed.
The editable install uses `maturin` to build the native extension.

## Install

For normal local development:

```bash
pip install -e ".[dev]"
```

For accurate biome backend experiments:

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

The desktop shell lives in `desktop/`.

```bash
cd desktop
npm install
npm run build
RUSTC_WRAPPER= cargo check --manifest-path src-tauri/Cargo.toml
```

Run the browser-based Vite preview:

```bash
npm run dev -- --port 5173
```

Run through Tauri during local desktop development:

```bash
npm run tauri dev
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

## Limitations

- Java Edition 1.18+ is the only implemented edition/version target.
- Bedrock is represented in the API but not implemented.
- Biome filtering is approximate unless an exact backend is installed and used.
- Structure candidate placement does not prove in-world structure validity.
- The desktop UI is an MVP scaffold, not a finished map viewer.
- GPU search planning is not implemented yet.

## References

- Cubitect's `cubiomes`: structure salts, region constants, and worldgen
  reference behavior.
- OpenJDK `java.util.Random`: canonical 48-bit LCG behavior.
- Minecraft Wiki: structure sets and stronghold placement rules.
- Tauri 2: desktop shell.
- React and TypeScript: desktop frontend.

## License

MIT. See `pyproject.toml` for package metadata.
