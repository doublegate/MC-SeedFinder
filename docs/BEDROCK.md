# Bedrock Edition status & roadmap

Phase 5 (committed) lays the **foundation** for Bedrock support. A full,
accuracy-correct Bedrock worldgen backend is a separate effort and is
deliberately not bundled until it can meet the same accuracy bar Java does.

## What's in Phase 5 today

- **Text-seed → numeric-seed hashing.** Bedrock (and Java) hash text seeds
  through Java's `String.hashCode()` — `h = 31*h + c` over each UTF-16 code
  unit, wrapping into a signed i32. We expose this in three places, all
  bit-identical to each other and to any conformant Java implementation:
  - Rust: [`bedrock::seed_from_string`](../crates/mcseedfinder-core/src/bedrock.rs)
  - PyO3: `mcseedfinder._native.bedrock_seed_from_string`
  - Python (with a pure-Python fallback): `mcseedfinder.bedrock.seed_from_string`
  - CLI: `python -m mcseedfinder --seed-string "your text"`
- **First-class `BedrockProvider`.** Replaces the Java-era stub with a real
  edition-aware validator: rejects seeds outside i32 range, enumerates
  which criterion types are unsupported, and refuses to silently fall back
  to Java math (which would invalidate every result). See
  [`engine.py`](../src/mcseedfinder/engine.py).
- **Seed-range hygiene.** `mcseedfinder.bedrock.is_valid_bedrock_seed`
  helps callers stay within Bedrock's signed i32 range.

## What's deliberately NOT in Phase 5

- Bedrock biome generation. Bedrock uses a different layered-noise pipeline
  than Java; the canonical cubiomes library only covers Java. The community
  forks ([reedacartwright/cubiomes@bedrock](https://github.com/reedacartwright/cubiomes/tree/bedrock),
  [FragrantResult186](https://github.com/FragrantResult186/cubiomes-viewer-bedrock),
  [bedrock-dev](https://github.com/bedrock-dev/cubiomes-viewer)) cover MC
  1.16/1.17-era Bedrock biomes but aren't drop-in compatible alongside the
  Java cubiomes we vendor — see the symbol-collision discussion below.
- Bedrock structure placement. Different salts, different spacing/separation,
  different PRNG (Mersenne Twister in modern Bedrock; the legacy LCG before).
- Stronghold rings. Bedrock has different stronghold ring counts and offsets.

The `BedrockProvider` rejects criteria that depend on any of the above with
a clear, enumerated error so users see exactly which feature is missing.

## Architectural options for Phase 5b (full Bedrock backend)

Three credible paths, in increasing order of effort + robustness:

### 1. Symbol-prefixed dual cubiomes (medium effort, medium robustness)

Vendor a second cubiomes (e.g. Reed Cartwright's `bedrock` branch) alongside
the existing Cubitect one. The hard part is **symbol collisions** — both
forks export `setupGenerator`, `applySeed`, `getBiomeAt`, etc.

Options:
- Compile each fork with `-fvisibility=hidden`, mark only the
  `mcsf_*` / `mcsf_be_*` shim wrappers as visible. With distinct static
  archives plus careful link-script work, only one set of symbols
  participates in final linking. Works on Linux/macOS; less clean on
  Windows.
- Run `objcopy --prefix-symbols=be_` on the Bedrock archive before linking.
  Simple in principle, but adds a Windows portability hurdle (no `objcopy`).
- Use a build-time preprocessor rename header that `#define`s every public
  cubiomes function to a `be_`-prefixed name in the Bedrock build only.
  Brittle (the list of public functions is large and changes with cubiomes
  versions) but cross-platform.

### 2. Pure-Rust port of Bedrock worldgen (high effort, highest robustness)

Mirror what the project already did for Java: a hand-ported, golden-tested
`bedrock_random`, `bedrock_structures`, `bedrock_biomes` in pure Rust. Same
discipline — golden vectors captured from a reference implementation, no
silent drift. Largest up-front cost but no C dependency, easy to ship to
WASM, and the same approach that already won for Java structures.

### 3. Subprocess sidecar (low code effort, weak ergonomics)

Build the Bedrock cubiomes fork as a *separate* binary, spawn it as a
subprocess, communicate via JSON over stdio. No symbol collisions. Pays a
process-spawn + IPC cost per query, and complicates packaging
(.deb/.AppImage/.msi all need to bundle the helper).

### Recommendation

For Phase 5b: **option 1 (symbol-prefixed dual cubiomes)** for Bedrock
biomes — fastest to a credible accuracy backend — combined with a
**pure-Rust port of the smaller surface** (Bedrock RNG + structure
salts/spacing), which is well-documented and the same shape as the existing
Java pure-Rust core. Strongholds can ride on whichever path lands first.

Whatever path is chosen, the contract on the Python/JS side stays the same:
`BedrockProvider.evaluate_seed` becomes able to evaluate the criteria types
the new backend supports, and the validator's "unsupported" set shrinks
accordingly. No engine-layer changes required.

## Verification standard for Phase 5b

The same accuracy bar Java meets:
- Golden test vectors per cubiomes version pin, captured from a reference
  Bedrock implementation (the chosen fork or Mojang's own behaviour),
  asserted in `cargo test` and `python -m unittest`.
- `SeedReport.exactness` keys are `"exact"` only when the result came from
  a vector-verified Bedrock path. Anything approximate must say so.
- The provider's "unsupported" set is *enumerated* in errors, never silent.

## See also

- [`src/mcseedfinder/bedrock.py`](../src/mcseedfinder/bedrock.py)
- [`crates/mcseedfinder-core/src/bedrock.rs`](../crates/mcseedfinder-core/src/bedrock.rs)
- [`src/mcseedfinder/engine.py`](../src/mcseedfinder/engine.py) — `BedrockProvider`
- [`tests/test_bedrock.py`](../tests/test_bedrock.py)
