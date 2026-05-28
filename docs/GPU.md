# GPU acceleration — status & roadmap

This project uses [wgpu](https://wgpu.rs) for compute-on-GPU. Phase 6 + 6b
established the foundation; Phase 6c is the big follow-on for full biome
acceleration. Java-RNG and structure-placement parity with the CPU code path
is guarded by GPU↔CPU parity tests (`crates/mcseedfinder-core/src/gpu.rs` —
the GPU output must agree byte-for-byte with the existing pure-Rust
implementation for every dispatched seed).

## What's on GPU today (Phase 6 + 6b)

| Feature | Status | Notes |
|---|---|---|
| Java 48-bit LCG (`java.util.Random`) | ✅ | WGSL with u32-limb arithmetic |
| Random-spread structure placement | ✅ | Linear + triangular spreads |
| Single `NearbyStructure` predicate | ✅ | `any_of` w/ 1 predicate |
| `Cluster` predicate (quad-hut, etc.) | ✅ Phase 6b | Counts hits across structure list |
| `all_of` over `NearbyStructure` leaves | ✅ Phase 6b | Up to 8 leaves |
| `any_of` over `NearbyStructure` leaves | ✅ Phase 6b | Up to 8 leaves |
| Stronghold ring math | ❌ | Different RNG path; CPU fallback |
| Logic gates nested inside groups | ❌ | Up to one level of single-child wrap |
| Biome lookups | ❌ | See Phase 6c below |
| Biome map rendering (tiles) | ❌ | See Phase 6c below |

### Auto-routing

`desktop/src-tauri/src/main.rs::run_search` routes to GPU only when:

- spec is structure-only (no biome leaves), **and**
- the compiled tree fits the multi-predicate kernel
  (`try_extract_gpu_spec`), **and**
- `spec.count >= GPU_MIN_COUNT` (currently 500,000).

Below the threshold the CPU evaluator wins — wgpu's per-dispatch overhead
(uniform write + queue submit + map_async + poll) and one-time init
(~200–500 ms first call) dominate small searches. The CPU's tight Rust
loop has no IPC / GPU-sync cost.

### Performance properties

- **Buffer reuse**: `GpuSearcher` owns persistent params/output/staging
  buffers and the bind group, created once. Per dispatch we only
  `queue.write_buffer` the 88-byte+predicate-array uniform.
- **Chunk size**: 65,536 seeds per dispatch, ~tens of ms on a modern GPU.
  Cancellation flag is checked between chunks for snappy cancel UX.
- **Max predicates**: 8. Bigger trees fall back to the CPU evaluator —
  if you need more, increase `MAX_PREDICATES` in both `gpu.rs` and
  `gpu.wgsl` (it's a single-knob change).

## Phase 6c roadmap — GPU biome generation

This is the multi-day effort that the user's "make tiles faster" / "GPU
biome conditions" requests both reduce to.

### Why it's hard

cubiomes' biome generation in modern Minecraft (1.18+) is a stack of:

1. **Multiple octaves of Perlin/Simplex noise** seeded from the world seed.
2. **Climate fields**: temperature, humidity, continentalness, erosion,
   weirdness/PV. Each is a multi-octave noise.
3. **6-D biome lookup**: the climate vector is searched against a
   per-version b-tree of biome regions
   (`tables/btree18.h`, `btree192.h`, `btree19.h`, `btree20.h`,
   `btree21wd.h` in the vendored cubiomes).
4. **Surface refinements**: ocean variants, beach/river detection.
5. **Version-specific quirks** for 1.16, 1.17, 1.18, 1.19, 1.20, 1.21+.

Porting this to WGSL means:
- Writing all the noise functions (Perlin, simplex, octave summation) in
  WGSL with deterministic results matching cubiomes' C float math.
- Encoding the per-version biome b-trees as GPU buffers + writing a tree
  walker in WGSL.
- Handling each MC version's variations.
- Building golden-vector tests so the GPU biome IDs match cubiomes for
  every supported version.

The cubiomes maintainers themselves have GPU biome generation as a
long-standing roadmap item (see Cubitect/cubiomes#48 and related issues);
nobody has shipped a complete port.

### Three credible approaches, in order of effort

**A. Port biome generation entirely to WGSL.** Highest performance ceiling,
fully portable, no native deps in the renderer. Largest effort (~weeks).
Accuracy risk requires comprehensive golden-vector tests per MC version.

**B. Read cubiomes' biome output, upload to GPU, render on GPU.**
cubiomes still does the compute (CPU); the GPU only does the
biome-ID → RGB → display step. Much smaller effort, but tile rendering
isn't actually compute-bound on the GPU step — it's bound on cubiomes
itself. Wouldn't visibly help the user's "slow tile" complaint.

**C. Approximate biome map on GPU + cubiomes-exact criteria on CPU.**
Render a fast, visually-plausible biome map on the GPU (lossy noise,
no exact lookups) for the preview. Search criteria continue to evaluate
on exact cubiomes (CPU). Honest if labeled: the user sees a fast
preview map, but the search results are still bit-exact. Smaller effort
than (A), but requires a clear UX label and risks user confusion.

### Recommendation

**(A) is the right long-term answer.** It's the only path to true
"GPU-accelerated biome lookups for search" *and* "GPU-rendered tiles".
The first deliverable should be Java 1.21 only — match the project's
"Java 1.18+, 1.21-first" stance — with golden vectors imported from
the existing cubiomes oracle. Earlier versions can follow.

A scoped first cut:
1. WGSL ports of cubiomes' `noise.c` (Perlin + Octave + simplex helpers).
2. WGSL port of the 1.21 climate noise stack (continentalness, erosion,
   weirdness, temperature, humidity).
3. WGSL b-tree walker over `btree21wd.h` (the lookup table for 1.21).
4. A new `GpuBiomeBackend` implementing the same `get_biome` contract as
   the cubiomes-backed `BiomeBackend`.
5. Tile rendering: a separate compute kernel that fills a tile buffer of
   biome IDs (1 thread per pixel) and a fragment-shader pass that maps
   biome IDs → RGB via the existing colormap. Output a GPU texture; ship
   to the WebView via a Tauri command that returns a base64 PNG (or
   bypass PNG entirely with a Canvas2D `putImageData` from raw bytes,
   which would save another 10-20 ms per tile).
6. Search-criteria biome eval: extend the existing multi-predicate kernel
   with biome leaf variants (`spawn_biome`, `nearby_biomes`, `biome_area`)
   that call into the new GPU biome path.

Until (A) ships, the desktop's tile rendering stays on cubiomes-CPU
with the BiomePool amortising setup cost. The biome-criteria search path
stays on CPU + cubiomes too. Both are honest about that — `SeedReport.
exactness` reads "exact" for both today, and would continue to.

## Files

| Path | Role |
|---|---|
| `crates/mcseedfinder-core/src/gpu.rs` | Searcher + uniform layout + parity tests |
| `crates/mcseedfinder-core/src/gpu.wgsl` | Compute kernel (Java RNG + structure + combinators) |
| `crates/mcseedfinder-core/Cargo.toml` | `gpu` feature, `wgpu`/`bytemuck`/`pollster` deps |
| `desktop/src-tauri/src/main.rs` | `try_extract_gpu_spec`, GPU search wiring |
| `docs/GPU.md` | This document |
