# Supported Minecraft Versions

mc-seed-finder evaluates worldgen via the bundled
[cubiomes](https://github.com/Cubitect/cubiomes) submodule. Which
Minecraft versions are supported depends entirely on what that submodule
recognises in `vendor/cubiomes/biomes.h`. This page is the source of
truth — keep it in sync on every `git submodule update`.

The CLI runs a pre-flight check against this list: `mcseedfinder --version <X>`
with an unrecognised `<X>` exits with a clear error before any search starts.
The check uses `rust_backend.is_supported_version()` which calls into
cubiomes' `mcsf_str2mc` (the same parser cubiomes-viewer uses).

## Current bundle

| Field | Value |
| --- | --- |
| Cubiomes commit | `e61f905` (2024-11-10, "Renamed MC_1_21_2 to MC_1_21_3") |
| Upstream `master` HEAD | `e61f905` — **same commit** (verified 2026-05-28) |
| Upstream source | https://github.com/Cubitect/cubiomes |

> **Note (2026-05-28):** A planned bump to MC 1.21.5/1.21.6/26.x is blocked
> on upstream cubiomes shipping those versions first. The bundled submodule
> is already at the latest upstream `master`; the gap below describes what
> Mojang has released but cubiomes has not yet supported. When cubiomes
> publishes a newer commit (or accepts a community PR adding those
> versions), follow the **Bumping the bundled cubiomes** steps to roll it in.

## Versions you can pass to `--version`

The values below are accepted by `mcsf_str2mc` (see `vendor/cubiomes/biomes.h`
for the canonical enum, and `vendor/cubiomes/util.c` for the string parser).
For each major.minor track, the highest-patch alias is the one cubiomes treats
as "the version" — e.g. passing `1.20` is equivalent to `1.20.6`.

| Java Edition | Cubiomes enum | mc-seed-finder coverage |
| --- | --- | --- |
| 1.0 → 1.17.x | `MC_1_0` → `MC_1_17_1` | Structure RNG + biome filtering; not the target of the project (search defaults to 1.21) |
| 1.18, 1.18.2 | `MC_1_18`, `MC_1_18_2` | Full support — first version mc-seed-finder targets |
| 1.19.2 | `MC_1_19_2` | Full support |
| 1.19, 1.19.4 | `MC_1_19`, `MC_1_19_4` | Full support |
| 1.20, 1.20.6 | `MC_1_20`, `MC_1_20_6` | Full support |
| 1.21.1 | `MC_1_21_1` | Full support |
| 1.21.3 | `MC_1_21_3` | Full support (incl. Pale Garden) — bundled cubiomes maps "1.21" here |
| 1.21 alias | `MC_1_21` → `MC_1_21_WD` | Maps to the Winter Drop preview; treat as 1.21.3-equivalent biomes |

### Structures recognised

The `--list-structures` CLI flag is authoritative. As of this revision:

```
ancient_city
desert_pyramid
igloo
jungle_temple
ocean_monument
ocean_ruin
pillager_outpost
ruined_portal
shipwreck
stronghold
swamp_hut
trial_chambers
village
woodland_mansion
```

`buried_treasure` is supported. It uses a per-chunk `nextFloat() < 0.01`
placement (cubiomes `case Treasure`) routed through a dedicated
`roll_buried_treasure_chunk` path in both Rust and Python instead of the
standard region-grid framework. Anchor block is `(chunk_x * 16 + 9,
chunk_z * 16 + 9)` — note the `+9`, not `+8`. Parity with cubiomes is
covered by a sweep test in `structures::buried_treasure_tests`.

## Known gap vs upstream Mojang

| Mojang release | Status |
| --- | --- |
| 1.21.5 (Spring to Life, Mar 2025) | **Upstream cubiomes does not support this yet.** Adds pale-garden negative-weirdness, leaf litter, mansions-in-pale-garden, fallen-tree decorator. Will land here when cubiomes ships it. |
| 1.21.6 (Chase the Skies, Jun 2025) | Upstream cubiomes does not support this yet. No worldgen changes that affect seed-finding (ambient sounds, music) — adoption is mostly enum-tracking. |
| 1.21.7 → 1.21.11 | Upstream cubiomes does not support these (community-reported Mojang versions). |
| 26.1 (Tiny Takeover, Mar 2026) | Upstream cubiomes does not support this yet. First release using the `year.drop.hotfix` numbering scheme. No new biomes/structures, but version-enum churn will require a Rust-side rename when cubiomes adopts it. |

When a user requests a newer version than the bundle supports, the CLI
emits the error message documented above and exits with code 2. The
desktop app accepts the version string in its UI but its searches will
hit the same backend error path; future work will pull this guard up
into the desktop validator.

## Bumping the bundled cubiomes

1. `git -C crates/mcseedfinder-core/vendor/cubiomes fetch && git -C crates/mcseedfinder-core/vendor/cubiomes checkout <commit>`
2. Re-run `cargo test --features pyo3` and the Python suite. Expect renames
   like `MC_1_21_2 → MC_1_21_3` to surface as compile errors in
   `crates/mcseedfinder-core/src/biomes.rs` and `gpu_climate.rs`.
3. Refresh the golden vectors in `crates/mcseedfinder-core/src/biomes.rs::frozen_reference_vectors_*`
   if any new biome IDs landed (e.g. pale garden's `184` / similar).
4. Add a new row to **Current bundle** above and a new line under
   **Versions you can pass to `--version`** if any enum was added.
5. Update the comment in `crates/mcseedfinder-core/src/gpu_climate.rs::parse_mc_version`
   so the GPU path recognises the new versions.
6. Commit the submodule bump + the doc + golden-vector updates in a single
   PR; the PR title should call out the cubiomes commit hash for traceability.
