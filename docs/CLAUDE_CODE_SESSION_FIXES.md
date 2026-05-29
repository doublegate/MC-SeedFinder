# Claude Code Session Fixes

This document summarizes the final applied fixes from the desktop performance
and 3D-map stabilization session. It intentionally omits intermediate
trial-and-error attempts and records only the behavior that should be preserved.

## Scope

Primary files changed:

- `desktop/src-tauri/src/main.rs`
- `desktop/src/main.tsx`
- `desktop/src/Map3D.tsx`
- `desktop/src/styles.css`
- `desktop/vite.config.ts`
- `AGENTS.md`

The existing dirty cubiomes submodule state was not modified.

## Desktop Stability And IPC

The Tauri command layer had several request-supersession paths that could make
foreground map rendering fail after unrelated background work.

Final applied behavior:

- Tile request supersession is purpose-aware.
- Foreground map tile requests use the `map` purpose and can supersede older
  foreground map work.
- Thumbnail and speculative prefetch tile requests use non-map purposes and do
  not cancel visible map tile rendering.
- Height tile requests use a separate `height_counter`, so heightmap work no
  longer cancels foreground biome tiles.
- Pins and analysis retain separate counters.
- `search-started` is emitted before the search worker thread is spawned, so
  fast searches cannot emit match/completion events before the frontend receives
  search metadata.

These changes live in `desktop/src-tauri/src/main.rs`.

## Search Event Filtering

The frontend now filters streamed search events by active job id more strictly.
This prevents stale events from previous jobs from updating results or job
state.

Final applied behavior in `desktop/src/main.tsx`:

- `search-started` is accepted only when it matches the active wildcard/job.
- `search-matches` uses `jobMatches(...)` instead of only checking whether any
  active job exists.
- If `start_search` resolves after a tiny search has already completed and
  cleared the active ref, the frontend does not revive the job id.

## 3D Tile Sizing And Resizing

The 3D map previously reshaped the rendered tile grid to the pane aspect ratio.
On wide panes this could refetch a narrow strip, making the map collapse into a
long sliver after resize.

Final applied behavior:

- 3D tiles use a fixed square `256x256` sample grid.
- Window resize changes the canvas and camera framing, not the shape of the
  sampled world tile.
- The R3F canvas is forced to fill the full `.map3dCanvas` wrapper.
- The camera uses a resize-aware cover fit so the scene uses the available pane
  instead of preserving blank letterbox space.

Relevant files:

- `desktop/src/main.tsx`
- `desktop/src/Map3D.tsx`
- `desktop/src/styles.css`

## 3D Input Model

The 3D interaction model was split into display-only controls and map-scale
controls.

Final behavior:

- Mouse wheel in 3D changes only `displayZoom3D`, a camera/display zoom.
- Mouse wheel in 3D does not change cubiomes scale, tile loading, or map zoom
  steps.
- `+` and `-` buttons are restored and change `zoomLevel`, the real map zoom
  step that can affect tile scale/loading.
- Left mouse drag pans the 3D view.
- Right mouse drag rotates the 3D view.
- 3D pan completion commits the local camera target offset back into
  `viewCenter`, triggering the existing tile/pin/height fetch pipeline for
  newly exposed surrounding terrain.
- After committing a 3D pan, the local camera target is reset so the replacement
  tile recenters correctly.
- The Y slider remains the layer/depth control.

Implementation notes:

- `Map3D` uses `OrbitControls` instead of `MapControls`.
- `OrbitControls` has wheel zoom disabled because React owns display-only wheel
  zoom.
- Mouse button mapping is: left pan, middle dolly unused, right rotate.

## 3D Voxel Rendering

The 3D terrain went through several visual fixes. The final state avoids the
bad rendering paths that caused diagonal wireframe X-lines, black top faces,
muddy colors, and missing terrain.

Final behavior in `desktop/src/Map3D.tsx`:

- The renderer uses one instanced terrain column per tile cell for performance.
- Columns are filled from Minecraft `yMin` upward, so side views do not show
  floating surface voxels with large vertical holes.
- Scene Y now maps from Minecraft world Y using `yMin` instead of treating
  scene `Y=0` as Minecraft `Y=0`.
- Below-ground content is represented by mapping Minecraft `yMin` (`-64`) to
  scene zero.
- The Y slider clips actual rendered column height to
  `min(surfaceHeight, selectedYLayer)`; it no longer only moves a translucent
  plane.
- The voxel material remains `meshBasicMaterial` so biome colors are not
  darkened by lighting.
- Do not re-add the `vertexColors` material prop: it caused black/missing top
  face rendering with the instanced color path.
- Do not use Three.js `wireframe` for block lines: it draws triangulated face
  diagonals and creates X-shaped aliasing.
- Do not use Lambert lighting for biome colors unless the color pipeline is
  redesigned; it made terrain muddy/dark.
- Do not use polygon offset on the terrain material; it contributed to face
  rendering artifacts.
- Block/layer seams are injected with `onBeforeCompile` on the known-good
  `meshBasicMaterial`, preserving Three's built-in instancing and instance
  color handling.

The current seam shader darkens `outgoingLight` before `opaque_fragment`:

- top faces get square grid seams,
- side faces get vertical seams,
- side faces get repeated horizontal layer seams so tall columns read as
  stacked blocks rather than smooth pillars.

## Result List Performance

Large result sets no longer render every match as a live DOM row.

Final behavior in `desktop/src/main.tsx`:

- The full result array is preserved for counts, histogram, and export.
- The visible result list renders only the first `500` rows.
- A hint reports how many additional matches are kept for export.

This avoids UI churn and thumbnail fan-out during searches with many matches.

## Vite Chunk Warning

A Vite config was added at `desktop/vite.config.ts`.

Final behavior:

- React, Tauri, React Three Fiber/Drei, and Three.js are split into explicit
  manual chunks.
- The chunk warning threshold is set to `750` KB because Three.js is a known
  large vendor dependency for the 3D map.
- The build no longer emits the previous `>500 kB` warning.

## Verification Commands Used

The following commands passed after the applied fixes:

```bash
npm --prefix desktop run build
RUSTC_WRAPPER= cargo check --manifest-path desktop/src-tauri/Cargo.toml
PYTHONPATH=src python -m unittest
cargo test --features pyo3 --manifest-path crates/mcseedfinder-core/Cargo.toml
RUSTC_WRAPPER= cargo clippy --features pyo3 --manifest-path crates/mcseedfinder-core/Cargo.toml --all-targets -- -D warnings
RUSTC_WRAPPER= cargo clippy --manifest-path desktop/src-tauri/Cargo.toml --all-targets -- -D warnings
```

Only the first two commands were rerun after the final Y-min/below-ground fix,
because that final change was isolated to the desktop renderer and Tauri build
surface.

## Known Repository State During This Session

At the time of this document:

- Local `main` tracks `origin/main`.
- `AGENTS.md` has been updated with current project guidance.
- `crates/mcseedfinder-core/vendor/cubiomes` remains dirty from pre-existing
  local submodule edits and was not reset.

