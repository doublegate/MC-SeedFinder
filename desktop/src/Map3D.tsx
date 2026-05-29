/**
 * 3D isometric biome map (Phase 7+).
 *
 * Renders the world as Minecraft-like surface voxels — one cube instance per
 * scale-grid cell — using:
 *   • per-instance y-position = approximate surface height (from cubiomes'
 *     `mapApproxHeight`, labelled "approximate terrain" in the HUD),
 *   • per-instance colour = biome at the slider-selected Y level (bit-exact
 *     via cubiomes' `genBiomes` at that Y).
 *
 * The side Y slider selects the biome sample depth → biome IDs change →
 * instance colours repaint. The heightmap doesn't depend on Y, so column
 * heights persist while you scrub the layer slider.
 *
 * Pointer-driven biome readout uses R3F's `onPointerMove` on the
 * InstancedMesh. `event.instanceId` identifies the cell; we look up its
 * (x, z) world coord and biome ID in the cached tile arrays, then push
 * the result to the parent's HUD via `onHover`.
 *
 * Accuracy: biome IDs and colours are bit-exact. The heightmap is
 * approximate (cubiomes spline-based, NOT bit-exact Java terrain) and
 * the "approximate terrain" pill in the HUD makes that visible.
 */

import { Canvas, useThree, type ThreeEvent } from "@react-three/fiber";
import { Html, OrbitControls, OrthographicCamera } from "@react-three/drei";
import { useEffect, useMemo, useRef, type ComponentRef } from "react";
import * as THREE from "three";
import { YSlider } from "./YSlider";

/** True when the host WebView advertises WebGPU. Tauri's WebView is one
 *  of WebKitGTK (Linux, no WebGPU as of late 2025), WebView2 (Windows,
 *  WebGPU behind --enable-unsafe-webgpu), or WKWebView (macOS, WebGPU
 *  baseline in Safari 26+). The current `<Canvas>` ships on WebGL2 in
 *  all cases — we only surface the capability so the HUD can communicate
 *  which path is in use. The async-gl swap to WebGPURenderer is a
 *  follow-up commit once R3F v9 + three/webgpu integration is stable
 *  enough to gate on. */
export function hasWebGPU(): boolean {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}

export type HoverInfo = {
  /** Block X coordinate of the hovered cell (centre of the scale-grid cell). */
  worldX: number;
  /** Block Z coordinate of the hovered cell. */
  worldZ: number;
  /** Wheel-driven block Y (echoed for the HUD; biome is sampled at this Y). */
  y: number;
  /** cubiomes biome ID at (worldX, y, worldZ), 0..255. */
  biomeId: number;
};

export type Map3DProps = {
  tile: {
    // Aliased over the binary IPC payload (D1). Random index access works
    // exactly like a number[] but with no JSON parse / no per-element copy.
    bytes: Uint8Array;
    biomeIds?: Uint8Array; // per-cell cubiomes biome IDs (u8)
    sx: number;
    sz: number;
    x: number; // top-left block X
    z: number; // top-left block Z
    scale: number;
  };
  heights: number[] | null; // sx*sz f32 block-heights, or null for flat plane
  pins: { structure: string; block_x: number; block_z: number }[];
  /** Camera zoom (orthographic). 1.0 = baseline; wheel adjusts continuously. */
  cameraZoom: number;
  /** Block Y currently selected by the side Y slider. */
  yLevel: number;
  /** 1.18+ build range — defines the YSlider track extent. */
  yMin: number;
  yMax: number;
  /** Wheel zoom callback. Factor > 1 zooms in, factor < 1 zooms out. */
  onZoomFactor: (factor: number) => void;
  /** Commit a completed 3D pan back to the app-level map centre in blocks. */
  onPanByBlocks: (dx: number, dz: number) => void;
  /** Direct Y setter (used by the side YSlider's drag handler). */
  onYSet: (y: number) => void;
  /** Pointer hover callback. Fires when the hovered cell changes; null when
   *  the pointer leaves the voxel mesh. */
  onHover: (info: HoverInfo | null) => void;
  /** Optional slime-chunk overlay: `[chunkX, chunkZ]` pairs that fall in
   *  the tile span. Rendered as flat green plates at the ground plane. */
  slimeChunks?: number[][];
  /** Whether to render the world-border wireframe (the canonical
   *  ±29 999 984 block rectangle). */
  showBorder?: boolean;
  /** Whether to render the 8 concentric stronghold ring constraint
   *  annuli (Java Edition canonical radii in blocks). */
  showStrongholdRings?: boolean;
  /** Cubiomes-computed world spawn (x, z) in block coords. Rendered
   *  as a star marker via drei `<Html>` when present and in tile span. */
  spawnPos?: { x: number; z: number } | null;
  /** Render the 16×16 spawn-chunks square centred on `spawnPos`. */
  showSpawnChunks?: boolean;
};

/** Java Edition stronghold ring (inner, outer) block radii. cubiomes'
 *  internal STRUCTURE_RING_DISTANCES; replicated here so the overlay is
 *  pure frontend (no extra Tauri command). The 8 rings each contain a
 *  fixed number of strongholds at random angles inside the annulus. */
const STRONGHOLD_RING_RADII: [number, number][] = [
  [1280, 2816],
  [4352, 5888],
  [7424, 8960],
  [10496, 12032],
  [13568, 15104],
  [16640, 18176],
  [19712, 21248],
  [22784, 24320],
];

/** Eight transparent ring annuli centred at world (0, 0). Each annulus
 *  is a `RingGeometry` (flat disk with a hole) rotated to lie on the
 *  ground plane. Rendered at the heightmap's ground level (y ≈ 0). */
function StrongholdRingsOverlay({ tile }: { tile: Map3DProps["tile"] }) {
  // Map world block coords → tile-grid units. The mesh is centred at
  // origin in grid units; world origin (0, 0) is at
  // (-tile.x / scale + sx/2, -tile.z / scale + sz/2). To draw rings
  // centred on world (0, 0), translate the ring meshes accordingly.
  const halfX = (tile.sx * tile.scale) / 2;
  const halfZ = (tile.sz * tile.scale) / 2;
  const cx = (0 - (tile.x + halfX)) / tile.scale;
  const cz = (0 - (tile.z + halfZ)) / tile.scale;
  return (
    <group position={[cx, 0.05, cz]} rotation={[-Math.PI / 2, 0, 0]}>
      {STRONGHOLD_RING_RADII.map(([inner, outer], i) => {
        const ri = inner / tile.scale;
        const ro = outer / tile.scale;
        return (
          <mesh key={i}>
            <ringGeometry args={[ri, ro, 64]} />
            <meshBasicMaterial
              color="#88ccff"
              transparent
              opacity={0.18}
              side={THREE.DoubleSide}
              depthWrite={false}
              toneMapped={false}
            />
          </mesh>
        );
      })}
    </group>
  );
}

/** Spawn-chunks tint: the 16×16-chunk (= 256×256 block) region centred
 *  on the world spawn point. Always loaded in vanilla, so e.g. AFK farm
 *  builders care about this rect. */
function SpawnChunksOverlay({
  tile,
  spawnPos,
}: {
  tile: Map3DProps["tile"];
  spawnPos: { x: number; z: number };
}) {
  const halfX = (tile.sx * tile.scale) / 2;
  const halfZ = (tile.sz * tile.scale) / 2;
  const centreX = tile.x + halfX;
  const centreZ = tile.z + halfZ;
  // Spawn chunks are 16 chunks * 16 blocks/chunk = 256 blocks across,
  // centred on spawn. In grid units: 256 / tile.scale.
  const sideGrid = 256 / tile.scale;
  const gx = (spawnPos.x - centreX) / tile.scale;
  const gz = (spawnPos.z - centreZ) / tile.scale;
  return (
    <mesh
      position={[gx, 0.12, gz]}
      rotation={[-Math.PI / 2, 0, 0]}
      renderOrder={1}
    >
      <planeGeometry args={[sideGrid, sideGrid]} />
      <meshBasicMaterial
        color="#ffcf3a"
        transparent
        opacity={0.18}
        side={THREE.DoubleSide}
        depthWrite={false}
        toneMapped={false}
      />
    </mesh>
  );
}

/** Spawn marker — golden star at the cubiomes-computed world spawn,
 *  rendered via drei `<Html>` so the star tracks the world point. */
function SpawnMarker({
  tile,
  spawnPos,
}: {
  tile: Map3DProps["tile"];
  spawnPos: { x: number; z: number };
}) {
  const halfX = (tile.sx * tile.scale) / 2;
  const halfZ = (tile.sz * tile.scale) / 2;
  const centreX = tile.x + halfX;
  const centreZ = tile.z + halfZ;
  // Skip if the spawn point is well outside the tile span.
  if (
    spawnPos.x < tile.x - halfX ||
    spawnPos.x > tile.x + tile.sx * tile.scale + halfX ||
    spawnPos.z < tile.z - halfZ ||
    spawnPos.z > tile.z + tile.sz * tile.scale + halfZ
  ) {
    return null;
  }
  return (
    <Html
      transform={false}
      center
      position={[
        (spawnPos.x - centreX) / tile.scale,
        1.2,
        (spawnPos.z - centreZ) / tile.scale,
      ]}
    >
      <div
        className="spawnStar"
        title={`World spawn @ (${spawnPos.x}, ${spawnPos.z}) — cubiomes getSpawn`}
      >
        ★
      </div>
    </Html>
  );
}

/** Imperative camera-zoom sync — wheel/key zoom drives a prop, and the
 *  orthographic camera's `zoom` field is updated here in response. */
function CameraZoomSync({ cameraZoom }: { cameraZoom: number }) {
  const { camera } = useThree();
  useEffect(() => {
    if (camera instanceof THREE.OrthographicCamera) {
      camera.zoom = cameraZoom;
      camera.updateProjectionMatrix();
    }
  }, [camera, cameraZoom]);
  return null;
}

function CameraFitSync({
  tile,
  cameraZoom,
}: {
  tile: Map3DProps["tile"];
  cameraZoom: number;
}) {
  const { camera, size } = useThree();
  useEffect(() => {
    if (!(camera instanceof THREE.OrthographicCamera)) return;

    const aspect = Math.max(0.1, size.width / Math.max(1, size.height));
    // The voxel map is a square grid viewed from an isometric camera. Use a
    // cover-style fit against the live canvas aspect so the scene grows into
    // wide/tall panes instead of preserving blank letterbox space.
    const footprintW = (tile.sx + tile.sz) * 0.78;
    const maxHeight = Math.max(tile.sx, tile.sz) * 0.58;
    const terrainH = Math.max(24, maxHeight);
    const footprintH = (tile.sx + tile.sz) * 0.38 + terrainH;
    const margin = 1.0;

    let halfW = (footprintW * margin) / 2;
    let halfH = (footprintH * margin) / 2;
    const footprintAspect = halfW / halfH;

    if (aspect > footprintAspect) {
      halfH = halfW / aspect;
    } else {
      halfW = halfH * aspect;
    }

    camera.left = -halfW;
    camera.right = halfW;
    camera.top = halfH;
    camera.bottom = -halfH;
    camera.zoom = cameraZoom;
    camera.updateProjectionMatrix();
  }, [camera, size.width, size.height, tile.sx, tile.sz, cameraZoom]);
  return null;
}

/** RELIEF maps block Y into grid units. Each rendered instance is a filled
 *  terrain column whose top sits on an integer voxel layer. This avoids side
 *  view "floating surface cube" holes while keeping the instance count bounded
 *  to one terrain column per tile cell.
 *
 *  COLUMN_SIZE is only slightly below 1.0. The clear colour shows through as
 *  subtle seams between neighboring columns without overwhelming biome colour.
 *  This avoids Three.js wireframe diagonals, which draw the cube face
 *  triangulation and alias badly at zoomed-out isometric views. */
const RELIEF = 2.0;
const COLUMN_SIZE = 0.9;
const SCENE_CLEAR = "#0f1711";

function worldYToSceneY(worldY: number, scale: number, yMin: number): number {
  return Math.round(((worldY - yMin) / scale) * RELIEF);
}

function columnHeight(rawY: number | undefined, scale: number, yMin: number): number {
  if (rawY == null) return 1;
  const h = worldYToSceneY(rawY, scale, yMin) + 1;
  return Math.max(1, h);
}

/** InstancedMesh of one filled terrain column per (i, j) cell. Per-instance
 *  height → approximate surface height; per-instance colour → biome at
 *  current Y. Fine dark seams come from the sub-1.0 column width/depth
 *  exposing the dark background between adjacent columns.
 *
 *  Update strategy:
 *  - geometry/material/mesh are created once per (sx, sz) — recreating
 *    65k instances per frame would be a real cost.
 *  - per-instance MATRIX (scale + position) is rewritten when the
 *    heightmap arrives or changes.
 *  - per-instance COLOR is rewritten whenever the tile bytes change
 *    (each wheel-Y scrub fires new bytes).
 *  - both updates flag `instanceMatrix.needsUpdate` / `instanceColor.needsUpdate`. */
function VoxelColumns({
  tile,
  heights,
  yLevel,
  yMin,
  onHover,
}: {
  tile: Map3DProps["tile"];
  heights: number[] | null;
  yLevel: number;
  yMin: number;
  onHover: (info: HoverInfo | null) => void;
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const sx = tile.sx;
  const sz = tile.sz;
  const count = sx * sz;

  // Shared scratch objects — avoid per-instance allocations.
  const tmpMatrix = useMemo(() => new THREE.Matrix4(), []);
  const tmpColor = useMemo(() => new THREE.Color(), []);
  const decorateBlockMaterial = useMemo(
    () => (shader: { vertexShader: string; fragmentShader: string }) => {
      shader.vertexShader = shader.vertexShader
        .replace(
          "#include <common>",
          `#include <common>
varying vec3 vBlockLocal;
varying vec3 vBlockNormal;
varying float vBlockHeight;`,
        )
        .replace(
          "#include <begin_vertex>",
          `#include <begin_vertex>
vBlockLocal = position + vec3(0.5);
vBlockNormal = normal;
#ifdef USE_INSTANCING
  vBlockHeight = max(1.0, length(instanceMatrix[1].xyz));
#else
  vBlockHeight = 1.0;
#endif`,
        );
      shader.fragmentShader = shader.fragmentShader
        .replace(
          "#include <common>",
          `#include <common>
varying vec3 vBlockLocal;
varying vec3 vBlockNormal;
varying float vBlockHeight;

float mcsfLine(float coord, float width) {
  float d = min(fract(coord), 1.0 - fract(coord));
  return 1.0 - smoothstep(width, width + 0.015, d);
}`,
        )
        .replace(
          "#include <opaque_fragment>",
          `vec3 n = normalize(vBlockNormal);
float topFace = step(0.55, n.y);
float sideFace = max(step(0.55, abs(n.x)), step(0.55, abs(n.z)));
float sideCoord = mix(vBlockLocal.x, vBlockLocal.z, step(0.55, abs(n.x)));
float topGrid = max(mcsfLine(vBlockLocal.x, 0.045), mcsfLine(vBlockLocal.z, 0.045));
float sideGrid = max(mcsfLine(sideCoord, 0.038), mcsfLine(vBlockLocal.y * vBlockHeight, 0.036));
float grid = max(topGrid * topFace, sideGrid * sideFace);
outgoingLight = mix(outgoingLight, outgoingLight * 0.18, grid * 0.82);
#include <opaque_fragment>`,
        );
    },
    [],
  );

  // (Re)write per-instance matrices whenever sx/sz/heights change.
  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const halfX = sx / 2;
    const halfZ = sz / 2;
    const layerH = columnHeight(yLevel, tile.scale, yMin);
    for (let j = 0; j < sz; j++) {
      for (let i = 0; i < sx; i++) {
        const idx = j * sx + i;
        const surfaceH = columnHeight(heights?.[idx], tile.scale, yMin);
        const h = Math.min(surfaceH, layerH);
        const xCell = i + 0.5 - halfX;
        const zCell = j + 0.5 - halfZ;
        tmpMatrix.makeScale(COLUMN_SIZE, h, COLUMN_SIZE);
        tmpMatrix.setPosition(xCell, h * 0.5, zCell);
        mesh.setMatrixAt(idx, tmpMatrix);
      }
    }
    mesh.instanceMatrix.needsUpdate = true;
    mesh.computeBoundingSphere(); // raycast culling
  }, [heights, sx, sz, tile.scale, yLevel, yMin, tmpMatrix]);

  // Write per-instance colors once per tile. The previous crossfade rewrote
  // every instance color on every animation frame; with large panes that can
  // monopolise the WebView for seconds. A direct update is cheaper and more
  // stable, especially while the user scrubs Y slices with the slider.
  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    for (let i = 0; i < count; i++) {
      const bOff = i * 4;
      tmpColor.setRGB(
        tile.bytes[bOff] / 255,
        tile.bytes[bOff + 1] / 255,
        tile.bytes[bOff + 2] / 255,
      );
      mesh.setColorAt(i, tmpColor);
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [tile.bytes, count, tmpColor]);

  // Throttle hover-state writes: only push to the parent when the hovered
  // cell index changes. Without this we'd fire a setState 60+ times per
  // second while the cursor moves across cells.
  const lastHoverIdRef = useRef<number | null>(null);

  const handlePointerMove = (e: ThreeEvent<PointerEvent>) => {
    const instanceId = e.instanceId;
    if (instanceId == null) return;
    if (lastHoverIdRef.current === instanceId) return;
    lastHoverIdRef.current = instanceId;
    const i = instanceId % sx;
    const j = Math.floor(instanceId / sx);
    const worldX = tile.x + (i + 0.5) * tile.scale; // centre of cell, block coords
    const worldZ = tile.z + (j + 0.5) * tile.scale;
    const biomeId = tile.biomeIds?.[instanceId] ?? 255;
    onHover({ worldX: Math.floor(worldX), worldZ: Math.floor(worldZ), y: yLevel, biomeId });
  };

  const handlePointerOut = () => {
    if (lastHoverIdRef.current != null) {
      lastHoverIdRef.current = null;
      onHover(null);
    }
  };

  return (
    <group>
      <instancedMesh
        ref={meshRef}
        args={[undefined, undefined, count]}
        onPointerMove={handlePointerMove}
        onPointerOut={handlePointerOut}
      >
        <boxGeometry args={[1, 1, 1]} />
        <meshBasicMaterial
          toneMapped={false}
          fog={false}
          dithering
          onBeforeCompile={decorateBlockMaterial}
          customProgramCacheKey={() => "mcsf-block-grid-v1"}
        />
      </instancedMesh>
    </group>
  );
}

/** Translucent yellow plane at Y = `yLevel` so the user sees where they
 *  are inside the column stack. Only rendered when the heightmap is
 *  available (otherwise there are no real columns to cut through). */
function YPlaneIndicator({
  tile,
  yLevel,
  yMin,
  show,
}: {
  tile: Map3DProps["tile"];
  yLevel: number;
  yMin: number;
  show: boolean;
}) {
  if (!show) return null;
  const planeY = columnHeight(yLevel, tile.scale, yMin);
  return (
    <mesh position={[0, planeY, 0]} rotation={[-Math.PI / 2, 0, 0]} renderOrder={2}>
      <planeGeometry args={[tile.sx, tile.sz]} />
      <meshBasicMaterial
        color="#ffd966"
        transparent
        opacity={0.12}
        side={THREE.DoubleSide}
        depthWrite={false}
      />
    </mesh>
  );
}

function PinOverlay({ tile, pins }: { tile: Map3DProps["tile"]; pins: Map3DProps["pins"] }) {
  // Mesh spans [-sx/2, +sx/2] × [-sz/2, +sz/2] grid units. World (block) coord
  // for centre of pane is (tile.x + sx*scale/2, tile.z + sz*scale/2); each
  // pin is offset by (blockX - centre) / scale grid units.
  const halfX = (tile.sx * tile.scale) / 2;
  const halfZ = (tile.sz * tile.scale) / 2;
  const centreX = tile.x + halfX;
  const centreZ = tile.z + halfZ;

  return (
    <>
      {Math.abs(0 - centreX) <= halfX && Math.abs(0 - centreZ) <= halfZ && (
        <Html
          transform={false}
          center
          position={[(0 - centreX) / tile.scale, 1.0, (0 - centreZ) / tile.scale]}
          className="spawn3d"
        >
          <div className="spawn">0,0</div>
        </Html>
      )}
      {pins.map((p) => (
        <Html
          key={`${p.structure}-${p.block_x}-${p.block_z}`}
          transform={false}
          center
          position={[
            (p.block_x - centreX) / tile.scale,
            1.0,
            (p.block_z - centreZ) / tile.scale,
          ]}
        >
          <button
            className={`pin pin-${p.structure}`}
            title={`${p.structure.replace(/_/g, " ")} @ (${p.block_x}, ${p.block_z}) — right-click to copy coords`}
            onContextMenu={(e) => {
              e.preventDefault();
              // Copy coords on right-click. Browser-native; succeeds in
              // any Tauri WebView (clipboard-write is allowed by default).
              const coords = `${p.block_x}, ${p.block_z}`;
              navigator.clipboard?.writeText(coords).catch(() => {
                /* clipboard denied; silently noop */
              });
              // Visual cue: briefly add a class so CSS animates a flash.
              const target = e.currentTarget;
              target.classList.add("pinFlash");
              window.setTimeout(() => target.classList.remove("pinFlash"), 600);
            }}
          />
        </Html>
      ))}
    </>
  );
}

/** Flat green plates for slime chunks. One InstancedMesh of small boxes
 *  at the ground plane (y ≈ 0.1, just above the bottom of the voxel
 *  columns). Each chunk = 16 blocks = 16/scale grid units. */
function SlimeChunksOverlay({
  tile,
  chunks,
}: {
  tile: Map3DProps["tile"];
  chunks: number[][];
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const tmpMatrix = useMemo(() => new THREE.Matrix4(), []);
  const halfX = tile.sx / 2;
  const halfZ = tile.sz / 2;
  // Tile origin (top-left) in block coords; chunk (cx, cz) covers
  // [cx*16, (cx+1)*16). Map to grid units (1 unit = scale blocks).
  const chunkSpan = 16 / tile.scale; // 4 grid units at scale=4

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    for (let i = 0; i < chunks.length; i++) {
      const cx = chunks[i][0];
      const cz = chunks[i][1];
      const worldX = cx * 16; // top-left block of chunk
      const worldZ = cz * 16;
      // Centre of the chunk in grid units relative to tile centre.
      const gx = (worldX + 8 - tile.x) / tile.scale - halfX;
      const gz = (worldZ + 8 - tile.z) / tile.scale - halfZ;
      tmpMatrix.makeScale(chunkSpan, 0.2, chunkSpan);
      tmpMatrix.setPosition(gx, 0.1, gz);
      mesh.setMatrixAt(i, tmpMatrix);
    }
    mesh.count = chunks.length;
    mesh.instanceMatrix.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [chunks, tile.x, tile.z, tile.scale, halfX, halfZ, chunkSpan, tmpMatrix]);

  if (chunks.length === 0) return null;
  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, Math.max(1, chunks.length)]}>
      <boxGeometry args={[1, 1, 1]} />
      <meshBasicMaterial color="#3aff8a" transparent opacity={0.55} toneMapped={false} />
    </instancedMesh>
  );
}

/** World border wireframe — the canonical ±29 999 984 block rectangle.
 *  Rendered as four vertical line segments at the cardinal edges when
 *  any of them falls inside the tile's world bounds. */
function WorldBorderWireframe({ tile, height }: { tile: Map3DProps["tile"]; height: number }) {
  const B = 29_999_984;
  // Map a world block coord to grid units relative to tile centre.
  const halfX = (tile.sx * tile.scale) / 2;
  const halfZ = (tile.sz * tile.scale) / 2;
  const centreX = tile.x + halfX;
  const centreZ = tile.z + halfZ;
  const toGrid = (worldX: number, worldZ: number) => ({
    gx: (worldX - centreX) / tile.scale,
    gz: (worldZ - centreZ) / tile.scale,
  });
  // The border is far outside any practical tile, so usually nothing renders.
  // Build a vertex array for whichever edges are in view. Each edge is a
  // line segment at y=0 ↔ y=height so it's visible above the columns.
  const points = useMemo(() => {
    const out: number[] = [];
    const push = (x1: number, z1: number, x2: number, z2: number) => {
      const a = toGrid(x1, z1);
      const b = toGrid(x2, z2);
      out.push(a.gx, 0, a.gz, b.gx, 0, b.gz);
      out.push(a.gx, height, a.gz, b.gx, height, b.gz);
      out.push(a.gx, 0, a.gz, a.gx, height, a.gz);
      out.push(b.gx, 0, b.gz, b.gx, height, b.gz);
    };
    // West (x = -B) — visible if any of the tile's X range is east of it.
    if (tile.x <= -B && tile.x + tile.sx * tile.scale >= -B) {
      push(-B, tile.z, -B, tile.z + tile.sz * tile.scale);
    }
    // East (x = +B)
    if (tile.x <= B && tile.x + tile.sx * tile.scale >= B) {
      push(B, tile.z, B, tile.z + tile.sz * tile.scale);
    }
    // North (z = -B)
    if (tile.z <= -B && tile.z + tile.sz * tile.scale >= -B) {
      push(tile.x, -B, tile.x + tile.sx * tile.scale, -B);
    }
    // South (z = +B)
    if (tile.z <= B && tile.z + tile.sz * tile.scale >= B) {
      push(tile.x, B, tile.x + tile.sx * tile.scale, B);
    }
    return new Float32Array(out);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tile.x, tile.z, tile.sx, tile.sz, tile.scale, height]);
  if (points.length === 0) return null;
  return (
    <lineSegments>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[points, 3]} />
      </bufferGeometry>
      <lineBasicMaterial color="#ff5050" transparent opacity={0.9} toneMapped={false} />
    </lineSegments>
  );
}

/** Stable renderer factory for the R3F `<Canvas>`. Keep the interactive map on
 * WebGL2 for now. The earlier optimistic WebGPURenderer path required async
 * init through R3F's `gl` callback and could leave the pane blank or tear down
 * the WebView on some desktop WebViews after Run-triggered re-renders. */
function makeRenderer(props: { canvas: unknown }): THREE.WebGLRenderer {
  const renderer = new THREE.WebGLRenderer({
    canvas: props.canvas as HTMLCanvasElement,
    antialias: true,
    alpha: false,
    powerPreference: "high-performance",
  });
  renderer.setClearColor(SCENE_CLEAR, 1);
  return renderer;
}

export function Map3D(props: Map3DProps) {
  const {
    tile,
    heights,
    pins,
    cameraZoom,
    yLevel,
    yMin,
    yMax,
    onZoomFactor,
    onPanByBlocks,
    onYSet,
    onHover,
    slimeChunks,
    showBorder,
    showStrongholdRings,
    spawnPos,
    showSpawnChunks,
  } = props;

  // Wheel handler — zooms the rendered 3D view. Depth/Y is adjusted only
  // through the side slider so scrolling matches normal 3D viewport behavior.
  const onWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    e.preventDefault();
    onZoomFactor(Math.exp(-e.deltaY / 500));
  };

  // Isometric-ish camera position: above and to the front-right, looking at
  // the origin. Orthographic ignores distance for size; `zoom` does scale.
  const camPos: [number, number, number] = [tile.sx * 0.7, tile.sx * 0.9, tile.sz * 0.7];
  const span = Math.max(tile.sx, tile.sz);
  const controlsRef = useRef<ComponentRef<typeof OrbitControls>>(null);

  const commitPan = () => {
    const controls = controlsRef.current;
    if (!controls) return;
    const dxBlocks = Math.round(controls.target.x * tile.scale);
    const dzBlocks = Math.round(controls.target.z * tile.scale);
    if (Math.abs(dxBlocks) < tile.scale && Math.abs(dzBlocks) < tile.scale) {
      return;
    }
    // OrbitControls pans the camera target inside the currently-loaded tile.
    // Commit that offset to the parent viewCenter so the normal tile-fetch
    // effect requests the newly exposed world area, then reset the local pan
    // so the replacement tile is centred in the 3D viewport.
    onPanByBlocks(dxBlocks, dzBlocks);
    controls.object.position.sub(controls.target);
    controls.target.set(0, 0, 0);
    controls.update();
  };

  return (
    <div
      className="map3dCanvas"
      onWheel={onWheel}
      onContextMenu={(e) => e.preventDefault()}
      onPointerLeave={() => onHover(null)}
      style={{ width: "100%", height: "100%" }}
    >
      <Canvas
        orthographic
        dpr={1}
        gl={makeRenderer}
        flat
        style={{ background: SCENE_CLEAR }}
      >
        <OrthographicCamera
          makeDefault
          position={camPos}
          zoom={cameraZoom}
          near={-span * 4}
          far={span * 4}
        />
        <CameraFitSync tile={tile} cameraZoom={cameraZoom} />
        <CameraZoomSync cameraZoom={cameraZoom} />
        <OrbitControls
          ref={controlsRef}
          makeDefault
          enableRotate
          enablePan
          enableZoom={false}
          enableDamping
          dampingFactor={0.08}
          mouseButtons={{
            LEFT: THREE.MOUSE.PAN,
            MIDDLE: THREE.MOUSE.DOLLY,
            RIGHT: THREE.MOUSE.ROTATE,
          }}
          screenSpacePanning
          target={[0, 0, 0]}
          onEnd={commitPan}
        />
        <VoxelColumns
          tile={tile}
          heights={heights}
          yLevel={yLevel}
          yMin={yMin}
          onHover={onHover}
        />
        <YPlaneIndicator tile={tile} yLevel={yLevel} yMin={yMin} show={heights != null} />
        {slimeChunks && slimeChunks.length > 0 && (
          <SlimeChunksOverlay tile={tile} chunks={slimeChunks} />
        )}
        {showBorder && <WorldBorderWireframe tile={tile} height={Math.max(tile.sx, tile.sz) * 0.1} />}
        {showStrongholdRings && <StrongholdRingsOverlay tile={tile} />}
        {showSpawnChunks && spawnPos && <SpawnChunksOverlay tile={tile} spawnPos={spawnPos} />}
        {spawnPos && <SpawnMarker tile={tile} spawnPos={spawnPos} />}
        <PinOverlay tile={tile} pins={pins} />
      </Canvas>
      <YSlider y={yLevel} yMin={yMin} yMax={yMax} onChange={onYSet} />
    </div>
  );
}
