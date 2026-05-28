/**
 * 3D isometric biome map (Phase 7+).
 *
 * Renders the world as **true voxel columns** — one `InstancedMesh` instance
 * per scale-grid cell — using:
 *   • per-instance scale = approximate surface height (from cubiomes'
 *     `mapApproxHeight`, labelled "approximate terrain" in the HUD),
 *   • per-instance colour = biome at the wheel-driven Y level (bit-exact
 *     via cubiomes' `genBiomes` at that Y).
 *
 * The wheel scrubs the Y slice → biome IDs change → instance colours
 * repaint. The heightmap doesn't depend on Y, so column heights persist
 * while you scroll through Y.
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

import { Canvas, useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { Html, MapControls, OrthographicCamera } from "@react-three/drei";
import { useEffect, useMemo, useRef } from "react";
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
    bytes: number[];
    biomeIds?: number[]; // per-cell cubiomes biome IDs (u8 in number[])
    sx: number;
    sz: number;
    x: number; // top-left block X
    z: number; // top-left block Z
    scale: number;
  };
  heights: number[] | null; // sx*sz f32 block-heights, or null for flat plane
  pins: { structure: string; block_x: number; block_z: number }[];
  /** Camera zoom (orthographic). 1.0 = baseline; +/- buttons multiply by 1.4. */
  cameraZoom: number;
  /** Block Y the wheel is currently scrubbed to. */
  yLevel: number;
  /** 1.18+ build range — defines the YSlider track extent. */
  yMin: number;
  yMax: number;
  /** Wheel delta callback (4-block step per notch; Shift = ×4 / 16 blocks). */
  onYDelta: (delta: number) => void;
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

/** Imperative camera-zoom sync — the +/- buttons drive a prop, and the
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

/** RELIEF scales the heightmap into Three.js world units. heights are in
 *  blocks; the voxel grid is in scale-grid units (1 unit = `scale` blocks).
 *  We divide by scale to keep heights proportional to width, then multiply
 *  by RELIEF to exaggerate vertical relief — 2.0 makes a 200-block
 *  mountain ~100 grid units tall, unmistakeably 3D.
 *
 *  MIN_H ensures cells whose approximate height is ≤ 0 (deep ocean floor,
 *  or absent heightmap) still render a thin slab the user can hover over.
 *
 *  FADE_MS is the crossfade duration when new tile bytes arrive. ~120ms
 *  is long enough to read as a "transition" rather than a snap, short
 *  enough not to feel laggy. */
const RELIEF = 2.0;
const MIN_H = 0.6;
const FADE_MS = 120;

function cellHeight(rawY: number | undefined, scale: number): number {
  if (rawY == null) return MIN_H;
  const h = (rawY / scale) * RELIEF;
  return Math.max(MIN_H, h);
}

/** InstancedMesh of one thin box per (i, j) cell. Per-instance scale →
 *  voxel column height; per-instance colour → biome at current Y.
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
  onHover,
}: {
  tile: Map3DProps["tile"];
  heights: number[] | null;
  yLevel: number;
  onHover: (info: HoverInfo | null) => void;
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const sx = tile.sx;
  const sz = tile.sz;
  const count = sx * sz;

  // Shared scratch objects — avoid per-instance allocations.
  const tmpMatrix = useMemo(() => new THREE.Matrix4(), []);
  const tmpColor = useMemo(() => new THREE.Color(), []);

  // (Re)write per-instance matrices whenever sx/sz/heights change.
  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const halfX = sx / 2;
    const halfZ = sz / 2;
    for (let j = 0; j < sz; j++) {
      for (let i = 0; i < sx; i++) {
        const idx = j * sx + i;
        const h = cellHeight(heights?.[idx], tile.scale);
        // Centre each box at (i + 0.5 - sx/2, h/2, j + 0.5 - sz/2). Bottom
        // sits on the ground plane (y=0); top reaches y=h.
        const xCell = i + 0.5 - halfX;
        const zCell = j + 0.5 - halfZ;
        tmpMatrix.makeScale(1, h, 1);
        tmpMatrix.setPosition(xCell, h * 0.5, zCell);
        mesh.setMatrixAt(idx, tmpMatrix);
      }
    }
    mesh.instanceMatrix.needsUpdate = true;
    mesh.computeBoundingSphere(); // raycast culling
  }, [heights, sx, sz, tile.scale, tmpMatrix]);

  // Crossfade animation state. When new tile bytes arrive, snapshot the
  // CURRENTLY-displayed RGB into `prevColorsRef` and start a fresh fade
  // timer. The useFrame loop below blends prev → new over FADE_MS.
  //
  // We snapshot the live instanceColor buffer (which itself may be
  // mid-fade) so rapid wheel scrubs animate from "whatever you're seeing
  // right now" → "the freshly-arrived bytes", never snapping back to a
  // pre-fade baseline.
  const prevColorsRef = useRef<Float32Array | null>(null);
  const animStartRef = useRef<number | null>(null);

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    // Initialise the prev buffer on first tile, or resize on (sx, sz) change.
    if (!prevColorsRef.current || prevColorsRef.current.length !== count * 3) {
      prevColorsRef.current = new Float32Array(count * 3);
      // First render: bake new bytes directly so we don't fade from black.
      for (let i = 0; i < count; i++) {
        const off = i * 4;
        const cOff = i * 3;
        prevColorsRef.current[cOff] = tile.bytes[off] / 255;
        prevColorsRef.current[cOff + 1] = tile.bytes[off + 1] / 255;
        prevColorsRef.current[cOff + 2] = tile.bytes[off + 2] / 255;
      }
      // Write to instanceColor so something is visible before useFrame runs.
      for (let i = 0; i < count; i++) {
        const cOff = i * 3;
        tmpColor.setRGB(
          prevColorsRef.current[cOff],
          prevColorsRef.current[cOff + 1],
          prevColorsRef.current[cOff + 2],
        );
        mesh.setColorAt(i, tmpColor);
      }
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      animStartRef.current = null;
      return;
    }
    // Subsequent updates: snapshot live colours (possibly mid-fade) as
    // the new starting point, then start a fresh animation.
    if (mesh.instanceColor) {
      prevColorsRef.current.set(mesh.instanceColor.array as Float32Array);
    }
    animStartRef.current = performance.now();
  }, [tile.bytes, count, tmpColor]);

  useFrame(() => {
    if (animStartRef.current == null) return;
    const mesh = meshRef.current;
    if (!mesh) return;
    const prev = prevColorsRef.current;
    if (!prev) return;
    const elapsed = performance.now() - animStartRef.current;
    const t = Math.min(1, elapsed / FADE_MS);
    for (let i = 0; i < count; i++) {
      const cOff = i * 3;
      const bOff = i * 4;
      const newR = tile.bytes[bOff] / 255;
      const newG = tile.bytes[bOff + 1] / 255;
      const newB = tile.bytes[bOff + 2] / 255;
      tmpColor.setRGB(
        prev[cOff] + (newR - prev[cOff]) * t,
        prev[cOff + 1] + (newG - prev[cOff + 1]) * t,
        prev[cOff + 2] + (newB - prev[cOff + 2]) * t,
      );
      mesh.setColorAt(i, tmpColor);
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    if (t >= 1) {
      animStartRef.current = null;
    }
  });

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
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, count]}
      onPointerMove={handlePointerMove}
      onPointerOut={handlePointerOut}
    >
      <boxGeometry args={[1, 1, 1]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  );
}

/** Translucent yellow plane at Y = `yLevel` so the user sees where they
 *  are inside the column stack. Only rendered when the heightmap is
 *  available (otherwise there are no real columns to cut through). */
function YPlaneIndicator({
  tile,
  yLevel,
  show,
}: {
  tile: Map3DProps["tile"];
  yLevel: number;
  show: boolean;
}) {
  if (!show) return null;
  const planeY = cellHeight(yLevel, tile.scale);
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

/** Async-gl factory for the R3F `<Canvas>`. Tries WebGPURenderer when
 *  navigator.gpu is available; falls back to R3F's default WebGLRenderer
 *  by returning `null`-equivalent (a plain WebGLRenderer params object).
 *  On Linux WebKitGTK navigator.gpu is undefined → the WebGL2 path is
 *  taken unchanged. */
async function makeRenderer(props: {
  canvas: HTMLCanvasElement;
}): Promise<THREE.WebGLRenderer> {
  if (hasWebGPU()) {
    try {
      // Dynamic import so WebGL-only platforms never load three/webgpu.
      // Three.js 0.171+: WebGPURenderer is the default export of three/webgpu.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const mod: any = await import("three/webgpu");
      const WebGPURenderer = mod.WebGPURenderer ?? mod.default;
      if (WebGPURenderer) {
        const r = new WebGPURenderer({ canvas: props.canvas, antialias: false });
        await r.init();
        // R3F's gl callback contract wants a WebGLRenderer-compatible
        // shape; WebGPURenderer implements the same surface (render,
        // setSize, setPixelRatio, dispose, etc.) but TS doesn't know.
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        return r as any;
      }
    } catch (e) {
      console.warn("[Map3D] WebGPU init failed, falling back to WebGL2:", e);
    }
  }
  const fallback = new THREE.WebGLRenderer({
    canvas: props.canvas,
    antialias: false,
    alpha: false,
  });
  return fallback;
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
    onYDelta,
    onYSet,
    onHover,
    slimeChunks,
    showBorder,
    showStrongholdRings,
    spawnPos,
    showSpawnChunks,
  } = props;

  // Wheel handler — bypasses MapControls (which has wheel-zoom off). Step
  // size = 4 blocks (one scale-Y unit at cubScale=4) so every notch crosses
  // a cubiomes scale-Y boundary; Shift = ×4 (one chunk-section).
  const onWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    e.preventDefault();
    const sign = e.deltaY > 0 ? -1 : 1;
    const mag = e.shiftKey ? 16 : 4;
    onYDelta(sign * mag);
  };

  // Isometric-ish camera position: above and to the front-right, looking at
  // the origin. Orthographic ignores distance for size; `zoom` does scale.
  const camPos: [number, number, number] = [tile.sx * 0.7, tile.sx * 0.9, tile.sz * 0.7];
  const span = Math.max(tile.sx, tile.sz);

  return (
    <div
      className="map3dCanvas"
      onWheel={onWheel}
      onPointerLeave={() => onHover(null)}
      style={{ width: "100%", height: "100%" }}
    >
      <Canvas
        orthographic
        dpr={[1, 2]}
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        gl={makeRenderer as any}
        flat
        style={{ background: "#0e1410" }}
      >
        <OrthographicCamera
          makeDefault
          position={camPos}
          zoom={cameraZoom}
          near={-span * 4}
          far={span * 4}
          left={-span * 0.6}
          right={span * 0.6}
          top={span * 0.6}
          bottom={-span * 0.6}
        />
        <CameraZoomSync cameraZoom={cameraZoom} />
        <MapControls
          enableRotate={false}
          enableZoom={false}
          screenSpacePanning
          target={[0, 0, 0]}
        />
        <ambientLight intensity={1.0} />
        <VoxelColumns tile={tile} heights={heights} yLevel={yLevel} onHover={onHover} />
        <YPlaneIndicator tile={tile} yLevel={yLevel} show={heights != null} />
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
