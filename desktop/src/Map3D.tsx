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

import { Canvas, useThree, type ThreeEvent } from "@react-three/fiber";
import { Html, MapControls, OrthographicCamera } from "@react-three/drei";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

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
  /** Wheel delta callback (4-block step per notch; Shift = ×4 / 16 blocks). */
  onYDelta: (delta: number) => void;
  /** Pointer hover callback. Fires when the hovered cell changes; null when
   *  the pointer leaves the voxel mesh. */
  onHover: (info: HoverInfo | null) => void;
};

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
 */
const RELIEF = 2.0;
const MIN_H = 0.6;

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

  // (Re)write per-instance colours whenever tile.bytes change. Each wheel
  // notch produces a fresh `tile` object with new bytes; this rewrites the
  // 4 bytes/cell into normalised RGB on the instanceColor buffer.
  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    for (let i = 0; i < count; i++) {
      const off = i * 4;
      tmpColor.setRGB(
        tile.bytes[off] / 255,
        tile.bytes[off + 1] / 255,
        tile.bytes[off + 2] / 255,
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
            title={`${p.structure.replace(/_/g, " ")} @ (${p.block_x}, ${p.block_z})`}
          />
        </Html>
      ))}
    </>
  );
}

export function Map3D(props: Map3DProps) {
  const { tile, heights, pins, cameraZoom, yLevel, onYDelta, onHover } = props;

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
        gl={{ antialias: false, alpha: false }}
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
        <PinOverlay tile={tile} pins={pins} />
      </Canvas>
    </div>
  );
}
