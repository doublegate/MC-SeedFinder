/**
 * 3D isometric biome map (Phase 7+ — PR 2/3).
 *
 * Renders the same biome tile bytes the 2D `<TileCanvas>` uses, but as a
 * `THREE.DataTexture` painted onto a `PlaneGeometry` whose per-vertex Y is
 * displaced by cubiomes' **approximate** surface heightmap. Pan = left-drag
 * (drei `<MapControls>`); zoom = buttons (orthographic `camera.zoom`); the
 * mouse wheel scrubs the Y biome-sample level (`onYDelta`).
 *
 * Accuracy: the biome texture is bit-exact via cubiomes (PR 1's
 * `render_tile_rgba_at_y`). The heightmap is approximate — the
 * "approximate terrain" badge in the HUD makes that visible to the user.
 */

import { Canvas, useThree } from "@react-three/fiber";
import { Html, MapControls, OrthographicCamera } from "@react-three/drei";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

export type Map3DProps = {
  tile: {
    bytes: number[];
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
  /** Wheel delta callback (one Δy block per notch; Shift = ×16). */
  onYDelta: (delta: number) => void;
};

/** Imperative helpers — the inner scene needs access to the camera and
 *  geometry to apply zoom changes and rebuild on (sx,sz) changes. */
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

/** Ground mesh: a unit `PlaneGeometry(sx-1, sz-1)` subdivisions rotated to
 *  lie flat on the XZ plane, textured with the biome RGBA bytes, and
 *  per-vertex Y displaced by the heightmap (when available). */
function GroundMesh({
  tile,
  heights,
}: {
  tile: Map3DProps["tile"];
  heights: number[] | null;
}) {
  // DataTexture lives across renders as long as (sx, sz, bytes) match. We
  // recreate when sx/sz change (different tile dimensions); otherwise we
  // mutate the buffer in place via `texture.image.data.set(...)` and flip
  // `needsUpdate`. For now we recreate per change — cheap at 256x256 and
  // keeps the React lifecycle simple.
  const texture = useMemo(() => {
    const data = new Uint8Array(tile.bytes);
    const t = new THREE.DataTexture(data, tile.sx, tile.sz, THREE.RGBAFormat);
    t.minFilter = THREE.NearestFilter;
    t.magFilter = THREE.NearestFilter;
    t.flipY = false; // tile bytes are top-left origin like ImageData
    t.needsUpdate = true;
    return t;
  }, [tile.bytes, tile.sx, tile.sz]);

  useEffect(() => () => texture.dispose(), [texture]);

  // PlaneGeometry with (sx-1, sz-1) segments produces sx*sz vertices on the
  // XZ grid. Index (j, i) → vertex (j*sx + i). Three.js's PlaneGeometry is
  // initially in the XY plane; we rotate it -π/2 around X so it sits flat.
  const geometry = useMemo(() => {
    const g = new THREE.PlaneGeometry(tile.sx, tile.sz, tile.sx - 1, tile.sz - 1);
    g.rotateX(-Math.PI / 2);
    return g;
  }, [tile.sx, tile.sz]);

  useEffect(() => () => geometry.dispose(), [geometry]);

  // Apply heightmap to position.y when available. heights[j*sx + i] is in
  // (fractional) block Y; the plane is currently in scale-grid units, so
  // we divide by the tile scale to keep the height visually proportional
  // to the horizontal span (1 scale-grid step = `scale` blocks). For
  // 1:4 that's a /4 — a 200-block-high mountain at scale 4 shows as 50
  // grid units tall. Tunable scalar below ("relief") scales the result.
  // RELIEF=2.0 makes a 200-block mountain visibly tower at ~100 grid units
  // over a 192-unit-wide tile — exaggerated but unmistakeably 3D.
  const RELIEF = 2.0;
  useEffect(() => {
    const pos = geometry.attributes.position as THREE.BufferAttribute;
    const arr = pos.array as Float32Array;
    const sx = tile.sx;
    const sz = tile.sz;
    if (heights && heights.length === sx * sz) {
      for (let j = 0; j < sz; j++) {
        for (let i = 0; i < sx; i++) {
          const idx = (j * sx + i) * 3;
          // PlaneGeometry after rotateX puts vertices as (x, y, z) where
          // y is the up axis. position.y is what we want to displace.
          arr[idx + 1] = (heights[j * sx + i] / tile.scale) * RELIEF;
        }
      }
    } else {
      // Flat — no heightmap available, fall back to a single plane.
      for (let i = 1; i < arr.length; i += 3) arr[i] = 0;
    }
    pos.needsUpdate = true;
    geometry.computeVertexNormals();
  }, [geometry, heights, tile.scale, tile.sx, tile.sz]);

  return (
    <mesh geometry={geometry}>
      <meshBasicMaterial map={texture} toneMapped={false} />
    </mesh>
  );
}

/** drei `<Html transform={false}>` markers for each structure pin. World
 *  coordinates are block-local; the tile mesh is centred at the origin and
 *  has integer-grid extents `(sx, sz)`, so we offset pins by the tile's
 *  top-left block coord. */
function PinOverlay({ tile, pins }: { tile: Map3DProps["tile"]; pins: Map3DProps["pins"] }) {
  // The mesh spans world coords [-sx/2, +sx/2] × [-sz/2, +sz/2] (after
  // PlaneGeometry centring). The pin's block coord falls at
  // (blockX - (tile.x + tile.sx*tile.scale/2)) / tile.scale in the same
  // unit space.
  const halfX = (tile.sx * tile.scale) / 2;
  const halfZ = (tile.sz * tile.scale) / 2;
  const centreX = tile.x + halfX;
  const centreZ = tile.z + halfZ;

  return (
    <>
      {/* Spawn marker at world (0, 0). Only rendered if it's in the tile span. */}
      {Math.abs(0 - centreX) <= halfX && Math.abs(0 - centreZ) <= halfZ && (
        <Html
          transform={false}
          center
          position={[(0 - centreX) / tile.scale, 0.5, (0 - centreZ) / tile.scale]}
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
            0.5,
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
  const { tile, heights, pins, cameraZoom, onYDelta } = props;
  const wrapperRef = useRef<HTMLDivElement | null>(null);

  // Vanilla wheel handler — bypasses MapControls (which has wheel-zoom off).
  // Step size = 4 blocks (one scale-Y unit at cubScale=4). cubiomes' Range.y
  // is in scale-relative units, so adjacent block-Y values < 4 apart all
  // map to the same scale-Y → identical biome bytes. Stepping by 4 makes
  // every wheel notch produce a different cubiomes scale-Y. Shift = ×4
  // (= 16-block / chunk-section step) for fast Y traversal.
  const onWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    e.preventDefault();
    const sign = e.deltaY > 0 ? -1 : 1;
    const mag = e.shiftKey ? 16 : 4;
    onYDelta(sign * mag);
  };

  // Initial camera position for an isometric look: hover above the plane,
  // rotated 45° around Y, tilted down ~30°. Orthographic projection means
  // distance doesn't change apparent size; `zoom` does.
  const camPos: [number, number, number] = [tile.sx * 0.6, tile.sx * 0.8, tile.sz * 0.6];
  // Frustum derived from the tile span — keeps the entire mesh visible at
  // zoom=1 regardless of pane aspect.
  const span = Math.max(tile.sx, tile.sz);

  return (
    <div
      ref={wrapperRef}
      className="map3dCanvas"
      onWheel={onWheel}
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
        <GroundMesh tile={tile} heights={heights} />
        <PinOverlay tile={tile} pins={pins} />
      </Canvas>
    </div>
  );
}
