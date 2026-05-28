import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import {
  ConditionBuilder,
  defaultRoot,
  nodeToWire,
  PRESET_TEMPLATES,
  wireToNode,
  type TreeNode,
} from "./conditions";
import { Map3D, hasWebGPU, type HoverInfo } from "./Map3D";
import { biomeLabel } from "./biomes";
import "./styles.css";

// Cached at module-init so we don't repeatedly check navigator.gpu.
const WEBGPU_AVAILABLE = hasWebGPU();

// ---------------------------------------------------------------------------
// Types mirroring the Tauri command surface
// ---------------------------------------------------------------------------

type SearchResult = {
  seed: number;
  edition: string;
  version: string;
  dimension: string;
  score: number;
  matched_features: string[];
  exactness: Record<string, string>;
  warnings: string[];
};

type JobState = {
  jobId: string;
  status: string; // idle | running | completed | cancelled | error
  scanned: number;
  matches: number;
};

type Analysis = {
  seed: number;
  version: string;
  dimension: string;
  origin_biome_id: number;
  origin_biome_exact: boolean;
  nearest_village: { block_x: number; block_z: number } | null;
  strongholds: { block_x: number; block_z: number }[];
};

// Raw RGBA tile from `render_tile_rgba_cmd`. `bytes` is sx*sz*4 (one byte per
// channel, row-major, top-left origin). The frontend blits it via a Canvas2D
// `putImageData`, skipping PNG encode + base64 + browser decode (~30-45 ms
// saved per tile vs the legacy PNG path).
type TileResponse = {
  bytes: number[];
  /** Per-cell cubiomes biome IDs (u8 stored as JS numbers), row-major,
   *  same order as `bytes`. Used by the 3D voxel renderer and the cursor
   *  biome-name readout. Optional for back-compat with older backends. */
  biome_ids?: number[];
  seed: number;
  scale: number;
  x: number;
  z: number;
  sx: number;
  sz: number;
  /** Minecraft block Y at which the biomes were sampled. Optional for
   *  back-compat with older backends that don't echo it. */
  y?: number;
};

/** Response from `surface_height_tile_cmd` — per-pixel approximate surface
 *  block Y (cubiomes `mapApproxHeight`, NOT bit-exact Java terrain) plus
 *  the biome IDs at those surface points. Always carries
 *  `exactness: "approximate"` so the UI can label the heightmap. */
type HeightTileResponse = {
  heights: number[];
  biome_ids: number[];
  exactness: string;
  seed: number;
  scale: number;
  x: number;
  z: number;
  sx: number;
  sz: number;
};

type StructurePin = {
  structure: string;
  block_x: number;
  block_z: number;
};

// ---------------------------------------------------------------------------
// Map constants
// ---------------------------------------------------------------------------

// Tiles used to be a fixed 256x256. They now adapt to the map pane's actual
// pixel size (cap below) so the biome map fills the available area without
// letterboxing and reflows on window resize. The cap bounds cubiomes work +
// PNG encoding per tile; the CSS then upscales to fill larger panes.
const TILE_MAX_PX = 1024;
const TILE_MIN_PX = 64;

// Over-render the tile slightly past the visible pane on each side. Lets a
// drag-pan reveal already-rendered map content instead of black margins;
// refetches with a new centre only on pointer-up. 1.3 = 15% over-render per
// side, tile area = 1.69× visible pane. Dropped from 1.5 (which made each
// tile 2.25× — noticeably slower to render on a per-tile basis) once the
// per-call cubiomes setupGenerator cost was amortised via the BiomePool on
// the Tauri side; the smaller overscan is now sufficient because tiles
// render in ~50 ms on warm cache instead of 200+ ms.
const OVERSCAN = 1.3;
// Inset of the over-rendered tile so its centre sits at the pane centre.
const OVERSCAN_INSET_PCT = (1 - OVERSCAN) * 50;

/** Pick tile dimensions that match the pane's aspect ratio, capped at TILE_MAX_PX
 *  along the longer axis. Caller multiplies by `scale` to get block coverage. */
function tileSizeForPane(w: number, h: number): { sx: number; sz: number } {
  const safeW = Math.max(TILE_MIN_PX, Math.round(w));
  const safeH = Math.max(TILE_MIN_PX, Math.round(h));
  const aspect = safeW / safeH;
  if (aspect >= 1) {
    const sx = Math.min(TILE_MAX_PX, safeW);
    return { sx, sz: Math.max(TILE_MIN_PX, Math.round(sx / aspect)) };
  }
  const sz = Math.min(TILE_MAX_PX, safeH);
  return { sx: Math.max(TILE_MIN_PX, Math.round(sz * aspect)), sz };
}
// cubiomes' supported scales: 1, 4, 16, 64, 256. Lower index = closer zoom.
const SCALE_LEVELS = [1, 4, 16, 64, 256] as const;
const DEFAULT_SCALE = 4;
const MIN_ZOOM = 0.05;
const MAX_ZOOM = 8;

/** Pick the cubiomes scale closest to "ideal" in log space.
 *  ideal = DEFAULT_SCALE / zoomLevel — the cubiomesScale that lets cssScale ≈ 1. */
function pickScale(zoomLevel: number): number {
  const ideal = DEFAULT_SCALE / zoomLevel;
  for (let i = 0; i < SCALE_LEVELS.length; i++) {
    const s = SCALE_LEVELS[i];
    if (s >= ideal) {
      if (i === 0) return s;
      const lower = SCALE_LEVELS[i - 1];
      // Compare in log space: pick whichever is closer to `ideal`.
      const logMid = Math.sqrt(lower * s);
      return ideal < logMid ? lower : s;
    }
  }
  return SCALE_LEVELS[SCALE_LEVELS.length - 1];
}

function clampZoom(z: number): number {
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, z));
}

/**
 * Render a tile's raw RGBA bytes into a `<canvas>` via `putImageData`.
 * Skips the PNG round-trip the legacy `<img src="data:image/png;base64,...">`
 * required (PNG encode ~15-25 ms + base64 ~5 ms + browser decode ~5-10 ms);
 * the canvas absorbs the bytes directly. The canvas's natural size matches
 * the tile's `(sx, sz)`; CSS stretches it like the old `<img>` did.
 */
function TileCanvas({
  tile,
  canvasRef,
}: {
  tile: TileResponse;
  canvasRef?: React.MutableRefObject<HTMLCanvasElement | null>;
}) {
  const localRef = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const canvas = localRef.current;
    if (!canvas) return;
    if (canvasRef) canvasRef.current = canvas;
    canvas.width = tile.sx;
    canvas.height = tile.sz;
    const ctx = canvas.getContext("2d", { willReadFrequently: false });
    if (!ctx) return;
    // tile.bytes arrives as number[] over JSON IPC — wrap as Uint8ClampedArray
    // (one allocation, no copy). ImageData expects RGBA in row-major order,
    // top-left origin — exactly what BiomeBackend::render_tile_rgba produces.
    const clamped = new Uint8ClampedArray(tile.bytes);
    const img = new ImageData(clamped, tile.sx, tile.sz);
    ctx.putImageData(img, 0, 0);
  }, [tile, canvasRef]);
  return (
    <canvas
      ref={localRef}
      className="tileImage"
      aria-label={`Biome tile for seed ${tile.seed}`}
      style={{ imageRendering: "pixelated" }}
    />
  );
}

/** Tauri commands bail out with "superseded" when a newer request of the
 *  same kind invalidates their work. Treat that as a benign signal, not an
 *  error to surface in the UI. */
function isSupersededError(e: unknown): boolean {
  const s = String(e);
  return s === "superseded" || s.endsWith(": superseded") || s.includes('"superseded"');
}

// ---------------------------------------------------------------------------
// Tile LRU cache helpers
// ---------------------------------------------------------------------------

function tileKey(
  seed: number,
  version: string,
  dimension: string,
  x: number,
  z: number,
  sx: number,
  sz: number,
  scale: number,
  y: number,
): string {
  return `${seed}|${version}|${dimension}|${x},${z}|${sx}x${sz}@${scale}|y${y}`;
}

function heightKey(
  seed: number,
  version: string,
  dimension: string,
  x: number,
  z: number,
  sx: number,
  sz: number,
): string {
  // Heightmaps are seed/dimension-dependent but NOT y-dependent — there's
  // one surface per (seed, dim, x, z). Separate key from biome tiles so
  // they share the cache cleanly.
  return `h:${seed}|${version}|${dimension}|${x},${z}|${sx}x${sz}`;
}

/** Minecraft 1.18+ build range. The wheel-driven Y scrubber in the 3D
 *  isometric view is clamped to this; default starting Y is sea level. */
const Y_MIN = -64;
const Y_MAX = 319;
const Y_DEFAULT = 63;

function clampY(y: number): number {
  return Math.max(Y_MIN, Math.min(Y_MAX, y | 0));
}

/** Cosmetic depth-band label next to the Y readout. The biome at the Y
 *  is the source of truth — these names are just visual orientation. */
function yBandLabel(y: number): string {
  if (y >= 192) return "(mountain)";
  if (y >= 96) return "(highlands)";
  if (y === 63) return "(sea level)";
  if (y >= 0) return "(surface)";
  if (y >= -32) return "(caves)";
  return "(deepslate)";
}

function getCachedTile(cache: Map<string, TileResponse>, key: string): TileResponse | undefined {
  const value = cache.get(key);
  if (value !== undefined) {
    // Touch — reinsert at end (Map iteration order = insertion order, which
    // doubles as a tiny LRU without an extra data structure).
    cache.delete(key);
    cache.set(key, value);
  }
  return value;
}

function putCachedTile(
  cache: Map<string, TileResponse>,
  key: string,
  value: TileResponse,
  capacity: number,
): void {
  cache.delete(key);
  cache.set(key, value);
  while (cache.size > capacity) {
    const oldest = cache.keys().next().value;
    if (oldest === undefined) break;
    cache.delete(oldest);
  }
}
// Cache type for the result-list thumbnails. Keyed on
// `${seed}|${version}|${dimension}`. Values are the full TileResponse
// so we can re-blit if the canvas re-mounts (e.g. result list scrolls).
type ThumbCache = Map<string, TileResponse>;
const THUMB_CACHE_MAX = 96;

const THUMB_SX = 36;
const THUMB_SZ = 36;
// scale=16 → each thumbnail covers 36*16 = 576 blocks across, centred
// near origin. Fast cubiomes scale that gives a recognisable terrain
// silhouette without burning backend budget.
const THUMB_SCALE = 16;
const THUMB_X = -(THUMB_SX * THUMB_SCALE) / 2;
const THUMB_Z = -(THUMB_SZ * THUMB_SCALE) / 2;

/**
 * Small biome preview canvas rendered inline with each search result —
 * the side-by-side seed compare feature (brainstorm #6) at MVP scope.
 * Lets the user scan the result list visually and notice that "seed
 * 12345 has a mountain near origin" vs "seed 67890 is mostly ocean"
 * without having to click each one to load the full map.
 *
 * Implementation notes:
 *  - Cache is shared across all thumbnails via a ref-passed Map so a
 *    re-render of the list (selectedSeed change, scroll) doesn't refetch.
 *  - Fetches on mount; aborts via the cancelled flag if unmounted before
 *    the response arrives. Errors silently noop — the canvas just stays
 *    empty.
 *  - 36×36 at scale 1:16 is the sweet spot: enough resolution to see
 *    biome patches, fast enough that 25 thumbnails finish in <100ms with
 *    the warm BiomePool.
 */
function ResultThumbnail({
  seed,
  version,
  dimension,
  cacheRef,
}: {
  seed: number;
  version: string;
  dimension: string;
  cacheRef: React.MutableRefObject<ThumbCache>;
}) {
  const [bytes, setBytes] = useState<number[] | null>(() => {
    const cached = cacheRef.current.get(`${seed}|${version}|${dimension}`);
    return cached ? cached.bytes : null;
  });
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const key = `${seed}|${version}|${dimension}`;
    const cached = cacheRef.current.get(key);
    if (cached) {
      setBytes(cached.bytes);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const t = await invoke<TileResponse>("render_tile_rgba_cmd", {
          request: {
            seed,
            version,
            dimension,
            x: THUMB_X,
            z: THUMB_Z,
            scale: THUMB_SCALE,
            sx: THUMB_SX,
            sz: THUMB_SZ,
          },
        });
        if (cancelled) return;
        cacheRef.current.set(key, t);
        // LRU cap — drop oldest entries first (Map preserves insertion order).
        while (cacheRef.current.size > THUMB_CACHE_MAX) {
          const oldest = cacheRef.current.keys().next().value;
          if (oldest === undefined) break;
          cacheRef.current.delete(oldest);
        }
        setBytes(t.bytes);
      } catch {
        /* superseded or backend error — leave the canvas empty */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [seed, version, dimension, cacheRef]);

  useEffect(() => {
    if (!bytes) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { willReadFrequently: false });
    if (!ctx) return;
    const clamped = new Uint8ClampedArray(bytes);
    const img = new ImageData(clamped, THUMB_SX, THUMB_SZ);
    ctx.putImageData(img, 0, 0);
  }, [bytes]);

  return (
    <canvas
      ref={canvasRef}
      width={THUMB_SX}
      height={THUMB_SZ}
      className={`resultThumb ${bytes ? "" : "loading"}`}
      title={`Biome preview at 1:${THUMB_SCALE} around origin for seed ${seed}`}
      aria-hidden
    />
  );
}

// Structures shown as pins on the map (and queried in batch from the backend).
const PIN_STRUCTURES = [
  "village",
  "pillager_outpost",
  "ocean_monument",
  "woodland_mansion",
  "stronghold",
];

// ---------------------------------------------------------------------------

function App() {
  const [edition, setEdition] = useState("java");
  const [version, setVersion] = useState("1.21");
  const [count, setCount] = useState(100000);
  const [maxMatches, setMaxMatches] = useState(25);
  const [conditionTree, setConditionTree] = useState<TreeNode>(() => defaultRoot());
  const [job, setJob] = useState<JobState>({
    jobId: "",
    status: "idle",
    scanned: 0,
    matches: 0,
  });
  const [results, setResults] = useState<SearchResult[]>([]);
  const [selectedSeed, setSelectedSeed] = useState<number | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [tile, setTile] = useState<TileResponse | null>(null);
  const [heightTile, setHeightTile] = useState<HeightTileResponse | null>(null);
  const [pins, setPins] = useState<StructurePin[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Map view state. `zoomLevel` is a continuous float; `pickScale(zoomLevel)`
  // chooses the cubiomes discrete scale to render with, and CSS scales the
  // tile by the residual ratio so pinch/wheel feels smooth without forcing a
  // cubiomes re-render at every micro-tick. zoomLevel = 1.0 → DEFAULT_SCALE.
  const [viewCenter, setViewCenter] = useState({ x: 0, z: 0 });
  const [zoomLevel, setZoomLevel] = useState(1.0);
  // The cubiomes scale derived from zoomLevel. Memoized so the tile fetch
  // effect doesn't re-run on every continuous-zoom tick — only when the
  // chosen discrete scale actually changes.
  const cubScale = useMemo(() => pickScale(zoomLevel), [zoomLevel]);

  // Pane dimensions in CSS pixels — drives the tile request size so the
  // rendered biome map fills the available area and reflows on resize.
  const [paneSize, setPaneSize] = useState({ w: 800, h: 600 });

  // In-memory LRU cache of rendered biome tiles, keyed by the full request
  // signature. Hits are served instantly so panning to recently-visited
  // areas (or to a neighbor we prefetched) doesn't blink to the loading
  // state. CSS transforms already run on the GPU compositor; the missing
  // piece for fluid pan was tile *availability*, which this fixes.
  const tileCacheRef = useRef<Map<string, TileResponse>>(new Map());
  // Bumped from 32 → 96 in PR 2 (3D view) — the working set grows from
  // (zoom × pan) to (zoom × pan × Y slice) once the wheel scrubs Y.
  const TILE_CACHE_MAX = 96;
  // Separate cache for the inline result-list thumbnails (#6). Keeps the
  // 36×36 previews from churning the main-map LRU.
  const thumbCacheRef = useRef<ThumbCache>(new Map());
  // Heightmaps are smaller per-tile (single f32 + i32 vs RGBA) and don't
  // multiply with Y, so a tighter cache is fine.
  const heightCacheRef = useRef<Map<string, HeightTileResponse>>(new Map());
  const HEIGHT_CACHE_MAX = 32;

  // 3D isometric map view + the Y the wheel scrubs to. Behind a toggle
  // during PR 2 (default 2D); flipped to 3D-default in PR 3 once the
  // 3D path proves out across Tauri WebView backends.
  // PR 3 (Phase 7+): flipped default to "3D" once the voxel pipeline +
  // overlays proved stable across WebKitGTK / WebView2 / WKWebView.
  // 2D mode (TileCanvas) remains accessible via the in-pane toggle for
  // any user who prefers the flat view or hits a WebGL2 init failure.
  const [mapView, setMapView] = useState<"2D" | "3D">("3D");
  const [yLevel, setYLevel] = useState<number>(Y_DEFAULT);
  // Per-cell biome readout under the cursor in 3D mode. null when not
  // hovering a column. Populated by Map3D's raycaster-driven onHover.
  const [hover, setHover] = useState<HoverInfo | null>(null);
  // Dimension tabs (Overworld / Nether / End). All three are supported
  // by cubiomes; tile rendering threads the value through directly.
  // The Nether/End heightmaps behave differently (Nether is constant Y=127,
  // End uses getEndSurfaceHeight) — the existing Tauri command handles both.
  type Dimension = "overworld" | "nether" | "end";
  const [dimension, setDimension] = useState<Dimension>("overworld");
  // Overlay toggles (slime chunks, world border). Slime chunks are
  // computed via the new list_slime_chunks_cmd Tauri command. World
  // border is the static ±29,999,984 block rectangle.
  const [overlays, setOverlays] = useState({
    slime: false,
    border: false,
    rings: false,
    spawn: true, // spawn star + chunks ON by default; visually subtle
    npDebug: false,
  });
  // Cached climate np[6] for the most recently rested hover. Cleared on
  // dimension/seed change. Six i64 values cubiomes computes at the
  // hovered cell — temperature, humidity, continentalness, erosion,
  // depth, weirdness — useful for power users.
  const [climateNp, setClimateNp] = useState<number[] | null>(null);
  const [slimeChunks, setSlimeChunks] = useState<number[][]>([]);
  // World spawn (x, z) for the selected seed. Fetched lazily on seed/
  // dimension change via world_spawn_cmd. Rendered as a star pin in
  // 3D mode (and as a label-anchored marker in 2D — already shows 0,0).
  const [worldSpawn, setWorldSpawn] = useState<{ x: number; z: number } | null>(null);

  // Ref to the on-screen canvas so `downloadTile` can pull the rendered PNG
  // from it (the canvas owns the rendered pixels; we don't ship a separate
  // PNG over IPC).
  const tileCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Active-drag pixel offset (CSS translate). Committed to viewCenter on
  // pointer-up, then the tile/pin fetch effect re-fires.
  const [drag, setDrag] = useState<{ dx: number; dz: number } | null>(null);
  const dragStartRef = useRef<{ x: number; y: number } | null>(null);
  const mapPaneRef = useRef<HTMLDivElement | null>(null);

  // Stale-event filter for streamed search.
  // The ref holds one of:
  //   ""    — no active search (drop any incoming event)
  //   "*"   — search just kicked off, real jobId not yet known (accept anything)
  //   <id>  — accept only events tagged with this job id
  // The "*" sentinel exists because a structure-only search of a few hundred
  // seeds can complete BEFORE invoke("start_search") resolves with the real
  // jobId — without the sentinel, every event would arrive while the ref is
  // still "" and get filtered out.
  const activeJobIdRef = useRef<string>("");

  function jobMatches(eventJobId: string): boolean {
    const active = activeJobIdRef.current;
    return active !== "" && (active === "*" || active === eventJobId);
  }

  const spec = useMemo(
    () => ({
      edition,
      version,
      dimension: "overworld",
      start_seed: 0,
      count,
      max_matches: maxMatches,
      criteria: {
        conditions: nodeToWire(conditionTree),
      },
    }),
    [count, maxMatches, edition, version, conditionTree],
  );

  // ---- Streamed search events ----
  useEffect(() => {
    let unlistenFns: UnlistenFn[] = [];
    (async () => {
      unlistenFns.push(
        await listen<SearchResult>("search-match", (event) => {
          if (!activeJobIdRef.current) return;
          setResults((prev) => [...prev, event.payload]);
          setJob((j) => ({ ...j, matches: j.matches + 1 }));
          setSelectedSeed((s) => (s == null ? event.payload.seed : s));
        }),
        await listen<{ job_id: string; scanned: number; matches: number }>(
          "search-progress",
          (event) => {
            if (!jobMatches(event.payload.job_id)) return;
            setJob((j) => ({
              ...j,
              scanned: event.payload.scanned,
              matches: event.payload.matches,
            }));
          },
        ),
        await listen<{
          job_id: string;
          scanned: number;
          matches: number;
          reason: string;
          error: string | null;
        }>("search-completed", (event) => {
          if (!jobMatches(event.payload.job_id)) return;
          setJob((j) => ({
            ...j,
            status:
              event.payload.reason === "cancelled" ? "cancelled" : "completed",
            scanned: event.payload.scanned,
            matches: event.payload.matches,
          }));
          if (event.payload.error) setError(event.payload.error);
          // Job is over — drop the active flag so any straggler events from
          // this job (or events emitted before a new Run begins) are ignored.
          activeJobIdRef.current = "";
        }),
      );
    })();
    return () => {
      unlistenFns.forEach((fn) => fn());
    };
  }, []);

  async function startSearch() {
    setError(null);
    setResults([]);
    setSelectedSeed(null);
    setAnalysis(null);
    setTile(null);
    setPins([]);
    setJob({ jobId: "", status: "running", scanned: 0, matches: 0 });
    // Request notification permission on first search (user-initiated, per
    // the spec's "user gesture" requirement). Stash the result so the
    // search-match listener can decide whether to fire toasts.
    if (
      typeof window.Notification !== "undefined" &&
      notifyPermissionRef.current === "unknown"
    ) {
      try {
        const perm =
          Notification.permission === "default"
            ? await Notification.requestPermission()
            : Notification.permission;
        notifyPermissionRef.current = perm;
      } catch {
        notifyPermissionRef.current = "denied";
      }
    }
    // CRITICAL: set the wildcard BEFORE invoke so events that fire during
    // the IPC round-trip (instant for tiny structure-only searches) aren't
    // dropped by the activeJobIdRef filter.
    activeJobIdRef.current = "*";
    try {
      const jobId = await invoke<string>("start_search", { spec });
      activeJobIdRef.current = jobId;
      setJob((j) => ({ ...j, jobId }));
    } catch (e) {
      activeJobIdRef.current = "";
      setError(String(e));
      setJob((j) => ({ ...j, status: "error" }));
    }
  }

  async function cancelSearch() {
    if (!job.jobId) return;
    await invoke("cancel_search", { jobId: job.jobId }).catch(() => undefined);
    setJob((j) => ({ ...j, status: "cancelling" }));
  }

  async function analyzeSelected() {
    if (selectedSeed == null) return;
    try {
      const a = await invoke<Analysis>("analyze_seed", {
        seed: selectedSeed,
        version,
        dimension: "overworld",
      });
      setAnalysis(a);
    } catch (e) {
      if (isSupersededError(e)) return;
      setError(String(e));
    }
  }

  // Build / consume a share blob: base64-encoded JSON with the current spec
  // and (if a seed is selected) the map view. Lets users copy a setup and
  // recreate it later — the desktop equivalent of a deep link.
  function buildShareBlob(): string {
    const payload = {
      v: 2,
      spec: {
        edition,
        version,
        count,
        max_matches: maxMatches,
        criteria: { conditions: nodeToWire(conditionTree) },
      },
      view:
        selectedSeed != null
          ? {
              seed: selectedSeed,
              x: viewCenter.x,
              z: viewCenter.z,
              zoom: zoomLevel,
              // v2 additions — share-link now restores the full 3D view
              // state, including Y level, dimension, mapView, and
              // overlay toggles.
              y: yLevel,
              mapView,
              dimension,
              overlays,
            }
          : null,
    };
    // btoa-safe UTF-8 round-trip.
    return btoa(unescape(encodeURIComponent(JSON.stringify(payload))));
  }

  async function copyShareBlob() {
    try {
      await navigator.clipboard.writeText(buildShareBlob());
    } catch (e) {
      setError(`Copy failed: ${e}`);
    }
  }

  function applyShareBlob(blob: string) {
    setError(null);
    try {
      const payload = JSON.parse(decodeURIComponent(escape(atob(blob.trim()))));
      if (![1, 2].includes(payload.v) || !payload.spec) {
        throw new Error("not a mc-seed-finder share blob");
      }
      const s = payload.spec;
      if (typeof s.edition === "string") setEdition(s.edition);
      if (typeof s.version === "string") setVersion(s.version);
      if (typeof s.count === "number") setCount(s.count);
      if (typeof s.max_matches === "number") setMaxMatches(s.max_matches);
      if (s.criteria?.conditions) {
        setConditionTree(wireToNode(s.criteria.conditions));
      }
      const v = payload.view;
      if (v && typeof v.seed === "number") {
        setSelectedSeed(v.seed);
        if (typeof v.x === "number" && typeof v.z === "number") {
          // Defer view-center update so the seed-change reset effect doesn't
          // immediately clobber it.
          setTimeout(() => {
            setViewCenter({ x: v.x, z: v.z });
            if (typeof v.zoom === "number") setZoomLevel(clampZoom(v.zoom));
            // v2: restore the 3D view state if present.
            if (typeof v.y === "number") setYLevel(clampY(v.y));
            if (v.mapView === "2D" || v.mapView === "3D") setMapView(v.mapView);
            if (
              v.dimension === "overworld" ||
              v.dimension === "nether" ||
              v.dimension === "end"
            ) {
              setDimension(v.dimension);
            }
            if (v.overlays && typeof v.overlays === "object") {
              setOverlays({
                slime: !!v.overlays.slime,
                border: !!v.overlays.border,
                rings: !!v.overlays.rings,
                spawn: v.overlays.spawn !== false, // default true
                npDebug: !!v.overlays.npDebug,
              });
            }
          }, 0);
        }
      }
    } catch (e) {
      setError(`Import share failed: ${e}`);
    }
  }

  function downloadTile() {
    if (!tile) return;
    // The canvas owns the rendered pixels — Canvas2D.toDataURL gives us a
    // PNG without a round-trip back to Rust. Falls back silently if the
    // canvas isn't mounted yet.
    const canvas = tileCanvasRef.current;
    if (!canvas) return;
    const a = document.createElement("a");
    a.href = canvas.toDataURL("image/png");
    a.download = `seed-${tile.seed}-x${viewCenter.x}-z${viewCenter.z}-zoom${zoomLevel.toFixed(2)}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  // Import seed from a Minecraft world's level.dat. The browser file input
  // gives us a File whose bytes we ship to Rust (gzipped NBT) — Rust returns
  // the seed (and friendly labels), and we treat it like any other selected
  // result so the map renders immediately.
  async function importLevelDat(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setError(null);
    try {
      const buf = await file.arrayBuffer();
      const bytes = Array.from(new Uint8Array(buf));
      const res = await invoke<{
        seed: number;
        version_name: string | null;
        level_name: string | null;
      }>("import_level_dat", { bytes });
      // Pre-populate as if the user picked it from results.
      const importedResult: SearchResult = {
        seed: res.seed,
        edition: "java",
        version: res.version_name ?? version,
        dimension: "overworld",
        score: 0,
        matched_features: ["imported_from_level_dat"],
        exactness: { structures: "exact" },
        warnings: res.level_name ? [`level.dat "${res.level_name}"`] : [],
      };
      setResults((prev) => {
        // Avoid duplicating the same seed if re-imported.
        if (prev.some((r) => r.seed === res.seed)) return prev;
        return [importedResult, ...prev];
      });
      setSelectedSeed(res.seed);
    } catch (err) {
      setError(`Import failed: ${err}`);
    } finally {
      // Reset the input so re-selecting the same file fires onChange again.
      e.target.value = "";
    }
  }

  // Re-centre on origin when a new seed is selected (don't carry pan/zoom).
  useEffect(() => {
    setViewCenter({ x: 0, z: 0 });
    setZoomLevel(1.0);
  }, [selectedSeed]);

  // Observe the map pane's CSS size and republish on resize (debounced so a
  // window-resize drag doesn't fire dozens of tile fetches).
  useEffect(() => {
    const el = mapPaneRef.current;
    if (!el) return;
    let timer: number | null = null;
    const ro = new ResizeObserver((entries) => {
      const entry = entries[entries.length - 1];
      if (!entry) return;
      const w = Math.round(entry.contentRect.width);
      const h = Math.round(entry.contentRect.height);
      if (timer != null) window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        setPaneSize((prev) => (prev.w === w && prev.h === h ? prev : { w, h }));
      }, 150);
    });
    ro.observe(el);
    return () => {
      ro.disconnect();
      if (timer != null) window.clearTimeout(timer);
    };
  }, []);

  // Fetch the biome tile AND structure pins whenever the view changes. Tiles
  // pass through an LRU cache; once a tile is rendered (here or by a prefetch
  // for a neighboring view) panning back to it is instant. Pins are not
  // cached — they're cheap to recompute and we always want them current.
  useEffect(() => {
    if (selectedSeed == null) {
      setTile(null);
      setPins([]);
      return;
    }
    // Rendering happens at the cubiomes scale derived from zoomLevel (memoized
    // above); CSS scale (computed downstream from the rendered tile's actual
    // scale) handles the residual continuous zoom in/out.
    // Request a tile larger than the visible pane (over-render) so a
    // drag-pan reveals already-loaded content instead of black margins.
    const { sx, sz } = tileSizeForPane(
      paneSize.w * OVERSCAN,
      paneSize.h * OVERSCAN,
    );
    const tileSpanX = sx * cubScale;
    const tileSpanZ = sz * cubScale;
    const tileX = viewCenter.x - Math.round(tileSpanX / 2);
    const tileZ = viewCenter.z - Math.round(tileSpanZ / 2);

    let cancelled = false;

    // 2D mode samples biomes at sea level (legacy behaviour, byte-for-byte
    // unchanged); 3D mode samples at the wheel-driven Y. Cache key includes
    // Y in both modes so an A/B toggle never serves a wrong-Y tile.
    const fetchY = mapView === "3D" ? yLevel : Y_DEFAULT;
    const key = tileKey(
      selectedSeed,
      version,
      dimension,
      tileX,
      tileZ,
      sx,
      sz,
      cubScale,
      fetchY,
    );
    const cached = getCachedTile(tileCacheRef.current, key);

    // The 3D view also needs the approximate surface heightmap to extrude
    // the ground plane. Heightmap is Y-independent — one per (seed, dim, x,
    // z, sx, sz) — and only fetched in 3D Overworld (Nether returns Y=127,
    // End uses different math the surface_height_tile_cmd doesn't surface yet).
    const hKey = heightKey(selectedSeed, version, dimension, tileX, tileZ, sx, sz);
    const cachedHeight =
      mapView === "3D" && dimension === "overworld"
        ? heightCacheRef.current.get(hKey)
        : undefined;

    (async () => {
      // Show any cached tile immediately, then refresh pins (which we never
      // cache) alongside a re-confirmation fetch only if the cache missed.
      if (cached) {
        setTile(cached);
      }
      if (cachedHeight) {
        setHeightTile(cachedHeight);
      }

      try {
        const tilePromise = cached
          ? Promise.resolve(cached)
          : invoke<TileResponse>("render_tile_rgba_cmd", {
              request: {
                seed: selectedSeed,
                version,
                dimension,
                x: tileX,
                z: tileZ,
                scale: cubScale,
                sx,
                sz,
                y: fetchY,
              },
            });
        const pinsPromise = invoke<StructurePin[]>("list_structures_in_view", {
          request: {
            seed: selectedSeed,
            structures: PIN_STRUCTURES,
            x: tileX,
            z: tileZ,
            sx: tileSpanX,
            sz: tileSpanZ,
          },
        });
        // Only the 3D path needs the heightmap. cubScale === 4 is the GPU/
        // heightmap-friendly scale; at coarser scales we'd need the bigger
        // surface_height_map (still scale-4 internally) sampled wider, so
        // for now restrict to scale 4 and fall back to a flat plane otherwise.
        const heightPromise: Promise<HeightTileResponse | null> =
          mapView === "3D" && cubScale === 4 && dimension === "overworld" && !cachedHeight
            ? invoke<HeightTileResponse>("surface_height_tile_cmd", {
                request: {
                  seed: selectedSeed,
                  version,
                  dimension,
                  x: tileX,
                  z: tileZ,
                  sx,
                  sz,
                },
              })
            : Promise.resolve(cachedHeight ?? null);
        const [t, ps, h] = await Promise.all([tilePromise, pinsPromise, heightPromise]);
        if (cancelled) return;
        setTile(t);
        setPins(ps);
        putCachedTile(tileCacheRef.current, key, t, TILE_CACHE_MAX);
        if (h) {
          setHeightTile(h);
          // LRU put for heightmap
          heightCacheRef.current.delete(hKey);
          heightCacheRef.current.set(hKey, h);
          while (heightCacheRef.current.size > HEIGHT_CACHE_MAX) {
            const oldest = heightCacheRef.current.keys().next().value;
            if (oldest === undefined) break;
            heightCacheRef.current.delete(oldest);
          }
        } else if (mapView !== "3D") {
          setHeightTile(null);
        }
      } catch (e) {
        if (cancelled) return;
        // Rust "superseded" rejection means a newer request invalidated this
        // one — silently drop, the newer fetch will fill the UI.
        if (isSupersededError(e)) return;
        setError(String(e));
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [
    selectedSeed,
    version,
    dimension,
    viewCenter.x,
    viewCenter.z,
    cubScale,
    paneSize.w,
    paneSize.h,
    mapView,
    yLevel,
  ]);

  // Y-axis prefetch (3D only). After the current tile settles, fire two
  // background fetches at y±4 (the wheel step size). When the user scrolls
  // the wheel a notch in either direction, the next tile is already in
  // cache and the wheel-driven refresh is instant. Debounced 200ms so a
  // rapid scroll doesn't queue noise.
  useEffect(() => {
    if (mapView !== "3D" || selectedSeed == null) return;
    if (cubScale !== 4) return; // mirrors the heightmap path's gate

    const handle = window.setTimeout(() => {
      const { sx, sz } = tileSizeForPane(
        paneSize.w * OVERSCAN,
        paneSize.h * OVERSCAN,
      );
      const tileSpanX = sx * cubScale;
      const tileSpanZ = sz * cubScale;
      const tileX = viewCenter.x - Math.round(tileSpanX / 2);
      const tileZ = viewCenter.z - Math.round(tileSpanZ / 2);

      const prefetchAt = async (targetY: number) => {
        if (targetY < Y_MIN || targetY > Y_MAX) return;
        const key = tileKey(
          selectedSeed,
          version,
          dimension,
          tileX,
          tileZ,
          sx,
          sz,
          cubScale,
          targetY,
        );
        if (tileCacheRef.current.has(key)) return; // already cached
        try {
          const t = await invoke<TileResponse>("render_tile_rgba_cmd", {
            request: {
              seed: selectedSeed,
              version,
              dimension,
              x: tileX,
              z: tileZ,
              scale: cubScale,
              sx,
              sz,
              y: targetY,
            },
          });
          // Cache only — don't update visible state.
          putCachedTile(tileCacheRef.current, key, t, TILE_CACHE_MAX);
        } catch (e) {
          // Silently drop supersession; ignore other errors (best effort).
          if (!isSupersededError(e)) {
            // Don't surface — prefetch is opportunistic.
          }
        }
      };

      // Wheel step size — see Map3D's wheel handler. Mirror it here so
      // a single notch lands in cache. Shift+wheel jumps 16; we don't
      // prefetch those bigger steps (they're explicitly fast-traversal).
      void prefetchAt(yLevel + 4);
      void prefetchAt(yLevel - 4);
    }, 200);

    return () => window.clearTimeout(handle);
  }, [mapView, selectedSeed, version, dimension, viewCenter.x, viewCenter.z, cubScale, paneSize.w, paneSize.h, yLevel]);

  // Slime-chunk fetch (#10). Fires when the slime overlay is toggled on
  // and on any view-rectangle change. Slime is dimension-independent
  // (pure Java RNG of (seed, chunk_x, chunk_z)), so we don't refetch on
  // dimension change. The list goes into local state; Map3D renders it
  // as a green flat overlay over the voxel columns.
  useEffect(() => {
    if (!overlays.slime || selectedSeed == null) {
      setSlimeChunks([]);
      return;
    }
    const { sx, sz } = tileSizeForPane(paneSize.w * OVERSCAN, paneSize.h * OVERSCAN);
    const tileSpanX = sx * cubScale;
    const tileSpanZ = sz * cubScale;
    const tileX = viewCenter.x - Math.round(tileSpanX / 2);
    const tileZ = viewCenter.z - Math.round(tileSpanZ / 2);
    let cancelled = false;
    (async () => {
      try {
        const chunks = await invoke<number[][]>("list_slime_chunks_cmd", {
          request: { seed: selectedSeed, x: tileX, z: tileZ, sx: tileSpanX, sz: tileSpanZ },
        });
        if (!cancelled) setSlimeChunks(chunks);
      } catch (e) {
        if (!cancelled && !isSupersededError(e)) {
          // best-effort; don't surface in UI
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [overlays.slime, selectedSeed, viewCenter.x, viewCenter.z, cubScale, paneSize.w, paneSize.h]);

  // Climate np[6] debug overlay (#11). Debounced: only fires after the
  // cursor rests on the same cell for ~300ms, so dragging the mouse
  // doesn't spam Tauri with single-point cubiomes calls. Disabled
  // entirely unless overlays.npDebug is on.
  useEffect(() => {
    if (!overlays.npDebug || !hover || selectedSeed == null) {
      setClimateNp(null);
      return;
    }
    const handle = window.setTimeout(async () => {
      try {
        const np = await invoke<number[]>("climate_np_cmd", {
          request: {
            seed: selectedSeed,
            version,
            dimension,
            x: hover.worldX,
            y: hover.y,
            z: hover.worldZ,
          },
        });
        setClimateNp(np);
      } catch {
        setClimateNp(null);
      }
    }, 300);
    return () => window.clearTimeout(handle);
  }, [overlays.npDebug, hover, selectedSeed, version, dimension]);

  // World-spawn fetch (#10). Lazy; only when a seed is selected. Backed
  // by cubiomes getSpawn which is fast (single call, no batch).
  useEffect(() => {
    if (selectedSeed == null) {
      setWorldSpawn(null);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const [x, z] = await invoke<[number, number]>("world_spawn_cmd", {
          request: { seed: selectedSeed, version, dimension },
        });
        if (!cancelled) setWorldSpawn({ x, z });
      } catch (_e) {
        if (!cancelled) setWorldSpawn(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedSeed, version, dimension]);

  // Match notification (#19). Show a desktop toast whenever a search match
  // arrives while the window is backgrounded. Permission asked on first
  // user-initiated search (start_search button click) — never autorequested.
  const notifyPermissionRef = useRef<NotificationPermission | "unknown">("unknown");
  useEffect(() => {
    const unlistens: UnlistenFn[] = [];
    (async () => {
      unlistens.push(
        await listen<SearchResult>("search-match", (event) => {
          if (
            typeof window.Notification === "undefined" ||
            notifyPermissionRef.current !== "granted" ||
            !document.hidden
          ) {
            return;
          }
          try {
            new window.Notification(`mc-seed-finder: seed ${event.payload.seed}`, {
              body: `${event.payload.matched_features.length} feature${
                event.payload.matched_features.length === 1 ? "" : "s"
              } matched`,
              silent: false,
            });
          } catch {
            /* notification API quirks; ignore */
          }
        }),
      );
    })();
    return () => {
      unlistens.forEach((u) => u());
    };
  }, []);

  // First-run onboarding (#15). One-time modal explaining the major UI
  // surfaces. Dismissal is persisted to localStorage so it never returns
  // unless the user explicitly clears their data.
  const [showOnboarding, setShowOnboarding] = useState<boolean>(() => {
    try {
      return localStorage.getItem("mcsf.onboarded.v1") !== "1";
    } catch {
      return false;
    }
  });
  function dismissOnboarding() {
    try {
      localStorage.setItem("mcsf.onboarded.v1", "1");
    } catch {
      /* localStorage disabled — accept the dismissal in memory only */
    }
    setShowOnboarding(false);
  }

  // Keyboard shortcuts (#18). Vim-like. j/k = Y -4/+4 (Shift = ×4 = 16);
  // h/l = zoom out/in (one notch); 2/3 = mapView toggle; arrows = pan;
  // r = recenter. Ignored while typing in an input/textarea/contentEditable.
  useEffect(() => {
    function isTyping(target: EventTarget | null): boolean {
      if (!(target instanceof HTMLElement)) return false;
      if (target.isContentEditable) return true;
      const tag = target.tagName;
      return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
    }
    function onKey(e: KeyboardEvent) {
      if (isTyping(e.target)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const big = e.shiftKey;
      // Pan step in BLOCKS — same step as the view-pan integration.
      const panStep = (big ? 4 : 1) * cubScale * 16;
      switch (e.key) {
        case "j":
        case "J":
          setYLevel((y) => clampY(y - (big ? 16 : 4)));
          e.preventDefault();
          return;
        case "k":
        case "K":
          setYLevel((y) => clampY(y + (big ? 16 : 4)));
          e.preventDefault();
          return;
        case "h":
        case "H":
          setZoomLevel((z) => clampZoom(z / 1.4));
          e.preventDefault();
          return;
        case "l":
        case "L":
          setZoomLevel((z) => clampZoom(z * 1.4));
          e.preventDefault();
          return;
        case "2":
          setMapView("2D");
          e.preventDefault();
          return;
        case "3":
          setMapView("3D");
          e.preventDefault();
          return;
        case "r":
        case "R":
          setViewCenter({ x: 0, z: 0 });
          setZoomLevel(1.0);
          e.preventDefault();
          return;
        case "ArrowLeft":
          setViewCenter((v) => ({ ...v, x: v.x - panStep }));
          e.preventDefault();
          return;
        case "ArrowRight":
          setViewCenter((v) => ({ ...v, x: v.x + panStep }));
          e.preventDefault();
          return;
        case "ArrowUp":
          setViewCenter((v) => ({ ...v, z: v.z - panStep }));
          e.preventDefault();
          return;
        case "ArrowDown":
          setViewCenter((v) => ({ ...v, z: v.z + panStep }));
          e.preventDefault();
          return;
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cubScale]);

  // Background prefetch was disabled in Phase 6b. It compounded with rapid
  // pan/click into a Tauri command-pool storm (5× tile fetches per view
  // change), and after the foreground-only path got fast via the BiomePool +
  // smaller OVERSCAN, the prefetch was net-negative for users who pan more
  // than they re-read. The tileCacheRef LRU still serves repeat views.
  // If you want it back: build a debounced setTimeout that fires four
  // neighbor render_tile calls and stuffs them into tileCacheRef via
  // putCachedTile. See the Phase-6a commit history for the original.

  // ---- Map pan handlers ----
  const onPanStart = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (selectedSeed == null) return;
      (e.target as Element).setPointerCapture?.(e.pointerId);
      dragStartRef.current = { x: e.clientX, y: e.clientY };
      setDrag({ dx: 0, dz: 0 });
    },
    [selectedSeed],
  );

  const onPanMove = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragStartRef.current) return;
    setDrag({
      dx: e.clientX - dragStartRef.current.x,
      dz: e.clientY - dragStartRef.current.y,
    });
  }, []);

  const onPanEnd = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!dragStartRef.current || !drag) {
        dragStartRef.current = null;
        setDrag(null);
        return;
      }
      if (drag.dx !== 0 || drag.dz !== 0) {
        // Drag right → reveal LEFT of world → centre.x decreases.
        // At the current view, 1 pane CSS pixel covers (DEFAULT_SCALE/zoomLevel)
        // blocks — this is the user-perceived "blocks per screen pixel" and
        // already accounts for any CSS scale, OVERSCAN, and tile.scale state.
        const blocksPerPx = DEFAULT_SCALE / zoomLevel;
        setViewCenter((c) => ({
          x: Math.round(c.x + -drag.dx * blocksPerPx),
          z: Math.round(c.z + -drag.dz * blocksPerPx),
        }));
      }
      dragStartRef.current = null;
      setDrag(null);
      (e.target as Element).releasePointerCapture?.(e.pointerId);
    },
    [drag, zoomLevel],
  );

  // Zoom by a multiplicative step. 1.4× per click gives a noticeable but not
  // jarring jump, ~5 clicks to traverse the full range. Continuous wheel
  // (below) uses much finer steps.
  function zoomIn() {
    setZoomLevel((z) => clampZoom(z * 1.4));
  }
  function zoomOut() {
    setZoomLevel((z) => clampZoom(z / 1.4));
  }
  function resetView() {
    setViewCenter({ x: 0, z: 0 });
    setZoomLevel(1.0);
  }

  // Wheel-to-zoom: each wheel tick multiplies zoom by exp(-deltaY/500),
  // ~+10% per notch on most mice — natural, continuous, and rounds into the
  // same cubScale boundary logic the buttons use.
  // Wheel-zoom anchored to the cursor: the world point under the mouse stays
  // put as you zoom. Computed by figuring out which world coord is currently
  // beneath the cursor, then shifting `viewCenter` so the same world coord
  // sits at the same screen position at the new zoom.
  //
  // Implementation note: every wheel tick shifts viewCenter, which fires the
  // tile-fetch effect. Phase 6b's per-request cancellation in Tauri means
  // only the latest tick's tile actually renders — without that the wheel
  // would create a long tail of stale fetches.
  const onWheel = useCallback(
    (e: React.WheelEvent<HTMLDivElement>) => {
      if (selectedSeed == null) return;
      e.preventDefault();
      const factor = Math.exp(-e.deltaY / 500);
      const newZoom = clampZoom(zoomLevel * factor);
      if (newZoom === zoomLevel) return;
      const pane = mapPaneRef.current;
      if (!pane) {
        setZoomLevel(newZoom);
        return;
      }
      const rect = pane.getBoundingClientRect();
      // Cursor offset from pane centre, in pane CSS pixels.
      const dxScreen = e.clientX - rect.left - rect.width / 2;
      const dyScreen = e.clientY - rect.top - rect.height / 2;
      // Current and new blocks-per-pane-pixel — derived from zoomLevel and
      // DEFAULT_SCALE (this is the same conversion the drag-delta math uses).
      const bppOld = DEFAULT_SCALE / zoomLevel;
      const bppNew = DEFAULT_SCALE / newZoom;
      // World point currently under the cursor.
      const worldX = viewCenter.x + dxScreen * bppOld;
      const worldZ = viewCenter.z + dyScreen * bppOld;
      // New viewCenter that keeps that world point under the same cursor.
      setZoomLevel(newZoom);
      setViewCenter({
        x: Math.round(worldX - dxScreen * bppNew),
        z: Math.round(worldZ - dyScreen * bppNew),
      });
    },
    [selectedSeed, zoomLevel, viewCenter.x, viewCenter.z],
  );

  // Smooth CSS-scale factor: applied to the over-rendered tile container so
  // continuous zoomLevel changes show immediately without waiting for a
  // cubiomes re-render. When `tile.scale` and `pickScale(zoomLevel)` agree
  // (the steady-state after a refetch), cssScale ≈ 1 at the matching
  // zoomLevel and grows/shrinks linearly until a scale boundary crosses.
  const cssScale = tile ? (zoomLevel * tile.scale) / DEFAULT_SCALE : 1;

  // Helper: world (block) coord → percentage within the rendered tile.
  // Uses per-axis spans because the tile is no longer guaranteed square
  // (it now adapts to the pane's aspect ratio).
  function worldToTilePct(blockX: number, blockZ: number) {
    if (!tile) return null;
    const spanX = tile.sx * tile.scale;
    const spanZ = tile.sz * tile.scale;
    const left = ((blockX - tile.x) / spanX) * 100;
    const top = ((blockZ - tile.z) / spanZ) * 100;
    if (left < 0 || left > 100 || top < 0 || top > 100) return null;
    return { left, top };
  }

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand">mcseedfinder</div>
        <div className="control">
          <label>Edition</label>
          <div className="segments">
            <button className={edition === "java" ? "active" : ""} onClick={() => setEdition("java")}>
              Java
            </button>
            <button className={edition === "bedrock" ? "active" : ""} onClick={() => setEdition("bedrock")}>
              Bedrock
            </button>
          </div>
        </div>
        <label className="control">
          Version
          <select value={version} onChange={(event) => setVersion(event.target.value)}>
            <option>1.21</option>
            <option>1.20</option>
            <option>1.19</option>
            <option>1.18</option>
          </select>
        </label>
        <div className="control control-row">
          <label>
            Seeds
            <input value={count} min={1} type="number" onChange={(event) => setCount(Number(event.target.value))} />
          </label>
          <label>
            Max matches
            <input value={maxMatches} min={1} type="number" onChange={(event) => setMaxMatches(Number(event.target.value))} />
          </label>
        </div>
        <div className="conditionsHeader">
          <h3>Conditions</h3>
          <small>Build a tree; any seed matching the root will be returned.</small>
        </div>
        <div className="presetTemplates">
          <small>Quick start:</small>
          <div className="presetTemplateButtons">
            {PRESET_TEMPLATES.map((p) => (
              <button
                key={p.key}
                className="presetTemplate"
                title={p.description}
                onClick={() => {
                  // Wrap existing tree + preset in an all_of unless the
                  // current tree is the default sole leaf — in which case
                  // just replace.
                  const preset = p.build();
                  if (
                    conditionTree.type === "nearby_structure" &&
                    conditionTree.structure === "village" &&
                    conditionTree.max_distance === 1000
                  ) {
                    setConditionTree(preset);
                  } else {
                    setConditionTree({
                      id: `preset-wrap-${Date.now()}`,
                      type: "all_of",
                      of: [conditionTree, preset],
                    });
                  }
                }}
              >
                + {p.label}
              </button>
            ))}
          </div>
        </div>
        <ConditionBuilder root={conditionTree} onChange={setConditionTree} />
        <div className="actions">
          <button onClick={startSearch} disabled={job.status === "running"}>Run</button>
          <button className="secondary" onClick={cancelSearch} disabled={job.status !== "running"}>Cancel</button>
        </div>
        <div className="control importControl">
          <label htmlFor="levelDatInput">Import world (level.dat)</label>
          <input
            id="levelDatInput"
            type="file"
            accept=".dat,application/octet-stream"
            onChange={importLevelDat}
          />
        </div>
        {error && <div className="error">{error}</div>}
      </aside>

      <section className="mapPane" ref={mapPaneRef}>
        {/* Dimension tabs + overlay toggles — shared across 2D and 3D
            modes. Positioned absolutely so they overlay both
            conditional branches without code duplication. */}
        <div className="mapTopBar">
          <div className="dimTabs" role="tablist" aria-label="Dimension">
            {(["overworld", "nether", "end"] as const).map((dim) => (
              <button
                key={dim}
                role="tab"
                aria-selected={dimension === dim}
                className={`dimTab ${dimension === dim ? "active" : ""}`}
                onClick={() => setDimension(dim)}
                title={`Render the ${dim} dimension`}
              >
                {dim === "overworld" ? "Overworld" : dim === "nether" ? "Nether" : "End"}
              </button>
            ))}
          </div>
          <div className="overlayToggles">
            <button
              className={`overlayToggle ${overlays.slime ? "active" : ""}`}
              onClick={() => setOverlays((o) => ({ ...o, slime: !o.slime }))}
              title="Toggle slime-chunk overlay (green plates over slime chunks)"
            >
              🟢 slime
            </button>
            <button
              className={`overlayToggle ${overlays.border ? "active" : ""}`}
              onClick={() => setOverlays((o) => ({ ...o, border: !o.border }))}
              title="Toggle world border wireframe (±29 999 984)"
            >
              ▢ border
            </button>
            <button
              className={`overlayToggle ${overlays.rings ? "active" : ""}`}
              onClick={() => setOverlays((o) => ({ ...o, rings: !o.rings }))}
              title="Toggle 8 concentric stronghold ring constraint annuli"
            >
              ◯ rings
            </button>
            <button
              className={`overlayToggle ${overlays.spawn ? "active" : ""}`}
              onClick={() => setOverlays((o) => ({ ...o, spawn: !o.spawn }))}
              title="Toggle world-spawn star + spawn chunks (16×16 chunks centred on spawn)"
            >
              ★ spawn
            </button>
            <button
              className={`overlayToggle ${overlays.npDebug ? "active" : ""}`}
              onClick={() => setOverlays((o) => ({ ...o, npDebug: !o.npDebug }))}
              title="Show cubiomes np[6] climate values at cursor (debounced)"
            >
              🔬 np
            </button>
          </div>
        </div>
        {mapView === "3D" && tile ? (
          <div className="mapGrid">
            <Map3D
              tile={{
                bytes: tile.bytes,
                biomeIds: tile.biome_ids,
                sx: tile.sx,
                sz: tile.sz,
                x: tile.x,
                z: tile.z,
                scale: tile.scale,
              }}
              heights={heightTile?.heights ?? null}
              pins={pins}
              cameraZoom={zoomLevel}
              yLevel={yLevel}
              yMin={Y_MIN}
              yMax={Y_MAX}
              onYDelta={(d) => setYLevel((y) => clampY(y + d))}
              onYSet={(y) => setYLevel(clampY(y))}
              onHover={setHover}
              slimeChunks={overlays.slime ? slimeChunks : []}
              showBorder={overlays.border}
              showStrongholdRings={overlays.rings}
              spawnPos={overlays.spawn ? worldSpawn : null}
              showSpawnChunks={overlays.spawn}
            />
            <div className="mapControls">
              <button onClick={zoomIn} title="Zoom in (smaller scale)">+</button>
              <button onClick={zoomOut} title="Zoom out (larger scale)">−</button>
              <button onClick={resetView} title="Recenter on origin">⌂</button>
              <button
                onClick={() => setMapView("2D")}
                title="Switch to flat 2D map"
                className="viewToggle"
              >
                2D
              </button>
            </div>
            <div className="mapLegend">
              <span className="legendItem"><span className="legendDot pin-village" />Village</span>
              <span className="legendItem"><span className="legendDot pin-pillager_outpost" />Outpost</span>
              <span className="legendItem"><span className="legendDot pin-ocean_monument" />Monument</span>
              <span className="legendItem"><span className="legendDot pin-woodland_mansion" />Mansion</span>
              <span className="legendItem"><span className="legendDot pin-stronghold" />Stronghold</span>
              <span className="legendItem"><span className="legendDot legendSpawn" />Spawn (0,0)</span>
            </div>
            <div className="badgeStack">
              {heightTile ? (
                <div
                  className="terrainBadge"
                  title={`Surface heights come from cubiomes' mapApproxHeight — depth-spline-based, not bit-exact Java terrain. (${heightTile.heights.length} samples)`}
                >
                  approximate terrain (y {Math.round(Math.min(...heightTile.heights))}..{Math.round(Math.max(...heightTile.heights))})
                </div>
              ) : (
                cubScale !== 4 && (
                  <div className="terrainBadge" style={{ background: "rgba(60, 60, 60, 0.75)", color: "#bbb" }}>
                    heightmap available at 1:4 only — zoom in
                  </div>
                )
              )}
              <div
                className="rendererBadge"
                title={
                  WEBGPU_AVAILABLE
                    ? "Host WebView advertises navigator.gpu — WebGPU available. (Renderer is still WebGL2; async-gl swap to WebGPURenderer is a future PR.)"
                    : "Host WebView has no navigator.gpu (typical on Linux/WebKitGTK as of 2025). Running on WebGL2."
                }
              >
                {WEBGPU_AVAILABLE ? "webgl2 · webgpu ready" : "webgl2"}
              </div>
            </div>
            {hover && (
              <div className="cursorReadout">
                <strong>{biomeLabel(hover.biomeId)}</strong>{" "}
                <span className="cursorReadoutSub">
                  @ ({hover.worldX}, Y={hover.y}, {hover.worldZ}) · id {hover.biomeId}
                </span>
                {climateNp && (
                  <div className="climateNp" title="cubiomes np[6] climate values at cursor (×10000, i64)">
                    {(
                      [
                        ["T", climateNp[0]],
                        ["H", climateNp[1]],
                        ["C", climateNp[2]],
                        ["E", climateNp[3]],
                        ["D", climateNp[4]],
                        ["W", climateNp[5]],
                      ] as const
                    ).map(([k, v]) => (
                      <span key={k} className="climateNpField">
                        <span className="climateNpKey">{k}</span>
                        {(v / 10000).toFixed(3)}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )}
            <div className="tileLabel">
              seed {tile.seed} · Y = {yLevel} {yBandLabel(yLevel)}
              {tile.y != null && tile.y !== yLevel ? ` [rendered @ y=${tile.y}]` : ""}
              {" "}· zoom {zoomLevel.toFixed(2)}× (1:{tile.scale}) ·
              {" "}centre ({viewCenter.x}, {viewCenter.z}) ·
              {" "}{pins.length} structure{pins.length === 1 ? "" : "s"} in view
            </div>
          </div>
        ) : (
          <div
            className="mapGrid"
            onPointerDown={onPanStart}
            onPointerMove={onPanMove}
            onPointerUp={onPanEnd}
            onPointerCancel={onPanEnd}
            onWheel={onWheel}
            style={{ cursor: drag ? "grabbing" : selectedSeed != null ? "grab" : "default" }}
          >
            {tile ? (
              <>
                <div
                  className="mapContent"
                  style={{
                    // Over-rendered tile: 150% of the pane, inset by -25% on
                    // each side so its centre aligns with the pane centre.
                    // Transform combines: continuous zoom (CSS scale) + pan
                    // drag (CSS translate during a drag, identity otherwise).
                    top: `${OVERSCAN_INSET_PCT}%`,
                    left: `${OVERSCAN_INSET_PCT}%`,
                    width: `${OVERSCAN * 100}%`,
                    height: `${OVERSCAN * 100}%`,
                    transformOrigin: "center center",
                    transform: drag
                      ? `translate(${drag.dx}px, ${drag.dz}px) scale(${cssScale})`
                      : `scale(${cssScale})`,
                  }}
                >
                  <TileCanvas tile={tile} canvasRef={tileCanvasRef} />
                  {(() => {
                    const origin = worldToTilePct(0, 0);
                    return origin ? (
                      <div
                        className="spawn"
                        style={{ left: `${origin.left}%`, top: `${origin.top}%` }}
                      >
                        0,0
                      </div>
                    ) : null;
                  })()}
                  {pins.map((p) => {
                    const pos = worldToTilePct(p.block_x, p.block_z);
                    if (!pos) return null;
                    return (
                      <button
                        key={`${p.structure}-${p.block_x}-${p.block_z}`}
                        className={`pin pin-${p.structure}`}
                        style={{ left: `${pos.left}%`, top: `${pos.top}%` }}
                        title={`${p.structure.replace(/_/g, " ")} @ (${p.block_x}, ${p.block_z})`}
                      />
                    );
                  })}
                </div>
                <div className="mapControls">
                  <button onClick={zoomIn} title="Zoom in (smaller scale)">+</button>
                  <button onClick={zoomOut} title="Zoom out (larger scale)">−</button>
                  <button onClick={resetView} title="Recenter on origin">⌂</button>
                  <button
                    onClick={() => setMapView("3D")}
                    title="Switch to 3D isometric map (wheel scrubs Y)"
                    className="viewToggle"
                  >
                    3D
                  </button>
                </div>
                <div className="mapLegend">
                  <span className="legendItem"><span className="legendDot pin-village" />Village</span>
                  <span className="legendItem"><span className="legendDot pin-pillager_outpost" />Outpost</span>
                  <span className="legendItem"><span className="legendDot pin-ocean_monument" />Monument</span>
                  <span className="legendItem"><span className="legendDot pin-woodland_mansion" />Mansion</span>
                  <span className="legendItem"><span className="legendDot pin-stronghold" />Stronghold</span>
                  <span className="legendItem"><span className="legendDot legendSpawn" />Spawn (0,0)</span>
                </div>
                <div className="tileLabel">
                  seed {tile.seed} · zoom {zoomLevel.toFixed(2)}× (1:{tile.scale}) ·
                  {" "}centre ({viewCenter.x}, {viewCenter.z}) ·
                  {" "}{pins.length} structure{pins.length === 1 ? "" : "s"} in view
                </div>
              </>
            ) : (
              <div className="mapEmpty">
                {selectedSeed == null
                  ? "Pick a seed from the results list to render its biome map."
                  : "Rendering biome tile…"}
              </div>
            )}
          </div>
        )}
      </section>

      <aside className="inspector">
        <section className="panel">
          <h2>Live Results ({results.length})</h2>
          {/* #5 — Match histogram across the searched seed range. Buckets
              the results into 32 columns by `(seed - start_seed) / count`
              ratio, so the user can see at a glance which parts of the
              range were productive. Pure visualisation — clicking a bucket
              does nothing (yet). */}
          {results.length > 1 && (
            <div className="matchHistogram" title={`${results.length} matches across the search range`}>
              {(() => {
                const BUCKETS = 32;
                const counts = new Array(BUCKETS).fill(0);
                for (const r of results) {
                  // seed in i64 range; bucket on the low 32 bits relative to count.
                  const idx = Math.abs(Math.floor(((r.seed % count) / count) * BUCKETS));
                  counts[Math.min(BUCKETS - 1, idx)] += 1;
                }
                const maxCount = Math.max(1, ...counts);
                return counts.map((c, i) => (
                  <div
                    key={i}
                    className="histBar"
                    style={{ height: `${(c / maxCount) * 100}%` }}
                    title={`bucket ${i}: ${c} matches`}
                  />
                ));
              })()}
            </div>
          )}
          <div className="resultList">
            {results.map((result) => (
              <button
                key={result.seed}
                className={selectedSeed === result.seed ? "result active" : "result"}
                onClick={() => setSelectedSeed(result.seed)}
              >
                <ResultThumbnail
                  seed={result.seed}
                  version={result.version}
                  dimension={result.dimension}
                  cacheRef={thumbCacheRef}
                />
                <div className="resultText">
                  <span>{result.seed}</span>
                  <small>{result.edition} {result.version}</small>
                </div>
              </button>
            ))}
            {results.length === 0 && job.status === "idle" && (
              <div className="hint">Press Run to start a search. Matches stream in live.</div>
            )}
          </div>
        </section>
        <section className="panel">
          <h2>Share</h2>
          <div className="shareActions">
            <button className="secondary" onClick={copyShareBlob}>Copy share link</button>
            <button className="secondary" onClick={downloadTile} disabled={!tile}>Download tile PNG</button>
          </div>
          <details className="shareImport">
            <summary>Paste a share link to import…</summary>
            <textarea
              rows={3}
              placeholder="Paste the base64 blob copied by another session"
              onBlur={(e) => {
                if (e.target.value.trim()) {
                  applyShareBlob(e.target.value);
                  e.target.value = "";
                }
              }}
            />
            <small>Auto-imports on blur (click outside the textarea).</small>
          </details>
        </section>
        <section className="panel analyzer">
          <h2>Analyzer</h2>
          <dl>
            <dt>Seed</dt>
            <dd>{selectedSeed ?? "none"}</dd>
            <dt>Conditions</dt>
            <dd>
              {conditionTree.type === "all_of" ||
              conditionTree.type === "any_of" ||
              conditionTree.type === "none_of"
                ? `${conditionTree.type.replace("_", " ")} of ${conditionTree.of.length}`
                : conditionTree.type.replace(/_/g, " ")}
            </dd>
            {analysis && (
              <>
                <dt>Origin biome</dt>
                <dd>{analysis.origin_biome_id} {analysis.origin_biome_exact ? "(exact)" : "(approx)"}</dd>
                <dt>Nearest village</dt>
                <dd>
                  {analysis.nearest_village
                    ? `(${analysis.nearest_village.block_x}, ${analysis.nearest_village.block_z})`
                    : "none in range"}
                </dd>
                <dt>Strongholds (ring 1)</dt>
                <dd>{analysis.strongholds.length}</dd>
              </>
            )}
          </dl>
          <button className="secondary" onClick={analyzeSelected} disabled={selectedSeed == null}>
            Analyze
          </button>
        </section>
      </aside>

      <footer className="status">
        <span>{job.status}</span>
        <span>job {job.jobId || "—"}</span>
        <span>scanned {job.scanned.toLocaleString()}</span>
        <span>matches {job.matches}</span>
      </footer>
      {showOnboarding && (
        <div className="onboardingScrim" role="dialog" aria-modal>
          <div className="onboardingCard">
            <h2>Welcome to mc-seed-finder</h2>
            <p className="onboardingLead">
              A bit-exact Minecraft Java seed search toolkit. Here's the lay of the land.
            </p>
            <ul className="onboardingList">
              <li>
                <strong>Left sidebar</strong> — build a condition tree (gates +
                structure / biome leaves) and start a search. Match seeds stream
                into the results panel on the right.
              </li>
              <li>
                <strong>Centre map</strong> — pick a result to render its biome
                map. Switch to <kbd>3D</kbd> in the top-right for voxel columns
                with a wheel-driven Y scrubber. Hover any cell for the bit-exact
                cubiomes biome name.
              </li>
              <li>
                <strong>Top bar</strong> — flip dimension (Overworld / Nether / End)
                or toggle overlays (slime chunks, world border, stronghold rings).
              </li>
              <li>
                <strong>Right sidebar</strong> — live results, share-blob import /
                export, level.dat import, results export.
              </li>
              <li>
                <strong>Keyboard</strong>: <kbd>j</kbd>/<kbd>k</kbd> Y scrub,
                {" "}<kbd>h</kbd>/<kbd>l</kbd> zoom, <kbd>2</kbd>/<kbd>3</kbd>{" "}
                2D/3D, arrows pan, <kbd>r</kbd> recenter. Hold <kbd>Shift</kbd>{" "}
                for bigger steps.
              </li>
              <li>
                Biomes are <strong>bit-exact</strong> via vendored cubiomes; the
                heightmap is <strong>approximate</strong> (cubiomes spline) and
                labelled as such in the HUD.
              </li>
            </ul>
            <button className="onboardingDismiss" onClick={dismissOnboarding}>
              Get started
            </button>
          </div>
        </div>
      )}
    </main>
  );
}

createRoot(document.getElementById("root") as HTMLElement).render(<App />);
