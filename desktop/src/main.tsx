import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import "./styles.css";

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

type TileResponse = {
  png_base64: string;
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

const TILE_PX = 256;
// cubiomes' supported scales: 1, 4, 16, 64, 256. Lower index = closer zoom.
const SCALE_LEVELS = [1, 4, 16, 64, 256] as const;
const DEFAULT_SCALE = 4;
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
  const [structure, setStructure] = useState("village");
  const [distance, setDistance] = useState(1000);
  const [count, setCount] = useState(100000);
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
  const [pins, setPins] = useState<StructurePin[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Map view state (block coordinates of the view centre + cubiomes scale).
  const [viewCenter, setViewCenter] = useState({ x: 0, z: 0 });
  const [scale, setScale] = useState<number>(DEFAULT_SCALE);

  // Active-drag pixel offset (CSS translate). Committed to viewCenter on
  // pointer-up, then the tile/pin fetch effect re-fires.
  const [drag, setDrag] = useState<{ dx: number; dz: number } | null>(null);
  const dragStartRef = useRef<{ x: number; y: number } | null>(null);
  const mapPaneRef = useRef<HTMLDivElement | null>(null);

  // Stale-event filter for streamed search.
  const activeJobIdRef = useRef<string>("");

  const spec = useMemo(
    () => ({
      edition,
      version,
      dimension: "overworld",
      start_seed: 0,
      count,
      max_matches: 25,
      criteria: {
        nearby_structures: [
          { structure, max_distance: distance, centre_x: 0, centre_z: 0 },
        ],
      },
    }),
    [count, distance, edition, structure, version],
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
            if (event.payload.job_id !== activeJobIdRef.current) return;
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
          if (event.payload.job_id !== activeJobIdRef.current) return;
          setJob((j) => ({
            ...j,
            status:
              event.payload.reason === "cancelled" ? "cancelled" : "completed",
            scanned: event.payload.scanned,
            matches: event.payload.matches,
          }));
          if (event.payload.error) setError(event.payload.error);
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
      setError(String(e));
    }
  }

  // Re-centre on origin when a new seed is selected (don't carry pan/zoom).
  useEffect(() => {
    setViewCenter({ x: 0, z: 0 });
    setScale(DEFAULT_SCALE);
  }, [selectedSeed]);

  // Fetch the biome tile AND structure pins whenever the view changes.
  useEffect(() => {
    if (selectedSeed == null) {
      setTile(null);
      setPins([]);
      return;
    }
    const tileWorldSpan = TILE_PX * scale;
    const tileX = viewCenter.x - tileWorldSpan / 2;
    const tileZ = viewCenter.z - tileWorldSpan / 2;
    let cancelled = false;
    (async () => {
      try {
        const [t, ps] = await Promise.all([
          invoke<TileResponse>("render_tile", {
            request: {
              seed: selectedSeed,
              version,
              dimension: "overworld",
              x: tileX,
              z: tileZ,
              scale,
              sx: TILE_PX,
              sz: TILE_PX,
            },
          }),
          invoke<StructurePin[]>("list_structures_in_view", {
            request: {
              seed: selectedSeed,
              structures: PIN_STRUCTURES,
              x: tileX,
              z: tileZ,
              sx: tileWorldSpan,
              sz: tileWorldSpan,
            },
          }),
        ]);
        if (cancelled) return;
        setTile(t);
        setPins(ps);
      } catch (e) {
        if (!cancelled) setError(String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedSeed, version, viewCenter.x, viewCenter.z, scale]);

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
      const pane = mapPaneRef.current;
      if (pane && (drag.dx !== 0 || drag.dz !== 0)) {
        const rect = pane.getBoundingClientRect();
        // Drag right → reveal LEFT of world → centre.x decreases.
        const blockDx = (-drag.dx / rect.width) * TILE_PX * scale;
        const blockDz = (-drag.dz / rect.height) * TILE_PX * scale;
        setViewCenter((c) => ({
          x: Math.round(c.x + blockDx),
          z: Math.round(c.z + blockDz),
        }));
      }
      dragStartRef.current = null;
      setDrag(null);
      (e.target as Element).releasePointerCapture?.(e.pointerId);
    },
    [drag, scale],
  );

  function zoomIn() {
    setScale((s) => {
      const i = SCALE_LEVELS.indexOf(s as (typeof SCALE_LEVELS)[number]);
      return SCALE_LEVELS[Math.max(0, i - 1)];
    });
  }
  function zoomOut() {
    setScale((s) => {
      const i = SCALE_LEVELS.indexOf(s as (typeof SCALE_LEVELS)[number]);
      return SCALE_LEVELS[Math.min(SCALE_LEVELS.length - 1, i + 1)];
    });
  }
  function resetView() {
    setViewCenter({ x: 0, z: 0 });
    setScale(DEFAULT_SCALE);
  }

  // Helper: world (block) coord → percentage within the rendered tile.
  function worldToTilePct(blockX: number, blockZ: number) {
    if (!tile) return null;
    const span = tile.sx * tile.scale;
    const left = ((blockX - tile.x) / span) * 100;
    const top = ((blockZ - tile.z) / span) * 100;
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
        <label className="control">
          Structure
          <select value={structure} onChange={(event) => setStructure(event.target.value)}>
            <option value="village">Village</option>
            <option value="pillager_outpost">Pillager Outpost</option>
            <option value="ocean_monument">Ocean Monument</option>
            <option value="stronghold">Stronghold</option>
            <option value="woodland_mansion">Woodland Mansion</option>
          </select>
        </label>
        <label className="control">
          Distance
          <input value={distance} min={1} max={8000} type="number" onChange={(event) => setDistance(Number(event.target.value))} />
        </label>
        <label className="control">
          Seeds
          <input value={count} min={1} type="number" onChange={(event) => setCount(Number(event.target.value))} />
        </label>
        <div className="actions">
          <button onClick={startSearch} disabled={job.status === "running"}>Run</button>
          <button className="secondary" onClick={cancelSearch} disabled={job.status !== "running"}>Cancel</button>
        </div>
        {error && <div className="error">{error}</div>}
      </aside>

      <section className="mapPane" ref={mapPaneRef}>
        <div
          className="mapGrid"
          onPointerDown={onPanStart}
          onPointerMove={onPanMove}
          onPointerUp={onPanEnd}
          onPointerCancel={onPanEnd}
          style={{ cursor: drag ? "grabbing" : selectedSeed != null ? "grab" : "default" }}
        >
          {tile ? (
            <>
              <div
                className="mapContent"
                style={{
                  transform: drag ? `translate(${drag.dx}px, ${drag.dz}px)` : undefined,
                }}
              >
                <img
                  className="tileImage"
                  src={`data:image/png;base64,${tile.png_base64}`}
                  alt={`Biome tile for seed ${tile.seed}`}
                  draggable={false}
                />
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
              </div>
              <div className="tileLabel">
                seed {tile.seed} · 1:{scale} · centre ({viewCenter.x}, {viewCenter.z}) ·
                {" "}
                {pins.length} structure{pins.length === 1 ? "" : "s"} in view
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
      </section>

      <aside className="inspector">
        <div className="tabs">
          <button className="active">Finder</button>
          <button>Results</button>
          <button>Analyzer</button>
        </div>
        <section className="panel">
          <h2>Live Results ({results.length})</h2>
          <div className="resultList">
            {results.map((result) => (
              <button
                key={result.seed}
                className={selectedSeed === result.seed ? "result active" : "result"}
                onClick={() => setSelectedSeed(result.seed)}
              >
                <span>{result.seed}</span>
                <small>{result.edition} {result.version}</small>
              </button>
            ))}
            {results.length === 0 && job.status === "idle" && (
              <div className="hint">Press Run to start a search. Matches stream in live.</div>
            )}
          </div>
        </section>
        <section className="panel analyzer">
          <h2>Analyzer</h2>
          <dl>
            <dt>Seed</dt>
            <dd>{selectedSeed ?? "none"}</dd>
            <dt>Target</dt>
            <dd>{structure.replace("_", " ")} within {distance} blocks</dd>
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
    </main>
  );
}

createRoot(document.getElementById("root") as HTMLElement).render(<App />);
