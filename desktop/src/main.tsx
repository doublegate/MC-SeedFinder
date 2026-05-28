import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import {
  ConditionBuilder,
  defaultRoot,
  nodeToWire,
  wireToNode,
  type TreeNode,
} from "./conditions";
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
      setError(String(e));
    }
  }

  // Build / consume a share blob: base64-encoded JSON with the current spec
  // and (if a seed is selected) the map view. Lets users copy a setup and
  // recreate it later — the desktop equivalent of a deep link.
  function buildShareBlob(): string {
    const payload = {
      v: 1,
      spec: {
        edition,
        version,
        count,
        max_matches: maxMatches,
        criteria: { conditions: nodeToWire(conditionTree) },
      },
      view:
        selectedSeed != null
          ? { seed: selectedSeed, x: viewCenter.x, z: viewCenter.z, scale }
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
      if (payload.v !== 1 || !payload.spec) {
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
            if (typeof v.scale === "number") setScale(v.scale);
          }, 0);
        }
      }
    } catch (e) {
      setError(`Import share failed: ${e}`);
    }
  }

  function downloadTile() {
    if (!tile) return;
    const a = document.createElement("a");
    a.href = `data:image/png;base64,${tile.png_base64}`;
    a.download = `seed-${tile.seed}-x${viewCenter.x}-z${viewCenter.z}-scale${scale}.png`;
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
    </main>
  );
}

createRoot(document.getElementById("root") as HTMLElement).render(<App />);
