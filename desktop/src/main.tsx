import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import "./styles.css";

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
  status: string;       // idle | running | completed | cancelled | error
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
  const [error, setError] = useState<string | null>(null);

  // The backend emits events with a `job_id` payload; we ignore anything not
  // for the currently-active job (stale events from a cancelled/restarted run).
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

  // Subscribe once on mount; the Rust side streams events for the whole app
  // lifetime, and we filter by job_id in the handler.
  useEffect(() => {
    let unlistenFns: UnlistenFn[] = [];
    (async () => {
      unlistenFns.push(
        await listen<SearchResult>("search-match", (event) => {
          const payload = event.payload;
          // (job_id is in started/progress/completed events; match events carry
          // the report directly. We accept any match while a job is active.)
          if (!activeJobIdRef.current) return;
          setResults((prev) => [...prev, payload]);
          setJob((j) => ({ ...j, matches: j.matches + 1 }));
          if (!selectedSeed) setSelectedSeed(payload.seed);
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function startSearch() {
    setError(null);
    setResults([]);
    setSelectedSeed(null);
    setAnalysis(null);
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
    // The completed event with reason="cancelled" will flip status; this is
    // just optimistic UI in case the worker is between chunk boundaries.
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

      <section className="mapPane">
        <div className="mapGrid">
          <div className="spawn">0,0</div>
          <div className="ring" />
          {results.slice(0, 8).map((result, index) => (
            <button
              key={`${result.seed}-${index}`}
              className="pin"
              style={{ left: `${18 + index * 9}%`, top: `${35 + (index % 3) * 12}%` }}
              onClick={() => setSelectedSeed(result.seed)}
              title={`Seed ${result.seed}`}
            />
          ))}
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
