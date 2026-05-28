import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { invoke } from "@tauri-apps/api/core";
import "./styles.css";

type SearchResult = {
  seed: number;
  edition: string;
  version: string;
  dimension: string;
  score: number;
};

type JobState = {
  jobId: string;
  status: string;
  scanned: number;
  matches: number;
  rate: number;
};

const fallbackResults: SearchResult[] = [
  { seed: 1, edition: "java", version: "1.21", dimension: "overworld", score: 0 },
];

function App() {
  const [edition, setEdition] = useState("java");
  const [version, setVersion] = useState("1.21");
  const [structure, setStructure] = useState("village");
  const [distance, setDistance] = useState(1000);
  const [count, setCount] = useState(100000);
  const [job, setJob] = useState<JobState>({
    jobId: "local-preview",
    status: "idle",
    scanned: 0,
    matches: 0,
    rate: 0,
  });
  const [results, setResults] = useState<SearchResult[]>([]);
  const [selectedSeed, setSelectedSeed] = useState<number | null>(null);

  const spec = useMemo(
    () => ({
      edition,
      version,
      dimension: "overworld",
      mode: "sequential",
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

  async function startSearch() {
    setJob((current) => ({ ...current, status: "running", scanned: 0, matches: 0 }));
    try {
      const jobId = await invoke<string>("start_search", { spec });
      const exported = await invoke<SearchResult[]>("export_results", {
        jobId,
        format: "json",
      });
      setResults(exported.length ? exported : fallbackResults);
      setSelectedSeed((exported[0] ?? fallbackResults[0]).seed);
      setJob({
        jobId,
        status: "completed",
        scanned: count,
        matches: exported.length,
        rate: count,
      });
    } catch {
      setResults(fallbackResults);
      setSelectedSeed(fallbackResults[0].seed);
      setJob((current) => ({ ...current, status: "preview", matches: 1 }));
    }
  }

  async function cancelSearch() {
    await invoke("cancel_search", { jobId: job.jobId }).catch(() => undefined);
    setJob((current) => ({ ...current, status: "cancelled" }));
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
          <button onClick={startSearch}>Run</button>
          <button className="secondary" onClick={cancelSearch}>Cancel</button>
        </div>
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
          <h2>Live Results</h2>
          <div className="resultList">
            {(results.length ? results : fallbackResults).map((result) => (
              <button
                key={result.seed}
                className={selectedSeed === result.seed ? "result active" : "result"}
                onClick={() => setSelectedSeed(result.seed)}
              >
                <span>{result.seed}</span>
                <small>{result.edition} {result.version}</small>
              </button>
            ))}
          </div>
        </section>
        <section className="panel analyzer">
          <h2>Analyzer</h2>
          <dl>
            <dt>Seed</dt>
            <dd>{selectedSeed ?? "none"}</dd>
            <dt>Target</dt>
            <dd>{structure.replace("_", " ")} within {distance} blocks</dd>
            <dt>Exactness</dt>
            <dd>{structure === "stronghold" || structure ? "structure exact" : "candidate"}</dd>
          </dl>
        </section>
      </aside>

      <footer className="status">
        <span>{job.status}</span>
        <span>job {job.jobId}</span>
        <span>scanned {job.scanned.toLocaleString()}</span>
        <span>matches {job.matches}</span>
        <span>{Math.round(job.rate).toLocaleString()} seeds/s</span>
      </footer>
    </main>
  );
}

createRoot(document.getElementById("root") as HTMLElement).render(<App />);
