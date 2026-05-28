//! Tauri command surface wired to the real Rust `mcseedfinder-core` engine.
//!
//! Phase 4a focuses on the **backend wiring**: `start_search` runs a real
//! search in a worker thread and streams matches back to the UI via the Tauri
//! event system; `cancel_search` flips a cooperative-cancellation flag;
//! `analyze_seed` returns the exact origin biome from the cubiomes backend.
//!
//! Biome criteria in a *search* spec are deferred to Phase 4a-2 (biome
//! evaluation in Rust is a separate piece of work) — for now the command
//! rejects such specs with a clear typed error instead of silently dropping
//! the biome check.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State};
use uuid::Uuid;

// The mcseedfinder-core lib is named `_native` so the pyo3 macro produces the
// right `PyInit__native` symbol for the Python extension; from a Rust consumer
// it just looks unusual in `use` paths.
use _native::biomes::{BiomeBackend, DEFAULT_Y};
use _native::conditions::{self, CompiledNode, Node};
use _native::structures::{iter_strongholds, iter_structures_in_radius, StructureType};

// ---------------------------------------------------------------------------
// Wire-format types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
struct SearchSpec {
    edition: String,
    version: String,
    dimension: String,
    #[serde(default)]
    start_seed: i64,
    count: u64,
    max_matches: u64,
    /// The Python criteria JSON shape. May contain a recursive `conditions`
    /// tree and/or the flat `nearby_structures` array. Biome criteria are
    /// rejected for now (see file-level docstring).
    criteria: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
struct SeedReport {
    seed: i64,
    edition: String,
    version: String,
    dimension: String,
    score: f64,
    matched_features: Vec<String>,
    exactness: HashMap<String, String>,
    warnings: Vec<String>,
}

// Tauri event payloads. Names match what the React side listens for.
#[derive(Debug, Clone, Serialize)]
struct ProgressEvent {
    job_id: String,
    scanned: u64,
    matches: u64,
}

#[derive(Debug, Clone, Serialize)]
struct CompletedEvent {
    job_id: String,
    scanned: u64,
    matches: u64,
    reason: String, // "max_matches" | "range_exhausted" | "cancelled" | "error"
    error: Option<String>,
}

// ---------------------------------------------------------------------------
// App state: tracks per-job cancellation + collected results
// ---------------------------------------------------------------------------

#[derive(Default)]
struct AppState {
    jobs: Mutex<HashMap<String, JobInner>>,
}

#[derive(Default)]
struct JobInner {
    results: Vec<SeedReport>,
    cancel: Arc<AtomicBool>,
}

// ---------------------------------------------------------------------------
// Criteria → compiled native tree (structure-only)
// ---------------------------------------------------------------------------

/// Build a structure-only [`CompiledNode`] from a Python-style criteria spec,
/// or return an error string if anything biome-related is present.
fn compile_criteria_tree(criteria: &serde_json::Value) -> Result<CompiledNode, String> {
    let obj = criteria.as_object().ok_or("criteria must be a JSON object")?;

    if obj.contains_key("spawn_biome") || obj.contains_key("nearby_biomes") {
        return Err(
            "biome criteria are not yet supported in desktop search — \
             use only structure conditions for now"
                .to_string(),
        );
    }

    let mut children: Vec<Node> = Vec::new();

    // Flat `nearby_structures` → NearbyStructure leaves.
    if let Some(list) = obj.get("nearby_structures").and_then(|v| v.as_array()) {
        for entry in list {
            let structure = entry
                .get("structure")
                .and_then(|s| s.as_str())
                .ok_or("nearby_structures entry missing 'structure'")?
                .to_string();
            let max_distance = entry
                .get("max_distance")
                .and_then(|v| v.as_i64())
                .unwrap_or(1500) as i32;
            let centre_x = entry
                .get("centre_x")
                .and_then(|v| v.as_i64())
                .unwrap_or(0) as i32;
            let centre_z = entry
                .get("centre_z")
                .and_then(|v| v.as_i64())
                .unwrap_or(0) as i32;
            children.push(Node::NearbyStructure {
                structure,
                max_distance,
                centre_x,
                centre_z,
            });
        }
    }

    // Recursive `conditions` tree → Node. Any biome leaf type causes serde
    // deserialisation to fail (Node only knows structure types), which we
    // surface as an error rather than silently dropping the criterion.
    if let Some(tree) = obj.get("conditions") {
        let tree_node: Node = serde_json::from_value(tree.clone()).map_err(|e| {
            format!(
                "conditions tree contains unsupported / biome criteria for desktop search ({e})"
            )
        })?;
        children.push(tree_node);
    }

    if children.is_empty() {
        return Err("criteria has no structure conditions — nothing to search for".into());
    }

    let root = if children.len() == 1 {
        children.pop().unwrap()
    } else {
        Node::AllOf { of: children }
    };

    conditions::compile(&root).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

#[tauri::command]
fn start_search(
    spec: SearchSpec,
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<String, String> {
    if spec.edition != "java" {
        return Err(format!(
            "edition {:?} not supported in desktop search yet",
            spec.edition
        ));
    }
    let compiled = compile_criteria_tree(&spec.criteria)?;

    let job_id = Uuid::new_v4().to_string();
    let cancel = Arc::new(AtomicBool::new(false));

    state
        .jobs
        .lock()
        .map_err(|_| "job store lock poisoned".to_string())?
        .insert(
            job_id.clone(),
            JobInner {
                results: Vec::new(),
                cancel: cancel.clone(),
            },
        );

    let app_clone = app.clone();
    let job_id_clone = job_id.clone();
    let spec_clone = spec.clone();
    thread::spawn(move || {
        run_search(app_clone, job_id_clone, spec_clone, compiled, cancel);
    });

    // Emit a "started" lifecycle event so the UI can flip into a running state.
    let _ = app.emit(
        "search-started",
        serde_json::json!({
            "job_id": job_id,
            "count": spec.count,
            "version": spec.version,
            "dimension": spec.dimension,
        }),
    );

    Ok(job_id)
}

fn emit_completed(
    app: &AppHandle,
    job_id: &str,
    scanned: u64,
    matches: u64,
    reason: &str,
    error: Option<String>,
) {
    let _ = app.emit(
        "search-completed",
        CompletedEvent {
            job_id: job_id.to_string(),
            scanned,
            matches,
            reason: reason.to_string(),
            error,
        },
    );
}

fn run_search(
    app: AppHandle,
    job_id: String,
    spec: SearchSpec,
    compiled: CompiledNode,
    cancel: Arc<AtomicBool>,
) {
    let chunk: u64 = 4096;
    let mut scanned: u64 = 0;
    let mut matches: u64 = 0;

    while scanned < spec.count {
        if cancel.load(Ordering::Relaxed) {
            emit_completed(&app, &job_id, scanned, matches, "cancelled", None);
            return;
        }
        let this_chunk = chunk.min(spec.count - scanned);
        for i in 0..this_chunk {
            let seed = spec.start_seed.wrapping_add((scanned + i) as i64);
            if conditions::evaluate(&compiled, seed) {
                let report = SeedReport {
                    seed,
                    edition: spec.edition.clone(),
                    version: spec.version.clone(),
                    dimension: spec.dimension.clone(),
                    score: 0.0,
                    matched_features: vec!["structure_conditions".to_string()],
                    exactness: HashMap::from([("structures".to_string(), "exact".to_string())]),
                    warnings: Vec::new(),
                };
                if let Some(jobs) = app.try_state::<AppState>() {
                    if let Ok(mut guard) = jobs.jobs.lock() {
                        if let Some(inner) = guard.get_mut(&job_id) {
                            inner.results.push(report.clone());
                        }
                    }
                }
                let _ = app.emit("search-match", &report);
                matches += 1;
                if matches >= spec.max_matches {
                    let scanned_final = scanned + i + 1;
                    emit_completed(&app, &job_id, scanned_final, matches, "max_matches", None);
                    return;
                }
            }
        }
        scanned += this_chunk;
        let _ = app.emit(
            "search-progress",
            ProgressEvent {
                job_id: job_id.clone(),
                scanned,
                matches,
            },
        );
    }

    emit_completed(&app, &job_id, scanned, matches, "range_exhausted", None);
}

#[tauri::command]
fn pause_search(_job_id: String) -> Result<(), String> {
    // Pause/resume is Phase 4a-2 — for now this is a no-op (cancellation works).
    Ok(())
}

#[tauri::command]
fn resume_search(_job_id: String) -> Result<(), String> {
    Ok(())
}

#[tauri::command]
fn cancel_search(job_id: String, state: State<'_, AppState>) -> Result<(), String> {
    let guard = state
        .jobs
        .lock()
        .map_err(|_| "job store lock poisoned".to_string())?;
    if let Some(inner) = guard.get(&job_id) {
        inner.cancel.store(true, Ordering::Relaxed);
    }
    Ok(())
}

#[tauri::command]
fn analyze_seed(
    seed: i64,
    version: Option<String>,
    dimension: Option<String>,
) -> Result<serde_json::Value, String> {
    let version = version.unwrap_or_else(|| "1.21".to_string());
    let dimension = dimension.unwrap_or_else(|| "overworld".to_string());

    let mut backend = BiomeBackend::from_strs(&version, &dimension, DEFAULT_Y).map_err(|e| e)?;
    let origin_biome = backend.get_biome(seed, 0, 0);

    // Nearest village (block coords + distance), iter exact placements within 8000.
    let nearest_village = iter_structures_in_radius(StructureType::Village, seed, 0, 0, 8000)
        .min_by_key(|pos| {
            let dx = pos.block_x() as i64;
            let dz = pos.block_z() as i64;
            dx * dx + dz * dz
        })
        .map(|pos| serde_json::json!({
            "block_x": pos.block_x(),
            "block_z": pos.block_z(),
        }));

    // First-ring stronghold positions (always 3 placements in ring 1).
    let strongholds: Vec<_> = iter_strongholds(seed, 1)
        .map(|pos| serde_json::json!({
            "block_x": pos.block_x(),
            "block_z": pos.block_z(),
        }))
        .collect();

    Ok(serde_json::json!({
        "seed": seed,
        "version": version,
        "dimension": dimension,
        "origin_biome_id": origin_biome,
        "origin_biome_exact": true,
        "nearest_village": nearest_village,
        "strongholds": strongholds,
    }))
}

#[tauri::command]
fn render_tile(request: serde_json::Value) -> serde_json::Value {
    // Phase 4a-2 — biome tile rendering with a base64 PNG payload.
    serde_json::json!({
        "tile": request,
        "status": "not_yet_implemented",
    })
}

#[tauri::command]
fn import_level_dat(path: String) -> serde_json::Value {
    // Phase 4a-2 — NBT parsing via fastnbt.
    serde_json::json!({
        "path": path,
        "status": "not_yet_implemented",
    })
}

#[tauri::command]
fn export_results(
    job_id: String,
    format: String,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let jobs = state
        .jobs
        .lock()
        .map_err(|_| "job store lock poisoned".to_string())?;
    let results = jobs
        .get(&job_id)
        .map(|inner| inner.results.clone())
        .unwrap_or_default();
    if format == "plain" {
        let seeds = results
            .iter()
            .map(|r| r.seed.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        return Ok(serde_json::Value::String(seeds));
    }
    serde_json::to_value(results).map_err(|e| e.to_string())
}

fn main() {
    tauri::Builder::default()
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            start_search,
            pause_search,
            resume_search,
            cancel_search,
            analyze_seed,
            render_tile,
            import_level_dat,
            export_results
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
