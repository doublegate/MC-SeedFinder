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

/// Build a [`CompiledNode`] from a Python-style criteria spec. Supports the
/// flat-keys form (`nearby_structures`) and the recursive `conditions` tree,
/// including biome leaves. Flat `spawn_biome` / `nearby_biomes` keys are
/// rejected with a hint to use the tree form (the flat shape carries biome
/// *names*; the desktop tree path takes numeric cubiomes IDs to avoid
/// duplicating the Python biome catalog in Rust).
fn compile_criteria_tree(criteria: &serde_json::Value) -> Result<CompiledNode, String> {
    let obj = criteria
        .as_object()
        .ok_or("criteria must be a JSON object")?;

    if obj.contains_key("spawn_biome") || obj.contains_key("nearby_biomes") {
        return Err(
            "flat spawn_biome / nearby_biomes keys aren't supported by the \
             desktop search; use a `conditions` tree with biome leaves carrying \
             numeric cubiomes biome IDs"
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
            let centre_x = entry.get("centre_x").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
            let centre_z = entry.get("centre_z").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
            children.push(Node::NearbyStructure {
                structure,
                max_distance,
                centre_x,
                centre_z,
            });
        }
    }

    // Recursive `conditions` tree → Node. Supports structure leaves
    // (NearbyStructure, Cluster), biome leaves (SpawnBiome, NearbyBiomes,
    // BiomeArea — biomes given as numeric cubiomes IDs), and logic gates.
    if let Some(tree) = obj.get("conditions") {
        let tree_node: Node = serde_json::from_value(tree.clone())
            .map_err(|e| format!("invalid conditions tree ({e})"))?;
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
    // Build a cubiomes biome backend ONCE per search, reused across seeds and
    // all biome leaves. Only when the spec actually contains biome leaves —
    // pure-structure searches stay on the cheap structure-only evaluator path.
    let needs_biomes = conditions::has_biome_leaves(&compiled);
    let mut biome_backend = if needs_biomes {
        match BiomeBackend::from_strs(&spec.version, &spec.dimension, DEFAULT_Y) {
            Ok(b) => Some(b),
            Err(e) => {
                emit_completed(
                    &app,
                    &job_id,
                    0,
                    0,
                    "error",
                    Some(format!("biome backend init failed: {e}")),
                );
                return;
            }
        }
    } else {
        None
    };

    let report_match = |seed: i64, scanned_at: u64, matches_so_far: u64| {
        let mut exactness = HashMap::new();
        exactness.insert("structures".to_string(), "exact".to_string());
        if needs_biomes {
            exactness.insert("biomes".to_string(), "exact".to_string());
        }
        let report = SeedReport {
            seed,
            edition: spec.edition.clone(),
            version: spec.version.clone(),
            dimension: spec.dimension.clone(),
            score: 0.0,
            matched_features: if needs_biomes {
                vec!["structure_conditions".into(), "biome_conditions".into()]
            } else {
                vec!["structure_conditions".into()]
            },
            exactness,
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
        let _ = scanned_at; // reserved for richer telemetry
        let _ = matches_so_far;
    };

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
            let hit = match biome_backend.as_mut() {
                Some(backend) => conditions::evaluate_with_biomes(&compiled, seed, backend),
                None => conditions::evaluate(&compiled, seed),
            };
            if hit {
                matches += 1;
                report_match(seed, scanned + i, matches);
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
        .map(|pos| {
            serde_json::json!({
                "block_x": pos.block_x(),
                "block_z": pos.block_z(),
            })
        });

    // First-ring stronghold positions (always 3 placements in ring 1).
    let strongholds: Vec<_> = iter_strongholds(seed, 1)
        .map(|pos| {
            serde_json::json!({
                "block_x": pos.block_x(),
                "block_z": pos.block_z(),
            })
        })
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

#[derive(Debug, Deserialize)]
struct TileRequest {
    seed: i64,
    #[serde(default = "default_version")]
    version: String,
    #[serde(default = "default_dimension")]
    dimension: String,
    /// Top-left block coordinate of the tile.
    x: i32,
    z: i32,
    /// cubiomes scale (1, 4, 16, 64, 256). Defaults to 1:4 (the standard
    /// overworld biome-map scale).
    #[serde(default = "default_scale")]
    scale: i32,
    /// Tile size in *scaled* pixels (so a 256-pixel tile at scale=4 covers
    /// 1024 blocks across).
    #[serde(default = "default_size")]
    sx: u32,
    #[serde(default = "default_size")]
    sz: u32,
}

fn default_version() -> String {
    "1.21".to_string()
}
fn default_dimension() -> String {
    "overworld".to_string()
}
fn default_scale() -> i32 {
    4
}
fn default_size() -> u32 {
    256
}

#[tauri::command]
fn render_tile(request: TileRequest) -> Result<serde_json::Value, String> {
    let mut backend = BiomeBackend::from_strs(&request.version, &request.dimension, DEFAULT_Y)?;
    let png_b64 = backend.render_tile_base64(
        request.seed,
        request.scale,
        request.x,
        request.z,
        request.sx,
        request.sz,
    )?;
    Ok(serde_json::json!({
        "png_base64": png_b64,
        "seed": request.seed,
        "version": request.version,
        "dimension": request.dimension,
        "scale": request.scale,
        "x": request.x,
        "z": request.z,
        "sx": request.sx,
        "sz": request.sz,
    }))
}

#[derive(Debug, Deserialize)]
struct StructuresInViewRequest {
    seed: i64,
    /// Structure types to query (e.g. ["village", "pillager_outpost"]).
    structures: Vec<String>,
    /// View rectangle in block coordinates.
    x: i32,
    z: i32,
    sx: u32,
    sz: u32,
}

#[derive(Debug, Serialize)]
struct StructurePinOut {
    structure: String,
    block_x: i32,
    block_z: i32,
}

#[tauri::command]
fn list_structures_in_view(
    request: StructuresInViewRequest,
) -> Result<Vec<StructurePinOut>, String> {
    // Use the diagonal as a generous search radius around the view centre so
    // we don't miss placements that just barely overlap the rectangle.
    let centre_x = request.x + (request.sx as i32) / 2;
    let centre_z = request.z + (request.sz as i32) / 2;
    let half_diag =
        (((request.sx as f64).powi(2) + (request.sz as f64).powi(2)).sqrt() / 2.0) as i32 + 16;

    let mut out: Vec<StructurePinOut> = Vec::new();
    for name in &request.structures {
        let kind =
            StructureType::from_name(name).ok_or_else(|| format!("unknown structure {name:?}"))?;
        if kind == StructureType::Stronghold {
            // Ring 1 strongholds (always 3) — only emit those inside the view.
            for pos in iter_strongholds(request.seed, 1) {
                if pos.block_x() >= request.x
                    && pos.block_x() < request.x + request.sx as i32
                    && pos.block_z() >= request.z
                    && pos.block_z() < request.z + request.sz as i32
                {
                    out.push(StructurePinOut {
                        structure: name.clone(),
                        block_x: pos.block_x(),
                        block_z: pos.block_z(),
                    });
                }
            }
            continue;
        }
        for pos in iter_structures_in_radius(kind, request.seed, centre_x, centre_z, half_diag) {
            if pos.block_x() >= request.x
                && pos.block_x() < request.x + request.sx as i32
                && pos.block_z() >= request.z
                && pos.block_z() < request.z + request.sz as i32
            {
                out.push(StructurePinOut {
                    structure: name.clone(),
                    block_x: pos.block_x(),
                    block_z: pos.block_z(),
                });
            }
        }
    }
    Ok(out)
}

/// Read the world seed (and a few labels) out of a Minecraft `level.dat`.
/// Accepts the raw gzipped NBT bytes — the frontend reads the file with an
/// HTML `<input type="file">` and passes the bytes through, so we don't need a
/// file-dialog plugin and don't care where on disk the file lives.
#[tauri::command]
fn import_level_dat(bytes: Vec<u8>) -> Result<serde_json::Value, String> {
    use flate2::read::GzDecoder;
    use std::io::Read;

    // level.dat is gzipped NBT.
    let mut decoder = GzDecoder::new(&bytes[..]);
    let mut nbt_buf = Vec::new();
    decoder
        .read_to_end(&mut nbt_buf)
        .map_err(|e| format!("gunzip level.dat: {e}"))?;

    let val: fastnbt::Value =
        fastnbt::from_bytes(&nbt_buf).map_err(|e| format!("parse NBT: {e}"))?;

    let top = match &val {
        fastnbt::Value::Compound(m) => m,
        _ => return Err("level.dat root is not a Compound tag".into()),
    };
    let data = match top.get("Data") {
        Some(fastnbt::Value::Compound(m)) => m,
        _ => return Err("level.dat has no Data compound".into()),
    };

    // Modern (1.16+): Data.WorldGenSettings.seed (Long).
    // Legacy (pre-1.16): Data.RandomSeed (Long).
    let seed: i64 = match data.get("WorldGenSettings") {
        Some(fastnbt::Value::Compound(ws)) => match ws.get("seed") {
            Some(fastnbt::Value::Long(s)) => *s,
            _ => match data.get("RandomSeed") {
                Some(fastnbt::Value::Long(s)) => *s,
                _ => return Err("no seed found (WorldGenSettings.seed or RandomSeed)".into()),
            },
        },
        _ => match data.get("RandomSeed") {
            Some(fastnbt::Value::Long(s)) => *s,
            _ => return Err("no seed found (WorldGenSettings.seed or RandomSeed)".into()),
        },
    };

    // Optional: friendly labels — version name and world level name.
    let version_name = match data.get("Version") {
        Some(fastnbt::Value::Compound(v)) => match v.get("Name") {
            Some(fastnbt::Value::String(s)) => Some(s.clone()),
            _ => None,
        },
        _ => None,
    };
    let level_name = match data.get("LevelName") {
        Some(fastnbt::Value::String(s)) => Some(s.clone()),
        _ => None,
    };

    Ok(serde_json::json!({
        "seed": seed,
        "version_name": version_name,
        "level_name": level_name,
    }))
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
    // WebKitGTK 2.42+ ships a new DMA-BUF renderer that crashes the WebView on
    // a number of Wayland compositors with "Gdk-Message: Error 71 (Protocol
    // error) dispatching to Wayland display." Disabling it falls back to the
    // older shared-memory path, which is universally compatible. Safe / no-op
    // on X11 and on builds where the new renderer works fine; only set if the
    // user hasn't already chosen a value.
    #[cfg(target_os = "linux")]
    {
        if std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
            std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
        }
    }

    tauri::Builder::default()
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            start_search,
            pause_search,
            resume_search,
            cancel_search,
            analyze_seed,
            render_tile,
            list_structures_in_view,
            import_level_dat,
            export_results
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
