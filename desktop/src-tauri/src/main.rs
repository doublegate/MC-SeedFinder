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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State};
use uuid::Uuid;

// The mcseedfinder-core lib is named `_native` so the pyo3 macro produces the
// right `PyInit__native` symbol for the Python extension; from a Rust consumer
// it just looks unusual in `use` paths.
use _native::biomes::{BiomeBackend, DEFAULT_Y};
use _native::conditions::{self, CompiledNode, Node};
use _native::gpu::{Combinator, GpuPredicate, GpuSearchSpec, GpuSearcher, MAX_PREDICATES};
use _native::structures::{
    is_slime_chunk, iter_strongholds, iter_structures_in_radius, StructureRequirement,
    StructureType,
};

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
    /// Lazy GPU searcher. First search to qualify (single NearbyStructure leaf,
    /// no biomes) attempts to init wgpu; subsequent searches reuse the result.
    /// `Outer = "init has been attempted"`, `inner = "did it succeed"`.
    gpu: OnceLock<Option<GpuSearcher>>,
    /// Pool of warm `BiomeBackend` instances keyed by (version, dimension).
    /// Building one calls cubiomes' `setupGenerator`, which is heavyweight —
    /// re-running it per tile / per analyze caused multi-second freezes when
    /// rapidly panning or clicking between matches. The pool amortises that
    /// cost to one-time-per-(version,dimension).
    biome_pool: Mutex<HashMap<(String, String), Vec<BiomeBackend>>>,
    /// Per-kind monotonic request counters. Each command claims a sequence
    /// number on entry; if a newer request bumps the counter past that
    /// sequence number before the command finishes its expensive work, it
    /// bails out early. Net effect: rapid pan/click only does the LATEST
    /// request's work, even though every click already issued an invoke.
    /// Each kind has its own counter so unrelated commands don't cancel
    /// each other (a fresh tile fetch shouldn't kill a still-running analyze).
    tile_counter: AtomicU64,
    pins_counter: AtomicU64,
    analyze_counter: AtomicU64,
}

/// Maximum cached BiomeBackends per (version, dimension). Concurrent tile
/// requests grab from this pool in parallel; if more than this number are
/// in-flight at once, extras are allocated and dropped on release rather
/// than retained. 4 covers foreground + 3 prefetches comfortably.
const BIOME_POOL_MAX: usize = 4;

fn acquire_biome_backend(
    state: &AppState,
    version: &str,
    dimension: &str,
) -> Result<BiomeBackend, String> {
    let key = (version.to_string(), dimension.to_string());
    {
        let mut guard = state
            .biome_pool
            .lock()
            .map_err(|_| "biome pool lock poisoned".to_string())?;
        if let Some(vec) = guard.get_mut(&key) {
            if let Some(b) = vec.pop() {
                return Ok(b);
            }
        }
        // Drop the lock before the slow setupGenerator call so concurrent
        // requests can keep grabbing pre-warmed instances.
    }
    BiomeBackend::from_strs(version, dimension, DEFAULT_Y)
}

fn release_biome_backend(state: &AppState, version: &str, dimension: &str, backend: BiomeBackend) {
    let key = (version.to_string(), dimension.to_string());
    if let Ok(mut guard) = state.biome_pool.lock() {
        let vec = guard.entry(key).or_default();
        if vec.len() < BIOME_POOL_MAX {
            vec.push(backend);
        }
        // Else: drop. Excess backends release their cubiomes allocation.
    }
}

/// Sentinel string a command returns when it's been superseded by a newer
/// request of the same kind. React-side handlers silently ignore promise
/// rejections matching this so the UI doesn't render a spurious error.
const SUPERSEDED: &str = "superseded";

/// True iff the counter has advanced past `my_seq` since this request started
/// — meaning a newer request is on the way and our work would be thrown out
/// anyway. Cheap (one relaxed atomic load). Call at every checkpoint a slow
/// step could be skipped: backend acquire, pre-cubiomes, post-cubiomes.
fn superseded(counter: &AtomicU64, my_seq: u64) -> bool {
    counter.load(Ordering::Relaxed) > my_seq
}

#[derive(Default)]
struct JobInner {
    results: Vec<SeedReport>,
    cancel: Arc<AtomicBool>,
}

/// Translate a [`StructureRequirement`] into the GPU predicate shape.
/// Returns None for strongholds (different RNG path; not supported by the
/// kernel) or for an invalid (negative) max_distance.
fn req_to_predicate(req: &StructureRequirement) -> Option<GpuPredicate> {
    if req.structure == StructureType::Stronghold || req.max_distance < 0 {
        return None;
    }
    Some(GpuPredicate {
        structure: req.structure,
        max_distance: req.max_distance,
        centre_x: req.centre_x,
        centre_z: req.centre_z,
    })
}

/// Detect whether a compiled conditions tree is structure-only and shaped
/// such that the multi-predicate GPU kernel can evaluate it. Recognises:
///   - single `NearbyStructure` (possibly wrapped in single-child groups)
///     → any_of with 1 predicate
///   - `Cluster` → cluster combinator
///   - `AllOf` of N `NearbyStructure` leaves (N ≤ MAX_PREDICATES)
///   - `AnyOf` of N `NearbyStructure` leaves (N ≤ MAX_PREDICATES)
/// Anything else returns None and the search stays on the CPU path.
fn try_extract_gpu_spec(node: &CompiledNode) -> Option<(Combinator, Vec<GpuPredicate>)> {
    match node {
        CompiledNode::Nearby(req) => Some((Combinator::AnyOf, vec![req_to_predicate(req)?])),
        CompiledNode::Cluster {
            structures,
            max_distance,
            min_count,
            centre_x,
            centre_z,
        } => {
            if *max_distance < 0 || structures.is_empty() || structures.len() > MAX_PREDICATES {
                return None;
            }
            // Any stronghold disqualifies — not supported by the kernel.
            if structures.iter().any(|s| *s == StructureType::Stronghold) {
                return None;
            }
            let preds: Vec<GpuPredicate> = structures
                .iter()
                .map(|s| GpuPredicate {
                    structure: *s,
                    max_distance: *max_distance,
                    centre_x: *centre_x,
                    centre_z: *centre_z,
                })
                .collect();
            Some((
                Combinator::Cluster {
                    min_count: *min_count,
                },
                preds,
            ))
        }
        CompiledNode::AllOf(children) | CompiledNode::AnyOf(children) => {
            if children.len() == 1 {
                // Group wrapping one leaf; recurse so groups-around-clusters
                // also qualify.
                return try_extract_gpu_spec(&children[0]);
            }
            if children.is_empty() || children.len() > MAX_PREDICATES {
                return None;
            }
            // All children must be NearbyStructure leaves (the GPU kernel
            // doesn't support nested groups or clusters as predicates).
            let mut preds: Vec<GpuPredicate> = Vec::with_capacity(children.len());
            for c in children {
                match c {
                    CompiledNode::Nearby(req) => preds.push(req_to_predicate(req)?),
                    _ => return None,
                }
            }
            let comb = match node {
                CompiledNode::AllOf(_) => Combinator::AllOf,
                _ => Combinator::AnyOf,
            };
            Some((comb, preds))
        }
        _ => None,
    }
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

/// GPU fast-path. Chunks dispatches so the cancel flag can be checked between
/// each, and so a `search-progress` event fires at chunk boundaries (same
/// shape as the CPU path so the UI doesn't have to special-case it).
fn run_search_gpu(
    app: &AppHandle,
    job_id: &str,
    spec: &SearchSpec,
    combinator: Combinator,
    predicates: &[GpuPredicate],
    searcher: &GpuSearcher,
    cancel: &AtomicBool,
) {
    // 65k seeds per dispatch ≈ a few tens of ms on a modern GPU — small enough
    // for snappy cancellation, large enough that the per-dispatch overhead
    // (uniform write + map_async + poll) stays a small fraction of GPU time.
    const CHUNK: u64 = 65_536;
    let mut scanned: u64 = 0;
    let mut matches: u64 = 0;

    while scanned < spec.count {
        if cancel.load(Ordering::Relaxed) {
            emit_completed(app, job_id, scanned, matches, "cancelled", None);
            return;
        }
        let this_chunk = CHUNK.min(spec.count - scanned);
        let chunk_start = spec.start_seed.wrapping_add(scanned as i64);
        let chunk_matches = match searcher.find_matches(&GpuSearchSpec {
            start_seed: chunk_start,
            count: this_chunk,
            combinator,
            predicates,
        }) {
            Ok(v) => v,
            Err(e) => {
                emit_completed(
                    app,
                    job_id,
                    scanned,
                    matches,
                    "error",
                    Some(format!("gpu dispatch failed: {e}")),
                );
                return;
            }
        };

        for seed in chunk_matches {
            let mut exactness = HashMap::new();
            exactness.insert("structures".to_string(), "exact".to_string());
            let report = SeedReport {
                seed,
                edition: spec.edition.clone(),
                version: spec.version.clone(),
                dimension: spec.dimension.clone(),
                score: 0.0,
                matched_features: vec!["structure_conditions".into(), "gpu_prefilter".into()],
                exactness,
                warnings: Vec::new(),
            };
            if let Some(jobs) = app.try_state::<AppState>() {
                if let Ok(mut guard) = jobs.jobs.lock() {
                    if let Some(inner) = guard.get_mut(job_id) {
                        inner.results.push(report.clone());
                    }
                }
            }
            let _ = app.emit("search-match", &report);
            matches += 1;
            if matches >= spec.max_matches {
                let scanned_final = scanned + this_chunk;
                emit_completed(app, job_id, scanned_final, matches, "max_matches", None);
                return;
            }
        }

        scanned += this_chunk;
        let _ = app.emit(
            "search-progress",
            ProgressEvent {
                job_id: job_id.to_string(),
                scanned,
                matches,
            },
        );
    }

    emit_completed(app, job_id, scanned, matches, "range_exhausted", None);
}

fn run_search(
    app: AppHandle,
    job_id: String,
    spec: SearchSpec,
    compiled: CompiledNode,
    cancel: Arc<AtomicBool>,
) {
    // GPU fast-path: route to wgpu ONLY when the dispatch is large enough to
    // amortise the per-dispatch overhead (buffer-create + queue-submit +
    // map_async + poll, ~5-10 ms each) and the one-time init (~200-500 ms on
    // first use). Below this threshold the CPU evaluator wins comfortably —
    // it's a tight branch-predicted Rust loop with no IPC / GPU sync cost.
    // Empirically GPU starts paying off around half a million seeds; the
    // threshold below leaves headroom and keeps interactive UI searches on
    // the fast CPU path.
    const GPU_MIN_COUNT: u64 = 500_000;
    if spec.count >= GPU_MIN_COUNT && !conditions::has_biome_leaves(&compiled) {
        if let Some((combinator, predicates)) = try_extract_gpu_spec(&compiled) {
            let state = app.try_state::<AppState>();
            let searcher = state
                .as_ref()
                .and_then(|s| s.gpu.get_or_init(GpuSearcher::try_new).as_ref());
            if let Some(searcher) = searcher {
                run_search_gpu(
                    &app,
                    &job_id,
                    &spec,
                    combinator,
                    &predicates,
                    searcher,
                    &cancel,
                );
                return;
            }
        }
    }

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
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let my_seq = state.analyze_counter.fetch_add(1, Ordering::Relaxed) + 1;
    if superseded(&state.analyze_counter, my_seq) {
        return Err(SUPERSEDED.into());
    }
    let version = version.unwrap_or_else(|| "1.21".to_string());
    let dimension = dimension.unwrap_or_else(|| "overworld".to_string());

    // Pool-acquired backend — avoids the heavy setupGenerator call per analyze
    // when the user clicks through many results in a row.
    let mut backend = acquire_biome_backend(&state, &version, &dimension)?;
    if superseded(&state.analyze_counter, my_seq) {
        release_biome_backend(&state, &version, &dimension, backend);
        return Err(SUPERSEDED.into());
    }
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

    let result = serde_json::json!({
        "seed": seed,
        "version": version,
        "dimension": dimension,
        "origin_biome_id": origin_biome,
        "origin_biome_exact": true,
        "nearest_village": nearest_village,
        "strongholds": strongholds,
    });
    release_biome_backend(&state, &version, &dimension, backend);
    Ok(result)
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
    /// Minecraft block Y at which to sample biomes. Defaults to sea level
    /// (63). For 1.18+ valid range is `[-64, 319]`; the frontend's
    /// wheel-driven Y scrubber drives this. Older callers that omit `y`
    /// keep the previous fixed-sea-level behaviour.
    #[serde(default = "default_y")]
    y: i32,
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
fn default_y() -> i32 {
    63
}

#[tauri::command]
fn render_tile(
    request: TileRequest,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    // Claim the latest sequence number for the "tile" kind. Any earlier tile
    // request that hasn't reached an expensive step yet will see the bumped
    // counter and bail out. Cancellation is "best effort" — we can't
    // interrupt cubiomes mid-call (it's a single C function), so the check
    // points are: before backend acquire, before the cubiomes fill, and after.
    let my_seq = state.tile_counter.fetch_add(1, Ordering::Relaxed) + 1;
    if superseded(&state.tile_counter, my_seq) {
        return Err(SUPERSEDED.into());
    }

    // Pool-acquired backend — avoids re-running cubiomes' setupGenerator on
    // every tile.
    let mut backend = acquire_biome_backend(&state, &request.version, &request.dimension)?;
    if superseded(&state.tile_counter, my_seq) {
        release_biome_backend(&state, &request.version, &request.dimension, backend);
        return Err(SUPERSEDED.into());
    }

    let png_b64 = backend.render_tile_base64(
        request.seed,
        request.scale,
        request.x,
        request.z,
        request.sx,
        request.sz,
    );
    let version = request.version.clone();
    let dimension = request.dimension.clone();
    release_biome_backend(&state, &version, &dimension, backend);

    if superseded(&state.tile_counter, my_seq) {
        return Err(SUPERSEDED.into());
    }
    png_b64.map(|png| {
        serde_json::json!({
            "png_base64": png,
            "seed": request.seed,
            "version": request.version,
            "dimension": request.dimension,
            "scale": request.scale,
            "x": request.x,
            "z": request.z,
            "sx": request.sx,
            "sz": request.sz,
        })
    })
}

/// Faster variant of [`render_tile`]: returns the raw RGBA pixel buffer
/// instead of a PNG. The frontend blits it via Canvas2D `putImageData`,
/// skipping PNG encode (~15-25 ms), base64 (~5 ms), and the browser's PNG
/// decode (~5-10 ms) — together a ~30-45 ms speedup per tile.
///
/// Wire shape: returns `bytes` (Vec<u8>, length `sx * sz * 4`) alongside
/// the same metadata `render_tile` returns. Tauri serialises Vec<u8> as a
/// JSON array of numbers; for the typical 768x576 tile (~1.7 MB raw) that
/// adds ~3-5 MB of JSON text, but the in-process IPC handles it quickly
/// and the net is still well ahead of the PNG path.
#[tauri::command]
fn render_tile_rgba_cmd(
    request: TileRequest,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let my_seq = state.tile_counter.fetch_add(1, Ordering::Relaxed) + 1;
    if superseded(&state.tile_counter, my_seq) {
        return Err(SUPERSEDED.into());
    }
    let mut backend = acquire_biome_backend(&state, &request.version, &request.dimension)?;
    if superseded(&state.tile_counter, my_seq) {
        release_biome_backend(&state, &request.version, &request.dimension, backend);
        return Err(SUPERSEDED.into());
    }
    let result = backend.render_tile_rgba_and_ids_at_y(
        request.seed,
        request.scale,
        request.x,
        request.z,
        request.sx,
        request.sz,
        request.y,
    );
    let version = request.version.clone();
    let dimension = request.dimension.clone();
    release_biome_backend(&state, &version, &dimension, backend);

    if superseded(&state.tile_counter, my_seq) {
        return Err(SUPERSEDED.into());
    }
    result.map(|(bytes, biome_ids)| {
        serde_json::json!({
            "bytes": bytes,
            // Per-cell cubiomes biome IDs (u8). Same row-major order as
            // `bytes`. Used by the 3D voxel renderer for per-instance
            // colours and by the cursor-hover readout for instant biome
            // name lookup with no IPC round-trip.
            "biome_ids": biome_ids,
            "seed": request.seed,
            "version": request.version,
            "dimension": request.dimension,
            "scale": request.scale,
            "x": request.x,
            "z": request.z,
            "sx": request.sx,
            "sz": request.sz,
            "y": request.y,
        })
    })
}

/// Heightmap request: same view rectangle as `TileRequest` but the
/// response is per-pixel surface-block-Y instead of biome RGBA. Used by
/// the 3D isometric view to extrude the ground plane.
///
/// **Approximate.** The heights come from cubiomes' `mapApproxHeight`,
/// which derives Y from the depth-spline output, not from Java's full
/// per-block surface noise. The response advertises this via the
/// `"exactness": "approximate"` field so the UI can label it.
#[derive(Debug, Deserialize)]
struct HeightTileRequest {
    seed: i64,
    #[serde(default = "default_version")]
    version: String,
    #[serde(default = "default_dimension")]
    dimension: String,
    /// Top-left block coordinate of the tile (matches `TileRequest`).
    x: i32,
    z: i32,
    /// Tile size in *scale-4* pixels (cubiomes' `mapApproxHeight`
    /// canonical unit). One pixel = 4 blocks. A 256-pixel tile covers
    /// 1024 blocks across.
    #[serde(default = "default_size")]
    sx: u32,
    #[serde(default = "default_size")]
    sz: u32,
}

#[tauri::command]
fn surface_height_tile_cmd(
    request: HeightTileRequest,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    // Share the tile_counter with `render_tile_rgba_cmd` — the front-end
    // fires both for the same pan/Y-scroll, so a newer pan should
    // supersede both together.
    let my_seq = state.tile_counter.fetch_add(1, Ordering::Relaxed) + 1;
    if superseded(&state.tile_counter, my_seq) {
        return Err(SUPERSEDED.into());
    }
    let mut backend = acquire_biome_backend(&state, &request.version, &request.dimension)?;
    if superseded(&state.tile_counter, my_seq) {
        release_biome_backend(&state, &request.version, &request.dimension, backend);
        return Err(SUPERSEDED.into());
    }
    // mapApproxHeight expects scale-4 coords. The Rust API mirrors that.
    let result = backend.surface_height_tile(
        request.seed,
        request.x >> 2,
        request.z >> 2,
        request.sx,
        request.sz,
    );
    let version = request.version.clone();
    let dimension = request.dimension.clone();
    release_biome_backend(&state, &version, &dimension, backend);

    if superseded(&state.tile_counter, my_seq) {
        return Err(SUPERSEDED.into());
    }
    result.map(|(heights, biome_ids)| {
        serde_json::json!({
            "heights": heights,
            "biome_ids": biome_ids,
            "exactness": "approximate",
            "seed": request.seed,
            "version": request.version,
            "dimension": request.dimension,
            "x": request.x,
            "z": request.z,
            "sx": request.sx,
            "sz": request.sz,
            "scale": 4,
        })
    })
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
    state: State<'_, AppState>,
) -> Result<Vec<StructurePinOut>, String> {
    let my_seq = state.pins_counter.fetch_add(1, Ordering::Relaxed) + 1;
    if superseded(&state.pins_counter, my_seq) {
        return Err(SUPERSEDED.into());
    }
    // Use the diagonal as a generous search radius around the view centre so
    // we don't miss placements that just barely overlap the rectangle.
    let centre_x = request.x + (request.sx as i32) / 2;
    let centre_z = request.z + (request.sz as i32) / 2;
    let half_diag =
        (((request.sx as f64).powi(2) + (request.sz as f64).powi(2)).sqrt() / 2.0) as i32 + 16;

    let mut out: Vec<StructurePinOut> = Vec::new();
    for name in &request.structures {
        if superseded(&state.pins_counter, my_seq) {
            return Err(SUPERSEDED.into());
        }
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

/// Wire-format for `list_slime_chunks_cmd` — view rectangle in BLOCK
/// coords; we convert to chunk bounds internally.
#[derive(Debug, Deserialize)]
struct SlimeChunksRequest {
    seed: i64,
    x: i32,
    z: i32,
    sx: u32,
    sz: u32,
}

/// Enumerate slime chunks whose chunk rect (`16×16 blocks each`) overlaps
/// the requested block-coord view rectangle. Returns `[chunk_x, chunk_z]`
/// pairs as flat arrays of i32. Dimension-independent — slime chunks are
/// a property of Java RNG mixing of (world_seed, chunk_x, chunk_z) only.
#[tauri::command]
fn list_slime_chunks_cmd(request: SlimeChunksRequest) -> Vec<[i32; 2]> {
    // Convert block bounds to chunk bounds (inclusive). `div_euclid`
    // handles negative coords correctly (Rust integer division rounds
    // toward zero, which is wrong for negative-block → chunk mapping).
    let cx0 = request.x.div_euclid(16);
    let cz0 = request.z.div_euclid(16);
    let cx1 = (request.x + request.sx as i32 - 1).div_euclid(16);
    let cz1 = (request.z + request.sz as i32 - 1).div_euclid(16);
    let mut out: Vec<[i32; 2]> = Vec::new();
    // The expected chunk count for a typical tile (1024×1024 blocks = 64×64
    // chunks = 4096 chunks) caps at ~410 slime chunks; allocating up front
    // avoids reallocation during the inner loop.
    let total = ((cx1 - cx0 + 1) as usize) * ((cz1 - cz0 + 1) as usize);
    out.reserve(total / 8);
    for cz in cz0..=cz1 {
        for cx in cx0..=cx1 {
            if is_slime_chunk(request.seed, cx, cz) {
                out.push([cx, cz]);
            }
        }
    }
    out
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
            render_tile_rgba_cmd,
            surface_height_tile_cmd,
            list_slime_chunks_cmd,
            list_structures_in_view,
            import_level_dat,
            export_results
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
