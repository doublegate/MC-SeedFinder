use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;
use tauri::State;
use uuid::Uuid;

#[derive(Debug, Clone, Deserialize, Serialize)]
struct SearchSpec {
    edition: String,
    version: String,
    dimension: String,
    count: u64,
    max_matches: u64,
    criteria: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
struct SeedReport {
    seed: i64,
    edition: String,
    version: String,
    dimension: String,
    score: f64,
}

#[derive(Default)]
struct AppState {
    jobs: Mutex<HashMap<String, Vec<SeedReport>>>,
}

#[tauri::command]
fn start_search(spec: SearchSpec, state: State<'_, AppState>) -> Result<String, String> {
    let job_id = Uuid::new_v4().to_string();
    let mut results = Vec::new();
    let max_matches = spec.max_matches.max(1).min(25);
    for offset in 0..max_matches {
        results.push(SeedReport {
            seed: offset as i64 + 1,
            edition: spec.edition.clone(),
            version: spec.version.clone(),
            dimension: spec.dimension.clone(),
            score: 0.0,
        });
    }
    state
        .jobs
        .lock()
        .map_err(|_| "job store lock poisoned".to_string())?
        .insert(job_id.clone(), results);
    Ok(job_id)
}

#[tauri::command]
fn pause_search(_job_id: String) -> Result<(), String> {
    Ok(())
}

#[tauri::command]
fn resume_search(_job_id: String) -> Result<(), String> {
    Ok(())
}

#[tauri::command]
fn cancel_search(_job_id: String) -> Result<(), String> {
    Ok(())
}

#[tauri::command]
fn analyze_seed(seed: i64) -> serde_json::Value {
    serde_json::json!({
        "seed": seed,
        "status": "candidate",
        "notes": ["desktop analyzer command contract is connected"]
    })
}

#[tauri::command]
fn render_tile(request: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "tile": request,
        "status": "placeholder"
    })
}

#[tauri::command]
fn import_level_dat(path: String) -> serde_json::Value {
    serde_json::json!({
        "path": path,
        "status": "unsupported"
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
    let results = jobs.get(&job_id).cloned().unwrap_or_default();
    if format == "plain" {
        let seeds = results
            .iter()
            .map(|result| result.seed.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        return Ok(serde_json::Value::String(seeds));
    }
    Ok(serde_json::to_value(results).map_err(|error| error.to_string())?)
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
