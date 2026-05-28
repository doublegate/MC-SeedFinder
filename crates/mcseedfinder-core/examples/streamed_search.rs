//! Verifies the Phase 4a streamed-search behavior end to end without a Tauri
//! window. The chunked loop, cancellation flag, and event-firing pattern below
//! mirror `desktop/src-tauri/src/main.rs::run_search` 1-for-1, but the "event"
//! sink prints to stdout instead of calling `app.emit(...)`. If `MATCH` lines
//! stream out chunk by chunk and `COMPLETED` fires at the end, the desktop's
//! streamed events do the same — `app.emit` is a single-line side effect at
//! each of the same points.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use _native::conditions::{self, Node};

fn main() {
    // Tight radius so matches are sparse and PROGRESS/MATCH events interleave,
    // making the streaming behaviour visible. (Python evaluator confirms ~60
    // matches in seeds 1..5000 with this spec.)
    let tree = Node::NearbyStructure {
        structure: "village".to_string(),
        max_distance: 100,
        centre_x: 0,
        centre_z: 0,
    };
    let compiled = conditions::compile(&tree).expect("compile");

    let start_seed: i64 = 1;
    let count: u64 = 5_000;
    let max_matches: u64 = 10;
    let chunk: u64 = 1024;
    let cancel = Arc::new(AtomicBool::new(false));

    println!(
        "STARTED  start_seed={} count={} chunk={} max_matches={}",
        start_seed, count, chunk, max_matches
    );
    let t0 = Instant::now();
    let mut scanned: u64 = 0;
    let mut matches: u64 = 0;

    'outer: while scanned < count {
        if cancel.load(Ordering::Relaxed) {
            println!(
                "COMPLETED reason=cancelled scanned={} matches={}",
                scanned, matches
            );
            return;
        }
        let this_chunk = chunk.min(count - scanned);
        for i in 0..this_chunk {
            let seed = start_seed.wrapping_add((scanned + i) as i64);
            if conditions::evaluate(&compiled, seed) {
                println!("MATCH    seed={}", seed);
                matches += 1;
                if matches >= max_matches {
                    let scanned_final = scanned + i + 1;
                    println!(
                        "COMPLETED reason=max_matches scanned={} matches={} elapsed_ms={}",
                        scanned_final,
                        matches,
                        t0.elapsed().as_millis()
                    );
                    break 'outer;
                }
            }
        }
        scanned += this_chunk;
        println!("PROGRESS scanned={} matches={}", scanned, matches);
    }
    if matches < max_matches {
        println!(
            "COMPLETED reason=range_exhausted scanned={} matches={} elapsed_ms={}",
            scanned,
            matches,
            t0.elapsed().as_millis()
        );
    }
}
