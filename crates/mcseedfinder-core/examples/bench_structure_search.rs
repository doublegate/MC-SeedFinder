use std::time::Instant;

use _native::{seed_matches_structure_requirements, StructureRequirement, StructureType};

fn main() {
    let count = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse::<u64>().ok())
        .unwrap_or(1_000_000);
    let requirement = StructureRequirement {
        structure: StructureType::Village,
        max_distance: 100,
        centre_x: 0,
        centre_z: 0,
    };
    let requirements = [requirement];

    let start = Instant::now();
    let matches = (0..count)
        .filter(|seed| seed_matches_structure_requirements(*seed as i64, &requirements))
        .count();
    let elapsed = start.elapsed();
    let rate = count as f64 / elapsed.as_secs_f64();

    println!("count={count}");
    println!("matches={matches}");
    println!("elapsed_seconds={:.6}", elapsed.as_secs_f64());
    println!("seeds_per_second={rate:.0}");
}
