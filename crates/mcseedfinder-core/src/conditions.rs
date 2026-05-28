//! Native fast-path evaluator for the **structure-only** subset of the Python
//! `conditions` tree (logic gates over `NearbyStructure` / `StructureCluster`).
//!
//! Biome conditions are intentionally out of scope here — they live behind the
//! cubiomes backend in [`crate::biomes`] and stay on the Python evaluation path
//! for now. The split mirrors the planned prefilter/confirm boundary: this
//! evaluator is the cheap, branchless-friendly structure RNG prefilter that a
//! future wgpu compute shader (Phase 6) can replace wholesale without touching
//! the (biome) confirmation layer.
//!
//! The Python side serialises the structure-only sub-tree to JSON and passes
//! it as a string over PyO3; this module deserialises it once per worker and
//! evaluates it across a seed range.

use std::collections::HashSet;

use serde::Deserialize;

#[cfg(feature = "biomes")]
use crate::biomes::BiomeBackend;
use crate::structures::{
    count_structures_in_radius, has_structure_in_radius, StructureRequirement, StructureType,
};

/// Sampling grid count used by the spawn-biome predicate (matches the Python
/// `SpawnBiome.samples_per_axis` default).
#[cfg(feature = "biomes")]
const SPAWN_BIOME_SAMPLES_PER_AXIS: i32 = 5;

/// Wire-format tree node. Tagged by `type` so it mirrors the Python schema 1:1.
/// Biome leaves carry numeric cubiomes biome IDs (not names) — name resolution
/// happens on the frontend / caller side using the project's biome catalog.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Node {
    NearbyStructure {
        structure: String,
        max_distance: i32,
        #[serde(default)]
        centre_x: i32,
        #[serde(default)]
        centre_z: i32,
    },
    Cluster {
        structures: Vec<String>,
        max_distance: i32,
        min_count: u32,
        #[serde(default)]
        centre_x: i32,
        #[serde(default)]
        centre_z: i32,
    },
    SpawnBiome {
        biomes: Vec<i32>,
        #[serde(default = "default_spawn_radius")]
        spawn_radius: i32,
    },
    NearbyBiomes {
        biomes: Vec<i32>,
        #[serde(default = "default_nearby_biomes_radius")]
        radius: i32,
        #[serde(default, rename = "all")]
        all_required: bool,
        #[serde(default = "default_biome_samples")]
        samples_per_axis: u32,
    },
    BiomeArea {
        biomes: Vec<i32>,
        #[serde(default = "default_biome_area_radius")]
        radius: i32,
        #[serde(default = "default_biome_samples")]
        samples_per_axis: u32,
        #[serde(default = "default_min_samples")]
        min_samples: u32,
        #[serde(default)]
        centre_x: i32,
        #[serde(default)]
        centre_z: i32,
    },
    AllOf {
        of: Vec<Node>,
    },
    AnyOf {
        of: Vec<Node>,
    },
    NoneOf {
        of: Vec<Node>,
    },
}

fn default_spawn_radius() -> i32 {
    64
}
fn default_nearby_biomes_radius() -> i32 {
    2000
}
fn default_biome_area_radius() -> i32 {
    1000
}
fn default_biome_samples() -> u32 {
    16
}
fn default_min_samples() -> u32 {
    8
}

/// Pre-validated, structure-name-resolved form of [`Node`]. Resolving structure
/// strings to [`StructureType`] once at compile time (and biome ID lists to
/// `HashSet`s) keeps the per-seed loop free of allocations and string
/// comparisons.
#[derive(Debug, Clone)]
pub enum CompiledNode {
    Nearby(StructureRequirement),
    Cluster {
        structures: Vec<StructureType>,
        max_distance: i32,
        min_count: u32,
        centre_x: i32,
        centre_z: i32,
    },
    SpawnBiome {
        biomes: HashSet<i32>,
        spawn_radius: i32,
    },
    NearbyBiomes {
        biomes: HashSet<i32>,
        radius: i32,
        all_required: bool,
        samples_per_axis: u32,
    },
    BiomeArea {
        biomes: HashSet<i32>,
        radius: i32,
        samples_per_axis: u32,
        min_samples: u32,
        centre_x: i32,
        centre_z: i32,
    },
    AllOf(Vec<CompiledNode>),
    AnyOf(Vec<CompiledNode>),
    NoneOf(Vec<CompiledNode>),
}

#[derive(Debug)]
pub enum CompileError {
    UnknownStructure(String),
    EmptyGroup(&'static str),
    EmptyCluster,
    NegativeDistance,
    ClusterMinCountZero,
    EmptyBiomeSet(&'static str),
    InvalidMinSamples,
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::UnknownStructure(s) => write!(f, "unknown structure {s:?}"),
            CompileError::EmptyGroup(s) => write!(f, "{s} group has no children"),
            CompileError::EmptyCluster => write!(f, "cluster has empty structure list"),
            CompileError::NegativeDistance => write!(f, "max_distance must be non-negative"),
            CompileError::ClusterMinCountZero => write!(f, "cluster min_count must be >= 1"),
            CompileError::EmptyBiomeSet(kind) => write!(f, "{kind} requires a non-empty biome set"),
            CompileError::InvalidMinSamples => {
                write!(
                    f,
                    "biome_area min_samples must be within [1, samples_per_axis^2]"
                )
            }
        }
    }
}

impl std::error::Error for CompileError {}

/// Resolve all structure names to [`StructureType`] and validate up-front so the
/// per-seed loop can never hit a "unknown structure" branch.
pub fn compile(node: &Node) -> Result<CompiledNode, CompileError> {
    fn parse_struct(name: &str) -> Result<StructureType, CompileError> {
        StructureType::from_name(name)
            .ok_or_else(|| CompileError::UnknownStructure(name.to_string()))
    }
    match node {
        Node::NearbyStructure {
            structure,
            max_distance,
            centre_x,
            centre_z,
        } => {
            if *max_distance < 0 {
                return Err(CompileError::NegativeDistance);
            }
            Ok(CompiledNode::Nearby(StructureRequirement {
                structure: parse_struct(structure)?,
                max_distance: *max_distance,
                centre_x: *centre_x,
                centre_z: *centre_z,
            }))
        }
        Node::Cluster {
            structures,
            max_distance,
            min_count,
            centre_x,
            centre_z,
        } => {
            if structures.is_empty() {
                return Err(CompileError::EmptyCluster);
            }
            if *min_count == 0 {
                return Err(CompileError::ClusterMinCountZero);
            }
            if *max_distance < 0 {
                return Err(CompileError::NegativeDistance);
            }
            let resolved = structures
                .iter()
                .map(|s| parse_struct(s))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(CompiledNode::Cluster {
                structures: resolved,
                max_distance: *max_distance,
                min_count: *min_count,
                centre_x: *centre_x,
                centre_z: *centre_z,
            })
        }
        Node::SpawnBiome {
            biomes,
            spawn_radius,
        } => {
            if biomes.is_empty() {
                return Err(CompileError::EmptyBiomeSet("spawn_biome"));
            }
            Ok(CompiledNode::SpawnBiome {
                biomes: biomes.iter().copied().collect(),
                spawn_radius: *spawn_radius,
            })
        }
        Node::NearbyBiomes {
            biomes,
            radius,
            all_required,
            samples_per_axis,
        } => {
            if biomes.is_empty() {
                return Err(CompileError::EmptyBiomeSet("nearby_biomes"));
            }
            Ok(CompiledNode::NearbyBiomes {
                biomes: biomes.iter().copied().collect(),
                radius: *radius,
                all_required: *all_required,
                samples_per_axis: *samples_per_axis,
            })
        }
        Node::BiomeArea {
            biomes,
            radius,
            samples_per_axis,
            min_samples,
            centre_x,
            centre_z,
        } => {
            if biomes.is_empty() {
                return Err(CompileError::EmptyBiomeSet("biome_area"));
            }
            let total = (*samples_per_axis).saturating_mul(*samples_per_axis);
            if *min_samples == 0 || *min_samples > total {
                return Err(CompileError::InvalidMinSamples);
            }
            Ok(CompiledNode::BiomeArea {
                biomes: biomes.iter().copied().collect(),
                radius: *radius,
                samples_per_axis: *samples_per_axis,
                min_samples: *min_samples,
                centre_x: *centre_x,
                centre_z: *centre_z,
            })
        }
        Node::AllOf { of } => {
            if of.is_empty() {
                return Err(CompileError::EmptyGroup("all_of"));
            }
            of.iter()
                .map(compile)
                .collect::<Result<Vec<_>, _>>()
                .map(CompiledNode::AllOf)
        }
        Node::AnyOf { of } => {
            if of.is_empty() {
                return Err(CompileError::EmptyGroup("any_of"));
            }
            of.iter()
                .map(compile)
                .collect::<Result<Vec<_>, _>>()
                .map(CompiledNode::AnyOf)
        }
        Node::NoneOf { of } => {
            if of.is_empty() {
                return Err(CompileError::EmptyGroup("none_of"));
            }
            of.iter()
                .map(compile)
                .collect::<Result<Vec<_>, _>>()
                .map(CompiledNode::NoneOf)
        }
    }
}

/// True if `node` (or any descendant) is a biome leaf. Callers use this to
/// pick the right evaluator: structure-only paths can stay on the cheap
/// `evaluate` route; biome-touching paths need `evaluate_with_biomes` plus a
/// `BiomeBackend`.
pub fn has_biome_leaves(node: &CompiledNode) -> bool {
    match node {
        CompiledNode::SpawnBiome { .. }
        | CompiledNode::NearbyBiomes { .. }
        | CompiledNode::BiomeArea { .. } => true,
        CompiledNode::AllOf(c) | CompiledNode::AnyOf(c) | CompiledNode::NoneOf(c) => {
            c.iter().any(has_biome_leaves)
        }
        CompiledNode::Nearby(_) | CompiledNode::Cluster { .. } => false,
    }
}

/// Mirrors the Python `StructureCluster` semantics: at least `min_count`
/// placements (totaled across the structure list) within `max_distance` of
/// `(cx, cz)`. Counts come from [`count_structures_in_radius`] so this and
/// "in radius" predicates always agree on whether any single placement is in
/// range.
fn cluster_holds(
    seed: i64,
    structures: &[StructureType],
    max_distance: i32,
    min_count: u32,
    cx: i32,
    cz: i32,
) -> bool {
    let mut hits: u32 = 0;
    for &kind in structures {
        let req = StructureRequirement {
            structure: kind,
            max_distance,
            centre_x: cx,
            centre_z: cz,
        };
        hits = hits.saturating_add(count_structures_in_radius(seed, &req));
        if hits >= min_count {
            return true;
        }
    }
    false
}

/// Evaluate a compiled tree against a single seed (structure-only).
///
/// Biome leaves require a [`BiomeBackend`] — call [`has_biome_leaves`] first
/// and route to [`evaluate_with_biomes`] when it returns `true`. This function
/// panics on biome leaves (a programmer error, not a search-time condition).
pub fn evaluate(node: &CompiledNode, seed: i64) -> bool {
    match node {
        CompiledNode::Nearby(req) => has_structure_in_radius(seed, req),
        CompiledNode::Cluster {
            structures,
            max_distance,
            min_count,
            centre_x,
            centre_z,
        } => cluster_holds(
            seed,
            structures,
            *max_distance,
            *min_count,
            *centre_x,
            *centre_z,
        ),
        CompiledNode::SpawnBiome { .. }
        | CompiledNode::NearbyBiomes { .. }
        | CompiledNode::BiomeArea { .. } => panic!(
            "biome leaf reached structure-only evaluate(); call has_biome_leaves \
             and route to evaluate_with_biomes"
        ),
        CompiledNode::AllOf(children) => children.iter().all(|c| evaluate(c, seed)),
        CompiledNode::AnyOf(children) => children.iter().any(|c| evaluate(c, seed)),
        CompiledNode::NoneOf(children) => !children.iter().any(|c| evaluate(c, seed)),
    }
}

/// Convenience wrapper: filter a contiguous seed range, returning matches.
/// The seed-loop is intentionally simple and branch-predictable — the future
/// wgpu prefilter will replace this loop without touching the evaluator above.
pub fn find_matches_range(start_seed: i64, count: u64, root: &CompiledNode) -> Vec<i64> {
    (0..count)
        .filter_map(|offset| {
            let seed = start_seed.wrapping_add(offset as i64);
            evaluate(root, seed).then_some(seed)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Biome-aware evaluator. The cubiomes-backed BiomeBackend lives in
// `crate::biomes` and is only compiled with the `biomes` feature, so the
// biome-aware code paths are gated on the same feature.
// ---------------------------------------------------------------------------

#[cfg(feature = "biomes")]
fn spawn_biome_holds(
    biomes: &HashSet<i32>,
    spawn_radius: i32,
    seed: i64,
    backend: &mut BiomeBackend,
) -> bool {
    if spawn_radius <= 0 {
        return biomes.contains(&backend.get_biome(seed, 0, 0));
    }
    let samples = SPAWN_BIOME_SAMPLES_PER_AXIS;
    let step = std::cmp::max(1, (2 * spawn_radius) / (samples - 1));
    for ix in 0..samples {
        let x = -spawn_radius + ix * step;
        for iz in 0..samples {
            let z = -spawn_radius + iz * step;
            if biomes.contains(&backend.get_biome(seed, x, z)) {
                return true;
            }
        }
    }
    false
}

#[cfg(feature = "biomes")]
fn nearby_biomes_holds(
    target: &HashSet<i32>,
    radius: i32,
    all_required: bool,
    samples_per_axis: u32,
    seed: i64,
    backend: &mut BiomeBackend,
) -> bool {
    let samples = samples_per_axis.max(1) as i32;
    let step = std::cmp::max(1, (2 * radius) / std::cmp::max(1, samples - 1));
    let mut found: HashSet<i32> = HashSet::new();
    for ix in 0..samples {
        let x = -radius + ix * step;
        for iz in 0..samples {
            let z = -radius + iz * step;
            let bid = backend.get_biome(seed, x, z);
            if target.contains(&bid) {
                if !all_required {
                    return true;
                }
                found.insert(bid);
                if &found == target {
                    return true;
                }
            }
        }
    }
    if all_required {
        &found == target
    } else {
        false
    }
}

#[cfg(feature = "biomes")]
fn biome_area_holds(
    target: &HashSet<i32>,
    radius: i32,
    samples_per_axis: u32,
    min_samples: u32,
    centre_x: i32,
    centre_z: i32,
    seed: i64,
    backend: &mut BiomeBackend,
) -> bool {
    let samples = samples_per_axis.max(1) as i32;
    let step = std::cmp::max(1, (2 * radius) / std::cmp::max(1, samples - 1));
    let mut hits: u32 = 0;
    for ix in 0..samples {
        let x = centre_x - radius + ix * step;
        for iz in 0..samples {
            let z = centre_z - radius + iz * step;
            if target.contains(&backend.get_biome(seed, x, z)) {
                hits += 1;
                if hits >= min_samples {
                    return true;
                }
            }
        }
    }
    false
}

/// Evaluate a compiled tree against a single seed, with biome support.
/// Requires a [`BiomeBackend`] — caller is responsible for re-using a single
/// backend across seeds (the backend re-applies the seed internally only when
/// it changes).
#[cfg(feature = "biomes")]
pub fn evaluate_with_biomes(node: &CompiledNode, seed: i64, backend: &mut BiomeBackend) -> bool {
    match node {
        CompiledNode::Nearby(req) => has_structure_in_radius(seed, req),
        CompiledNode::Cluster {
            structures,
            max_distance,
            min_count,
            centre_x,
            centre_z,
        } => cluster_holds(
            seed,
            structures,
            *max_distance,
            *min_count,
            *centre_x,
            *centre_z,
        ),
        CompiledNode::SpawnBiome {
            biomes,
            spawn_radius,
        } => spawn_biome_holds(biomes, *spawn_radius, seed, backend),
        CompiledNode::NearbyBiomes {
            biomes,
            radius,
            all_required,
            samples_per_axis,
        } => nearby_biomes_holds(
            biomes,
            *radius,
            *all_required,
            *samples_per_axis,
            seed,
            backend,
        ),
        CompiledNode::BiomeArea {
            biomes,
            radius,
            samples_per_axis,
            min_samples,
            centre_x,
            centre_z,
        } => biome_area_holds(
            biomes,
            *radius,
            *samples_per_axis,
            *min_samples,
            *centre_x,
            *centre_z,
            seed,
            backend,
        ),
        CompiledNode::AllOf(children) => children
            .iter()
            .all(|c| evaluate_with_biomes(c, seed, backend)),
        CompiledNode::AnyOf(children) => children
            .iter()
            .any(|c| evaluate_with_biomes(c, seed, backend)),
        CompiledNode::NoneOf(children) => !children
            .iter()
            .any(|c| evaluate_with_biomes(c, seed, backend)),
    }
}

/// Filter a contiguous seed range with biome-aware evaluation. Use this when
/// [`has_biome_leaves`] returns `true` for the compiled tree.
#[cfg(feature = "biomes")]
pub fn find_matches_range_with_biomes(
    start_seed: i64,
    count: u64,
    root: &CompiledNode,
    backend: &mut BiomeBackend,
) -> Vec<i64> {
    (0..count)
        .filter_map(|offset| {
            let seed = start_seed.wrapping_add(offset as i64);
            evaluate_with_biomes(root, seed, backend).then_some(seed)
        })
        .collect()
}

#[cfg(all(test, feature = "biomes"))]
mod biome_eval_tests {
    use super::*;
    use crate::biomes::{BiomeBackend, DEFAULT_Y};

    fn backend() -> BiomeBackend {
        BiomeBackend::from_strs("1.21", "overworld", DEFAULT_Y).expect("backend")
    }

    #[test]
    fn has_biome_leaves_detects_nested() {
        let plain = compile(&Node::NearbyStructure {
            structure: "village".into(),
            max_distance: 100,
            centre_x: 0,
            centre_z: 0,
        })
        .unwrap();
        assert!(!has_biome_leaves(&plain));

        let with_biome = compile(&Node::AllOf {
            of: vec![
                Node::NearbyStructure {
                    structure: "village".into(),
                    max_distance: 100,
                    centre_x: 0,
                    centre_z: 0,
                },
                Node::SpawnBiome {
                    biomes: vec![0],
                    spawn_radius: 0,
                },
            ],
        })
        .unwrap();
        assert!(has_biome_leaves(&with_biome));
    }

    // Mirrors the existing Python test `test_provider_marks_biomes_exact`:
    // seed 12345 has ocean (id 0) at origin in 1.21; seed 1 has deep_ocean
    // (24), not ocean. The Rust biome-aware evaluator must agree.
    #[test]
    fn spawn_biome_matches_known_ocean_seed() {
        let compiled = compile(&Node::SpawnBiome {
            biomes: vec![0],
            spawn_radius: 0,
        })
        .unwrap();
        let mut b = backend();
        assert!(evaluate_with_biomes(&compiled, 12345, &mut b));
        assert!(!evaluate_with_biomes(&compiled, 1, &mut b));
    }

    #[test]
    fn biome_area_finds_ocean_neighbourhood_for_seed_12345() {
        // Same shape as the Python `test_biome_area_finds_ocean` test:
        // ocean variants within 1000 blocks of origin, 16x16 grid, >=8 hits.
        let compiled = compile(&Node::BiomeArea {
            biomes: vec![0, 24, 45, 46], // ocean, deep_ocean, lukewarm_ocean, cold_ocean
            radius: 1000,
            samples_per_axis: 16,
            min_samples: 8,
            centre_x: 0,
            centre_z: 0,
        })
        .unwrap();
        let mut b = backend();
        assert!(evaluate_with_biomes(&compiled, 12345, &mut b));
    }

    #[test]
    fn structure_only_evaluator_panics_on_biome_leaf() {
        let compiled = compile(&Node::SpawnBiome {
            biomes: vec![0],
            spawn_radius: 0,
        })
        .unwrap();
        // The structure-only path is a programmer error for biome leaves —
        // assert it panics so callers must use has_biome_leaves before routing.
        let result = std::panic::catch_unwind(|| evaluate(&compiled, 1));
        assert!(result.is_err());
    }

    #[test]
    fn empty_biome_set_rejected_at_compile() {
        let err = compile(&Node::SpawnBiome {
            biomes: vec![],
            spawn_radius: 0,
        });
        assert!(matches!(err, Err(CompileError::EmptyBiomeSet(_))));
    }
}
