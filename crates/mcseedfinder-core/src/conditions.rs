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

use serde::Deserialize;

use crate::structures::{
    count_structures_in_radius, has_structure_in_radius, StructureRequirement, StructureType,
};

/// Wire-format tree node. Tagged by `type` so it mirrors the Python schema 1:1.
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

/// Pre-validated, structure-name-resolved form of [`Node`]. Resolving structure
/// strings to [`StructureType`] once at compile time keeps the per-seed loop
/// allocation-free and free of string comparisons.
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
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::UnknownStructure(s) => write!(f, "unknown structure {s:?}"),
            CompileError::EmptyGroup(s) => write!(f, "{s} group has no children"),
            CompileError::EmptyCluster => write!(f, "cluster has empty structure list"),
            CompileError::NegativeDistance => write!(f, "max_distance must be non-negative"),
            CompileError::ClusterMinCountZero => write!(f, "cluster min_count must be >= 1"),
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

/// Evaluate a compiled tree against a single seed.
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
