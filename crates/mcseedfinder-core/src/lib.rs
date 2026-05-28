//! Accuracy-critical Minecraft seed search core.
//!
//! This crate starts the Rust-first engine path with deterministic Java RNG and
//! structure-placement math ported from the Python implementation. It is kept
//! dependency-free while the public API settles.

pub mod bedrock;
#[cfg(feature = "biomes")]
pub mod biomes;
pub mod conditions;
#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(all(feature = "gpu", feature = "biomes"))]
pub mod gpu_btree;
#[cfg(all(feature = "gpu", feature = "biomes"))]
pub mod gpu_climate;
#[cfg(all(feature = "gpu", feature = "biomes"))]
pub mod gpu_double_perlin;
#[cfg(all(feature = "gpu", feature = "biomes"))]
pub mod gpu_noise;
pub mod java_random;
pub mod structures;

pub use java_random::JavaRandom;
pub use structures::{
    get_structure_pos, iter_strongholds, seed_matches_structure_requirements, StructureConfig,
    StructurePos, StructureRequirement, StructureType,
};

#[cfg(feature = "pyo3")]
use pyo3::exceptions::{PyKeyError, PyValueError};
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

#[cfg(feature = "pyo3")]
#[pyclass(name = "JavaRandom")]
struct PyJavaRandom {
    inner: JavaRandom,
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl PyJavaRandom {
    #[new]
    fn new(seed: i64) -> Self {
        Self {
            inner: JavaRandom::new(seed),
        }
    }

    fn set_seed(&mut self, seed: i64) {
        self.inner.set_seed(seed);
    }

    fn next_int(&mut self) -> i32 {
        self.inner.next_int()
    }

    fn next_int_bound(&mut self, bound: i32) -> PyResult<i32> {
        if bound <= 0 {
            return Err(PyValueError::new_err("bound must be positive"));
        }
        Ok(self.inner.next_int_bound(bound))
    }

    fn next_long(&mut self) -> i64 {
        self.inner.next_long()
    }

    fn next_double(&mut self) -> f64 {
        self.inner.next_double()
    }
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn get_structure_pos_py(
    structure: &str,
    world_seed: i64,
    region_x: i64,
    region_z: i64,
) -> PyResult<(String, i32, i32)> {
    let structure_type = StructureType::from_name(structure)
        .ok_or_else(|| PyKeyError::new_err(format!("unknown structure {structure:?}")))?;
    if structure_type == StructureType::Stronghold {
        return Err(PyValueError::new_err("strongholds use iter_strongholds_py"));
    }
    let pos = get_structure_pos(structure_type, world_seed, region_x, region_z);
    Ok((pos.structure.name().to_string(), pos.chunk_x, pos.chunk_z))
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn iter_strongholds_py(world_seed: i64, max_rings: usize) -> PyResult<Vec<(String, i32, i32)>> {
    if !(1..=8).contains(&max_rings) {
        return Err(PyValueError::new_err("max_rings must be between 1 and 8"));
    }
    Ok(iter_strongholds(world_seed, max_rings)
        .map(|pos| (pos.structure.name().to_string(), pos.chunk_x, pos.chunk_z))
        .collect())
}

/// Exact biome backend exposed to Python. Implements the duck-typed
/// `get_biome(world_seed, x, z) -> int` contract that `BiomeLookup` expects, so
/// it slots into the criteria layer in place of the approximate Perlin backend.
#[cfg(all(feature = "pyo3", feature = "biomes"))]
#[pyclass(name = "CubiomesBiomeBackend")]
struct PyCubiomesBiomeBackend {
    inner: biomes::BiomeBackend,
}

#[cfg(all(feature = "pyo3", feature = "biomes"))]
#[pymethods]
impl PyCubiomesBiomeBackend {
    #[new]
    #[pyo3(signature = (version, dimension = "overworld", y = biomes::DEFAULT_Y))]
    fn new(version: &str, dimension: &str, y: i32) -> PyResult<Self> {
        let inner = biomes::BiomeBackend::from_strs(version, dimension, y)
            .map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// Numeric cubiomes biome ID at block coord `(x, z)`; -1 if unresolved.
    fn get_biome(&mut self, world_seed: i64, x: i32, z: i32) -> i32 {
        self.inner.get_biome(world_seed, x, z)
    }
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn find_structure_matches_range(
    start_seed: i64,
    count: u64,
    requirements: Vec<(String, i32, i32, i32)>,
) -> PyResult<Vec<i64>> {
    let parsed = requirements
        .into_iter()
        .map(|(name, max_distance, centre_x, centre_z)| {
            let structure = StructureType::from_name(&name)
                .ok_or_else(|| PyKeyError::new_err(format!("unknown structure {name:?}")))?;
            if max_distance < 0 {
                return Err(PyValueError::new_err("max_distance must be non-negative"));
            }
            Ok(StructureRequirement {
                structure,
                max_distance,
                centre_x,
                centre_z,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;

    let mut matches = Vec::new();
    for offset in 0..count {
        let seed = start_seed.wrapping_add(offset as i64);
        if seed_matches_structure_requirements(seed, &parsed) {
            matches.push(seed);
        }
    }
    Ok(matches)
}

/// Structure-only conditions-tree fast-path. Accepts the Python tree spec as a
/// JSON string, compiles it once (resolving structure names + validation), and
/// filters a contiguous seed range. Biome conditions are NOT supported here —
/// callers must keep those on the Python evaluation path.
#[cfg(feature = "pyo3")]
#[pyfunction]
fn find_tree_matches_range(start_seed: i64, count: u64, tree_json: &str) -> PyResult<Vec<i64>> {
    let parsed: conditions::Node = serde_json::from_str(tree_json)
        .map_err(|e| PyValueError::new_err(format!("invalid tree JSON: {e}")))?;
    let compiled = conditions::compile(&parsed).map_err(|e| match e {
        conditions::CompileError::UnknownStructure(s) => {
            PyKeyError::new_err(format!("unknown structure {s:?}"))
        }
        other => PyValueError::new_err(other.to_string()),
    })?;
    Ok(conditions::find_matches_range(start_seed, count, &compiled))
}

/// Convert a Bedrock text seed to its canonical i32 game-seed via Java's
/// `String.hashCode()`. Same algorithm Java and Bedrock both use when a
/// player types a non-numeric seed into the world creation screen.
#[cfg(feature = "pyo3")]
#[pyfunction]
fn bedrock_seed_from_string(text: &str) -> i32 {
    bedrock::seed_from_string(text)
}

#[cfg(feature = "pyo3")]
#[pymodule]
fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyJavaRandom>()?;
    m.add_function(wrap_pyfunction!(get_structure_pos_py, m)?)?;
    m.add_function(wrap_pyfunction!(iter_strongholds_py, m)?)?;
    m.add_function(wrap_pyfunction!(find_structure_matches_range, m)?)?;
    m.add_function(wrap_pyfunction!(find_tree_matches_range, m)?)?;
    m.add_function(wrap_pyfunction!(bedrock_seed_from_string, m)?)?;
    // Whether this build links cubiomes for exact biomes. Lets the Python side
    // decide between the exact backend and the approximate fallback.
    #[cfg(feature = "biomes")]
    {
        m.add_class::<PyCubiomesBiomeBackend>()?;
        m.add("HAS_CUBIOMES", true)?;
    }
    #[cfg(not(feature = "biomes"))]
    m.add("HAS_CUBIOMES", false)?;
    Ok(())
}
