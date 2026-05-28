//! Accuracy-critical Minecraft seed search core.
//!
//! This crate starts the Rust-first engine path with deterministic Java RNG and
//! structure-placement math ported from the Python implementation. It is kept
//! dependency-free while the public API settles.

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

#[cfg(feature = "pyo3")]
#[pymodule]
fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyJavaRandom>()?;
    m.add_function(wrap_pyfunction!(get_structure_pos_py, m)?)?;
    m.add_function(wrap_pyfunction!(iter_strongholds_py, m)?)?;
    m.add_function(wrap_pyfunction!(find_structure_matches_range, m)?)?;
    Ok(())
}
