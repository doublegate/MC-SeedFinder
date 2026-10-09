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
pub mod gpu_biome;
#[cfg(all(feature = "gpu", feature = "biomes"))]
pub mod gpu_btree;
#[cfg(all(feature = "gpu", feature = "biomes"))]
pub mod gpu_climate;
#[cfg(all(feature = "gpu", feature = "biomes"))]
pub mod gpu_depth;
#[cfg(all(feature = "gpu", feature = "biomes"))]
pub mod gpu_double_perlin;
#[cfg(all(feature = "gpu", feature = "biomes"))]
pub mod gpu_noise;
#[cfg(all(feature = "gpu", feature = "biomes"))]
pub mod gpu_xoroshiro;
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
    if structure_type == StructureType::BuriedTreasure {
        return Err(PyValueError::new_err(
            "buried_treasure uses roll_buried_treasure_chunk_py \
             (per-chunk roll, not region-grid placement)",
        ));
    }
    let pos = get_structure_pos(structure_type, world_seed, region_x, region_z);
    Ok((pos.structure.name().to_string(), pos.chunk_x, pos.chunk_z))
}

/// Bridge for the buried_treasure per-chunk roll. Returns `True` iff a
/// buried treasure is placed at `(chunk_x, chunk_z)` for `world_seed`.
/// Anchor block when true is `(chunk_x * 16 + 9, chunk_z * 16 + 9)`.
#[cfg(feature = "pyo3")]
#[pyfunction]
fn roll_buried_treasure_chunk_py(world_seed: i64, chunk_x: i32, chunk_z: i32) -> bool {
    structures::roll_buried_treasure_chunk(world_seed, chunk_x, chunk_z)
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
#[pyclass(name = "CubiomesBiomeBackend", unsendable)]
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

/// Parse + compile a conditions-tree JSON spec once at the Python boundary,
/// returning a handle that downstream search calls reuse. Avoids re-parsing
/// the JSON and re-resolving structure names on every chunk.
#[cfg(feature = "pyo3")]
fn compile_tree_from_json(tree_json: &str) -> PyResult<conditions::CompiledNode> {
    let parsed: conditions::Node = serde_json::from_str(tree_json)
        .map_err(|e| PyValueError::new_err(format!("invalid tree JSON: {e}")))?;
    conditions::compile(&parsed).map_err(|e| match e {
        conditions::CompileError::UnknownStructure(s) => {
            PyKeyError::new_err(format!("unknown structure {s:?}"))
        }
        other => PyValueError::new_err(other.to_string()),
    })
}

/// Reusable handle wrapping a validated [`conditions::CompiledNode`]. The
/// Python worker compiles once at pool init and passes this handle to every
/// `find_tree_matches_range_compiled` call, skipping the per-chunk
/// `json.dumps` + `serde_json::from_str` + `compile` overhead that
/// [`find_tree_matches_range`] pays on each call.
#[cfg(feature = "pyo3")]
#[pyclass(name = "CompiledTree", frozen)]
struct PyCompiledTree {
    inner: conditions::CompiledNode,
    /// True iff the tree contains any biome leaf. Reported to the Python
    /// side so the coordinator can route to the biome-aware native path
    /// (when wired up) versus the structure-only fast path.
    has_biome_leaves: bool,
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl PyCompiledTree {
    #[getter]
    fn has_biome_leaves(&self) -> bool {
        self.has_biome_leaves
    }
}

/// Compile a conditions-tree JSON spec once and return a reusable handle.
/// Call once at worker init; pass the handle to
/// [`find_tree_matches_range_compiled`] for every chunk.
#[cfg(feature = "pyo3")]
#[pyfunction]
fn compile_tree(tree_json: &str) -> PyResult<PyCompiledTree> {
    let compiled = compile_tree_from_json(tree_json)?;
    let has_biome_leaves = conditions::has_biome_leaves(&compiled);
    Ok(PyCompiledTree {
        inner: compiled,
        has_biome_leaves,
    })
}

/// Structure-only conditions-tree fast-path. Accepts the Python tree spec as a
/// JSON string, compiles it once (resolving structure names + validation), and
/// filters a contiguous seed range. Biome conditions are NOT supported here —
/// callers must keep those on the Python evaluation path. For repeated calls
/// against the same tree (e.g. inside a worker pool chunking through a search),
/// prefer [`compile_tree`] + [`find_tree_matches_range_compiled`] to skip the
/// per-call JSON parse + compile.
#[cfg(feature = "pyo3")]
#[pyfunction]
fn find_tree_matches_range(start_seed: i64, count: u64, tree_json: &str) -> PyResult<Vec<i64>> {
    let compiled = compile_tree_from_json(tree_json)?;
    Ok(conditions::find_matches_range(start_seed, count, &compiled))
}

/// Run a structure-only search using a pre-compiled tree handle. Identical
/// semantics to [`find_tree_matches_range`] but avoids the per-call JSON
/// parse + compile cost when many chunks share the same criteria.
#[cfg(feature = "pyo3")]
#[pyfunction]
fn find_tree_matches_range_compiled(
    py: Python<'_>,
    start_seed: i64,
    count: u64,
    compiled: &PyCompiledTree,
) -> Vec<i64> {
    // Release the GIL around the inner search loop so Python threads (e.g.
    // progress callbacks, signal handlers) aren't blocked while we churn
    // through 4 k–1 M seeds. The compiled tree is `frozen`, so no GIL is
    // needed to touch its fields.
    py.detach(|| conditions::find_matches_range(start_seed, count, &compiled.inner))
}

/// Biome-aware sibling of [`find_tree_matches_range_compiled`]. Accepts a
/// pre-compiled tree that may contain biome leaves (`spawn_biome`,
/// `nearby_biomes`, `biome_area`) and a cubiomes `BiomeBackend` reused for
/// the whole call. Replaces the previous Python-side fallback path that
/// called back into PyO3 (and hence cubiomes) per coordinate — the
/// per-coord PyO3 trip cost ~5 µs each, dragging the search rate from
/// ~10⁶ seeds/s down to ~10² seeds/s on biome-touching criteria.
///
/// The backend is taken as `&Bound<PyCubiomesBiomeBackend>` so we can
/// mutably borrow it for the duration of the call (cubiomes' `Generator`
/// is single-threaded; the GIL release below keeps it that way at the
/// Python level).
#[cfg(all(feature = "pyo3", feature = "biomes"))]
#[pyfunction]
fn find_tree_matches_range_compiled_with_biomes(
    start_seed: i64,
    count: u64,
    compiled: &PyCompiledTree,
    backend: &Bound<'_, PyCubiomesBiomeBackend>,
) -> PyResult<Vec<i64>> {
    // Borrow the backend mutably for the whole search; cubiomes' Generator
    // is mutated by every `apply_seed` call inside the evaluator. Any
    // attempt to use the backend from another Python thread will fail
    // with the standard PyO3 RuntimeError, which is the correct semantics
    // — concurrent access would corrupt the generator state.
    //
    // No `Python::allow_threads` here: the `PyRefMut` we hold carries a GIL
    // marker that PyO3 (correctly) refuses to send into a no-GIL closure.
    // The typical deployment is multiprocessing (one Python interpreter
    // per worker), where GIL contention is moot. A future optimisation
    // could lift the BiomeBackend out via an Arc<Mutex<_>> for the
    // threaded case.
    let mut backend_ref = backend.borrow_mut();
    Ok(conditions::find_matches_range_with_biomes(
        start_seed,
        count,
        &compiled.inner,
        &mut backend_ref.inner,
    ))
}

/// Rayon-parallel sibling of [`find_tree_matches_range_compiled`]. Spreads
/// the seed range across rayon's global thread pool; returns matches in the
/// same order as the serial path (parity test guards this). Use this from a
/// single-process search loop (Tauri's per-job thread, CLI runs with
/// `--workers=1`) — NOT from inside a multiprocessing worker, where it
/// would oversubscribe the CPU.
#[cfg(all(feature = "pyo3", feature = "parallel"))]
#[pyfunction]
fn find_tree_matches_range_compiled_parallel(
    py: Python<'_>,
    start_seed: i64,
    count: u64,
    compiled: &PyCompiledTree,
) -> Vec<i64> {
    py.detach(|| conditions::find_matches_range_parallel(start_seed, count, &compiled.inner))
}

/// Convert a Bedrock text seed to its canonical i32 game-seed via Java's
/// `String.hashCode()`. Same algorithm Java and Bedrock both use when a
/// player types a non-numeric seed into the world creation screen.
#[cfg(feature = "pyo3")]
#[pyfunction]
fn bedrock_seed_from_string(text: &str) -> i32 {
    bedrock::seed_from_string(text)
}

/// True iff cubiomes (via `mcsf_str2mc`) recognises the given Minecraft
/// version string. Used by the CLI's pre-flight check to surface a clear
/// "this version isn't supported by the bundled cubiomes" error before
/// the user spends time setting up a search.
#[cfg(all(feature = "pyo3", feature = "biomes"))]
#[pyfunction]
fn is_supported_version(version: &str) -> bool {
    biomes::BiomeBackend::from_strs(version, "overworld", biomes::DEFAULT_Y).is_ok()
}

#[cfg(feature = "pyo3")]
#[pymodule]
fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyJavaRandom>()?;
    m.add_function(wrap_pyfunction!(get_structure_pos_py, m)?)?;
    m.add_function(wrap_pyfunction!(iter_strongholds_py, m)?)?;
    m.add_function(wrap_pyfunction!(roll_buried_treasure_chunk_py, m)?)?;
    m.add_function(wrap_pyfunction!(find_structure_matches_range, m)?)?;
    m.add_function(wrap_pyfunction!(find_tree_matches_range, m)?)?;
    m.add_function(wrap_pyfunction!(compile_tree, m)?)?;
    m.add_function(wrap_pyfunction!(find_tree_matches_range_compiled, m)?)?;
    m.add_class::<PyCompiledTree>()?;
    #[cfg(feature = "parallel")]
    {
        m.add_function(wrap_pyfunction!(
            find_tree_matches_range_compiled_parallel,
            m
        )?)?;
        m.add("HAS_PARALLEL", true)?;
    }
    #[cfg(not(feature = "parallel"))]
    m.add("HAS_PARALLEL", false)?;
    m.add_function(wrap_pyfunction!(bedrock_seed_from_string, m)?)?;
    // Whether this build links cubiomes for exact biomes. Lets the Python side
    // decide between the exact backend and the approximate fallback.
    #[cfg(feature = "biomes")]
    {
        m.add_class::<PyCubiomesBiomeBackend>()?;
        m.add_function(wrap_pyfunction!(
            find_tree_matches_range_compiled_with_biomes,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(is_supported_version, m)?)?;
        m.add("HAS_CUBIOMES", true)?;
    }
    #[cfg(not(feature = "biomes"))]
    m.add("HAS_CUBIOMES", false)?;
    Ok(())
}
