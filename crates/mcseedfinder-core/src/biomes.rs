//! Exact biome generation via the vendored cubiomes C library (the oracle).
//!
//! This module hand-declares the FFI to the small shim in `csrc/shim.c` (built
//! by `build.rs` with the `cc` crate) and wraps it in a safe `BiomeBackend`.
//! It answers "what biome is at block coordinate `(x, z)`?" for a given
//! Minecraft Java version, dimension, and world seed. Results are **bit-exact**
//! for the configured version — this is what lets the project drop its
//! "approximate" caveat for biome filtering.
//!
//! Numeric biome IDs returned here are cubiomes `enum BiomeID` values, which the
//! Python `biomes.py` catalog mirrors for MC 1.18+, so the IDs slot straight
//! into the criteria layer's `numeric_ids_for()` set checks with no translation.
//!
//! Biome is queried at a fixed `y` (default sea level, 63). Post-1.18 biomes are
//! 3D, so the answer is exact *for that `(x, y, z)`*; callers pick `y` to mean
//! "surface biome" the same way map viewers do.

use std::ffi::{c_char, c_int, CString};

/// Sentinel returned when cubiomes cannot resolve a biome at a coordinate.
/// Mirrors cubiomes' own `-1` failure value; treated as "no biome" upstream.
pub const NO_BIOME: i32 = -1;

/// Default sampling height: sea level. Surface biomes for the overworld read
/// the same at or above sea level for nearly all terrain.
pub const DEFAULT_Y: i32 = 63;

/// cubiomes coordinate scale: 1 = block (1:1). 4 would be the 1:4 biome grid.
const SCALE_BLOCK: c_int = 1;

/// cubiomes `enum Dimension` integer values (from biomes.h).
const DIM_NETHER: c_int = -1;
const DIM_OVERWORLD: c_int = 0;
const DIM_END: c_int = 1;

/// Opaque handle to a cubiomes `Generator` (allocated/freed on the C side).
#[repr(C)]
struct CubGenerator {
    _private: [u8; 0],
}

extern "C" {
    fn mcsf_generator_alloc() -> *mut CubGenerator;
    fn mcsf_generator_free(g: *mut CubGenerator);
    fn mcsf_setup(g: *mut CubGenerator, mc: c_int, flags: u32);
    fn mcsf_apply_seed(g: *mut CubGenerator, dim: c_int, seed: u64);
    fn mcsf_biome_at(g: *const CubGenerator, scale: c_int, x: c_int, y: c_int, z: c_int) -> c_int;
    fn mcsf_str2mc(s: *const c_char) -> c_int;
}

/// A reusable exact-biome backend bound to a version + dimension.
///
/// The underlying cubiomes generator is allocated once; [`Self::get_biome`]
/// re-applies the seed only when it changes, so a multi-coordinate query for a
/// single seed pays the seed cost once.
pub struct BiomeBackend {
    generator: *mut CubGenerator,
    dimension: c_int,
    y: i32,
    current_seed: i64,
    seeded: bool,
}

// SAFETY: each BiomeBackend exclusively owns its generator pointer and never
// shares it. Moving ownership between threads is sound; `get_biome` requires
// `&mut self`, so there is no aliased mutation. (Send only, not Sync.)
unsafe impl Send for BiomeBackend {}

impl BiomeBackend {
    /// Build a backend for a cubiomes `MCVersion` int and dimension int,
    /// sampling biomes at height `y`. Returns `None` if allocation fails.
    fn new(mc_version: c_int, dimension: c_int, y: i32) -> Option<Self> {
        // SAFETY: alloc returns a pointer to uninitialized Generator storage;
        // setup initializes it before any generation call.
        let generator = unsafe { mcsf_generator_alloc() };
        if generator.is_null() {
            return None;
        }
        // SAFETY: generator is non-null and freshly allocated.
        unsafe { mcsf_setup(generator, mc_version, 0) };
        Some(Self {
            generator,
            dimension,
            y,
            current_seed: 0,
            seeded: false,
        })
    }

    /// Parse human-friendly `version` ("1.21", "1.18.2") and `dimension`
    /// ("overworld", "nether"/"the_nether", "end"/"the_end") strings.
    pub fn from_strs(version: &str, dimension: &str, y: i32) -> Result<Self, String> {
        let cversion = CString::new(version)
            .map_err(|_| format!("version string contains NUL: {version:?}"))?;
        // SAFETY: cversion is a valid NUL-terminated C string.
        let mc = unsafe { mcsf_str2mc(cversion.as_ptr()) };
        if mc == 0 {
            return Err(format!(
                "unknown or unsupported Minecraft version {version:?}"
            ));
        }
        let dim = parse_dimension(dimension)?;
        Self::new(mc, dim, y).ok_or_else(|| "failed to allocate cubiomes generator".to_string())
    }

    /// Numeric cubiomes biome ID at `(x, z)` for `world_seed`, or [`NO_BIOME`].
    pub fn get_biome(&mut self, world_seed: i64, x: i32, z: i32) -> i32 {
        if !self.seeded || world_seed != self.current_seed {
            // SAFETY: generator is initialized (setup in `new`); reapplying a
            // seed is the documented way to reuse a generator.
            unsafe { mcsf_apply_seed(self.generator, self.dimension, world_seed as u64) };
            self.current_seed = world_seed;
            self.seeded = true;
        }
        // SAFETY: generator is initialized and seeded above.
        unsafe { mcsf_biome_at(self.generator, SCALE_BLOCK, x, self.y, z) }
    }
}

impl Drop for BiomeBackend {
    fn drop(&mut self) {
        // SAFETY: generator was allocated by mcsf_generator_alloc and is freed
        // exactly once here.
        unsafe { mcsf_generator_free(self.generator) };
    }
}

/// Map dimension aliases to cubiomes' `Dimension` integer.
fn parse_dimension(dimension: &str) -> Result<c_int, String> {
    match dimension.trim().to_ascii_lowercase().as_str() {
        "overworld" | "the_overworld" => Ok(DIM_OVERWORLD),
        "nether" | "the_nether" => Ok(DIM_NETHER),
        "end" | "the_end" => Ok(DIM_END),
        _ => Err(format!("unknown dimension {dimension:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_version_and_dimension() {
        assert!(BiomeBackend::from_strs("not-a-version", "overworld", DEFAULT_Y).is_err());
        assert!(BiomeBackend::from_strs("1.21", "moon", DEFAULT_Y).is_err());
    }

    #[test]
    fn biome_query_is_valid_and_deterministic() {
        let mut g = BiomeBackend::from_strs("1.21", "overworld", DEFAULT_Y).expect("backend");
        // Repeated queries for the same (seed, x, z) are stable.
        let a = g.get_biome(1, 0, 0);
        let b = g.get_biome(1, 0, 0);
        assert_eq!(a, b);
        assert_ne!(a, NO_BIOME, "biome at origin should resolve");

        // Re-applying a different seed then returning works (exercises the
        // seed-change path) and remains stable.
        let other = g.get_biome(-123, 0, 0);
        assert_ne!(other, NO_BIOME);
        assert_eq!(g.get_biome(1, 0, 0), a, "switching seeds back is consistent");

        // Generation is not constant across a wide region (sanity: real worldgen).
        let mut seen = std::collections::HashSet::new();
        for x in (-3000..3000).step_by(250) {
            seen.insert(g.get_biome(1, x, 0));
        }
        assert!(seen.len() > 1, "expected biome variety across a transect");
    }

    // Integration-regression vectors for MC 1.21, overworld, y=63. Captured from
    // the vendored cubiomes build at integration time; they guard the FFI wiring
    // (seed sign, dimension int, scale, struct handling) against silent drift.
    // Each maps to a real `biomes.py` ID: 24=deep_ocean, 16=beach, 0=ocean,
    // 45=lukewarm_ocean. If a cubiomes submodule bump legitimately changes these,
    // re-capture and note it in the changelog.
    #[test]
    fn frozen_reference_vectors_1_21() {
        let mut g = BiomeBackend::from_strs("1.21", "overworld", DEFAULT_Y).expect("backend");
        // (seed, x, z, expected cubiomes BiomeID)
        let cases: &[(i64, i32, i32, i32)] = &[
            (1, 0, 0, 24),
            (1, 1000, -1000, 24),
            (1, -2000, 1500, 16),
            (12345, 0, 0, 0),
            (-4172144997902289642, 0, 0, 45),
        ];
        for &(seed, x, z, expected) in cases {
            assert_eq!(g.get_biome(seed, x, z), expected, "seed={seed} x={x} z={z}");
        }
    }
}
