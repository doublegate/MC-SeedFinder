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
    fn mcsf_gen_biomes(
        g: *mut CubGenerator,
        out: *mut c_int,
        scale: c_int,
        x: c_int,
        z: c_int,
        sx: c_int,
        sz: c_int,
        y: c_int,
    ) -> c_int;
    fn mcsf_init_biome_colors(out: *mut u8);
    fn mcsf_min_cache_size(
        g: *const CubGenerator,
        scale: c_int,
        sx: c_int,
        sy: c_int,
        sz: c_int,
    ) -> usize;
}

/// 256-entry cubiomes biome RGB colormap, fetched once.
fn biome_colormap() -> [[u8; 3]; 256] {
    let mut buf = [0u8; 256 * 3];
    // SAFETY: cubiomes writes exactly 256*3 bytes; we hand it a matching buffer.
    unsafe { mcsf_init_biome_colors(buf.as_mut_ptr()) };
    let mut out = [[0u8; 3]; 256];
    for (i, slot) in out.iter_mut().enumerate() {
        let off = i * 3;
        *slot = [buf[off], buf[off + 1], buf[off + 2]];
    }
    out
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
        self.ensure_seed(world_seed);
        // SAFETY: generator is initialized and seeded above.
        unsafe { mcsf_biome_at(self.generator, SCALE_BLOCK, x, self.y, z) }
    }

    /// (Re)apply a world seed if it differs from the cached one.
    fn ensure_seed(&mut self, world_seed: i64) {
        if !self.seeded || world_seed != self.current_seed {
            // SAFETY: generator is initialized (setup in `new`); reapplying a
            // seed is the documented way to reuse a generator.
            unsafe { mcsf_apply_seed(self.generator, self.dimension, world_seed as u64) };
            self.current_seed = world_seed;
            self.seeded = true;
        }
    }

    /// Render a biome tile to raw RGBA pixels (4 bytes per pixel, row-major,
    /// top-left origin). `(x, z)` is the top-left **block** coordinate; `(sx,
    /// sz)` is the tile size in *scaled* units (so the tile covers
    /// `sx*scale` × `sz*scale` blocks). `scale` must be a cubiomes scale
    /// (1, 4, 16, 64, or 256); 4 is the standard biome-map scale.
    ///
    /// Returns `sx * sz * 4` bytes. Prefer this over `render_tile_png` when
    /// the consumer can blit raw RGBA (e.g. a Canvas2D `putImageData`); it
    /// skips PNG encode + base64 + browser decode (~30–45 ms / tile saved).
    pub fn render_tile_rgba(
        &mut self,
        world_seed: i64,
        scale: i32,
        x: i32,
        z: i32,
        sx: u32,
        sz: u32,
    ) -> Result<Vec<u8>, String> {
        if !matches!(scale, 1 | 4 | 16 | 64 | 256) {
            return Err(format!(
                "invalid scale {scale}; expected 1, 4, 16, 64, or 256"
            ));
        }
        if sx == 0 || sz == 0 {
            return Err("tile size must be positive".into());
        }
        self.ensure_seed(world_seed);

        // CRITICAL: cubiomes' genBiomes may use the output buffer as scratch
        // for its layered noise pipeline; the buffer MUST be at least
        // getMinCacheSize ints, which can exceed sx*sz at certain scales.
        // (Earlier this was sized at sx*sz and a zoom-out crashed glibc with
        // "malloc(): corrupted top size" — the overrun trashed heap metadata.)
        let total = (sx as usize) * (sz as usize);
        // SAFETY: generator is initialized + seeded.
        let min_cache =
            unsafe { mcsf_min_cache_size(self.generator, scale, sx as c_int, 0, sz as c_int) };
        let cache_len = min_cache.max(total);
        let mut ids = vec![0i32; cache_len];
        // SAFETY: generator is initialized + seeded; the buffer is at least
        // mcsf_min_cache_size ints; cubiomes y=0 means a 2D plane.
        let rc = unsafe {
            mcsf_gen_biomes(
                self.generator,
                ids.as_mut_ptr(),
                scale,
                x / scale, // cubiomes Range takes scaled coords
                z / scale,
                sx as c_int,
                sz as c_int,
                0,
            )
        };
        if rc != 0 {
            return Err(format!("cubiomes genBiomes failed with rc={rc}"));
        }

        // Map biome IDs → cubiomes RGB colormap → RGBA pixel buffer. Only the
        // first `total` ints are the readable output (the rest of the cache
        // is post-generation scratch).
        let colors = biome_colormap();
        let mut rgba = Vec::with_capacity(total * 4);
        for &bid in &ids[..total] {
            let [r, g, b] = if (0..256).contains(&bid) {
                colors[bid as usize]
            } else {
                // Unknown / failure → magenta sentinel so it's visually obvious.
                [255, 0, 255]
            };
            rgba.extend_from_slice(&[r, g, b, 255]);
        }
        Ok(rgba)
    }

    /// Render a biome tile as PNG bytes. Wraps [`render_tile_rgba`] with a
    /// PNG encode — useful when the consumer is an `<img src="data:…">` or
    /// the network needs the compression. Prefer `render_tile_rgba` for
    /// in-process Tauri↔WebView transport: it skips PNG encode + base64 +
    /// browser decode (saves ~30–45 ms / tile).
    pub fn render_tile_png(
        &mut self,
        world_seed: i64,
        scale: i32,
        x: i32,
        z: i32,
        sx: u32,
        sz: u32,
    ) -> Result<Vec<u8>, String> {
        let rgba = self.render_tile_rgba(world_seed, scale, x, z, sx, sz)?;
        let mut png_bytes: Vec<u8> = Vec::with_capacity(rgba.len() / 2 + 1024);
        {
            let mut encoder = png::Encoder::new(&mut png_bytes, sx, sz);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder
                .write_header()
                .map_err(|e| format!("png header: {e}"))?;
            writer
                .write_image_data(&rgba)
                .map_err(|e| format!("png write: {e}"))?;
        }
        Ok(png_bytes)
    }

    /// Convenience: render a tile and return a base64 string (no data: prefix).
    pub fn render_tile_base64(
        &mut self,
        world_seed: i64,
        scale: i32,
        x: i32,
        z: i32,
        sx: u32,
        sz: u32,
    ) -> Result<String, String> {
        use base64::Engine;
        let png_bytes = self.render_tile_png(world_seed, scale, x, z, sx, sz)?;
        Ok(base64::engine::general_purpose::STANDARD.encode(png_bytes))
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
    fn render_tile_produces_valid_png_bytes() {
        let mut g = BiomeBackend::from_strs("1.21", "overworld", DEFAULT_Y).expect("backend");
        // 32x32 tile at 1:4 around origin → 128x128 blocks; small but real.
        let bytes = g
            .render_tile_png(1, 4, -64, -64, 32, 32)
            .expect("render_tile_png");
        // PNG magic: 0x89 P N G \r \n 0x1a \n
        assert_eq!(
            &bytes[..8],
            &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a],
            "render_tile_png didn't produce PNG magic"
        );
        assert!(bytes.len() > 64, "PNG suspiciously small: {}", bytes.len());
    }

    #[test]
    fn render_tile_rejects_invalid_scale() {
        let mut g = BiomeBackend::from_strs("1.21", "overworld", DEFAULT_Y).expect("backend");
        assert!(g.render_tile_png(1, 3, 0, 0, 32, 32).is_err());
        assert!(g.render_tile_png(1, 4, 0, 0, 0, 32).is_err());
    }

    // Regression: previously, sizing the genBiomes output buffer at sx*sz
    // overran the heap at coarser scales (cubiomes uses the buffer as scratch
    // and requires getMinCacheSize ints). A zoom-out from scale 4 → 16/64/256
    // would corrupt the allocator and crash the app with
    // "malloc(): corrupted top size". Exercise every supported scale at a
    // realistic tile size; if any of them ever corrupts memory again, address
    // sanitizers or just a fresh allocator state on the next test will trip.
    #[test]
    fn render_tile_succeeds_at_every_supported_scale() {
        let mut g = BiomeBackend::from_strs("1.21", "overworld", DEFAULT_Y).expect("backend");
        for &scale in &[1, 4, 16, 64, 256] {
            let bytes = g
                .render_tile_png(1, scale, -512, -512, 256, 256)
                .unwrap_or_else(|e| panic!("scale {scale} failed: {e}"));
            assert_eq!(
                &bytes[..8],
                &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a],
                "scale {scale} produced bad PNG header"
            );
        }
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
        assert_eq!(
            g.get_biome(1, 0, 0),
            a,
            "switching seeds back is consistent"
        );

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
