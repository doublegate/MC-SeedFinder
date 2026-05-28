//! Phase 6c-5: End-to-end GPU biome generation.
//!
//! Orchestrates the four GPU primitives built in Phase 6c-1..6c-4 plus a
//! small CPU pass for the depth-spline to reproduce cubiomes'
//! `sampleBiomeNoise` for a grid of (x, z) coordinates. The full pipeline:
//!
//!   1. **Shift** sample on GPU at (x, 0, z) → `shift_x`
//!   2. **Shift** sample on GPU at (z, x, 0) → `shift_z`  [args rotated]
//!   3. `(px, pz) = (x + shift_x * 4, z + shift_z * 4)`
//!   4. **Climate** sample on GPU for the 5 fields (T, H, C, E, W) at
//!      (px, 0, pz)
//!   5. **CPU**: compute depth from (c, e, w) via the seed-independent
//!      cubiomes spline tree (single eval per pixel; the spline is f32
//!      and small per-pixel work compared to the 6-field × ~18-octave
//!      climate stack on GPU).
//!   6. Assemble `np[6] = i64-cast(10000 × {t, h, c, e, d, w})`
//!   7. **B-tree** walk on GPU → biome IDs
//!
//! The CPU path through `mcsf_compute_depth` keeps the spline tree
//! unchanged in cubiomes; only the heavy work (climate noise = 5
//! double-perlins × up to 18 octaves) runs on GPU.
//!
//! Accuracy target: f32 climate drift at biome boundaries can flip a
//! handful of pixels. The parity test requires ≥99% agreement with
//! cubiomes' `genBiomes` over a 32×32 grid for seed 12345 MC 1.21.

#![cfg(feature = "gpu")]

use crate::gpu_btree::GpuBiomeTree;
use crate::gpu_climate::{ClimateField, ClimateNoise};
use crate::gpu_double_perlin::GpuDoublePerlin;

extern "C" {
    fn mcsf_compute_depth(
        mc: std::os::raw::c_int,
        c: f64,
        e: f64,
        w: f64,
        y: std::os::raw::c_int,
    ) -> f64;
}

/// All-in-one GPU biome renderer: holds the GPU pipelines and the
/// per-seed [`ClimateNoise`] stack. One instance per seed; cheap to
/// rebuild when the seed changes (ClimateNoise extraction is fast — six
/// `setBiomeSeed` calls).
pub struct GpuBiomeRenderer {
    gpu_dp: GpuDoublePerlin,
    gpu_bt: GpuBiomeTree,
    climate: ClimateNoise,
}

impl std::fmt::Debug for GpuBiomeRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBiomeRenderer")
            .field("mc", &self.climate.mc)
            .field("seed", &self.climate.seed)
            .field("large", &self.climate.large)
            .finish_non_exhaustive()
    }
}

impl GpuBiomeRenderer {
    /// Build the renderer for an Overworld seed. Returns `None` if no GPU
    /// adapter is available.
    pub fn try_new(mc: i32, seed: u64, large: bool) -> Option<Self> {
        let gpu_dp = GpuDoublePerlin::try_new()?;
        let gpu_bt = GpuBiomeTree::try_new()?;
        let climate = ClimateNoise::for_overworld(mc, seed, large);
        Some(Self {
            gpu_dp,
            gpu_bt,
            climate,
        })
    }

    pub fn climate(&self) -> &ClimateNoise {
        &self.climate
    }

    /// Generate biome IDs for a 2D grid in **block** coordinates,
    /// matching `genBiomeNoise3D(bn, out, r, opt=0)` for `r.scale==4`.
    ///
    /// `coords` is a flat list of `(x, y, z)` triples; output is one
    /// cubiomes biome ID per input. (3D structure is irrelevant here —
    /// shift and climate are 2D, depth uses y only as a small linear
    /// offset.)
    pub fn biome_grid(&self, coords: &[(i32, i32, i32)]) -> Result<Vec<i32>, &'static str> {
        if coords.is_empty() {
            return Ok(Vec::new());
        }
        let n = coords.len();

        // 1. Build two shift-sample input lists: (x, 0, z) and (z, x, 0).
        let mut shift_in_a: Vec<[f32; 3]> = Vec::with_capacity(n);
        let mut shift_in_b: Vec<[f32; 3]> = Vec::with_capacity(n);
        for &(x, _y, z) in coords {
            shift_in_a.push([x as f32, 0.0, z as f32]);
            shift_in_b.push([z as f32, x as f32, 0.0]);
        }
        let shift_spec = self.climate.field(ClimateField::Shift);

        // 2. Two GPU dispatches for the two shift samples.
        let shift_a = self.gpu_dp.sample_batch(shift_spec, &shift_in_a)?;
        let shift_b = self.gpu_dp.sample_batch(shift_spec, &shift_in_b)?;

        // 3. Build perturbed climate-sample coords.
        let mut climate_in: Vec<[f32; 3]> = Vec::with_capacity(n);
        for (i, &(x, _y, z)) in coords.iter().enumerate() {
            let px = x as f32 + shift_a[i] * 4.0;
            let pz = z as f32 + shift_b[i] * 4.0;
            climate_in.push([px, 0.0, pz]);
        }

        // 4. Sample the 5 climate fields at the perturbed coords.
        // Order matters: np[0]=T, np[1]=H, np[2]=C, np[3]=E, np[4]=D, np[5]=W
        let t = self
            .gpu_dp
            .sample_batch(self.climate.field(ClimateField::Temperature), &climate_in)?;
        let h = self
            .gpu_dp
            .sample_batch(self.climate.field(ClimateField::Humidity), &climate_in)?;
        let c = self.gpu_dp.sample_batch(
            self.climate.field(ClimateField::Continentalness),
            &climate_in,
        )?;
        let e = self
            .gpu_dp
            .sample_batch(self.climate.field(ClimateField::Erosion), &climate_in)?;
        let w = self
            .gpu_dp
            .sample_batch(self.climate.field(ClimateField::Weirdness), &climate_in)?;

        // 5. CPU pass: compute depth per pixel via cubiomes' spline tree.
        // 6. Assemble np[6] as i64 values (10000 × climate, signed truncated).
        let mut np_inputs: Vec<[i64; 6]> = Vec::with_capacity(n);
        for (i, &(_x, y, _z)) in coords.iter().enumerate() {
            let ci = c[i] as f64;
            let ei = e[i] as f64;
            let wi = w[i] as f64;
            let d = unsafe { mcsf_compute_depth(self.climate.mc, ci, ei, wi, y) };
            // cubiomes does `(int64_t)(10000.0F * value)` — note `F` suffix.
            // The cast goes through f32 first; replicate that to avoid
            // a half-ULP drift compared to cubiomes.
            let ti = t[i];
            let hi = h[i];
            np_inputs.push([
                (10000.0_f32 * ti) as i64,
                (10000.0_f32 * hi) as i64,
                (10000.0_f32 * c[i]) as i64,
                (10000.0_f32 * e[i]) as i64,
                (10000.0_f32 * d as f32) as i64,
                (10000.0_f32 * w[i]) as i64,
            ]);
        }

        // 7. GPU b-tree walk.
        let biomes_u = self.gpu_bt.lookup_batch(&np_inputs)?;
        Ok(biomes_u.into_iter().map(|b| b as i32).collect())
    }

    /// Phase 6c-7 primitive: evaluate a "biome at coord is in `allowed`"
    /// predicate across a batch of coords. Returns one `bool` per input
    /// — the per-coord biome ID is computed via [`Self::biome_grid`] then
    /// tested against the `allowed` set (sorted, binary search).
    ///
    /// This is the primitive a per-seed biome-area condition in the seed
    /// searcher would call: a candidate seed is built once via
    /// [`Self::try_new`], then this method evaluates the area's biome
    /// constraint at the trial coords in a single GPU dispatch worth of
    /// noise + b-tree work.
    ///
    /// **Multi-seed search integration is NOT done here.** Extending the
    /// existing multi-predicate seed kernel (Phase 6b) to evaluate biome
    /// conditions per seed requires porting cubiomes' Xoroshiro128++
    /// (`xSetSeed`/`xNextLong`) and `setBiomeSeed` to WGSL so the
    /// climate-noise stack can be (re)initialised per-thread on the GPU.
    /// That is a sub-phase in its own right — tracked as future work —
    /// and is intentionally out of scope for "Phase 6c: one MC version,
    /// matching cubiomes". For now, search continues to evaluate biome
    /// conditions on the CPU staging path; this primitive becomes the
    /// per-seed building block once GPU climate init lands.
    pub fn biomes_in_set(
        &self,
        coords: &[(i32, i32, i32)],
        allowed: &[i32],
    ) -> Result<Vec<bool>, &'static str> {
        let biomes = self.biome_grid(coords)?;
        // Sort + dedup `allowed` once; binary-search per result. For the
        // typical small biome sets the search uses (<10 biome IDs), a
        // sorted Vec scan is faster than HashSet hashing.
        let mut set: Vec<i32> = allowed.to_vec();
        set.sort_unstable();
        set.dedup();
        Ok(biomes
            .into_iter()
            .map(|b| set.binary_search(&b).is_ok())
            .collect())
    }

    /// Render a biome tile to raw RGBA bytes, matching the layout produced
    /// by [`crate::biomes::BiomeBackend::render_tile_rgba`] at the canonical
    /// biome scale (1:4).
    ///
    /// Currently only `scale == 4` is supported. cubiomes' `r.scale` values
    /// 1 (voronoi) and ≥16 (inner-scaling) need an extra coord-shaping pass
    /// that hasn't been ported yet — for those, fall back to
    /// `BiomeBackend::render_tile_rgba` (cubiomes CPU).
    ///
    /// `(x, z)` is the top-left **block** coordinate; `(sx, sz)` is the
    /// tile size in scale-grid units (so the tile covers `sx*scale`
    /// blocks). Returns `sx * sz * 4` bytes.
    pub fn render_tile_rgba(
        &self,
        scale: i32,
        x: i32,
        z: i32,
        sx: u32,
        sz: u32,
    ) -> Result<Vec<u8>, String> {
        if scale != 4 {
            return Err(format!(
                "GpuBiomeRenderer::render_tile_rgba currently only supports \
                 scale=4 (1:4 biome grid); got scale={scale}"
            ));
        }
        if sx == 0 || sz == 0 {
            return Err("tile size must be positive".into());
        }
        let total = (sx as usize) * (sz as usize);

        // cubiomes' genBiomeNoise3D at r.scale=4: inner scale=1, mid=0.
        // r.x = caller's x / scale. xi = r.x + i (in scale-4 grid units).
        // sampleBiomeNoise input is the scale-4 grid coord.
        let gx = x / scale;
        let gz = z / scale;
        let mut coords: Vec<(i32, i32, i32)> = Vec::with_capacity(total);
        for j in 0..sz {
            for i in 0..sx {
                coords.push((gx + i as i32, 0, gz + j as i32));
            }
        }
        let biomes = self
            .biome_grid(&coords)
            .map_err(|e| format!("biome_grid failed: {e}"))?;

        let colors = crate::biomes::biome_colormap();
        let mut rgba = Vec::with_capacity(total * 4);
        for &bid in &biomes {
            let [r, g, b] = if (0..256).contains(&bid) {
                colors[bid as usize]
            } else {
                // Unknown / failure — same magenta sentinel as the CPU path.
                [255, 0, 255]
            };
            rgba.extend_from_slice(&[r, g, b, 255]);
        }
        Ok(rgba)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_climate::parse_mc_version;

    extern "C" {
        fn mcsf_sample_biome_at(
            mc: std::os::raw::c_int,
            seed: u64,
            large: std::os::raw::c_int,
            x: std::os::raw::c_int,
            y: std::os::raw::c_int,
            z: std::os::raw::c_int,
            out_np6: *mut i64,
        ) -> std::os::raw::c_int;
    }

    /// Phase 6c-5 cornerstone test. Compare the GPU-orchestrated biome
    /// pipeline (shift+climate on GPU, depth on CPU, b-tree on GPU) to
    /// cubiomes' real `sampleBiomeNoise` for a 32×32 block grid centred
    /// near origin. f32 climate drift at biome boundaries can flip a few
    /// pixels; we require ≥99% agreement.
    #[test]
    fn gpu_biome_renderer_matches_cubiomes_on_32x32_grid_mc_1_21() {
        let Some(mc) = parse_mc_version("1.21") else {
            return;
        };
        let Some(rend) = GpuBiomeRenderer::try_new(mc, 12345, false) else {
            eprintln!("skipping: no GPU adapter on this host");
            return;
        };

        // 32×32 block grid at 8-block spacing → 256m × 256m around (0, 0),
        // sampled at the same scale cubiomes uses for tile rendering.
        // Climate features change on a 100-200 m scale, so we'll cross
        // many biome boundaries.
        let mut coords: Vec<(i32, i32, i32)> = Vec::with_capacity(1024);
        let mut expected: Vec<i32> = Vec::with_capacity(1024);
        for ix in -16..16 {
            for iz in -16..16 {
                let x = ix * 8;
                let z = iz * 8;
                coords.push((x, 0, z));
                let mut np = [0i64; 6];
                let biome = unsafe { mcsf_sample_biome_at(mc, 12345, 0, x, 0, z, np.as_mut_ptr()) };
                expected.push(biome);
            }
        }

        let got = rend.biome_grid(&coords).expect("gpu biome grid");
        assert_eq!(got.len(), expected.len());

        let mut matches = 0usize;
        let mut mismatches = Vec::new();
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            if g == e {
                matches += 1;
            } else if mismatches.len() < 8 {
                mismatches.push((coords[i], *g, *e));
            }
        }
        let n = expected.len();
        let rate = matches as f64 / n as f64;
        eprintln!("biome match rate: {matches}/{n} = {:.2}%", rate * 100.0);
        if !mismatches.is_empty() {
            eprintln!("first mismatches (showing up to 8):");
            for (c, g, e) in &mismatches {
                eprintln!("  {c:?}: gpu={g} cubiomes={e}");
            }
        }
        // Require ≥99% match. f32 drift at biome edges is expected; a
        // catastrophic regression (wrong noise sum, wrong b-tree path)
        // would push this well below 50%.
        assert!(
            rate >= 0.99,
            "biome match rate {rate:.4} below 99% threshold"
        );
    }

    /// Phase 6c-7 primitive test: `biomes_in_set` correctly tags coords
    /// whose biome is in the allowed set. Uses the same 16-coord sample
    /// as the 6c-5 grid; expected biomes come from cubiomes directly so
    /// the test verifies (a) GPU produces the same biome IDs and (b) the
    /// set-membership predicate works.
    #[test]
    fn gpu_biomes_in_set_matches_cubiomes_membership() {
        let Some(mc) = parse_mc_version("1.21") else {
            return;
        };
        let Some(rend) = GpuBiomeRenderer::try_new(mc, 12345, false) else {
            eprintln!("skipping: no GPU adapter on this host");
            return;
        };

        let mut coords: Vec<(i32, i32, i32)> = Vec::new();
        let mut cubi_biomes: Vec<i32> = Vec::new();
        for ix in -4..4 {
            for iz in -4..4 {
                let x = ix * 32;
                let z = iz * 32;
                coords.push((x, 0, z));
                let mut np = [0i64; 6];
                let biome = unsafe { mcsf_sample_biome_at(mc, 12345, 0, x, 0, z, np.as_mut_ptr()) };
                cubi_biomes.push(biome);
            }
        }

        // Pick the most common biome from the sample as the allowed set.
        let mut counts = std::collections::HashMap::new();
        for &b in &cubi_biomes {
            *counts.entry(b).or_insert(0u32) += 1;
        }
        let common: i32 = *counts
            .iter()
            .max_by_key(|(_, c)| *c)
            .map(|(b, _)| b)
            .unwrap();
        let allowed = vec![common];

        let got = rend
            .biomes_in_set(&coords, &allowed)
            .expect("gpu biomes_in_set");
        assert_eq!(got.len(), coords.len());

        // Each `got[i]` should be true iff cubi_biomes[i] == common.
        for (i, (g, &b)) in got.iter().zip(cubi_biomes.iter()).enumerate() {
            assert_eq!(
                *g,
                b == common,
                "idx {i} {coords:?}: cubiomes biome={b} common={common} got={g}",
                coords = coords[i]
            );
        }
    }

    /// Phase 6c-6 cornerstone test: GPU-rendered RGBA tile must agree
    /// with `BiomeBackend::render_tile_rgba` (cubiomes CPU) byte-for-byte.
    /// Both paths feed through the same biome colormap, so any colour
    /// difference is a biome-ID mismatch. Match rate ≥99% (same f32 drift
    /// tolerance as the biome-grid test).
    #[test]
    fn gpu_render_tile_rgba_matches_biome_backend() {
        use crate::biomes::BiomeBackend;
        let Some(mc) = parse_mc_version("1.21") else {
            return;
        };
        let Some(rend) = GpuBiomeRenderer::try_new(mc, 12345, false) else {
            eprintln!("skipping: no GPU adapter on this host");
            return;
        };
        let mut cpu = match BiomeBackend::from_strs("1.21", "overworld", 0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping: BiomeBackend init failed: {e}");
                return;
            }
        };

        // 32×32 tile at scale 4 = 128 × 128 blocks. The GPU renderer's
        // top-left is at (x=0, z=0) block coords.
        let scale = 4i32;
        let sx = 32u32;
        let sz = 32u32;
        let gpu_rgba = rend
            .render_tile_rgba(scale, 0, 0, sx, sz)
            .expect("gpu render");
        let cpu_rgba = cpu
            .render_tile_rgba(12345, scale, 0, 0, sx, sz)
            .expect("cpu render");
        assert_eq!(gpu_rgba.len(), cpu_rgba.len());

        let mut equal_pixels = 0usize;
        let total = (sx * sz) as usize;
        for i in 0..total {
            let a = &gpu_rgba[i * 4..i * 4 + 4];
            let b = &cpu_rgba[i * 4..i * 4 + 4];
            if a == b {
                equal_pixels += 1;
            }
        }
        let rate = equal_pixels as f64 / total as f64;
        eprintln!(
            "tile RGBA match rate: {equal_pixels}/{total} = {:.2}%",
            rate * 100.0
        );
        assert!(rate >= 0.99, "tile RGBA match rate {rate:.4} below 99%");
    }
}
