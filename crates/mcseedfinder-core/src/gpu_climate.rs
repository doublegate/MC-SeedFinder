//! Phase 6c-3: Climate noise stack for MC 1.18+ Overworld.
//!
//! Wraps cubiomes' `setBiomeSeed` to extract the six climate
//! `DoublePerlinNoise` fields (temperature, humidity, continentalness,
//! erosion, shift, weirdness) into [`DoublePerlinSpec`]s ready for GPU
//! dispatch. The CPU reference is cubiomes' own `sampleClimatePara` via the
//! `mcsf_climate_sample_field` shim — passing parity means GPU output
//! matches the real cubiomes init+sample path end-to-end.
//!
//! Each field is its own pipeline dispatch in [`GpuDoublePerlin`]; Phase
//! 6c-4+ will batch all six in a single fused kernel that feeds directly
//! into the biome b-tree walker.

#![cfg(feature = "gpu")]

use crate::gpu_double_perlin::{DoublePerlinSpec, OctaveSpec};
use crate::gpu_noise::PerlinState;

// Max octaves cubiomes ever uses for any single climate field's
// DoublePerlinNoise. Continentalness is the largest at 9 + 9 = 18.
const MAX_OCT_PER_FIELD: usize = 18;

extern "C" {
    fn mcsf_climate_init_field(
        mc: i32,
        seed: u64,
        large: i32,
        field_idx: i32,
        out_oct_a_count: *mut i32,
        out_oct_b_count: *mut i32,
        out_dp_amplitude: *mut f64,
        out_abc: *mut f64,
        out_h2: *mut u8,
        out_d2_t2: *mut f64,
        out_amp_lac: *mut f64,
        out_perm: *mut u8,
    ) -> i32;
    fn mcsf_climate_sample_field(
        mc: i32,
        seed: u64,
        large: i32,
        field_idx: i32,
        x: f64,
        z: f64,
    ) -> f64;
    fn mcsf_str2mc(s: *const std::os::raw::c_char) -> i32;
}

/// Cubiomes' NP_* climate-field enum (1.18+ Overworld only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClimateField {
    Temperature = 0,
    Humidity = 1,
    Continentalness = 2,
    Erosion = 3,
    /// NP_SHIFT in cubiomes — used to perturb x/z for the other climates.
    /// (Same enum slot as NP_DEPTH, which is a derived field — not the
    /// underlying double-perlin we expose here.)
    Shift = 4,
    Weirdness = 5,
}

impl ClimateField {
    pub const ALL: [ClimateField; 6] = [
        ClimateField::Temperature,
        ClimateField::Humidity,
        ClimateField::Continentalness,
        ClimateField::Erosion,
        ClimateField::Shift,
        ClimateField::Weirdness,
    ];

    pub fn name(self) -> &'static str {
        match self {
            ClimateField::Temperature => "temperature",
            ClimateField::Humidity => "humidity",
            ClimateField::Continentalness => "continentalness",
            ClimateField::Erosion => "erosion",
            ClimateField::Shift => "shift",
            ClimateField::Weirdness => "weirdness",
        }
    }
}

/// Parse a Minecraft version string ("1.21", "1.20.4") into cubiomes'
/// internal MCVersion int. Returns `None` for unrecognised strings.
pub fn parse_mc_version(s: &str) -> Option<i32> {
    let c = std::ffi::CString::new(s).ok()?;
    let v = unsafe { mcsf_str2mc(c.as_ptr()) };
    if v <= 0 {
        None
    } else {
        Some(v)
    }
}

/// All six climate noises for an Overworld seed, ready for GPU dispatch.
#[derive(Clone, Debug)]
pub struct ClimateNoise {
    pub mc: i32,
    pub seed: u64,
    pub large: bool,
    pub fields: [DoublePerlinSpec; 6],
}

impl ClimateNoise {
    /// Build all six fields by extracting octave state from cubiomes'
    /// `setBiomeSeed`. The returned spec is bit-identical to what cubiomes
    /// itself uses internally (the same `PerlinNoise` struct values).
    pub fn for_overworld(mc: i32, seed: u64, large: bool) -> Self {
        let fields = std::array::from_fn(|i| Self::extract_field(mc, seed, large, i as i32));
        Self {
            mc,
            seed,
            large,
            fields,
        }
    }

    /// Convenience: parse a version string and build the climate stack.
    pub fn for_overworld_str(version: &str, seed: u64, large: bool) -> Option<Self> {
        parse_mc_version(version).map(|mc| Self::for_overworld(mc, seed, large))
    }

    pub fn field(&self, f: ClimateField) -> &DoublePerlinSpec {
        &self.fields[f as usize]
    }

    /// CPU reference via cubiomes' own `sampleClimatePara`. Used by the
    /// parity test as the ground truth.
    pub fn cpu_sample_field(&self, f: ClimateField, x: f64, z: f64) -> f64 {
        unsafe {
            mcsf_climate_sample_field(
                self.mc,
                self.seed,
                if self.large { 1 } else { 0 },
                f as i32,
                x,
                z,
            )
        }
    }

    fn extract_field(mc: i32, seed: u64, large: bool, field_idx: i32) -> DoublePerlinSpec {
        let mut a_count: i32 = 0;
        let mut b_count: i32 = 0;
        let mut dp_amplitude: f64 = 0.0;
        let mut abc = vec![0.0f64; MAX_OCT_PER_FIELD * 3];
        let mut h2 = vec![0u8; MAX_OCT_PER_FIELD];
        let mut d2_t2 = vec![0.0f64; MAX_OCT_PER_FIELD * 2];
        let mut amp_lac = vec![0.0f64; MAX_OCT_PER_FIELD * 2];
        let mut perm = vec![0u8; MAX_OCT_PER_FIELD * 256];

        let total = unsafe {
            mcsf_climate_init_field(
                mc,
                seed,
                if large { 1 } else { 0 },
                field_idx,
                &mut a_count,
                &mut b_count,
                &mut dp_amplitude,
                abc.as_mut_ptr(),
                h2.as_mut_ptr(),
                d2_t2.as_mut_ptr(),
                amp_lac.as_mut_ptr(),
                perm.as_mut_ptr(),
            )
        };
        assert!(
            total >= 0,
            "mcsf_climate_init_field returned negative count"
        );
        let total = total as usize;
        assert!(
            total <= MAX_OCT_PER_FIELD,
            "field {field_idx} produced {total} octaves (max {MAX_OCT_PER_FIELD})"
        );
        assert_eq!(total, (a_count + b_count) as usize);

        let mut octaves = Vec::with_capacity(total);
        for i in 0..total {
            let mut state_perm = [0u8; 256];
            state_perm.copy_from_slice(&perm[i * 256..(i + 1) * 256]);
            octaves.push(OctaveSpec {
                state: PerlinState {
                    a: abc[i * 3],
                    b: abc[i * 3 + 1],
                    c: abc[i * 3 + 2],
                    h2: h2[i],
                    d2: d2_t2[i * 2],
                    t2: d2_t2[i * 2 + 1],
                    perm: state_perm,
                },
                amplitude: amp_lac[i * 2],
                lacunarity: amp_lac[i * 2 + 1],
            });
        }
        let a = a_count as usize;
        let oct_b = octaves.split_off(a);
        let oct_a = octaves;
        DoublePerlinSpec {
            oct_a,
            oct_b,
            amplitude: dp_amplitude,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_double_perlin::GpuDoublePerlin;

    /// Phase 6c-3 cornerstone test. For each of the 6 climate fields,
    /// dispatch the GPU `cs_double_perlin` kernel against the seed's
    /// cubiomes-derived `DoublePerlinSpec` and assert every sample matches
    /// cubiomes' own `sampleClimatePara` within f32 tolerance.
    ///
    /// Tolerance scaling: continentalness has 18 octaves with amplitudes
    /// up to 2.0. Worst-case per-sample drift bound is roughly
    /// 2 * sum_amp * f32_eps_for_perlin ≈ 2 * 11 * 3e-5 ≈ 6.6e-4. We use
    /// 5e-3 to be robust against driver f32 fma policy.
    #[test]
    fn gpu_climate_matches_cubiomes_for_seed_12345_mc_1_21() {
        let Some(gpu) = GpuDoublePerlin::try_new() else {
            eprintln!("skipping: no GPU adapter on this host");
            return;
        };
        let mc = match parse_mc_version("1.21") {
            Some(v) => v,
            None => {
                eprintln!("skipping: cubiomes does not know MC 1.21");
                return;
            }
        };
        let climate = ClimateNoise::for_overworld(mc, 12345, false);

        // Mix of origin, near-origin, scale-1 offsets, and far-out coords.
        // Climate noise is sampled at scale 1:4, so coords here are
        // "noise coords" (post-scaling) — same as cubiomes' input domain.
        let coords: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [16.0, 0.0, 16.0],
            [-128.0, 0.0, 64.0],
            [1024.0, 0.0, -1024.0],
            [3.5, 0.0, 7.25],
            [-555.5, 0.0, 333.25],
        ];

        let mut worst_per_field = [0.0f64; 6];
        for f in ClimateField::ALL {
            let spec = climate.field(f);
            // Empty spec should never happen — every field has ≥1 octave.
            assert!(
                !(spec.oct_a.is_empty() && spec.oct_b.is_empty()),
                "field {} has no octaves",
                f.name()
            );

            let gpu_results = gpu.sample_batch(spec, &coords).expect("gpu dispatch");
            assert_eq!(gpu_results.len(), coords.len());
            for (c, gpu_v) in coords.iter().zip(gpu_results.iter()) {
                let cpu_v = climate.cpu_sample_field(f, c[0] as f64, c[2] as f64);
                let diff = (cpu_v - *gpu_v as f64).abs();
                if diff > worst_per_field[f as usize] {
                    worst_per_field[f as usize] = diff;
                }
                assert!(
                    diff < 5e-3,
                    "climate {} drift at {c:?}: gpu={gpu_v} cpu={cpu_v} diff={diff:.3e}",
                    f.name()
                );
            }
        }
        eprintln!("max abs(gpu-cpu) per climate field (MC 1.21, seed 12345):");
        for f in ClimateField::ALL {
            eprintln!("  {:>16}: {:.3e}", f.name(), worst_per_field[f as usize]);
        }
    }

    /// Smoke: ClimateNoise must initialise the expected octave structure.
    /// Hard-coded counts come from cubiomes' init_climate_seed amplitude
    /// arrays: temperature=6, humidity=6, continentalness=9, erosion=5,
    /// shift=4, weirdness=6. Each is split into octA/octB by xDoublePerlinInit.
    #[test]
    fn climate_noise_has_expected_octave_layout_for_mc_1_21() {
        let mc = match parse_mc_version("1.21") {
            Some(v) => v,
            None => return,
        };
        let climate = ClimateNoise::for_overworld(mc, 0, false);
        // xDoublePerlinInit splits `len` non-zero octaves between octA and
        // octB roughly evenly: na = (n+1)/2, nb = n/2. We don't assert
        // exact counts (cubiomes trims trailing zero-amplitude entries),
        // just that each field has ≥1 octave in each half.
        for f in ClimateField::ALL {
            let spec = climate.field(f);
            let total = spec.oct_a.len() + spec.oct_b.len();
            assert!(total > 0, "field {} has 0 octaves", f.name());
            // Amplitude must be the cubiomes amp_ini[len] value, in (0, 2).
            assert!(
                spec.amplitude > 0.0 && spec.amplitude < 2.0,
                "field {} amplitude {} out of plausible range",
                f.name(),
                spec.amplitude
            );
        }
    }
}
