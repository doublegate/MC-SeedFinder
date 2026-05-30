//! Phase A: bit-exact WGSL Xoroshiro128++ port + parity test.
//!
//! This is the foundation for GPU per-seed biome-noise initialisation
//! (`setBiomeSeed`). cubiomes builds each climate field's Perlin octaves from a
//! Xoroshiro128++ stream seeded by the world seed; to evaluate biome conditions
//! per seed on the GPU, each thread must reproduce that stream itself.
//!
//! Unlike the float noise primitives (`gpu_noise`, `gpu_double_perlin`), this is
//! pure 64-bit integer RNG: the GPU output must be **bit-identical** to
//! cubiomes' `xNextLong`, not merely within an f32 tolerance. WGSL has no u64,
//! so `gpu_xoroshiro.wgsl` emulates it with `vec2<u32>` limbs; the test below is
//! the proof that the emulation is exact.
//!
//! Nothing here is wired into search yet — this ships the primitive and its
//! correctness proof in isolation (the smallest de-risking slice of the GPU
//! biome-conditioned search feature).

#![cfg(feature = "gpu")]

use bytemuck::{Pod, Zeroable};

extern "C" {
    /// Fill `out[0..k]` with the first `k` `xNextLong` outputs of a
    /// Xoroshiro128++ seeded from `seed`. Defined in `csrc/shim.c`.
    fn mcsf_xoroshiro_stream(seed: u64, k: std::os::raw::c_int, out: *mut u64);
    /// Run cubiomes' `xPerlinInit` from a raw Xoroshiro state `(lo, hi)` and
    /// dump 3 doubles (a, b, c) into `out_abc` and 256 perm bytes into
    /// `out_perm`. Defined in `csrc/shim.c`.
    fn mcsf_xperlin_init_dump(lo: u64, hi: u64, out_abc: *mut f64, out_perm: *mut u8);
}

// These oracles back the Phase C parity test only (the `set_biome_field_batch`
// public API takes its constants from CLIMATE_FIELDS, not cubiomes), so they
// are test-gated to avoid a dead-code warning in the non-test lib build.
#[cfg(test)]
extern "C" {
    /// cubiomes' `setBiomeSeed` octave-state dump for one climate field. Fills
    /// per-octave (octA then octB) `out_abc` (3 f64/oct), `out_amp_lac`
    /// (2 f64/oct), `out_perm` (256 bytes/oct), and the split counts. Returns
    /// the total octave count. Defined in `csrc/shim.c` (shared with
    /// `gpu_climate`). Capacity: 18 octaves/field.
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
    fn mcsf_str2mc(s: *const std::os::raw::c_char) -> i32;
}

/// CPU reference: cubiomes' Xoroshiro128++ stream for one seed.
pub fn cpu_xoroshiro_stream(seed: u64, k: u32) -> Vec<u64> {
    let mut out = vec![0u64; k as usize];
    unsafe {
        mcsf_xoroshiro_stream(seed, k as std::os::raw::c_int, out.as_mut_ptr());
    }
    out
}

/// CPU reference: cubiomes' `xPerlinInit` from a raw Xoroshiro state. Returns
/// `((a, b, c), perm[256])`.
pub fn cpu_xperlin_init(lo: u64, hi: u64) -> ((f64, f64, f64), [u8; 256]) {
    let mut abc = [0.0f64; 3];
    let mut perm = [0u8; 256];
    unsafe {
        mcsf_xperlin_init_dump(lo, hi, abc.as_mut_ptr(), perm.as_mut_ptr());
    }
    ((abc[0], abc[1], abc[2]), perm)
}

/// Result of one GPU `xPerlinInit`: the (f32) a/b/c offsets and the 256-byte
/// permutation table.
#[derive(Clone, Debug)]
pub struct PerlinInit {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub perm: [u8; 256],
}

// --- Phase C: setBiomeSeed climate-field init -------------------------------

/// Max octaves cubiomes produces for any single climate field (continentalness
/// = 9 + 9). Each half (octA/octB) is capped at 9; the GPU output reserves
/// `MAX_OCT_PER_FIELD` slots per (seed, field), octB starting at slot 9.
const MAX_OCT_PER_FIELD: usize = 18;
const MAX_OCT_PER_HALF: usize = 9;

/// cubiomes' `md5_octave_n` table (noise.c:394): md5 of "octave_-12".."octave_0".
/// One `(lo, hi)` u64 pair per octave (13 entries).
const MD5_OCTAVE: [(u64, u64); 13] = [
    (0xb198de63a8012672, 0x7b84cad43ef7b5a8), // octave_-12
    (0x0fd787bfbc403ec3, 0x74a4a31ca21b48b8), // octave_-11
    (0x36d326eed40efeb2, 0x5be9ce18223c636a), // octave_-10
    (0x082fe255f8be6631, 0x4e96119e22dedc81), // octave_-9
    (0x0ef68ec68504005e, 0x48b6bf93a2789640), // octave_-8
    (0xf11268128982754f, 0x257a1d670430b0aa), // octave_-7
    (0xe51c98ce7d1de664, 0x5f9478a733040c45), // octave_-6
    (0x6d7b49e7e429850a, 0x2e3063c622a24777), // octave_-5
    (0xbd90d5377ba1b762, 0xc07317d419a7548d), // octave_-4
    (0x53d39c6752dac858, 0xbcd1c5a80ab65b3e), // octave_-3
    (0xb4a24d7a84e7677b, 0x023ff9668e89b5c4), // octave_-2
    (0xdffa22b534c5f608, 0xb9b67517d3665ca9), // octave_-1
    (0xd50708086cef4d7c, 0x6e1651ecc7f43309), // octave_0
];

/// Seed-independent constants for one climate field's `init_climate_seed`
/// (biomenoise.c:844). `md5` is XORed into the field sub-Xoroshiro; `amps`,
/// `omin`, `len` drive `xDoublePerlinInit`. Non-large Overworld only for now.
#[derive(Clone, Copy, Debug)]
pub struct ClimateFieldConsts {
    pub md5: (u64, u64),
    pub omin: i32,
    pub len: u32,
    pub amps: &'static [f64],
}

/// The six climate fields, in cubiomes' `NP_*` order (temperature=0 ..
/// weirdness=5). Non-large constants from `init_climate_seed`.
pub const CLIMATE_FIELDS: [ClimateFieldConsts; 6] = [
    // NP_TEMPERATURE
    ClimateFieldConsts {
        md5: (0x5c7e6b29735f0d7f, 0xf7d86f1bbc734988),
        omin: -10,
        len: 6,
        amps: &[1.5, 0.0, 1.0, 0.0, 0.0, 0.0],
    },
    // NP_HUMIDITY ("vegetation")
    ClimateFieldConsts {
        md5: (0x81bb4d22e8dc168e, 0xf1c8b4bea16303cd),
        omin: -8,
        len: 6,
        amps: &[1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    },
    // NP_CONTINENTALNESS
    ClimateFieldConsts {
        md5: (0x83886c9d0ae3a662, 0xafa638a61b42e8ad),
        omin: -9,
        len: 9,
        amps: &[1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
    },
    // NP_EROSION
    ClimateFieldConsts {
        md5: (0xd02491e6058f6fd8, 0x4792512c94c17a80),
        omin: -9,
        len: 5,
        amps: &[1.0, 1.0, 0.0, 1.0, 1.0],
    },
    // NP_SHIFT ("offset")
    ClimateFieldConsts {
        md5: (0x080518cf6af25384, 0x3f3dfb40a54febd5),
        omin: -3,
        len: 4,
        amps: &[1.0, 1.0, 1.0, 0.0],
    },
    // NP_WEIRDNESS ("ridge")
    ClimateFieldConsts {
        md5: (0xefc8ef4d36102b34, 0x1beeeb324a0f24ea),
        omin: -7,
        len: 6,
        amps: &[1.0, 2.0, 1.0, 0.0, 0.0, 0.0],
    },
];

/// cubiomes' `persist_ini[len]` (noise.c:413): (2^(len-1))/(2^len - 1).
fn persist_start(len: u32) -> f64 {
    let l = len as i32;
    (1i64 << (l - 1)) as f64 / ((1i64 << l) - 1) as f64
}

/// One octave's init state as produced by the GPU. perm is bit-exact vs
/// cubiomes; the floats are f32 approximations.
#[derive(Clone, Debug)]
pub struct OctaveInit {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub amplitude: f32,
    pub lacunarity: f32,
    pub perm: [u8; 256],
}

/// One field's per-seed init: the octA and octB octave lists.
#[derive(Clone, Debug)]
pub struct FieldInit {
    pub oct_a: Vec<OctaveInit>,
    pub oct_b: Vec<OctaveInit>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n_seeds: u32,
    k: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Mirrors `PiParams` in `gpu_xoroshiro.wgsl` (Phase B perlin-init entry).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PiParams {
    n_states: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Mirrors `FieldParams` in `gpu_xoroshiro.wgsl` (Phase C setBiomeSeed entry).
/// `md5_*` and `amp` are little-endian (lo, hi) limbs / packed f32 quads.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FieldParams {
    n_seeds: u32,
    len: u32,
    omin: i32,
    _pad0: u32,
    md5_lo: [u32; 2],
    md5_hi: [u32; 2],
    persist_start: f32,
    lacuna_start: f32,
    _pad1: f32,
    _pad2: f32,
    amp: [[f32; 4]; 3],
}

/// wgpu pipeline that runs the WGSL Xoroshiro128++ port over a batch of seeds,
/// emitting `k` `xNextLong` outputs per seed. Persistent device/pipeline across
/// calls; buffers are sized per dispatch (this is a test/validation primitive).
pub struct GpuXoroshiro {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    // Phase B: per-octave Perlin init (`cs_perlin_init`). Separate layout — it
    // has a 4th binding (perm output) and two read_write storage buffers.
    pi_pipeline: wgpu::ComputePipeline,
    pi_bind_group_layout: wgpu::BindGroupLayout,
    // Phase C: per-seed setBiomeSeed field init (`cs_set_biome_field`).
    // 6 bindings: uniform + seeds(ro) + md5(ro) + scalars(rw) + perm(rw) +
    // counts(rw).
    sbs_pipeline: wgpu::ComputePipeline,
    sbs_bind_group_layout: wgpu::BindGroupLayout,
}

impl std::fmt::Debug for GpuXoroshiro {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuXoroshiro").finish_non_exhaustive()
    }
}

impl GpuXoroshiro {
    pub fn try_new() -> Option<Self> {
        pollster::block_on(Self::async_new()).ok()
    }

    async fn async_new() -> Result<Self, &'static str> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or("no compatible GPU adapter found")?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("mcsf-gpu-xoroshiro-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|_| "failed to acquire wgpu device")?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mcsf-gpu-xoroshiro-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_xoroshiro.wgsl").into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mcsf-gpu-xoroshiro-bgl"),
            entries: &[
                // params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // seeds storage
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // outputs storage
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mcsf-gpu-xoroshiro-pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mcsf-gpu-xoroshiro-cp"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("cs_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // Phase B: cs_perlin_init layout — uniform params, read-only states,
        // and two read_write storage outputs (abc + perm).
        let storage_rw = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let storage_ro = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let pi_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mcsf-gpu-perlin-init-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: storage_ro,
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: storage_rw,
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: storage_rw,
                    count: None,
                },
            ],
        });
        let pi_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mcsf-gpu-perlin-init-pl"),
            bind_group_layouts: &[&pi_bgl],
            push_constant_ranges: &[],
        });
        let pi_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mcsf-gpu-perlin-init-cp"),
            layout: Some(&pi_pl),
            module: &shader,
            entry_point: Some("cs_perlin_init"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // Phase C: cs_set_biome_field — uniform + 2 read-only storage (seeds,
        // md5) + 3 read_write storage (scalars, perm, counts).
        let uniform_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let sbs_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mcsf-gpu-setbiome-bgl"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
            ],
        });
        let sbs_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mcsf-gpu-setbiome-pl"),
            bind_group_layouts: &[&sbs_bgl],
            push_constant_ranges: &[],
        });
        let sbs_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mcsf-gpu-setbiome-cp"),
            layout: Some(&sbs_pl),
            module: &shader,
            entry_point: Some("cs_set_biome_field"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout: bgl,
            pi_pipeline,
            pi_bind_group_layout: pi_bgl,
            sbs_pipeline,
            sbs_bind_group_layout: sbs_bgl,
        })
    }

    /// Run the WGSL Xoroshiro128++ over `seeds`, emitting `k` `xNextLong`
    /// outputs per seed. Output is flat: `result[i*k + j]` is the j-th draw for
    /// `seeds[i]`. u64 is shipped to/from the GPU as (lo32, hi32) pairs.
    pub fn stream_batch(&self, seeds: &[u64], k: u32) -> Result<Vec<u64>, &'static str> {
        if seeds.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let n = seeds.len();
        let total = n * k as usize;

        let params = Params {
            n_seeds: n as u32,
            k,
            _pad0: 0,
            _pad1: 0,
        };
        // Seeds as (lo, hi) u32 pairs.
        let seeds_packed: Vec<[u32; 2]> = seeds
            .iter()
            .map(|&s| [(s & 0xFFFF_FFFF) as u32, (s >> 32) as u32])
            .collect();
        let seeds_bytes: &[u8] = bytemuck::cast_slice(&seeds_packed);
        let out_bytes = (total * 8) as wgpu::BufferAddress; // 8 bytes per u64

        let params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcsf-xoro-params"),
            size: std::mem::size_of::<Params>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        let seeds_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcsf-xoro-seeds"),
            size: seeds_bytes.len() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&seeds_buf, 0, seeds_bytes);

        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcsf-xoro-out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcsf-xoro-staging"),
            size: out_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mcsf-xoro-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: seeds_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mcsf-xoro-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mcsf-xoro-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let groups = (n as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_bytes);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| "wgpu map channel closed")?
            .map_err(|_| "wgpu map_async failed")?;

        let data = slice.get_mapped_range();
        // Each output u64 arrives as (lo32, hi32).
        let words: &[u32] = bytemuck::cast_slice(&data);
        let mut results = Vec::with_capacity(total);
        for i in 0..total {
            let lo = words[i * 2] as u64;
            let hi = words[i * 2 + 1] as u64;
            results.push((hi << 32) | lo);
        }
        drop(data);
        staging.unmap();
        Ok(results)
    }

    /// Run the WGSL `xPerlinInit` (Phase B) over a batch of raw Xoroshiro
    /// states `(lo, hi)`, returning one [`PerlinInit`] per state. The perm
    /// table is bit-exact vs cubiomes; a/b/c are f32 approximations.
    pub fn perlin_init_batch(
        &self,
        states: &[(u64, u64)],
    ) -> Result<Vec<PerlinInit>, &'static str> {
        if states.is_empty() {
            return Ok(Vec::new());
        }
        let n = states.len();

        let params = PiParams {
            n_states: n as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        // States as [lo_lo, lo_hi, hi_lo, hi_hi] u32 quads (two vec2<u32>).
        let mut states_packed: Vec<[u32; 2]> = Vec::with_capacity(n * 2);
        for &(lo, hi) in states {
            states_packed.push([(lo & 0xFFFF_FFFF) as u32, (lo >> 32) as u32]);
            states_packed.push([(hi & 0xFFFF_FFFF) as u32, (hi >> 32) as u32]);
        }
        let states_bytes: &[u8] = bytemuck::cast_slice(&states_packed);
        let abc_bytes = (n * 16) as wgpu::BufferAddress; // vec4<f32> per state
        let perm_bytes = (n * 256) as wgpu::BufferAddress; // 64 u32 per state

        let params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcsf-pi-params"),
            size: std::mem::size_of::<PiParams>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        let states_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcsf-pi-states"),
            size: states_bytes.len() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&states_buf, 0, states_bytes);

        let abc_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcsf-pi-abc"),
            size: abc_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let perm_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcsf-pi-perm"),
            size: perm_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let abc_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcsf-pi-abc-staging"),
            size: abc_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let perm_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcsf-pi-perm-staging"),
            size: perm_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mcsf-pi-bg"),
            layout: &self.pi_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: states_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: abc_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: perm_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mcsf-pi-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mcsf-pi-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pi_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let groups = (n as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&abc_buf, 0, &abc_staging, 0, abc_bytes);
        encoder.copy_buffer_to_buffer(&perm_buf, 0, &perm_staging, 0, perm_bytes);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Map both staging buffers, then poll once.
        let abc_slice = abc_staging.slice(..);
        let perm_slice = perm_staging.slice(..);
        let (atx, arx) = std::sync::mpsc::channel();
        let (ptx, prx) = std::sync::mpsc::channel();
        abc_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = atx.send(res);
        });
        perm_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = ptx.send(res);
        });
        let _ = self.device.poll(wgpu::Maintain::Wait);
        arx.recv()
            .map_err(|_| "wgpu map channel closed")?
            .map_err(|_| "wgpu abc map_async failed")?;
        prx.recv()
            .map_err(|_| "wgpu map channel closed")?
            .map_err(|_| "wgpu perm map_async failed")?;

        let abc_data = abc_slice.get_mapped_range();
        let perm_data = perm_slice.get_mapped_range();
        let abc_f: &[f32] = bytemuck::cast_slice(&abc_data);
        let perm_u: &[u8] = &perm_data;

        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut perm = [0u8; 256];
            perm.copy_from_slice(&perm_u[i * 256..(i + 1) * 256]);
            out.push(PerlinInit {
                a: abc_f[i * 4],
                b: abc_f[i * 4 + 1],
                c: abc_f[i * 4 + 2],
                perm,
            });
        }
        drop(abc_data);
        drop(perm_data);
        abc_staging.unmap();
        perm_staging.unmap();
        Ok(out)
    }

    /// Run the WGSL `cs_set_biome_field` (Phase C) over a batch of world seeds
    /// for ONE climate `field`, reproducing cubiomes' per-seed `setBiomeSeed`
    /// init entirely on the GPU. Returns one [`FieldInit`] (octA + octB octave
    /// lists) per seed. Perm tables are bit-exact vs cubiomes; the scalar
    /// fields (a/b/c/amplitude/lacunarity) are f32 approximations.
    pub fn set_biome_field_batch(
        &self,
        seeds: &[u64],
        field: ClimateFieldConsts,
    ) -> Result<Vec<FieldInit>, &'static str> {
        if seeds.is_empty() {
            return Ok(Vec::new());
        }
        let n = seeds.len();
        let oct_slots = n * MAX_OCT_PER_FIELD;

        // Uniform: pad the amplitude array (<=9 used) into 3 vec4<f32>.
        let mut amp = [[0.0f32; 4]; 3];
        for (i, &a) in field.amps.iter().enumerate() {
            amp[i >> 2][i & 3] = a as f32;
        }
        let params = FieldParams {
            n_seeds: n as u32,
            len: field.len,
            omin: field.omin,
            _pad0: 0,
            md5_lo: [
                (field.md5.0 & 0xFFFF_FFFF) as u32,
                (field.md5.0 >> 32) as u32,
            ],
            md5_hi: [
                (field.md5.1 & 0xFFFF_FFFF) as u32,
                (field.md5.1 >> 32) as u32,
            ],
            persist_start: persist_start(field.len) as f32,
            // cubiomes lacuna_ini[-omin] = 2^omin (omin is negative).
            lacuna_start: 2.0f32.powi(field.omin),
            _pad1: 0.0,
            _pad2: 0.0,
            amp,
        };

        let seeds_packed: Vec<[u32; 2]> = seeds
            .iter()
            .map(|&s| [(s & 0xFFFF_FFFF) as u32, (s >> 32) as u32])
            .collect();
        // md5 octave table as (lo, hi) u32 limb pairs, 2 per octave.
        let mut md5_packed: Vec<[u32; 2]> = Vec::with_capacity(MD5_OCTAVE.len() * 2);
        for &(lo, hi) in MD5_OCTAVE.iter() {
            md5_packed.push([(lo & 0xFFFF_FFFF) as u32, (lo >> 32) as u32]);
            md5_packed.push([(hi & 0xFFFF_FFFF) as u32, (hi >> 32) as u32]);
        }

        let params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcsf-sbs-params"),
            size: std::mem::size_of::<FieldParams>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        let seeds_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcsf-sbs-seeds"),
            size: bytemuck::cast_slice::<_, u8>(&seeds_packed).len() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&seeds_buf, 0, bytemuck::cast_slice(&seeds_packed));

        let md5_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcsf-sbs-md5"),
            size: bytemuck::cast_slice::<_, u8>(&md5_packed).len() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&md5_buf, 0, bytemuck::cast_slice(&md5_packed));

        let scalars_bytes = (oct_slots * 8 * 4) as wgpu::BufferAddress; // 8 f32/oct
        let perm_bytes = (oct_slots * 256) as wgpu::BufferAddress; // 64 u32/oct
        let counts_bytes = (n * 2 * 4) as wgpu::BufferAddress; // 2 u32/seed

        let make_storage = |label, size| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let make_staging = |label, size| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        let scalars_buf = make_storage("mcsf-sbs-scalars", scalars_bytes);
        let perm_buf = make_storage("mcsf-sbs-perm", perm_bytes);
        let counts_buf = make_storage("mcsf-sbs-counts", counts_bytes);
        let scalars_staging = make_staging("mcsf-sbs-scalars-staging", scalars_bytes);
        let perm_staging = make_staging("mcsf-sbs-perm-staging", perm_bytes);
        let counts_staging = make_staging("mcsf-sbs-counts-staging", counts_bytes);

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mcsf-sbs-bg"),
            layout: &self.sbs_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: seeds_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: md5_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scalars_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: perm_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: counts_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mcsf-sbs-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mcsf-sbs-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sbs_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let groups = (n as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&scalars_buf, 0, &scalars_staging, 0, scalars_bytes);
        encoder.copy_buffer_to_buffer(&perm_buf, 0, &perm_staging, 0, perm_bytes);
        encoder.copy_buffer_to_buffer(&counts_buf, 0, &counts_staging, 0, counts_bytes);
        self.queue.submit(std::iter::once(encoder.finish()));

        let sc_slice = scalars_staging.slice(..);
        let pm_slice = perm_staging.slice(..);
        let ct_slice = counts_staging.slice(..);
        let (stx, srx) = std::sync::mpsc::channel();
        let (ptx, prx) = std::sync::mpsc::channel();
        let (ctx, crx) = std::sync::mpsc::channel();
        sc_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = stx.send(r);
        });
        pm_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = ptx.send(r);
        });
        ct_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = ctx.send(r);
        });
        let _ = self.device.poll(wgpu::Maintain::Wait);
        srx.recv()
            .map_err(|_| "wgpu map channel closed")?
            .map_err(|_| "wgpu scalars map_async failed")?;
        prx.recv()
            .map_err(|_| "wgpu map channel closed")?
            .map_err(|_| "wgpu perm map_async failed")?;
        crx.recv()
            .map_err(|_| "wgpu map channel closed")?
            .map_err(|_| "wgpu counts map_async failed")?;

        let sc_data = sc_slice.get_mapped_range();
        let pm_data = pm_slice.get_mapped_range();
        let ct_data = ct_slice.get_mapped_range();
        let scalars: &[f32] = bytemuck::cast_slice(&sc_data);
        let perm_u: &[u8] = &pm_data;
        let counts: &[u32] = bytemuck::cast_slice(&ct_data);

        let read_octave = |gidx: usize| -> OctaveInit {
            let sb = gidx * 8;
            let mut perm = [0u8; 256];
            perm.copy_from_slice(&perm_u[gidx * 256..(gidx + 1) * 256]);
            OctaveInit {
                a: scalars[sb],
                b: scalars[sb + 1],
                c: scalars[sb + 2],
                amplitude: scalars[sb + 3],
                lacunarity: scalars[sb + 4],
                perm,
            }
        };

        let mut out = Vec::with_capacity(n);
        for s in 0..n {
            let na = counts[2 * s] as usize;
            let nb = counts[2 * s + 1] as usize;
            let base = s * MAX_OCT_PER_FIELD;
            let oct_a = (0..na).map(|i| read_octave(base + i)).collect();
            let oct_b = (0..nb)
                .map(|i| read_octave(base + MAX_OCT_PER_HALF + i))
                .collect();
            out.push(FieldInit { oct_a, oct_b });
        }
        drop(sc_data);
        drop(pm_data);
        drop(ct_data);
        scalars_staging.unmap();
        perm_staging.unmap();
        counts_staging.unmap();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Phase A cornerstone. The WGSL Xoroshiro128++ stream must be
    /// BIT-IDENTICAL to cubiomes' `xNextLong` for every seed and every draw —
    /// this is integer RNG, so there is zero tolerance. A green test proves the
    /// u32-limb u64 emulation (add64/shl64/shr64/rotl64/mul64 and the xSetSeed
    /// scrambler) is exact, unblocking the per-seed octave/permutation init.
    #[test]
    fn gpu_xoroshiro_matches_cubiomes_bit_exact() {
        let Some(gpu) = GpuXoroshiro::try_new() else {
            eprintln!("skipping: no GPU adapter on this host");
            return;
        };

        // Seeds chosen to exercise: zero, small, large, high-bit-set, classic
        // worldgen test seeds, and signed-negative-as-u64 values.
        let seeds: Vec<u64> = vec![
            0,
            1,
            42,
            12345,
            -1i64 as u64,
            -12345i64 as u64,
            0xDEAD_BEEFu64,
            0x0123_4567_89AB_CDEFu64,
            0xFFFF_FFFF_FFFF_FFFFu64,
            0x8000_0000_0000_0000u64,
            1_234_567_890_123_456_789u64,
        ];
        let k: u32 = 24;

        let gpu_stream = gpu.stream_batch(&seeds, k).expect("gpu dispatch");
        assert_eq!(gpu_stream.len(), seeds.len() * k as usize);

        for (si, &seed) in seeds.iter().enumerate() {
            let cpu = cpu_xoroshiro_stream(seed, k);
            for j in 0..k as usize {
                let g = gpu_stream[si * k as usize + j];
                let c = cpu[j];
                assert_eq!(
                    g, c,
                    "Xoroshiro mismatch seed={seed:#018x} draw={j}: gpu={g:#018x} cpu={c:#018x}"
                );
            }
        }
    }

    /// Independent sanity check on the seeding path: the very first draw for a
    /// handful of seeds must match cubiomes exactly. Cheap, and isolates
    /// xSetSeed (which uses mul64) from the per-draw xNextLong loop.
    #[test]
    fn gpu_xoroshiro_first_draw_matches_cubiomes() {
        let Some(gpu) = GpuXoroshiro::try_new() else {
            return;
        };
        for seed in [0u64, 1, 2, 100, 1_000_000, u64::MAX] {
            let g = gpu.stream_batch(&[seed], 1).unwrap()[0];
            let c = cpu_xoroshiro_stream(seed, 1)[0];
            assert_eq!(g, c, "seed {seed:#018x}: gpu={g:#018x} cpu={c:#018x}");
        }
    }

    /// Phase B cornerstone. cubiomes' `xPerlinInit` builds each octave's
    /// 256-byte permutation table via a Fisher-Yates shuffle driven by
    /// `xNextInt` (pure integer) — so the GPU table must be BIT-IDENTICAL to
    /// cubiomes for every input state. The a/b/c offsets come from
    /// `xNextDouble` and are only f32-tolerant. A green perm-table assertion
    /// proves the GPU consumes the RNG stream in lockstep with cubiomes
    /// (correct draw order and count), which is the load-bearing property for
    /// per-seed `setBiomeSeed` in later phases.
    #[test]
    fn gpu_xperlin_init_perm_table_bit_exact() {
        let Some(gpu) = GpuXoroshiro::try_new() else {
            eprintln!("skipping: no GPU adapter on this host");
            return;
        };

        // Raw Xoroshiro states (lo, hi). xPerlinInit takes a state directly, so
        // we exercise assorted bit patterns (incl. all-zero, all-one, and the
        // post-xSetSeed states of a few real seeds) rather than only seeds.
        let mut states: Vec<(u64, u64)> = vec![
            (0, 0),
            (1, 0),
            (0, 1),
            (u64::MAX, u64::MAX),
            (0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210),
            (0x9E37_79B9_7F4A_7C15, 0x6A09_E667_F3BC_C909),
        ];
        // Also use the seeded states of real world seeds: xSetSeed then read
        // the first xNextLong pair as a representative downstream state.
        for seed in [0u64, 1, 42, 12345] {
            let s = cpu_xoroshiro_stream(seed, 2);
            states.push((s[0], s[1]));
        }

        let gpu_inits = gpu.perlin_init_batch(&states).expect("gpu dispatch");
        assert_eq!(gpu_inits.len(), states.len());

        let mut max_abc_err = 0.0f64;
        for (i, &(lo, hi)) in states.iter().enumerate() {
            let ((ca, cb, cc), cperm) = cpu_xperlin_init(lo, hi);
            let g = &gpu_inits[i];

            // Permutation table: BIT-EXACT, byte for byte.
            assert_eq!(
                g.perm, cperm,
                "perm-table mismatch for state ({lo:#018x},{hi:#018x})"
            );

            // a/b/c: f32-tolerant. cubiomes values are in [0, 256).
            for (gv, cv, name) in [(g.a, ca, "a"), (g.b, cb, "b"), (g.c, cc, "c")] {
                let diff = (cv - gv as f64).abs();
                if diff > max_abc_err {
                    max_abc_err = diff;
                }
                assert!(
                    diff < 1e-2,
                    "{name} offset drift too large for state ({lo:#018x},{hi:#018x}): \
                     gpu={gv} cpu={cv} diff={diff:.3e}"
                );
            }
        }
        eprintln!("max abs(gpu-cpu) xPerlinInit a/b/c diff: {max_abc_err:.3e}");
    }

    /// CPU oracle: cubiomes' `setBiomeSeed` octave state for one field.
    /// Returns (octA octaves, octB octaves) where each octave is
    /// (a, b, c, amplitude, lacunarity, perm[256]).
    #[allow(clippy::type_complexity)]
    fn cpu_field_octaves(
        mc: i32,
        seed: u64,
        large: bool,
        field_idx: i32,
    ) -> (
        Vec<(f64, f64, f64, f64, f64, [u8; 256])>,
        Vec<(f64, f64, f64, f64, f64, [u8; 256])>,
    ) {
        const CAP: usize = 18;
        let mut a_count = 0i32;
        let mut b_count = 0i32;
        let mut dp_amp = 0.0f64;
        let mut abc = vec![0.0f64; CAP * 3];
        let mut h2 = vec![0u8; CAP];
        let mut d2_t2 = vec![0.0f64; CAP * 2];
        let mut amp_lac = vec![0.0f64; CAP * 2];
        let mut perm = vec![0u8; CAP * 256];
        let total = unsafe {
            mcsf_climate_init_field(
                mc,
                seed,
                large as i32,
                field_idx,
                &mut a_count,
                &mut b_count,
                &mut dp_amp,
                abc.as_mut_ptr(),
                h2.as_mut_ptr(),
                d2_t2.as_mut_ptr(),
                amp_lac.as_mut_ptr(),
                perm.as_mut_ptr(),
            )
        };
        assert!(total >= 0);
        let read = |i: usize| {
            let mut p = [0u8; 256];
            p.copy_from_slice(&perm[i * 256..(i + 1) * 256]);
            (
                abc[i * 3],
                abc[i * 3 + 1],
                abc[i * 3 + 2],
                amp_lac[i * 2],
                amp_lac[i * 2 + 1],
                p,
            )
        };
        let a = a_count as usize;
        let b = b_count as usize;
        let oct_a = (0..a).map(read).collect();
        let oct_b = (0..b).map(|i| read(a + i)).collect();
        (oct_a, oct_b)
    }

    /// Phase C cornerstone. The GPU reproduces cubiomes' per-seed
    /// `setBiomeSeed` climate-field init (xSetSeed -> field md5 -> xOctaveInit
    /// -> xPerlinInit per octave) entirely on-device. For every (seed, field):
    /// the per-half octave COUNTS must match, every octave's 256-byte perm
    /// table must be BIT-IDENTICAL, and a/b/c/amplitude/lacunarity must match
    /// within f32 tolerance. Bit-exact perm tables prove the GPU walks the RNG
    /// stream in exact lockstep across all six fields — the property the GPU
    /// search kernel relies on to evaluate biome conditions per seed.
    #[test]
    fn gpu_set_biome_field_matches_cubiomes() {
        let Some(gpu) = GpuXoroshiro::try_new() else {
            eprintln!("skipping: no GPU adapter on this host");
            return;
        };
        // MC 1.21, non-large Overworld.
        let mc = {
            let c = std::ffi::CString::new("1.21").unwrap();
            unsafe { mcsf_str2mc(c.as_ptr()) }
        };
        if mc <= 0 {
            eprintln!("skipping: cubiomes does not recognise 1.21");
            return;
        }

        let seeds: Vec<u64> = vec![0, 1, 42, 12345, 0xDEAD_BEEF, -1i64 as u64];
        let mut max_scalar_err = 0.0f64;

        for (fi, field) in CLIMATE_FIELDS.iter().enumerate() {
            let gpu_inits = gpu
                .set_biome_field_batch(&seeds, *field)
                .expect("gpu dispatch");
            assert_eq!(gpu_inits.len(), seeds.len());

            for (si, &seed) in seeds.iter().enumerate() {
                let (cpu_a, cpu_b) = cpu_field_octaves(mc, seed, false, fi as i32);
                let g = &gpu_inits[si];

                assert_eq!(
                    (g.oct_a.len(), g.oct_b.len()),
                    (cpu_a.len(), cpu_b.len()),
                    "field {fi} seed {seed:#018x}: octave count mismatch"
                );

                for (half_name, gh, ch) in [("octA", &g.oct_a, &cpu_a), ("octB", &g.oct_b, &cpu_b)]
                {
                    for (oi, (go, co)) in gh.iter().zip(ch.iter()).enumerate() {
                        assert_eq!(
                            go.perm, co.5,
                            "field {fi} seed {seed:#018x} {half_name}[{oi}]: perm mismatch"
                        );
                        for (gv, cv, name) in [
                            (go.a, co.0, "a"),
                            (go.b, co.1, "b"),
                            (go.c, co.2, "c"),
                            (go.amplitude, co.3, "amplitude"),
                            (go.lacunarity, co.4, "lacunarity"),
                        ] {
                            let diff = (cv - gv as f64).abs();
                            // a/b/c in [0,256); amplitude/lacunarity small.
                            // Use a relative-ish absolute bound.
                            let tol = 1e-2 + cv.abs() * 1e-4;
                            if diff > max_scalar_err {
                                max_scalar_err = diff;
                            }
                            assert!(
                                diff < tol,
                                "field {fi} seed {seed:#018x} {half_name}[{oi}] {name}: \
                                 gpu={gv} cpu={cv} diff={diff:.3e} tol={tol:.3e}"
                            );
                        }
                    }
                }
            }
        }
        eprintln!("max abs(gpu-cpu) setBiomeSeed octave scalar diff: {max_scalar_err:.3e}");
    }
}
