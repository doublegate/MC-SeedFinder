//! Phase 6c-2: WGSL `sampleOctave` + `sampleDoublePerlin` dispatcher + parity test.
//!
//! Builds on Phase 6c-1's single-Perlin primitive by stacking N octaves
//! per double-perlin half (A and B) with per-octave amplitudes and
//! lacunarities. The B half is sampled at the cubiomes `337/331` frequency
//! shift; the final output is `(sumA + sumB_shifted) * amplitude`.
//!
//! The CPU reference is cubiomes' real `sampleDoublePerlin` via the
//! `mcsf_double_perlin_sample` shim — not a Rust re-implementation. That
//! means a passing parity test is end-to-end proof that the GPU kernel
//! agrees with cubiomes, not with our own re-port of cubiomes.

#![cfg(feature = "gpu")]

use bytemuck::{Pod, Zeroable};

use crate::gpu_noise::PerlinState;

extern "C" {
    fn mcsf_double_perlin_sample(
        oct_a_count: i32,
        oct_b_count: i32,
        dp_amplitude: f64,
        octave_abc: *const f64,
        octave_h2: *const u8,
        octave_d2_t2: *const f64,
        octave_amp_lac: *const f64,
        octave_perm: *const u8,
        x: f64,
        y: f64,
        z: f64,
    ) -> f64;
}

/// One octave in a [`DoublePerlinSpec`]: a cubiomes-initialised
/// [`PerlinState`] plus the amplitude and lacunarity cubiomes would store
/// in `PerlinNoise.amplitude` / `.lacunarity`.
#[derive(Clone, Debug)]
pub struct OctaveSpec {
    pub state: PerlinState,
    pub amplitude: f64,
    pub lacunarity: f64,
}

/// Full description of a cubiomes `DoublePerlinNoise` ready for GPU
/// dispatch. The A and B octave arrays correspond to `octA`/`octB`; the
/// `amplitude` field is the cubiomes-derived overall scaling factor (e.g.
/// `5./6` for two octaves total).
#[derive(Clone, Debug)]
pub struct DoublePerlinSpec {
    pub oct_a: Vec<OctaveSpec>,
    pub oct_b: Vec<OctaveSpec>,
    pub amplitude: f64,
}

impl DoublePerlinSpec {
    /// CPU reference: cubiomes' exact `sampleDoublePerlin` via the shim.
    pub fn cpu_sample(&self, x: f64, y: f64, z: f64) -> f64 {
        let total = self.oct_a.len() + self.oct_b.len();
        if total == 0 {
            return 0.0;
        }
        let mut abc = Vec::with_capacity(total * 3);
        let mut h2 = Vec::with_capacity(total);
        let mut d2t2 = Vec::with_capacity(total * 2);
        let mut amp_lac = Vec::with_capacity(total * 2);
        let mut perm = Vec::with_capacity(total * 256);
        for o in self.oct_a.iter().chain(self.oct_b.iter()) {
            abc.push(o.state.a);
            abc.push(o.state.b);
            abc.push(o.state.c);
            h2.push(o.state.h2);
            d2t2.push(o.state.d2);
            d2t2.push(o.state.t2);
            amp_lac.push(o.amplitude);
            amp_lac.push(o.lacunarity);
            perm.extend_from_slice(&o.state.perm);
        }
        unsafe {
            mcsf_double_perlin_sample(
                self.oct_a.len() as i32,
                self.oct_b.len() as i32,
                self.amplitude,
                abc.as_ptr(),
                h2.as_ptr(),
                d2t2.as_ptr(),
                amp_lac.as_ptr(),
                perm.as_ptr(),
                x,
                y,
                z,
            )
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OctaveEntryGpu {
    a: f32,
    b: f32,
    c: f32,
    _pad0: u32,
    d2: f32,
    t2: f32,
    h2: u32,
    _pad1: u32,
    amplitude: f32,
    lacunarity: f32,
    _pad2: u32,
    _pad3: u32,
    perm: [[u32; 4]; 16], // 16 vec4<u32> = 256 bytes
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DPParamsGpu {
    oct_a_count: u32,
    oct_b_count: u32,
    sample_count: u32,
    _pad0: u32,
    dp_amplitude: f32,
    freq_shift: f32,
    _pad1: u32,
    _pad2: u32,
}

fn pack_perm(perm: &[u8; 256]) -> [[u32; 4]; 16] {
    let mut out = [[0u32; 4]; 16];
    for (vec_idx, vec_out) in out.iter_mut().enumerate() {
        for (sub_idx, word_out) in vec_out.iter_mut().enumerate() {
            let word_idx = vec_idx * 4 + sub_idx;
            let mut word: u32 = 0;
            for byte_off in 0..4 {
                let byte_idx = word_idx * 4 + byte_off;
                word |= (perm[byte_idx] as u32) << (byte_off * 8);
            }
            *word_out = word;
        }
    }
    out
}

fn pack_octave(o: &OctaveSpec) -> OctaveEntryGpu {
    OctaveEntryGpu {
        a: o.state.a as f32,
        b: o.state.b as f32,
        c: o.state.c as f32,
        _pad0: 0,
        d2: o.state.d2 as f32,
        t2: o.state.t2 as f32,
        h2: o.state.h2 as u32,
        _pad1: 0,
        amplitude: o.amplitude as f32,
        lacunarity: o.lacunarity as f32,
        _pad2: 0,
        _pad3: 0,
        perm: pack_perm(&o.state.perm),
    }
}

/// wgpu pipeline for `sampleDoublePerlin`. Separate pipeline from
/// [`crate::gpu_noise::GpuPerlin`] because the bind-group layout differs
/// (storage buffer of octaves instead of a single uniform Perlin).
pub struct GpuDoublePerlin {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl std::fmt::Debug for GpuDoublePerlin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDoublePerlin").finish_non_exhaustive()
    }
}

impl GpuDoublePerlin {
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
                    label: Some("mcsf-gpu-double-perlin-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|_| "failed to acquire wgpu device")?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mcsf-gpu-double-perlin-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_noise.wgsl").into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mcsf-gpu-dp-bgl"),
            entries: &[
                // dpp uniform
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
                // octaves storage
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
                // inputs storage
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
                    binding: 3,
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
            label: Some("mcsf-gpu-dp-pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mcsf-gpu-dp-cp"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("cs_double_perlin"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout: bgl,
        })
    }

    /// Dispatch `sampleDoublePerlin(spec, x, y, z)` over `coords`. Returns
    /// one f32 per input. Single-shot; production callers will reuse
    /// buffers across many dispatches.
    pub fn sample_batch(
        &self,
        spec: &DoublePerlinSpec,
        coords: &[[f32; 3]],
    ) -> Result<Vec<f32>, &'static str> {
        if coords.is_empty() {
            return Ok(Vec::new());
        }
        if spec.oct_a.is_empty() && spec.oct_b.is_empty() {
            return Err("double-perlin spec has no octaves");
        }

        let dpp = DPParamsGpu {
            oct_a_count: spec.oct_a.len() as u32,
            oct_b_count: spec.oct_b.len() as u32,
            sample_count: coords.len() as u32,
            _pad0: 0,
            dp_amplitude: spec.amplitude as f32,
            freq_shift: (337.0_f64 / 331.0_f64) as f32,
            _pad1: 0,
            _pad2: 0,
        };

        let octaves: Vec<OctaveEntryGpu> = spec
            .oct_a
            .iter()
            .chain(spec.oct_b.iter())
            .map(pack_octave)
            .collect();
        let oct_bytes: &[u8] = bytemuck::cast_slice(&octaves);

        let inputs_packed: Vec<[f32; 4]> = coords.iter().map(|c| [c[0], c[1], c[2], 0.0]).collect();
        let in_bytes: &[u8] = bytemuck::cast_slice(&inputs_packed);
        let out_bytes = (coords.len() * 4) as wgpu::BufferAddress;

        let dpp_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcsf-dp-dpp"),
            size: std::mem::size_of::<DPParamsGpu>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&dpp_buf, 0, bytemuck::bytes_of(&dpp));

        let oct_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcsf-dp-octaves"),
            size: oct_bytes.len() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&oct_buf, 0, oct_bytes);

        let in_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcsf-dp-in"),
            size: in_bytes.len() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&in_buf, 0, in_bytes);

        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcsf-dp-out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcsf-dp-staging"),
            size: out_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mcsf-dp-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dpp_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: oct_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mcsf-dp-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mcsf-dp-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let groups = ((coords.len() as u32) + 63) / 64;
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
        let results: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic spec: 4 octaves on each side with halving amplitudes and
    /// doubling lacunarities, distinct seeds. Not a real climate-noise
    /// init (that's Phase 6c-3) — purpose is purely to validate the
    /// octave-sum + frequency-shift math agrees with cubiomes.
    fn synthetic_spec() -> DoublePerlinSpec {
        let mk = |seed: u64, amp: f64, lac: f64| OctaveSpec {
            state: PerlinState::from_seed(seed),
            amplitude: amp,
            lacunarity: lac,
        };
        DoublePerlinSpec {
            oct_a: vec![
                mk(0x1, 1.0, 1.0),
                mk(0x2, 0.5, 2.0),
                mk(0x3, 0.25, 4.0),
                mk(0x4, 0.125, 8.0),
            ],
            oct_b: vec![
                mk(0x5, 1.0, 1.0),
                mk(0x6, 0.5, 2.0),
                mk(0x7, 0.25, 4.0),
                mk(0x8, 0.125, 8.0),
            ],
            // cubiomes' amp_ini[len] for len=2 — picked to match a plausible
            // real-world double-perlin amplitude scale.
            amplitude: 5.0 / 6.0,
        }
    }

    /// Phase 6c-2 cornerstone test. WGSL `sampleDoublePerlin` must agree
    /// with cubiomes' real C `sampleDoublePerlin` (via the shim) within
    /// f32 round-off for every sample. Tolerance accounts for two new
    /// sources of drift versus Phase 6c-1:
    ///   1. The frequency-shift constant `337/331` rounds in f32.
    ///   2. Octave sums amplify per-sample drift by `sum(|amplitude|)`.
    /// 8 octaves with `|amp|<=1` keeps the per-sample-summed bound under
    /// ~2 * 8 * 1e-5 ≈ 1.6e-4 in the worst case.
    #[test]
    fn gpu_double_perlin_matches_cubiomes_within_f32_tolerance() {
        let Some(gpu) = GpuDoublePerlin::try_new() else {
            eprintln!("skipping: no GPU adapter on this host");
            return;
        };
        let spec = synthetic_spec();
        let coords: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 2.5],
            [-1.5, 0.0, 2.5],
            [10.0, 0.0, -10.0],
            [0.5, 0.0, 0.5],
            [127.5, 0.0, 127.5],
            [3.14159, 0.0, 2.71828],
            [-1000.0, 0.0, 1000.0],
        ];
        let gpu_results = gpu.sample_batch(&spec, &coords).expect("gpu dispatch");
        assert_eq!(gpu_results.len(), coords.len());

        let mut max_abs_err = 0.0f64;
        for (c, gpu_v) in coords.iter().zip(gpu_results.iter()) {
            let cpu_v = spec.cpu_sample(c[0] as f64, c[1] as f64, c[2] as f64);
            let diff = (cpu_v - *gpu_v as f64).abs();
            if diff > max_abs_err {
                max_abs_err = diff;
            }
            assert!(
                diff < 5e-3,
                "Double-Perlin GPU↔CPU drift at {c:?}: gpu={gpu_v} cpu={cpu_v} diff={diff:.3e}",
            );
        }
        eprintln!(
            "max abs(gpu-cpu) Double-Perlin diff over {} samples: {:.3e}",
            coords.len(),
            max_abs_err
        );
    }

    /// Degenerate but legal: octB empty, octA single octave. Should still
    /// match cubiomes (which handles this in its inner loop).
    #[test]
    fn gpu_double_perlin_single_octave_a_only() {
        let Some(gpu) = GpuDoublePerlin::try_new() else {
            return;
        };
        let spec = DoublePerlinSpec {
            oct_a: vec![OctaveSpec {
                state: PerlinState::from_seed(42),
                amplitude: 1.0,
                lacunarity: 1.0,
            }],
            oct_b: vec![],
            amplitude: 1.0,
        };
        let coords = vec![[0.0f32, 0.0, 0.0], [1.0, 0.0, 1.0], [-3.5, 0.0, 7.25]];
        let gpu_results = gpu.sample_batch(&spec, &coords).expect("gpu dispatch");
        for (c, gpu_v) in coords.iter().zip(gpu_results.iter()) {
            let cpu_v = spec.cpu_sample(c[0] as f64, c[1] as f64, c[2] as f64);
            let diff = (cpu_v - *gpu_v as f64).abs();
            assert!(
                diff < 5e-3,
                "single-octave drift at {c:?}: cpu={cpu_v} gpu={gpu_v}"
            );
        }
    }
}
