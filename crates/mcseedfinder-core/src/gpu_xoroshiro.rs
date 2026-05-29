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
}

/// CPU reference: cubiomes' Xoroshiro128++ stream for one seed.
pub fn cpu_xoroshiro_stream(seed: u64, k: u32) -> Vec<u64> {
    let mut out = vec![0u64; k as usize];
    unsafe {
        mcsf_xoroshiro_stream(seed, k as std::os::raw::c_int, out.as_mut_ptr());
    }
    out
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n_seeds: u32,
    k: u32,
    _pad0: u32,
    _pad1: u32,
}

/// wgpu pipeline that runs the WGSL Xoroshiro128++ port over a batch of seeds,
/// emitting `k` `xNextLong` outputs per seed. Persistent device/pipeline across
/// calls; buffers are sized per dispatch (this is a test/validation primitive).
pub struct GpuXoroshiro {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
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
        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout: bgl,
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
}
