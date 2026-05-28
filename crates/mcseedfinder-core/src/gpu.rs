//! wgpu compute prefilter for the structure-RNG hot path.
//!
//! Phase 6 ships GPU acceleration for the most common case — a single
//! `NearbyStructure` predicate — dispatched in parallel over a contiguous
//! seed range. The WGSL kernel (`gpu.wgsl`) implements Java's 48-bit LCG and
//! the structure placement step bit-identically to the CPU code in
//! `java_random.rs` / `structures.rs`; a parity test below dispatches a
//! moderate seed range on both paths and asserts identical match lists.
//!
//! Init is async (wgpu's `request_adapter` is async); we drive it with
//! `pollster::block_on` so the rest of the code stays synchronous. If no
//! adapter is available (headless CI, no Vulkan/Metal/DX12 driver), `new`
//! returns `None` and the caller falls back to the CPU path. Same shape
//! covers older GPUs that fail pipeline creation.

#![cfg(feature = "gpu")]

use bytemuck::{Pod, Zeroable};

use crate::structures::{structure_config, SpreadType, StructureType};

/// Hand the host's i64 region multipliers to the shader as u32 limbs (so the
/// shader doesn't have to hard-code hex, which would risk transcription bugs).
const REGION_MUL_X: i64 = 341_873_128_712;
const REGION_MUL_Z: i64 = 132_897_987_541;

fn i64_to_limbs(v: i64) -> (u32, u32) {
    let u = v as u64;
    (u as u32, (u >> 32) as u32)
}

/// Uniform layout — must match the WGSL `SearchParams` struct exactly. The
/// `#[repr(C)]` + explicit ordering + 16-byte alignment make the wgpu uniform
/// buffer happy without padding shenanigans.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SearchParamsUniform {
    start_seed_lo: u32,
    start_seed_hi: u32,
    count: u32,
    spread_type: u32,
    salt_lo: u32,
    salt_hi: u32,
    spacing: i32,
    chunk_range: u32,
    rx_min: i32,
    rx_max: i32,
    rz_min: i32,
    rz_max: i32,
    centre_x: i32,
    centre_z: i32,
    region_mul_x_lo: u32,
    region_mul_x_hi: u32,
    region_mul_z_lo: u32,
    region_mul_z_hi: u32,
    max_dist_sq_lo: u32,
    max_dist_sq_hi: u32,
    // Pad up to a 16-byte multiple to keep wgpu's strict uniform layout happy.
    _pad0: u32,
    _pad1: u32,
}

/// One dispatch is bounded to `MAX_SEEDS_PER_DISPATCH` so we never blow past
/// the typical 128 MiB storage-buffer limit on modest adapters. Callers loop.
const MAX_SEEDS_PER_DISPATCH: u32 = 1_000_000;

/// Persistent GPU resources. Initialise once per process and reuse.
///
/// In v1 every dispatch allocated fresh params/output/staging buffers and a
/// fresh bind group. That per-dispatch churn was a large fraction of the
/// total GPU time (wgpu's docs explicitly call this out as the most common
/// performance mistake in compute pipelines). v2 pre-allocates a single set
/// of buffers sized for the max chunk and reuses them across dispatches;
/// per-dispatch we only `queue.write_buffer` the small (88-byte) uniform.
pub struct GpuSearcher {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    /// Persistent buffers + bind group reused across dispatches.
    params_buffer: wgpu::Buffer,
    out_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl std::fmt::Debug for GpuSearcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuSearcher").finish_non_exhaustive()
    }
}

impl GpuSearcher {
    /// Try to initialise wgpu. Returns `None` if no compatible adapter is
    /// reachable or pipeline creation fails — callers fall back to CPU.
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
                    label: Some("mcsf-gpu-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None, // optional trace path (debug only)
            )
            .await
            .map_err(|_| "failed to acquire wgpu device")?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mcsf-gpu-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mcsf-gpu-bgl"),
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
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mcsf-gpu-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mcsf-gpu-cp"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // Persistent buffers sized for the maximum chunk. Per-dispatch we only
        // queue.write_buffer the small uniform; the output + staging buffers
        // are reused. Bind group is built once because all bindings point at
        // these stable buffers.
        let params_size = std::mem::size_of::<SearchParamsUniform>() as wgpu::BufferAddress;
        let out_size = (MAX_SEEDS_PER_DISPATCH as wgpu::BufferAddress) * 4;
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcsf-gpu-params"),
            size: params_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcsf-gpu-out"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcsf-gpu-staging"),
            size: out_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mcsf-gpu-bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            params_buffer,
            out_buffer,
            staging_buffer,
            bind_group,
        })
    }

    /// Run the GPU prefilter for "structure within `max_distance` blocks of
    /// `(centre_x, centre_z)`" across `[start_seed, start_seed + count)`.
    /// Returns the matching seeds in ascending order. Strongholds are out of
    /// scope (they use a different RNG path) — caller must filter those out.
    pub fn find_nearby_structure_matches(
        &self,
        start_seed: i64,
        count: u64,
        structure: StructureType,
        max_distance: i32,
        centre_x: i32,
        centre_z: i32,
    ) -> Result<Vec<i64>, &'static str> {
        if structure == StructureType::Stronghold {
            return Err("strongholds use iter_strongholds, not the GPU prefilter");
        }
        if max_distance < 0 {
            return Err("max_distance must be non-negative");
        }
        if count == 0 {
            return Ok(Vec::new());
        }

        let cfg = structure_config(structure);
        // Region scan bounds mirror has_structure_in_radius's region walk.
        let chunk_radius = max_distance / 16 + 1;
        let cx_chunk = centre_x.div_euclid(16);
        let cz_chunk = centre_z.div_euclid(16);
        let cx_min = cx_chunk - chunk_radius;
        let cx_max = cx_chunk + chunk_radius;
        let cz_min = cz_chunk - chunk_radius;
        let cz_max = cz_chunk + chunk_radius;
        let rx_min = cx_min.div_euclid(cfg.spacing);
        let rx_max = cx_max.div_euclid(cfg.spacing);
        let rz_min = cz_min.div_euclid(cfg.spacing);
        let rz_max = cz_max.div_euclid(cfg.spacing);

        let (salt_lo, salt_hi) = i64_to_limbs(cfg.salt);
        let (rmx_lo, rmx_hi) = i64_to_limbs(REGION_MUL_X);
        let (rmz_lo, rmz_hi) = i64_to_limbs(REGION_MUL_Z);
        let max_dist_sq: u64 = (max_distance as u64) * (max_distance as u64);

        let mut matches: Vec<i64> = Vec::new();
        let mut offset: u64 = 0;
        while offset < count {
            let this_count = ((count - offset).min(MAX_SEEDS_PER_DISPATCH as u64)) as u32;
            let chunk_start = start_seed.wrapping_add(offset as i64);
            let (ss_lo, ss_hi) = i64_to_limbs(chunk_start);

            let params = SearchParamsUniform {
                start_seed_lo: ss_lo,
                start_seed_hi: ss_hi,
                count: this_count,
                spread_type: match cfg.spread_type {
                    SpreadType::Linear => 0,
                    SpreadType::Triangular => 1,
                },
                salt_lo,
                salt_hi,
                spacing: cfg.spacing,
                chunk_range: cfg.chunk_range() as u32,
                rx_min,
                rx_max,
                rz_min,
                rz_max,
                centre_x,
                centre_z,
                region_mul_x_lo: rmx_lo,
                region_mul_x_hi: rmx_hi,
                region_mul_z_lo: rmz_lo,
                region_mul_z_hi: rmz_hi,
                max_dist_sq_lo: max_dist_sq as u32,
                max_dist_sq_hi: (max_dist_sq >> 32) as u32,
                _pad0: 0,
                _pad1: 0,
            };
            let chunk_matches = self.dispatch_one(&params)?;
            matches.extend(
                chunk_matches
                    .into_iter()
                    .map(|i| chunk_start.wrapping_add(i as i64)),
            );
            offset += this_count as u64;
        }
        Ok(matches)
    }

    /// Drive one dispatch end-to-end using the persistent buffers + bind
    /// group. Only the small uniform is uploaded per call; output/staging
    /// are reused across dispatches.
    fn dispatch_one(&self, params: &SearchParamsUniform) -> Result<Vec<u32>, &'static str> {
        let count = params.count as u64;
        let out_byte_size = count * 4; // u32 per seed

        // Upload the per-dispatch uniform. (wgpu's queue.write_buffer is the
        // standard path for small/frequent uploads.)
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mcsf-gpu-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mcsf-gpu-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            // workgroup_size in WGSL is 64; dispatch ceil(count/64) workgroups.
            let groups = ((count as u32) + 63) / 64;
            pass.dispatch_workgroups(groups, 1, 1);
        }
        // Copy only the live portion (count u32s) — staging is sized for max
        // chunk but the trailing slots aren't meaningful for this dispatch.
        encoder.copy_buffer_to_buffer(&self.out_buffer, 0, &self.staging_buffer, 0, out_byte_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back. wgpu's map API is async; pollster blocks the calling thread.
        let slice = self.staging_buffer.slice(..out_byte_size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        // Drive the device until the map completes. wgpu 23's poll returns
        // MaintainResult (not Result) — discard it; the actual signal that
        // the buffer is ready comes through the map_async callback channel.
        let _ = self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| "wgpu map channel closed")?
            .map_err(|_| "wgpu map_async failed")?;

        let data = slice.get_mapped_range();
        let results: &[u32] = bytemuck::cast_slice(&data);
        let matches: Vec<u32> = results
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v != 0 { Some(i as u32) } else { None })
            .collect();
        drop(data);
        // Unmap so the staging buffer can be re-bound for the next dispatch.
        self.staging_buffer.unmap();
        Ok(matches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::{has_structure_in_radius, StructureRequirement};

    /// The reason this whole file exists: GPU output must match CPU output
    /// bit-for-bit. If parity fails this test surfaces it loudly with the
    /// disagreeing seeds. Skipped automatically on hosts without a GPU.
    #[test]
    fn gpu_matches_cpu_for_village_within_1000() {
        let Some(searcher) = GpuSearcher::try_new() else {
            eprintln!("skipping: no GPU adapter on this host");
            return;
        };

        let start: i64 = 1;
        let count: u64 = 2_000;
        let req = StructureRequirement {
            structure: StructureType::Village,
            max_distance: 1000,
            centre_x: 0,
            centre_z: 0,
        };

        let gpu_matches = searcher
            .find_nearby_structure_matches(
                start,
                count,
                req.structure,
                req.max_distance,
                req.centre_x,
                req.centre_z,
            )
            .expect("gpu dispatch");
        let cpu_matches: Vec<i64> = (0..count)
            .filter_map(|i| {
                let seed = start.wrapping_add(i as i64);
                has_structure_in_radius(seed, &req).then_some(seed)
            })
            .collect();
        assert_eq!(
            gpu_matches,
            cpu_matches,
            "GPU prefilter must agree with CPU. Seeds disagreed: \
             GPU returned {} matches, CPU returned {}",
            gpu_matches.len(),
            cpu_matches.len()
        );
    }

    /// Triangular-spread structures exercise the longer (4-call) RNG path.
    #[test]
    fn gpu_matches_cpu_for_triangular_structure() {
        let Some(searcher) = GpuSearcher::try_new() else {
            eprintln!("skipping: no GPU adapter on this host");
            return;
        };
        let start: i64 = 1_000_000;
        let count: u64 = 1_000;
        let req = StructureRequirement {
            structure: StructureType::OceanMonument, // Triangular spread
            max_distance: 3000,
            centre_x: 0,
            centre_z: 0,
        };
        let gpu = searcher
            .find_nearby_structure_matches(
                start,
                count,
                req.structure,
                req.max_distance,
                req.centre_x,
                req.centre_z,
            )
            .expect("gpu dispatch");
        let cpu: Vec<i64> = (0..count)
            .filter_map(|i| {
                let seed = start.wrapping_add(i as i64);
                has_structure_in_radius(seed, &req).then_some(seed)
            })
            .collect();
        assert_eq!(gpu, cpu, "triangular GPU output drifted from CPU");
    }

    /// Off-origin centre + a coarser radius exercise the rx/rz scan bounds.
    #[test]
    fn gpu_matches_cpu_with_off_origin_centre() {
        let Some(searcher) = GpuSearcher::try_new() else {
            eprintln!("skipping: no GPU adapter on this host");
            return;
        };
        let req = StructureRequirement {
            structure: StructureType::Village,
            max_distance: 2500,
            centre_x: 1500,
            centre_z: -800,
        };
        let gpu = searcher
            .find_nearby_structure_matches(
                0,
                500,
                req.structure,
                req.max_distance,
                req.centre_x,
                req.centre_z,
            )
            .expect("gpu dispatch");
        let cpu: Vec<i64> = (0..500)
            .filter_map(|i| {
                let seed = i as i64;
                has_structure_in_radius(seed, &req).then_some(seed)
            })
            .collect();
        assert_eq!(gpu, cpu);
    }
}
