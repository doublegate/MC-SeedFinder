//! wgpu compute prefilter for the structure-only conditions tree.
//!
//! Phase 6b generalises the v1 kernel: instead of a single `NearbyStructure`,
//! it now takes up to 8 predicates plus a combinator (any_of / all_of /
//! cluster), so the GPU also handles:
//!   - Cluster searches (quad-hut style).
//!   - `all_of` / `any_of` groups of `NearbyStructure` leaves.
//! The WGSL kernel (`gpu.wgsl`) implements Java's 48-bit LCG and the
//! structure placement step bit-identically to the CPU code; the parity
//! tests below dispatch GPU + CPU side by side and assert identical match
//! lists for each combinator.
//!
//! Init is async (wgpu's `request_adapter` is async); we drive it with
//! `pollster::block_on` so the rest of the code stays synchronous. If no
//! adapter is available, `try_new` returns `None` and callers fall back to
//! the CPU evaluator.

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

/// Maximum predicates the kernel supports. Matches the array length in
/// `gpu.wgsl`. Increasing it grows the uniform buffer linearly.
pub const MAX_PREDICATES: usize = 8;

/// What kind of structure-only predicate the GPU is being asked to evaluate.
#[derive(Debug, Clone, Copy)]
pub enum Combinator {
    /// Match if ≥1 predicate has any hit. `min_count` ignored.
    AnyOf,
    /// Match iff EVERY predicate has at least one hit.
    AllOf,
    /// Match iff the total hit count across all predicates ≥ `min_count`.
    /// This is the quad-hut shape: same structures, common centre, threshold.
    Cluster { min_count: u32 },
}

/// One element of the predicate list. Independently scoped:
/// `(structure, centre, max_distance)` triple.
#[derive(Debug, Clone, Copy)]
pub struct GpuPredicate {
    pub structure: StructureType,
    pub max_distance: i32,
    pub centre_x: i32,
    pub centre_z: i32,
}

/// Full host-side specification. Translated into the uniform layout below.
#[derive(Debug, Clone)]
pub struct GpuSearchSpec<'a> {
    pub start_seed: i64,
    pub count: u64,
    pub combinator: Combinator,
    pub predicates: &'a [GpuPredicate],
}

// ---------------------------------------------------------------------------
// Uniform layout — must match `gpu.wgsl` exactly.
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct PredicateUniform {
    salt_lo: u32,
    salt_hi: u32,
    spacing: i32,
    chunk_range: u32,
    spread_type: u32,
    centre_x: i32,
    centre_z: i32,
    rx_min: i32,
    rx_max: i32,
    rz_min: i32,
    rz_max: i32,
    max_dist_sq_lo: u32,
    max_dist_sq_hi: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SearchParamsUniform {
    start_seed_lo: u32,
    start_seed_hi: u32,
    count: u32,
    combinator: u32,
    num_predicates: u32,
    min_count: u32,
    region_mul_x_lo: u32,
    region_mul_x_hi: u32,
    region_mul_z_lo: u32,
    region_mul_z_hi: u32,
    _pad0: u32,
    _pad1: u32,
    predicates: [PredicateUniform; MAX_PREDICATES],
}

impl PredicateUniform {
    /// Build the per-predicate uniform from a host-side `GpuPredicate`,
    /// computing the region scan bounds the same way the CPU helpers do.
    fn from_predicate(p: &GpuPredicate) -> Result<Self, &'static str> {
        if p.structure == StructureType::Stronghold {
            return Err(
                "strongholds use iter_strongholds; GPU path is for random-spread structures",
            );
        }
        if p.max_distance < 0 {
            return Err("max_distance must be non-negative");
        }
        let cfg = structure_config(p.structure);
        let chunk_radius = p.max_distance / 16 + 1;
        let cx_chunk = p.centre_x.div_euclid(16);
        let cz_chunk = p.centre_z.div_euclid(16);
        let cx_min = cx_chunk - chunk_radius;
        let cx_max = cx_chunk + chunk_radius;
        let cz_min = cz_chunk - chunk_radius;
        let cz_max = cz_chunk + chunk_radius;
        let rx_min = cx_min.div_euclid(cfg.spacing);
        let rx_max = cx_max.div_euclid(cfg.spacing);
        let rz_min = cz_min.div_euclid(cfg.spacing);
        let rz_max = cz_max.div_euclid(cfg.spacing);

        let (salt_lo, salt_hi) = i64_to_limbs(cfg.salt);
        let max_dist_sq: u64 = (p.max_distance as u64) * (p.max_distance as u64);

        Ok(Self {
            salt_lo,
            salt_hi,
            spacing: cfg.spacing,
            chunk_range: cfg.chunk_range() as u32,
            spread_type: match cfg.spread_type {
                SpreadType::Linear => 0,
                SpreadType::Triangular => 1,
            },
            centre_x: p.centre_x,
            centre_z: p.centre_z,
            rx_min,
            rx_max,
            rz_min,
            rz_max,
            max_dist_sq_lo: max_dist_sq as u32,
            max_dist_sq_hi: (max_dist_sq >> 32) as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        })
    }
}

/// One dispatch is bounded so we never blow past the typical 128 MiB storage
/// limit on modest adapters. Callers loop until `spec.count` is exhausted.
const MAX_SEEDS_PER_DISPATCH: u32 = 1_000_000;

// ---------------------------------------------------------------------------
// Searcher (persistent buffers + bind group across dispatches)
// ---------------------------------------------------------------------------

pub struct GpuSearcher {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
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
                None,
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

    /// Generic find: runs the multi-predicate kernel and returns matches in
    /// ascending seed order. Handles any combinator the kernel knows.
    pub fn find_matches(&self, spec: &GpuSearchSpec<'_>) -> Result<Vec<i64>, &'static str> {
        if spec.predicates.is_empty() {
            return Err("at least one predicate required");
        }
        if spec.predicates.len() > MAX_PREDICATES {
            return Err("too many predicates for the GPU kernel");
        }
        if spec.count == 0 {
            return Ok(Vec::new());
        }

        // Pre-build predicate uniforms once (shared across all dispatches).
        let mut predicates_arr = [PredicateUniform::default(); MAX_PREDICATES];
        for (i, p) in spec.predicates.iter().enumerate() {
            predicates_arr[i] = PredicateUniform::from_predicate(p)?;
        }
        let (combinator, min_count) = match spec.combinator {
            Combinator::AnyOf => (0, 1),
            Combinator::AllOf => (1, 1),
            Combinator::Cluster { min_count } => (2, min_count.max(1)),
        };
        let (rmx_lo, rmx_hi) = i64_to_limbs(REGION_MUL_X);
        let (rmz_lo, rmz_hi) = i64_to_limbs(REGION_MUL_Z);

        let mut matches: Vec<i64> = Vec::new();
        let mut offset: u64 = 0;
        while offset < spec.count {
            let this_count = ((spec.count - offset).min(MAX_SEEDS_PER_DISPATCH as u64)) as u32;
            let chunk_start = spec.start_seed.wrapping_add(offset as i64);
            let (ss_lo, ss_hi) = i64_to_limbs(chunk_start);

            let params = SearchParamsUniform {
                start_seed_lo: ss_lo,
                start_seed_hi: ss_hi,
                count: this_count,
                combinator,
                num_predicates: spec.predicates.len() as u32,
                min_count,
                region_mul_x_lo: rmx_lo,
                region_mul_x_hi: rmx_hi,
                region_mul_z_lo: rmz_lo,
                region_mul_z_hi: rmz_hi,
                _pad0: 0,
                _pad1: 0,
                predicates: predicates_arr,
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

    /// Convenience wrapper preserving the v1 API for callers that only need a
    /// single `NearbyStructure` predicate (any_of of size 1).
    pub fn find_nearby_structure_matches(
        &self,
        start_seed: i64,
        count: u64,
        structure: StructureType,
        max_distance: i32,
        centre_x: i32,
        centre_z: i32,
    ) -> Result<Vec<i64>, &'static str> {
        let preds = [GpuPredicate {
            structure,
            max_distance,
            centre_x,
            centre_z,
        }];
        self.find_matches(&GpuSearchSpec {
            start_seed,
            count,
            combinator: Combinator::AnyOf,
            predicates: &preds,
        })
    }

    fn dispatch_one(&self, params: &SearchParamsUniform) -> Result<Vec<u32>, &'static str> {
        let count = params.count as u64;
        let out_byte_size = count * 4;

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
            let groups = ((count as u32) + 63) / 64;
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.out_buffer, 0, &self.staging_buffer, 0, out_byte_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = self.staging_buffer.slice(..out_byte_size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
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
        self.staging_buffer.unmap();
        Ok(matches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conditions::{compile, CompiledNode, Node};
    use crate::structures::{has_structure_in_radius, StructureRequirement};

    fn make_searcher() -> Option<GpuSearcher> {
        let s = GpuSearcher::try_new();
        if s.is_none() {
            eprintln!("skipping: no GPU adapter on this host");
        }
        s
    }

    // ---------- v1 parity: single NearbyStructure ----------

    #[test]
    fn gpu_matches_cpu_for_village_within_1000() {
        let Some(searcher) = make_searcher() else {
            return;
        };
        let req = StructureRequirement {
            structure: StructureType::Village,
            max_distance: 1000,
            centre_x: 0,
            centre_z: 0,
        };
        let count: u64 = 2_000;
        let gpu = searcher
            .find_nearby_structure_matches(
                1,
                count,
                req.structure,
                req.max_distance,
                req.centre_x,
                req.centre_z,
            )
            .expect("gpu dispatch");
        let cpu: Vec<i64> = (0..count)
            .filter_map(|i| {
                let s = 1i64.wrapping_add(i as i64);
                has_structure_in_radius(s, &req).then_some(s)
            })
            .collect();
        assert_eq!(gpu, cpu);
    }

    #[test]
    fn gpu_matches_cpu_for_triangular_structure() {
        let Some(searcher) = make_searcher() else {
            return;
        };
        let req = StructureRequirement {
            structure: StructureType::OceanMonument,
            max_distance: 3000,
            centre_x: 0,
            centre_z: 0,
        };
        let count: u64 = 1_000;
        let gpu = searcher
            .find_nearby_structure_matches(
                1_000_000,
                count,
                req.structure,
                req.max_distance,
                req.centre_x,
                req.centre_z,
            )
            .expect("gpu dispatch");
        let cpu: Vec<i64> = (0..count)
            .filter_map(|i| {
                let s = 1_000_000i64.wrapping_add(i as i64);
                has_structure_in_radius(s, &req).then_some(s)
            })
            .collect();
        assert_eq!(gpu, cpu);
    }

    #[test]
    fn gpu_matches_cpu_with_off_origin_centre() {
        let Some(searcher) = make_searcher() else {
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
            .filter_map(|i| has_structure_in_radius(i as i64, &req).then_some(i as i64))
            .collect();
        assert_eq!(gpu, cpu);
    }

    // ---------- v2 parity: multi-predicate kernels ----------

    /// Helper: evaluate a CompiledNode against a contiguous seed range on the
    /// CPU and return the matching seeds. Strict structure-only.
    fn cpu_matches(node: &CompiledNode, start: i64, count: u64) -> Vec<i64> {
        (0..count)
            .filter_map(|i| {
                let seed = start.wrapping_add(i as i64);
                crate::conditions::evaluate(node, seed).then_some(seed)
            })
            .collect()
    }

    #[test]
    fn gpu_cluster_matches_cpu() {
        let Some(searcher) = make_searcher() else {
            return;
        };
        // Cluster: ≥3 hits across {village, pillager_outpost} within 2000 of origin.
        let preds = [
            GpuPredicate {
                structure: StructureType::Village,
                max_distance: 2000,
                centre_x: 0,
                centre_z: 0,
            },
            GpuPredicate {
                structure: StructureType::PillagerOutpost,
                max_distance: 2000,
                centre_x: 0,
                centre_z: 0,
            },
        ];
        let gpu = searcher
            .find_matches(&GpuSearchSpec {
                start_seed: 1,
                count: 2_000,
                combinator: Combinator::Cluster { min_count: 3 },
                predicates: &preds,
            })
            .expect("gpu dispatch");
        let node = compile(&Node::Cluster {
            structures: vec!["village".into(), "pillager_outpost".into()],
            max_distance: 2000,
            min_count: 3,
            centre_x: 0,
            centre_z: 0,
        })
        .unwrap();
        let cpu = cpu_matches(&node, 1, 2_000);
        assert_eq!(gpu, cpu, "cluster GPU drifted from CPU");
    }

    #[test]
    fn gpu_any_of_matches_cpu() {
        let Some(searcher) = make_searcher() else {
            return;
        };
        // any_of: village within 500 OR pillager_outpost within 1500.
        let preds = [
            GpuPredicate {
                structure: StructureType::Village,
                max_distance: 500,
                centre_x: 0,
                centre_z: 0,
            },
            GpuPredicate {
                structure: StructureType::PillagerOutpost,
                max_distance: 1500,
                centre_x: 0,
                centre_z: 0,
            },
        ];
        let gpu = searcher
            .find_matches(&GpuSearchSpec {
                start_seed: 1,
                count: 2_000,
                combinator: Combinator::AnyOf,
                predicates: &preds,
            })
            .expect("gpu dispatch");
        let node = compile(&Node::AnyOf {
            of: vec![
                Node::NearbyStructure {
                    structure: "village".into(),
                    max_distance: 500,
                    centre_x: 0,
                    centre_z: 0,
                },
                Node::NearbyStructure {
                    structure: "pillager_outpost".into(),
                    max_distance: 1500,
                    centre_x: 0,
                    centre_z: 0,
                },
            ],
        })
        .unwrap();
        let cpu = cpu_matches(&node, 1, 2_000);
        assert_eq!(gpu, cpu, "any_of GPU drifted from CPU");
    }

    #[test]
    fn gpu_all_of_matches_cpu() {
        let Some(searcher) = make_searcher() else {
            return;
        };
        // all_of: village within 2000 AND ocean_monument within 3000 (loose
        // enough that some seeds will satisfy both).
        let preds = [
            GpuPredicate {
                structure: StructureType::Village,
                max_distance: 2000,
                centre_x: 0,
                centre_z: 0,
            },
            GpuPredicate {
                structure: StructureType::OceanMonument,
                max_distance: 3000,
                centre_x: 0,
                centre_z: 0,
            },
        ];
        let gpu = searcher
            .find_matches(&GpuSearchSpec {
                start_seed: 1,
                count: 2_000,
                combinator: Combinator::AllOf,
                predicates: &preds,
            })
            .expect("gpu dispatch");
        let node = compile(&Node::AllOf {
            of: vec![
                Node::NearbyStructure {
                    structure: "village".into(),
                    max_distance: 2000,
                    centre_x: 0,
                    centre_z: 0,
                },
                Node::NearbyStructure {
                    structure: "ocean_monument".into(),
                    max_distance: 3000,
                    centre_x: 0,
                    centre_z: 0,
                },
            ],
        })
        .unwrap();
        let cpu = cpu_matches(&node, 1, 2_000);
        assert_eq!(gpu, cpu, "all_of GPU drifted from CPU");
    }
}
