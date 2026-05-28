//! Phase 6c-4: Biome b-tree walker for MC 1.21 (btree21wd).
//!
//! cubiomes encodes its 6-D biome decision tree as three flat tables:
//!   - `steps[depth]` — fanout step at each tree depth (5 levels + 0 terminator)
//!   - `param[i][2]`  — `(min, max)` int32 pair per parameter range, indexed
//!                       by bytes packed into a node's low 48 bits
//!   - `nodes[i]`     — u64 packing: low 48 bits = 6 byte indices into
//!                       `param` (one per climate), top 16 bits = either the
//!                       biome ID (leaf) or the first child index (internal)
//!
//! The walker is `get_resulting_node` (cubiomes/biomenoise.c:1388). It's
//! naturally recursive but bounded by `len(steps)-1 = 5` levels. WGSL has
//! no recursion, so the GPU port turns it into an iterative state machine
//! with an explicit ≤5-frame stack.
//!
//! This module is laid out in two halves:
//!   1. **CPU walker** (`BiomeTree::resulting_node`) — pure Rust port of
//!      cubiomes' algorithm, runs against the f64 cubiomes np[6] inputs.
//!      Used as a self-test that the data extraction + algorithm port are
//!      correct independent of WGSL.
//!   2. **GPU walker** (`GpuBiomeTree`) — WGSL `cs_climate_to_biome` entry
//!      point. Reads the same `BiomeTree` data uploaded as storage buffers,
//!      runs the same iterative algorithm.
//!
//! Both walkers are tested against cubiomes' real `climateToBiome` so a
//! green test means GPU output is bit-identical to cubiomes (this stage is
//! pure integer arithmetic — no float precision tolerance involved).

#![cfg(feature = "gpu")]

use bytemuck::{Pod, Zeroable};

extern "C" {
    fn mcsf_btree21wd_order() -> std::os::raw::c_int;
    fn mcsf_btree21wd_steps_len() -> usize;
    fn mcsf_btree21wd_steps_copy(out: *mut u32);
    fn mcsf_btree21wd_param_len_i32() -> usize;
    fn mcsf_btree21wd_param_copy(out: *mut i32);
    fn mcsf_btree21wd_nodes_len() -> usize;
    fn mcsf_btree21wd_nodes_copy(out: *mut u64);
    fn mcsf_climate_to_biome(mc: std::os::raw::c_int, np6: *const u64) -> std::os::raw::c_int;
}

/// In-memory copy of cubiomes' btree21wd tables. Built once via
/// [`BiomeTree::btree21wd`]; used by both the CPU walker and the GPU
/// dispatcher.
#[derive(Clone, Debug)]
pub struct BiomeTree {
    pub steps: Vec<u32>,
    pub param: Vec<i32>, // 2 i32 per parameter range (flat)
    pub nodes: Vec<u64>,
    pub order: u32,
}

impl BiomeTree {
    /// Extract cubiomes' btree21wd into Rust-owned buffers.
    pub fn btree21wd() -> Self {
        let order = unsafe { mcsf_btree21wd_order() } as u32;
        let n_steps = unsafe { mcsf_btree21wd_steps_len() };
        let n_param = unsafe { mcsf_btree21wd_param_len_i32() };
        let n_nodes = unsafe { mcsf_btree21wd_nodes_len() };
        let mut steps = vec![0u32; n_steps];
        let mut param = vec![0i32; n_param];
        let mut nodes = vec![0u64; n_nodes];
        unsafe {
            mcsf_btree21wd_steps_copy(steps.as_mut_ptr());
            mcsf_btree21wd_param_copy(param.as_mut_ptr());
            mcsf_btree21wd_nodes_copy(nodes.as_mut_ptr());
        }
        Self {
            steps,
            param,
            nodes,
            order,
        }
    }

    pub fn len(&self) -> u32 {
        self.nodes.len() as u32
    }

    /// cubiomes' `get_np_dist` — squared distance from `np` to the
    /// per-parameter `[min, max]` ranges referenced by node `idx`.
    ///
    /// Note: `np` is u64 storing int64 bit patterns. Subtraction is
    /// modular; the signed-positive test is "is the underflow not
    /// happening". Sum of 6 squared diffs needs u64 headroom (max
    /// ~9.6×10⁹ > 2³²).
    fn get_np_dist(&self, np: &[u64; 6], idx: u32) -> u64 {
        let node = self.nodes[idx as usize];
        let mut ds: u64 = 0;
        for i in 0..6u32 {
            let pidx = ((node >> (8 * i)) & 0xFF) as usize;
            let pmin = self.param[2 * pidx] as i64 as u64;
            let pmax = self.param[2 * pidx + 1] as i64 as u64;
            let a = np[i as usize].wrapping_sub(pmax);
            let b = pmin.wrapping_sub(np[i as usize]);
            let d = if (a as i64) > 0 {
                a
            } else if (b as i64) > 0 {
                b
            } else {
                0
            };
            // d ≤ ~40000, d² ≤ ~1.6×10⁹ — fits in u64 trivially.
            ds = ds.wrapping_add(d.wrapping_mul(d));
        }
        ds
    }

    /// cubiomes' `get_resulting_node`. Returns the index of the leaf node
    /// whose [min,max] hyperrectangle is closest (in 6-D squared distance)
    /// to `np`. Recursive in cubiomes; ported as recursive Rust here for
    /// the CPU oracle. The WGSL port (Phase 6c-4 GPU half) flattens this.
    fn resulting_node(&self, np: &[u64; 6], idx: u32, alt: u32, ds: u64, depth: usize) -> u32 {
        let len = self.len();
        if depth >= self.steps.len() || self.steps[depth] == 0 {
            return idx;
        }
        let mut step;
        let mut d = depth;
        loop {
            step = self.steps[d];
            d += 1;
            if idx + step < len || step == 0 {
                break;
            }
        }
        let node = self.nodes[idx as usize];
        let mut inner = (node >> 48) as u32;
        let mut leaf = alt;
        let mut ds = ds;
        for _ in 0..self.order {
            let ds_inner = self.get_np_dist(np, inner);
            if ds_inner < ds {
                let leaf2 = self.resulting_node(np, inner, leaf, ds, d);
                let ds_leaf2 = if inner == leaf2 {
                    ds_inner
                } else {
                    self.get_np_dist(np, leaf2)
                };
                if ds_leaf2 < ds {
                    ds = ds_leaf2;
                    leaf = leaf2;
                }
            }
            inner += step;
            if inner >= len {
                break;
            }
        }
        leaf
    }

    /// Top-level entry: `climateToBiome(mc, np)` equivalent. Returns the
    /// cubiomes biome ID (low 8 bits of the leaf node's high 16 bits).
    pub fn climate_to_biome(&self, np: &[u64; 6]) -> u32 {
        let leaf = self.resulting_node(np, 0, 0, u64::MAX, 0);
        ((self.nodes[leaf as usize] >> 48) & 0xFF) as u32
    }
}

/// Reference: cubiomes' real `climateToBiome` via the shim. Used by tests
/// as the absolute ground truth.
pub fn cubiomes_climate_to_biome(mc: i32, np: &[i64; 6]) -> i32 {
    let np_u: [u64; 6] = std::array::from_fn(|i| np[i] as u64);
    unsafe { mcsf_climate_to_biome(mc, np_u.as_ptr()) }
}

// =============================================================================
// GPU walker
// =============================================================================

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BTreeParams {
    order: u32,
    nodes_len: u32,
    steps_len: u32,
    sample_count: u32,
}

/// wgpu pipeline for `climateToBiome`. One workgroup-thread per np[6]
/// input; output is one u32 biome ID per input.
pub struct GpuBiomeTree {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    tree: BiomeTree,
}

impl std::fmt::Debug for GpuBiomeTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBiomeTree").finish_non_exhaustive()
    }
}

impl GpuBiomeTree {
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
        // We need 5 storage buffers per stage (steps, param, nodes,
        // np_in, biome_out); the downlevel default is 4.
        let limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 6,
            ..wgpu::Limits::downlevel_defaults()
        };
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("mcsf-gpu-btree-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|_| "failed to acquire wgpu device")?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mcsf-gpu-btree-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_btree.wgsl").into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mcsf-gpu-btree-bgl"),
            entries: &[
                // params
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
                // steps storage
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
                // param storage
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
                // nodes storage (u64 packed as vec2<u32>: x=lo, y=hi)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // inputs (np[6] per sample, packed as 6 × vec2<u32>: x=lo, y=hi)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // outputs (u32 biome IDs)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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
            label: Some("mcsf-gpu-btree-pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mcsf-gpu-btree-cp"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("cs_climate_to_biome"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        let tree = BiomeTree::btree21wd();
        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout: bgl,
            tree,
        })
    }

    pub fn tree(&self) -> &BiomeTree {
        &self.tree
    }

    /// Look up biome IDs for a batch of np[6] climate-value sets.
    /// Inputs are i64 (cubiomes' canonical format); they're transmitted as
    /// vec2<u32> pairs.
    pub fn lookup_batch(&self, np_inputs: &[[i64; 6]]) -> Result<Vec<u32>, &'static str> {
        if np_inputs.is_empty() {
            return Ok(Vec::new());
        }

        let params = BTreeParams {
            order: self.tree.order,
            nodes_len: self.tree.nodes.len() as u32,
            steps_len: self.tree.steps.len() as u32,
            sample_count: np_inputs.len() as u32,
        };

        // Pack u64s as [lo: u32, hi: u32].
        let nodes_packed: Vec<[u32; 2]> = self
            .tree
            .nodes
            .iter()
            .map(|&n| [(n & 0xFFFF_FFFF) as u32, (n >> 32) as u32])
            .collect();
        // np[6] per input; pack each i64 as [lo, hi] u32 pair → 6 vec2 per input
        // = 12 u32 per input; align nicely as 3×vec4 doesn't apply, so just
        // contiguous u32 pairs.
        let mut np_packed: Vec<[u32; 2]> = Vec::with_capacity(np_inputs.len() * 6);
        for inp in np_inputs {
            for &v in inp {
                let u = v as u64;
                np_packed.push([(u & 0xFFFF_FFFF) as u32, (u >> 32) as u32]);
            }
        }

        let nodes_bytes: &[u8] = bytemuck::cast_slice(&nodes_packed);
        let np_bytes: &[u8] = bytemuck::cast_slice(&np_packed);
        let steps_bytes: &[u8] = bytemuck::cast_slice(&self.tree.steps);
        let param_bytes: &[u8] = bytemuck::cast_slice(&self.tree.param);
        let out_bytes = (np_inputs.len() * 4) as wgpu::BufferAddress;

        let params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcsf-btree-params"),
            size: std::mem::size_of::<BTreeParams>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        let steps_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcsf-btree-steps"),
            size: steps_bytes.len() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&steps_buf, 0, steps_bytes);

        let param_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcsf-btree-param"),
            size: param_bytes.len() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&param_buf, 0, param_bytes);

        let nodes_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcsf-btree-nodes"),
            size: nodes_bytes.len() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&nodes_buf, 0, nodes_bytes);

        let np_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcsf-btree-np"),
            size: np_bytes.len() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&np_buf, 0, np_bytes);

        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcsf-btree-out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcsf-btree-staging"),
            size: out_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mcsf-btree-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: steps_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: param_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: nodes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: np_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mcsf-btree-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mcsf-btree-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let groups = ((np_inputs.len() as u32) + 63) / 64;
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
        let results: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(results)
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

    fn sample_climate(mc: i32, seed: u64, large: bool, x: i32, z: i32) -> ([i64; 6], i32) {
        let mut np = [0i64; 6];
        let biome = unsafe {
            mcsf_sample_biome_at(
                mc,
                seed,
                if large { 1 } else { 0 },
                x,
                0,
                z,
                np.as_mut_ptr(),
            )
        };
        (np, biome)
    }

    /// CPU walker port — must bit-exactly match cubiomes' `climateToBiome`
    /// across a grid of seed 12345 1.21 samples. Catches any data-extraction
    /// or algorithm-port bug before the GPU dispatcher gets involved.
    #[test]
    fn cpu_btree_matches_cubiomes_on_grid_seed_12345_mc_1_21() {
        let mc = match parse_mc_version("1.21") {
            Some(v) => v,
            None => return,
        };
        let tree = BiomeTree::btree21wd();

        let mut mismatches = Vec::new();
        // 8×8 grid at 64-block spacing covers a meaningful spread of
        // biomes (~500m × 500m around origin).
        for ix in -4..4 {
            for iz in -4..4 {
                let x = ix * 64;
                let z = iz * 64;
                let (np, cubi_biome) = sample_climate(mc, 12345, false, x, z);
                let np_u: [u64; 6] = std::array::from_fn(|i| np[i] as u64);
                let cpu_biome = tree.climate_to_biome(&np_u) as i32;
                if cpu_biome != cubi_biome {
                    mismatches.push((x, z, cubi_biome, cpu_biome, np));
                }
            }
        }
        if !mismatches.is_empty() {
            for (x, z, c, p, np) in &mismatches[..mismatches.len().min(5)] {
                eprintln!("mismatch at ({x},{z}): cubiomes={c} cpu={p} np={np:?}");
            }
            panic!(
                "{} of 64 CPU walker results disagreed with cubiomes",
                mismatches.len()
            );
        }
    }

    /// GPU walker — must produce identical biome IDs to the CPU walker
    /// (and thus to cubiomes) for the same grid. This stage is pure
    /// integer arithmetic on i64 (emulated as 2×u32 in WGSL) so equality
    /// is exact, not tolerance-based.
    #[test]
    fn gpu_btree_matches_cubiomes_on_grid_seed_12345_mc_1_21() {
        let Some(gpu) = GpuBiomeTree::try_new() else {
            eprintln!("skipping: no GPU adapter on this host");
            return;
        };
        let mc = match parse_mc_version("1.21") {
            Some(v) => v,
            None => return,
        };

        let mut inputs: Vec<[i64; 6]> = Vec::new();
        let mut expected: Vec<i32> = Vec::new();
        for ix in -4..4 {
            for iz in -4..4 {
                let x = ix * 64;
                let z = iz * 64;
                let (np, biome) = sample_climate(mc, 12345, false, x, z);
                inputs.push(np);
                expected.push(biome);
            }
        }

        let gpu_results = gpu.lookup_batch(&inputs).expect("gpu dispatch");
        assert_eq!(gpu_results.len(), expected.len());

        let mut mismatches = 0;
        for (i, (got, want)) in gpu_results.iter().zip(expected.iter()).enumerate() {
            if *got as i32 != *want {
                if mismatches < 5 {
                    eprintln!("idx {i}: gpu={got} cubiomes={want} np={:?}", inputs[i]);
                }
                mismatches += 1;
            }
        }
        assert_eq!(
            mismatches,
            0,
            "{mismatches} of {} GPU walker results disagreed",
            expected.len()
        );
    }
}
