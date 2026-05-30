//! Phase D1: GPU depth-spline evaluator.
//!
//! cubiomes derives `np[NP_DEPTH]` from the climate vector via a recursive
//! offset spline (`getSpline`, biomenoise.c:1074): the depth value blends
//! continentalness / erosion / weirdness through a small seed-independent tree.
//!
//! For one-seed *tile rendering* (`gpu_biome.rs`) this is done on the CPU per
//! pixel. For per-seed *search* that would force a GPU→CPU→GPU round-trip every
//! dispatch, so the spline is moved onto the GPU: the seed-independent tree is
//! flattened once on the CPU (`mcsf_depth_spline_flatten`) such that every child
//! node precedes its parent in the flat order, then a WGSL walker
//! (`gpu_depth.wgsl`) evaluates it forward with no recursion.
//!
//! Accuracy: cubiomes' `getSpline` is f32 throughout, but the climate inputs
//! reach the GPU as f32 (vs f64 on cubiomes), so depth is f32-tolerant — the
//! same concession as the other Phase 6c noise primitives.

#![cfg(feature = "gpu")]

use bytemuck::{Pod, Zeroable};

const MAX_NODES: usize = 256;

extern "C" {
    /// Flatten MC version `mc`'s offset spline tree into parallel arrays
    /// (capacity `MAX_NODES` each). Children precede parents; the root is the
    /// last node. Returns the node count, or -1 on overflow. Defined in
    /// `csrc/shim.c`.
    fn mcsf_depth_spline_flatten(
        mc: i32,
        out_typ: *mut i32,
        out_len: *mut i32,
        out_fix: *mut f32,
        out_loc: *mut f32,
        out_der: *mut f32,
        out_child: *mut i32,
    ) -> i32;
    /// CPU reference depth (cubiomes `getSpline` + the depth formula at y=0).
    /// Defined in `csrc/shim.c`.
    fn mcsf_compute_depth(mc: i32, c: f64, e: f64, w: f64, y: i32) -> f64;
}

/// CPU-flattened depth spline for one MC version. Seed-independent; build once
/// via [`DepthSpline::flatten`] and reuse across all seeds.
#[derive(Clone, Debug)]
pub struct DepthSpline {
    pub n_nodes: u32,
    pub typ: Vec<i32>,
    pub len: Vec<u32>,
    pub fix: Vec<f32>,
    pub loc: Vec<f32>,   // 12 per node
    pub der: Vec<f32>,   // 12 per node
    pub child: Vec<i32>, // 12 per node
}

impl DepthSpline {
    /// Flatten cubiomes' offset spline for `mc`. Returns `None` if the tree
    /// overflows the flattener capacity (should not happen for supported
    /// Overworld versions).
    pub fn flatten(mc: i32) -> Option<Self> {
        let mut typ = vec![0i32; MAX_NODES];
        let mut len = vec![0i32; MAX_NODES];
        let mut fix = vec![0.0f32; MAX_NODES];
        let mut loc = vec![0.0f32; MAX_NODES * 12];
        let mut der = vec![0.0f32; MAX_NODES * 12];
        let mut child = vec![0i32; MAX_NODES * 12];
        let count = unsafe {
            mcsf_depth_spline_flatten(
                mc,
                typ.as_mut_ptr(),
                len.as_mut_ptr(),
                fix.as_mut_ptr(),
                loc.as_mut_ptr(),
                der.as_mut_ptr(),
                child.as_mut_ptr(),
            )
        };
        if count <= 0 {
            return None;
        }
        let n = count as usize;
        typ.truncate(n);
        let len_u: Vec<u32> = len[..n].iter().map(|&v| v as u32).collect();
        fix.truncate(n);
        loc.truncate(n * 12);
        der.truncate(n * 12);
        child.truncate(n * 12);
        Some(Self {
            n_nodes: n as u32,
            typ,
            len: len_u,
            fix,
            loc,
            der,
            child,
        })
    }

    /// CPU reference depth at y=0 for one climate vector, via cubiomes.
    pub fn cpu_depth(mc: i32, c: f64, e: f64, w: f64) -> f64 {
        unsafe { mcsf_compute_depth(mc, c, e, w, 0) }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DepthParams {
    n_inputs: u32,
    n_nodes: u32,
    _pad0: u32,
    _pad1: u32,
}

/// wgpu pipeline evaluating the depth spline for many climate vectors. Holds
/// the device/pipeline; spline-node buffers are uploaded per call.
pub struct GpuDepth {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl std::fmt::Debug for GpuDepth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDepth").finish_non_exhaustive()
    }
}

impl GpuDepth {
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
                    label: Some("mcsf-gpu-depth-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|_| "failed to acquire wgpu device")?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mcsf-gpu-depth-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_depth.wgsl").into()),
        });
        let uniform = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mcsf-gpu-depth-bgl"),
            entries: &[
                uniform(0),
                storage(1, true),  // typ
                storage(2, true),  // len
                storage(3, true),  // fix
                storage(4, true),  // loc
                storage(5, true),  // der
                storage(6, true),  // child
                storage(7, true),  // inputs
                storage(8, false), // outputs
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mcsf-gpu-depth-pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mcsf-gpu-depth-cp"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("cs_depth"),
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

    /// Evaluate depth for each `(c, e, w)` climate vector against `spline`.
    /// Returns one depth value per input.
    pub fn depth_batch(
        &self,
        spline: &DepthSpline,
        climate: &[(f32, f32, f32)],
    ) -> Result<Vec<f32>, &'static str> {
        if climate.is_empty() {
            return Ok(Vec::new());
        }
        let n = climate.len();

        let params = DepthParams {
            n_inputs: n as u32,
            n_nodes: spline.n_nodes,
            _pad0: 0,
            _pad1: 0,
        };
        let inputs: Vec<[f32; 4]> = climate.iter().map(|&(c, e, w)| [c, e, w, 0.0]).collect();

        let mk_uniform = |label: &str, bytes: &[u8]| {
            let b = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.len() as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&b, 0, bytes);
            b
        };
        let mk_storage = |label: &str, bytes: &[u8]| {
            let b = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.len() as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&b, 0, bytes);
            b
        };

        let params_buf = mk_uniform("mcsf-depth-params", bytemuck::bytes_of(&params));
        let typ_buf = mk_storage("mcsf-depth-typ", bytemuck::cast_slice(&spline.typ));
        let len_buf = mk_storage("mcsf-depth-len", bytemuck::cast_slice(&spline.len));
        let fix_buf = mk_storage("mcsf-depth-fix", bytemuck::cast_slice(&spline.fix));
        let loc_buf = mk_storage("mcsf-depth-loc", bytemuck::cast_slice(&spline.loc));
        let der_buf = mk_storage("mcsf-depth-der", bytemuck::cast_slice(&spline.der));
        let child_buf = mk_storage("mcsf-depth-child", bytemuck::cast_slice(&spline.child));
        let in_buf = mk_storage("mcsf-depth-in", bytemuck::cast_slice(&inputs));

        let out_bytes = (n * 4) as wgpu::BufferAddress;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcsf-depth-out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcsf-depth-staging"),
            size: out_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mcsf-depth-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: typ_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: len_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: fix_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: loc_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: der_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: child_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mcsf-depth-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mcsf-depth-pass"),
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
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
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

    fn mc_1_21() -> Option<i32> {
        extern "C" {
            fn mcsf_str2mc(s: *const std::os::raw::c_char) -> i32;
        }
        let c = std::ffi::CString::new("1.21").ok()?;
        let v = unsafe { mcsf_str2mc(c.as_ptr()) };
        if v > 0 {
            Some(v)
        } else {
            None
        }
    }

    /// Phase D1 cornerstone. The WGSL depth-spline walker must match cubiomes'
    /// `getSpline`-based depth within f32 tolerance for a spread of climate
    /// vectors. cubiomes' spline is f32, so the only drift source is the f32
    /// climate inputs — the tolerance covers that. A green test proves the
    /// flatten-and-walk port (and the children-before-parents ordering) is
    /// correct, which is the missing piece for the fused per-seed kernel.
    #[test]
    fn gpu_depth_matches_cubiomes_within_f32_tolerance() {
        let Some(gpu) = GpuDepth::try_new() else {
            eprintln!("skipping: no GPU adapter on this host");
            return;
        };
        let Some(mc) = mc_1_21() else {
            eprintln!("skipping: cubiomes does not recognise 1.21");
            return;
        };
        let Some(spline) = DepthSpline::flatten(mc) else {
            panic!("depth spline flatten failed / overflowed");
        };
        assert!(spline.n_nodes > 0);

        // Climate vectors spanning the [-1, 1]-ish range the splines branch on,
        // including the knot boundaries where getSpline switches segments.
        let climate: Vec<(f32, f32, f32)> = vec![
            (0.0, 0.0, 0.0),
            (-1.0, -1.0, -1.0),
            (1.0, 1.0, 1.0),
            (-0.5, 0.3, 0.7),
            (0.4, -0.2, -0.65),
            (0.2, 0.45, 0.05),
            (-0.9, 0.9, -0.05),
            (0.11, -0.78, 0.66),
            (0.05, 0.05, 0.6666667),
            (-0.3, -0.3, -0.6666667),
        ];

        let gpu_depths = gpu.depth_batch(&spline, &climate).expect("gpu dispatch");
        assert_eq!(gpu_depths.len(), climate.len());

        let mut max_err = 0.0f64;
        for (&(c, e, w), &gd) in climate.iter().zip(gpu_depths.iter()) {
            let cd = DepthSpline::cpu_depth(mc, c as f64, e as f64, w as f64);
            let diff = (cd - gd as f64).abs();
            if diff > max_err {
                max_err = diff;
            }
            assert!(
                diff < 5e-3,
                "depth GPU↔CPU drift too large at (c={c}, e={e}, w={w}): \
                 gpu={gd} cpu={cd} diff={diff:.3e}"
            );
        }
        eprintln!(
            "max abs(gpu-cpu) depth diff over {} vecs: {max_err:.3e}",
            climate.len()
        );
    }
}
