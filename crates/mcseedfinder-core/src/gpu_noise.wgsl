// Phase 6c-1: WGSL port of cubiomes' `samplePerlin`.
//
// Mirrors `noise.c`/`samplePerlin` step for step, with one fundamental
// concession: WebGPU compute is f32-only, cubiomes is f64. That means our
// sampled values differ from cubiomes' by f32 round-off (~1e-7 relative
// error) — bit-exact equality is mathematically impossible without f64.
// The parity test (in `gpu_noise.rs`) asserts an absolute-difference
// tolerance that covers the worst case.
//
// Permutation table layout: 256 bytes packed little-endian into 64 u32s.
// cubiomes' implementation keeps a 257th duplicate (idx[256] = idx[0]); we
// emulate that by masking every "h+1" index with `& 0xFFu`, which gives
// identical lookups without the extra entry.
//
// `yamp != 0` is the 3D-terrain branch; climate noise sampling always
// passes yamp=0, so we keep the simpler path here. 3D yamp support lands
// in Phase 6c-2 alongside octave-summation.

struct PerlinUniform {
    a: f32,
    b: f32,
    c: f32,
    _pad0: u32,
    d2: f32,
    t2: f32,
    h2: u32,
    _pad1: u32,
    // 256 bytes packed little-endian.
    perm: array<vec4<u32>, 16>,  // 16 × 4 × 4 bytes = 256 bytes
};

struct SampleParams {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> perlin: PerlinUniform;
@group(0) @binding(1) var<uniform> sp: SampleParams;
/// Inputs: one vec4 per sample — (x, y, z, _unused). Phase 6c-1 ignores
/// the 4th component; Phase 6c-2+ will repurpose it for yamp.
@group(0) @binding(2) var<storage, read> sample_inputs: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> sample_outputs: array<f32>;

/// Extract byte `idx` (0..256) from the packed perm table; indices wrap
/// at 256 to emulate cubiomes' duplicated 257th entry.
fn perm_byte(idx: u32) -> u32 {
    let i = idx & 0xFFu;
    let word_idx = i >> 2u;           // which u32 (0..63)
    let byte_off = (i & 3u) * 8u;      // bit offset within u32
    let vec_idx = word_idx >> 2u;      // which vec4<u32> (0..15)
    let sub_idx = word_idx & 3u;       // component within vec4
    let v = perlin.perm[vec_idx];
    var word: u32;
    if (sub_idx == 0u) { word = v.x; }
    else if (sub_idx == 1u) { word = v.y; }
    else if (sub_idx == 2u) { word = v.z; }
    else { word = v.w; }
    return (word >> byte_off) & 0xFFu;
}

/// cubiomes' fade(): t^3 (t (6t - 15) + 10)
fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn lerp_f(t: f32, a: f32, b: f32) -> f32 {
    return a + t * (b - a);
}

/// cubiomes' indexedLerp: switch-on-low-4-bits picks one of 16 gradient
/// directions and dots it with (a, b, c). The switch table maps directly to
/// pairwise sums/differences of (a, b, c).
fn indexed_lerp(idx: u32, a: f32, b: f32, c: f32) -> f32 {
    switch (idx & 0xFu) {
        case 0u:  { return  a + b; }
        case 1u:  { return -a + b; }
        case 2u:  { return  a - b; }
        case 3u:  { return -a - b; }
        case 4u:  { return  a + c; }
        case 5u:  { return -a + c; }
        case 6u:  { return  a - c; }
        case 7u:  { return -a - c; }
        case 8u:  { return  b + c; }
        case 9u:  { return -b + c; }
        case 10u: { return  b - c; }
        case 11u: { return -b - c; }
        case 12u: { return  a + b; }
        case 13u: { return -b + c; }
        case 14u: { return -a + b; }
        default:  { return -b - c; }
    }
}

fn sample_perlin(d1_in: f32, d2_in: f32, d3_in: f32) -> f32 {
    var d2: f32;
    var t2: f32;
    var h2: u32;
    if (d2_in == 0.0) {
        // Fast path: cubiomes pre-computes h2/d2/t2 from p.b at init.
        d2 = perlin.d2;
        t2 = perlin.t2;
        h2 = perlin.h2;
    } else {
        let v = d2_in + perlin.b;
        let i2 = floor(v);
        d2 = v - i2;
        h2 = u32(i32(i2)) & 0xFFu;
        t2 = fade(d2);
    }

    let v1 = d1_in + perlin.a;
    let v3 = d3_in + perlin.c;
    let i1 = floor(v1);
    let i3 = floor(v3);
    let d1 = v1 - i1;
    let d3 = v3 - i3;
    let h1 = u32(i32(i1)) & 0xFFu;
    let h3 = u32(i32(i3)) & 0xFFu;
    let t1 = fade(d1);
    let t3 = fade(d3);

    // Gradient lookups mirror cubiomes' two-step "vec2" path:
    //   p_h1   = idx[h1]
    //   p_h1_1 = idx[h1+1]
    let p_h1   = perm_byte(h1);
    let p_h1_1 = perm_byte(h1 + 1u);

    //   v1a = p_h1 + h2  (mod 256 by uint8_t wrap)
    let v1a = (p_h1   + h2) & 0xFFu;
    let v1b = (p_h1_1 + h2) & 0xFFu;

    let p_v1a   = perm_byte(v1a);
    let p_v1a_1 = perm_byte(v1a + 1u);
    let p_v1b   = perm_byte(v1b);
    let p_v1b_1 = perm_byte(v1b + 1u);

    //   v2.a += h3 etc.
    let v2a = (p_v1a   + h3) & 0xFFu;
    let v2b = (p_v1a_1 + h3) & 0xFFu;
    let v3a = (p_v1b   + h3) & 0xFFu;
    let v3b = (p_v1b_1 + h3) & 0xFFu;

    let v4a = perm_byte(v2a);
    let v4b = perm_byte(v2a + 1u);
    let v5a = perm_byte(v2b);
    let v5b = perm_byte(v2b + 1u);
    let v6a = perm_byte(v3a);
    let v6b = perm_byte(v3a + 1u);
    let v7a = perm_byte(v3b);
    let v7b = perm_byte(v3b + 1u);

    // 8 gradient samples at the unit-cube corners.
    let l1 = indexed_lerp(v4a, d1,       d2,       d3      );
    let l5 = indexed_lerp(v4b, d1,       d2,       d3 - 1.0);
    let l2 = indexed_lerp(v6a, d1 - 1.0, d2,       d3      );
    let l6 = indexed_lerp(v6b, d1 - 1.0, d2,       d3 - 1.0);
    let l3 = indexed_lerp(v5a, d1,       d2 - 1.0, d3      );
    let l7 = indexed_lerp(v5b, d1,       d2 - 1.0, d3 - 1.0);
    let l4 = indexed_lerp(v7a, d1 - 1.0, d2 - 1.0, d3      );
    let l8 = indexed_lerp(v7b, d1 - 1.0, d2 - 1.0, d3 - 1.0);

    // Trilinear interpolation along x, then z (paired), then y.
    let m1 = lerp_f(t1, l1, l2);
    let m3 = lerp_f(t1, l3, l4);
    let m5 = lerp_f(t1, l5, l6);
    let m7 = lerp_f(t1, l7, l8);
    let n1 = lerp_f(t2, m1, m3);
    let n5 = lerp_f(t2, m5, m7);
    return lerp_f(t3, n1, n5);
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sp.count) { return; }
    let inp = sample_inputs[i];
    sample_outputs[i] = sample_perlin(inp.x, inp.y, inp.z);
}

// -----------------------------------------------------------------------------
// Phase 6c-2: sampleOctave + sampleDoublePerlin.
//
// Mirrors cubiomes' `sampleOctave` and `sampleDoublePerlin` exactly. Note
// that cubiomes' `maintainPrecision` is a no-op (the floor/wrap is commented
// out), so we just multiply by lacunarity directly.
//
// Bind group layout (separate pipeline from cs_main above):
//   binding 0: DPParams uniform (octA_count, octB_count, sample_count,
//              dp_amplitude, freq_shift = 337/331)
//   binding 1: storage<read> octaves[]   (one OctaveEntry per PerlinNoise,
//              concatenated as [octA..., octB...])
//   binding 2: storage<read> sample_inputs[]    (vec4<f32>, .xyz used)
//   binding 3: storage<read_write> sample_outputs[]
// -----------------------------------------------------------------------------

struct OctaveEntry {
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
    perm: array<vec4<u32>, 16>,  // 256 bytes packed little-endian.
};

struct DPParams {
    oct_a_count: u32,
    oct_b_count: u32,
    sample_count: u32,
    _pad0: u32,
    dp_amplitude: f32,
    freq_shift: f32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> dpp: DPParams;
@group(0) @binding(1) var<storage, read> octaves: array<OctaveEntry>;
@group(0) @binding(2) var<storage, read> dp_inputs: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> dp_outputs: array<f32>;

fn perm_byte_oct(oct_idx: u32, idx: u32) -> u32 {
    let i = idx & 0xFFu;
    let word_idx = i >> 2u;
    let byte_off = (i & 3u) * 8u;
    let vec_idx = word_idx >> 2u;
    let sub_idx = word_idx & 3u;
    let v = octaves[oct_idx].perm[vec_idx];
    var word: u32;
    if (sub_idx == 0u) { word = v.x; }
    else if (sub_idx == 1u) { word = v.y; }
    else if (sub_idx == 2u) { word = v.z; }
    else { word = v.w; }
    return (word >> byte_off) & 0xFFu;
}

fn sample_perlin_oct(oct_idx: u32, d1_in: f32, d2_in: f32, d3_in: f32) -> f32 {
    let oa = octaves[oct_idx].a;
    let ob = octaves[oct_idx].b;
    let oc = octaves[oct_idx].c;

    var d2: f32;
    var t2: f32;
    var h2: u32;
    if (d2_in == 0.0) {
        d2 = octaves[oct_idx].d2;
        t2 = octaves[oct_idx].t2;
        h2 = octaves[oct_idx].h2;
    } else {
        let v = d2_in + ob;
        let i2 = floor(v);
        d2 = v - i2;
        h2 = u32(i32(i2)) & 0xFFu;
        t2 = fade(d2);
    }

    let v1 = d1_in + oa;
    let v3 = d3_in + oc;
    let i1 = floor(v1);
    let i3 = floor(v3);
    let d1 = v1 - i1;
    let d3 = v3 - i3;
    let h1 = u32(i32(i1)) & 0xFFu;
    let h3 = u32(i32(i3)) & 0xFFu;
    let t1 = fade(d1);
    let t3 = fade(d3);

    let p_h1   = perm_byte_oct(oct_idx, h1);
    let p_h1_1 = perm_byte_oct(oct_idx, h1 + 1u);
    let v1a = (p_h1   + h2) & 0xFFu;
    let v1b = (p_h1_1 + h2) & 0xFFu;
    let p_v1a   = perm_byte_oct(oct_idx, v1a);
    let p_v1a_1 = perm_byte_oct(oct_idx, v1a + 1u);
    let p_v1b   = perm_byte_oct(oct_idx, v1b);
    let p_v1b_1 = perm_byte_oct(oct_idx, v1b + 1u);
    let v2a = (p_v1a   + h3) & 0xFFu;
    let v2b = (p_v1a_1 + h3) & 0xFFu;
    let v3a = (p_v1b   + h3) & 0xFFu;
    let v3b = (p_v1b_1 + h3) & 0xFFu;
    let v4a = perm_byte_oct(oct_idx, v2a);
    let v4b = perm_byte_oct(oct_idx, v2a + 1u);
    let v5a = perm_byte_oct(oct_idx, v2b);
    let v5b = perm_byte_oct(oct_idx, v2b + 1u);
    let v6a = perm_byte_oct(oct_idx, v3a);
    let v6b = perm_byte_oct(oct_idx, v3a + 1u);
    let v7a = perm_byte_oct(oct_idx, v3b);
    let v7b = perm_byte_oct(oct_idx, v3b + 1u);

    let l1 = indexed_lerp(v4a, d1,       d2,       d3      );
    let l5 = indexed_lerp(v4b, d1,       d2,       d3 - 1.0);
    let l2 = indexed_lerp(v6a, d1 - 1.0, d2,       d3      );
    let l6 = indexed_lerp(v6b, d1 - 1.0, d2,       d3 - 1.0);
    let l3 = indexed_lerp(v5a, d1,       d2 - 1.0, d3      );
    let l7 = indexed_lerp(v5b, d1,       d2 - 1.0, d3 - 1.0);
    let l4 = indexed_lerp(v7a, d1 - 1.0, d2 - 1.0, d3      );
    let l8 = indexed_lerp(v7b, d1 - 1.0, d2 - 1.0, d3 - 1.0);

    let m1 = lerp_f(t1, l1, l2);
    let m3 = lerp_f(t1, l3, l4);
    let m5 = lerp_f(t1, l5, l6);
    let m7 = lerp_f(t1, l7, l8);
    let n1 = lerp_f(t2, m1, m3);
    let n5 = lerp_f(t2, m5, m7);
    return lerp_f(t3, n1, n5);
}

fn sample_octave_range(start: u32, count: u32, x: f32, y: f32, z: f32) -> f32 {
    var v = 0.0;
    for (var i = 0u; i < count; i = i + 1u) {
        let oct_idx = start + i;
        let lf = octaves[oct_idx].lacunarity;
        // cubiomes' maintainPrecision is a no-op — plain multiplication.
        let ax = x * lf;
        let ay = y * lf;
        let az = z * lf;
        let pv = sample_perlin_oct(oct_idx, ax, ay, az);
        v = v + octaves[oct_idx].amplitude * pv;
    }
    return v;
}

@compute @workgroup_size(64)
fn cs_double_perlin(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dpp.sample_count) { return; }
    let inp = dp_inputs[i];
    let x = inp.x;
    let y = inp.y;
    let z = inp.z;
    let f = dpp.freq_shift;
    let va = sample_octave_range(0u, dpp.oct_a_count, x, y, z);
    let vb = sample_octave_range(dpp.oct_a_count, dpp.oct_b_count, x * f, y * f, z * f);
    dp_outputs[i] = (va + vb) * dpp.dp_amplitude;
}
