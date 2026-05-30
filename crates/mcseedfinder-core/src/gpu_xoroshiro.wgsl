// Phase A: bit-exact Xoroshiro128++ port of cubiomes' `rng.h` for per-seed
// climate-noise initialisation on the GPU.
//
// WGSL has no u64, so every 64-bit value is emulated as vec2<u32> = (lo, hi)
// with lo = bits[0..31], hi = bits[32..63]. Every operation here is BIT-EXACT
// against cubiomes (this is integer RNG, NOT float noise — no tolerance).
//
// Reference (verified): cubiomes/rng.h
//   xSetSeed (line 185), xNextLong (line 203).
//
// This kernel only produces a raw xNextLong stream for the parity test; it is
// not yet wired into any search. Subsequent phases build octave/permutation
// init on top of these primitives.

struct Params {
    n_seeds: u32,
    k: u32,        // xNextLong draws emitted per seed
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// World seeds, one per thread, as (lo32, hi32).
@group(0) @binding(1) var<storage, read> seeds: array<vec2<u32>>;
// Output stream: n_seeds * k outputs, each a u64 as (lo32, hi32).
@group(0) @binding(2) var<storage, read_write> outp: array<vec2<u32>>;

struct Xoro { lo: vec2<u32>, hi: vec2<u32> };

fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    let carry = select(0u, 1u, lo < a.x);
    let hi = a.y + b.y + carry;
    return vec2<u32>(lo, hi);
}

// Left shift by n in [0, 63].
fn shl64(a: vec2<u32>, n: u32) -> vec2<u32> {
    if (n == 0u) { return a; }
    if (n < 32u) {
        let lo = a.x << n;
        let hi = (a.y << n) | (a.x >> (32u - n));
        return vec2<u32>(lo, hi);
    }
    return vec2<u32>(0u, a.x << (n - 32u));
}

// Logical right shift by n in [0, 63].
fn shr64(a: vec2<u32>, n: u32) -> vec2<u32> {
    if (n == 0u) { return a; }
    if (n < 32u) {
        let hi = a.y >> n;
        let lo = (a.x >> n) | (a.y << (32u - n));
        return vec2<u32>(lo, hi);
    }
    return vec2<u32>(a.y >> (n - 32u), 0u);
}

// Rotate left by n in [1, 63]. shl64(a,n) and shr64(a,64-n) occupy disjoint
// bit positions, so bitwise-OR is the correct recombination.
fn rotl64(a: vec2<u32>, n: u32) -> vec2<u32> {
    return shl64(a, n) | shr64(a, 64u - n);
}

// Full 32x32 -> 64 unsigned multiply (WGSL has no umulhi). Returns (lo, hi).
fn mul_u32_full(x: u32, y: u32) -> vec2<u32> {
    let xl = x & 0xFFFFu; let xh = x >> 16u;
    let yl = y & 0xFFFFu; let yh = y >> 16u;
    let ll = xl * yl;
    let lh = xl * yh;
    let hl = xh * yl;
    let hh = xh * yh;
    let cross = lh + hl;                       // up to ~2^33
    let cross_carry = select(0u, 1u, cross < lh);
    let lo = ll + (cross << 16u);
    let lo_carry = select(0u, 1u, lo < ll);
    let hi = hh + (cross >> 16u) + (cross_carry << 16u) + lo_carry;
    return vec2<u32>(lo, hi);
}

// Low 64 bits of a*b. The high half of the product is discarded (matches C
// uint64_t multiply wraparound). Only the low 32 bits of the cross term
// (a.x*b.y + a.y*b.x) survive the <<32, so wrapping u32 math is exact there.
fn mul64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let t1 = mul_u32_full(a.x, b.x);
    let cross = a.x * b.y + a.y * b.x;         // low 32 bits only
    return vec2<u32>(t1.x, t1.y + cross);
}

// cubiomes xSetSeed (rng.h:185).
fn x_set_seed(value: vec2<u32>) -> Xoro {
    let XL = vec2<u32>(0x7f4a7c15u, 0x9e3779b9u);
    let XH = vec2<u32>(0xf3bcc909u, 0x6a09e667u);
    let A  = vec2<u32>(0x1ce4e5b9u, 0xbf58476du);
    let B  = vec2<u32>(0x133111ebu, 0x94d049bbu);
    var l = value ^ XH;
    var h = add64(l, XL);
    l = mul64(l ^ shr64(l, 30u), A);
    h = mul64(h ^ shr64(h, 30u), A);
    l = mul64(l ^ shr64(l, 27u), B);
    h = mul64(h ^ shr64(h, 27u), B);
    l = l ^ shr64(l, 31u);
    h = h ^ shr64(h, 31u);
    return Xoro(l, h);
}

// cubiomes xNextLong (rng.h:203). Multiply-free.
fn x_next_long(xr: ptr<function, Xoro>) -> vec2<u32> {
    let l = (*xr).lo;
    let h = (*xr).hi;
    let n = add64(rotl64(add64(l, h), 17u), l);
    let h2 = h ^ l;
    (*xr).lo = rotl64(l, 49u) ^ h2 ^ shl64(h2, 21u);
    (*xr).hi = rotl64(h2, 28u);
    return n;
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_seeds) { return; }
    var xr = x_set_seed(seeds[i]);
    let base = i * params.k;
    for (var j: u32 = 0u; j < params.k; j = j + 1u) {
        outp[base + j] = x_next_long(&xr);
    }
}

// -----------------------------------------------------------------------------
// Phase B: per-octave Perlin init (cubiomes `xPerlinInit`, noise.c:79).
//
// Given a raw Xoroshiro state, derive a PerlinNoise's a/b/c offsets and its
// 256-byte permutation table. Reuses the RNG above (x_next_long / mul_u32_full).
//
// Accuracy contract:
//   * a, b, c = xNextDouble * 256  — f32-TOLERANT (xNextDouble is a double in
//     cubiomes; here it is approximated in f32). These offsets are added to
//     coordinates and floored, so f32 round-off is acceptable (same concession
//     as the existing Phase 6c noise primitives).
//   * the permutation table is BIT-EXACT. The Fisher-Yates shuffle is driven
//     only by xNextInt (pure integer), and xNextDouble/xNextInt each advance
//     the Xoroshiro stream by a fixed number of xNextLong draws regardless of
//     the f32 imprecision in a/b/c, so stream alignment never drifts.
// -----------------------------------------------------------------------------

struct PiParams { n_states: u32, _p0: u32, _p1: u32, _p2: u32 };

// Bindings are reused per-entry-point (naga only validates bindings reachable
// from a given entry point — see gpu_noise.wgsl, which likewise reuses
// @group(0) @binding(0..3) for two distinct pipelines).
@group(0) @binding(0) var<uniform> pi_params: PiParams;
// Two vec2<u32> per state: [lo_limbs, hi_limbs].
@group(0) @binding(1) var<storage, read> pi_states: array<vec2<u32>>;
// One vec4<f32> per state: (a, b, c, _unused).
@group(0) @binding(2) var<storage, read_write> pi_abc: array<vec4<f32>>;
// 64 u32 per state: 256 perm bytes packed little-endian.
@group(0) @binding(3) var<storage, read_write> pi_perm: array<u32>;

// cubiomes xNextDouble (rng.h:227): (xNextLong >> 11) * 2^-53. Approximated in
// f32 as hi*2^-32 + lo*2^-64 (only a/b/c depend on the value; f32-tolerant).
fn x_next_double_f32(xr: ptr<function, Xoro>) -> f32 {
    let n = x_next_long(xr);
    return f32(n.y) * 2.3283064365386963e-10 + f32(n.x) * 5.421010862427522e-20;
}

// cubiomes xNextInt (rng.h:214) with bounded n (here always 1..256). Pure
// integer; BIT-EXACT, including the rejection loop that preserves stream
// alignment with the C implementation.
fn x_next_int(xr: ptr<function, Xoro>, n: u32) -> u32 {
    let nl = x_next_long(xr);
    var r = mul_u32_full(nl.x, n); // (low32, high32) of (xNextLong & 0xFFFFFFFF) * n
    if (r.x < n) {
        let thresh = (0u - n) % n; // (~n + 1) % n in uint32 wraparound
        loop {
            if (r.x >= thresh) { break; }
            let nl2 = x_next_long(xr);
            r = mul_u32_full(nl2.x, n);
        }
    }
    return r.y; // r >> 32
}

@compute @workgroup_size(64)
fn cs_perlin_init(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if (s >= pi_params.n_states) { return; }
    var xr = Xoro(pi_states[2u * s], pi_states[2u * s + 1u]);

    // a/b/c drawn first, in order (xPerlinInit noise.c:83-85).
    let a = x_next_double_f32(&xr) * 256.0;
    let b = x_next_double_f32(&xr) * 256.0;
    let c = x_next_double_f32(&xr) * 256.0;
    pi_abc[s] = vec4<f32>(a, b, c, 0.0);

    // Identity table, then Fisher-Yates: j = xNextInt(256 - i) + i; swap(i, j).
    var idx: array<u32, 256>;
    for (var i = 0u; i < 256u; i = i + 1u) { idx[i] = i; }
    for (var i = 0u; i < 256u; i = i + 1u) {
        let j = x_next_int(&xr, 256u - i) + i;
        let tmp = idx[i];
        idx[i] = idx[j];
        idx[j] = tmp;
    }

    let base = s * 64u;
    for (var w = 0u; w < 64u; w = w + 1u) {
        let b0 = idx[w * 4u + 0u];
        let b1 = idx[w * 4u + 1u];
        let b2 = idx[w * 4u + 2u];
        let b3 = idx[w * 4u + 3u];
        pi_perm[base + w] = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
    }
}
