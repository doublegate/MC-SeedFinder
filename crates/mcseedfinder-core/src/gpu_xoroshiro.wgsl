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
