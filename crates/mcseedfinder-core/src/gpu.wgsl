// GPU compute prefilter for "is structure X within R blocks of (cx, cz)?"
//
// Implements Java's 48-bit LCG (`java.util.Random`) plus the structure
// placement step in WGSL — bit-identical to the Rust pure-RNG / pure-
// structure code in `java_random.rs` and `structures.rs`, dispatched in
// parallel across a seed range. Output is a u32 per seed (1 = match, else 0).
//
// WGSL has no native u64; the 48-bit RNG seed lives in two u32 limbs
// `(lo, hi)` where `hi` uses only its low 16 bits. The 48×48 → 48 multiply
// uses schoolbook u32×u32 with manual carries; the 64×64 → low-64 multiply
// (region-seed mixing) is similarly schoolbook. Every multi-limb constant
// (multiplier, salt, region mixers, distance threshold) arrives via the
// uniform buffer so the host owns hex/decimal conversion in one place.

const MULT_LO: u32 = 0xDEECE66Du;       // low 32 of the LCG multiplier 0x5DEECE66D
const MULT_HI: u32 = 0x5u;              // high 16 of the multiplier
const INC_LO: u32  = 0xBu;
const MASK_HI: u32 = 0xFFFFu;           // 48-bit seed mask: hi limb keeps only low 16 bits

// --- u32 × u32 → (low32, high32) -------------------------------------------
fn umul32(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;

    let ll = a_lo * b_lo;     // bits 0..32
    let lh = a_lo * b_hi;     // bits 16..48
    let hl = a_hi * b_lo;     // bits 16..48
    let hh = a_hi * b_hi;     // bits 32..64

    // Sum cross terms with explicit carry.
    let mid = lh + hl;
    let mid_carry = select(0u, 1u, mid < lh);

    let low_from_mid = mid << 16u;
    let high_from_mid = (mid >> 16u) | (mid_carry << 16u);

    let lo32 = ll + low_from_mid;
    let lo_carry = select(0u, 1u, lo32 < ll);
    let hi32 = hh + high_from_mid + lo_carry;

    return vec2<u32>(lo32, hi32);
}

// --- 64-bit wrapping multiply (low-64-only result) -------------------------
//     (a_lo + a_hi<<32) * (b_lo + b_hi<<32)  mod 2^64
//
// Contributions to bits 0..64 of a 64×64 → 128 product:
//   low(a)*low(b)               full 64 bits
//   low(a)*high(b) << 32        only its low 32 land in bits 32..64
//   high(a)*low(b) << 32        same
//   high(a)*high(b) << 64       dropped
fn umul64_lo(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    let p_ll = umul32(a_lo, b_lo);
    let p_lh = a_lo * b_hi;        // u32 wrap is correct here — overflow above bit 64 is dropped
    let p_hl = a_hi * b_lo;
    let hi32 = p_ll.y + p_lh + p_hl;
    return vec2<u32>(p_ll.x, hi32);
}

// --- 64-bit wrapping add ---------------------------------------------------
fn uadd64(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    let lo = a_lo + b_lo;
    let carry = select(0u, 1u, lo < a_lo);
    let hi = a_hi + b_hi + carry;
    return vec2<u32>(lo, hi);
}

// Sign-extend an i32 into 64-bit two's-complement limbs.
fn signext64_i32(v: i32) -> vec2<u32> {
    let lo = bitcast<u32>(v);
    let hi: u32 = select(0u, 0xFFFFFFFFu, v < 0);
    return vec2<u32>(lo, hi);
}

// ---------------------------------------------------------------------------
// Java RNG
// ---------------------------------------------------------------------------

// set_seed: ((seed_in ^ MULTIPLIER) & MASK48), where seed_in is held as
// (seed_lo, seed_hi) u32 limbs of a signed 64-bit seed.
fn rng_set_seed(seed_lo: u32, seed_hi: u32) -> vec2<u32> {
    let lo = seed_lo ^ MULT_LO;
    let hi = (seed_hi ^ MULT_HI) & MASK_HI;
    return vec2<u32>(lo, hi);
}

// Advance the 48-bit seed by one LCG step AND extract `next(bits)`.
// Returns (new_lo, new_hi, next_bits_value).
fn rng_step_and_next(state_lo: u32, state_hi: u32, bits: u32) -> vec3<u32> {
    // seed * MULT, mod 2^48:
    //   low(seed)*low(MULT)  → full 64 contribution
    //   low(seed)*high(MULT) → shifted 32; low 32 of product lands in bits 32..64
    //   high(seed)*low(MULT) → shifted 32; ditto
    //   high(seed)*high(MULT) → above bit 48; dropped by mask anyway
    let p0 = umul32(state_lo, MULT_LO);
    let p1 = state_lo * MULT_HI;
    let p2 = state_hi * MULT_LO;

    let lo = p0.x + INC_LO;
    let lo_carry = select(0u, 1u, lo < p0.x);
    let hi_raw = p0.y + p1 + p2 + lo_carry;
    let hi = hi_raw & MASK_HI;

    // next(bits) = (seed >> (48 - bits)) cast to u32. The combined u48 has
    // `lo` in bits 0..32 and `hi` (low 16 bits) in bits 32..48.
    let shift = 48u - bits;
    var result: u32;
    if (shift >= 32u) {
        result = hi >> (shift - 32u);
    } else if (shift == 0u) {
        result = lo;
    } else {
        result = (hi << (32u - shift)) | (lo >> shift);
    }

    return vec3<u32>(lo, hi, result);
}

// next_int_bound: power-of-two fast path matching the CPU impl.
fn rng_next_int_bound_pow2(state_lo: u32, state_hi: u32, bound: u32) -> vec3<u32> {
    let r = rng_step_and_next(state_lo, state_hi, 31u);
    // ((bound as i64) * (bits as i64)) >> 31
    let prod = umul32(bound, r.z);
    let result = (prod.x >> 31u) | (prod.y << 1u);
    return vec3<u32>(r.x, r.y, result);
}

// next_int_bound: general case, rejection-sampling on next(31) % bound.
fn rng_next_int_bound(state_lo: u32, state_hi: u32, bound: u32) -> vec3<u32> {
    if ((bound & (bound - 1u)) == 0u) {
        return rng_next_int_bound_pow2(state_lo, state_hi, bound);
    }
    var lo = state_lo;
    var hi = state_hi;
    for (var attempts: u32 = 0u; attempts < 32u; attempts = attempts + 1u) {
        let r = rng_step_and_next(lo, hi, 31u);
        lo = r.x;
        hi = r.y;
        let bits = r.z;
        let val = bits % bound;
        // Java: bits - val + (bound - 1) ≥ 0   (signed i32)
        // The sum is in [0, 2*(2^31 - 1)]; bit-31 set ⇒ wrapped past i32::MAX.
        let candidate = (bits - val) + (bound - 1u);
        if ((candidate & 0x80000000u) == 0u) {
            return vec3<u32>(lo, hi, val);
        }
    }
    return vec3<u32>(lo, hi, 0u);
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

struct SearchParams {
    // Seed range (start_seed is i64, carried as limbs).
    start_seed_lo: u32,
    start_seed_hi: u32,
    count: u32,
    // Structure config:
    spread_type: u32,              // 0 = Linear, 1 = Triangular
    salt_lo: u32,
    salt_hi: u32,
    spacing: i32,
    chunk_range: u32,              // = spacing - separation (positive in our table)
    // Region scan bounds (computed CPU-side).
    rx_min: i32,
    rx_max: i32,
    rz_min: i32,
    rz_max: i32,
    centre_x: i32,
    centre_z: i32,
    // Region mixers (host hands these in to avoid in-shader hex / decimal conversion).
    region_mul_x_lo: u32,
    region_mul_x_hi: u32,
    region_mul_z_lo: u32,
    region_mul_z_hi: u32,
    // Distance threshold (max_distance squared, as unsigned i64 limbs).
    max_dist_sq_lo: u32,
    max_dist_sq_hi: u32,
}

@group(0) @binding(0) var<uniform> params: SearchParams;
@group(0) @binding(1) var<storage, read_write> matches_out: array<u32>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    // seed = start_seed + idx (i64 wrap)
    let seed_pair = uadd64(params.start_seed_lo, params.start_seed_hi, idx, 0u);
    let world_seed_lo = seed_pair.x;
    let world_seed_hi = seed_pair.y;

    var hit: u32 = 0u;
    var rx: i32 = params.rx_min;
    loop {
        if (rx > params.rx_max || hit == 1u) { break; }
        let rx_pair = signext64_i32(rx);
        let rx_mul = umul64_lo(rx_pair.x, rx_pair.y,
                               params.region_mul_x_lo, params.region_mul_x_hi);

        var rz: i32 = params.rz_min;
        loop {
            if (rz > params.rz_max || hit == 1u) { break; }
            let rz_pair = signext64_i32(rz);
            let rz_mul = umul64_lo(rz_pair.x, rz_pair.y,
                                   params.region_mul_z_lo, params.region_mul_z_hi);

            // region_seed = world_seed + salt + rx*MX + rz*MZ (i64 wrap)
            let s1 = uadd64(world_seed_lo, world_seed_hi, params.salt_lo, params.salt_hi);
            let s2 = uadd64(s1.x, s1.y, rx_mul.x, rx_mul.y);
            let s3 = uadd64(s2.x, s2.y, rz_mul.x, rz_mul.y);

            // Java RNG init.
            var rng = rng_set_seed(s3.x, s3.y);

            var ox: i32;
            var oz: i32;
            if (params.spread_type == 0u) {
                let a = rng_next_int_bound(rng.x, rng.y, params.chunk_range);
                rng = vec2<u32>(a.x, a.y);
                let b = rng_next_int_bound(rng.x, rng.y, params.chunk_range);
                rng = vec2<u32>(b.x, b.y);
                ox = i32(a.z);
                oz = i32(b.z);
            } else {
                let a = rng_next_int_bound(rng.x, rng.y, params.chunk_range);
                rng = vec2<u32>(a.x, a.y);
                let b = rng_next_int_bound(rng.x, rng.y, params.chunk_range);
                rng = vec2<u32>(b.x, b.y);
                let c = rng_next_int_bound(rng.x, rng.y, params.chunk_range);
                rng = vec2<u32>(c.x, c.y);
                let d = rng_next_int_bound(rng.x, rng.y, params.chunk_range);
                rng = vec2<u32>(d.x, d.y);
                ox = (i32(a.z) + i32(c.z)) / 2;
                oz = (i32(b.z) + i32(d.z)) / 2;
            }

            let chunk_x = rx * params.spacing + ox;
            let chunk_z = rz * params.spacing + oz;
            let block_x = chunk_x * 16 + 8;
            let block_z = chunk_z * 16 + 8;

            let dx_pair = signext64_i32(block_x - params.centre_x);
            let dz_pair = signext64_i32(block_z - params.centre_z);
            let dx_sq = umul64_lo(dx_pair.x, dx_pair.y, dx_pair.x, dx_pair.y);
            let dz_sq = umul64_lo(dz_pair.x, dz_pair.y, dz_pair.x, dz_pair.y);
            let sum = uadd64(dx_sq.x, dx_sq.y, dz_sq.x, dz_sq.y);

            // Both `sum` and `max_dist_sq` are non-negative i64 values (squared
            // distances + a positive threshold), so unsigned compare suffices.
            if (sum.y < params.max_dist_sq_hi ||
                (sum.y == params.max_dist_sq_hi && sum.x <= params.max_dist_sq_lo)) {
                hit = 1u;
            }
            rz = rz + 1;
        }
        rx = rx + 1;
    }
    matches_out[idx] = hit;
}
