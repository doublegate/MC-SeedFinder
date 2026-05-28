// GPU compute prefilter for structure-only conditions, generalised in 6b to
// handle multiple predicates with a combinator (any_of / all_of / cluster).
//
// Each predicate is a single NearbyStructure (structure type + centre +
// max_distance + pre-computed region scan bounds). The combinator decides
// how per-predicate matches combine:
//   - 0 = ANY_OF:   true iff ≥1 predicate has any structure-in-radius hit
//   - 1 = ALL_OF:   true iff EVERY predicate has any structure-in-radius hit
//   - 2 = CLUSTER:  true iff the TOTAL count of hits across all predicates
//                   reaches min_count (the marquee quad-hut search)
//
// Java's 48-bit LCG and the structure placement step are bit-identical to the
// Rust pure-RNG / pure-structure code in `java_random.rs` / `structures.rs`;
// dedicated parity tests dispatch GPU + CPU side by side and assert equal
// match lists, so any drift in this shader is loud.
//
// All multi-limb constants (RNG multiplier, region mixers, salts, distance
// thresholds) arrive via the uniform buffer so the host owns hex/decimal
// transcription in one place.

const MULT_LO: u32 = 0xDEECE66Du;
const MULT_HI: u32 = 0x5u;
const INC_LO: u32  = 0xBu;
const MASK_HI: u32 = 0xFFFFu;

const COMB_ANY: u32 = 0u;
const COMB_ALL: u32 = 1u;
const COMB_CLUSTER: u32 = 2u;

const MAX_PREDICATES: u32 = 8u;

// --- u32 × u32 → (low32, high32) -------------------------------------------
fn umul32(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;

    let ll = a_lo * b_lo;
    let lh = a_lo * b_hi;
    let hl = a_hi * b_lo;
    let hh = a_hi * b_hi;

    let mid = lh + hl;
    let mid_carry = select(0u, 1u, mid < lh);

    let low_from_mid = mid << 16u;
    let high_from_mid = (mid >> 16u) | (mid_carry << 16u);

    let lo32 = ll + low_from_mid;
    let lo_carry = select(0u, 1u, lo32 < ll);
    let hi32 = hh + high_from_mid + lo_carry;

    return vec2<u32>(lo32, hi32);
}

fn umul64_lo(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    let p_ll = umul32(a_lo, b_lo);
    let p_lh = a_lo * b_hi;
    let p_hl = a_hi * b_lo;
    let hi32 = p_ll.y + p_lh + p_hl;
    return vec2<u32>(p_ll.x, hi32);
}

fn uadd64(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    let lo = a_lo + b_lo;
    let carry = select(0u, 1u, lo < a_lo);
    let hi = a_hi + b_hi + carry;
    return vec2<u32>(lo, hi);
}

fn signext64_i32(v: i32) -> vec2<u32> {
    let lo = bitcast<u32>(v);
    let hi: u32 = select(0u, 0xFFFFFFFFu, v < 0);
    return vec2<u32>(lo, hi);
}

// --- Java RNG -------------------------------------------------------------

fn rng_set_seed(seed_lo: u32, seed_hi: u32) -> vec2<u32> {
    let lo = seed_lo ^ MULT_LO;
    let hi = (seed_hi ^ MULT_HI) & MASK_HI;
    return vec2<u32>(lo, hi);
}

fn rng_step_and_next(state_lo: u32, state_hi: u32, bits: u32) -> vec3<u32> {
    let p0 = umul32(state_lo, MULT_LO);
    let p1 = state_lo * MULT_HI;
    let p2 = state_hi * MULT_LO;

    let lo = p0.x + INC_LO;
    let lo_carry = select(0u, 1u, lo < p0.x);
    let hi_raw = p0.y + p1 + p2 + lo_carry;
    let hi = hi_raw & MASK_HI;

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

fn rng_next_int_bound_pow2(state_lo: u32, state_hi: u32, bound: u32) -> vec3<u32> {
    let r = rng_step_and_next(state_lo, state_hi, 31u);
    let prod = umul32(bound, r.z);
    let result = (prod.x >> 31u) | (prod.y << 1u);
    return vec3<u32>(r.x, r.y, result);
}

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
        let candidate = (bits - val) + (bound - 1u);
        if ((candidate & 0x80000000u) == 0u) {
            return vec3<u32>(lo, hi, val);
        }
    }
    return vec3<u32>(lo, hi, 0u);
}

// --- Uniform layout --------------------------------------------------------
//
// `Predicate` is sized to 64 bytes (16 u32s) so the WGSL uniform-array
// element stride matches what the host writes. Mirrors `PredicateUniform`
// in `gpu.rs` 1:1 — keep these in sync.

struct Predicate {
    salt_lo: u32,
    salt_hi: u32,
    spacing: i32,
    chunk_range: u32,
    spread_type: u32,         // 0=Linear, 1=Triangular
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
};

struct SearchParams {
    start_seed_lo: u32,
    start_seed_hi: u32,
    count: u32,
    combinator: u32,           // 0=any_of, 1=all_of, 2=cluster
    num_predicates: u32,
    min_count: u32,            // only meaningful for cluster
    region_mul_x_lo: u32,
    region_mul_x_hi: u32,
    region_mul_z_lo: u32,
    region_mul_z_hi: u32,
    _pad0: u32,
    _pad1: u32,
    // Array of predicates. WGSL stride = 64 bytes per Predicate.
    predicates: array<Predicate, 8>,
};

@group(0) @binding(0) var<uniform> params: SearchParams;
@group(0) @binding(1) var<storage, read_write> matches_out: array<u32>;

// Returns the hit count for a single predicate against a given seed.
// Walks the region grid; for any_of/all_of the caller breaks early after the
// first hit, but for cluster we want the full count, so the inner loop never
// short-circuits here.
fn count_predicate_hits(
    pred: Predicate,
    world_seed_lo: u32,
    world_seed_hi: u32,
    early_exit_on_first: bool,
) -> u32 {
    var hits: u32 = 0u;
    var rx: i32 = pred.rx_min;
    loop {
        if (rx > pred.rx_max) { break; }
        let rx_pair = signext64_i32(rx);
        let rx_mul = umul64_lo(rx_pair.x, rx_pair.y,
                               params.region_mul_x_lo, params.region_mul_x_hi);

        var rz: i32 = pred.rz_min;
        loop {
            if (rz > pred.rz_max) { break; }
            let rz_pair = signext64_i32(rz);
            let rz_mul = umul64_lo(rz_pair.x, rz_pair.y,
                                   params.region_mul_z_lo, params.region_mul_z_hi);

            // region_seed = world_seed + salt + rx*MX + rz*MZ
            let s1 = uadd64(world_seed_lo, world_seed_hi, pred.salt_lo, pred.salt_hi);
            let s2 = uadd64(s1.x, s1.y, rx_mul.x, rx_mul.y);
            let s3 = uadd64(s2.x, s2.y, rz_mul.x, rz_mul.y);

            var rng = rng_set_seed(s3.x, s3.y);

            var ox: i32;
            var oz: i32;
            if (pred.spread_type == 0u) {
                let a = rng_next_int_bound(rng.x, rng.y, pred.chunk_range);
                rng = vec2<u32>(a.x, a.y);
                let b = rng_next_int_bound(rng.x, rng.y, pred.chunk_range);
                rng = vec2<u32>(b.x, b.y);
                ox = i32(a.z);
                oz = i32(b.z);
            } else {
                let a = rng_next_int_bound(rng.x, rng.y, pred.chunk_range);
                rng = vec2<u32>(a.x, a.y);
                let b = rng_next_int_bound(rng.x, rng.y, pred.chunk_range);
                rng = vec2<u32>(b.x, b.y);
                let c = rng_next_int_bound(rng.x, rng.y, pred.chunk_range);
                rng = vec2<u32>(c.x, c.y);
                let d = rng_next_int_bound(rng.x, rng.y, pred.chunk_range);
                rng = vec2<u32>(d.x, d.y);
                ox = (i32(a.z) + i32(c.z)) / 2;
                oz = (i32(b.z) + i32(d.z)) / 2;
            }

            let chunk_x = rx * pred.spacing + ox;
            let chunk_z = rz * pred.spacing + oz;
            let block_x = chunk_x * 16 + 8;
            let block_z = chunk_z * 16 + 8;

            let dx_pair = signext64_i32(block_x - pred.centre_x);
            let dz_pair = signext64_i32(block_z - pred.centre_z);
            let dx_sq = umul64_lo(dx_pair.x, dx_pair.y, dx_pair.x, dx_pair.y);
            let dz_sq = umul64_lo(dz_pair.x, dz_pair.y, dz_pair.x, dz_pair.y);
            let sum = uadd64(dx_sq.x, dx_sq.y, dz_sq.x, dz_sq.y);

            if (sum.y < pred.max_dist_sq_hi ||
                (sum.y == pred.max_dist_sq_hi && sum.x <= pred.max_dist_sq_lo)) {
                hits = hits + 1u;
                if (early_exit_on_first) {
                    return hits;
                }
            }
            rz = rz + 1;
        }
        rx = rx + 1;
    }
    return hits;
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let seed_pair = uadd64(params.start_seed_lo, params.start_seed_hi, idx, 0u);
    let world_seed_lo = seed_pair.x;
    let world_seed_hi = seed_pair.y;

    // For any_of / all_of we only care whether each predicate had ANY hit, so
    // pass early_exit_on_first=true to stop the inner loop on first hit.
    // For cluster we need the full count.
    let early_exit: bool = params.combinator != COMB_CLUSTER;

    var any_matched: bool = false;
    var all_matched: bool = true;
    var hits_total: u32 = 0u;

    var p: u32 = 0u;
    loop {
        if (p >= params.num_predicates) { break; }
        let pred = params.predicates[p];
        let pred_hits = count_predicate_hits(pred, world_seed_lo, world_seed_hi, early_exit);

        if (pred_hits > 0u) {
            any_matched = true;
            hits_total = hits_total + pred_hits;
        } else {
            all_matched = false;
        }

        // Combinator-specific early exit:
        if (params.combinator == COMB_ANY && any_matched) {
            matches_out[idx] = 1u;
            return;
        }
        if (params.combinator == COMB_ALL && !all_matched) {
            matches_out[idx] = 0u;
            return;
        }
        if (params.combinator == COMB_CLUSTER && hits_total >= params.min_count) {
            matches_out[idx] = 1u;
            return;
        }
        p = p + 1u;
    }

    var result: u32 = 0u;
    if (params.combinator == COMB_ANY) {
        result = select(0u, 1u, any_matched);
    } else if (params.combinator == COMB_ALL) {
        result = select(0u, 1u, all_matched);
    } else {
        result = select(0u, 1u, hits_total >= params.min_count);
    }
    matches_out[idx] = result;
}
