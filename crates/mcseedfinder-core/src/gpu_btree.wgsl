// Phase 6c-4: WGSL port of cubiomes' biome b-tree walker (`climateToBiome`).
//
// cubiomes implements `get_resulting_node` recursively (biomenoise.c:1388).
// WGSL has no function recursion, so we flatten it with an explicit stack.
// Max stack depth = len(steps) - 1 (btree21wd: 5 levels), so a 6-frame
// stack is always sufficient.
//
// Arithmetic notes:
//   - cubiomes' `np[6]` are u64 (storing int64 bit-patterns). WGSL has no
//     native u64, so we represent each value as `vec2<u32>` = (lo, hi).
//   - cubiomes' `param[][2]` are int32 but get sign-extended to int64 on
//     subtraction. We do the same with `i32` → `i64` (also vec2<u32>).
//   - `ds` (sum of 6 squared diffs) can exceed u32, so it's also u64.
//   - The signed-positive test `(int64_t)a > 0` becomes "hi bit clear AND
//     value non-zero" in our u64 representation.
//
// Bind group:
//   binding 0: BTreeParams uniform
//   binding 1: storage<read> steps[]   (u32)
//   binding 2: storage<read> param[]   (i32, flat — pairs are adjacent)
//   binding 3: storage<read> nodes[]   (u64 packed as vec2<u32>: x=lo y=hi)
//   binding 4: storage<read> np_inputs[]  (6 × vec2<u32> per sample)
//   binding 5: storage<read_write> biomes[]  (u32)

struct BTreeParams {
    order: u32,
    nodes_len: u32,
    steps_len: u32,
    sample_count: u32,
};

@group(0) @binding(0) var<uniform> bt: BTreeParams;
@group(0) @binding(1) var<storage, read> bt_steps: array<u32>;
@group(0) @binding(2) var<storage, read> bt_param: array<i32>;
@group(0) @binding(3) var<storage, read> bt_nodes: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read> bt_np_in: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read_write> bt_biome_out: array<u32>;

// --- 64-bit unsigned helpers (lo = x, hi = y) ---

fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    let carry = select(0u, 1u, lo < a.x);
    let hi = a.y + b.y + carry;
    return vec2<u32>(lo, hi);
}

fn u64_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x - b.x;
    let borrow = select(0u, 1u, a.x < b.x);
    let hi = a.y - b.y - borrow;
    return vec2<u32>(lo, hi);
}

fn u64_lt(a: vec2<u32>, b: vec2<u32>) -> bool {
    if (a.y < b.y) { return true; }
    if (a.y > b.y) { return false; }
    return a.x < b.x;
}

/// True iff the value (interpreted as int64) is strictly positive.
/// I.e., sign bit clear (hi bit 31 = 0) AND value non-zero.
fn i64_positive(a: vec2<u32>) -> bool {
    if ((a.y & 0x80000000u) != 0u) { return false; }
    return (a.x | a.y) != 0u;
}

/// 32×32 → 64-bit unsigned multiply. Splits inputs into 16-bit halves and
/// does four 16×16 multiplies that fit in u32, then accumulates with
/// carry. Used for d * d where d ≤ ~40000 fits in u32 unsigned.
fn u32_mul_to_u64(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;

    let ll = a_lo * b_lo;        // bits 0..31 (uses bits 0..32 of product)
    let lh = a_lo * b_hi;        // bits 16..47
    let hl = a_hi * b_lo;        // bits 16..47
    let hh = a_hi * b_hi;        // bits 32..63

    // Assemble: result = ll + ((lh+hl) << 16) + (hh << 32)
    let mid = lh + hl;                       // might overflow → carry bit
    let mid_carry = select(0u, 1u, mid < lh);
    let mid_lo = (mid & 0xFFFFu) << 16u;     // contributes to lo
    let mid_hi = (mid >> 16u) | (mid_carry << 16u);  // contributes to hi
    let lo = ll + mid_lo;
    let lo_carry = select(0u, 1u, lo < ll);
    let hi = hh + mid_hi + lo_carry;
    return vec2<u32>(lo, hi);
}

/// cubiomes' `get_np_dist` — 6-D squared distance between `np` (read from
/// `bt_np_in` at `np_base`) and the param ranges referenced by node `idx`.
fn get_np_dist(np_base: u32, idx: u32) -> vec2<u32> {
    let node = bt_nodes[idx];   // (lo, hi)
    var ds = vec2<u32>(0u, 0u);
    for (var i: u32 = 0u; i < 6u; i = i + 1u) {
        // Param byte index: byte i of the low 6 bytes of node (i.e. (lo, hi)
        // first then hi). i < 4 → from lo; i in {4,5} → from hi.
        var pidx: u32;
        if (i < 4u) {
            pidx = (node.x >> (8u * i)) & 0xFFu;
        } else {
            pidx = (node.y >> (8u * (i - 4u))) & 0xFFu;
        }

        // Load (min, max) i32 pair and sign-extend to i64 representation.
        let pmin_i32 = bt_param[2u * pidx];
        let pmax_i32 = bt_param[2u * pidx + 1u];
        let pmin_lo = bitcast<u32>(pmin_i32);
        let pmin_hi = select(0u, 0xFFFFFFFFu, (pmin_lo & 0x80000000u) != 0u);
        let pmax_lo = bitcast<u32>(pmax_i32);
        let pmax_hi = select(0u, 0xFFFFFFFFu, (pmax_lo & 0x80000000u) != 0u);
        let pmin = vec2<u32>(pmin_lo, pmin_hi);
        let pmax = vec2<u32>(pmax_lo, pmax_hi);
        let n = bt_np_in[np_base + i];

        let a = u64_sub(n, pmax);   // np - max
        let b = u64_sub(pmin, n);   // min - np
        var d = vec2<u32>(0u, 0u);
        if (i64_positive(a)) {
            d = a;
        } else if (i64_positive(b)) {
            d = b;
        }
        // d fits in u32 (max ≈ 40000 < 2^16); squaring it produces u64.
        let d_sq = u32_mul_to_u64(d.x, d.x);
        ds = u64_add(ds, d_sq);
    }
    return ds;
}

// --- Iterative get_resulting_node ---
//
// Stack frames mirror the recursion exactly. State machine inside each
// frame:
//   STATE_PRE: entry — set up step/inner/leaf/i, then transition to LOOP
//              (or return immediately if leaf level)
//   STATE_LOOP: at the top of the for-loop iteration; check inner, recurse
//               or advance to next i
//   STATE_RETURN: post-recursion — incorporate child's result, advance
//                 inner, advance i

const STATE_PRE: u32     = 0u;
const STATE_LOOP: u32    = 1u;
const STATE_RETURN: u32  = 2u;

struct Frame {
    idx: u32,
    alt: u32,
    ds_lo: u32,
    ds_hi: u32,
    depth: u32,
    step: u32,
    inner: u32,
    leaf: u32,
    i: u32,
    state: u32,
    saved_ds_lo: u32,
    saved_ds_hi: u32,
};

fn climate_to_biome(np_base: u32) -> u32 {
    var stack: array<Frame, 6>;
    var sp: u32 = 0u;
    stack[0] = Frame(
        0u,                     // idx (root)
        0u,                     // alt
        0xFFFFFFFFu,           // ds_lo (u64::MAX)
        0xFFFFFFFFu,           // ds_hi
        0u,                     // depth
        0u, 0u, 0u, 0u,         // step, inner, leaf, i
        STATE_PRE,              // state
        0u, 0u
    );
    sp = 1u;
    var result: u32 = 0u;

    // Bound on iterations: per level we do up to `order=6` children;
    // recursion depth ≤ 5; so a generous safety limit is 6^5 = 7776.
    // In practice the early-out via ds-comparison keeps this tiny.
    for (var iter: u32 = 0u; iter < 8192u; iter = iter + 1u) {
        if (sp == 0u) { break; }
        let top = sp - 1u;

        if (stack[top].state == STATE_PRE) {
            // Leaf detection: depth past end of `steps`, or step at this
            // depth is the 0 terminator. Either way, return idx.
            if (stack[top].depth >= bt.steps_len || bt_steps[stack[top].depth] == 0u) {
                result = stack[top].idx;
                sp = sp - 1u;
                continue;
            }
            // Advance depth past steps that overflow the tree.
            var d = stack[top].depth;
            var step = bt_steps[d];
            d = d + 1u;
            // Match cubiomes' do/while: always read step at least once,
            // and only re-advance while idx+step overflows the tree.
            for (var safety: u32 = 0u; safety < bt.steps_len; safety = safety + 1u) {
                if (stack[top].idx + step < bt.nodes_len) { break; }
                step = bt_steps[d];
                d = d + 1u;
            }
            stack[top].step = step;
            stack[top].depth = d;
            let node = bt_nodes[stack[top].idx];
            stack[top].inner = node.y >> 16u; // top 16 bits = node.y >> 16
            stack[top].leaf = stack[top].alt;
            stack[top].i = 0u;
            stack[top].state = STATE_LOOP;
            continue;
        }

        if (stack[top].state == STATE_LOOP) {
            if (stack[top].i >= bt.order) {
                result = stack[top].leaf;
                sp = sp - 1u;
                continue;
            }
            let ds_inner = get_np_dist(np_base, stack[top].inner);
            let cur_ds = vec2<u32>(stack[top].ds_lo, stack[top].ds_hi);
            if (u64_lt(ds_inner, cur_ds)) {
                // Save ds_inner for use after the recursive return.
                stack[top].saved_ds_lo = ds_inner.x;
                stack[top].saved_ds_hi = ds_inner.y;
                stack[top].state = STATE_RETURN;
                // Push child frame.
                stack[sp] = Frame(
                    stack[top].inner,
                    stack[top].leaf,
                    stack[top].ds_lo,
                    stack[top].ds_hi,
                    stack[top].depth,
                    0u, 0u, 0u, 0u,
                    STATE_PRE,
                    0u, 0u
                );
                sp = sp + 1u;
                continue;
            }
            // Else fall through to advance inner.
            stack[top].inner = stack[top].inner + stack[top].step;
            if (stack[top].inner >= bt.nodes_len) {
                result = stack[top].leaf;
                sp = sp - 1u;
                continue;
            }
            stack[top].i = stack[top].i + 1u;
            continue;
        }

        // STATE_RETURN
        let leaf2 = result;
        var ds_leaf2: vec2<u32>;
        if (stack[top].inner == leaf2) {
            ds_leaf2 = vec2<u32>(stack[top].saved_ds_lo, stack[top].saved_ds_hi);
        } else {
            ds_leaf2 = get_np_dist(np_base, leaf2);
        }
        let cur_ds = vec2<u32>(stack[top].ds_lo, stack[top].ds_hi);
        if (u64_lt(ds_leaf2, cur_ds)) {
            stack[top].ds_lo = ds_leaf2.x;
            stack[top].ds_hi = ds_leaf2.y;
            stack[top].leaf = leaf2;
        }
        stack[top].inner = stack[top].inner + stack[top].step;
        if (stack[top].inner >= bt.nodes_len) {
            result = stack[top].leaf;
            sp = sp - 1u;
            continue;
        }
        stack[top].i = stack[top].i + 1u;
        stack[top].state = STATE_LOOP;
    }
    return result;
}

@compute @workgroup_size(64)
fn cs_climate_to_biome(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= bt.sample_count) { return; }
    let np_base = i * 6u;
    let leaf = climate_to_biome(np_base);
    // Top 16 bits of leaf node, low 8 bits = biome id.
    let node = bt_nodes[leaf];
    let biome = (node.y >> 16u) & 0xFFu;
    bt_biome_out[i] = biome;
}
