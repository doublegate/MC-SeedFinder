// Phase D1: WGSL port of cubiomes' depth-spline (`getSpline`, biomenoise.c:1074).
//
// np[NP_DEPTH] = getSpline(offset_tree, {c, e, w_t, w}) + 0.015, where
//   w_t = -3 * (|(|w| - 0.6666667)| - 0.33333334).
// The spline tree is seed-independent (built by initBiomeNoise from the MC
// version) and is flattened on the CPU (`mcsf_depth_spline_flatten`) so every
// child node has a LOWER flat index than its parent. That lets this walker
// evaluate nodes forward (0..count) into a results array with no recursion and
// no explicit stack — the root is the final node.
//
// Accuracy: cubiomes' getSpline is f32 throughout, so this is naturally close,
// but climate inputs (c, e, w) arrive as f32 here vs f64 in cubiomes — depth is
// therefore f32-TOLERANT, matching the existing Phase 6c noise concession.
//
// Node encoding (parallel arrays, one slot per node, <=12 children per node):
//   typ[i]      : SP_* climate index (0..3) for internal nodes, -1 for leaves
//   len[i]      : child/knot count (1 for a leaf)
//   fix[i]      : leaf constant value (internal nodes ignore it)
//   loc[i*12+k] : knot location k
//   der[i*12+k] : knot derivative k
//   child[i*12+k]: flat index of child spline k

struct DepthParams {
    n_inputs: u32,
    n_nodes: u32,      // root index = n_nodes - 1
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> dp: DepthParams;
// Per-node scalars: (typ as f32-bitcast i32, len, fix, _pad).
@group(0) @binding(1) var<storage, read> node_typ: array<i32>;
@group(0) @binding(2) var<storage, read> node_len: array<u32>;
@group(0) @binding(3) var<storage, read> node_fix: array<f32>;
// Flattened 12-wide knot arrays.
@group(0) @binding(4) var<storage, read> node_loc: array<f32>;
@group(0) @binding(5) var<storage, read> node_der: array<f32>;
@group(0) @binding(6) var<storage, read> node_child: array<i32>;
// Inputs: one vec4<f32> per sample = (c, e, w, _unused). w_t derived here.
@group(0) @binding(7) var<storage, read> depth_inputs: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> depth_outputs: array<f32>;

fn lerp_f(t: f32, a: f32, b: f32) -> f32 {
    return a + t * (b - a);
}

// Max nodes the flattener can emit (SplineStack stack[42]+fstack[151] = 193).
const MAX_NODES: u32 = 256u;

// Evaluate the whole flattened spline tree for one (c, e, w_t, w) vals vector.
// `res[i]` holds node i's value; because children precede parents, a forward
// pass fills every dependency before it is read.
fn eval_spline(vals: vec4<f32>) -> f32 {
    var res: array<f32, 256>;
    let n = dp.n_nodes;
    for (var i = 0u; i < n; i = i + 1u) {
        if (node_typ[i] < 0) {
            res[i] = node_fix[i];
            continue;
        }
        let typ = u32(node_typ[i]);
        let len = node_len[i];
        // f = vals[typ]; typ in 0..3 indexes (c, e, w_t, w).
        var f: f32;
        switch (typ) {
            case 0u: { f = vals.x; }
            case 1u: { f = vals.y; }
            case 2u: { f = vals.z; }
            default: { f = vals.w; }
        }
        let base = i * 12u;

        // Find first knot whose loc >= f (cubiomes' linear scan).
        var k = len;
        for (var j = 0u; j < len; j = j + 1u) {
            if (node_loc[base + j] >= f) { k = j; break; }
        }

        if (k == 0u || k == len) {
            var kk = k;
            if (kk != 0u) { kk = kk - 1u; }
            let cidx = u32(node_child[base + kk]);
            res[i] = res[cidx] + node_der[base + kk] * (f - node_loc[base + kk]);
            continue;
        }

        let c1 = u32(node_child[base + k - 1u]);
        let c2 = u32(node_child[base + k]);
        let g = node_loc[base + k - 1u];
        let h = node_loc[base + k];
        let kf = (f - g) / (h - g);
        let l = node_der[base + k - 1u];
        let m = node_der[base + k];
        let nv = res[c1];
        let ov = res[c2];
        let pv = l * (h - g) - (ov - nv);
        let qv = -m * (h - g) + (ov - nv);
        res[i] = lerp_f(kf, nv, ov) + kf * (1.0 - kf) * lerp_f(kf, pv, qv);
    }
    return res[n - 1u];
}

@compute @workgroup_size(64)
fn cs_depth(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dp.n_inputs) { return; }
    let inp = depth_inputs[i];
    let c = inp.x;
    let e = inp.y;
    let w = inp.z;
    // w_t = -3 * (|(|w| - 0.6666667)| - 0.33333334)  (biomenoise.c:1520)
    let w_t = -3.0 * (abs(abs(w) - 0.6666667) - 0.33333334);
    let off = eval_spline(vec4<f32>(c, e, w_t, w)) + 0.015;
    // mcsf_compute_depth at y=0: 1 - 83/160 + off (the y term is 0).
    depth_outputs[i] = 1.0 - 83.0 / 160.0 + off;
}
