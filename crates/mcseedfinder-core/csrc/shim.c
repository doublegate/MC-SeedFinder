/*
 * Minimal C shim over Cubitect's cubiomes, exposing just the biome-generation
 * surface mcseedfinder-core needs. Hand-written FFI (no bindgen) keeps the build
 * deterministic across clang/libclang versions and CI.
 *
 * Structures and strongholds are NOT bridged here — those are already exact in
 * the pure-Rust core (structures.rs / java_random.rs). cubiomes is used solely
 * as the authoritative biome oracle.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "biomenoise.h"
#include "finders.h"
#include "generator.h"
#include "layers.h"
#include "noise.h"
#include "rng.h"
#include "util.h"

/* Allocate/free the (large, version-dependent) cubiomes Generator on the C side
 * so Rust never needs to know its exact size or alignment. */
Generator *mcsf_generator_alloc(void) {
    return (Generator *)malloc(sizeof(Generator));
}

void mcsf_generator_free(Generator *g) {
    free(g);
}

/* Configure the generator for a Minecraft version (cubiomes MCVersion int) and
 * flags. Must be called before applying a seed. */
void mcsf_setup(Generator *g, int mc, uint32_t flags) {
    setupGenerator(g, mc, flags);
}

/* (Re)apply a world seed for the given dimension
 * (DIM_NETHER=-1, DIM_OVERWORLD=0, DIM_END=1). */
void mcsf_apply_seed(Generator *g, int dim, uint64_t seed) {
    applySeed(g, dim, seed);
}

/* Exact biome id at block coordinate (x, y, z). scale is 1 (block) or 4 (1:4).
 * Returns cubiomes' BiomeID int, or -1 on failure. */
int mcsf_biome_at(const Generator *g, int scale, int x, int y, int z) {
    return getBiomeAt(g, scale, x, y, z);
}

/* Parse a version string ("1.21", "1.18.2") to a cubiomes MCVersion int.
 * Returns 0 when the string is not a recognized version. */
int mcsf_str2mc(const char *s) {
    return str2mc(s);
}

/* Phase A oracle: emit the first `k` xNextLong outputs of a Xoroshiro128++
 * seeded from `seed` via cubiomes' xSetSeed. This is the bit-exact reference
 * the WGSL Xoroshiro port (gpu_xoroshiro.wgsl) is tested against. `out` must
 * have room for `k` uint64_t values. */
void mcsf_xoroshiro_stream(uint64_t seed, int k, uint64_t *out) {
    Xoroshiro xr;
    xSetSeed(&xr, seed);
    for (int i = 0; i < k; i++) {
        out[i] = xNextLong(&xr);
    }
}

/* Phase B oracle: run cubiomes' xPerlinInit from a raw Xoroshiro state
 * (lo, hi) and dump the resulting a/b/c offsets and 256-byte permutation
 * table. Reference for the WGSL `cs_perlin_init` entry point. `out_abc` gets 3
 * doubles; `out_perm` gets 256 bytes (the 257th duplicate entry is omitted —
 * the GPU emulates it by masking indices). */
void mcsf_xperlin_init_dump(
    uint64_t lo, uint64_t hi, double *out_abc, uint8_t *out_perm
) {
    Xoroshiro xr;
    xr.lo = lo;
    xr.hi = hi;
    PerlinNoise pn;
    memset(&pn, 0, sizeof(pn));
    xPerlinInit(&pn, &xr);
    out_abc[0] = pn.a;
    out_abc[1] = pn.b;
    out_abc[2] = pn.c;
    memcpy(out_perm, pn.d, 256);
}

/* Reference oracle for the Rust-side buried_treasure per-chunk roll
 * (structures.rs::roll_buried_treasure_chunk). Returns 1 iff cubiomes'
 * own getStructurePos(Treasure, ...) succeeds at (chunkX, chunkZ),
 * 0 otherwise. The MC version controls which structure config is used,
 * though for Treasure the config is stable from 1.13 onwards.
 *
 * Lets the Rust unit tests compare us bit-exactly against cubiomes
 * instead of just self-checking against the math we ported.
 */
int mcsf_buried_treasure_at(int mc, uint64_t world_seed, int chunk_x, int chunk_z) {
    StructureConfig sconf;
    if (!getStructureConfig(Treasure, mc, &sconf)) {
        return 0;
    }
    Pos pos;
    return getStructurePos(Treasure, mc, world_seed, chunk_x, chunk_z, &pos);
}

/* Batch biome generation for a rectangular area. Mirrors cubiomes' genBiomes
 * — far faster than per-pixel getBiomeAt for tile rendering. `out` must hold
 * at least sx*sz int32s. Returns 0 on success. y/sy = 0 → 2D plane. */
int mcsf_gen_biomes(
    Generator *g,
    int *out,
    int scale,
    int x, int z,
    int sx, int sz,
    int y
) {
    Range r = { scale, x, z, sx, sz, y, 0 };
    return genBiomes(g, out, r);
}

/* Fill `out` (must be 256*3 bytes) with the cubiomes biome RGB colormap. */
void mcsf_init_biome_colors(unsigned char *out) {
    /* cubiomes initBiomeColors expects unsigned char[256][3]. */
    initBiomeColors((unsigned char (*)[3]) out);
}

/* The buffer cubiomes needs for genBiomes can be larger than sx*sy*sz at
 * certain (scale, dimension) combinations — cubiomes reuses it for layered
 * scratch storage. Always allocate this many int32s before calling
 * mcsf_gen_biomes, or you'll get heap corruption (malloc: corrupted top size).
 * Pass sy=0 for a 2D plane. */
size_t mcsf_min_cache_size(
    const Generator *g, int scale, int sx, int sy, int sz
) {
    return getMinCacheSize(g, scale, sx, sy, sz);
}

/* --- Phase 6c: Perlin noise state extraction & CPU reference sample ----- */

/* Initialise a cubiomes PerlinNoise from `seed` (Java-RNG init path), copy
 * its state out for upload to the GPU. The 256-byte permutation table is
 * cubiomes' shuffled byte array; cubiomes' implementation also stores a
 * duplicated 257th byte but it equals byte 0, so we only export 256 and
 * the WGSL kernel wraps with `& 0xFF` to get the same effect. */
void mcsf_perlin_init(
    uint64_t seed,
    double *a_out, double *b_out, double *c_out,
    uint8_t *h2_out, double *d2_out, double *t2_out,
    uint8_t *perm_out_256
) {
    PerlinNoise p;
    uint64_t s = seed;
    perlinInit(&p, &s);
    *a_out = p.a;
    *b_out = p.b;
    *c_out = p.c;
    *h2_out = p.h2;
    *d2_out = p.d2;
    *t2_out = p.t2;
    memcpy(perm_out_256, p.d, 256);
}

/* CPU reference sample. Rebuilds a PerlinNoise from the extracted state and
 * calls cubiomes' samplePerlin — used by the GPU↔CPU parity test. */
double mcsf_perlin_sample(
    double a, double b, double c,
    uint8_t h2, double d2, double t2,
    const uint8_t *perm_256,
    double x, double y, double z,
    double yamp, double ymin
) {
    PerlinNoise p;
    p.a = a;
    p.b = b;
    p.c = c;
    p.h2 = h2;
    p.d2 = d2;
    p.t2 = t2;
    p.amplitude = 1.0;
    p.lacunarity = 1.0;
    memcpy(p.d, perm_256, 256);
    p.d[256] = p.d[0];
    return samplePerlin(&p, x, y, z, yamp, ymin);
}

/* --- Phase 6c-2: Double-Perlin reference sample ------------------------- */
/* Reconstruct cubiomes' DoublePerlinNoise from packed per-octave arrays and
 * sample at (x, y, z). This is the *exact* cubiomes oracle — the GPU result
 * must match it within f32 tolerance. */
double mcsf_double_perlin_sample(
    int oct_a_count,
    int oct_b_count,
    double dp_amplitude,
    const double *octave_abc,        /* 3 doubles per octave: a, b, c */
    const uint8_t *octave_h2,        /* 1 byte per octave */
    const double *octave_d2_t2,      /* 2 doubles per octave: d2, t2 */
    const double *octave_amp_lac,    /* 2 doubles per octave: amplitude, lac */
    const uint8_t *octave_perm,      /* 256 bytes per octave */
    double x, double y, double z
) {
    int total = oct_a_count + oct_b_count;
    if (total <= 0) return 0.0;
    PerlinNoise *pns = (PerlinNoise *) malloc(sizeof(PerlinNoise) * (size_t) total);
    if (!pns) return 0.0;
    for (int i = 0; i < total; i++) {
        pns[i].a = octave_abc[i*3 + 0];
        pns[i].b = octave_abc[i*3 + 1];
        pns[i].c = octave_abc[i*3 + 2];
        pns[i].h2 = octave_h2[i];
        pns[i].d2 = octave_d2_t2[i*2 + 0];
        pns[i].t2 = octave_d2_t2[i*2 + 1];
        pns[i].amplitude = octave_amp_lac[i*2 + 0];
        pns[i].lacunarity = octave_amp_lac[i*2 + 1];
        memcpy(pns[i].d, octave_perm + i*256, 256);
        pns[i].d[256] = pns[i].d[0];
    }
    DoublePerlinNoise dp;
    dp.amplitude = dp_amplitude;
    dp.octA.octcnt = oct_a_count;
    dp.octA.octaves = pns;
    dp.octB.octcnt = oct_b_count;
    dp.octB.octaves = pns + oct_a_count;
    double v = sampleDoublePerlin(&dp, x, y, z);
    free(pns);
    return v;
}

/* --- Phase 6c-3: Climate noise init for MC 1.18+ ------------------------ */
/* Field indices match cubiomes' NP_* enum (0=temp, 1=humidity, 2=cont,
 * 3=erosion, 4=shift, 5=weirdness). Builds a BiomeNoise via setBiomeSeed,
 * then walks the requested field's DoublePerlinNoise and copies each octave
 * (octA followed by octB) into the caller's flat arrays.
 *
 * Caller must allocate output capacity for at least 18 octaves per field
 * (cubiomes' continentalness uses 9 + 9 = 18 max; others are smaller).
 * Returns the total octave count (oct_a_count + oct_b_count), or 0 on error.
 */
int mcsf_climate_init_field(
    int mc,
    uint64_t seed,
    int large,
    int field_idx,
    /* outputs: */
    int *out_oct_a_count,
    int *out_oct_b_count,
    double *out_dp_amplitude,
    double *out_abc,        /* 3 doubles per octave */
    uint8_t *out_h2,        /* 1 byte per octave */
    double *out_d2_t2,      /* 2 doubles per octave */
    double *out_amp_lac,    /* 2 doubles per octave */
    uint8_t *out_perm       /* 256 bytes per octave */
) {
    if (field_idx < 0 || field_idx >= NP_MAX) return 0;
    BiomeNoise bn;
    memset(&bn, 0, sizeof(bn));
    initBiomeNoise(&bn, mc);
    setBiomeSeed(&bn, seed, large);

    const DoublePerlinNoise *dpn = &bn.climate[field_idx];
    int a_count = dpn->octA.octcnt;
    int b_count = dpn->octB.octcnt;
    *out_oct_a_count = a_count;
    *out_oct_b_count = b_count;
    *out_dp_amplitude = dpn->amplitude;

    int idx = 0;
    for (int half = 0; half < 2; half++) {
        const PerlinNoise *octs = (half == 0) ? dpn->octA.octaves : dpn->octB.octaves;
        int n = (half == 0) ? a_count : b_count;
        for (int i = 0; i < n; i++) {
            const PerlinNoise *p = &octs[i];
            out_abc[idx*3 + 0] = p->a;
            out_abc[idx*3 + 1] = p->b;
            out_abc[idx*3 + 2] = p->c;
            out_h2[idx] = p->h2;
            out_d2_t2[idx*2 + 0] = p->d2;
            out_d2_t2[idx*2 + 1] = p->t2;
            out_amp_lac[idx*2 + 0] = p->amplitude;
            out_amp_lac[idx*2 + 1] = p->lacunarity;
            memcpy(out_perm + idx*256, p->d, 256);
            idx++;
        }
    }
    return a_count + b_count;
}

/* CPU reference: sample one climate field's underlying DoublePerlinNoise
 * directly. NOTE: cubiomes aliases NP_SHIFT == NP_DEPTH (both = 4), and
 * `sampleClimatePara` interprets nptype==NP_DEPTH as the composite
 * continentalness/erosion/weirdness spline — NOT the underlying shift
 * noise. The GPU computes the raw double-perlin, so the CPU reference must
 * also use the raw double-perlin path for parity. This bypasses
 * sampleClimatePara entirely and uses cubiomes' setBiomeSeed +
 * sampleDoublePerlin on the requested field. */
double mcsf_climate_sample_field(
    int mc, uint64_t seed, int large, int field_idx,
    double x, double z
) {
    if (field_idx < 0 || field_idx >= NP_MAX) return 0.0;
    BiomeNoise bn;
    memset(&bn, 0, sizeof(bn));
    initBiomeNoise(&bn, mc);
    setBiomeSeed(&bn, seed, large);
    return sampleDoublePerlin(&bn.climate[field_idx], x, 0, z);
}

/* --- Phase 6c-4: Biome b-tree data + reference walker ------------------- */
/* We include btree21wd.h here to get a second copy of the data tables (the
 * originals are static const inside biomenoise.c). The data is ~22KB and
 * read-only, so the duplication is negligible. */
#include "tables/btree21wd.h"

int mcsf_btree21wd_order(void) {
    return btree21wd_order;
}

/* Each accessor returns the array length. If `out` is non-NULL and
 * `cap >= length`, copies data into `out`. */
size_t mcsf_btree21wd_steps_len(void) {
    return sizeof(btree21wd_steps) / sizeof(*btree21wd_steps);
}
void mcsf_btree21wd_steps_copy(uint32_t *out) {
    memcpy(out, btree21wd_steps, sizeof(btree21wd_steps));
}

/* btree21wd_param is int32_t [][2] — total int32 count = 2 * #pairs. */
size_t mcsf_btree21wd_param_len_i32(void) {
    return sizeof(btree21wd_param) / sizeof(int32_t);
}
void mcsf_btree21wd_param_copy(int32_t *out) {
    memcpy(out, btree21wd_param, sizeof(btree21wd_param));
}

size_t mcsf_btree21wd_nodes_len(void) {
    return sizeof(btree21wd_nodes) / sizeof(uint64_t);
}
void mcsf_btree21wd_nodes_copy(uint64_t *out) {
    memcpy(out, btree21wd_nodes, sizeof(btree21wd_nodes));
}

/* Reference biome lookup: cubiomes' real `climateToBiome` for the given mc
 * version. Used by the parity test as the ground truth. */
int mcsf_climate_to_biome(int mc, const uint64_t *np6) {
    return climateToBiome(mc, np6, NULL);
}

/* --- World spawn point (cubiomes getSpawn) ----------------------------- */
/* Computes the canonical (x, z) of the world spawn for the seed applied
 * to `g`. Only meaningful in the Overworld; cubiomes' Nether/End paths
 * have their own spawn semantics. */
void mcsf_get_spawn(const Generator *g, int *out_x, int *out_z) {
    Pos p = getSpawn(g);
    *out_x = p.x;
    *out_z = p.z;
}

/* --- Climate np[6] at a single coord (for the debug overlay) ----------- */
/* Returns the six raw climate values cubiomes computes at (x, y, z) — the
 * same values fed to the b-tree to pick a biome. Useful for power-user
 * debugging. Values are i64 (cubiomes' internal canonical form: the i64
 * truncation of `10000 * f32_climate`). */
int mcsf_climate_np(
    Generator *g,
    int x, int y, int z,
    int64_t *out_np6
) {
    return sampleBiomeNoise(&g->bn, out_np6, x, y, z, NULL, 0);
}

/* Full biome sample at (x, z) — runs cubiomes' setBiomeSeed +
 * sampleBiomeNoise (which does shift+climate+depth-spline+b-tree). Also
 * writes the np[6] climate values to out_np6 (length 6 int64s). */
int mcsf_sample_biome_at(
    int mc, uint64_t seed, int large,
    int x, int y, int z,
    int64_t *out_np6
) {
    BiomeNoise bn;
    memset(&bn, 0, sizeof(bn));
    initBiomeNoise(&bn, mc);
    setBiomeSeed(&bn, seed, large);
    return sampleBiomeNoise(&bn, out_np6, x, y, z, NULL, 0);
}

/* --- Phase 6c-5: depth-spline pre-computation -------------------------- */
/* `getSpline` is exported from biomenoise.c but has no prototype in any
 * cubiomes public header. Forward-declare it here so we can call it. */
extern float getSpline(const Spline *sp, const float *vals);

/* The spline tree in BiomeNoise.sp is seed-independent (built by
 * initBiomeNoise from the MC version). This computes the `d` (depth)
 * climate value the same way `sampleBiomeNoise` does internally, given
 * the three already-sampled climate values c, e, w. Pulling the spline
 * out of the inner loop lets the GPU pipeline batch-sample c/e/w then
 * compute depth on the CPU once per pixel — the spline is f32 in
 * cubiomes and is small per-pixel work compared to the noise stack. */
double mcsf_compute_depth(int mc, double c, double e, double w, int y) {
    BiomeNoise bn;
    memset(&bn, 0, sizeof(bn));
    initBiomeNoise(&bn, mc);
    float w_t = -3.0F * (fabsf(fabsf((float)w) - 0.6666667F) - 0.33333334F);
    float np_param[4] = {(float)c, (float)e, w_t, (float)w};
    double off = getSpline(bn.sp, np_param) + 0.015F;
    return 1.0 - (y * 4) / 128.0 - 83.0 / 160.0 + off;
}

/* --- Phase D1: depth-spline flattening for GPU per-seed search --------- */
/* For per-seed GPU search the depth value (np[NP_DEPTH]) must be computed on
 * the GPU, so the seed-independent spline tree is flattened here and uploaded
 * once. Post-order DFS with pointer dedup guarantees every child has a LOWER
 * flat index than its parent, so the WGSL walker evaluates nodes forward
 * (0..count) with no recursion/stack and the root is the last node.
 *
 * Capacity: SplineStack holds stack[42] + fstack[151] = 193 nodes max; callers
 * pass 256-slot buffers. A leaf (FixSpline, len==1) emits typ=-1 + out_fix;
 * an internal node emits typ + len + loc/der/child triples (<=12). Returns the
 * node count (root index = count-1), or -1 on overflow. */
#define MCSF_SPLINE_CAP 256

static int mcsf_spline_flatten_rec(
    const Spline *sp, const void **seen, int *count,
    int *o_typ, int *o_len, float *o_fix, float *o_loc, float *o_der, int *o_child
) {
    for (int k = 0; k < *count; k++)
        if (seen[k] == (const void *)sp)
            return k;

    int len = sp->len;
    int child_idx[12];
    if (len != 1) {
        for (int i = 0; i < len; i++) {
            child_idx[i] = mcsf_spline_flatten_rec(
                sp->val[i], seen, count, o_typ, o_len, o_fix, o_loc, o_der, o_child);
            if (child_idx[i] < 0)
                return -1;
        }
    }

    if (*count >= MCSF_SPLINE_CAP)
        return -1;
    int my = (*count)++;
    seen[my] = (const void *)sp;
    if (len == 1) {
        o_typ[my] = -1;
        o_len[my] = 1;
        o_fix[my] = ((const FixSpline *)sp)->val;
    } else {
        o_typ[my] = sp->typ;
        o_len[my] = len;
        for (int i = 0; i < len; i++) {
            o_loc[my * 12 + i] = sp->loc[i];
            o_der[my * 12 + i] = sp->der[i];
            o_child[my * 12 + i] = child_idx[i];
        }
    }
    return my;
}

int mcsf_depth_spline_flatten(
    int mc, int *out_typ, int *out_len, float *out_fix,
    float *out_loc, float *out_der, int *out_child
) {
    BiomeNoise bn;
    memset(&bn, 0, sizeof(bn));
    initBiomeNoise(&bn, mc);
    const void *seen[MCSF_SPLINE_CAP];
    int count = 0;
    int root = mcsf_spline_flatten_rec(
        bn.sp, seen, &count, out_typ, out_len, out_fix, out_loc, out_der, out_child);
    if (root < 0)
        return -1;
    /* Post-order guarantees the root is the final node. */
    return count;
}

/* --- Approximate surface heightmap (cubiomes mapApproxHeight) ---------- */
/* Important: NOT bit-exact Java terrain. cubiomes itself names this
 * `mapApproxHeight` because it's derived from the depth-spline output
 * (np[NP_DEPTH] / 76.0 in 1.18+) — fast and deterministic, but post-
 * climate-noise, not the per-block Java surface column. The shim exposes
 * it for use ONLY as visual relief, never as exact altitude. Every
 * downstream caller must label its output "approximate". */

/* Single-point variant. Coords (x, z) are in cubiomes scale-4 units
 * (i.e. block_x >> 2). Returns the surface height in (fractional) blocks
 * via *out_y; biome at that surface point goes to *out_biome (NULL OK).
 * Returns the cubiomes mapApproxHeight return code, or non-zero on error. */
int mcsf_surface_height_at(
    Generator *g, int x, int z,
    float *out_y, int *out_biome
) {
    SurfaceNoise sn;
    initSurfaceNoise(&sn, g->dim, g->seed);
    float y = 0.0f;
    int id = 0;
    int rc = mapApproxHeight(&y, &id, g, &sn, x, z, 1, 1);
    if (rc != 0) return rc;
    *out_y = y;
    if (out_biome) *out_biome = id;
    return 0;
}

/* Batch variant: fills out_y (w*h floats) and out_ids (w*h int32s) for a
 * rectangle starting at (x, z) in scale-4 units. Output stride matches
 * mapApproxHeight: y[j*w+i]. */
int mcsf_surface_height_map(
    Generator *g,
    int x, int z, int w, int h,
    float *out_y, int *out_ids
) {
    SurfaceNoise sn;
    initSurfaceNoise(&sn, g->dim, g->seed);
    return mapApproxHeight(out_y, out_ids, g, &sn, x, z, w, h);
}
