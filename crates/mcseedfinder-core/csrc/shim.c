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

#include "generator.h"
#include "layers.h"
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
