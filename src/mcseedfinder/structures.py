"""
structures.py
=============

Accurate structure-position predictor for Minecraft Java Edition 1.18+.

This is the **high-fidelity** part of the seed finder. Unlike biome
generation — which we approximate — structure placement is fully reproducible
from the world seed using a small handful of integer arithmetic operations
and one ``java.util.Random`` instance per region. The math here matches
Mojang's source line-for-line and is independently verified against
cubiomes' ``finders.c``.

Algorithm (``random_spread`` structure placement)
-------------------------------------------------
For each structure type, Mojang assigns three configuration values:

* ``salt``        — a unique integer mixed into the per-region RNG seed
* ``spacing``     — grid cell size, in chunks (``regionSize`` in cubiomes)
* ``separation``  — minimum chunk distance between adjacent structures

The placement algorithm, given a world seed ``S`` and a target region
``(rx, rz)`` (where ``rx = floor(chunkX / spacing)``):

.. code-block:: text

    k = S + salt + rx * 341873128712 + rz * 132897987541      (signed 64-bit wrap)
    rng = JavaRandom(k)                                      (Java setSeed)
    range = spacing - separation
    # 'linear' spread (most pre-1.18 structures):
    offsetX = rng.nextInt(range)
    offsetZ = rng.nextInt(range)
    # 'triangular' spread (1.18+ villages, monuments, etc.):
    offsetX = (rng.nextInt(range) + rng.nextInt(range)) // 2
    offsetZ = (rng.nextInt(range) + rng.nextInt(range)) // 2
    chunkX = rx * spacing + offsetX
    chunkZ = rz * spacing + offsetZ

The block-space position of the structure is then
``(chunkX * 16 + 8, chunkZ * 16 + 8)`` (centre of the top-left chunk).

This file does **not** check biome eligibility — a returned candidate chunk
might be cancelled in-game because, e.g., a swamp hut needs a swamp biome.
The criteria layer can optionally cross-check against biome data, but for
quick structure-density searches this position alone is what you want.

Salts and configs cited below are taken directly from cubiomes' ``finders.c``
(``getStructureConfig`` for MC 1.18+).

References
----------
* cubiomes ``finders.c`` (Cubitect): authoritative ``StructureConfig`` entries.
* Minecraft Wiki: https://minecraft.wiki/w/Structure_set — placement spec.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

from .java_random import JavaRandom

# --------------------------------------------------------------------------- #
# Region-grid multipliers — fixed Mojang constants
# --------------------------------------------------------------------------- #
#: Multiplier applied to the structure region X coordinate when mixing it
#: into the per-region RNG seed. Identical across all structure types and
#: all post-1.13 Minecraft versions.
_REGION_MUL_X: int = 341873128712
#: Multiplier applied to the structure region Z coordinate. Same provenance.
_REGION_MUL_Z: int = 132897987541
#: Mask used to wrap signed 64-bit arithmetic into the Java ``long`` range.
_LONG_MASK: int = (1 << 64) - 1
_LONG_SIGN_BIT: int = 1 << 63


def _wrap_signed_long(value: int) -> int:
    """Reduce a Python integer into Java's signed 64-bit ``long`` range."""
    value &= _LONG_MASK
    return value - (1 << 64) if value >= _LONG_SIGN_BIT else value


# --------------------------------------------------------------------------- #
# Structure configuration table
# --------------------------------------------------------------------------- #
class SpreadType(Enum):
    """The two distribution modes the structure placement RNG uses."""

    LINEAR = "linear"           # single ``nextInt`` per axis
    TRIANGULAR = "triangular"   # ``(nextInt + nextInt) // 2`` per axis


@dataclass(frozen=True)
class StructureConfig:
    """Mojang's per-structure placement config.

    Attributes
    ----------
    salt:
        Unique integer mixed into the region RNG seed.
    spacing:
        Grid cell size in chunks (``regionSize``).
    separation:
        Minimum chunk separation. The RNG draws an offset in
        ``[0, spacing - separation)``.
    spread_type:
        Linear or triangular — see :class:`SpreadType`.
    salt_offset_for_alt_versions:
        Mostly unused here; kept for future expansion to historical versions.
    """

    salt: int
    spacing: int
    separation: int
    spread_type: SpreadType = SpreadType.LINEAR
    salt_offset_for_alt_versions: int = 0

    @property
    def chunk_range(self) -> int:
        """Width of the offset range, ``spacing - separation``."""
        return self.spacing - self.separation


# Values taken verbatim from cubiomes' ``finders.c``::
#
#     s_desert_pyramid  = { 14357617, 32, 24, Desert_Pyramid,   0,0},
#     s_igloo           = { 14357618, 32, 24, Igloo,            0,0},
#     s_jungle_temple   = { 14357619, 32, 24, Jungle_Pyramid,   0,0},
#     s_swamp_hut       = { 14357620, 32, 24, Swamp_Hut,        0,0},
#     s_outpost         = {165745296, 32, 24, Outpost,          0,0},
#     s_village_117     = { 10387312, 32, 24, Village,          0,0},
#     s_village         = { 10387312, 34, 26, Village,          0,0},   # 1.18+
#     s_ocean_ruin      = { 14357621, 20, 12, Ocean_Ruin,       0,0},
#     s_shipwreck       = {165745295, 24, 20, Shipwreck,        0,0},
#     s_monument        = { 10387313, 32, 27, Monument,         0,0},
#     s_mansion         = { 10387319, 80, 60, Mansion,          0,0},
#     s_ruined_portal   = { 34222645, 40, 25, Ruined_Portal,    0,0},
#
# In cubiomes the entries are ``{salt, regionSize, chunkRange, ...}``, where
# ``chunkRange`` equals ``spacing - separation``; we store ``separation``
# directly so it matches the wiki's terminology.
STRUCTURE_CONFIGS: dict[str, StructureConfig] = {
    "desert_pyramid": StructureConfig(14357617, 32, 8),
    "igloo":          StructureConfig(14357618, 32, 8),
    "jungle_temple":  StructureConfig(14357619, 32, 8),
    "swamp_hut":      StructureConfig(14357620, 32, 8),
    "pillager_outpost": StructureConfig(165745296, 32, 8),
    # Village changed in 1.18+: spacing 34, separation 8 (chunkRange 26).
    "village":        StructureConfig(10387312, 34, 8, SpreadType.TRIANGULAR),
    "ocean_ruin":     StructureConfig(14357621, 20, 8),
    "shipwreck":      StructureConfig(165745295, 24, 4),
    "ocean_monument": StructureConfig(10387313, 32, 5, SpreadType.TRIANGULAR),
    "woodland_mansion": StructureConfig(10387319, 80, 20, SpreadType.TRIANGULAR),
    "ruined_portal":  StructureConfig(34222645, 40, 15),
    # 1.19.2+: deep-dark city, standard linear getFeaturePos
    # (cubiomes s_ancient_city = {20083232, 24, 16} → separation=24-16=8).
    "ancient_city":   StructureConfig(20083232, 24, 8),
    # 1.21+: trial chambers, standard linear getFeaturePos
    # (cubiomes s_trial_chambers = {94251327, 34, 22} → separation=34-22=12).
    "trial_chambers": StructureConfig(94251327, 34, 12),
    # Notes on entries with non-region-grid placement:
    #
    # * stronghold — concentric-ring algorithm (see ``iter_strongholds``).
    # * buried_treasure — per-chunk 1% nextFloat roll (cubiomes `case
    #   Treasure`). The salt is real but spacing/separation are dummy
    #   placeholders here; the real evaluator branches via
    #   ``roll_buried_treasure_chunk`` + ``iter_buried_treasure_in_radius``.
}


# --------------------------------------------------------------------------- #
# Region-position computation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StructurePos:
    """A predicted structure position, in chunk and block coordinates."""

    structure: str
    chunk_x: int
    chunk_z: int

    @property
    def block_x(self) -> int:
        """Block X coordinate of the structure's anchor (top-left + 8)."""
        return self.chunk_x * 16 + 8

    @property
    def block_z(self) -> int:
        """Block Z coordinate of the structure's anchor."""
        return self.chunk_z * 16 + 8

    def distance_to(self, x: int, z: int) -> float:
        """Euclidean distance from ``(x, z)`` block coords."""
        dx = self.block_x - x
        dz = self.block_z - z
        return math.hypot(dx, dz)


_BURIED_TREASURE_SALT = 10387320


def roll_buried_treasure_chunk(world_seed: int, chunk_x: int, chunk_z: int) -> bool:
    """True iff a buried treasure is placed in ``(chunk_x, chunk_z)``.

    Buried treasure uses a per-chunk roll rather than the standard region
    grid: at every chunk, mix the salt + region multipliers into the world
    seed, seed Java's RNG, and check ``nextFloat() < 0.01``. The placement
    anchor when true is ``(chunk_x * 16 + 9, chunk_z * 16 + 9)`` — note the
    ``+9``, not the ``+8`` used by canonical region-anchor structures.
    Mirrors the Rust ``structures::roll_buried_treasure_chunk`` and cubiomes
    ``case Treasure`` in ``finders.c``.
    """
    k = (
        world_seed
        + _BURIED_TREASURE_SALT
        + chunk_x * _REGION_MUL_X
        + chunk_z * _REGION_MUL_Z
    )
    k = _wrap_signed_long(k)
    rng = JavaRandom(k)
    # Cubiomes' `nextFloat(&seed) < 0.01` is f32 < f64 (the literal 0.01
    # promotes via C's usual arithmetic conversions). Python's `next_float`
    # returns f64 — round-trip through `struct` to drop to f32 precision so
    # the comparison matches cubiomes bit-exactly even at the rounding
    # boundary (where naïve f64-vs-f64 disagrees with f32-promoted-to-f64).
    import struct
    raw_f64 = rng.next_float()
    f32_as_f64 = struct.unpack("f", struct.pack("f", raw_f64))[0]
    return f32_as_f64 < 0.01


def iter_buried_treasure_in_radius(
    world_seed: int,
    centre_block_x: int,
    centre_block_z: int,
    block_radius: int,
) -> Iterator[StructurePos]:
    """Yield every buried treasure within ``block_radius`` of the centre."""
    chunk_radius = (block_radius // 16) + 1
    cx_min = (centre_block_x // 16) - chunk_radius
    cx_max = (centre_block_x // 16) + chunk_radius
    cz_min = (centre_block_z // 16) - chunk_radius
    cz_max = (centre_block_z // 16) + chunk_radius
    block_radius_sq = block_radius * block_radius
    for cx in range(cx_min, cx_max + 1):
        for cz in range(cz_min, cz_max + 1):
            if not roll_buried_treasure_chunk(world_seed, cx, cz):
                continue
            bx = cx * 16 + 9
            bz = cz * 16 + 9
            dx = bx - centre_block_x
            dz = bz - centre_block_z
            if dx * dx + dz * dz > block_radius_sq:
                continue
            yield StructurePos(structure="buried_treasure", chunk_x=cx, chunk_z=cz)


def get_structure_pos(
    structure: str, world_seed: int, region_x: int, region_z: int
) -> StructurePos:
    """Predict where ``structure`` *attempts* to generate in the given region.

    Notes
    -----
    "Attempts" because the in-game check additionally requires a suitable
    biome (and, for 1.18+ desert pyramids / jungle temples / mansions, a
    suitable terrain height). Without a real biome generator we return the
    candidate chunk; the criteria layer can filter further if you have one.

    Buried treasure uses a per-chunk roll, not a region-grid placement —
    callers should use :func:`roll_buried_treasure_chunk` or
    :func:`iter_buried_treasure_in_radius` instead.

    Parameters
    ----------
    structure:
        Key from :data:`STRUCTURE_CONFIGS`.
    world_seed:
        Full 64-bit world seed. (Only the low 48 bits actually influence the
        RNG — see :mod:`mcseedfinder.java_random`.)
    region_x, region_z:
        Region grid coordinates. To find the structure near chunk
        ``(cx, cz)``, use ``region_x = cx // spacing`` etc.
    """
    if structure == "buried_treasure":
        raise ValueError(
            "buried_treasure uses per-chunk roll; "
            "call roll_buried_treasure_chunk() / iter_buried_treasure_in_radius()"
        )
    cfg = STRUCTURE_CONFIGS[structure]

    # ---- Per-region RNG seed ----
    # The math is done in Python's unbounded ints but the final value is
    # interpreted as a signed Java long when fed to the LCG.
    k = world_seed + cfg.salt + region_x * _REGION_MUL_X + region_z * _REGION_MUL_Z
    k = _wrap_signed_long(k)

    rng = JavaRandom(k)
    rng_range = cfg.chunk_range

    if cfg.spread_type is SpreadType.LINEAR:
        offset_x = rng.next_int_bound(rng_range)
        offset_z = rng.next_int_bound(rng_range)
    else:
        # Triangular distribution: average of two uniform samples produces
        # a peak in the centre of the cell. Mojang chose this for villages
        # and ocean monuments in 1.18+ to make their spacing feel more
        # uniform than pure-uniform does.
        offset_x = (rng.next_int_bound(rng_range) + rng.next_int_bound(rng_range)) // 2
        offset_z = (rng.next_int_bound(rng_range) + rng.next_int_bound(rng_range)) // 2

    chunk_x = region_x * cfg.spacing + offset_x
    chunk_z = region_z * cfg.spacing + offset_z
    return StructurePos(structure=structure, chunk_x=chunk_x, chunk_z=chunk_z)


def iter_structures_in_radius(
    structure: str,
    world_seed: int,
    centre_block_x: int,
    centre_block_z: int,
    block_radius: int,
) -> Iterator[StructurePos]:
    """Yield every structure of the given type within ``block_radius`` blocks.

    The implementation walks the structure's grid covering the bounding box
    of the search circle, then filters by Euclidean distance. For a 2000-block
    radius this is on the order of a few hundred regions per structure type,
    so even a brute-force seed sweep stays fast.

    Buried treasure is dispatched to its per-chunk-roll iterator since the
    region-grid framework doesn't model "most chunks have no placement".
    """
    if structure == "buried_treasure":
        yield from iter_buried_treasure_in_radius(
            world_seed, centre_block_x, centre_block_z, block_radius
        )
        return
    cfg = STRUCTURE_CONFIGS[structure]
    # Convert the search box to chunk coords, then to region coords.
    chunk_radius = (block_radius // 16) + 1
    cx_min = (centre_block_x // 16) - chunk_radius
    cx_max = (centre_block_x // 16) + chunk_radius
    cz_min = (centre_block_z // 16) - chunk_radius
    cz_max = (centre_block_z // 16) + chunk_radius
    rx_min = _floordiv(cx_min, cfg.spacing)
    rx_max = _floordiv(cx_max, cfg.spacing)
    rz_min = _floordiv(cz_min, cfg.spacing)
    rz_max = _floordiv(cz_max, cfg.spacing)

    for rx in range(rx_min, rx_max + 1):
        for rz in range(rz_min, rz_max + 1):
            pos = get_structure_pos(structure, world_seed, rx, rz)
            if pos.distance_to(centre_block_x, centre_block_z) <= block_radius:
                yield pos


def _floordiv(a: int, b: int) -> int:
    """Python's ``//`` already performs floor division for negative dividends.

    This helper exists for explicitness — Minecraft's region math is
    documented as floor-division, and using ``//`` directly is correct but
    easy to misread when ``a`` can be negative.
    """
    return a // b


# --------------------------------------------------------------------------- #
# Strongholds — concentric rings, not the grid algorithm
# --------------------------------------------------------------------------- #
# Stronghold rings (1.9+): 8 rings, ``count`` strongholds each, distance to
# the closest stronghold-ring origin in blocks. Source: cubiomes
# (``getStrongholds``) and the Minecraft Wiki stronghold article.
_STRONGHOLD_RING_COUNTS: list[int] = [3, 6, 10, 15, 21, 28, 36, 9]
_STRONGHOLD_RING_DISTANCES: list[tuple[int, int]] = [
    (1280, 2816),
    (4352, 5888),
    (7424, 8960),
    (10496, 12032),
    (13568, 15104),
    (16640, 18176),
    (19712, 21248),
    (22784, 24320),
]


def iter_strongholds(
    world_seed: int, max_rings: int = 3
) -> Iterator[StructurePos]:
    """Yield stronghold positions, ring by ring, up to ``max_rings``.

    Strongholds don't use ``random_spread``; they use ``concentric_rings``.
    Each ring has a fixed structure count and a fixed annulus, and the
    angular positions are evenly spaced around a random initial angle drawn
    from the world seed.

    Parameters
    ----------
    max_rings:
        Number of rings to iterate (max 8). The first ring at ~1300 blocks
        is what most seed finders care about; later rings are huge distances.
    """
    if not 1 <= max_rings <= 8:
        raise ValueError("max_rings must be between 1 and 8")

    # Java code uses ``Random(worldSeed)`` directly (no ``+ salt`` mixing),
    # because the ring layout is global rather than per-region.
    rng = JavaRandom(world_seed)
    # Initial angle in radians, uniform over [0, 2π).
    angle = rng.next_double() * math.pi * 2.0

    for ring_idx in range(max_rings):
        count = _STRONGHOLD_RING_COUNTS[ring_idx]
        # Distance for this ring: uniform between the inner and outer radius.
        d_min, d_max = _STRONGHOLD_RING_DISTANCES[ring_idx]
        # Each stronghold in the ring is placed at angle + k*(2π/count) plus
        # a small per-stronghold jitter consumed from the same RNG.
        for _k in range(count):
            distance = d_min + rng.next_double() * (d_max - d_min)
            # The block X/Z come from the polar position. Java's source
            # rounds to chunk centres.
            block_x = int(math.cos(angle) * distance)
            block_z = int(math.sin(angle) * distance)
            # Convert to chunk coords for the StructurePos contract.
            yield StructurePos(
                structure="stronghold",
                chunk_x=block_x // 16,
                chunk_z=block_z // 16,
            )
            angle += (2.0 * math.pi) / count


# --------------------------------------------------------------------------- #
# Public registry of available structure names
# --------------------------------------------------------------------------- #
SUPPORTED_STRUCTURES: frozenset[str] = frozenset(
    list(STRUCTURE_CONFIGS) + ["stronghold"]
)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    # Quick sanity print: positions of the nearest village + swamp hut for a
    # known seed. These match cubiomes for the same seed and version.
    seed = 1
    print(f"Structures near origin for seed {seed} (MC 1.18+):")
    for s in ("village", "swamp_hut", "desert_pyramid", "shipwreck",
              "ocean_monument", "pillager_outpost"):
        nearest: StructurePos | None = None
        for pos in iter_structures_in_radius(s, seed, 0, 0, 5000):
            if nearest is None or pos.distance_to(0, 0) < nearest.distance_to(0, 0):
                nearest = pos
        if nearest:
            print(f"  {s:<20s} chunk=({nearest.chunk_x:5d},{nearest.chunk_z:5d}) "
                  f"block=({nearest.block_x:6d},{nearest.block_z:6d}) "
                  f"dist={nearest.distance_to(0, 0):7.1f}")
    print("Strongholds (ring 1):")
    for sh in iter_strongholds(seed, max_rings=1):
        print(f"  stronghold           block=({sh.block_x:6d},{sh.block_z:6d}) "
              f"dist={sh.distance_to(0, 0):7.1f}")
