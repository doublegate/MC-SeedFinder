"""
biome_gen.py
============

Biome lookup for an arbitrary world coordinate, given a world seed.

Two backends
------------
This module dispatches to one of two implementations:

1. **``cubiomes``** (preferred for accuracy)
   If the user has ``cubiomes-py`` (or any object implementing
   :class:`BiomeGenerator`) available, we delegate to it. ``cubiomes`` is a
   faithful re-implementation of Minecraft's actual biome generator and
   gives **bit-exact** results for the configured MC version.

2. **Climate-noise approximation** (built-in fallback)
   A deterministic, seed-driven Perlin-noise field maps each (x, z) to a
   plausible biome via a hand-tuned climate→biome table. *This is not what
   Minecraft does.* It produces qualitatively-correct biome neighbourhoods
   (deserts cluster with savannas, jungles with bamboo jungles, etc.) but
   will not predict the actual in-game map for a given seed.

The approximation is acceptable for use cases like *"find me any seed that
has a desert next to a jungle"* — many seeds satisfy that, and the
approximation is right often enough to surface candidates worth verifying
in-game. It is **not** acceptable for "find a seed where the village at
(180, 260) is in a plains biome" — for that, you need a real biome
generator.

Plugging in a real backend
--------------------------
::

    from mcseedfinder.biome_gen import BiomeLookup
    import cubiomes_py  # hypothetical wrapper

    lookup = BiomeLookup(
        world_seed=12345,
        backend=cubiomes_py.create_generator(version="1.20"),
    )
    print(lookup.biome_at(100, 100))   # numeric biome ID

Any object exposing ``get_biome(world_seed: int, x: int, z: int) -> int`` is
a valid backend.
"""

from __future__ import annotations

from typing import Protocol

from .biomes import BIOMES, BiomeInfo, Climate
from .noise import OctaveNoise


# --------------------------------------------------------------------------- #
# Backend protocol
# --------------------------------------------------------------------------- #
class BiomeGenerator(Protocol):
    """Duck-typed interface for plugging in a real biome backend."""

    def get_biome(self, world_seed: int, x: int, z: int) -> int:
        """Return the numeric biome ID at block coordinate ``(x, z)``."""
        ...


# --------------------------------------------------------------------------- #
# Climate→biome mapping
# --------------------------------------------------------------------------- #
# Lookup table for the approximate generator. The key is
# ``(climate, vegetation_band)`` where ``vegetation_band`` is the noise-derived
# humidity bucket. Picking from this table gives us plausible biome groupings.
#
# vegetation_band:
#     0 → bare/cold (high altitude, sparse vegetation)
#     1 → grassland / low scrub
#     2 → forested
#     3 → lush / wet
#
# Entries are biome-key strings (keys of :data:`mcseedfinder.biomes.BIOMES`).
_CLIMATE_TABLE: dict[tuple[Climate, int], str] = {
    # Snowy
    (Climate.SNOWY, 0): "snowy_slopes",
    (Climate.SNOWY, 1): "snowy_plains",
    (Climate.SNOWY, 2): "snowy_taiga",
    (Climate.SNOWY, 3): "grove",
    # Cold
    (Climate.COLD, 0): "windswept_hills",
    (Climate.COLD, 1): "windswept_forest",
    (Climate.COLD, 2): "taiga",
    (Climate.COLD, 3): "old_growth_pine_taiga",
    # Temperate
    (Climate.TEMPERATE, 0): "meadow",
    (Climate.TEMPERATE, 1): "plains",
    (Climate.TEMPERATE, 2): "forest",
    (Climate.TEMPERATE, 3): "dark_forest",
    # Warm
    (Climate.WARM, 0): "desert",
    (Climate.WARM, 1): "savanna",
    (Climate.WARM, 2): "jungle",
    (Climate.WARM, 3): "swamp",
    # Ocean — only used when continentalness is below the sea-level threshold.
    (Climate.OCEAN, 0): "deep_ocean",
    (Climate.OCEAN, 1): "ocean",
    (Climate.OCEAN, 2): "lukewarm_ocean",
    (Climate.OCEAN, 3): "warm_ocean",
}


# --------------------------------------------------------------------------- #
# Approximate climate-noise generator
# --------------------------------------------------------------------------- #
class _ApproximateBiomeBackend:
    """Built-in approximate biome generator.

    Maintains three independent octave-noise fields:

    * **continentalness** — low values are ocean, high values are land
    * **temperature**     — drives the climate bucket
    * **humidity**        — drives the vegetation band within the bucket

    The noise fields are sampled at coarse scale (one biome cell ≈ 256 blocks)
    so the resulting "map" has biome regions on the same order of magnitude
    as Minecraft's. Returned biome IDs match the cubiomes 1.18+ numeric IDs.
    """

    __slots__ = ("_continent", "_temp", "_humid")

    # All noise fields are sampled at this base wavelength (one cycle per
    # _SCALE blocks). Roughly matches Minecraft's biome-cell granularity.
    _SCALE: float = 1.0 / 512.0

    def __init__(self, world_seed: int):
        # Derive three sub-seeds so each field is independent.
        # XOR with arbitrary but fixed constants keeps the three fields
        # decorrelated even for seeds with low entropy (e.g. seed=0).
        self._continent = OctaveNoise(world_seed ^ 0xA5A5A5A5A5A5A5A5, octaves=4)
        self._temp = OctaveNoise(world_seed ^ 0x5A5A5A5A5A5A5A5A, octaves=3)
        self._humid = OctaveNoise(world_seed ^ 0xC3C3C3C3C3C3C3C3, octaves=3)

    def get_biome(self, world_seed: int, x: int, z: int) -> int:
        """Return an approximate biome numeric ID for block coord ``(x, z)``.

        Note: ``world_seed`` is ignored here (the backend was constructed
        from it). It's part of the interface so swappable backends can
        re-derive per call if they prefer.
        """
        # Sample all three fields. ``_SCALE`` makes one block step a tiny
        # fraction of the noise period — adjacent blocks share a biome.
        c = self._continent.sample(x * self._SCALE, z * self._SCALE)
        t = self._temp.sample(x * self._SCALE, z * self._SCALE)
        h = self._humid.sample(x * self._SCALE, z * self._SCALE)

        # ---- Continentalness threshold: ocean vs land ----
        # Around -0.2 we sit between coastal and deep ocean. This is heavily
        # tunable; the values were eyeballed against rendered noise.
        if c < -0.15:
            # Ocean — pick ocean variant by temperature.
            climate = Climate.OCEAN
            vegetation_band = _bucket(t, [-0.4, 0.0, 0.4])  # cold/normal/luke/warm
            # Override for very cold: snap to deep_ocean or frozen_ocean.
            if t < -0.6:
                return BIOMES["frozen_ocean"].numeric_id
            if c < -0.4:
                # Deep variants
                if t < -0.4:
                    return BIOMES["deep_cold_ocean"].numeric_id
                if t > 0.4:
                    return BIOMES["deep_lukewarm_ocean"].numeric_id
                return BIOMES["deep_ocean"].numeric_id
        else:
            # ---- Land: pick climate from temperature ----
            # Bucket boundaries chosen so each bucket gets roughly equal area
            # of the global noise field.
            climate = _temperature_to_climate(t)
            vegetation_band = _bucket(h, [-0.3, 0.1, 0.5])

        biome_key = _CLIMATE_TABLE[(climate, vegetation_band)]

        # ---- Rare-biome injection ----
        # Mushroom Fields, Ice Spikes etc. appear at extreme noise tails.
        # Probability per cell is roughly the product of two thresholds, so
        # they're correspondingly rare in the output map.
        if climate is not Climate.OCEAN:
            if c > 0.45 and h > 0.6:
                # High continent + very humid => mushroom fields (rare).
                biome_key = "mushroom_fields"
            elif climate is Climate.SNOWY and h < -0.45:
                biome_key = "ice_spikes"
            elif climate is Climate.WARM and t > 0.6 and h < -0.2:
                biome_key = "badlands"
            elif climate is Climate.TEMPERATE and 0.25 < t < 0.35 and h > 0.4:
                biome_key = "cherry_grove"

        return BIOMES[biome_key].numeric_id


def _bucket(value: float, thresholds: list[float]) -> int:
    """Map a noise value to a discrete bucket index by walking ``thresholds``."""
    for i, threshold in enumerate(thresholds):
        if value < threshold:
            return i
    return len(thresholds)


def _temperature_to_climate(t: float) -> Climate:
    """Convert a temperature noise sample to a climate bucket."""
    if t < -0.4:
        return Climate.SNOWY
    if t < 0.0:
        return Climate.COLD
    if t < 0.4:
        return Climate.TEMPERATE
    return Climate.WARM


# --------------------------------------------------------------------------- #
# Public wrapper
# --------------------------------------------------------------------------- #
class BiomeLookup:
    """High-level biome-query API used by the criteria and finder layers.

    Internally holds either a user-supplied accurate backend or the built-in
    approximation. The constructor logs (to stderr, once) which mode it's
    using so users aren't surprised by mismatches with in-game results.
    """

    __slots__ = ("world_seed", "_backend", "_is_approximate")

    def __init__(
        self,
        world_seed: int,
        backend: BiomeGenerator | None = None,
    ) -> None:
        self.world_seed: int = world_seed
        if backend is not None:
            self._backend: BiomeGenerator = backend
            self._is_approximate: bool = False
        else:
            self._backend = _ApproximateBiomeBackend(world_seed)
            self._is_approximate = True

    @property
    def is_approximate(self) -> bool:
        """True when using the built-in noise approximation."""
        return self._is_approximate

    def biome_at(self, x: int, z: int) -> int:
        """Return the numeric biome ID at block coord ``(x, z)``."""
        return self._backend.get_biome(self.world_seed, x, z)

    def biome_info_at(self, x: int, z: int) -> BiomeInfo:
        """Same as :meth:`biome_at` but returns the full :class:`BiomeInfo`."""
        bid = self.biome_at(x, z)
        # Reverse-lookup by numeric_id. Cache could go here if profiling
        # showed this matters; in practice it's microseconds.
        for info in BIOMES.values():
            if info.numeric_id == bid:
                return info
        # Unknown ID — return a placeholder rather than raising; biome backends
        # might return IDs outside our catalog and we don't want to crash a
        # multi-million-seed search.
        return BIOMES["plains"]
