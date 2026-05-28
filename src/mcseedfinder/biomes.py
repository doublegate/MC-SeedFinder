"""
biomes.py
=========

Catalog of Minecraft biome identifiers, climate categories, and convenience
groupings used by the seed-finder's criteria DSL.

Why a catalog?
--------------
Modern Minecraft (1.18+) addresses biomes by namespaced *string* identifiers
(``minecraft:plains``, ``minecraft:cherry_grove`` etc.) rather than the legacy
numeric IDs that earlier versions used and that ``cubiomes`` exposes via its
``enum BiomeID``. Both forms are useful:

* The string form is what users actually type into a CLI or JSON criteria
  file (more readable).
* The numeric form is what a fast biome generator (or a `cubiomes-py` backend)
  returns, and what we use internally for set-membership checks.

This module exposes a single ``BIOMES`` registry mapping every supported
identifier to a :class:`BiomeInfo` record. It also provides handy groupings —
e.g. ``BIOME_GROUPS["warm_dry"]`` — so a user can write
``"spawn_biome": "warm_dry"`` and have the criteria layer expand it.

Versions
--------
The numeric IDs here match the cubiomes ``enum BiomeID`` for Minecraft Java
Edition 1.18+. Earlier-edition IDs (1.7–1.17) differ — if you need historical
compatibility, swap in the appropriate cubiomes header.

References
----------
* cubiomes ``biomes.h`` (Cubitect): authoritative numeric biome IDs.
* https://minecraft.wiki/w/Biome — namespaced identifiers and category info.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Climate(IntEnum):
    """Coarse climate category used by the approximate biome generator.

    Matches the four-way temperature bucket Minecraft itself uses in its
    legacy temperature-category system (snowy/cold/temperate/warm), plus a
    distinct ocean bucket because oceans short-circuit normal climate logic.
    """

    OCEAN = 0
    SNOWY = 1
    COLD = 2
    TEMPERATE = 3
    WARM = 4


@dataclass(frozen=True)
class BiomeInfo:
    """Static descriptor for a single biome.

    Attributes
    ----------
    numeric_id:
        Integer ID matching cubiomes' ``enum BiomeID`` for MC 1.18+.
    namespaced_id:
        The ``minecraft:<name>`` string the game itself uses.
    climate:
        Coarse climate bucket — used by our approximate biome generator.
    temperature:
        Approximate biome temperature in the ``[-0.5, 2.0]`` range Minecraft
        uses internally. Mostly informational; the generator uses
        :attr:`climate` for bucketing.
    humidity:
        Approximate biome humidity (``[0.0, 1.0]``).
    is_ocean:
        Convenience flag — true for any ocean variant.
    """

    numeric_id: int
    namespaced_id: str
    climate: Climate
    temperature: float
    humidity: float
    is_ocean: bool = False


# --------------------------------------------------------------------------- #
# Biome registry
# --------------------------------------------------------------------------- #
# Only the subset that users actually filter on. Add to this freely if you
# need finer granularity; just keep ``numeric_id`` in sync with cubiomes.
BIOMES: dict[str, BiomeInfo] = {
    # ---- Warm / dry ----
    "desert": BiomeInfo(2, "minecraft:desert", Climate.WARM, 2.0, 0.0),
    "savanna": BiomeInfo(35, "minecraft:savanna", Climate.WARM, 1.2, 0.0),
    "savanna_plateau": BiomeInfo(36, "minecraft:savanna_plateau", Climate.WARM, 1.0, 0.0),
    "badlands": BiomeInfo(37, "minecraft:badlands", Climate.WARM, 2.0, 0.0),
    "eroded_badlands": BiomeInfo(165, "minecraft:eroded_badlands", Climate.WARM, 2.0, 0.0),
    "wooded_badlands": BiomeInfo(38, "minecraft:wooded_badlands", Climate.WARM, 2.0, 0.0),
    # ---- Warm / humid ----
    "jungle": BiomeInfo(21, "minecraft:jungle", Climate.WARM, 0.95, 0.9),
    "sparse_jungle": BiomeInfo(23, "minecraft:sparse_jungle", Climate.WARM, 0.95, 0.8),
    "bamboo_jungle": BiomeInfo(168, "minecraft:bamboo_jungle", Climate.WARM, 0.95, 0.9),
    "swamp": BiomeInfo(6, "minecraft:swamp", Climate.WARM, 0.8, 0.9),
    "mangrove_swamp": BiomeInfo(184, "minecraft:mangrove_swamp", Climate.WARM, 0.8, 0.9),
    # ---- Temperate ----
    "plains": BiomeInfo(1, "minecraft:plains", Climate.TEMPERATE, 0.8, 0.4),
    "sunflower_plains": BiomeInfo(129, "minecraft:sunflower_plains", Climate.TEMPERATE, 0.8, 0.4),
    "forest": BiomeInfo(4, "minecraft:forest", Climate.TEMPERATE, 0.7, 0.8),
    "flower_forest": BiomeInfo(132, "minecraft:flower_forest", Climate.TEMPERATE, 0.7, 0.8),
    "birch_forest": BiomeInfo(27, "minecraft:birch_forest", Climate.TEMPERATE, 0.6, 0.6),
    "dark_forest": BiomeInfo(29, "minecraft:dark_forest", Climate.TEMPERATE, 0.7, 0.8),
    "old_growth_birch_forest": BiomeInfo(155, "minecraft:old_growth_birch_forest", Climate.TEMPERATE, 0.6, 0.6),
    "cherry_grove": BiomeInfo(192, "minecraft:cherry_grove", Climate.TEMPERATE, 0.5, 0.8),
    "meadow": BiomeInfo(186, "minecraft:meadow", Climate.TEMPERATE, 0.5, 0.8),
    # ---- Cold ----
    "taiga": BiomeInfo(5, "minecraft:taiga", Climate.COLD, 0.25, 0.8),
    "old_growth_pine_taiga": BiomeInfo(32, "minecraft:old_growth_pine_taiga", Climate.COLD, 0.3, 0.8),
    "old_growth_spruce_taiga": BiomeInfo(160, "minecraft:old_growth_spruce_taiga", Climate.COLD, 0.25, 0.8),
    "windswept_hills": BiomeInfo(3, "minecraft:windswept_hills", Climate.COLD, 0.2, 0.3),
    "windswept_forest": BiomeInfo(34, "minecraft:windswept_forest", Climate.COLD, 0.2, 0.3),
    "stony_peaks": BiomeInfo(189, "minecraft:stony_peaks", Climate.COLD, 1.0, 0.3),
    # ---- Snowy ----
    "snowy_plains": BiomeInfo(12, "minecraft:snowy_plains", Climate.SNOWY, 0.0, 0.5),
    "snowy_taiga": BiomeInfo(30, "minecraft:snowy_taiga", Climate.SNOWY, -0.5, 0.4),
    "snowy_beach": BiomeInfo(26, "minecraft:snowy_beach", Climate.SNOWY, 0.05, 0.3),
    "snowy_slopes": BiomeInfo(188, "minecraft:snowy_slopes", Climate.SNOWY, -0.3, 0.9),
    "frozen_peaks": BiomeInfo(183, "minecraft:frozen_peaks", Climate.SNOWY, -0.7, 0.9),
    "jagged_peaks": BiomeInfo(182, "minecraft:jagged_peaks", Climate.SNOWY, -0.7, 0.9),
    "ice_spikes": BiomeInfo(140, "minecraft:ice_spikes", Climate.SNOWY, 0.0, 0.5),
    "grove": BiomeInfo(185, "minecraft:grove", Climate.SNOWY, -0.2, 0.8),
    # ---- Special ----
    "mushroom_fields": BiomeInfo(14, "minecraft:mushroom_fields", Climate.TEMPERATE, 0.9, 1.0),
    # ---- Ocean variants ----
    "ocean": BiomeInfo(0, "minecraft:ocean", Climate.OCEAN, 0.5, 0.5, is_ocean=True),
    "deep_ocean": BiomeInfo(24, "minecraft:deep_ocean", Climate.OCEAN, 0.5, 0.5, is_ocean=True),
    "warm_ocean": BiomeInfo(44, "minecraft:warm_ocean", Climate.OCEAN, 0.8, 0.5, is_ocean=True),
    "lukewarm_ocean": BiomeInfo(45, "minecraft:lukewarm_ocean", Climate.OCEAN, 0.6, 0.5, is_ocean=True),
    "deep_lukewarm_ocean": BiomeInfo(48, "minecraft:deep_lukewarm_ocean", Climate.OCEAN, 0.6, 0.5, is_ocean=True),
    "cold_ocean": BiomeInfo(46, "minecraft:cold_ocean", Climate.OCEAN, 0.3, 0.5, is_ocean=True),
    "deep_cold_ocean": BiomeInfo(49, "minecraft:deep_cold_ocean", Climate.OCEAN, 0.3, 0.5, is_ocean=True),
    "frozen_ocean": BiomeInfo(10, "minecraft:frozen_ocean", Climate.OCEAN, 0.0, 0.5, is_ocean=True),
    "deep_frozen_ocean": BiomeInfo(50, "minecraft:deep_frozen_ocean", Climate.OCEAN, 0.0, 0.5, is_ocean=True),
    # ---- Beaches / rivers ----
    "beach": BiomeInfo(16, "minecraft:beach", Climate.TEMPERATE, 0.8, 0.4),
    "river": BiomeInfo(7, "minecraft:river", Climate.TEMPERATE, 0.5, 0.5),
    "frozen_river": BiomeInfo(11, "minecraft:frozen_river", Climate.SNOWY, 0.0, 0.5),
}


# --------------------------------------------------------------------------- #
# Convenience groupings — exposed in the criteria DSL
# --------------------------------------------------------------------------- #
#: Maps a "group name" to the set of biome keys (the ``BIOMES`` dict keys)
#: that belong to it. Users can write ``"warm_dry"`` in their criteria file
#: and the resolver will expand it.
BIOME_GROUPS: dict[str, frozenset[str]] = {
    "warm_dry": frozenset({"desert", "savanna", "savanna_plateau", "badlands",
                           "eroded_badlands", "wooded_badlands"}),
    "warm_humid": frozenset({"jungle", "sparse_jungle", "bamboo_jungle",
                             "swamp", "mangrove_swamp"}),
    "temperate": frozenset({"plains", "sunflower_plains", "forest",
                            "flower_forest", "birch_forest", "dark_forest",
                            "old_growth_birch_forest", "cherry_grove",
                            "meadow"}),
    "cold": frozenset({"taiga", "old_growth_pine_taiga",
                       "old_growth_spruce_taiga", "windswept_hills",
                       "windswept_forest", "stony_peaks"}),
    "snowy": frozenset({"snowy_plains", "snowy_taiga", "snowy_beach",
                        "snowy_slopes", "frozen_peaks", "jagged_peaks",
                        "ice_spikes", "grove"}),
    "ocean": frozenset(k for k, v in BIOMES.items() if v.is_ocean),
    "rare": frozenset({"mushroom_fields", "ice_spikes", "cherry_grove",
                       "eroded_badlands", "flower_forest"}),
}


# --------------------------------------------------------------------------- #
# Resolver utilities
# --------------------------------------------------------------------------- #
def resolve_biome_selector(selector: str | list[str]) -> frozenset[str]:
    """Expand a user-supplied biome selector to a concrete set of biome keys.

    A selector is either:

    * A single biome key (``"plains"``)
    * A group name (``"warm_dry"``)
    * A list of either, which is unioned

    Unknown selectors raise :class:`KeyError`. Returns the canonical biome
    keys (suitable for indexing :data:`BIOMES`).
    """
    if isinstance(selector, str):
        selector = [selector]

    resolved: set[str] = set()
    for token in selector:
        token = token.lower().strip()
        # Strip the optional "minecraft:" namespace prefix.
        if token.startswith("minecraft:"):
            token = token[len("minecraft:"):]
        if token in BIOMES:
            resolved.add(token)
        elif token in BIOME_GROUPS:
            resolved.update(BIOME_GROUPS[token])
        else:
            raise KeyError(f"Unknown biome or group: {token!r}")
    return frozenset(resolved)


def numeric_ids_for(selector: str | list[str]) -> frozenset[int]:
    """Same as :func:`resolve_biome_selector` but returns numeric IDs.

    Used by the criteria checker, which works against numeric biome IDs for
    speed (set-of-ints membership is much faster than set-of-strings).
    """
    keys = resolve_biome_selector(selector)
    return frozenset(BIOMES[k].numeric_id for k in keys)
