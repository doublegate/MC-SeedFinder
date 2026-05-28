"""
criteria.py
===========

Criteria DSL: declarative specification of what a "matching seed" looks like.

A :class:`CriteriaSet` is a list of individual :class:`Criterion` objects,
each of which can answer ``matches(world_seed)`` in isolation. The finder
runs them in *cost order* (cheapest first) and short-circuits as soon as one
fails — important when sweeping millions of seeds.

The JSON schema (also accepted as a dict from Python code) looks like::

    {
      "spawn_biome": "plains" | ["plains", "savanna"] | "temperate",
      "spawn_radius": 128,                    // optional, default 64

      "nearby_structures": [
        {"structure": "village",        "max_distance": 1500},
        {"structure": "ocean_monument", "max_distance": 3000, "required": true}
      ],

      "nearby_biomes": {
        "biomes":    ["jungle", "desert", "mushroom_fields"],
        "all":       true,                    // require all (else any)
        "radius":    2000,
        "samples":   64                       // grid resolution, optional
      }
    }

Each entry compiles to a Criterion subclass; the resolver below performs the
compilation. New criterion types can be added in three steps:

1. Subclass :class:`Criterion` and implement :meth:`evaluate`.
2. Set a sensible :attr:`cost` (1 = trivial, 10 = heavyweight grid sample).
3. Register a handler in :func:`_compile_one`.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, FrozenSet, Iterable, List, Mapping, Optional, Tuple, Union

from .biome_gen import BiomeLookup
from .biomes import numeric_ids_for
from .structures import (
    STRUCTURE_CONFIGS,
    SUPPORTED_STRUCTURES,
    iter_strongholds,
    iter_structures_in_radius,
)


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
@dataclass
class Criterion(ABC):
    """Abstract criterion. Subclasses declare a static ``cost`` for ordering."""

    #: Heuristic cost — used to put the cheapest predicates first so that a
    #: million-seed sweep spends as little time as possible on doomed seeds.
    cost: int = 5
    #: Staged-search phase. Stage 1 is cheap placement math, stage 2 is exact
    #: validation, and stage 3 is heavier sampling/scoring-like work.
    stage: int = 2

    @abstractmethod
    def evaluate(self, world_seed: int, lookup: Optional[BiomeLookup]) -> bool:
        """Return True if ``world_seed`` satisfies this criterion."""
        ...

    def describe(self) -> str:
        """Human-readable line for the report output. Override for nicer text."""
        return self.__class__.__name__


# --------------------------------------------------------------------------- #
# Structure-distance criterion (cheap: pure integer math, no biome lookup)
# --------------------------------------------------------------------------- #
@dataclass
class NearbyStructure(Criterion):
    """Require a given structure within ``max_distance`` blocks of origin."""

    structure: str = ""
    max_distance: int = 1500
    cost: int = 2  # cheap — structure math is pure arithmetic
    stage: int = 1
    centre_x: int = 0
    centre_z: int = 0

    def __post_init__(self) -> None:
        # Validate up-front so a bad criteria file fails fast, not after
        # spinning up worker processes.
        if self.structure not in SUPPORTED_STRUCTURES:
            raise ValueError(
                f"Unknown structure {self.structure!r}. Supported: "
                f"{sorted(SUPPORTED_STRUCTURES)}"
            )

    def evaluate(self, world_seed: int, lookup: Optional[BiomeLookup]) -> bool:
        if self.structure == "stronghold":
            # Strongholds use a different generator; ring 1 is what's near origin.
            for sh in iter_strongholds(world_seed, max_rings=1):
                if sh.distance_to(self.centre_x, self.centre_z) <= self.max_distance:
                    return True
            return False
        # Standard random_spread structure search.
        for _ in iter_structures_in_radius(
            self.structure, world_seed,
            self.centre_x, self.centre_z, self.max_distance
        ):
            return True
        return False

    def describe(self) -> str:
        return (f"{self.structure} within {self.max_distance} blocks of "
                f"({self.centre_x}, {self.centre_z})")


# --------------------------------------------------------------------------- #
# Spawn-biome criterion (single biome lookup at origin)
# --------------------------------------------------------------------------- #
@dataclass
class SpawnBiome(Criterion):
    """Require the biome at the origin (approximate spawn) to be in a set.

    Notes
    -----
    Minecraft's *actual* world-spawn point is the result of a multi-step
    process (search radius around the origin for valid terrain, prefer
    spawn-compatible biomes, etc.). We approximate it by sampling a small
    grid centred on the origin and accepting if **any** sample lies in the
    target set. Set ``spawn_radius`` to 0 for a strict single-point check.
    """

    biomes: FrozenSet[int] = field(default_factory=frozenset)
    spawn_radius: int = 64
    samples_per_axis: int = 5
    cost: int = 4  # one or a handful of biome samples
    stage: int = 2

    def evaluate(self, world_seed: int, lookup: Optional[BiomeLookup]) -> bool:
        assert lookup is not None, "SpawnBiome needs a BiomeLookup"
        if self.spawn_radius <= 0:
            return lookup.biome_at(0, 0) in self.biomes
        # Grid sample within +/- spawn_radius.
        step = max(1, (2 * self.spawn_radius) // (self.samples_per_axis - 1))
        for ix in range(self.samples_per_axis):
            x = -self.spawn_radius + ix * step
            for iz in range(self.samples_per_axis):
                z = -self.spawn_radius + iz * step
                if lookup.biome_at(x, z) in self.biomes:
                    return True
        return False

    def describe(self) -> str:
        return f"spawn biome in {sorted(self.biomes)} (within {self.spawn_radius} blocks)"


# --------------------------------------------------------------------------- #
# Nearby-biome criterion (expensive: grid sample)
# --------------------------------------------------------------------------- #
@dataclass
class NearbyBiomes(Criterion):
    """Require certain biomes to exist anywhere within a radius."""

    biomes: FrozenSet[int] = field(default_factory=frozenset)
    radius: int = 2000
    all_required: bool = False
    samples_per_axis: int = 16
    cost: int = 10  # grid sample is the heaviest predicate we support
    stage: int = 3

    def evaluate(self, world_seed: int, lookup: Optional[BiomeLookup]) -> bool:
        assert lookup is not None, "NearbyBiomes needs a BiomeLookup"
        target = set(self.biomes)
        found: set[int] = set()
        step = max(1, (2 * self.radius) // (self.samples_per_axis - 1))
        for ix in range(self.samples_per_axis):
            x = -self.radius + ix * step
            for iz in range(self.samples_per_axis):
                z = -self.radius + iz * step
                bid = lookup.biome_at(x, z)
                if bid in target:
                    found.add(bid)
                    if not self.all_required:
                        return True
                    if found == target:
                        return True
        return found == target if self.all_required else False

    def describe(self) -> str:
        mode = "all of" if self.all_required else "any of"
        return f"{mode} {sorted(self.biomes)} within {self.radius} blocks"


# --------------------------------------------------------------------------- #
# Criteria set
# --------------------------------------------------------------------------- #
@dataclass
class CriteriaSet:
    """An ordered list of criteria with short-circuit evaluation.

    ``needs_biome_lookup`` is precomputed so the finder can skip building a
    :class:`BiomeLookup` for purely-structural searches (much faster).
    """

    criteria: List[Criterion]
    needs_biome_lookup: bool

    def __post_init__(self) -> None:
        # Sort by cost so cheap predicates fail-fast on bad seeds.
        self.criteria.sort(key=lambda c: c.cost)

    def matches(self, world_seed: int) -> bool:
        """True iff every criterion holds for ``world_seed``."""
        matched, _, _ = self.evaluate_staged(world_seed)
        return matched

    def evaluate_staged(self, world_seed: int) -> Tuple[bool, Optional[int], List[str]]:
        """Evaluate criteria and return match status, failed stage, and passes.

        The existing boolean :meth:`matches` API remains the simple path used
        by the CLI. This richer form supports product search events and
        rejected-stage telemetry without duplicating criterion evaluation.
        """
        lookup = BiomeLookup(world_seed) if self.needs_biome_lookup else None
        passed: List[str] = []
        for crit in self.criteria:
            if not crit.evaluate(world_seed, lookup):
                return False, crit.stage, passed
            passed.append(crit.describe())
        return True, None, passed

    def describe(self) -> List[str]:
        """One human-readable line per criterion, in evaluation order."""
        return [c.describe() for c in self.criteria]


# --------------------------------------------------------------------------- #
# Compilation: spec dict / file → CriteriaSet
# --------------------------------------------------------------------------- #
def load_criteria_file(path: Union[str, Path]) -> CriteriaSet:
    """Load a JSON criteria file and compile it."""
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    return compile_criteria(spec)


def compile_criteria(spec: Mapping[str, Any]) -> CriteriaSet:
    """Compile a spec dict (the JSON schema described at module top) to a set."""
    criteria: List[Criterion] = []
    needs_biome = False

    # ---- nearby_structures (cheapest first) ----
    for entry in spec.get("nearby_structures", []) or []:
        criteria.append(_compile_structure_entry(entry))

    # ---- spawn_biome ----
    spawn = spec.get("spawn_biome")
    if spawn is not None:
        criteria.append(SpawnBiome(
            biomes=numeric_ids_for(spawn),
            spawn_radius=int(spec.get("spawn_radius", 64)),
        ))
        needs_biome = True

    # ---- nearby_biomes ----
    nb = spec.get("nearby_biomes")
    if nb is not None:
        criteria.append(NearbyBiomes(
            biomes=numeric_ids_for(nb["biomes"]),
            radius=int(nb.get("radius", 2000)),
            all_required=bool(nb.get("all", False)),
            samples_per_axis=int(nb.get("samples", 16)),
        ))
        needs_biome = True

    if not criteria:
        raise ValueError("criteria spec is empty — nothing to search for")

    return CriteriaSet(criteria=criteria, needs_biome_lookup=needs_biome)


def _compile_structure_entry(entry: Mapping[str, Any]) -> NearbyStructure:
    """Convert one ``nearby_structures`` JSON entry to a criterion object."""
    structure = entry["structure"]
    return NearbyStructure(
        structure=structure,
        max_distance=int(entry.get("max_distance", 1500)),
        centre_x=int(entry.get("centre_x", 0)),
        centre_z=int(entry.get("centre_z", 0)),
    )


# --------------------------------------------------------------------------- #
# Convenience: build a CriteriaSet from CLI-style kwargs
# --------------------------------------------------------------------------- #
def compile_cli_criteria(
    spawn_biome: Optional[Iterable[str]] = None,
    spawn_radius: int = 64,
    structures: Optional[Iterable[str]] = None,
    structure_radius: int = 1500,
    nearby_biomes: Optional[Iterable[str]] = None,
    nearby_biomes_radius: int = 2000,
    nearby_biomes_all: bool = False,
) -> CriteriaSet:
    """Build a CriteriaSet from the most common CLI arguments.

    This is a thin convenience layer; for anything beyond the basics use the
    JSON schema with :func:`load_criteria_file`.
    """
    spec: dict[str, Any] = {}
    if spawn_biome:
        spec["spawn_biome"] = list(spawn_biome)
        spec["spawn_radius"] = spawn_radius
    if structures:
        spec["nearby_structures"] = [
            {"structure": s, "max_distance": structure_radius}
            for s in structures
        ]
    if nearby_biomes:
        spec["nearby_biomes"] = {
            "biomes": list(nearby_biomes),
            "radius": nearby_biomes_radius,
            "all": nearby_biomes_all,
        }
    return compile_criteria(spec)
