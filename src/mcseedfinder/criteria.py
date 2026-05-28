"""
criteria.py
===========

Criteria DSL: declarative specification of what a "matching seed" looks like.

A :class:`CriteriaSet` is a list of individual :class:`Criterion` objects,
each of which can answer ``matches(world_seed)`` in isolation. The finder
runs them in *cost order* (cheapest first) and short-circuits as soon as one
fails — important when sweeping millions of seeds.

The JSON schema (also accepted as a dict from Python code) supports a flat
form for common searches and a recursive ``conditions`` tree for advanced ones
(logic gates, structure clusters, biome-area). The two compose — flat keys and
the tree are all AND'd at the top level.

Flat keys::

    {
      "spawn_biome": "plains" | ["plains", "savanna"] | "temperate",
      "spawn_radius": 128,                    // optional, default 64

      "nearby_structures": [
        {"structure": "village",        "max_distance": 1500},
        {"structure": "ocean_monument", "max_distance": 3000}
      ],

      "nearby_biomes": {
        "biomes":    ["jungle", "desert", "mushroom_fields"],
        "all":       true,                    // require all (else any)
        "radius":    2000,
        "samples":   64                       // grid resolution, optional
      }
    }

Advanced ``conditions`` tree::

    {
      "conditions": {
        "type": "all_of",                     // any_of | all_of | none_of
        "of": [
          {"type": "cluster",                 // quad-hut / multi-structure
           "structures": ["swamp_hut"],
           "min_count": 4, "max_distance": 128,
           "centre_x": 0, "centre_z": 0},

          {"type": "any_of",                  // nested logic gates
           "of": [
             {"type": "nearby_structure",
              "structure": "village", "max_distance": 1500},
             {"type": "nearby_structure",
              "structure": "pillager_outpost", "max_distance": 1500}
           ]},

          {"type": "biome_area",              // exact biome area (cubiomes)
           "biomes": ["plains", "savanna"],
           "centre_x": 0, "centre_z": 0,
           "radius": 1000, "samples_per_axis": 16, "min_samples": 32}
        ]
      }
    }

Each entry compiles to a Criterion subclass; the resolver below performs the
compilation. New criterion types can be added in three steps:

1. Subclass :class:`Criterion` and implement :meth:`evaluate`.
2. Set a sensible :attr:`cost` (1 = trivial, 10 = heavyweight grid sample).
3. Register a handler in :func:`_compile_one` and :func:`_node_compile`.
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
# Structure-cluster criterion — generalizes quad-hut / quad-monument
# --------------------------------------------------------------------------- #
@dataclass
class StructureCluster(Criterion):
    """Require at least ``min_count`` structure placements (counted across the
    given structure list) within ``max_distance`` of a reference centre.

    Setting ``structures=["swamp_hut"]`` with ``min_count=4`` and a tight radius
    is the classic quad-hut search. Mixed lists let users look for dense
    multi-structure neighbourhoods (e.g. a village + outpost + pyramid cluster).
    Each structure type is validated up-front.
    """

    structures: Tuple[str, ...] = ()
    max_distance: int = 1500
    min_count: int = 4
    centre_x: int = 0
    centre_z: int = 0
    cost: int = 3  # cheaper than a biome grid; more work than a single structure
    stage: int = 1

    def __post_init__(self) -> None:
        if not self.structures:
            raise ValueError("StructureCluster requires a non-empty `structures` list")
        if self.min_count < 1:
            raise ValueError("StructureCluster.min_count must be >= 1")
        for name in self.structures:
            if name not in SUPPORTED_STRUCTURES:
                raise ValueError(
                    f"Unknown structure {name!r}. Supported: "
                    f"{sorted(SUPPORTED_STRUCTURES)}"
                )

    def evaluate(self, world_seed: int, lookup: Optional[BiomeLookup]) -> bool:
        hits = 0
        for name in self.structures:
            if name == "stronghold":
                # Strongholds use a different generator; only ring 1 is near origin.
                for sh in iter_strongholds(world_seed, max_rings=1):
                    if sh.distance_to(self.centre_x, self.centre_z) <= self.max_distance:
                        hits += 1
                        if hits >= self.min_count:
                            return True
                continue
            for _ in iter_structures_in_radius(
                name, world_seed, self.centre_x, self.centre_z, self.max_distance
            ):
                hits += 1
                if hits >= self.min_count:
                    return True
        return False

    def describe(self) -> str:
        return (
            f"≥{self.min_count} of {list(self.structures)} within "
            f"{self.max_distance} blocks of ({self.centre_x}, {self.centre_z})"
        )


# --------------------------------------------------------------------------- #
# Biome-area criterion — minimum area of a biome (set) within a region
# --------------------------------------------------------------------------- #
@dataclass
class BiomeArea(Criterion):
    """Require at least ``min_samples`` grid samples whose biome is in the set,
    within a square of side ``2 * radius`` centred on ``(centre_x, centre_z)``.

    This is meaningful only with the exact cubiomes backend — the approximate
    fallback will under/over-count by definition. With biomes exact, it answers
    "is there a sizeable patch of <biome> near <point>?" which Cubiomes Viewer
    handles via its biome analysis tab.
    """

    biomes: FrozenSet[int] = field(default_factory=frozenset)
    radius: int = 1000
    samples_per_axis: int = 16
    min_samples: int = 8
    centre_x: int = 0
    centre_z: int = 0
    cost: int = 9  # comparable to NearbyBiomes; depends on grid resolution
    stage: int = 3

    def __post_init__(self) -> None:
        if not self.biomes:
            raise ValueError("BiomeArea requires a non-empty `biomes` set")
        total = self.samples_per_axis * self.samples_per_axis
        if not (1 <= self.min_samples <= total):
            raise ValueError(
                f"BiomeArea.min_samples must be in [1, {total}], got {self.min_samples}"
            )

    def evaluate(self, world_seed: int, lookup: Optional[BiomeLookup]) -> bool:
        assert lookup is not None, "BiomeArea needs a BiomeLookup"
        target = set(self.biomes)
        step = max(1, (2 * self.radius) // max(1, self.samples_per_axis - 1))
        hits = 0
        # Early-exit threshold: as soon as we have min_samples hits we can stop.
        for ix in range(self.samples_per_axis):
            x = self.centre_x - self.radius + ix * step
            for iz in range(self.samples_per_axis):
                z = self.centre_z - self.radius + iz * step
                if lookup.biome_at(x, z) in target:
                    hits += 1
                    if hits >= self.min_samples:
                        return True
        return False

    def describe(self) -> str:
        return (
            f"≥{self.min_samples}/{self.samples_per_axis ** 2} samples of "
            f"{sorted(self.biomes)} within {self.radius} blocks of "
            f"({self.centre_x}, {self.centre_z})"
        )


# --------------------------------------------------------------------------- #
# Logic-gate group — hierarchical any_of / all_of / none_of
# --------------------------------------------------------------------------- #
@dataclass
class GroupCriterion(Criterion):
    """Compose child criteria with a boolean combinator.

    Combinators:
      * ``"all_of"`` — every child must hold (logical AND).
      * ``"any_of"`` — at least one child must hold (logical OR).
      * ``"none_of"`` — no child may hold (logical NOR / NOT-any).

    Children are cost-sorted within the group so the cheapest fail-fast.
    """

    combinator: str = "all_of"
    children: List[Criterion] = field(default_factory=list)
    stage: int = 2

    def __post_init__(self) -> None:
        if self.combinator not in ("all_of", "any_of", "none_of"):
            raise ValueError(f"unknown combinator {self.combinator!r}")
        if not self.children:
            raise ValueError(f"{self.combinator} group needs at least one child")
        # Cost-sort children so cheap predicates fail-fast inside the group.
        self.children.sort(key=lambda c: c.cost)
        # Effective cost: sum of children. all_of short-circuits on first false,
        # any_of on first true; sum is a reasonable upper-bound used purely for
        # ordering this group against its siblings.
        self.cost = sum(c.cost for c in self.children)

    def evaluate(self, world_seed: int, lookup: Optional[BiomeLookup]) -> bool:
        if self.combinator == "all_of":
            for child in self.children:
                if not child.evaluate(world_seed, lookup):
                    return False
            return True
        if self.combinator == "any_of":
            for child in self.children:
                if child.evaluate(world_seed, lookup):
                    return True
            return False
        # none_of
        for child in self.children:
            if child.evaluate(world_seed, lookup):
                return False
        return True

    def describe(self) -> str:
        joiner = {"all_of": " AND ", "any_of": " OR ", "none_of": " NOR "}[self.combinator]
        return "(" + joiner.join(c.describe() for c in self.children) + ")"


# --------------------------------------------------------------------------- #
# Criteria set
# --------------------------------------------------------------------------- #
@dataclass
class CriteriaSet:
    """An ordered list of criteria with short-circuit evaluation.

    ``needs_biome_lookup`` is precomputed so the finder can skip building a
    :class:`BiomeLookup` for purely-structural searches (much faster).

    When ``biome_version`` is set and the native cubiomes backend is available,
    biome lookups are **exact** for that version/dimension; otherwise they fall
    back to the approximate climate-noise generator. The backend is built once
    and reused across seeds (it re-applies the world seed internally), so a
    multi-million-seed sweep allocates a single generator per process.
    """

    criteria: List[Criterion]
    needs_biome_lookup: bool
    biome_version: Optional[str] = None
    biome_dimension: str = "overworld"
    biome_y: Optional[int] = None

    def __post_init__(self) -> None:
        # Sort by cost so cheap predicates fail-fast on bad seeds.
        self.criteria.sort(key=lambda c: c.cost)
        # Build the exact biome backend eagerly (cheap) so callers can report
        # exactness deterministically. None means "use the approximate fallback".
        self._biome_backend = None
        if self.needs_biome_lookup and self.biome_version is not None:
            # Imported lazily to keep the optional native bridge truly optional.
            from .rust_backend import make_biome_backend

            self._biome_backend = make_biome_backend(
                self.biome_version, self.biome_dimension, self.biome_y
            )

    @property
    def uses_exact_biomes(self) -> bool:
        """True when biome lookups are cubiomes-exact (not the approximation)."""
        return self.needs_biome_lookup and self._biome_backend is not None

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
        lookup = (
            BiomeLookup(world_seed, backend=self._biome_backend)
            if self.needs_biome_lookup
            else None
        )
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
def load_criteria_file(
    path: Union[str, Path],
    *,
    biome_version: Optional[str] = None,
    biome_dimension: str = "overworld",
    biome_y: Optional[int] = None,
) -> CriteriaSet:
    """Load a JSON criteria file and compile it."""
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    return compile_criteria(
        spec,
        biome_version=biome_version,
        biome_dimension=biome_dimension,
        biome_y=biome_y,
    )


def compile_criteria(
    spec: Mapping[str, Any],
    *,
    biome_version: Optional[str] = None,
    biome_dimension: str = "overworld",
    biome_y: Optional[int] = None,
) -> CriteriaSet:
    """Compile a spec dict (the JSON schema described at module top) to a set.

    Pass ``biome_version``/``biome_dimension`` to enable the exact cubiomes
    biome backend for biome criteria; without them, biome lookups use the
    approximate fallback.
    """
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

    # ---- conditions tree (logic gates, clusters, biome-area) ----
    tree = spec.get("conditions")
    if tree is not None:
        node, tree_needs_biome = _node_compile(tree)
        criteria.append(node)
        needs_biome = needs_biome or tree_needs_biome

    if not criteria:
        raise ValueError("criteria spec is empty — nothing to search for")

    return CriteriaSet(
        criteria=criteria,
        needs_biome_lookup=needs_biome,
        biome_version=biome_version,
        biome_dimension=biome_dimension,
        biome_y=biome_y,
    )


# --------------------------------------------------------------------------- #
# Recursive condition-tree compiler
# --------------------------------------------------------------------------- #
_GROUP_TYPES: FrozenSet[str] = frozenset({"all_of", "any_of", "none_of"})

_MAX_TREE_DEPTH = 16  # hard cap to keep pathological specs from blowing the stack


def _node_compile(node: Mapping[str, Any], *, depth: int = 0) -> Tuple[Criterion, bool]:
    """Compile one condition-tree node. Returns ``(criterion, needs_biome)``."""
    if not isinstance(node, Mapping):
        raise ValueError(
            f"condition node must be a JSON object, got {type(node).__name__}"
        )
    if depth > _MAX_TREE_DEPTH:
        raise ValueError(
            f"condition tree nested too deep (>{_MAX_TREE_DEPTH} levels)"
        )
    t = node.get("type")
    if t in _GROUP_TYPES:
        of = node.get("of") or []
        if not of:
            raise ValueError(f"{t!r} group has empty 'of' list")
        children: List[Criterion] = []
        any_biome = False
        for child in of:
            crit, child_biome = _node_compile(child, depth=depth + 1)
            children.append(crit)
            any_biome = any_biome or child_biome
        return GroupCriterion(combinator=t, children=children), any_biome
    if t == "nearby_structure":
        return (
            NearbyStructure(
                structure=str(node["structure"]),
                max_distance=int(node.get("max_distance", 1500)),
                centre_x=int(node.get("centre_x", 0)),
                centre_z=int(node.get("centre_z", 0)),
            ),
            False,
        )
    if t == "cluster":
        return (
            StructureCluster(
                structures=tuple(node["structures"]),
                max_distance=int(node.get("max_distance", 1500)),
                min_count=int(node.get("min_count", 4)),
                centre_x=int(node.get("centre_x", 0)),
                centre_z=int(node.get("centre_z", 0)),
            ),
            False,
        )
    if t == "spawn_biome":
        return (
            SpawnBiome(
                biomes=numeric_ids_for(node["biomes"]),
                spawn_radius=int(node.get("spawn_radius", 64)),
            ),
            True,
        )
    if t == "nearby_biomes":
        return (
            NearbyBiomes(
                biomes=numeric_ids_for(node["biomes"]),
                radius=int(node.get("radius", 2000)),
                all_required=bool(node.get("all", False)),
                samples_per_axis=int(node.get("samples", 16)),
            ),
            True,
        )
    if t == "biome_area":
        return (
            BiomeArea(
                biomes=numeric_ids_for(node["biomes"]),
                radius=int(node.get("radius", 1000)),
                samples_per_axis=int(node.get("samples", 16)),
                min_samples=int(node.get("min_samples", 8)),
                centre_x=int(node.get("centre_x", 0)),
                centre_z=int(node.get("centre_z", 0)),
            ),
            True,
        )
    raise ValueError(f"unknown condition type {t!r}")


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
