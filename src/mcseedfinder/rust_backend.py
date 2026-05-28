"""
Optional Rust backend bridge.

The package remains importable without a compiled extension. When the PyO3
module is present, structure-only searches can use native batch filtering.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - availability depends on build mode
    from . import _native
except ImportError:  # pragma: no cover
    _native = None


@dataclass(frozen=True)
class RustStructureRequirement:
    """Requirement shape accepted by the Rust batch search."""

    structure: str
    max_distance: int
    centre_x: int = 0
    centre_z: int = 0

    def as_tuple(self) -> tuple[str, int, int, int]:
        return (self.structure, self.max_distance, self.centre_x, self.centre_z)


def is_available() -> bool:
    """Return True if the native extension is importable."""
    return _native is not None


def java_random(seed: int) -> Any:
    """Create the native JavaRandom object."""
    if _native is None:
        raise RuntimeError("Rust backend is not available")
    return _native.JavaRandom(seed)


def has_cubiomes() -> bool:
    """Return True if the native extension links the exact cubiomes biome backend."""
    return _native is not None and bool(getattr(_native, "HAS_CUBIOMES", False))


def make_biome_backend(
    version: str,
    dimension: str = "overworld",
    y: int | None = None,
) -> Any | None:
    """Construct an exact cubiomes biome backend, or ``None`` if unavailable.

    The returned object implements ``get_biome(world_seed, x, z) -> int`` (the
    ``BiomeGenerator`` protocol), so it drops straight into ``BiomeLookup``.
    Returns ``None`` when the native extension lacks cubiomes or the version /
    dimension is not recognised — callers then fall back to the approximate
    backend rather than failing a search.
    """
    if not has_cubiomes():
        return None
    try:
        if y is None:
            return _native.CubiomesBiomeBackend(version, dimension)
        return _native.CubiomesBiomeBackend(version, dimension, y)
    except (ValueError, RuntimeError):
        return None


def get_structure_pos(
    structure: str,
    world_seed: int,
    region_x: int,
    region_z: int,
) -> tuple[str, int, int]:
    """Return ``(structure, chunk_x, chunk_z)`` from the native backend."""
    if _native is None:
        raise RuntimeError("Rust backend is not available")
    return _native.get_structure_pos_py(structure, world_seed, region_x, region_z)


def iter_strongholds(world_seed: int, max_rings: int = 3) -> list[tuple[str, int, int]]:
    """Return stronghold positions from the native backend."""
    if _native is None:
        raise RuntimeError("Rust backend is not available")
    return _native.iter_strongholds_py(world_seed, max_rings)


def find_structure_matches_range(
    start_seed: int,
    count: int,
    requirements: Iterable[RustStructureRequirement],
) -> list[int]:
    """Find seeds matching all native structure requirements."""
    if _native is None:
        raise RuntimeError("Rust backend is not available")
    return _native.find_structure_matches_range(
        start_seed,
        count,
        [req.as_tuple() for req in requirements],
    )


def compile_structure_only_requirements(
    criteria_spec: Mapping[str, Any],
) -> list[RustStructureRequirement] | None:
    """Return native requirements when a criteria spec is structure-only.

    The native backend supports normal random-spread structures and the first
    stronghold ring used by the existing criteria path. Only handles the *flat*
    ``nearby_structures`` form; for the recursive ``conditions`` tree see
    :func:`compile_structure_only_tree`.
    """
    allowed_keys = {"nearby_structures"}
    if set(criteria_spec) - allowed_keys:
        return None

    entries = criteria_spec.get("nearby_structures") or []
    if not entries:
        return None

    requirements: list[RustStructureRequirement] = []
    for entry in entries:
        structure = str(entry["structure"])
        requirements.append(
            RustStructureRequirement(
                structure=structure,
                max_distance=int(entry.get("max_distance", 1500)),
                centre_x=int(entry.get("centre_x", 0)),
                centre_z=int(entry.get("centre_z", 0)),
            )
        )
    return requirements


# Node types the native tree evaluator handles. Anything else (biome-touching
# leaves: spawn_biome, nearby_biomes, biome_area) forces fallback to Python.
_NATIVE_TREE_LEAFS = {"nearby_structure", "cluster"}
_NATIVE_TREE_GROUPS = {"all_of", "any_of", "none_of"}


def _is_structure_only_node(node: Any) -> bool:
    """True when ``node`` is a conditions-tree node the native evaluator covers."""
    if not isinstance(node, Mapping):
        return False
    t = node.get("type")
    if t in _NATIVE_TREE_LEAFS:
        return True
    if t in _NATIVE_TREE_GROUPS:
        of = node.get("of") or []
        return bool(of) and all(_is_structure_only_node(c) for c in of)
    return False


def compile_structure_only_tree(
    criteria_spec: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Return a tree dict for the native evaluator when the spec is purely
    structure-only — covering the new ``conditions`` tree and the legacy flat
    ``nearby_structures`` keys (which compile to an ``all_of`` of leaves).

    Returns ``None`` when the spec mixes in any biome criterion or anything the
    native evaluator does not recognise, so the caller can fall back to the
    Python evaluation path without ever silently dropping a criterion.
    """
    if _native is None:
        return None

    # Reject specs that include biome-touching criteria.
    if criteria_spec.get("spawn_biome") is not None:
        return None
    if criteria_spec.get("nearby_biomes") is not None:
        return None

    children: list[Mapping[str, Any]] = []

    # Flat nearby_structures → cluster of structure leaves.
    for entry in criteria_spec.get("nearby_structures") or []:
        children.append(
            {
                "type": "nearby_structure",
                "structure": str(entry["structure"]),
                "max_distance": int(entry.get("max_distance", 1500)),
                "centre_x": int(entry.get("centre_x", 0)),
                "centre_z": int(entry.get("centre_z", 0)),
            }
        )

    # Conditions tree must be entirely structure-only to qualify.
    tree = criteria_spec.get("conditions")
    if tree is not None:
        if not _is_structure_only_node(tree):
            return None
        children.append(tree)

    if not children:
        return None
    if len(children) == 1:
        return children[0]
    return {"type": "all_of", "of": children}


def find_tree_matches_range(
    start_seed: int,
    count: int,
    tree: Mapping[str, Any],
) -> list[int]:
    """Filter ``[start_seed, start_seed+count)`` via the native tree evaluator."""
    if _native is None:
        raise RuntimeError("Rust backend is not available")
    import json

    return _native.find_tree_matches_range(start_seed, count, json.dumps(tree))
