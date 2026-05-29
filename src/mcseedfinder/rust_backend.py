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


# Node types the native tree evaluator handles. The structure-only set drives
# the cheap structure-only fast path; the full set (with biome leaves) drives
# the new biome-aware native path that uses a shared cubiomes BiomeBackend.
_NATIVE_TREE_LEAFS = {"nearby_structure", "cluster"}
_NATIVE_TREE_BIOME_LEAFS = {"spawn_biome", "nearby_biomes", "biome_area"}
_NATIVE_TREE_LEAFS_FULL = _NATIVE_TREE_LEAFS | _NATIVE_TREE_BIOME_LEAFS
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


def _resolve_biome_leaf(
    node: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Convert a biome leaf (with biome NAMES) into the wire shape expected by
    the Rust evaluator (numeric cubiomes IDs). Returns None for unrecognised
    leaf types or when the biome catalog can't resolve a name."""
    from .biomes import numeric_ids_for

    t = node.get("type")
    biome_names = node.get("biomes")
    if biome_names is None:
        return None
    try:
        ids = sorted(numeric_ids_for(biome_names))
    except KeyError:
        return None
    if t == "spawn_biome":
        return {
            "type": "spawn_biome",
            "biomes": ids,
            "spawn_radius": int(node.get("spawn_radius", 64)),
        }
    if t == "nearby_biomes":
        return {
            "type": "nearby_biomes",
            "biomes": ids,
            "radius": int(node.get("radius", 2000)),
            "all_required": bool(node.get("all", node.get("all_required", False))),
            "samples_per_axis": int(node.get("samples", node.get("samples_per_axis", 16))),
        }
    if t == "biome_area":
        return {
            "type": "biome_area",
            "biomes": ids,
            "radius": int(node.get("radius", 1000)),
            "samples_per_axis": int(node.get("samples", node.get("samples_per_axis", 16))),
            "min_samples": int(node.get("min_samples", 8)),
            "centre_x": int(node.get("centre_x", 0)),
            "centre_z": int(node.get("centre_z", 0)),
        }
    return None


def _node_for_native(node: Any) -> Mapping[str, Any] | None:
    """Recursively rebuild a conditions-tree node into the wire shape the
    Rust evaluator consumes (structure leaves are pass-through; biome leaves
    have names resolved to IDs; groups recurse). Returns None if any leaf
    has a type the native evaluator doesn't recognise."""
    if not isinstance(node, Mapping):
        return None
    t = node.get("type")
    if t in _NATIVE_TREE_LEAFS:
        # Structure leaves already use the wire-format keys.
        return dict(node)
    if t in _NATIVE_TREE_BIOME_LEAFS:
        return _resolve_biome_leaf(node)
    if t in _NATIVE_TREE_GROUPS:
        of = node.get("of") or []
        if not of:
            return None
        children: list[Mapping[str, Any]] = []
        for c in of:
            converted = _node_for_native(c)
            if converted is None:
                return None
            children.append(converted)
        return {"type": t, "of": children}
    return None


def compile_native_tree_full(
    criteria_spec: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Build a tree dict for the biome-aware native evaluator.

    Accepts both structure leaves and biome leaves (with biome names; resolved
    to numeric IDs here). Returns ``None`` when the spec contains anything the
    native evaluator doesn't recognise so the caller can fall back to the
    Python path without ever silently dropping a criterion. The companion
    [`find_tree_matches_range_compiled_with_biomes`] is what actually runs the
    search — this helper is just the spec-to-wire-format conversion.
    """
    if _native is None:
        return None

    children: list[Mapping[str, Any]] = []

    # Flat nearby_structures → structure leaves.
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

    # Flat spawn_biome key → spawn_biome leaf with resolved IDs.
    spawn = criteria_spec.get("spawn_biome")
    if spawn is not None:
        try:
            from .biomes import numeric_ids_for

            children.append(
                {
                    "type": "spawn_biome",
                    "biomes": sorted(numeric_ids_for(spawn)),
                    "spawn_radius": int(criteria_spec.get("spawn_radius", 64)),
                }
            )
        except KeyError:
            return None

    # Flat nearby_biomes key.
    nb = criteria_spec.get("nearby_biomes")
    if nb is not None:
        try:
            from .biomes import numeric_ids_for

            children.append(
                {
                    "type": "nearby_biomes",
                    "biomes": sorted(numeric_ids_for(nb["biomes"])),
                    "radius": int(nb.get("radius", 2000)),
                    "all_required": bool(nb.get("all", nb.get("all_required", False))),
                    "samples_per_axis": int(nb.get("samples", 16)),
                }
            )
        except KeyError:
            return None

    # Recursive conditions tree.
    tree = criteria_spec.get("conditions")
    if tree is not None:
        converted = _node_for_native(tree)
        if converted is None:
            return None
        children.append(converted)

    if not children:
        return None
    if len(children) == 1:
        return children[0]
    return {"type": "all_of", "of": children}


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
    """Filter ``[start_seed, start_seed+count)`` via the native tree evaluator.

    Re-parses and re-compiles the tree on every call. For repeated calls
    against the same criteria — the worker-pool case — prefer
    :func:`compile_native_tree` once at worker init plus
    :func:`find_tree_matches_range_compiled` per chunk.
    """
    if _native is None:
        raise RuntimeError("Rust backend is not available")
    import json

    return _native.find_tree_matches_range(start_seed, count, json.dumps(tree))


def compile_native_tree(tree: Mapping[str, Any]) -> Any | None:
    """Compile a structure-only tree once and return a reusable native handle.

    Returns ``None`` when the native extension is unavailable, or when the
    JSON encode fails for some reason. Errors from the validator (unknown
    structure, negative distance, …) are still raised so they surface at
    pool-init time rather than per chunk.
    """
    if _native is None or not hasattr(_native, "compile_tree"):
        return None
    import json

    return _native.compile_tree(json.dumps(tree))


def find_tree_matches_range_compiled(
    start_seed: int,
    count: int,
    compiled: Any,
) -> list[int]:
    """Filter ``[start_seed, start_seed+count)`` using a pre-compiled handle.

    The handle is the value returned by :func:`compile_native_tree`. Skips
    the per-chunk JSON parse + compile and releases the GIL around the
    inner Rust loop.
    """
    if _native is None:
        raise RuntimeError("Rust backend is not available")
    return _native.find_tree_matches_range_compiled(start_seed, count, compiled)


def find_tree_matches_range_compiled_with_biomes(
    start_seed: int,
    count: int,
    compiled: Any,
    backend: Any,
) -> list[int]:
    """Biome-aware sibling of :func:`find_tree_matches_range_compiled`.

    Runs entirely inside Rust + cubiomes — no per-coord PyO3 callback into
    Python. Requires that ``backend`` be a ``CubiomesBiomeBackend`` (the
    object returned by :func:`make_biome_backend`).

    Returns an empty list and falls back to the caller's Python path if the
    extension was built without cubiomes or the biome-aware native entry
    point isn't present.
    """
    if _native is None or not hasattr(_native, "find_tree_matches_range_compiled_with_biomes"):
        raise RuntimeError("biome-aware native search not available")
    return _native.find_tree_matches_range_compiled_with_biomes(
        start_seed, count, compiled, backend
    )


def has_biome_aware_native() -> bool:
    """True iff this build exposes :func:`find_tree_matches_range_compiled_with_biomes`."""
    return _native is not None and hasattr(
        _native, "find_tree_matches_range_compiled_with_biomes"
    )


def is_supported_version(version: str) -> bool:
    """True iff the bundled cubiomes recognises ``version``.

    Lets the CLI surface a clear "this version isn't in the bundled cubiomes"
    error before the user invests time in a search. Returns ``False`` if the
    extension wasn't built with cubiomes — the caller should then either
    accept the version (approximate-biomes builds) or warn separately.
    """
    if _native is None or not hasattr(_native, "is_supported_version"):
        return False
    return bool(_native.is_supported_version(version))
