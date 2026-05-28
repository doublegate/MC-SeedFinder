"""
Optional Rust backend bridge.

The package remains importable without a compiled extension. When the PyO3
module is present, structure-only searches can use native batch filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Tuple


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

    def as_tuple(self) -> Tuple[str, int, int, int]:
        return (self.structure, self.max_distance, self.centre_x, self.centre_z)


def is_available() -> bool:
    """Return True if the native extension is importable."""
    return _native is not None


def java_random(seed: int) -> Any:
    """Create the native JavaRandom object."""
    if _native is None:
        raise RuntimeError("Rust backend is not available")
    return _native.JavaRandom(seed)


def get_structure_pos(
    structure: str,
    world_seed: int,
    region_x: int,
    region_z: int,
) -> Tuple[str, int, int]:
    """Return ``(structure, chunk_x, chunk_z)`` from the native backend."""
    if _native is None:
        raise RuntimeError("Rust backend is not available")
    return _native.get_structure_pos_py(structure, world_seed, region_x, region_z)


def iter_strongholds(world_seed: int, max_rings: int = 3) -> List[Tuple[str, int, int]]:
    """Return stronghold positions from the native backend."""
    if _native is None:
        raise RuntimeError("Rust backend is not available")
    return _native.iter_strongholds_py(world_seed, max_rings)


def find_structure_matches_range(
    start_seed: int,
    count: int,
    requirements: Iterable[RustStructureRequirement],
) -> List[int]:
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
) -> Optional[List[RustStructureRequirement]]:
    """Return native requirements when a criteria spec is structure-only.

    The native backend supports normal random-spread structures and the first
    stronghold ring used by the existing criteria path.
    """
    allowed_keys = {"nearby_structures"}
    if set(criteria_spec) - allowed_keys:
        return None

    entries = criteria_spec.get("nearby_structures") or []
    if not entries:
        return None

    requirements: List[RustStructureRequirement] = []
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
