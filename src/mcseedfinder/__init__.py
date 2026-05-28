"""
mcseedfinder
============

Minecraft Java Edition world-seed finder.

Public API
----------
* :class:`~mcseedfinder.java_random.JavaRandom`
* :func:`~mcseedfinder.structures.get_structure_pos`
* :func:`~mcseedfinder.structures.iter_structures_in_radius`
* :func:`~mcseedfinder.structures.iter_strongholds`
* :class:`~mcseedfinder.biome_gen.BiomeLookup`
* :func:`~mcseedfinder.criteria.compile_criteria`
* :func:`~mcseedfinder.criteria.load_criteria_file`
* :func:`~mcseedfinder.finder.run_search`
* :func:`~mcseedfinder.engine.run_staged_search`

See ``README.md`` in the project root for the full design discussion and
accuracy disclaimers.
"""

from .biome_gen import BiomeLookup
from .biomes import BIOMES, BIOME_GROUPS, BiomeInfo
from .app_contract import AsyncCommandHost, JobStore, LocalCommandHost
from .criteria import (
    CriteriaSet,
    compile_criteria,
    compile_cli_criteria,
    load_criteria_file,
)
from .finder import Match, SearchConfig, SearchPlan, run_search
from .engine import (
    BedrockProvider,
    JavaPythonProvider,
    SearchEvent,
    SearchSpec,
    SeedReport,
    provider_for,
    run_staged_search,
)
from .rust_backend import is_available as rust_backend_available
from .java_random import JavaRandom
from .structures import (
    STRUCTURE_CONFIGS,
    SUPPORTED_STRUCTURES,
    StructureConfig,
    StructurePos,
    get_structure_pos,
    iter_strongholds,
    iter_structures_in_radius,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # App contract
    "AsyncCommandHost", "JobStore", "LocalCommandHost",
    # Random
    "JavaRandom",
    # Biomes
    "BIOMES", "BIOME_GROUPS", "BiomeInfo", "BiomeLookup",
    # Structures
    "STRUCTURE_CONFIGS", "SUPPORTED_STRUCTURES",
    "StructureConfig", "StructurePos",
    "get_structure_pos", "iter_structures_in_radius", "iter_strongholds",
    # Criteria
    "CriteriaSet", "compile_criteria", "compile_cli_criteria",
    "load_criteria_file",
    # Finder
    "Match", "SearchConfig", "SearchPlan", "run_search",
    # Product engine
    "BedrockProvider", "JavaPythonProvider", "SearchEvent", "SearchSpec",
    "SeedReport", "provider_for", "run_staged_search",
    # Optional Rust backend
    "rust_backend_available",
]
