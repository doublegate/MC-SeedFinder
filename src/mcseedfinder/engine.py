"""
Product-facing seed search contracts and staged event stream.

This module is the compatibility bridge toward the Rust/Tauri architecture:
it defines stable request, event, report, and provider shapes while using the
current Python criteria engine underneath. A future Rust backend can implement
the same provider methods without changing CLI or UI-facing contracts.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Protocol

from .criteria import CriteriaSet, compile_criteria
from .finder import SearchPlan, _iter_chunks


Exactness = str


@dataclass(frozen=True)
class SearchSpec:
    """Versioned product search request.

    ``criteria`` uses the existing JSON criteria schema today. The surrounding
    fields are explicit so desktop UI, Python CLI, and a later Rust engine all
    agree on edition/version/dimension, limits, and enumeration strategy.
    """

    criteria: Mapping[str, Any]
    edition: str = "java"
    version: str = "1.21"
    dimension: str = "overworld"
    mode: str = "sequential"
    start_seed: int = 0
    count: int = 1_000_000
    chunk_size: int = 4096
    random_seed: Optional[int] = None
    max_matches: int = 10
    scoring: Mapping[str, Any] = field(default_factory=dict)
    spec_version: int = 1

    def to_plan(self) -> SearchPlan:
        """Convert enumeration fields into the legacy finder plan."""
        return SearchPlan(
            mode=self.mode,
            start_seed=self.start_seed,
            count=self.count,
            chunk_size=self.chunk_size,
            random_seed=self.random_seed,
        )


@dataclass(frozen=True)
class SearchEvent:
    """Event emitted by a staged seed search."""

    type: str
    job_id: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    timestamp_seconds: float = field(default_factory=time.time)


@dataclass(frozen=True)
class SeedReport:
    """A verified seed result suitable for UI cards, exports, and APIs."""

    seed: int
    edition: str
    version: str
    dimension: str
    matched_features: List[str]
    exactness: Mapping[str, Exactness]
    score: float = 0.0
    map_preview: Optional[Mapping[str, Any]] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "seed": self.seed,
            "edition": self.edition,
            "version": self.version,
            "dimension": self.dimension,
            "matched_features": list(self.matched_features),
            "exactness": dict(self.exactness),
            "score": self.score,
            "map_preview": self.map_preview,
            "warnings": list(self.warnings),
        }


class Provider(Protocol):
    """Edition/version provider contract for future exact backends."""

    edition: str

    def validate_spec(self, spec: SearchSpec) -> None:
        """Raise ``ValueError`` if this provider cannot run ``spec``."""
        ...

    def evaluate_seed(
        self,
        seed: int,
        criteria: CriteriaSet,
        spec: SearchSpec,
    ) -> SeedReport | None:
        """Return a report for a verified match, else ``None``."""
        ...


class JavaPythonProvider:
    """Current Java provider backed by local Python structure/biome logic."""

    edition = "java"

    _SUPPORTED_DIMENSIONS = ("overworld", "the_overworld", "nether", "the_nether", "end", "the_end")

    def validate_spec(self, spec: SearchSpec) -> None:
        if spec.edition != self.edition:
            raise ValueError(f"Java provider cannot run edition {spec.edition!r}")
        if spec.dimension not in self._SUPPORTED_DIMENSIONS:
            raise ValueError(
                f"unsupported dimension {spec.dimension!r}; expected one of "
                f"{self._SUPPORTED_DIMENSIONS}"
            )

    def evaluate_seed(
        self,
        seed: int,
        criteria: CriteriaSet,
        spec: SearchSpec,
    ) -> SeedReport | None:
        matched, _, passed = criteria.evaluate_staged(seed)
        if not matched:
            return None
        exactness = _exactness_for_spec(spec.criteria, criteria.uses_exact_biomes)
        warnings: List[str] = []
        if "candidate" in exactness.values():
            warnings.append(
                "biome filters use the approximate local fallback; the exact "
                "cubiomes backend was not available for this build"
            )
        return SeedReport(
            seed=seed,
            edition=spec.edition,
            version=spec.version,
            dimension=spec.dimension,
            matched_features=passed,
            exactness=exactness,
            warnings=warnings,
        )


class BedrockProvider:
    """Bedrock Edition provider — Phase 5 foundation.

    Bedrock differs from Java at the worldgen level (different PRNG, biome
    layers, structure salts and spacing). Phase 5 plumbs Bedrock as a
    first-class edition through the engine, but the worldgen backend is
    deliberately not implemented yet — see ``docs/BEDROCK.md``. This
    provider:

    * validates the rest of the spec (edition, seed range — Bedrock uses
      signed i32, not Java's i64);
    * rejects worldgen-dependent criteria with a clear edition-aware
      error naming exactly what's missing (no silent fall-back to Java
      math, which would invalidate every match).

    Bedrock text-seed → numeric-seed hashing is supported and works
    independently of any search — see :func:`mcseedfinder.bedrock.seed_from_string`.
    """

    edition = "bedrock"

    def validate_spec(self, spec: SearchSpec) -> None:
        if spec.edition != self.edition:
            raise ValueError(f"Bedrock provider cannot run edition {spec.edition!r}")
        # Bedrock seeds are signed i32. Reject explicit Java-range starts.
        from .bedrock import is_valid_bedrock_seed

        if not is_valid_bedrock_seed(spec.start_seed):
            raise ValueError(
                f"Bedrock seeds are signed i32; start_seed {spec.start_seed} "
                f"is outside [-2**31, 2**31). For text seeds, hash them with "
                f"`mcseedfinder.bedrock.seed_from_string` first."
            )
        # Refuse criteria the Bedrock backend can't faithfully evaluate yet.
        unsupported = self._unsupported_criteria(spec.criteria)
        if unsupported:
            joined = ", ".join(sorted(unsupported))
            raise ValueError(
                f"Bedrock search does not yet support these criteria: {joined}. "
                f"Bedrock worldgen (biomes, structures, strongholds) is on the "
                f"roadmap (docs/BEDROCK.md). Text-seed hashing works today via "
                f"`mcseedfinder.bedrock.seed_from_string`."
            )

    @staticmethod
    def _unsupported_criteria(criteria_spec: Mapping[str, Any]) -> set[str]:
        """Set of criterion type names the Bedrock backend cannot evaluate.

        Today this is *everything* worldgen-dependent. Returning a set makes
        the error message enumerable rather than vague.
        """
        unsupported: set[str] = set()
        if criteria_spec.get("nearby_structures"):
            unsupported.add("nearby_structures")
        if criteria_spec.get("spawn_biome") is not None:
            unsupported.add("spawn_biome")
        if criteria_spec.get("nearby_biomes") is not None:
            unsupported.add("nearby_biomes")
        tree = criteria_spec.get("conditions")
        if tree is not None:
            unsupported.update(_collect_tree_leaf_types(tree))
        return unsupported

    def evaluate_seed(
        self,
        seed: int,
        criteria: CriteriaSet,
        spec: SearchSpec,
    ) -> SeedReport | None:
        # validate_spec already rejected any spec that gets here with non-empty
        # criteria, so reaching evaluate_seed implies the criteria set was
        # empty — which compile_criteria would have already refused. Belt and
        # braces: never silently return a Java-evaluated report for a Bedrock
        # spec.
        raise NotImplementedError(
            "Bedrock worldgen backend not yet implemented (docs/BEDROCK.md)"
        )


def _collect_tree_leaf_types(node: Any, out: set[str] | None = None) -> set[str]:
    """Walk a conditions tree dict and gather every leaf type name."""
    if out is None:
        out = set()
    if not isinstance(node, Mapping):
        return out
    t = node.get("type")
    if t in {"all_of", "any_of", "none_of"}:
        for child in node.get("of") or []:
            _collect_tree_leaf_types(child, out)
    elif isinstance(t, str):
        out.add(t)
    return out


def provider_for(edition: str) -> Provider:
    """Return the default provider for an edition."""
    if edition == "java":
        return JavaPythonProvider()
    if edition == "bedrock":
        return BedrockProvider()
    raise ValueError(f"unknown edition {edition!r}")


def run_staged_search(
    spec: SearchSpec,
    provider: Optional[Provider] = None,
    job_id: Optional[str] = None,
    progress_interval: float = 2.0,
    cancel_callback: Optional[Callable[[], bool]] = None,
    pause_callback: Optional[Callable[[], bool]] = None,
) -> Iterator[SearchEvent]:
    """Run a local staged search and yield product events.

    The stream mirrors the future Tauri event contract: lifecycle, progress,
    candidate, verified match, rejected-stage summary, and completion.
    """
    active_provider = provider or provider_for(spec.edition)
    active_provider.validate_spec(spec)
    criteria = compile_criteria(
        spec.criteria,
        biome_version=spec.version,
        biome_dimension=spec.dimension,
    )

    jid = job_id or str(uuid.uuid4())
    start = time.monotonic()
    last_progress = start
    scanned = 0
    matches = 0
    rejected_by_stage: Dict[int, int] = {}

    yield SearchEvent(
        type="started",
        job_id=jid,
        payload={
            "edition": spec.edition,
            "version": spec.version,
            "dimension": spec.dimension,
            "count": spec.count,
        },
    )

    for chunk in _iter_chunks(spec.to_plan()):
        for seed in chunk:
            if cancel_callback and cancel_callback():
                yield SearchEvent(
                    type="cancelled",
                    job_id=jid,
                    payload={
                        "scanned": scanned,
                        "matches": matches,
                        "rejected_by_stage": dict(rejected_by_stage),
                    },
                )
                return
            while pause_callback and pause_callback():
                if cancel_callback and cancel_callback():
                    yield SearchEvent(
                        type="cancelled",
                        job_id=jid,
                        payload={
                            "scanned": scanned,
                            "matches": matches,
                            "rejected_by_stage": dict(rejected_by_stage),
                        },
                    )
                    return
                time.sleep(0.05)
            scanned += 1
            matched, failed_stage, _ = criteria.evaluate_staged(seed)
            if not matched:
                assert failed_stage is not None
                rejected_by_stage[failed_stage] = rejected_by_stage.get(failed_stage, 0) + 1
                continue

            yield SearchEvent(
                type="candidate",
                job_id=jid,
                payload={"seed": seed, "stage": "criteria"},
            )
            report = active_provider.evaluate_seed(seed, criteria, spec)
            if report is None:
                rejected_by_stage[4] = rejected_by_stage.get(4, 0) + 1
                continue

            matches += 1
            yield SearchEvent(
                type="verified_match",
                job_id=jid,
                payload=report.to_dict(),
            )
            if matches >= spec.max_matches:
                break

        now = time.monotonic()
        if (now - last_progress) >= progress_interval:
            elapsed = now - start
            yield SearchEvent(
                type="progress",
                job_id=jid,
                payload={
                    "scanned": scanned,
                    "matches": matches,
                    "rate": scanned / elapsed if elapsed > 0 else 0.0,
                    "rejected_by_stage": dict(rejected_by_stage),
                },
            )
            last_progress = now
        if matches >= spec.max_matches:
            break

    yield SearchEvent(
        type="rejected_summary",
        job_id=jid,
        payload={"rejected_by_stage": dict(rejected_by_stage)},
    )
    yield SearchEvent(
        type="completed",
        job_id=jid,
        payload={
            "scanned": scanned,
            "matches": matches,
            "elapsed_seconds": time.monotonic() - start,
        },
    )


def _exactness_for_spec(
    criteria_spec: Mapping[str, Any], biome_exact: bool
) -> Dict[str, Exactness]:
    # Structure placement is always exact (pure-Rust/Python golden-tested math).
    # Biome filters are exact only when the cubiomes backend was active for the
    # search; otherwise they are approximate candidates.
    biome_level: Exactness = "exact" if biome_exact else "candidate"
    exactness: Dict[str, Exactness] = {}
    if criteria_spec.get("nearby_structures"):
        exactness["structures"] = "exact"
    if criteria_spec.get("spawn_biome") is not None:
        exactness["spawn_biome"] = biome_level
    if criteria_spec.get("nearby_biomes") is not None:
        exactness["nearby_biomes"] = biome_level
    return exactness
