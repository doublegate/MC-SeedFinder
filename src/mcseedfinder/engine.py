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

    def validate_spec(self, spec: SearchSpec) -> None:
        if spec.edition != self.edition:
            raise ValueError(f"Java provider cannot run edition {spec.edition!r}")
        if spec.dimension != "overworld":
            raise ValueError("only the overworld dimension is currently supported")

    def evaluate_seed(
        self,
        seed: int,
        criteria: CriteriaSet,
        spec: SearchSpec,
    ) -> SeedReport | None:
        matched, _, passed = criteria.evaluate_staged(seed)
        if not matched:
            return None
        exactness = _exactness_for_spec(spec.criteria)
        warnings: List[str] = []
        if "candidate" in exactness.values():
            warnings.append(
                "biome filters use the local fallback unless cubiomes-backed "
                "lookup is installed"
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
    """Placeholder provider with explicit unsupported status.

    The product contract supports Bedrock, but this Python package does not yet
    contain Bedrock worldgen. Keeping a concrete provider that fails clearly is
    preferable to silently treating Bedrock as Java.
    """

    edition = "bedrock"

    def validate_spec(self, spec: SearchSpec) -> None:
        if spec.edition != self.edition:
            raise ValueError(f"Bedrock provider cannot run edition {spec.edition!r}")
        raise ValueError("Bedrock search is not implemented in this backend yet")

    def evaluate_seed(
        self,
        seed: int,
        criteria: CriteriaSet,
        spec: SearchSpec,
    ) -> SeedReport | None:
        raise NotImplementedError("Bedrock search is not implemented")


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
    criteria = compile_criteria(spec.criteria)

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


def _exactness_for_spec(criteria_spec: Mapping[str, Any]) -> Dict[str, Exactness]:
    exactness: Dict[str, Exactness] = {}
    if criteria_spec.get("nearby_structures"):
        exactness["structures"] = "exact"
    if criteria_spec.get("spawn_biome") is not None:
        exactness["spawn_biome"] = "candidate"
    if criteria_spec.get("nearby_biomes") is not None:
        exactness["nearby_biomes"] = "candidate"
    return exactness
