"""
cli.py
======

Command-line interface for the seed finder.

Two ways to specify search criteria
-----------------------------------
1. **Inline flags** for simple searches::

       python -m mcseedfinder \\
           --spawn-biome plains \\
           --nearby-structure village:1500 \\
           --nearby-structure ocean_monument:3000 \\
           --max-matches 5 --workers 8

2. **JSON config file** for anything non-trivial::

       python -m mcseedfinder --config my_criteria.json

Run ``python -m mcseedfinder --help`` for the full reference.

The CLI prints matching seeds to stdout (one per line, integer form so you
can pipe to other tools) and a human-readable report to stderr. ``--quiet``
strips the report.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .biome_gen import BiomeLookup
from .biomes import BIOMES
from .criteria import (
    CriteriaSet,
    compile_criteria,
)
from .finder import Match, SearchConfig, SearchPlan, run_search
from .structures import (
    SUPPORTED_STRUCTURES,
    iter_strongholds,
    iter_structures_in_radius,
)


# --------------------------------------------------------------------------- #
# Argument parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser. Factored out for testability."""
    p = argparse.ArgumentParser(
        prog="mcseedfinder",
        description="Find Minecraft Java Edition world seeds matching given "
                    "criteria. Structure placement is exact; biome filtering is "
                    "exact when the cubiomes backend is built, else approximate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- Criteria source ----
    src = p.add_argument_group("criteria source")
    src.add_argument(
        "--config", type=Path, default=None,
        help="JSON file containing the full criteria specification."
    )
    src.add_argument(
        "--spec", type=Path, default=None,
        help="Versioned SearchSpec JSON file. If it contains a top-level "
             "'criteria' object, that object is used for the search."
    )
    src.add_argument(
        "--spawn-biome", action="append", default=None,
        metavar="BIOME_OR_GROUP",
        help="Require this biome (or group like 'warm_dry') near spawn. "
             "Repeatable; biomes are OR'd."
    )
    src.add_argument(
        "--spawn-radius", type=int, default=64,
        help="Spawn-biome search radius in blocks (default: 64)."
    )
    src.add_argument(
        "--nearby-structure", action="append", default=None,
        metavar="STRUCTURE[:DISTANCE]",
        help=(
            "Require structure within DISTANCE blocks of origin (default "
            "1500). Repeatable. Use 'list' as the structure name to see all."
        ),
    )
    src.add_argument(
        "--nearby-biomes", action="append", default=None,
        metavar="BIOME_OR_GROUP",
        help="Require any of these biomes within --nearby-biomes-radius."
    )
    src.add_argument(
        "--nearby-biomes-radius", type=int, default=2000,
        help="Radius for --nearby-biomes (default: 2000)."
    )
    src.add_argument(
        "--nearby-biomes-all", action="store_true",
        help="Require ALL biomes in --nearby-biomes (default: any)."
    )

    # ---- Search plan ----
    plan = p.add_argument_group("search plan")
    plan.add_argument(
        "--edition", choices=("java", "bedrock"), default="java",
        help="Minecraft edition to search (default: java). Bedrock is reserved "
             "for the product contract and is not implemented yet."
    )
    plan.add_argument(
        "--version", default="1.21",
        help="Minecraft version for exact biome generation and report/export "
             "labels (default: 1.21). Accepts versions cubiomes knows, e.g. "
             "1.18, 1.20.4, 1.21."
    )
    plan.add_argument(
        "--dimension", choices=("overworld", "nether", "end"), default="overworld",
        help="Dimension for biome lookups (default: overworld)."
    )
    plan.add_argument(
        "--mode", choices=("sequential", "random"), default="sequential",
        help="How to enumerate candidate seeds (default: sequential)."
    )
    plan.add_argument(
        "--start", type=int, default=0,
        help="Sequential mode: first seed to test (default: 0)."
    )
    plan.add_argument(
        "--count", type=int, default=1_000_000,
        help="Number of seeds to test (default: 1,000,000)."
    )
    plan.add_argument(
        "--random-seed", type=int, default=None,
        help="Reproducible RNG seed for random-mode search."
    )

    # ---- Runtime ----
    rt = p.add_argument_group("runtime")
    rt.add_argument(
        "--workers", type=int, default=None,
        help="Worker process count (default: CPU count − 1)."
    )
    rt.add_argument(
        "--max-matches", type=int, default=10,
        help="Stop after this many matches (default: 10)."
    )
    rt.add_argument(
        "--chunk-size", type=int, default=4096,
        help="Seeds per worker chunk (default: 4096)."
    )
    rt.add_argument(
        "--report-each", action="store_true",
        help="Print a human-readable per-match report to stderr."
    )
    rt.add_argument(
        "--quiet", action="store_true",
        help="Suppress the per-match report and progress lines."
    )
    rt.add_argument(
        "--export", type=Path, default=None,
        help="Write matches to a file in addition to canonical stdout."
    )
    rt.add_argument(
        "--export-format", choices=("json", "csv", "plain"), default="json",
        help="Export file format when --export is used (default: json)."
    )

    # ---- Special-mode helpers ----
    info = p.add_argument_group("information")
    info.add_argument(
        "--list-structures", action="store_true",
        help="Print all supported structure names and exit."
    )
    info.add_argument(
        "--list-biomes", action="store_true",
        help="Print all known biome keys and exit."
    )
    info.add_argument(
        "--show-seed", type=int, default=None,
        metavar="SEED",
        help="Skip the search; print a structure/biome report for the given "
             "seed instead. Useful for verifying a candidate.",
    )
    info.add_argument(
        "--seed-string", default=None,
        metavar="TEXT",
        help="Convert a Bedrock-style text seed into the i32 game seed Bedrock "
             "(and Java) stores. Same algorithm as Java's String.hashCode(). "
             "Prints the signed value to stdout and exits.",
    )

    return p


# --------------------------------------------------------------------------- #
# Argument compilation
# --------------------------------------------------------------------------- #
def _parse_structure_spec(spec: str) -> tuple[str, int]:
    """Parse a ``"structure[:distance]"`` CLI argument."""
    if ":" in spec:
        name, dist_str = spec.split(":", 1)
        return name.strip(), int(dist_str)
    return spec.strip(), 1500


def compile_args_to_criteria(args: argparse.Namespace) -> CriteriaSet:
    """Turn parsed CLI args into a :class:`CriteriaSet`.

    ``--config`` takes precedence over inline flags when both are present
    (the assumption being that anyone with a config file knows what they want).
    """
    dimension = getattr(args, "dimension", "overworld")
    if args.spec:
        return compile_criteria(
            _load_criteria_spec(args.spec),
            biome_version=args.version,
            biome_dimension=dimension,
        )
    if args.config:
        return compile_criteria(
            _load_criteria_spec(args.config),
            biome_version=args.version,
            biome_dimension=dimension,
        )

    # Otherwise, build from inline flags.
    structures: list[tuple[str, int]] = []
    if args.nearby_structure:
        for spec in args.nearby_structure:
            structures.append(_parse_structure_spec(spec))

    # Use the spec-dict path so per-structure max_distance values are preserved
    # (the convenience helper applies a single radius to all of them).
    spec: dict[str, Any] = {}
    if args.spawn_biome:
        spec["spawn_biome"] = args.spawn_biome
        spec["spawn_radius"] = args.spawn_radius
    if structures:
        spec["nearby_structures"] = [
            {"structure": s, "max_distance": d} for s, d in structures
        ]
    if args.nearby_biomes:
        spec["nearby_biomes"] = {
            "biomes": args.nearby_biomes,
            "radius": args.nearby_biomes_radius,
            "all": args.nearby_biomes_all,
        }
    return compile_criteria(
        spec, biome_version=args.version, biome_dimension=dimension
    )


def _load_criteria_spec(path: Path) -> Mapping[str, Any]:
    """Load either a raw criteria file or a versioned SearchSpec file."""
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    if "criteria" in spec and isinstance(spec["criteria"], dict):
        return spec["criteria"]
    return spec


# --------------------------------------------------------------------------- #
# Seed verification report
# --------------------------------------------------------------------------- #
def report_seed(
    seed: int,
    out=sys.stdout,
    *,
    version: str = "1.21",
    dimension: str = "overworld",
) -> None:
    """Pretty-print structure positions and a spawn biome for a single seed.

    This is the inverse of the search: given a candidate seed, show what's
    in it so the user can sanity-check before loading the world. Uses the
    exact cubiomes biome backend when available, else the approximation.
    """
    print(f"Seed: {seed}", file=out)
    print("=" * 70, file=out)
    print("Nearest structures from origin:", file=out)
    print("-" * 70, file=out)
    print(f"{'structure':<20s} {'chunk':>14s} {'block':>14s} {'distance':>10s}",
          file=out)
    for structure in sorted(SUPPORTED_STRUCTURES):
        if structure == "stronghold":
            continue
        nearest = None
        for pos in iter_structures_in_radius(structure, seed, 0, 0, 8000):
            if nearest is None or pos.distance_to(0, 0) < nearest.distance_to(0, 0):
                nearest = pos
        if nearest:
            print(
                f"{structure:<20s} "
                f"{('(' + str(nearest.chunk_x) + ',' + str(nearest.chunk_z) + ')'):>14s} "
                f"{('(' + str(nearest.block_x) + ',' + str(nearest.block_z) + ')'):>14s} "
                f"{nearest.distance_to(0, 0):>10.1f}",
                file=out,
            )
    print("-" * 70, file=out)
    print("First stronghold ring:", file=out)
    for sh in iter_strongholds(seed, max_rings=1):
        print(f"  block=({sh.block_x:6d},{sh.block_z:6d}) "
              f"dist={sh.distance_to(0, 0):7.1f}", file=out)
    # Biome at origin — exact via cubiomes when available, else approximate.
    from .rust_backend import make_biome_backend

    backend = make_biome_backend(version, dimension)
    lookup = BiomeLookup(seed, backend=backend)
    info = lookup.biome_info_at(0, 0)
    label = " (approximate)" if lookup.is_approximate else " (exact)"
    print("-" * 70, file=out)
    print(f"Origin biome{label}: {info.namespaced_id}", file=out)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    """Parse args and run the search. Returns a UNIX-style exit code."""
    args = build_parser().parse_args(argv)

    # ---- Version guard rail ----
    # Catch unsupported Minecraft versions early — otherwise the user sees
    # either a hard cubiomes "from_strs" failure later, or silently
    # incorrect biome results if the spec only used structure criteria.
    # Only checked for Java + when cubiomes is in the build (the
    # approximate-biomes / no-cubiomes path accepts any version string).
    if args.edition == "java":
        from .rust_backend import has_cubiomes, is_supported_version

        if has_cubiomes() and not is_supported_version(args.version):
            print(
                f"error: Minecraft version {args.version!r} is not supported by the "
                "bundled cubiomes. Pass --version with a value the bundled cubiomes "
                "recognises (e.g. 1.18, 1.19.2, 1.20.4, 1.21). See "
                "docs/SUPPORTED_VERSIONS.md for the current matrix.",
                file=sys.stderr,
            )
            return 2

    # ---- Information modes ----
    if args.list_structures:
        for s in sorted(SUPPORTED_STRUCTURES):
            print(s)
        return 0
    if args.list_biomes:
        for k in sorted(BIOMES):
            print(k)
        return 0
    if args.show_seed is not None:
        report_seed(args.show_seed, version=args.version, dimension=args.dimension)
        return 0
    if args.seed_string is not None:
        # Bedrock + Java both hash text seeds via Java's String.hashCode().
        from .bedrock import seed_from_string

        print(seed_from_string(args.seed_string))
        return 0

    if args.edition == "bedrock":
        # Bedrock worldgen backend is on the roadmap (docs/BEDROCK.md). The
        # provider can already validate / reject — surface its error here so
        # the user sees exactly what's missing, rather than a generic refusal.
        from .engine import BedrockProvider, SearchSpec

        try:
            criteria_spec = _criteria_set_to_spec_via_args(args)
            BedrockProvider().validate_spec(
                SearchSpec(
                    criteria=criteria_spec,
                    edition="bedrock",
                    version=args.version,
                    dimension=args.dimension,
                    count=args.count,
                    start_seed=args.start,
                    max_matches=args.max_matches,
                )
            )
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        # Shouldn't be reachable today (every criterion is unsupported), but
        # leave the door open for a future Phase 5b that wires Bedrock for
        # specific criteria types.
        print("error: Bedrock search backend not yet implemented", file=sys.stderr)
        return 2

    # ---- Compile criteria ----
    try:
        criteria = compile_args_to_criteria(args)
    except (KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if not args.quiet:
        print("Active criteria (in evaluation order):", file=sys.stderr)
        for line in criteria.describe():
            print(f"  - {line}", file=sys.stderr)

    # ---- Build search plan ----
    plan = SearchPlan(
        mode=args.mode,
        start_seed=args.start,
        count=args.count,
        chunk_size=args.chunk_size,
        random_seed=args.random_seed,
    )
    # Recompile the spec just for serialisation to workers — the worker
    # processes don't share the parent's compiled CriteriaSet.
    spec = _criteria_set_to_spec_via_args(args)
    # If the user didn't pass --workers, let SearchConfig's default_factory
    # decide (CPU count - 1). Otherwise honour the explicit value.
    cfg_kwargs: dict = {
        "criteria_spec": spec,
        "plan": plan,
        "max_matches": args.max_matches,
        "biome_version": args.version,
        "biome_dimension": args.dimension,
    }
    if args.workers is not None:
        cfg_kwargs["workers"] = max(1, args.workers)
    cfg = SearchConfig(**cfg_kwargs)

    # ---- Progress reporter ----
    def progress(seeds_checked: int, matches: int, elapsed: float) -> None:
        if args.quiet:
            return
        rate = seeds_checked / elapsed if elapsed > 0 else 0
        print(f"[progress] checked={seeds_checked:>10d} matches={matches:>3d} "
              f"elapsed={elapsed:5.1f}s rate={rate:>8.0f} seeds/s",
              file=sys.stderr)

    # ---- Run ----
    found: list[Match] = []
    for match in run_search(cfg, progress_callback=progress):
        print(match.seed)            # canonical machine-readable output
        sys.stdout.flush()
        found.append(match)
        if args.report_each and not args.quiet:
            print(file=sys.stderr)
            report_seed(match.seed, out=sys.stderr)

    if not args.quiet:
        print(f"\nFound {len(found)} matching seed(s).", file=sys.stderr)
    if args.export:
        _export_matches(
            args.export,
            args.export_format,
            found,
            edition=args.edition,
            version=args.version,
        )
    return 0 if found else 1


def _criteria_set_to_spec_via_args(args: argparse.Namespace) -> Mapping[str, Any]:
    """Rebuild the spec dict used to seed worker processes.

    We rebuild rather than reuse because :class:`CriteriaSet` contains
    non-picklable bits (frozen sets, abstract subclasses) and recompiling
    inside each worker is trivial.
    """
    if args.spec:
        return _load_criteria_spec(args.spec)
    if args.config:
        return _load_criteria_spec(args.config)

    spec: dict[str, Any] = {}
    if args.spawn_biome:
        spec["spawn_biome"] = args.spawn_biome
        spec["spawn_radius"] = args.spawn_radius
    if args.nearby_structure:
        spec["nearby_structures"] = []
        for entry in args.nearby_structure:
            name, dist = _parse_structure_spec(entry)
            spec["nearby_structures"].append(
                {"structure": name, "max_distance": dist}
            )
    if args.nearby_biomes:
        spec["nearby_biomes"] = {
            "biomes": args.nearby_biomes,
            "radius": args.nearby_biomes_radius,
            "all": args.nearby_biomes_all,
        }
    return spec


def _export_matches(
    path: Path,
    fmt: str,
    matches: Sequence[Match],
    *,
    edition: str,
    version: str,
) -> None:
    """Write search matches to a file without changing stdout semantics."""
    if fmt == "plain":
        with open(path, "w", encoding="utf-8") as f:
            for match in matches:
                print(match.seed, file=f)
        return

    rows = [
        {
            "seed": match.seed,
            "edition": edition,
            "version": version,
            "found_at_seconds": match.found_at_seconds,
        }
        for match in matches
    ]
    if fmt == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
            f.write("\n")
        return
    if fmt == "csv":
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["seed", "edition", "version", "found_at_seconds"],
            )
            writer.writeheader()
            writer.writerows(rows)
        return
    raise ValueError(f"unknown export format {fmt!r}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
