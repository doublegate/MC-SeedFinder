"""
Micro-benchmark the Python and optional Rust structure search paths.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Mapping, Sequence

from .criteria import compile_criteria
from .rust_backend import (
    compile_structure_only_requirements,
    find_structure_matches_range,
    is_available as rust_backend_available,
)


DEFAULT_SPEC: Mapping[str, Any] = {
    "nearby_structures": [
        {"structure": "village", "max_distance": 100},
    ],
}


def run_benchmark(
    count: int,
    start_seed: int = 0,
    criteria_spec: Mapping[str, Any] = DEFAULT_SPEC,
) -> dict[str, Any]:
    """Return throughput metrics for Python and Rust-backed structure search."""
    criteria = compile_criteria(criteria_spec)

    py_start = time.perf_counter()
    py_matches = [
        seed
        for seed in range(start_seed, start_seed + count)
        if criteria.matches(seed)
    ]
    py_elapsed = time.perf_counter() - py_start

    result: dict[str, Any] = {
        "count": count,
        "start_seed": start_seed,
        "python": {
            "elapsed_seconds": py_elapsed,
            "seeds_per_second": count / py_elapsed if py_elapsed > 0 else 0.0,
            "matches": len(py_matches),
        },
        "rust": None,
        "speedup": None,
    }

    requirements = compile_structure_only_requirements(criteria_spec)
    if requirements and rust_backend_available():
        rust_start = time.perf_counter()
        rust_matches = find_structure_matches_range(start_seed, count, requirements)
        rust_elapsed = time.perf_counter() - rust_start
        result["rust"] = {
            "elapsed_seconds": rust_elapsed,
            "seeds_per_second": count / rust_elapsed if rust_elapsed > 0 else 0.0,
            "matches": len(rust_matches),
            "parity": rust_matches == py_matches,
        }
        result["speedup"] = (
            py_elapsed / rust_elapsed if rust_elapsed > 0 else None
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mcseedfinder.benchmark",
        description="Benchmark Python vs optional Rust-backed structure search.",
    )
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--start", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run_benchmark(args.count, args.start), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
