"""
finder.py
=========

Parallel seed-search engine.

The finder takes a :class:`~mcseedfinder.criteria.CriteriaSet` and a search
plan (seed range / random sampling / explicit seed list) and returns a stream
of matching seeds. Search is parallelised with :mod:`multiprocessing`, which
sidesteps the GIL — a single Python worker can check ~1k–10k seeds/sec
depending on which criteria are involved, and N workers give a near-linear
speedup until you saturate memory bandwidth.

Search modes
------------
* **Sequential range**: ``[start, start+count)``. Best for systematic sweeps,
  reproducible results, and resuming after interruption.
* **Random sampling**: draw seeds uniformly from the 64-bit space. Best when
  you don't care about coverage and want any matching seed quickly.
* **Explicit list**: check a user-supplied iterable of candidate seeds.
  Useful when post-processing a coarse pre-filter from another tool.

Output
------
Matches are yielded as ``(seed, elapsed_seconds)`` tuples in the order they
are found (not seed order — workers complete out-of-order). The finder also
periodically reports progress to stderr or a callback.

Design notes
------------
* **No shared state in the hot loop.** Workers each compile their own copy
  of the criteria from a pickled spec; the only IPC traffic is the
  ``(seed_chunk_start, seed_chunk_count)`` work item and the rare match
  event. This keeps the multiprocessing overhead negligible.

* **Chunked work distribution.** Workers receive ranges of seeds (default
  4096 per chunk) rather than one seed at a time, amortising IPC cost.

* **Cooperative cancellation.** A shared ``stop`` flag lets the parent
  signal workers to wind down when ``max_matches`` has been hit.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, List, Mapping, Optional

from .criteria import CriteriaSet, compile_criteria
from .rust_backend import (
    RustStructureRequirement,
    compile_structure_only_requirements,
    find_structure_matches_range,
    is_available as rust_backend_available,
)


# --------------------------------------------------------------------------- #
# Configuration dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class SearchPlan:
    """Describes *how* the finder enumerates candidate seeds."""

    mode: str = "sequential"            # "sequential" | "random" | "explicit"
    start_seed: int = 0
    count: int = 1_000_000
    explicit_seeds: Optional[Iterable[int]] = None
    chunk_size: int = 4096              # seeds per work unit
    random_seed: Optional[int] = None   # for reproducible random sampling


@dataclass
class SearchConfig:
    """Top-level search configuration handed to :func:`run_search`."""

    criteria_spec: Mapping[str, Any]    # raw spec dict, pickled to workers
    plan: SearchPlan = field(default_factory=SearchPlan)
    max_matches: int = 10
    workers: int = field(default_factory=lambda: max(1, (os.cpu_count() or 2) - 1))
    progress_interval: float = 2.0      # seconds between progress callbacks
    # Enable the exact cubiomes biome backend in each worker. None keeps the
    # approximate fallback (e.g. for pure-structure searches or no-cubiomes builds).
    biome_version: Optional[str] = None
    biome_dimension: str = "overworld"


@dataclass
class Match:
    """A successful seed match plus light metadata for the report."""

    seed: int
    found_at_seconds: float


# --------------------------------------------------------------------------- #
# Worker entry point
# --------------------------------------------------------------------------- #
# These globals live *inside the worker process* and are populated once at
# pool startup so we don't recompile the criteria on every chunk.
_WORKER_CRITERIA: Optional[CriteriaSet] = None
_WORKER_RUST_STRUCTURE_REQUIREMENTS: Optional[List[RustStructureRequirement]] = None


def _worker_init(
    criteria_spec: Mapping[str, Any],
    biome_version: Optional[str] = None,
    biome_dimension: str = "overworld",
) -> None:
    """Pool initialiser — compile the criteria once per worker.

    Each worker builds its own cubiomes backend (PyO3 objects aren't picklable),
    so only the version/dimension strings cross the process boundary.
    """
    global _WORKER_CRITERIA, _WORKER_RUST_STRUCTURE_REQUIREMENTS
    _WORKER_CRITERIA = compile_criteria(
        criteria_spec,
        biome_version=biome_version,
        biome_dimension=biome_dimension,
    )
    _WORKER_RUST_STRUCTURE_REQUIREMENTS = compile_structure_only_requirements(
        criteria_spec
    )


def _check_chunk(seeds: List[int]) -> List[int]:
    """Test every seed in ``seeds`` and return those that match.

    Kept as a free function (not a method) so it pickles cleanly across
    process boundaries on every Python version, including 3.13+.
    """
    assert _WORKER_CRITERIA is not None, "worker not initialised"
    if (
        _WORKER_RUST_STRUCTURE_REQUIREMENTS
        and rust_backend_available()
        and _is_contiguous(seeds)
    ):
        return find_structure_matches_range(
            seeds[0],
            len(seeds),
            _WORKER_RUST_STRUCTURE_REQUIREMENTS,
        )
    return [s for s in seeds if _WORKER_CRITERIA.matches(s)]


def _is_contiguous(seeds: List[int]) -> bool:
    """True when ``seeds`` is a simple sequential range."""
    return bool(seeds) and seeds[-1] - seeds[0] == len(seeds) - 1


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def run_search(
    config: SearchConfig,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> Iterator[Match]:
    """Run the search and yield matches as they're found.

    Parameters
    ----------
    config:
        The full :class:`SearchConfig`.
    progress_callback:
        Optional callable ``(seeds_checked, matches_so_far, elapsed_s) -> None``
        invoked roughly every ``config.progress_interval`` seconds.

    Yields
    ------
    :class:`Match` instances. The generator stops when either ``max_matches``
    is reached or the search space is exhausted.
    """
    # Validate the criteria spec immediately (so a bad spec fails before we
    # spend cycles spinning up workers).
    compile_criteria(config.criteria_spec)

    # ---- Worker pool ----
    ctx = mp.get_context("spawn")  # 'spawn' is the safe default cross-platform
    pool = ctx.Pool(
        processes=config.workers,
        initializer=_worker_init,
        initargs=(dict(config.criteria_spec), config.biome_version, config.biome_dimension),
    )

    start_time = time.monotonic()
    last_progress = start_time
    seeds_checked = 0
    matches_found = 0

    try:
        # Stream of chunks. Each chunk is a list of integer seeds; the pool
        # will feed them to ``_check_chunk`` in any worker, in any order.
        chunk_iter = _iter_chunks(config.plan)

        # ``imap_unordered`` keeps the producer paced (won't queue infinite
        # work) and returns results in completion order, which is what we
        # want for streaming matches as fast as possible.
        result_iter = pool.imap_unordered(_check_chunk, chunk_iter)

        for chunk_matches in result_iter:
            seeds_checked += config.plan.chunk_size  # approximate
            for s in chunk_matches:
                matches_found += 1
                yield Match(
                    seed=s,
                    found_at_seconds=time.monotonic() - start_time,
                )
                if matches_found >= config.max_matches:
                    return

            # Progress callback
            now = time.monotonic()
            if progress_callback and (now - last_progress) >= config.progress_interval:
                progress_callback(seeds_checked, matches_found, now - start_time)
                last_progress = now
    finally:
        # Always tear the pool down — workers don't outlive the search.
        pool.terminate()
        pool.join()


# --------------------------------------------------------------------------- #
# Chunk iterator — produces lists of seeds per the plan
# --------------------------------------------------------------------------- #
def _iter_chunks(plan: SearchPlan) -> Iterator[List[int]]:
    """Yield seed-chunks (lists of ints) according to the plan."""
    if plan.mode == "sequential":
        yield from _seq_chunks(plan.start_seed, plan.count, plan.chunk_size)
    elif plan.mode == "random":
        yield from _random_chunks(plan.count, plan.chunk_size, plan.random_seed)
    elif plan.mode == "explicit":
        assert plan.explicit_seeds is not None
        yield from _explicit_chunks(plan.explicit_seeds, plan.chunk_size)
    else:
        raise ValueError(f"unknown search mode {plan.mode!r}")


def _seq_chunks(start: int, count: int, chunk_size: int) -> Iterator[List[int]]:
    """Walk ``[start, start+count)`` in chunks."""
    end = start + count
    cur = start
    while cur < end:
        nxt = min(cur + chunk_size, end)
        yield list(range(cur, nxt))
        cur = nxt


def _random_chunks(
    count: int, chunk_size: int, rng_seed: Optional[int]
) -> Iterator[List[int]]:
    """Draw ``count`` random seeds uniformly from the signed 64-bit range."""
    rng = random.Random(rng_seed)
    # Use the full signed 64-bit range — same domain Minecraft itself uses.
    lo, hi = -(1 << 63), (1 << 63) - 1
    remaining = count
    while remaining > 0:
        this_chunk = min(chunk_size, remaining)
        yield [rng.randint(lo, hi) for _ in range(this_chunk)]
        remaining -= this_chunk


def _explicit_chunks(seeds: Iterable[int], chunk_size: int) -> Iterator[List[int]]:
    """Re-chunk an arbitrary iterable of seeds."""
    buffer: List[int] = []
    for s in seeds:
        buffer.append(s)
        if len(buffer) >= chunk_size:
            yield buffer
            buffer = []
    if buffer:
        yield buffer
