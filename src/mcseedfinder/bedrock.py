"""
Bedrock Edition support (foundation).
=====================================

Bedrock Edition differs from Java at the worldgen level (different PRNG,
different biome layers, different structure salts/spacing). A faithful
in-process port is on the roadmap — see ``docs/BEDROCK.md``. Phase 5
ships only the parts that are exact today without a second backend:

* :func:`seed_from_string` — Bedrock (and Java) hash text seeds the same
  way: Java's ``String.hashCode()`` over UTF-16 code units, wrapping into
  a signed 32-bit integer. Bedrock then stores that integer as the world
  seed. This is the #1 Bedrock UX feature for any seed tool.

Anything that requires Bedrock worldgen (biome lookups, structure
placement, stronghold rings) is intentionally **not** implemented here.
:class:`mcseedfinder.engine.BedrockProvider` rejects such criteria with a
clear edition-aware error so callers see exactly which feature is missing.

The native Rust implementation in ``mcseedfinder._native`` is preferred when
available; the pure-Python fallback below produces identical results and
exists so the public ``mcseedfinder.bedrock`` API works even on builds
without the compiled extension.
"""

from __future__ import annotations

from . import rust_backend


def seed_from_string(text: str) -> int:
    """Return the i32 game seed Bedrock stores for a text seed ``text``.

    Algorithm (same as Java ``String.hashCode()``):

    .. code-block:: text

        h = 0
        for unit in text (UTF-16 code units):
            h = 31 * h + unit         # wrap as i32

    Examples
    --------
    >>> seed_from_string("")
    0
    >>> seed_from_string("Hello")
    69609650
    >>> seed_from_string("abc")
    96354
    """
    if rust_backend.is_available():
        return rust_backend._native.bedrock_seed_from_string(text)  # type: ignore[union-attr]
    return _python_seed_from_string(text)


def _python_seed_from_string(text: str) -> int:
    """Pure-Python fallback. Identical semantics to the native version."""
    h = 0
    # Java's String is UTF-16 internally; the hashCode walks code units, not
    # Unicode codepoints. ``str.encode("utf-16-le")`` gives the same units.
    encoded = text.encode("utf-16-le")
    for i in range(0, len(encoded), 2):
        unit = encoded[i] | (encoded[i + 1] << 8)
        h = (h * 31 + unit) & 0xFFFFFFFF
    # Convert the unsigned i32 accumulator back to signed.
    if h >= 0x80000000:
        h -= 0x100000000
    return h


def is_valid_bedrock_seed(value: int) -> bool:
    """True iff ``value`` is within Bedrock's signed i32 seed range.

    Bedrock stores world seeds as i32 (unlike Java's i64). Players can still
    enter any text — :func:`seed_from_string` handles that — but a numeric
    seed must fit in [-2**31, 2**31).
    """
    return -(2**31) <= value < 2**31
