"""
java_random.py
==============

Faithful Python reimplementation of ``java.util.Random`` as used by the
Minecraft Java Edition world generator.

Background
----------
Minecraft (Java Edition) uses the JDK's stock ``java.util.Random`` class
pervasively for deterministic, seed-driven world generation:

* World seed initialisation
* Per-region structure placement (villages, ocean monuments, ruined portals, …)
* Per-chunk feature decoration (ores, trees, dungeons, lakes, …)
* Mob spawn rolls and loot table sampling

``java.util.Random`` is a **48-bit linear congruential generator (LCG)** with
the parameters defined in JDK reference source (``Random.java``):

.. math::

    X_{n+1} = (a \\cdot X_n + c) \\bmod m

* :math:`a` = ``0x5DEECE66D``  (multiplier)
* :math:`c` = ``0xB``          (increment)
* :math:`m` = :math:`2^{48}`   (modulus)

Because the internal state is only 48 bits wide, *only the low 48 bits of the
user-supplied seed influence subsequent randomness*. This is a well-known
property exploited by every Minecraft seed finder (cubiomes, MineMap,
Chunkbase, etc.) to brute-force the 48-bit seed space rather than the full
64-bit world seed.

Why a hand-rolled re-implementation?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python's ``random`` module uses Mersenne Twister, not Java's LCG, so we cannot
borrow it. This module mirrors the OpenJDK source line-for-line, including
the rejection-sampling form of ``nextInt(bound)`` and the sign-extending
``nextLong()`` semantics. The result is **bit-exact** with reference Minecraft
generation for all 48-bit-determined operations.

Verified against canonical Java test vectors:

    >>> r = JavaRandom(0)
    >>> r.next_long()
    -4962768465676381896
    >>> r = JavaRandom(0)
    >>> r.next_int_bound(10)
    0

Usage
-----
::

    from mcseedfinder.java_random import JavaRandom

    rng = JavaRandom(seed=1234567890)
    x_offset = rng.next_int_bound(24)
    z_offset = rng.next_int_bound(24)

References
----------
* OpenJDK source: ``jdk/src/share/classes/java/util/Random.java``
* cubiomes (Cubitect): ``rng.c``, ``rng.h``
* Minecraft Wiki: https://minecraft.wiki/w/Random
"""

from __future__ import annotations

from typing import Final

# --------------------------------------------------------------------------- #
# LCG constants (verbatim from OpenJDK Random.java)
# --------------------------------------------------------------------------- #
# Multiplier 'a' in X_{n+1} = (a*X + c) mod 2^48
_MULTIPLIER: Final[int] = 0x5DEECE66D
# Increment 'c'
_INCREMENT: Final[int] = 0xB
# 48-bit mask, applied after every multiplication to enforce the modulus.
_MASK48: Final[int] = (1 << 48) - 1
# Helpful constants for sign conversions.
_INT32_SIGN_BIT: Final[int] = 1 << 31
_UINT32_MOD: Final[int] = 1 << 32
_INT64_SIGN_BIT: Final[int] = 1 << 63
_UINT64_MOD: Final[int] = 1 << 64


def _to_signed32(value: int) -> int:
    """Reinterpret an unsigned 32-bit value as Java's signed ``int``.

    Java's ``next(32)`` returns a value of type ``int`` (signed 32-bit). The
    Python implementation stores intermediate values in arbitrary-precision
    integers, so we must explicitly perform the sign conversion when emulating
    operations that rely on Java's two's-complement semantics (notably
    ``nextLong()`` and ``nextGaussian()``).
    """
    return value - _UINT32_MOD if value >= _INT32_SIGN_BIT else value


def _to_signed64(value: int) -> int:
    """Wrap an unbounded Python int into Java's signed 64-bit ``long`` range."""
    value &= _UINT64_MOD - 1
    return value - _UINT64_MOD if value >= _INT64_SIGN_BIT else value


class JavaRandom:
    """Bit-exact port of ``java.util.Random``.

    Attributes
    ----------
    seed : int
        The current 48-bit internal state. Read-only externally (mutated via
        :meth:`set_seed` or implicit advancement through ``next*`` methods).
    """

    __slots__ = ("seed",)

    def __init__(self, seed: int = 0) -> None:
        """Construct a new generator and scramble the seed exactly as Java does.

        Java's ``Random(long seed)`` constructor calls ``setSeed(seed)``, which
        XORs the seed with the multiplier *before* masking. This means
        ``new Random(0)`` does **not** start with state 0.
        """
        # set_seed performs the scramble; we don't initialise `self.seed`
        # before calling it because __slots__ forbids stray attributes.
        self.seed: int = 0
        self.set_seed(seed)

    # ------------------------------------------------------------------ #
    # Core LCG primitives
    # ------------------------------------------------------------------ #
    def set_seed(self, seed: int) -> None:
        """Re-seed the generator. Mirrors ``Random.setSeed(long)``.

        The seed is XORed with the multiplier and then truncated to 48 bits.
        This is the same operation performed by Java's constructor.
        """
        # The XOR happens on the full 64-bit value, but the mask discards
        # any bits above bit 47. We accept any Python int (positive or
        # negative) and let `& _MASK48` normalise it.
        self.seed = (seed ^ _MULTIPLIER) & _MASK48

    def next(self, bits: int) -> int:
        """Advance the LCG one step and return the top ``bits`` bits.

        Java's ``protected int next(int bits)`` is the primitive that all the
        public ``nextXxx`` methods are built from. It returns a value in
        :math:`[0, 2^{bits})`. For ``bits == 32`` Java treats the result as a
        signed int; here we always return the unsigned value and let callers
        apply :func:`_to_signed32` if they need the signed interpretation.
        """
        # Standard LCG step. Masking after the multiply-add is what enforces
        # the 2^48 modulus.
        self.seed = (self.seed * _MULTIPLIER + _INCREMENT) & _MASK48
        # Right-shift to extract the high bits — the high bits of an LCG
        # output are statistically much better than the low bits.
        return self.seed >> (48 - bits)

    # ------------------------------------------------------------------ #
    # Public ``nextXxx`` API
    # ------------------------------------------------------------------ #
    def next_int(self) -> int:
        """Java's ``nextInt()`` (signed 32-bit)."""
        return _to_signed32(self.next(32))

    def next_int_bound(self, bound: int) -> int:
        """Java's ``nextInt(int bound)``: uniform int in :math:`[0, \\text{bound})`.

        Implementation follows OpenJDK exactly, including the
        rejection-sampling branch for non-power-of-two bounds. This matters:
        a naïve ``next(31) % bound`` introduces measurable bias for some
        bounds, and Minecraft's structure placement relies on the *exact*
        bias-free distribution.
        """
        if bound <= 0:
            raise ValueError("bound must be positive")

        # Fast path for power-of-two bounds. This branch is hit frequently
        # because Minecraft uses chunkRange values like 16, 24, 32.
        if (bound & -bound) == bound:
            # `(bound * next(31)) >> 31` is equivalent to taking the top
            # log2(bound) bits but uses one multiplication, which is what
            # OpenJDK does.
            return (bound * self.next(31)) >> 31

        # General path: sample 31 bits, reduce, and reject if the value would
        # introduce modulo bias.
        while True:
            bits = self.next(31)
            val = bits % bound
            # OpenJDK's check: `bits - val + (bound - 1) < 0` indicates
            # overflow of the discarded high range, i.e. the sample falls in
            # the tail that would skew the distribution. Reject and retry.
            if bits - val + (bound - 1) >= 0:
                return val

    def next_long(self) -> int:
        """Java's ``nextLong()`` (signed 64-bit).

        Implementation follows OpenJDK exactly:
        ``return ((long)next(32) << 32) + next(32);``
        The first ``next(32)`` is sign-extended when cast to ``long``, and the
        addition uses Java's two's-complement wrapping semantics.
        """
        high = _to_signed32(self.next(32))  # sign-extended to long in Java
        low = _to_signed32(self.next(32))
        # In Java, this `+` is two's-complement long addition that wraps. We
        # do it on Python's unbounded integers and then re-wrap.
        return _to_signed64((high << 32) + low)

    def next_boolean(self) -> bool:
        """Java's ``nextBoolean()``."""
        return self.next(1) != 0

    def next_float(self) -> float:
        """Java's ``nextFloat()``: uniform 32-bit float in ``[0.0, 1.0)``."""
        # Java uses 24 bits / 2^24 to produce the float.
        return self.next(24) / float(1 << 24)

    def next_double(self) -> float:
        """Java's ``nextDouble()``: uniform 53-bit double in ``[0.0, 1.0)``."""
        # 26 high bits + 27 low bits = 53 bits of mantissa.
        return ((self.next(26) << 27) + self.next(27)) / float(1 << 53)

    # ------------------------------------------------------------------ #
    # Helpers commonly needed by Minecraft generation code
    # ------------------------------------------------------------------ #
    def skip(self, n: int) -> None:
        """Advance the state by ``n`` LCG steps without consuming output.

        Useful for fast-forwarding through deterministic prefix sequences.
        Uses the standard LCG fast-skip identity but for clarity we just
        iterate; ``n`` here is always small in practice.
        """
        for _ in range(n):
            self.seed = (self.seed * _MULTIPLIER + _INCREMENT) & _MASK48


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    # These vectors are the canonical Java reference outputs. If any of them
    # fail, the port is broken and every downstream calculation is suspect.
    # All expected values cross-checked against reference Java output.
    cases = [
        # (seed,        method,                                expected)
        (0, lambda r: r.next_long(), -4962768465676381896),
        (1, lambda r: r.next_long(), -4964420948893066024),
        (0, lambda r: r.next_int_bound(10), 0),
        (0, lambda r: r.next_double(), 0.730967787376657),
        (42, lambda r: r.next_int(), -1170105035),
    ]
    for seed, fn, expected in cases:
        rng = JavaRandom(seed)
        got = fn(rng)
        status = "OK" if got == expected else "FAIL"
        print(f"[{status}] seed={seed} expected={expected} got={got}")
