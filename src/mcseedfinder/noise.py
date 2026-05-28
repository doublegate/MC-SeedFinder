"""
noise.py
========

Lightweight gradient-noise primitives used by the *approximate* biome
generator (see :mod:`mcseedfinder.biome_gen`).

Disclaimer
----------
**This is not a port of Minecraft's actual noise pipeline.** Minecraft 1.18+
uses a six-channel climate-noise stack (``temperature``, ``humidity``,
``continentalness``, ``erosion``, ``depth``, ``weirdness``) blended via
spline interpolation onto a hand-tuned biome lookup, plus density-noise
terrain shaping. Faithfully reproducing that is the entire purpose of
``cubiomes`` (~10k lines of C). What we provide here is a deterministic,
seed-derived noise field with broadly the *right shape* — enough for the
finder to produce plausible biome neighbourhoods, but not to predict the
exact in-game map for a given seed.

If you need real-game accuracy, install ``cubiomes-py`` and the biome
generator (:mod:`mcseedfinder.biome_gen`) will use it automatically.

What this module gives you
--------------------------
* :class:`PerlinNoise` — a single-octave 2-D Perlin noise field, deterministic
  in its 64-bit seed and using a Java-style RNG to shuffle its permutation
  table. This makes the noise reproducible across machines and Python
  versions (important for criteria that should match exactly when re-run).
* :class:`OctaveNoise` — a stack of :class:`PerlinNoise` instances summed
  with the classic ``persistence``/``lacunarity`` parameters. Used to give
  the climate fields multi-scale features.

References
----------
* Ken Perlin (2002): *Improving Noise*, SIGGRAPH.
* Minecraft Wiki: https://minecraft.wiki/w/Noise — describes the actual MC
  noise stack we are *approximating*.
"""

from __future__ import annotations

import math
from typing import List

from .java_random import JavaRandom


# --------------------------------------------------------------------------- #
# Permutation-table Perlin noise
# --------------------------------------------------------------------------- #
class PerlinNoise:
    """Single-octave 2-D Perlin noise with a Java-RNG-shuffled permutation.

    Using :class:`~mcseedfinder.java_random.JavaRandom` for the shuffle means
    two things:

    1. The noise field is bit-exactly reproducible across implementations
       that share this LCG (cubiomes, MineMap, our finder, …).
    2. Different world seeds give meaningfully different fields, exactly the
       property we want for a finder.

    Output range is approximately ``[-1, 1]`` (the theoretical bound for
    2-D Perlin is ``±sqrt(2)/2 ≈ ±0.707`` but we don't normalise; downstream
    code only cares about the qualitative landscape).
    """

    __slots__ = ("perm", "ox", "oy")

    def __init__(self, seed: int):
        """Build the permutation table and the random offset for this seed."""
        # We treat the world seed as the LCG seed for a Java-style shuffle.
        rng = JavaRandom(seed)

        # Per-instance origin offset: keeps adjacent seeds from producing
        # visually-similar fields (a known issue with naive Perlin seeding).
        # next_double() gives a uniform [0,1) which we scale to a useful range.
        self.ox: float = rng.next_double() * 256.0
        self.oy: float = rng.next_double() * 256.0

        # Classic Perlin 256-entry permutation, doubled for wrap-free lookup.
        # We Fisher–Yates shuffle [0..255] using the Java RNG so that the
        # permutation is fully determined by the seed.
        base: List[int] = list(range(256))
        # Java-style Fisher–Yates: iterate top-down, swap with random earlier.
        for i in range(255, 0, -1):
            j = rng.next_int_bound(i + 1)
            base[i], base[j] = base[j], base[i]
        # Double for cheap modulo: perm[i + 256] == perm[i].
        self.perm: List[int] = base + base

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fade(t: float) -> float:
        """6t^5 − 15t^4 + 10t^3 — Perlin's improved smoothstep."""
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    @staticmethod
    def _grad(hash_: int, x: float, y: float) -> float:
        """One of 8 gradient directions selected by the low bits of `hash_`.

        Returns the dot product of the chosen gradient with ``(x, y)``.
        """
        # Use 3 bits → 8 directions. This is the standard table used in
        # Ken Perlin's reference 2002 implementation collapsed to 2D.
        h = hash_ & 7
        u = x if h < 4 else y
        v = y if h < 4 else x
        # Sign bits drawn from the next two bits of the hash.
        return (-u if (h & 1) else u) + (-2.0 * v if (h & 2) else 2.0 * v)

    # ------------------------------------------------------------------ #
    # Public sampler
    # ------------------------------------------------------------------ #
    def sample(self, x: float, y: float) -> float:
        """Sample the noise field at world coordinates ``(x, y)``."""
        # Apply the per-instance origin offset so neighbouring seeds diverge.
        x += self.ox
        y += self.oy

        # Integer lattice cell containing the point.
        xi = math.floor(x)
        yi = math.floor(y)
        # Fractional part within the cell.
        xf = x - xi
        yf = y - yi
        # Mask to the doubled-perm range.
        xi &= 255
        yi &= 255

        # Eased fractions for smooth blending between corner gradients.
        u = self._fade(xf)
        v = self._fade(yf)

        # Hash the four corners of the cell.
        aa = self.perm[self.perm[xi] + yi]
        ab = self.perm[self.perm[xi] + yi + 1]
        ba = self.perm[self.perm[xi + 1] + yi]
        bb = self.perm[self.perm[xi + 1] + yi + 1]

        # Bilinear blend of the four corner gradients.
        x1 = _lerp(u, self._grad(aa, xf, yf), self._grad(ba, xf - 1.0, yf))
        x2 = _lerp(u, self._grad(ab, xf, yf - 1.0), self._grad(bb, xf - 1.0, yf - 1.0))
        return _lerp(v, x1, x2)


def _lerp(t: float, a: float, b: float) -> float:
    """Standard linear interpolation."""
    return a + t * (b - a)


# --------------------------------------------------------------------------- #
# Octave noise (fractal Brownian motion)
# --------------------------------------------------------------------------- #
class OctaveNoise:
    """Stack of Perlin octaves summed with persistence/lacunarity.

    This is the standard fBm construction: each octave doubles in frequency
    (via ``lacunarity``) and halves in amplitude (via ``persistence``). The
    sum is normalised so the output stays roughly in ``[-1, 1]``.
    """

    __slots__ = ("octaves", "persistence", "lacunarity", "_amplitude_sum")

    def __init__(
        self,
        seed: int,
        octaves: int = 4,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
    ) -> None:
        # Derive a distinct sub-seed per octave so they don't collude.
        # next_long() gives 64 bits of entropy per octave, plenty.
        rng = JavaRandom(seed)
        self.octaves: List[PerlinNoise] = [
            PerlinNoise(rng.next_long()) for _ in range(octaves)
        ]
        self.persistence: float = persistence
        self.lacunarity: float = lacunarity

        # Pre-compute the total amplitude so we can normalise on each sample.
        amp = 1.0
        total = 0.0
        for _ in range(octaves):
            total += amp
            amp *= persistence
        self._amplitude_sum: float = total

    def sample(self, x: float, y: float, base_frequency: float = 1.0) -> float:
        """Sample the summed octave stack at ``(x, y)``.

        ``base_frequency`` scales the input coordinates before any octave
        ramp-up, so callers can sample the same noise at world scale
        (low frequency, e.g. 1/512) vs feature scale (higher frequency).
        """
        amplitude = 1.0
        frequency = base_frequency
        total = 0.0
        for octave in self.octaves:
            total += octave.sample(x * frequency, y * frequency) * amplitude
            amplitude *= self.persistence
            frequency *= self.lacunarity
        return total / self._amplitude_sum
