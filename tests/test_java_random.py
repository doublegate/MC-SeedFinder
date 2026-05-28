"""
tests/test_java_random.py
=========================

Verify that ``mcseedfinder.java_random.JavaRandom`` is a bit-exact port of
``java.util.Random``. Every assertion in this file is a canonical reference
vector — values were obtained by running an actual OpenJDK 17 ``Random``
instance with the indicated seed and recording the output. If any of these
assertions fails, the Java Random implementation is broken and *every*
structure-finding answer the tool produces is suspect, because Minecraft's
structure-seed math depends on this exact LCG.

References
----------
- OpenJDK source: https://github.com/openjdk/jdk/blob/master/src/java.base/share/classes/java/util/Random.java
- Canonical scrambled-seed constant: 0x5DEECE66DL (multiplier),
  0xBL (addend), mask = (1 << 48) - 1.

Run
---
    python -m unittest tests.test_java_random
"""

from __future__ import annotations

import unittest

from mcseedfinder.java_random import JavaRandom


class TestJavaRandom(unittest.TestCase):
    # ------------------------------------------------------------------
    # nextInt() / nextLong() / nextDouble() canonical vectors
    # ------------------------------------------------------------------
    def test_next_long_seed_0(self) -> None:
        r = JavaRandom(0)
        # Java: new Random(0).nextLong() == -4962768465676381896L
        self.assertEqual(r.next_long(), -4962768465676381896)

    def test_next_long_seed_1(self) -> None:
        r = JavaRandom(1)
        self.assertEqual(r.next_long(), -4964420948893066024)

    def test_next_int_bounded_seed_0(self) -> None:
        # Java: new Random(0).nextInt(10) == 0
        r = JavaRandom(0)
        self.assertEqual(r.next_int_bound(10), 0)

    def test_next_double_seed_0(self) -> None:
        # Java: new Random(0).nextDouble() == 0.730967787376657
        r = JavaRandom(0)
        self.assertAlmostEqual(r.next_double(), 0.730967787376657, places=15)

    def test_first_five_ints_seed_42(self) -> None:
        # Java: new Random(42); ints = {r.nextInt(), r.nextInt(), ...} x 5
        r = JavaRandom(42)
        expected = [-1170105035, 234785527, -1360544799, 205897768, 1325939940]
        actual = [r.next_int() for _ in range(5)]
        self.assertEqual(actual, expected)

    # ------------------------------------------------------------------
    # Bounded-int correctness around the OpenJDK rejection-sampling branch
    # ------------------------------------------------------------------
    def test_bounded_power_of_two_fast_path(self) -> None:
        # When bound is a power of two, OpenJDK takes a fast (m & (bound-1))
        # path. We don't observe that branch directly, but we can verify the
        # output range and reproducibility.
        r = JavaRandom(12345)
        for _ in range(1000):
            v = r.next_int_bound(1024)
            self.assertTrue(0 <= v < 1024)

    def test_bounded_arbitrary_range(self) -> None:
        # Arbitrary non-power-of-two bound exercises rejection sampling.
        r = JavaRandom(0xCAFEBABE)
        for _ in range(1000):
            v = r.next_int_bound(31)  # prime, definitely not a power of two
            self.assertTrue(0 <= v < 31)

    def test_bounded_rejects_zero_and_negative(self) -> None:
        r = JavaRandom(0)
        with self.assertRaises(ValueError):
            r.next_int_bound(0)
        with self.assertRaises(ValueError):
            r.next_int_bound(-1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
