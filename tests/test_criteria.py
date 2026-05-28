"""
tests/test_criteria.py
======================

Smoke tests for the criteria-compilation pipeline:

* ``compile_criteria`` accepts the documented JSON spec shape.
* Compiled criteria evaluate to True/False correctly for known seeds.
* The cost-ordering invariant holds (cheap criteria are evaluated first
  inside ``CriteriaSet.matches``).

Run
---
    python -m unittest tests.test_criteria
"""

from __future__ import annotations

import unittest

from mcseedfinder.criteria import compile_criteria


class TestCriteriaCompilation(unittest.TestCase):
    def test_structure_only_spec(self) -> None:
        cs = compile_criteria({
            "nearby_structures": [
                {"structure": "village", "max_distance": 1000},
            ],
        })
        self.assertEqual(len(cs.criteria), 1)
        # Seed 1 has a village within 322 blocks of origin — must match.
        self.assertTrue(cs.matches(1))

    def test_structure_distance_negative(self) -> None:
        cs = compile_criteria({
            "nearby_structures": [
                {"structure": "village", "max_distance": 100},  # tight
            ],
        })
        # Closest village to origin at seed=1 is ~322 blocks → no match.
        self.assertFalse(cs.matches(1))

    def test_cost_ordering(self) -> None:
        # Mixing biome (expensive) and structure (cheap) criteria must
        # leave the structure first in the evaluation order.
        cs = compile_criteria({
            "spawn_biome": "plains",
            "nearby_structures": [
                {"structure": "village", "max_distance": 1000},
            ],
        })
        costs = [c.cost for c in cs.criteria]
        self.assertEqual(costs, sorted(costs))

    def test_unknown_structure_rejected(self) -> None:
        with self.assertRaises((KeyError, ValueError)):
            compile_criteria({
                "nearby_structures": [
                    {"structure": "totally_not_real", "max_distance": 1000},
                ],
            })


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
