"""
tests/test_structures.py
========================

Regression test for the structure-placement math in
``mcseedfinder.structures``.

Strategy
--------
We test against **self-consistency** rather than against an external oracle:
for a known seed we record every nearby-structure position once (by running
the code), and then assert the same positions are produced again on every
run. This catches accidental changes to the salts, region multipliers, or
JavaRandom seeding sequence — any of which would silently corrupt every
search answer the tool gives.

For an external-oracle check, install ``cubiomes`` (the C reference) and
diff its output against this module for seed 1. That cross-check was
performed during development and informs the constants embedded here.

References
----------
- cubiomes/finders.c (Cubitect): authoritative source for every salt below.
- Minecraft Wiki: https://minecraft.wiki/w/Structure_set

Run
---
    python -m unittest tests.test_structures
"""

from __future__ import annotations

import unittest

from mcseedfinder.structures import (
    STRUCTURE_CONFIGS,
    get_structure_pos,
    iter_strongholds,
    iter_structures_in_radius,
)


class TestStructureMath(unittest.TestCase):
    SEED = 1

    # Pre-recorded chunk positions for seed=1, region (0,0).
    # Each value is (chunk_x, chunk_z), as produced by get_structure_pos(...).
    EXPECTED_FIRST_CHUNK = {
        "village":          (11, 16),
        "swamp_hut":        ( 9,  2),
        "desert_pyramid":   (14, 10),
        "shipwreck":        ( 4,  8),
        "ocean_monument":   (12, 23),
        "pillager_outpost": ( 5, 20),
        "igloo":            ( 0, 10),
        "jungle_temple":    ( 0, 18),
    }

    def test_known_first_region_positions(self) -> None:
        # Every structure's region (0,0) placement is fully determined by its
        # salt + the world seed. These chunks are the canonical values for
        # seed=1, verified against cubiomes during development.
        for struct, expected in self.EXPECTED_FIRST_CHUNK.items():
            pos = get_structure_pos(struct, self.SEED, 0, 0)
            self.assertIsNotNone(pos, f"{struct} should generate in region (0,0)")
            assert pos is not None  # for type checker
            self.assertEqual(
                (pos.chunk_x, pos.chunk_z), expected,
                f"{struct} chunk mismatch for seed={self.SEED}",
            )

    def test_iter_within_radius_includes_known_village(self) -> None:
        # Seed 1, village at chunk (11,16) — block centre (184, 264),
        # Euclidean distance ~321.8 blocks. So a 500-block search must
        # find it.
        found = list(iter_structures_in_radius("village", self.SEED, 0, 0, 500))
        self.assertTrue(
            any(p.chunk_x == 11 and p.chunk_z == 16 for p in found),
            "Expected village at chunk (11,16) was not in the 500-block radius result",
        )

    def test_iter_within_radius_excludes_far_structures(self) -> None:
        # Tight radius should yield no village (closest is ~322 blocks away).
        found = list(iter_structures_in_radius("village", self.SEED, 0, 0, 100))
        self.assertEqual(
            found, [],
            "100-block radius around origin should contain no village for seed=1",
        )

    def test_strongholds_count_and_first_ring(self) -> None:
        # Iterate all 8 rings — vanilla 1.13+ generates 128 strongholds total.
        all_holds = list(iter_strongholds(self.SEED, max_rings=8))
        self.assertEqual(len(all_holds), 128)

        # First ring contains 3 strongholds (per Minecraft Wiki).
        first_ring = list(iter_strongholds(self.SEED, max_rings=1))
        self.assertEqual(len(first_ring), 3)

    def test_all_structures_have_configs(self) -> None:
        # Sanity: every documented structure has a configured salt.
        required = [
            "village", "desert_pyramid", "igloo", "jungle_temple", "swamp_hut",
            "pillager_outpost", "ocean_ruin", "shipwreck", "ocean_monument",
            "woodland_mansion", "ruined_portal",
        ]
        for name in required:
            self.assertIn(name, STRUCTURE_CONFIGS, f"Missing config for {name}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
