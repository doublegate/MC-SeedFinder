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

from mcseedfinder import rust_backend
from mcseedfinder.structures import (
    STRUCTURE_CONFIGS,
    get_structure_pos,
    iter_buried_treasure_in_radius,
    iter_strongholds,
    iter_structures_in_radius,
    roll_buried_treasure_chunk,
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
        # 1.19.2+ deep-dark city: salt 20083232, spread 16.
        "ancient_city":     ( 9, 11),
        # 1.21+ trial chambers: salt 94251327, spread 22.
        "trial_chambers":   (18, 14),
        # buried_treasure deliberately omitted — it uses the per-chunk roll
        # path (no region-grid placement). See `TestBuriedTreasureRoll`.
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


class TestBuriedTreasureRoll(unittest.TestCase):
    """Per-chunk roll placement for buried_treasure (cubiomes `case Treasure`)."""

    def test_python_matches_rust_native_for_seed_1(self) -> None:
        """Bit-exact parity between the Python and Rust implementations of
        the buried_treasure roll. The Rust side is in turn parity-checked
        against cubiomes in its own test suite, so this transitively ties
        Python to cubiomes."""
        if not rust_backend.is_available():
            self.skipTest("native extension not built")
        from mcseedfinder._native import roll_buried_treasure_chunk_py
        for cx in range(-16, 17):
            for cz in range(-16, 17):
                ours = roll_buried_treasure_chunk(1, cx, cz)
                native = roll_buried_treasure_chunk_py(1, cx, cz)
                self.assertEqual(
                    ours, native,
                    f"Python/Rust roll disagree at chunk ({cx}, {cz}) for seed=1"
                )

    def test_rate_is_about_one_percent(self) -> None:
        """~1% chance per chunk → expect 2..30 hits in a 33×33 grid (Poisson
        std ~3.3). Wide band so it doesn't flake; catches regressions where
        the salt math or float threshold drift drastically."""
        hits = sum(
            1
            for cx in range(-16, 17)
            for cz in range(-16, 17)
            if roll_buried_treasure_chunk(1, cx, cz)
        )
        self.assertTrue(
            2 <= hits <= 30,
            f"implausible buried_treasure rate: {hits}/1089 for seed=1",
        )

    def test_iter_yields_anchors_inside_radius(self) -> None:
        """Every yielded position must lie within the requested block
        radius of the centre — guards against bounding-box bugs at the
        circle's edge."""
        radius = 2000
        results = list(iter_buried_treasure_in_radius(1, 0, 0, radius))
        for pos in results:
            anchor_x = pos.chunk_x * 16 + 9  # buried_treasure uses +9, not +8
            anchor_z = pos.chunk_z * 16 + 9
            self.assertLessEqual(
                anchor_x * anchor_x + anchor_z * anchor_z, radius * radius,
                f"buried_treasure at ({anchor_x},{anchor_z}) outside {radius}-block radius"
            )

    def test_iter_in_radius_dispatches_through_structures_helper(self) -> None:
        """``iter_structures_in_radius('buried_treasure', ...)`` must route
        to the per-chunk path — not error or fall through to the region
        framework that doesn't model 1%-per-chunk placement."""
        direct = sorted(
            (p.chunk_x, p.chunk_z)
            for p in iter_buried_treasure_in_radius(1, 0, 0, 2000)
        )
        dispatched = sorted(
            (p.chunk_x, p.chunk_z)
            for p in iter_structures_in_radius("buried_treasure", 1, 0, 0, 2000)
        )
        self.assertEqual(direct, dispatched)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
