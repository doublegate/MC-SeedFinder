"""
Tests for the advanced condition system: logic gates, structure clusters,
biome-area, and the recursive condition tree.
"""

from __future__ import annotations

import unittest

from mcseedfinder import rust_backend
from mcseedfinder.criteria import (
    BiomeArea,
    GroupCriterion,
    NearbyStructure,
    StructureCluster,
    compile_criteria,
)

_HAS_CUBIOMES = rust_backend.has_cubiomes()


class TestStructureCluster(unittest.TestCase):
    def test_validates_structures(self) -> None:
        with self.assertRaises(ValueError):
            StructureCluster(structures=("not_a_structure",))
        with self.assertRaises(ValueError):
            StructureCluster(structures=())
        with self.assertRaises(ValueError):
            StructureCluster(structures=("village",), min_count=0)

    def test_min_count_one_behaves_like_nearby(self) -> None:
        # With min_count=1, the cluster reduces to "at least one of these in range".
        cluster = StructureCluster(
            structures=("village",), max_distance=1000, min_count=1
        )
        nearby = NearbyStructure(structure="village", max_distance=1000)
        # Cross-check on a known-matching seed (seed=1 has a village within 1000
        # per the existing rust_backend test vector).
        self.assertEqual(cluster.evaluate(1, None), nearby.evaluate(1, None))

    def test_high_min_count_fails(self) -> None:
        # No seed has hundreds of villages within 1000 blocks of origin.
        cluster = StructureCluster(
            structures=("village",), max_distance=1000, min_count=999
        )
        self.assertFalse(cluster.evaluate(1, None))


class TestGroupCriterion(unittest.TestCase):
    def test_combinator_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            GroupCriterion(combinator="weird", children=[
                NearbyStructure(structure="village", max_distance=10)
            ])

    def test_empty_group_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GroupCriterion(combinator="all_of", children=[])

    def test_all_of_short_circuits_false(self) -> None:
        # NearbyStructure with a tiny radius almost never holds → group fails.
        impossible = NearbyStructure(structure="village", max_distance=1)
        easy = NearbyStructure(structure="village", max_distance=100_000)
        group = GroupCriterion(combinator="all_of", children=[easy, impossible])
        self.assertFalse(group.evaluate(1, None))

    def test_any_of_short_circuits_true(self) -> None:
        impossible = NearbyStructure(structure="village", max_distance=1)
        easy = NearbyStructure(structure="village", max_distance=100_000)
        group = GroupCriterion(combinator="any_of", children=[impossible, easy])
        self.assertTrue(group.evaluate(1, None))

    def test_none_of_inverts(self) -> None:
        easy = NearbyStructure(structure="village", max_distance=100_000)
        group = GroupCriterion(combinator="none_of", children=[easy])
        self.assertFalse(group.evaluate(1, None))


class TestConditionTreeCompiler(unittest.TestCase):
    def test_unknown_type_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compile_criteria({"conditions": {"type": "nope"}})

    def test_depth_cap(self) -> None:
        # Nest deeper than the cap → should raise.
        node = {"type": "nearby_structure", "structure": "village", "max_distance": 1}
        for _ in range(20):
            node = {"type": "all_of", "of": [node]}
        with self.assertRaises(ValueError):
            compile_criteria({"conditions": node})

    def test_tree_with_logic_compiles_and_evaluates(self) -> None:
        criteria = compile_criteria({
            "conditions": {
                "type": "any_of",
                "of": [
                    {"type": "nearby_structure", "structure": "village", "max_distance": 1},
                    {"type": "nearby_structure", "structure": "village", "max_distance": 100_000},
                ],
            }
        })
        self.assertTrue(criteria.matches(1))

    def test_flat_and_tree_compose(self) -> None:
        # Both the flat nearby_structures AND the tree must hold.
        criteria = compile_criteria({
            "nearby_structures": [{"structure": "village", "max_distance": 100_000}],
            "conditions": {
                "type": "all_of",
                "of": [
                    {
                        "type": "cluster",
                        "structures": ["village"],
                        "max_distance": 100_000,
                        "min_count": 1,
                    }
                ],
            },
        })
        self.assertTrue(criteria.matches(1))


@unittest.skipUnless(_HAS_CUBIOMES, "native extension built without cubiomes")
class TestBiomeAreaExact(unittest.TestCase):
    def test_biome_area_finds_ocean(self) -> None:
        # Seed 12345 origin is ocean (ID 0). A 16x16 grid centred on origin
        # within radius 1000 should hit at least 8 ocean samples easily.
        criteria = compile_criteria(
            {
                "conditions": {
                    "type": "biome_area",
                    "biomes": ["ocean", "deep_ocean", "lukewarm_ocean", "cold_ocean"],
                    "centre_x": 0,
                    "centre_z": 0,
                    "radius": 1000,
                    "samples_per_axis": 16,
                    "min_samples": 8,
                }
            },
            biome_version="1.21",
        )
        self.assertTrue(criteria.uses_exact_biomes)
        self.assertTrue(criteria.matches(12345))

    def test_biome_area_validates_min_samples_bound(self) -> None:
        with self.assertRaises(ValueError):
            BiomeArea(biomes=frozenset({1}), samples_per_axis=4, min_samples=999)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
