"""
Tests for the exact cubiomes biome backend and its integration with the
criteria / engine layers.

When the native extension is built without cubiomes (HAS_CUBIOMES == False),
the cubiomes-specific tests are skipped, but the catalog-consistency and
fallback-plumbing tests still run.
"""

from __future__ import annotations

import unittest

from mcseedfinder import rust_backend
from mcseedfinder.biome_gen import BiomeLookup
from mcseedfinder.biomes import BIOMES
from mcseedfinder.criteria import compile_criteria
from mcseedfinder.engine import JavaPythonProvider, SearchSpec

_HAS_CUBIOMES = rust_backend.has_cubiomes()


class TestBiomeCatalogConsistency(unittest.TestCase):
    """The Python catalog must mirror cubiomes' numeric BiomeID values."""

    def test_known_ids(self) -> None:
        # Spot-check IDs that the cubiomes integration vectors rely on.
        self.assertEqual(BIOMES["ocean"].numeric_id, 0)
        self.assertEqual(BIOMES["plains"].numeric_id, 1)
        self.assertEqual(BIOMES["beach"].numeric_id, 16)
        self.assertEqual(BIOMES["deep_ocean"].numeric_id, 24)
        self.assertEqual(BIOMES["lukewarm_ocean"].numeric_id, 45)


@unittest.skipUnless(_HAS_CUBIOMES, "native extension built without cubiomes")
class TestCubiomesBackend(unittest.TestCase):
    def test_backend_constructs_and_is_deterministic(self) -> None:
        backend = rust_backend.make_biome_backend("1.21", "overworld")
        self.assertIsNotNone(backend)
        a = backend.get_biome(1, 0, 0)
        self.assertEqual(a, backend.get_biome(1, 0, 0))
        self.assertNotEqual(a, -1)

    def test_frozen_reference_vectors(self) -> None:
        # Mirror of the Rust-side integration vectors (MC 1.21, overworld, y=63).
        backend = rust_backend.make_biome_backend("1.21", "overworld")
        assert backend is not None
        self.assertEqual(backend.get_biome(1, 0, 0), 24)        # deep_ocean
        self.assertEqual(backend.get_biome(1, -2000, 1500), 16)  # beach
        self.assertEqual(backend.get_biome(12345, 0, 0), 0)      # ocean

    def test_unknown_version_falls_back_to_none(self) -> None:
        self.assertIsNone(rust_backend.make_biome_backend("not-a-version"))

    def test_lookup_uses_exact_backend(self) -> None:
        backend = rust_backend.make_biome_backend("1.21", "overworld")
        lookup = BiomeLookup(1, backend=backend)
        self.assertFalse(lookup.is_approximate)
        self.assertEqual(lookup.biome_at(0, 0), 24)

    def test_criteria_set_reports_exact_biomes(self) -> None:
        criteria = compile_criteria(
            {"spawn_biome": "ocean", "spawn_radius": 0},
            biome_version="1.21",
            biome_dimension="overworld",
        )
        self.assertTrue(criteria.uses_exact_biomes)
        # Seed 12345 is ocean at origin per the reference vectors.
        self.assertTrue(criteria.matches(12345))

    def test_provider_marks_biomes_exact(self) -> None:
        spec = SearchSpec(
            criteria={"spawn_biome": "ocean", "spawn_radius": 0},
            version="1.21",
            count=1,
            max_matches=1,
        )
        criteria = compile_criteria(
            spec.criteria, biome_version=spec.version, biome_dimension=spec.dimension
        )
        report = JavaPythonProvider().evaluate_seed(12345, criteria, spec)
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(report.exactness.get("spawn_biome"), "exact")
        self.assertEqual(report.warnings, [])


class TestApproximateFallback(unittest.TestCase):
    """Without a version (or cubiomes), biome lookups stay approximate."""

    def test_no_version_is_approximate(self) -> None:
        criteria = compile_criteria({"spawn_biome": "plains", "spawn_radius": 0})
        self.assertFalse(criteria.uses_exact_biomes)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
