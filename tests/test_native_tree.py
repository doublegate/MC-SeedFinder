"""
Parity between the native structure-only tree evaluator (Rust) and the
Python `CriteriaSet`. The Rust path is opt-in: it must NOT silently disagree
with Python on which seeds match, otherwise a tree fast-path search would
return different results than the same spec via the Python evaluator.

Also exercises the "force fallback when biomes are involved" routing — the
native path is structure-only by design and any biome criterion in the spec
must push the search to the Python evaluator.
"""

from __future__ import annotations

import unittest

from mcseedfinder import rust_backend
from mcseedfinder.criteria import compile_criteria


@unittest.skipUnless(rust_backend.is_available(), "Rust extension not built")
class TestNativeTreeParity(unittest.TestCase):
    def _python_matches(self, spec, seed_range):
        criteria = compile_criteria(spec)
        return [s for s in seed_range if criteria.matches(s)]

    def test_flat_nearby_structures_parity(self) -> None:
        spec = {
            "nearby_structures": [
                {"structure": "village", "max_distance": 1000},
            ],
        }
        tree = rust_backend.compile_structure_only_tree(spec)
        self.assertIsNotNone(tree)
        native = rust_backend.find_tree_matches_range(1, 200, tree)
        py = self._python_matches(spec, range(1, 201))
        self.assertEqual(native, py, "native flat-structure path drifted from Python")

    def test_logic_gate_parity(self) -> None:
        spec = {
            "conditions": {
                "type": "any_of",
                "of": [
                    {"type": "nearby_structure", "structure": "village", "max_distance": 800},
                    {"type": "nearby_structure", "structure": "pillager_outpost",
                     "max_distance": 800},
                ],
            }
        }
        tree = rust_backend.compile_structure_only_tree(spec)
        self.assertIsNotNone(tree)
        native = rust_backend.find_tree_matches_range(1, 200, tree)
        py = self._python_matches(spec, range(1, 201))
        self.assertEqual(native, py)

    def test_cluster_parity(self) -> None:
        spec = {
            "conditions": {
                "type": "cluster",
                "structures": ["village", "pillager_outpost"],
                "max_distance": 2000,
                "min_count": 2,
                "centre_x": 0,
                "centre_z": 0,
            }
        }
        tree = rust_backend.compile_structure_only_tree(spec)
        self.assertIsNotNone(tree)
        native = rust_backend.find_tree_matches_range(1, 500, tree)
        py = self._python_matches(spec, range(1, 501))
        self.assertEqual(native, py)

    def test_none_of_parity(self) -> None:
        spec = {
            "conditions": {
                "type": "none_of",
                "of": [
                    {"type": "nearby_structure", "structure": "village", "max_distance": 1},
                ],
            }
        }
        tree = rust_backend.compile_structure_only_tree(spec)
        native = rust_backend.find_tree_matches_range(1, 100, tree)
        py = self._python_matches(spec, range(1, 101))
        self.assertEqual(native, py)

    def test_biome_present_forces_fallback(self) -> None:
        # Mixing in any biome criterion must disable the native path so we don't
        # silently drop the biome check.
        spec = {
            "spawn_biome": "plains",
            "conditions": {
                "type": "nearby_structure", "structure": "village", "max_distance": 800,
            },
        }
        self.assertIsNone(rust_backend.compile_structure_only_tree(spec))

        spec2 = {
            "conditions": {
                "type": "all_of",
                "of": [
                    {"type": "nearby_structure", "structure": "village",
                     "max_distance": 800},
                    {"type": "biome_area", "biomes": ["plains"],
                     "radius": 1000, "min_samples": 8},
                ],
            }
        }
        self.assertIsNone(rust_backend.compile_structure_only_tree(spec2))

    def test_unknown_structure_rejected_by_native(self) -> None:
        tree = {"type": "nearby_structure", "structure": "not_a_real_structure",
                "max_distance": 100}
        with self.assertRaises(KeyError):
            rust_backend.find_tree_matches_range(1, 1, tree)

    def test_compiled_handle_matches_uncompiled_path(self) -> None:
        """The pre-compiled-tree fast-path (R2/P1) must produce the exact
        same matches as the uncompiled path that re-parses JSON per call."""
        spec = {
            "conditions": {
                "type": "any_of",
                "of": [
                    {"type": "nearby_structure", "structure": "village",
                     "max_distance": 800},
                    {"type": "nearby_structure", "structure": "desert_pyramid",
                     "max_distance": 800},
                ],
            }
        }
        tree = rust_backend.compile_structure_only_tree(spec)
        self.assertIsNotNone(tree)
        uncompiled = rust_backend.find_tree_matches_range(1, 500, tree)
        compiled = rust_backend.compile_native_tree(tree)
        self.assertIsNotNone(
            compiled, "compile_native_tree returned None despite Rust backend present"
        )
        from_compiled = rust_backend.find_tree_matches_range_compiled(
            1, 500, compiled
        )
        self.assertEqual(uncompiled, from_compiled)

    def test_compiled_handle_exposes_biome_leaves_flag(self) -> None:
        """The compiled handle reports whether the tree touches biomes — used
        by future routing to pick the biome-aware native path."""
        tree = {"type": "nearby_structure", "structure": "village",
                "max_distance": 800}
        compiled = rust_backend.compile_native_tree(tree)
        self.assertIsNotNone(compiled)
        self.assertFalse(compiled.has_biome_leaves)


@unittest.skipUnless(
    rust_backend.is_available() and rust_backend.has_biome_aware_native(),
    "biome-aware native search unavailable",
)
class TestBiomeAwareNativeSearch(unittest.TestCase):
    """Parity for the R3 fix: biome-aware native search must match the Python
    evaluator exactly on biome-touching specs."""

    def test_spawn_biome_parity_against_python(self) -> None:
        spec = {
            "spawn_biome": ["ocean", "deep_ocean", "frozen_ocean"],
        }
        backend = rust_backend.make_biome_backend("1.21", "overworld")
        self.assertIsNotNone(backend, "cubiomes backend required for parity test")
        # Resolve the tree the way finder.py does.
        full_tree = rust_backend.compile_native_tree_full(spec)
        self.assertIsNotNone(full_tree)
        compiled = rust_backend.compile_native_tree(full_tree)
        self.assertIsNotNone(compiled)
        self.assertTrue(compiled.has_biome_leaves)

        native = rust_backend.find_tree_matches_range_compiled_with_biomes(
            1, 200, compiled, backend
        )

        # Python reference: compile the SAME spec and evaluate seed-by-seed.
        criteria = compile_criteria(spec, biome_version="1.21",
                                    biome_dimension="overworld")
        py = [s for s in range(1, 201) if criteria.matches(s)]
        self.assertEqual(
            native, py,
            "biome-aware native search drifted from the Python evaluator"
        )

    def test_mixed_structure_and_biome_parity(self) -> None:
        spec = {
            "nearby_structures": [
                {"structure": "village", "max_distance": 2000},
            ],
            "spawn_biome": ["plains", "savanna", "desert"],
        }
        backend = rust_backend.make_biome_backend("1.21", "overworld")
        self.assertIsNotNone(backend)
        full_tree = rust_backend.compile_native_tree_full(spec)
        self.assertIsNotNone(full_tree)
        compiled = rust_backend.compile_native_tree(full_tree)
        self.assertIsNotNone(compiled)
        self.assertTrue(compiled.has_biome_leaves)

        native = rust_backend.find_tree_matches_range_compiled_with_biomes(
            1, 500, compiled, backend
        )

        criteria = compile_criteria(spec, biome_version="1.21",
                                    biome_dimension="overworld")
        py = [s for s in range(1, 501) if criteria.matches(s)]
        self.assertEqual(native, py)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
