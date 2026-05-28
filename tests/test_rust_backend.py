"""
Parity tests for the optional Rust extension.
"""

from __future__ import annotations

import unittest

from mcseedfinder import rust_backend
from mcseedfinder.java_random import JavaRandom
from mcseedfinder.structures import get_structure_pos


@unittest.skipUnless(rust_backend.is_available(), "Rust extension not built")
class TestRustBackendParity(unittest.TestCase):
    def test_java_random_vectors_match_python(self) -> None:
        py = JavaRandom(42)
        rs = rust_backend.java_random(42)
        self.assertEqual(rs.next_int(), py.next_int())
        self.assertEqual(rs.next_long(), py.next_long())

    def test_structure_position_matches_python(self) -> None:
        py = get_structure_pos("village", 1, 0, 0)
        name, chunk_x, chunk_z = rust_backend.get_structure_pos("village", 1, 0, 0)
        self.assertEqual(name, "village")
        self.assertEqual((chunk_x, chunk_z), (py.chunk_x, py.chunk_z))

    def test_batch_structure_search_matches_python_vector(self) -> None:
        reqs = rust_backend.compile_structure_only_requirements({
            "nearby_structures": [
                {"structure": "village", "max_distance": 1000},
            ],
        })
        assert reqs is not None
        matches = rust_backend.find_structure_matches_range(1, 1, reqs)
        self.assertEqual(matches, [1])

    def test_strongholds_match_python_count(self) -> None:
        strongholds = rust_backend.iter_strongholds(1, max_rings=1)
        self.assertEqual(len(strongholds), 3)
        self.assertTrue(all(name == "stronghold" for name, _, _ in strongholds))

    def test_batch_stronghold_search_uses_rust_path(self) -> None:
        reqs = rust_backend.compile_structure_only_requirements({
            "nearby_structures": [
                {"structure": "stronghold", "max_distance": 3000},
            ],
        })
        assert reqs is not None
        matches = rust_backend.find_structure_matches_range(1, 1, reqs)
        self.assertEqual(matches, [1])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
