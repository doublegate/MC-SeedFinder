"""
Tests for CLI compatibility helpers.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mcseedfinder.cli import _load_criteria_spec


class TestCliHelpers(unittest.TestCase):
    def test_load_versioned_search_spec_extracts_criteria(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spec.json"
            path.write_text(json.dumps({
                "spec_version": 1,
                "edition": "java",
                "version": "1.21",
                "criteria": {
                    "nearby_structures": [
                        {"structure": "village", "max_distance": 1000},
                    ],
                },
            }), encoding="utf-8")

            spec = _load_criteria_spec(path)

        self.assertEqual(
            spec,
            {
                "nearby_structures": [
                    {"structure": "village", "max_distance": 1000},
                ],
            },
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
