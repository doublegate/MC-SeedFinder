"""
Tests for the product-facing search contracts.
"""

from __future__ import annotations

import unittest

from mcseedfinder.engine import (
    BedrockProvider,
    JavaPythonProvider,
    SearchSpec,
    provider_for,
    run_staged_search,
)


class TestProductEngine(unittest.TestCase):
    def test_staged_search_emits_verified_match(self) -> None:
        spec = SearchSpec(
            criteria={
                "nearby_structures": [
                    {"structure": "village", "max_distance": 1000},
                ],
            },
            start_seed=1,
            count=1,
            max_matches=1,
        )

        events = list(run_staged_search(
            spec,
            provider=JavaPythonProvider(),
            job_id="test-job",
            progress_interval=999,
        ))

        self.assertEqual(events[0].type, "started")
        self.assertIn("candidate", [event.type for event in events])
        matches = [event for event in events if event.type == "verified_match"]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].payload["seed"], 1)
        self.assertEqual(matches[0].payload["exactness"]["structures"], "exact")
        self.assertEqual(events[-1].type, "completed")
        self.assertEqual(events[-1].payload["matches"], 1)

    def test_rejected_stage_summary_counts_stage_one_failure(self) -> None:
        spec = SearchSpec(
            criteria={
                "nearby_structures": [
                    {"structure": "village", "max_distance": 100},
                ],
            },
            start_seed=1,
            count=1,
            max_matches=1,
        )

        events = list(run_staged_search(
            spec,
            provider=JavaPythonProvider(),
            job_id="test-job",
            progress_interval=999,
        ))

        summaries = [event for event in events if event.type == "rejected_summary"]
        self.assertEqual(summaries[0].payload["rejected_by_stage"], {1: 1})
        self.assertEqual(events[-1].payload["matches"], 0)

    def test_provider_for_returns_bedrock_provider(self) -> None:
        # `provider_for("bedrock")` returns the real Phase-5 BedrockProvider.
        # The empty-criteria validation now succeeds (the provider only
        # rejects worldgen-dependent criteria); structure/biome criteria
        # rejection is covered exhaustively in tests/test_bedrock.py.
        provider = provider_for("bedrock")
        self.assertIsInstance(provider, BedrockProvider)
        provider.validate_spec(SearchSpec(criteria={}, edition="bedrock"))
        # And it still refuses structure criteria with an enumerated message.
        with self.assertRaisesRegex(ValueError, "nearby_structures"):
            provider.validate_spec(
                SearchSpec(
                    criteria={"nearby_structures": [{"structure": "village"}]},
                    edition="bedrock",
                )
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
