"""
Tests for Phase 5 Bedrock foundation: text-seed hashing + BedrockProvider
validation. The Rust crate has its own tests for the hash algorithm; these
verify the Python wrappers and parity with the native binding.
"""

from __future__ import annotations

import unittest

from mcseedfinder import bedrock, rust_backend
from mcseedfinder.bedrock import _python_seed_from_string, is_valid_bedrock_seed
from mcseedfinder.engine import BedrockProvider, SearchSpec


class TestSeedFromString(unittest.TestCase):
    """Documented Java String.hashCode() vectors — identical for Bedrock."""

    def test_empty_string_is_zero(self) -> None:
        self.assertEqual(bedrock.seed_from_string(""), 0)

    def test_short_ascii_vectors(self) -> None:
        # 'a' → 97 (just the char value).
        self.assertEqual(bedrock.seed_from_string("a"), 97)
        # 31 * 97 + 98 = 3105
        self.assertEqual(bedrock.seed_from_string("ab"), 3105)
        # 31 * 3105 + 99 = 96354
        self.assertEqual(bedrock.seed_from_string("abc"), 96354)

    def test_well_known_hello(self) -> None:
        # Matches every conformant Java String.hashCode() implementation.
        self.assertEqual(bedrock.seed_from_string("Hello"), 69609650)

    def test_pure_python_matches_native(self) -> None:
        # The pure-Python fallback must agree with the Rust implementation
        # for any input — otherwise users see different seeds depending on
        # whether the native extension is built.
        if not rust_backend.is_available():
            self.skipTest("native extension not built")
        # A spread of inputs covering empty, ascii, non-ASCII, surrogate pair,
        # and a long string that overflows the i32 accumulator.
        cases = [
            "",
            "a",
            "abc",
            "Hello",
            "Minecraft",
            "é",  # single UTF-16 code unit (BMP)
            "🌍",  # supplementary plane → surrogate pair
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",  # 32 'a's; overflows several times
            "the quick brown fox jumps over the lazy dog 1234567890",
        ]
        for s in cases:
            with self.subTest(text=s):
                native = rust_backend._native.bedrock_seed_from_string(s)  # type: ignore[union-attr]
                py = _python_seed_from_string(s)
                self.assertEqual(native, py)
                self.assertIs(type(native), int)
                self.assertTrue(-(2**31) <= native < 2**31)


class TestBedrockSeedRange(unittest.TestCase):
    def test_signed_i32_range(self) -> None:
        self.assertTrue(is_valid_bedrock_seed(0))
        self.assertTrue(is_valid_bedrock_seed(-(2**31)))
        self.assertTrue(is_valid_bedrock_seed(2**31 - 1))
        self.assertFalse(is_valid_bedrock_seed(2**31))
        self.assertFalse(is_valid_bedrock_seed(-(2**31) - 1))


class TestBedrockProvider(unittest.TestCase):
    def test_rejects_wrong_edition(self) -> None:
        spec = SearchSpec(criteria={}, edition="java", version="1.21", count=1)
        with self.assertRaisesRegex(ValueError, "Bedrock provider"):
            BedrockProvider().validate_spec(spec)

    def test_rejects_out_of_range_start_seed(self) -> None:
        spec = SearchSpec(
            criteria={},
            edition="bedrock",
            version="1.21",
            count=1,
            start_seed=2**40,  # way outside i32
        )
        with self.assertRaisesRegex(ValueError, r"signed i32"):
            BedrockProvider().validate_spec(spec)

    def test_rejects_structure_criteria_with_enumerated_message(self) -> None:
        spec = SearchSpec(
            criteria={"nearby_structures": [{"structure": "village", "max_distance": 1000}]},
            edition="bedrock",
            version="1.21",
            count=1,
        )
        with self.assertRaises(ValueError) as ctx:
            BedrockProvider().validate_spec(spec)
        self.assertIn("nearby_structures", str(ctx.exception))
        self.assertIn("roadmap", str(ctx.exception).lower())

    def test_rejects_biome_criteria_in_tree(self) -> None:
        spec = SearchSpec(
            criteria={
                "conditions": {
                    "type": "all_of",
                    "of": [
                        {"type": "biome_area", "biomes": [1], "min_samples": 4},
                    ],
                }
            },
            edition="bedrock",
            version="1.21",
            count=1,
        )
        with self.assertRaises(ValueError) as ctx:
            BedrockProvider().validate_spec(spec)
        self.assertIn("biome_area", str(ctx.exception))

    def test_evaluate_seed_raises_not_implemented(self) -> None:
        # Belt and braces: even if validate_spec is bypassed, evaluate_seed
        # never silently returns a Java-evaluated report.
        from mcseedfinder.criteria import compile_criteria

        raw_spec = {"nearby_structures": [{"structure": "village"}]}
        criteria = compile_criteria(raw_spec)
        spec = SearchSpec(criteria=raw_spec, edition="bedrock", version="1.21", count=1)
        with self.assertRaises(NotImplementedError):
            BedrockProvider().evaluate_seed(1, criteria, spec)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
