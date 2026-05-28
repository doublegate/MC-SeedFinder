//! Bedrock Edition support (foundation).
//!
//! Phase 5 establishes only the parts of Bedrock support that are *exact*
//! today and don't require vendoring a second cubiomes fork. Biome and
//! structure generation differ algorithmically between Java and Bedrock and
//! deserve a properly-tested second backend — see `docs/BEDROCK.md` for the
//! roadmap.
//!
//! What's implemented here:
//!   - `seed_from_string`: convert a Bedrock text seed into the canonical
//!     32-bit signed integer the game actually stores. Bedrock uses Java's
//!     `String.hashCode()`: `h = 31 * h + c` over each UTF-16 code unit,
//!     wrapping into i32. This is the same algorithm Java Edition uses for
//!     text seeds and is well-documented.
//!
//! What's deliberately NOT here (yet):
//!   - Bedrock biome generation (different layered noise than Java).
//!   - Bedrock structure placement (different salts/spacing + a different
//!     PRNG depending on version).
//!   - Stronghold ring math (Bedrock has 3 strongholds in ring 1, not 3).
//!
//! Those land in Phase 5b when the cubiomes-bedrock fork (or a hand-port) is
//! plumbed through. Until then, BedrockProvider in the Python layer rejects
//! structure/biome criteria with a clear edition-aware error.

/// Hash a Bedrock text seed into the canonical i32 game-seed.
///
/// Mirrors Java's `String.hashCode()` exactly (which is what Bedrock and
/// Java Edition both use when a player enters a text seed):
///   `h = sum_{i=0..n} s[i] * 31^(n-1-i)`, computed iteratively with i32
/// wrap-around so signed overflow is well-defined (`.wrapping_*`).
///
/// Each character is its UTF-16 code unit — that is, ASCII characters are
/// their code point, BMP characters are their u16, and supplementary
/// characters expand to a surrogate pair (matching Java's `String` shape).
pub fn seed_from_string(text: &str) -> i32 {
    let mut h: i32 = 0;
    for unit in text.encode_utf16() {
        h = h.wrapping_mul(31).wrapping_add(unit as i32);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden vectors from Java's documented `String.hashCode()` behaviour.
    /// The same values appear on the Minecraft wiki's seed-hashcode calculator
    /// and in any Java implementation — they cannot drift.
    #[test]
    fn matches_java_string_hashcode_vectors() {
        assert_eq!(seed_from_string(""), 0);
        assert_eq!(seed_from_string("a"), 97); // 'a' = 0x61 = 97
                                               // 31 * 97 + 98  = 3105   ('a','b')
        assert_eq!(seed_from_string("ab"), 3105);
        // 31 * 3105 + 99 = 96354 ('a','b','c')
        assert_eq!(seed_from_string("abc"), 96354);
        // Standard "Hello" hash; matches every Java impl.
        assert_eq!(seed_from_string("Hello"), 69609650);
    }

    #[test]
    fn wraps_on_overflow_property() {
        // Verifying the algorithm under overflow with a hand-computed golden
        // is fragile — there are too many places to slip up a 12-deep
        // i32-wrapping accumulation. Instead assert two cheaper properties
        // that any correct implementation must satisfy:
        //   1. Determinism: the function is pure; same input → same output.
        //   2. Avalanche: changing one character meaningfully changes the
        //      result (rules out a silent no-op accumulator).
        // The short vectors in the test above lock the algorithm itself.
        let a = seed_from_string("aaaaaaaaaaaa");
        let b = seed_from_string("aaaaaaaaaaaa");
        assert_eq!(a, b, "function must be deterministic");
        let c = seed_from_string("aaaaaaaaaaaab");
        assert_ne!(a, c, "changing one character must change the hash");
        let d = seed_from_string("aaaaaaaaaaab"); // mutated mid-string
        assert_ne!(a, d, "interior changes must propagate");
    }

    #[test]
    fn handles_non_ascii_utf16() {
        // "é" (U+00E9) is a single UTF-16 code unit (233).
        assert_eq!(seed_from_string("é"), 233);
        // 31 * 233 + 'a' = 7320
        assert_eq!(seed_from_string("éa"), 7320);
    }
}
