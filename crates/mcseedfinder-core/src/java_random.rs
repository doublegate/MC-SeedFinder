//! Bit-exact port of `java.util.Random`.

const MULTIPLIER: u64 = 0x5DEECE66D;
const INCREMENT: u64 = 0xB;
const MASK48: u64 = (1_u64 << 48) - 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JavaRandom {
    seed: u64,
}

impl JavaRandom {
    pub fn new(seed: i64) -> Self {
        let mut rng = Self { seed: 0 };
        rng.set_seed(seed);
        rng
    }

    pub fn set_seed(&mut self, seed: i64) {
        self.seed = ((seed as u64) ^ MULTIPLIER) & MASK48;
    }

    pub fn next(&mut self, bits: u32) -> u32 {
        self.seed = self.seed.wrapping_mul(MULTIPLIER).wrapping_add(INCREMENT) & MASK48;
        (self.seed >> (48 - bits)) as u32
    }

    pub fn next_int(&mut self) -> i32 {
        self.next(32) as i32
    }

    pub fn next_int_bound(&mut self, bound: i32) -> i32 {
        assert!(bound > 0, "bound must be positive");

        if (bound & -bound) == bound {
            return (((bound as i64) * (self.next(31) as i64)) >> 31) as i32;
        }

        loop {
            let bits = self.next(31) as i32;
            let val = bits % bound;
            if bits.wrapping_sub(val).wrapping_add(bound - 1) >= 0 {
                return val;
            }
        }
    }

    pub fn next_long(&mut self) -> i64 {
        let high = self.next(32) as i32 as i64;
        let low = self.next(32) as i32 as i64;
        (high << 32).wrapping_add(low)
    }

    pub fn next_double(&mut self) -> f64 {
        let high = (self.next(26) as u64) << 27;
        let low = self.next(27) as u64;
        (high + low) as f64 / ((1_u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::JavaRandom;

    #[test]
    fn next_long_seed_0() {
        assert_eq!(JavaRandom::new(0).next_long(), -4962768465676381896);
    }

    #[test]
    fn next_long_seed_1() {
        assert_eq!(JavaRandom::new(1).next_long(), -4964420948893066024);
    }

    #[test]
    fn next_int_bound_seed_0() {
        assert_eq!(JavaRandom::new(0).next_int_bound(10), 0);
    }

    #[test]
    fn next_double_seed_0() {
        let got = JavaRandom::new(0).next_double();
        assert!((got - 0.730967787376657).abs() < 1e-15);
    }

    #[test]
    fn first_five_ints_seed_42() {
        let mut rng = JavaRandom::new(42);
        let actual = [
            rng.next_int(),
            rng.next_int(),
            rng.next_int(),
            rng.next_int(),
            rng.next_int(),
        ];
        assert_eq!(
            actual,
            [-1170105035, 234785527, -1360544799, 205897768, 1325939940]
        );
    }
}
