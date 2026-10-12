use std::fmt;
use std::hash::Hash;

/// Values that can be compared exactly when grouping states by signature.
///
/// Exact types use themselves as key. Floating-point weights must map to a
/// discrete key; see [`Real`] for what that implies.
pub trait SignatureKey {
    type Key: Eq + Ord + Hash + Clone + fmt::Debug;
    fn key(&self) -> Self::Key;
}

/// A commutative semiring of transition weights.
///
/// Only `zero` and `add` are needed to decide bisimilarity (class sums);
/// `one` and `mul` are used by composition operators and weak saturation.
pub trait Semiring: Clone + fmt::Debug + SignatureKey {
    fn zero() -> Self;
    fn one() -> Self;
    fn add(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;

    fn is_zero(&self) -> bool {
        self.key() == Self::zero().key()
    }
}

impl SignatureKey for bool {
    type Key = bool;
    fn key(&self) -> bool {
        *self
    }
}

/// The Boolean semiring `(∨, ∧)`: plain labelled transition systems.
impl Semiring for bool {
    fn zero() -> Self {
        false
    }
    fn one() -> Self {
        true
    }
    fn add(&self, other: &Self) -> Self {
        *self || *other
    }
    fn mul(&self, other: &Self) -> Self {
        *self && *other
    }
}

impl SignatureKey for u64 {
    type Key = u64;
    fn key(&self) -> u64 {
        *self
    }
}

/// Natural numbers: transition multiplicities.
impl Semiring for u64 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn add(&self, other: &Self) -> Self {
        self.saturating_add(*other)
    }
    fn mul(&self, other: &Self) -> Self {
        self.saturating_mul(*other)
    }
}

/// Non-negative real weight (a rate in s⁻¹ or a probability).
///
/// Signature keys round to an absolute grid of [`Real::RESOLUTION`]. Two class
/// sums that differ only by rounding noise compare equal, but values that lie
/// close to a grid boundary can still fall into different buckets. ε-bucketing
/// is **not** transitive, so it is not a substitute for a behavioural metric.
#[derive(Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Real(pub f64);

impl Real {
    pub const RESOLUTION: f64 = 1e-9;
    /// `1 / RESOLUTION`, exactly representable, so keys convert back without drift.
    const SCALE: f64 = 1e9;

    pub fn value(self) -> f64 {
        self.0
    }
}

impl fmt::Debug for Real {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Display for Real {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Signature key of a [`Real`]: the value rounded to [`Real::RESOLUTION`].
/// Its `Debug` output is the rounded value, so witnesses read as rates.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RealKey(pub i64);

impl fmt::Debug for RealKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0 as f64 / Real::SCALE)
    }
}

impl SignatureKey for Real {
    type Key = RealKey;
    fn key(&self) -> RealKey {
        RealKey((self.0 * Self::SCALE).round() as i64)
    }
}

impl Semiring for Real {
    fn zero() -> Self {
        Real(0.0)
    }
    fn one() -> Self {
        Real(1.0)
    }
    fn add(&self, other: &Self) -> Self {
        Real(self.0 + other.0)
    }
    fn mul(&self, other: &Self) -> Self {
        Real(self.0 * other.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_keys_absorb_rounding_noise() {
        let a = Real(0.1).add(&Real(0.2));
        assert_eq!(a.key(), Real(0.3).key());
        assert!(Real(1e-12).is_zero());
        assert!(!Real(1e-6).is_zero());
        assert_eq!(format!("{:?}", Real(0.1 + 0.2).key()), "0.3");
    }
}
