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

    /// Whether this value may weigh a transition. [`WltsBuilder`] rejects
    /// invalid weights, e.g. NaN, infinite or negative rates.
    ///
    /// [`WltsBuilder`]: crate::WltsBuilder
    fn is_valid(&self) -> bool {
        true
    }

    /// The sum of `items`. Class sums and merged parallel transitions are
    /// computed this way, so the result must not depend on the order of the
    /// items: equal multisets of weights give equal sums. Folding with `add`
    /// does that for exact semirings.
    fn sum<'a>(items: impl IntoIterator<Item = &'a Self>) -> Self
    where
        Self: 'a,
    {
        items.into_iter().fold(Self::zero(), |acc, w| acc.add(w))
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
/// Transition weights must be finite and non-negative
/// ([`Semiring::is_valid`]). Only an exact `0.0` is zero, so tiny rates are
/// kept.
///
/// Signature keys round to a grid whose spacing is [`Real::RESOLUTION`]
/// relative to the value, so class sums that differ only by rounding noise
/// compare equal at any magnitude, and [`Semiring::sum`] adds in a fixed
/// order, so equal multisets of rates give identical sums. Values close to
/// a grid boundary can still fall into different buckets: rounding is
/// **not** transitive, so it is not a substitute for a behavioural metric.
#[derive(Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Real(pub f64);

impl Real {
    /// Relative spacing of the signature-key grid: 2⁻³⁰ ≈ 9.3e-10.
    pub const RESOLUTION: f64 = 9.313225746154785e-10;
    /// Mantissa bits dropped by signature keys (52 − 30).
    const DROPPED_BITS: u32 = 22;

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

/// Signature key of a [`Real`]: the value rounded to [`Real::RESOLUTION`]
/// relative to its magnitude. Keys order like the values they stand for.
/// Its `Debug` output is the rounded value, so witnesses read as rates.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RealKey(pub i64);

impl RealKey {
    /// The grid point this key stands for.
    pub fn value(self) -> f64 {
        let magnitude = f64::from_bits(self.0.unsigned_abs() << Real::DROPPED_BITS);
        if self.0 < 0 { -magnitude } else { magnitude }
    }
}

impl fmt::Debug for RealKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Nine significant digits: the grid's precision without its noise.
        let value = self.value();
        let shown: f64 = format!("{value:.8e}").parse().unwrap_or(value);
        write!(f, "{shown}")
    }
}

impl SignatureKey for Real {
    type Key = RealKey;
    fn key(&self) -> RealKey {
        if self.0 == 0.0 {
            return RealKey(0);
        }
        let v = if self.0.is_nan() { f64::NAN } else { self.0 };
        // The bit pattern of a non-negative float grows with its value, so
        // rounding off the low mantissa bits rounds to the nearest point of
        // a grid with relative spacing 2⁻³⁰ (carrying into the exponent).
        let half = 1u64 << (Real::DROPPED_BITS - 1);
        let rounded = ((v.abs().to_bits() + half) >> Real::DROPPED_BITS) as i64;
        RealKey(if v < 0.0 { -rounded } else { rounded })
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

    /// Exactly zero: a tiny rate is still a rate.
    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }

    /// Finite and non-negative.
    fn is_valid(&self) -> bool {
        self.0.is_finite() && self.0 >= 0.0
    }

    /// Adds in ascending order. Float addition is not associative, so this
    /// fixed order is what makes the sum (and its key) independent of the
    /// order of the items.
    fn sum<'a>(items: impl IntoIterator<Item = &'a Self>) -> Self {
        let mut values: Vec<f64> = items.into_iter().map(|w| w.0).collect();
        values.sort_by(f64::total_cmp);
        Real(values.into_iter().fold(0.0, |acc, v| acc + v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_keys_absorb_rounding_noise() {
        let a = Real(0.1).add(&Real(0.2));
        assert_eq!(a.key(), Real(0.3).key());
        assert_eq!(format!("{:?}", Real(0.1 + 0.2).key()), "0.3");
        assert_eq!(format!("{:?}", Real(2.0).key()), "2");
    }

    #[test]
    fn only_exact_zero_is_zero() {
        assert!(Real(0.0).is_zero());
        assert!(Real(-0.0).is_zero());
        // A failure rate of 4e-10 s⁻¹ (mean time to failure ≈ 79 years) is a
        // rate, not noise.
        assert!(!Real(4e-10).is_zero());
        assert!(!Real(1e-12).is_zero());
    }

    #[test]
    fn real_keys_are_relative_to_the_magnitude() {
        // Large rates stay distinct (an absolute grid saturated above ~9.2e9)…
        assert_ne!(Real(1e10).key(), Real(2e10).key());
        assert_ne!(Real(1e300).key(), Real(1e301).key());
        // …small rates are not rounded away…
        assert_ne!(Real(4e-10).key(), Real(0.0).key());
        assert_ne!(Real(4e-10).key(), Real(5e-10).key());
        // …and noise in the last bits is absorbed at every magnitude.
        for v in [1e-6, 0.3, 1e8, 1e15] {
            let noisy = v * (1.0 + 4.0 * f64::EPSILON);
            assert_ne!(v, noisy);
            assert_eq!(Real(v).key(), Real(noisy).key(), "{v}");
        }
        // Keys order like the values.
        assert!(Real(1.0).key() < Real(1.5).key());
        assert!(Real(0.0).key() < Real(4e-10).key());
        assert_eq!(Real(0.0).key(), Real(-0.0).key());
    }

    #[test]
    fn real_sums_do_not_depend_on_order() {
        let forward: Vec<Real> = [727511.0, 115478.57142857143, 172873.99999999997]
            .into_iter()
            .map(Real)
            .collect();
        let backward: Vec<Real> = forward.iter().rev().copied().collect();
        // Folding with `add` is order-sensitive in the last bits…
        let fold = |ws: &[Real]| ws.iter().fold(Real::zero(), |acc, w| acc.add(w));
        assert_ne!(fold(&forward).0.to_bits(), fold(&backward).0.to_bits());
        // …`sum` is not.
        assert_eq!(
            Real::sum(&forward).0.to_bits(),
            Real::sum(&backward).0.to_bits()
        );
        assert_eq!(Real::sum(&[]).0.to_bits(), 0.0f64.to_bits());
        assert_eq!(u64::sum(&[2, 3]), 5);
        assert!(bool::sum(&[false, true]));
    }

    #[test]
    fn real_weights_must_be_finite_and_non_negative() {
        assert!(Real(0.0).is_valid());
        assert!(Real(4e-10).is_valid());
        assert!(Real(2.5).is_valid());
        assert!(!Real(f64::NAN).is_valid());
        assert!(!Real(f64::INFINITY).is_valid());
        assert!(!Real(-1.0).is_valid());
        // NaN is not a zero weight that could silently disappear.
        assert!(!Real(f64::NAN).is_zero());
    }
}
