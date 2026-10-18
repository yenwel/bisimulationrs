/// Outcome of one behavioural check, parametric in the witness type `X`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Verdict<X> {
    /// The relation holds on the full (finite) systems.
    Holds,
    /// The relation holds up to the given number of steps (`∼ₙ`). On a prefix
    /// of an image-finite system this is all that can be claimed.
    HoldsUpTo { depth: u32 },
    /// The relation fails; the witness explains why.
    Violated { witness: X },
    /// No difference found, but some behaviour is still being observed
    /// (open states of a growing prefix): the relation may yet fail.
    HoldsSoFar,
    /// Not enough information yet.
    Unknown,
    /// Quantitative answer: behavioural distance and optional confidence interval.
    Approx {
        distance: f64,
        ci: Option<(f64, f64)>,
    },
}

impl<X> Verdict<X> {
    pub fn holds(&self) -> bool {
        matches!(
            self,
            Verdict::Holds | Verdict::HoldsUpTo { .. } | Verdict::HoldsSoFar
        )
    }

    pub fn is_violated(&self) -> bool {
        matches!(self, Verdict::Violated { .. })
    }
}

/// The three comparisons this crate reports for a pair of systems.
///
/// Bisimilarity implies mutual similarity, which implies that the right
/// system simulates the left, but none of the converses hold for
/// nondeterministic systems.
#[derive(Debug, Clone, PartialEq)]
pub struct ComparisonVerdict<X> {
    pub right_simulates_left: Verdict<X>,
    pub mutually_similar: Verdict<X>,
    pub bisimilar: Verdict<X>,
}
