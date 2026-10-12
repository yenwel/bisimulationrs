use std::fmt;

use crate::ids::StateId;
use crate::semiring::{Semiring, SignatureKey};
use crate::wlts::Wlts;

/// Quantitative Hennessy–Milner formula used as a distinguishing witness.
///
/// `Modal { label, weight, target }` holds in `s` iff the total weight of
/// `label`-transitions from `s` into states satisfying `target` has signature
/// key `weight`. For Boolean systems `weight = true` is the diamond `⟨a⟩φ`
/// and `weight = false` is `¬⟨a⟩φ`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Formula<L, K> {
    True,
    Not(Box<Formula<L, K>>),
    And(Vec<Formula<L, K>>),
    Modal {
        label: L,
        weight: K,
        target: Box<Formula<L, K>>,
    },
}

impl<L, K> Formula<L, K> {
    /// Modal depth: the number of steps needed to tell states apart.
    pub fn depth(&self) -> usize {
        match self {
            Formula::True => 0,
            Formula::Not(f) => f.depth(),
            Formula::And(fs) => fs.iter().map(Formula::depth).max().unwrap_or(0),
            Formula::Modal { target, .. } => 1 + target.depth(),
        }
    }
}

impl<L: Clone + Eq + std::hash::Hash, K: Eq> Formula<L, K> {
    /// Evaluates the formula in state `s` of `sys`.
    pub fn holds<W>(&self, sys: &Wlts<L, W>, s: StateId) -> bool
    where
        W: Semiring + SignatureKey<Key = K>,
    {
        match self {
            Formula::True => true,
            Formula::Not(f) => !f.holds(sys, s),
            Formula::And(fs) => fs.iter().all(|f| f.holds(sys, s)),
            Formula::Modal {
                label,
                weight,
                target,
            } => {
                let total = sys
                    .successors(s)
                    .iter()
                    .filter(|e| sys.label(e.label) == label && target.holds(sys, e.target))
                    .fold(W::zero(), |acc, e| acc.add(&e.weight));
                &total.key() == weight
            }
        }
    }
}

impl<L: fmt::Display, K: fmt::Debug> fmt::Display for Formula<L, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Formula::True => write!(f, "tt"),
            Formula::Not(g) => write!(f, "¬{g}"),
            Formula::And(gs) if gs.is_empty() => write!(f, "tt"),
            Formula::And(gs) => {
                write!(f, "(")?;
                for (i, g) in gs.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ∧ ")?;
                    }
                    write!(f, "{g}")?;
                }
                write!(f, ")")
            }
            Formula::Modal {
                label,
                weight,
                target,
            } => {
                let weight = format!("{weight:?}");
                match weight.as_str() {
                    "true" => write!(f, "⟨{label}⟩{target}"),
                    "false" => write!(f, "¬⟨{label}⟩{target}"),
                    w => write!(f, "⟨{label}⟩={w} {target}"),
                }
            }
        }
    }
}
