use std::collections::HashMap;
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
    ///
    /// Every (subformula, state) pair is evaluated at most once, so the cost
    /// is polynomial in the size of the formula and the part of the system
    /// it reaches (a plain recursion is exponential in the modal depth).
    pub fn holds<W>(&self, sys: &Wlts<L, W>, s: StateId) -> bool
    where
        W: Semiring + SignatureKey<Key = K>,
    {
        self.eval(sys, s, &mut HashMap::new())
    }

    /// The states of `sys` that satisfy the formula (sharing one cache).
    pub fn satisfying<W>(&self, sys: &Wlts<L, W>) -> Vec<StateId>
    where
        W: Semiring + SignatureKey<Key = K>,
    {
        let mut memo = HashMap::new();
        sys.states()
            .filter(|&s| self.eval(sys, s, &mut memo))
            .collect()
    }

    fn eval<W>(
        &self,
        sys: &Wlts<L, W>,
        s: StateId,
        memo: &mut HashMap<(*const Self, StateId), bool>,
    ) -> bool
    where
        W: Semiring + SignatureKey<Key = K>,
    {
        let key = (self as *const Self, s);
        if let Some(&known) = memo.get(&key) {
            return known;
        }
        let result = match self {
            Formula::True => true,
            Formula::Not(f) => !f.eval(sys, s, memo),
            Formula::And(fs) => fs.iter().all(|f| f.eval(sys, s, memo)),
            Formula::Modal {
                label,
                weight,
                target,
            } => {
                // Summed like class sums, so the key matches the witness's.
                let satisfying: Vec<&W> = sys
                    .successors(s)
                    .iter()
                    .filter(|e| sys.label(e.label) == label && target.eval(sys, e.target, memo))
                    .map(|e| &e.weight)
                    .collect();
                &W::sum(satisfying).key() == weight
            }
        };
        memo.insert(key, result);
        result
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NamedBuilder;

    #[test]
    fn deep_formulas_on_branching_systems_are_polynomial() {
        // Two parallel `a`-edges at every level: a plain recursion visits
        // 2^depth paths.
        let depth = 40u32;
        let mut b = NamedBuilder::<u32, char, bool>::new();
        for i in 0..depth {
            b.edge(i, 'a', i + 1);
            b.edge(i, 'a', i + 1000);
            b.edge(i + 1000, 'a', i + 1);
        }
        let (sys, names) = b.build();
        let mut f = Formula::True;
        for _ in 0..depth {
            f = Formula::Modal {
                label: 'a',
                weight: true,
                target: Box::new(Formula::Not(Box::new(Formula::Not(Box::new(f))))),
            };
        }
        let start = std::time::Instant::now();
        let sat = f.satisfying(&sys);
        assert!(start.elapsed().as_secs() < 5);
        let zero = StateId(names.iter().position(|n| *n == 0).unwrap() as u32);
        assert!(sat.contains(&zero));
        assert!(f.holds(&sys, zero));
    }
}
