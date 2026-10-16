//! On-the-fly bisimulation game (Stirling) for boolean LTSs.
//!
//! [`game_bisimilar`] decides bisimilarity of two states by playing the
//! Spoiler/Duplicator game, discovering pairs lazily from the initial
//! query: only pairs reachable from `(s, t)` via matching moves are ever
//! explored. A shallow disagreement is found without refining the whole
//! systems, which is what makes this the on-the-fly counterpart to
//! [`refine`](super::refine).
//!
//! Two budgets bound the search: at most `max_pairs` distinct pairs and
//! nesting depth `max_depth`. Exhausting the pair budget reports
//! [`Unknown`](bisim_core::Verdict::Unknown); hitting only the depth
//! limit reports [`HoldsUpTo`](bisim_core::Verdict::HoldsUpTo); a
//! completed search reports `Holds` or a Spoiler witness (`Violated`).
//! Cyclic plays are Duplicator wins (the coinductive rule), so a
//! completed `Holds` is exact.

use std::collections::HashMap;
use std::hash::Hash;

use bisim_core::{Formula, Lts, StateId, Verdict};

/// Search budgets for [`game_bisimilar`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GameBudget {
    /// Maximum number of distinct pairs explored (`0` decides nothing).
    pub max_pairs: usize,
    /// Maximum nesting depth of pairs (`0` still finds immediate mismatches).
    pub max_depth: u32,
}

impl Default for GameBudget {
    fn default() -> Self {
        Self {
            max_pairs: 100_000,
            max_depth: u32::MAX,
        }
    }
}

/// Decides whether state `s` of `left` and state `t` of `right` are
/// bisimilar, exploring pairs on demand under `budget`. A violation
/// carries an existential HML formula true in `s` and false in `t`
/// (or its negation when Spoiler wins from the right).
pub fn game_bisimilar<L: Eq + Hash + Clone>(
    left: &Lts<L>,
    s: StateId,
    right: &Lts<L>,
    t: StateId,
    budget: GameBudget,
) -> Verdict<Formula<L, bool>> {
    let union = left.disjoint_union(right);
    let t = StateId(t.0 + left.num_states() as u32);
    let mut solver = Solver {
        sys: &union,
        memo: HashMap::new(),
        stack: Vec::new(),
        pairs: 0,
        budget,
        shallow: false,
    };
    match solver.solve(s.0, t.0, 0) {
        Outcome::Dup => {
            if solver.shallow {
                Verdict::HoldsUpTo {
                    depth: budget.max_depth,
                }
            } else {
                Verdict::Holds
            }
        }
        Outcome::Spoil(witness) => Verdict::Violated { witness },
        Outcome::Out => Verdict::Unknown,
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Outcome<L> {
    Dup,
    Spoil(Formula<L, bool>),
    Out,
}

struct Solver<'a, L> {
    sys: &'a Lts<L>,
    memo: HashMap<(u32, u32), Outcome<L>>,
    stack: Vec<(u32, u32)>,
    pairs: usize,
    budget: GameBudget,
    shallow: bool,
}

impl<L: Eq + Hash + Clone> Solver<'_, L> {
    fn solve(&mut self, x: u32, y: u32, depth: u32) -> Outcome<L> {
        // Game positions are unordered (Spoiler attacks either side), so
        // mirror entries transfer: `Dup` directly, refutations negated
        // (keeping the true-in-first convention).
        if let Some(known) = self.memo.get(&(x, y)) {
            return known.clone();
        }
        if let Some(mirror) = self.memo.get(&(y, x)) {
            return match mirror.clone() {
                Outcome::Dup => Outcome::Dup,
                Outcome::Spoil(w) => Outcome::Spoil(Formula::Not(Box::new(w))),
                Outcome::Out => Outcome::Out,
            };
        }
        if self.stack.contains(&(x, y)) || self.stack.contains(&(y, x)) {
            // A cycle: Spoiler loops forever, so Duplicator wins.
            return Outcome::Dup;
        }
        if self.pairs >= self.budget.max_pairs {
            return Outcome::Out;
        }
        if depth > self.budget.max_depth {
            self.shallow = true;
            return Outcome::Dup;
        }
        self.pairs += 1;
        self.stack.push((x, y));
        let outcome = match (
            self.attack(StateId(x), StateId(y), depth, false),
            self.attack(StateId(y), StateId(x), depth, true),
        ) {
            (Attack::Refuted(witness), _) => Outcome::Spoil(witness),
            (_, Attack::Refuted(witness)) => Outcome::Spoil(witness),
            (Attack::Out, _) | (_, Attack::Out) => Outcome::Out,
            (Attack::Matched, Attack::Matched) => Outcome::Dup,
        };
        self.stack.pop();
        // `Out` is search-state dependent (remaining budget), so it is
        // never memoized; `Dup` (exact or depth-capped, tracked globally)
        // and `Spoil` (a genuine refutation) are safe to reuse.
        if !matches!(outcome, Outcome::Out) {
            self.memo.insert((x, y), outcome.clone());
        }
        outcome
    }

    /// One Spoiler direction: every move of `a` matched by `b` (`Matched`),
    /// an unmatched move (`Refuted` with witness), or undecided (`Out`).
    /// `flipped` means the attack runs from the right, so the witness is
    /// negated to keep the `true-in-left` convention.
    fn attack(&mut self, a: StateId, b: StateId, depth: u32, flipped: bool) -> Attack<L> {
        for ea in self.sys.successors(a) {
            let mut conj = Vec::new();
            let mut matched = false;
            let mut out = false;
            for eb in self.sys.successors(b) {
                if eb.label != ea.label {
                    continue;
                }
                match self.solve(ea.target.0, eb.target.0, depth + 1) {
                    Outcome::Dup => {
                        matched = true;
                        break;
                    }
                    Outcome::Spoil(w) => conj.push(w),
                    Outcome::Out => out = true,
                }
            }
            if matched {
                continue;
            }
            if out {
                // Some answers refute, others are beyond budget: no conclusion.
                return Attack::Out;
            }
            let target = match conj.len() {
                0 => Formula::True,
                1 => conj.pop().expect("one conjunct"),
                _ => Formula::And(conj),
            };
            let witness = Formula::Modal {
                label: self.sys.label(ea.label).clone(),
                weight: true,
                target: Box::new(target),
            };
            return Attack::Refuted(if flipped {
                Formula::Not(Box::new(witness))
            } else {
                witness
            });
        }
        Attack::Matched
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Attack<L> {
    Matched,
    Refuted(Formula<L, bool>),
    Out,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compare::bisimilar_states;
    use bisim_core::NamedBuilder;

    fn budget() -> GameBudget {
        GameBudget::default()
    }

    #[test]
    fn finds_shallow_mismatch_without_full_refinement() {
        // a.a.0 vs a.0: Spoiler wins in two plies.
        let (sys, _) = NamedBuilder::from_triples([("p0", 'a', "p1"), ("p1", 'a', "p2")]);
        let v = game_bisimilar(&sys, StateId(0), &sys, StateId(1), budget());
        let Verdict::Violated { witness } = v else {
            panic!("expected a violation, got {v:?}");
        };
        assert_eq!(witness.to_string(), "⟨a⟩⟨a⟩tt");
        assert!(witness.holds(&sys.disjoint_union(&sys), StateId(0)));
        // Pairs explored: only the reachable ones, not the full square.
        assert!(solver_pairs(&sys) < sys.num_states() * sys.num_states());
    }

    fn solver_pairs(sys: &Lts<char>) -> usize {
        let union = sys.disjoint_union(sys);
        let mut solver = Solver {
            sys: &union,
            memo: HashMap::new(),
            stack: Vec::new(),
            pairs: 0,
            budget: budget(),
            shallow: false,
        };
        let _ = solver.solve(0, 1, 0);
        solver.pairs
    }

    #[test]
    fn agrees_with_refinement_on_all_pairs() {
        // Chain, cycle and branching: every pair, both directions.
        let (sys, _) = NamedBuilder::from_triples([
            ("c0", 'a', "c1"),
            ("c1", 'a', "c2"),
            ("y0", 'a', "y0"),
            ("y1", 'a', "y0"),
            ("b", 'a', "b1"),
            ("b", 'a', "b2"),
            ("b1", 'b', "b1"),
        ]);
        for s in sys.states() {
            for t in sys.states() {
                let g = game_bisimilar(&sys, s, &sys, t, budget());
                let r = bisimilar_states(&sys, s, &sys, t);
                assert_eq!(g.holds(), r.holds(), "pair {s:?} vs {t:?}");
                if let Verdict::Violated { witness } = g {
                    let union = sys.disjoint_union(&sys);
                    let tt = StateId(t.0 + sys.num_states() as u32);
                    assert!(witness.holds(&union, s), "witness holds in left: {witness}");
                    assert!(
                        !witness.holds(&union, tt),
                        "witness fails in right: {witness}"
                    );
                }
            }
        }
    }

    #[test]
    fn budgets_report_unknown_and_holds_up_to() {
        let (sys, _) = NamedBuilder::from_triples([("p0", 'a', "p1"), ("p1", 'a', "p2")]);
        // No pairs allowed: nothing can be decided.
        let v = game_bisimilar(
            &sys,
            StateId(0),
            &sys,
            StateId(0),
            GameBudget {
                max_pairs: 0,
                max_depth: u32::MAX,
            },
        );
        assert_eq!(v, Verdict::Unknown);
        // Depth 0 on a pair that needs depth: holds so far, at depth 0.
        // (A self-loop pair would instead close by the cycle rule: exact.)
        let v = game_bisimilar(
            &sys,
            StateId(0),
            &sys,
            StateId(0),
            GameBudget {
                max_pairs: 100,
                max_depth: 0,
            },
        );
        assert_eq!(v, Verdict::HoldsUpTo { depth: 0 });
        // Depth 0 still finds immediate mismatches (no recursion needed).
        let v = game_bisimilar(
            &sys,
            StateId(2),
            &sys,
            StateId(0),
            GameBudget {
                max_pairs: 100,
                max_depth: 0,
            },
        );
        assert!(matches!(v, Verdict::Violated { .. }));
    }

    #[test]
    fn right_side_wins_are_negated() {
        // Left loops on `a`, right also offers `b`: Spoiler wins from the
        // right, and the negation keeps the true-in-left convention.
        let (left, _) = NamedBuilder::from_triples([("x", 'a', "x")]);
        let (right, _) = NamedBuilder::from_triples([("y", 'a', "y"), ("y", 'b', "y")]);
        let v = game_bisimilar(&left, StateId(0), &right, StateId(0), budget());
        let Verdict::Violated { witness } = v else {
            panic!("expected a violation");
        };
        assert_eq!(witness.to_string(), "¬⟨b⟩tt");
        let union = left.disjoint_union(&right);
        let t = StateId(1);
        assert!(witness.holds(&union, StateId(0)));
        assert!(!witness.holds(&union, t));
    }
}
