//! Verdicts over growing prefixes of two systems.
//!
//! [`PrefixChecker`] compares two systems that are only observed
//! piece by piece: transitions arrive over time, and a state may be declared
//! *complete* once all of its outgoing transitions have been seen. The
//! verdicts are sound for what is still to come:
//!
//! - [`Violated`](Verdict::Violated) is final: it is only reported when a
//!   transition that *was* observed has no matching answer from a state that
//!   is complete, so no later observation can repair it. It carries a
//!   distinguishing formula as usual.
//! - [`Holds`](Verdict::Holds) means the relation holds and every state
//!   involved is complete.
//! - [`HoldsSoFar`](Verdict::HoldsSoFar) means no difference has been found,
//!   but some state involved can still gain transitions.
//!
//! A state that is never declared complete can always gain an answer, so
//! missing answers from it never count against the relation. Each call to
//! [`verdict`](PrefixChecker::verdict) replays the on-the-fly game on the
//! prefix seen so far; it does not yet update its answer incrementally when
//! a transition arrives.

use std::collections::HashSet;
use std::hash::Hash;

use bisim_core::{ComparisonVerdict, Formula, Lts, NamedBuilder, StateId, Verdict};

use crate::game::{Game, GameBudget};

/// One observed side: its transitions and the states known to be complete.
struct Side<S, L> {
    initial: S,
    transitions: Vec<(S, L, S)>,
    complete: HashSet<S>,
}

impl<S: Eq + Hash + Clone, L: Eq + Hash + Clone> Side<S, L> {
    fn new(initial: S) -> Self {
        Self {
            initial,
            transitions: Vec::new(),
            complete: HashSet::new(),
        }
    }

    /// The system seen so far, its initial state, and which states are
    /// still unfinished (indexed by state).
    fn snapshot(&self) -> (Lts<L>, StateId, Vec<bool>) {
        let mut b = NamedBuilder::<S, L, bool>::new();
        b.initial(self.initial.clone());
        for (s, l, t) in &self.transitions {
            b.edge(s.clone(), l.clone(), t.clone());
        }
        for s in &self.complete {
            b.state(s.clone());
        }
        let (lts, names) = b.build();
        let unfinished = names.iter().map(|n| !self.complete.contains(n)).collect();
        let initial = lts.initial().expect("initial state was set");
        (lts, initial, unfinished)
    }
}

/// Compares the initial states of two systems observed incrementally.
pub struct PrefixChecker<S, L> {
    left: Side<S, L>,
    right: Side<S, L>,
    budget: GameBudget,
}

impl<S: Eq + Hash + Clone, L: Eq + Hash + Clone> std::fmt::Debug for PrefixChecker<S, L> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefixChecker")
            .field("left_transitions", &self.left.transitions.len())
            .field("right_transitions", &self.right.transitions.len())
            .finish()
    }
}

impl<S: Eq + Hash + Clone, L: Eq + Hash + Clone> PrefixChecker<S, L> {
    /// Starts with the two initial states and no transitions.
    pub fn new(left_initial: S, right_initial: S) -> Self {
        Self {
            left: Side::new(left_initial),
            right: Side::new(right_initial),
            budget: GameBudget::default(),
        }
    }

    /// Limits the search of each [`verdict`](Self::verdict).
    pub fn with_budget(mut self, budget: GameBudget) -> Self {
        self.budget = budget;
        self
    }

    /// Adds observed transitions `(source, label, target)` of the left system.
    pub fn observe_left(&mut self, transitions: impl IntoIterator<Item = (S, L, S)>) {
        self.left.transitions.extend(transitions);
    }

    /// Adds observed transitions of the right system.
    pub fn observe_right(&mut self, transitions: impl IntoIterator<Item = (S, L, S)>) {
        self.right.transitions.extend(transitions);
    }

    /// Declares that every transition out of `state` in the left system has
    /// been observed.
    pub fn complete_left(&mut self, state: S) {
        self.left.complete.insert(state);
    }

    /// Declares that every transition out of `state` in the right system has
    /// been observed.
    pub fn complete_right(&mut self, state: S) {
        self.right.complete.insert(state);
    }

    /// The three comparisons for the prefix seen so far.
    pub fn verdict(&self) -> ComparisonVerdict<Formula<L, bool>> {
        let (left, l0, left_open) = self.left.snapshot();
        let (right, r0, right_open) = self.right.snapshot();
        let union = left.disjoint_union(&right);
        let r0 = StateId(r0.0 + left.num_states() as u32);
        let unfinished: Vec<bool> = left_open.into_iter().chain(right_open).collect();

        let game = |s: StateId, t: StateId, simulation: bool| {
            Game::on_prefix(&union, self.budget, unfinished.clone(), simulation).play(s, t)
        };
        let right_simulates_left = game(l0, r0, true);
        let mutually_similar = match &right_simulates_left {
            Verdict::Violated { .. } => right_simulates_left.clone(),
            first => {
                let back = game(r0, l0, true);
                match (first, &back) {
                    (_, Verdict::Violated { .. }) => back,
                    (a, b) => weakest(a, b),
                }
            }
        };
        let bisimilar = match &mutually_similar {
            Verdict::Violated { .. } => mutually_similar.clone(),
            similar => weakest(similar, &game(l0, r0, false)),
        };
        ComparisonVerdict {
            right_simulates_left,
            mutually_similar,
            bisimilar,
        }
    }
}

/// The weaker of two non-violated verdicts about the same prefix; a
/// violation of the second wins.
fn weakest<X: Clone>(a: &Verdict<X>, b: &Verdict<X>) -> Verdict<X> {
    match (a, b) {
        (_, v @ Verdict::Violated { .. }) => v.clone(),
        (Verdict::Holds, other) => other.clone(),
        (other, Verdict::Holds) => other.clone(),
        (Verdict::Unknown, _) | (_, Verdict::Unknown) => Verdict::Unknown,
        (a, _) => a.clone(),
    }
}
