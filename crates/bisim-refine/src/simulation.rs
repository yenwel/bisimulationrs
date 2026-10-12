use std::collections::HashMap;
use std::hash::Hash;

use bisim_core::{Formula, Lts, StateId};

/// The greatest simulation on one LTS, with the round in which each pair was
/// removed (used to build witnesses).
#[derive(Debug, Clone)]
pub struct Preorder {
    n: usize,
    /// `removed[s * n + t]`: round in which "t simulates s" was refuted, or
    /// `u32::MAX` if t simulates s.
    removed: Vec<u32>,
}

impl Preorder {
    /// `s ≤ t`: state `t` simulates state `s`.
    pub fn le(&self, s: StateId, t: StateId) -> bool {
        self.removed[s.index() * self.n + t.index()] == u32::MAX
    }

    /// Simulation equivalence: `s ≤ t` and `t ≤ s`.
    pub fn equivalent(&self, s: StateId, t: StateId) -> bool {
        self.le(s, t) && self.le(t, s)
    }

    fn removed_at(&self, s: StateId, t: StateId) -> u32 {
        self.removed[s.index() * self.n + t.index()]
    }
}

/// Computes the simulation preorder as a greatest fixpoint.
///
/// Starts from the full relation and repeatedly removes `(s, t)` when some
/// move `s -a-> s'` cannot be matched by a move `t -a-> t'` with `(s', t')`
/// still related. This is the naive `O(n² · m)` per round algorithm; it is
/// simple and serves as an oracle for faster HHK-style implementations.
pub fn simulation_preorder<L: Eq + Hash + Clone>(sys: &Lts<L>) -> Preorder {
    let n = sys.num_states();
    let mut removed = vec![u32::MAX; n * n];
    let mut round = 0u32;
    loop {
        round += 1;
        let mut to_remove = Vec::new();
        for s in sys.states() {
            for t in sys.states() {
                if removed[s.index() * n + t.index()] != u32::MAX {
                    continue;
                }
                let matched = sys.successors(s).iter().all(|es| {
                    sys.successors(t).iter().any(|et| {
                        et.label == es.label
                            && removed[es.target.index() * n + et.target.index()] == u32::MAX
                    })
                });
                if !matched {
                    to_remove.push(s.index() * n + t.index());
                }
            }
        }
        if to_remove.is_empty() {
            return Preorder { n, removed };
        }
        // Removing pairs only after the whole sweep keeps rounds well defined:
        // a pair removed in round k only depends on pairs removed before k.
        for i in to_remove {
            removed[i] = round;
        }
    }
}

/// An existential HML formula (only `⟨a⟩`, `∧`, `tt`) that holds in `s` but
/// not in `t`, or `None` if `t` simulates `s`.
///
/// Existential formulas are preserved by simulation, so this proves that `t`
/// cannot simulate `s`.
pub fn simulation_witness<L: Eq + Hash + Clone>(
    sys: &Lts<L>,
    preorder: &Preorder,
    s: StateId,
    t: StateId,
) -> Option<Formula<L, bool>> {
    if preorder.le(s, t) {
        return None;
    }
    let mut memo = HashMap::new();
    Some(sim_witness(sys, preorder, s, t, &mut memo))
}

fn sim_witness<L: Eq + Hash + Clone>(
    sys: &Lts<L>,
    preorder: &Preorder,
    s: StateId,
    t: StateId,
    memo: &mut HashMap<(StateId, StateId), Formula<L, bool>>,
) -> Formula<L, bool> {
    if let Some(f) = memo.get(&(s, t)) {
        return f.clone();
    }
    let round = preorder.removed_at(s, t);
    // The move of s that t could not match in `round`: every a-successor t'
    // of t was already refuted for s' in an earlier round.
    let (es, answers) = sys
        .successors(s)
        .iter()
        .find_map(|es| {
            let answers: Vec<StateId> = sys
                .successors(t)
                .iter()
                .filter(|et| et.label == es.label)
                .map(|et| et.target)
                .collect();
            answers
                .iter()
                .all(|&t2| preorder.removed_at(es.target, t2) < round)
                .then_some((es, answers))
        })
        .expect("a refuted pair has an unmatched move");

    let mut conj: Vec<Formula<L, bool>> = answers
        .into_iter()
        .map(|t2| sim_witness(sys, preorder, es.target, t2, memo))
        .collect();
    conj.dedup();
    let target = match conj.len() {
        0 => Formula::True,
        1 => conj.pop().expect("one conjunct"),
        _ => Formula::And(conj),
    };
    let f = Formula::Modal {
        label: sys.label(es.label).clone(),
        weight: true,
        target: Box::new(target),
    };
    memo.insert((s, t), f.clone());
    f
}
