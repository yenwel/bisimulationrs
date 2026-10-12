use std::hash::Hash;

use bisim_core::{ComparisonVerdict, Formula, Lts, Semiring, StateId, Verdict, Wlts};

use crate::refine::{distinguishing_formula, refine};
use crate::simulation::{simulation_preorder, simulation_witness};

/// Minimal system bisimilar to `sys` (quotient by bisimilarity).
pub fn minimise<L: Eq + Hash + Clone, W: Semiring>(sys: &Wlts<L, W>) -> Wlts<L, W> {
    sys.quotient(refine(sys, None).partition())
}

/// Decides whether state `s` of `left` and state `t` of `right` are bisimilar.
///
/// Works for any weight semiring: strong bisimilarity for `bool`, Markovian
/// bisimilarity (lumpability) for rates. The systems may have different numbers
/// of states. A violation carries a formula true in `s` and false in `t`.
pub fn bisimilar_states<L, W>(
    left: &Wlts<L, W>,
    s: StateId,
    right: &Wlts<L, W>,
    t: StateId,
) -> Verdict<Formula<L, W::Key>>
where
    L: Eq + Hash + Clone,
    W: Semiring,
{
    let union = left.disjoint_union(right);
    let t = StateId(t.0 + left.num_states() as u32);
    let refinement = refine(&union, None);
    match distinguishing_formula(&union, &refinement, s, t) {
        None => Verdict::Holds,
        Some(witness) => Verdict::Violated { witness },
    }
}

/// Bisimilarity of the initial states of two systems.
///
/// # Panics
/// If either system has no initial state.
pub fn bisimilar<L, W>(left: &Wlts<L, W>, right: &Wlts<L, W>) -> Verdict<Formula<L, W::Key>>
where
    L: Eq + Hash + Clone,
    W: Semiring,
{
    bisimilar_states(left, initial(left), right, initial(right))
}

/// Simulation, mutual similarity and bisimilarity of the initial states of two
/// LTSs. Every violation carries a distinguishing formula that holds on the
/// left and fails on the right (or vice versa for the reverse simulation).
///
/// # Panics
/// If either system has no initial state.
pub fn compare<L: Eq + Hash + Clone>(
    left: &Lts<L>,
    right: &Lts<L>,
) -> ComparisonVerdict<Formula<L, bool>> {
    let union = left.disjoint_union(right);
    let s = initial(left);
    let t = StateId(initial(right).0 + left.num_states() as u32);

    let preorder = simulation_preorder(&union);
    let as_verdict = |w: Option<Formula<L, bool>>| match w {
        None => Verdict::Holds,
        Some(witness) => Verdict::Violated { witness },
    };
    let right_simulates_left = as_verdict(simulation_witness(&union, &preorder, s, t));
    let mutually_similar = match &right_simulates_left {
        Verdict::Holds => as_verdict(simulation_witness(&union, &preorder, t, s)),
        violated => violated.clone(),
    };
    let bisimilar = match &mutually_similar {
        Verdict::Holds => {
            let refinement = refine(&union, None);
            as_verdict(distinguishing_formula(&union, &refinement, s, t))
        }
        violated => violated.clone(),
    };
    ComparisonVerdict {
        right_simulates_left,
        mutually_similar,
        bisimilar,
    }
}

fn initial<L: Eq + Hash + Clone, W: Semiring>(sys: &Wlts<L, W>) -> StateId {
    sys.initial().expect("system needs an initial state")
}
