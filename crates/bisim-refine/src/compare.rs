use std::hash::Hash;

use bisim_core::{ComparisonVerdict, Formula, Lts, Semiring, StateId, Verdict, Wlts};

use crate::fast::refine_fast;
use crate::refine::{distinguishing_formula, refine};
use crate::simulation::{simulation_preorder, simulation_witness};

/// Minimal system bisimilar to `sys` (quotient by bisimilarity).
pub fn minimise<L: Eq + Hash + Clone, W: Semiring>(sys: &Wlts<L, W>) -> Wlts<L, W> {
    sys.quotient(&refine_fast(sys, None))
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
    distinguish(&union, s, t)
}

/// `Holds` if `s` and `t` are bisimilar in `union`, else a witness.
///
/// The worklist backend decides; the round-by-round refinement that
/// explains a difference only runs when there is one to explain.
fn distinguish<L, W>(union: &Wlts<L, W>, s: StateId, t: StateId) -> Verdict<Formula<L, W::Key>>
where
    L: Eq + Hash + Clone,
    W: Semiring,
{
    if refine_fast(union, None).same_block(s, t) {
        return Verdict::Holds;
    }
    let refinement = refine(union, None);
    match distinguishing_formula(union, &refinement, s, t) {
        None => Verdict::Holds,
        Some(witness) => Verdict::Violated { witness },
    }
}

/// A system without an initial state cannot be compared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoInitialState;

impl std::fmt::Display for NoInitialState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "the system has no initial state")
    }
}

impl std::error::Error for NoInitialState {}

/// Bisimilarity of the initial states of two systems.
///
/// # Panics
/// If either system has no initial state; see [`try_bisimilar`].
pub fn bisimilar<L, W>(left: &Wlts<L, W>, right: &Wlts<L, W>) -> Verdict<Formula<L, W::Key>>
where
    L: Eq + Hash + Clone,
    W: Semiring,
{
    bisimilar_states(left, initial(left), right, initial(right))
}

/// [`bisimilar`], or an error if a system has no initial state.
pub fn try_bisimilar<L, W>(
    left: &Wlts<L, W>,
    right: &Wlts<L, W>,
) -> Result<Verdict<Formula<L, W::Key>>, NoInitialState>
where
    L: Eq + Hash + Clone,
    W: Semiring,
{
    let (s, t) = left.initial().zip(right.initial()).ok_or(NoInitialState)?;
    Ok(bisimilar_states(left, s, right, t))
}

/// Simulation, mutual similarity and bisimilarity of the initial states of two
/// LTSs. Every violation carries a distinguishing formula that holds on the
/// left and fails on the right (or vice versa for the reverse simulation).
///
/// # Panics
/// If either system has no initial state; see [`try_compare`].
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
        Verdict::Holds => distinguish(&union, s, t),
        violated => violated.clone(),
    };
    ComparisonVerdict {
        right_simulates_left,
        mutually_similar,
        bisimilar,
    }
}

/// [`compare`], or an error if a system has no initial state.
pub fn try_compare<L: Eq + Hash + Clone>(
    left: &Lts<L>,
    right: &Lts<L>,
) -> Result<ComparisonVerdict<Formula<L, bool>>, NoInitialState> {
    if left.initial().is_none() || right.initial().is_none() {
        return Err(NoInitialState);
    }
    Ok(compare(left, right))
}

fn initial<L: Eq + Hash + Clone, W: Semiring>(sys: &Wlts<L, W>) -> StateId {
    sys.initial().expect("system needs an initial state")
}
