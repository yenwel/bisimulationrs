//! Comparison up to causality, for the bindings: do two PEPA models not
//! only allow the same steps but also have the same dependencies between
//! them? See `bisim_refine::hp_bisimilar` for the method and
//! `PepaModel::causal_net` for the causal semantics.

use bisim_algebra::pepa::parse_pepa;
use bisim_core::Verdict;
use bisim_refine::{CausalBudget, bisimilar, hp_bisimilar};
use serde::Serialize;

use crate::observe::error;
use crate::{CompareError, Diagnostic, Options, diagnostic};

/// Size of a model's causal net.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NetSummary {
    /// One per local derivative of every component instance.
    pub places: usize,
    pub transitions: usize,
    /// Component instances (array copies counted one by one).
    pub components: usize,
}

/// The comparison of two models with and without causality.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CausalComparison {
    /// Same sequences of actions, step for step (rates ignored).
    pub interleaving_bisimilar: bool,
    /// When not: a formula true of `a` and false of `b`.
    pub interleaving_witness: Option<String>,
    /// `"holds"`, `"violated"` or `"unknown"` (search budget exhausted).
    pub causal: String,
    /// When violated: a formula with causes, true of `a` and false of `b`.
    pub causal_witness: Option<String>,
    pub nets: [NetSummary; 2],
    pub explanation: String,
}

/// Compares the initial states of two models up to causality
/// (history-preserving bisimulation of their causal nets, rates ignored).
pub fn causal_compare(
    source_a: &str,
    source_b: &str,
    options: Options,
) -> Result<CausalComparison, CompareError> {
    let net = |source: &str| -> Result<_, Vec<Diagnostic>> {
        let model = parse_pepa(source).map_err(|e| vec![diagnostic(&e)])?;
        model.causal_net().map_err(|e| vec![diagnostic(&e)])
    };
    let (a, b) = match (net(source_a), net(source_b)) {
        (Ok(a), Ok(b)) => (a, b),
        (a, b) => {
            return Err(CompareError {
                a: a.err().unwrap_or_default(),
                b: b.err().unwrap_or_default(),
            });
        }
    };
    let both = |e: Vec<Diagnostic>| CompareError { a: e.clone(), b: e };
    let summary = |n: &bisim_core::SafeNet<_>| NetSummary {
        places: n.places.len(),
        transitions: n.transitions.len(),
        components: n.initial.len(),
    };
    let graph = |n: &bisim_core::SafeNet<_>| {
        n.reachability_graph(options.max_states)
            .map(|(lts, _)| lts)
            .map_err(|e| both(error(e.to_string())))
    };
    let (ga, gb) = (graph(&a)?, graph(&b)?);
    let interleaving = bisimilar(&ga, &gb);
    let interleaving_witness = match &interleaving {
        Verdict::Violated { witness } => Some(witness.to_string()),
        _ => None,
    };
    let causal = hp_bisimilar(
        &a,
        &b,
        CausalBudget {
            max_positions: options.max_states.max(1_000),
        },
    )
    .map_err(|e| both(error(e.to_string())))?;
    let (verdict, causal_witness) = match &causal {
        Verdict::Holds => ("holds", None),
        Verdict::Violated { witness } => ("violated", Some(witness.to_string())),
        _ => ("unknown", None),
    };
    let explanation = match (
        interleaving.holds(),
        verdict,
        &causal_witness,
        &interleaving_witness,
    ) {
        (_, "holds", _, _) => {
            "Same behaviour and the same causal structure: every step in one model \
             depends on the matching earlier steps in the other."
                .to_owned()
        }
        (true, "violated", Some(w), _) => format!(
            "The models allow the same sequences of actions, so an ordinary comparison calls them \
             equal, but their steps depend on different earlier steps. Witness: {w}, true of A and \
             false of B. Read ⟨a ← 1,2⟩ as “an a that depends on the 1st and 2nd step so far” and \
             ⟨a ← ∅⟩ as “an a that depends on none of them”."
        ),
        (false, _, _, Some(w)) => format!(
            "The models already differ as sequences of actions: {w} holds in A and not in B, so \
             they differ causally too."
        ),
        _ => "Undecided within the search budget; raise max_states.".to_owned(),
    };
    Ok(CausalComparison {
        interleaving_bisimilar: interleaving.holds(),
        interleaving_witness,
        causal: verdict.to_owned(),
        causal_witness,
        nets: [summary(&a), summary(&b)],
        explanation,
    })
}
