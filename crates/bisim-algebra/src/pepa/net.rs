//! Causal semantics of PEPA: the model as a safe Petri net.
//!
//! Every sequential component instance is a token moving through places,
//! one place per local derivative; each copy of an array `P[n]` is its own
//! instance. A local move is a transition from one place to another; a
//! shared action `a ∈ L` in `P ⋈_L Q` is one transition that moves a
//! participant on each side at once; hiding renames to τ. A step then
//! depends on an earlier one exactly when it moves a component that the
//! earlier step moved: the causality of the process algebra, which the
//! interleaving state space forgets. This is the standard net semantics of
//! process algebras with synchronisation; Ribaudo (PNPM 1995) gives one for
//! stochastic process algebras.
//!
//! Rates are left out: this is the untimed causal structure. A *functional*
//! rate (`#Name`) can be zero in some states, which would disable a step the
//! net allows, so such models are refused rather than misrepresented. The
//! reachability graph of the net is bisimilar to the untimed derivation
//! (checked in the tests).

use bisim_core::{NetTransition, SafeNet};

use super::ast::{Action, ModelTerm, PepaModel, SeqId};
use super::derive::{Act, Local};
use super::rate::Rate;
use super::validate::PepaError;

/// Most sequential component instances (array copies counted one by one).
pub const MAX_CAUSAL_COMPONENTS: usize = 64;

type Step<A> = (Act<A>, Vec<u32>, Vec<u32>);

struct Builder<'a, A> {
    model: &'a PepaModel<A>,
    local: Local<A>,
    names: Vec<String>,
    places: Vec<String>,
    initial: Vec<u32>,
    instances: usize,
}

impl<A: Action> Builder<'_, A> {
    /// One instance of sequential component `id`; `copy` numbers array
    /// copies for the place names.
    fn instance(&mut self, id: SeqId, copy: Option<u32>) -> Result<Vec<Step<A>>, PepaError> {
        self.instances += 1;
        if self.instances > MAX_CAUSAL_COMPONENTS {
            return Err(PepaError::NoCausalSemantics {
                reason: format!(
                    "more than {MAX_CAUSAL_COMPONENTS} component instances (array copies count one by one)"
                ),
            });
        }
        let name = self.model.seq_name(id).to_owned();
        let owner = match copy {
            Some(c) => format!("{name}#{c}"),
            None => name.clone(),
        };
        let nodes = self.local.reachable(self.local.root(id));
        let first = self.places.len() as u32;
        for &n in &nodes {
            self.places
                .push(format!("{owner}.{}", self.names[n as usize]));
        }
        self.initial.push(first);
        let place = |n: u32| first + nodes.iter().position(|&m| m == n).expect("reachable") as u32;
        let mut steps = Vec::new();
        let mut buf = Vec::new();
        for &n in &nodes {
            buf.clear();
            self.local.transitions(n, &mut buf);
            for (a, rate, t) in buf.drain(..) {
                if let Rate::Functional(_) = rate {
                    return Err(PepaError::NoCausalSemantics {
                        reason: format!(
                            "`{a}` in `{name}` has a functional rate, so whether it can happen \
                             depends on counts elsewhere"
                        ),
                    });
                }
                steps.push((Act::Act(a), vec![place(n)], vec![place(t)]));
            }
        }
        Ok(steps)
    }

    fn term(&mut self, t: &ModelTerm<A>) -> Result<Vec<Step<A>>, PepaError> {
        Ok(match t {
            ModelTerm::Component(id) => self.instance(*id, None)?,
            ModelTerm::Array(id, n) => {
                let mut steps = Vec::new();
                for c in 1..=*n {
                    steps.extend(self.instance(*id, Some(c))?);
                }
                steps
            }
            ModelTerm::Hide(x, set) => self
                .term(x)?
                .into_iter()
                .map(|(a, pre, post)| match a {
                    Act::Act(b) if set.contains(&b) => (Act::Tau, pre, post),
                    other => (other, pre, post),
                })
                .collect(),
            ModelTerm::Coop(l, set, r) => {
                let (left, right) = (self.term(l)?, self.term(r)?);
                let shared = |a: &Act<A>| matches!(a, Act::Act(b) if set.contains(b));
                let mut steps: Vec<Step<A>> = left
                    .iter()
                    .chain(&right)
                    .filter(|(a, _, _)| !shared(a))
                    .cloned()
                    .collect();
                for (a, pl, ql) in left.iter().filter(|s| shared(&s.0)) {
                    for (b, pr, qr) in right.iter().filter(|s| shared(&s.0)) {
                        if a == b {
                            let pre = pl.iter().chain(pr).copied().collect();
                            let post = ql.iter().chain(qr).copied().collect();
                            steps.push((a.clone(), pre, post));
                        }
                    }
                }
                steps
            }
        })
    }
}

impl<A: Action> PepaModel<A> {
    /// The causal semantics of the model as a safe Petri net (see the module
    /// documentation). Place names read `Component.Derivative`, with `#k`
    /// for the `k`-th copy of an array.
    pub fn causal_net(&self) -> Result<SafeNet<Act<A>>, PepaError> {
        self.validate()?;
        let (local, names) = Local::compile(self)?;
        let mut b = Builder {
            model: self,
            local,
            names,
            places: Vec::new(),
            initial: Vec::new(),
            instances: 0,
        };
        let steps = b.term(&self.system)?;
        let transitions = steps
            .into_iter()
            .map(|(label, pre, post)| NetTransition { label, pre, post })
            .collect();
        Ok(SafeNet::new(b.places, transitions, b.initial))
    }
}
