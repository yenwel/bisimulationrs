//! (ε, δ)-bisimulation for continuous-time Markov chains (Spork, Baier,
//! Katoen, Klüppelholz, Piribauer, "Approximate Probabilistic Bisimulation
//! for Continuous-Time Markov Chains", CAV 2025).
//!
//! For a CTMC with state labels, exit rates `E` and jump probabilities `P`,
//! a reflexive, symmetric relation `R` is an (ε, δ)-bisimulation when every
//! related pair `(s, s′)` has
//!
//! - equal labels;
//! - exit rates within a factor `e^δ`: `|ln E(s) − ln E(s′)| ≤ δ`;
//! - jump probabilities within `ε` in the sense
//!   `P(s, A) ≤ P(s′, R(A)) + ε` for every set `A` (`R(A)`: the states
//!   related to some state of `A`).
//!
//! The probability condition is checked as a maximum flow: from the
//! successors of `s` (capacities `P(s, ·)`) along `R` to the successors of
//! `s′` (capacities `P(s′, ·)`), the flow is at least `1 − ε` exactly when
//! the condition holds for every `A` (max-flow/min-cut). The largest
//! (ε, δ)-bisimulation is found from above: start with the pairs that agree
//! on labels and rates and remove pairs failing the flow test until none
//! does. `∼₀,₀` is strong bisimilarity (rates into every class equal).
//!
//! The relation is *not* transitive in general: tolerances add up along a
//! chain of related states (`ε₁ + ε₂`, `δ₁ + δ₂`). The payoff is a bound on
//! time-bounded reachability, [`reach_bound`] (the paper's Proposition 3),
//! and a sharper one when only rates differ, [`rate_only_bound`]
//! (Section 5).
//!
//! Conventions here: transitions are summed over action labels; self-loops
//! are dropped (they do not change the state, so the CTMC is the same, as in
//! `Ctmc::from_wlts`); states with no exit are related only to each other.

use std::hash::Hash;

use bisim_core::{Real, StateId, Wlts};

use crate::flow::FlowNetwork;
use crate::kantorovich::MetricError;

/// Largest number of states (both systems together) accepted.
pub const MAX_EPSDELTA_STATES: usize = 2_000;

/// A CTMC as needed here: exit rates and jump probabilities per state.
#[derive(Debug, Clone)]
struct Jumps {
    exit: Vec<f64>,
    /// `(target, probability)`, targets distinct.
    prob: Vec<Vec<(usize, f64)>>,
}

impl Jumps {
    fn union<L: Eq + Hash + Clone>(left: &Wlts<L, Real>, right: &Wlts<L, Real>) -> Self {
        let n = left.num_states();
        let total = n + right.num_states();
        let mut rates: Vec<Vec<(usize, f64)>> = vec![Vec::new(); total];
        for (sys, offset) in [(left, 0), (right, n)] {
            for (s, _, t, w) in sys.transitions() {
                if s != t {
                    rates[s.index() + offset].push((t.index() + offset, w.value()));
                }
            }
        }
        let mut exit = vec![0.0; total];
        let prob = rates
            .into_iter()
            .enumerate()
            .map(|(s, mut row)| {
                row.sort_by_key(|e| e.0);
                row.dedup_by(|next, kept| {
                    let same = next.0 == kept.0;
                    if same {
                        kept.1 += next.1;
                    }
                    same
                });
                exit[s] = row.iter().map(|e| e.1).sum();
                row.into_iter().map(|(t, r)| (t, r / exit[s])).collect()
            })
            .collect();
        Self { exit, prob }
    }
}

/// Largest (ε, δ)-bisimulation on the disjoint union of two systems.
#[derive(Debug, Clone)]
pub struct EpsDeltaRelation {
    /// States of the left system come first, then the right system's.
    related: Vec<Vec<bool>>,
    left_states: usize,
}

impl EpsDeltaRelation {
    /// Are state `s` of the left system and `t` of the right one related?
    pub fn related(&self, s: StateId, t: StateId) -> bool {
        self.related[s.index()][self.left_states + t.index()]
    }

    /// Related pairs within the union (both orders, including `s ∼ s`).
    pub fn len(&self) -> usize {
        self.related
            .iter()
            .map(|r| r.iter().filter(|&&b| b).count())
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The largest (ε, δ)-bisimulation between `left` and `right`, with state
/// labels from `label_left` / `label_right`.
pub fn epsdelta<L, K>(
    left: &Wlts<L, Real>,
    right: &Wlts<L, Real>,
    label_left: impl Fn(StateId) -> K,
    label_right: impl Fn(StateId) -> K,
    epsilon: f64,
    delta: f64,
) -> Result<EpsDeltaRelation, MetricError>
where
    L: Eq + Hash + Clone,
    K: Eq,
{
    if !(epsilon >= 0.0 && delta >= 0.0 && epsilon.is_finite() && delta.is_finite()) {
        return Err(MetricError::Invalid(format!(
            "tolerances must be finite and non-negative, got ε = {epsilon}, δ = {delta}"
        )));
    }
    let n = left.num_states();
    let total = n + right.num_states();
    if total > MAX_EPSDELTA_STATES {
        return Err(MetricError::TooManyPairs(MAX_EPSDELTA_STATES));
    }
    let jumps = Jumps::union(left, right);
    let labels: Vec<K> = (0..n)
        .map(|i| label_left(StateId(i as u32)))
        .chain((0..right.num_states()).map(|i| label_right(StateId(i as u32))))
        .collect();
    let rates_close = |a: f64, b: f64| match (a > 0.0, b > 0.0) {
        (true, true) => (a.ln() - b.ln()).abs() <= delta + 1e-12,
        (false, false) => true,
        _ => false,
    };
    let mut related = vec![vec![false; total]; total];
    for s in 0..total {
        for t in s..total {
            let ok = labels[s] == labels[t] && rates_close(jumps.exit[s], jumps.exit[t]);
            related[s][t] = ok;
            related[t][s] = ok;
        }
    }
    // Remove pairs failing the flow test until none does.
    loop {
        let mut removed = false;
        for s in 0..total {
            for t in s + 1..total {
                if !related[s][t] || jumps.exit[s] == 0.0 {
                    continue;
                }
                let ok = flow_ok(&jumps, &related, s, t, epsilon)
                    && flow_ok(&jumps, &related, t, s, epsilon);
                if !ok {
                    related[s][t] = false;
                    related[t][s] = false;
                    removed = true;
                }
            }
        }
        if !removed {
            break;
        }
    }
    Ok(EpsDeltaRelation {
        related,
        left_states: n,
    })
}

/// Does the flow from `s`'s successors along `related` to `t`'s successors
/// reach `1 − ε`?
fn flow_ok(jumps: &Jumps, related: &[Vec<bool>], s: usize, t: usize, epsilon: f64) -> bool {
    let (ps, pt) = (&jumps.prob[s], &jumps.prob[t]);
    let (a, b) = (ps.len(), pt.len());
    let (source, sink) = (a + b, a + b + 1);
    let mut net = FlowNetwork::new(a + b + 2);
    for (i, &(_, p)) in ps.iter().enumerate() {
        net.add_edge(source, i, p, 0.0);
    }
    for (j, &(_, p)) in pt.iter().enumerate() {
        net.add_edge(a + j, sink, p, 0.0);
    }
    for (i, &(u, _)) in ps.iter().enumerate() {
        for (j, &(v, _)) in pt.iter().enumerate() {
            if related[u][v] {
                net.add_edge(i, a + j, f64::INFINITY, 0.0);
            }
        }
    }
    net.max_flow(source, sink) >= 1.0 - epsilon - 1e-9
}

/// Are the initial states (ε, δ)-bisimilar?
pub fn epsdelta_bisimilar<L, K>(
    left: &Wlts<L, Real>,
    right: &Wlts<L, Real>,
    label_left: impl Fn(StateId) -> K,
    label_right: impl Fn(StateId) -> K,
    epsilon: f64,
    delta: f64,
) -> Result<bool, MetricError>
where
    L: Eq + Hash + Clone,
    K: Eq,
{
    let (s, t) = initials(left, right)?;
    Ok(epsdelta(left, right, label_left, label_right, epsilon, delta)?.related(s, t))
}

fn initials<L: Eq + Hash + Clone>(
    left: &Wlts<L, Real>,
    right: &Wlts<L, Real>,
) -> Result<(StateId, StateId), MetricError> {
    match (left.initial(), right.initial()) {
        (Some(s), Some(t)) => Ok((s, t)),
        _ => Err(MetricError::Invalid(
            "both systems need an initial state".into(),
        )),
    }
}

/// The smallest ε (within `1e-4`) at which the initial states are
/// (ε, δ)-bisimilar, or `None` if not even at ε = 1 (labels or rates
/// differ beyond `δ`). Bisection is valid because the relation grows with ε.
pub fn min_epsilon<L, K>(
    left: &Wlts<L, Real>,
    right: &Wlts<L, Real>,
    label_left: impl Fn(StateId) -> K + Copy,
    label_right: impl Fn(StateId) -> K + Copy,
    delta: f64,
) -> Result<Option<f64>, MetricError>
where
    L: Eq + Hash + Clone,
    K: Eq,
{
    let holds = |e: f64| epsdelta_bisimilar(left, right, label_left, label_right, e, delta);
    if !holds(1.0)? {
        return Ok(None);
    }
    if holds(0.0)? {
        return Ok(Some(0.0));
    }
    let (mut lo, mut hi) = (0.0, 1.0);
    while hi - lo > 1e-4 {
        let mid = 0.5 * (lo + hi);
        if holds(mid)? {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Ok(Some(hi))
}

/// Proposition 3: for (ε, δ)-bisimilar states and a unique absorbing goal,
/// the probabilities of reaching the goal within time `t` differ by at most
/// `1 − exp(−q·t·(e^δ·(1 + ε) − 1))`, with `q` at least every exit rate.
pub fn reach_bound(epsilon: f64, delta: f64, q: f64, t: f64) -> f64 {
    1.0 - (-q * t * (delta.exp() * (1.0 + epsilon) - 1.0)).exp()
}

/// Section 5: when only rates differ (ε = 0) and all exit rates equal `e`,
/// the bound `Diff_t(E_N)` of the Erlang chains with `N = ⌈(e^δ − 1)·t′/δ⌉`
/// at rescaled time `t′ = e·t`, where
/// `Diff_t(E_n) = Σ_{k<n} tᵏ/k! · (e^{−t} − cᵏ e^{−ct})`, `c = e^δ`.
pub fn rate_only_bound(delta: f64, exit_rate: f64, t: f64) -> f64 {
    if delta == 0.0 {
        return 0.0;
    }
    let t = exit_rate * t;
    let c = delta.exp();
    let n = ((c - 1.0) * t / delta).ceil().max(1.0) as usize;
    // Terms in log space: k·ln t − ln k! − t, and k·ln(ct) − ln k! − ct.
    let (mut sum, mut log_fact) = (0.0, 0.0);
    for k in 0..n {
        if k > 0 {
            log_fact += (k as f64).ln();
        }
        let kf = k as f64;
        let a = if t > 0.0 {
            (kf * t.ln() - log_fact - t).exp()
        } else if k == 0 {
            1.0
        } else {
            0.0
        };
        let b = if t > 0.0 {
            (kf * (c * t).ln() - log_fact - c * t).exp()
        } else if k == 0 {
            1.0
        } else {
            0.0
        };
        sum += a - b;
    }
    sum
}
