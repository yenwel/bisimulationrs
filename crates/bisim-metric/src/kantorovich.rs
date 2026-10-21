//! Bisimulation distance: how differently two Markov chains behave, as a
//! number in `[0, 1]`.
//!
//! The distance is the optimal-transport (Kantorovich) bisimulation metric
//! with discount `γ`, in the form of Calo, Jonsson, Neu, Schwartz and
//! Segovia-Aguas, "Bisimulation metrics are optimal transport distances,
//! and can be computed efficiently" (NeurIPS 2024): with an observation
//! `r(x) ∈ [0, 1]` on every state and ground cost
//! `c(x, y) = (1 − γ)·|r(x) − r(y)|`, the distance is the least fixed point
//!
//! ```text
//! d(x, y) = c(x, y) + γ · min over couplings π of P(x,·), P(y,·) of Σ π·d.
//! ```
//!
//! It is the smallest expected discounted observation gap over all ways of
//! running the two chains side by side (bicausal couplings). Since the
//! weights `(1 − γ)γᵗ` sum to 1, it reads as "the time-averaged difference in
//! the observation, with weights halving every half-life, when the two
//! systems are made to look as alike as they can". It is 0 exactly when the
//! two states are bisimilar for the observation (equal observations, matched
//! step by step), and it is a pseudometric.
//!
//! Rate-labelled systems are compared as uniformised discrete-time chains
//! (one common uniformisation rate `q` for both, so a step is the same
//! `1/q` of model time on both sides). The discount is given as a
//! *half-life* in model time: `γ = 2^(−1/(q·half_life))`.
//!
//! The solver is policy iteration on the optimal-transport Bellman
//! operator: couplings are improved by exact transport on the small
//! supports of sparse chains, and each set of couplings is evaluated by
//! Gauss–Seidel sweeps. The result reports a certified error bound from the
//! Bellman residual.

use std::collections::HashMap;
use std::hash::Hash;

use bisim_core::{Real, StateId, Wlts};

use crate::flow::transport;

/// Largest number of state pairs explored from the initial pair.
pub const MAX_METRIC_PAIRS: usize = 2_000_000;

/// A discrete-time Markov chain with an observation on every state.
#[derive(Debug, Clone)]
pub struct Chain {
    /// Transition probabilities per state (each row sums to 1).
    pub rows: Vec<Vec<(usize, f64)>>,
    /// Observation per state, in `[0, 1]`.
    pub observation: Vec<f64>,
    pub initial: usize,
}

/// Options for [`distance`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricOptions {
    /// Target accuracy of the distance.
    pub tolerance: f64,
    /// Bound on policy-improvement rounds.
    pub max_rounds: usize,
    /// How many pairs to list in [`Distance::pairs`].
    pub top: usize,
}

impl Default for MetricOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-6,
            max_rounds: 200,
            top: 10,
        }
    }
}

/// A state pair's share of the distance.
#[derive(Debug, Clone, PartialEq)]
pub struct PairContribution {
    pub left: usize,
    pub right: usize,
    /// `|r(x) − r(y)|` (normalised observations).
    pub gap: f64,
    /// Discounted time the optimal coupling spends in the pair, as a
    /// fraction (the occupancies sum to 1).
    pub occupancy: f64,
    /// `occupancy · gap`: these add up to the distance.
    pub contribution: f64,
}

/// The distance between the initial states, with how it arises.
#[derive(Debug, Clone, PartialEq)]
pub struct Distance {
    pub value: f64,
    /// `‖T V − V‖∞ / (1 − γ)`: the true distance is within this of `value`.
    pub error_bound: f64,
    pub gamma: f64,
    /// Pairs explored.
    pub pairs_explored: usize,
    /// The pairs contributing most, largest first.
    pub pairs: Vec<PairContribution>,
    /// Policy-improvement rounds used.
    pub rounds: usize,
}

/// Errors of the distance computations.
#[derive(Debug, Clone, PartialEq)]
pub enum MetricError {
    /// More than [`MAX_METRIC_PAIRS`] reachable pairs.
    TooManyPairs(usize),
    /// A parameter is out of range (message).
    Invalid(String),
}

impl std::fmt::Display for MetricError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooManyPairs(n) => write!(f, "more than {n} reachable state pairs"),
            Self::Invalid(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for MetricError {}

/// Uniformisation rate covering both systems: the largest exit rate
/// (self-loops excluded), or 1 if nothing moves.
pub fn common_rate<L: Eq + Hash + Clone>(a: &Wlts<L, Real>, b: &Wlts<L, Real>) -> f64 {
    let max_exit = |sys: &Wlts<L, Real>| {
        let mut exit = vec![0.0; sys.num_states()];
        for (s, _, t, w) in sys.transitions() {
            if s != t {
                exit[s.index()] += w.value();
            }
        }
        exit.into_iter().fold(0.0f64, f64::max)
    };
    let q = max_exit(a).max(max_exit(b));
    if q > 0.0 { q } else { 1.0 }
}

/// The discount for a half-life in model time at uniformisation rate `q`.
pub fn discount(q: f64, half_life: f64) -> f64 {
    2f64.powf(-1.0 / (q * half_life))
}

/// The uniformised chain of `sys` at rate `q ≥` every exit rate, with raw
/// observations (normalise with [`normalise`]).
///
/// # Panics
/// If `sys` has no initial state.
pub fn uniformise<L: Eq + Hash + Clone>(
    sys: &Wlts<L, Real>,
    q: f64,
    observe: impl Fn(StateId) -> f64,
) -> Chain {
    let n = sys.num_states();
    let mut out: Vec<HashMap<usize, f64>> = vec![HashMap::new(); n];
    let mut exit = vec![0.0; n];
    for (s, _, t, w) in sys.transitions() {
        if s == t {
            continue;
        }
        *out[s.index()].entry(t.index()).or_insert(0.0) += w.value() / q;
        exit[s.index()] += w.value();
    }
    let rows = out
        .into_iter()
        .enumerate()
        .map(|(s, m)| {
            let mut row: Vec<(usize, f64)> = m.into_iter().collect();
            let stay = 1.0 - exit[s] / q;
            if stay > 1e-15 {
                row.push((s, stay));
            }
            row.sort_by_key(|&(t, _)| t);
            row
        })
        .collect();
    Chain {
        rows,
        observation: sys.states().map(observe).collect(),
        initial: sys.initial().expect("system has no initial state").index(),
    }
}

/// Rescales the observations of both chains jointly to `[0, 1]` (min–max
/// over both); returns the scale (max − min), 0 if constant.
pub fn normalise(a: &mut Chain, b: &mut Chain) -> f64 {
    let all = a.observation.iter().chain(&b.observation);
    let lo = all.clone().copied().fold(f64::INFINITY, f64::min);
    let hi = all.copied().fold(f64::NEG_INFINITY, f64::max);
    let scale = hi - lo;
    for r in a.observation.iter_mut().chain(b.observation.iter_mut()) {
        *r = if scale > 0.0 { (*r - lo) / scale } else { 0.0 };
    }
    scale
}

/// The bisimulation distance between the initial states of two chains
/// (observations already in `[0, 1]`) at discount `gamma ∈ (0, 1)`.
pub fn distance(
    a: &Chain,
    b: &Chain,
    gamma: f64,
    options: MetricOptions,
) -> Result<Distance, MetricError> {
    if !(gamma > 0.0 && gamma < 1.0) {
        return Err(MetricError::Invalid(format!(
            "discount {gamma} not in (0, 1)"
        )));
    }
    if a.observation
        .iter()
        .chain(&b.observation)
        .any(|r| !(0.0..=1.0).contains(r))
    {
        return Err(MetricError::Invalid(
            "observations must be in [0, 1]".into(),
        ));
    }
    let pairs = Pairs::explore(a, b)?;
    let n = pairs.list.len();
    let cost: Vec<f64> = pairs
        .list
        .iter()
        .map(|&(x, y)| (1.0 - gamma) * (a.observation[x] - b.observation[y]).abs())
        .collect();

    // Start from the independent coupling.
    let mut policy: Vec<Vec<(usize, f64)>> = pairs
        .list
        .iter()
        .map(|&(x, y)| {
            let mut c = Vec::new();
            for &(x2, px) in &a.rows[x] {
                for &(y2, py) in &b.rows[y] {
                    c.push((pairs.index[&(x2, y2)], px * py));
                }
            }
            c
        })
        .collect();
    let mut v = vec![0.0; n];
    let tol = options.tolerance.max(1e-12);
    let mut rounds = 0;
    let mut residual = f64::INFINITY;
    while rounds < options.max_rounds {
        rounds += 1;
        evaluate(&policy, &cost, gamma, &mut v, tol * (1.0 - gamma) * 0.1);
        // Improvement: exact transport against the current values.
        residual = 0.0;
        let mut improved = false;
        for (k, &(x, y)) in pairs.list.iter().enumerate() {
            let (best, coupling) = best_coupling(a, b, &pairs, x, y, &v);
            let current: f64 = policy[k].iter().map(|&(j, m)| m * v[j]).sum();
            let backup = cost[k] + gamma * best;
            residual = f64::max(residual, (backup - v[k]).abs());
            if best < current - 1e-12 {
                policy[k] = coupling;
                improved = true;
            }
        }
        if !improved || residual <= tol * (1.0 - gamma) {
            break;
        }
    }
    let error_bound = residual / (1.0 - gamma);
    let occupancy = occupancy(&policy, gamma, n);
    let mut contributions: Vec<PairContribution> = pairs
        .list
        .iter()
        .zip(&occupancy)
        .map(|(&(x, y), &o)| {
            let gap = (a.observation[x] - b.observation[y]).abs();
            PairContribution {
                left: x,
                right: y,
                gap,
                occupancy: o,
                contribution: o * gap,
            }
        })
        .filter(|c| c.contribution > 0.0)
        .collect();
    contributions.sort_by(|p, q| q.contribution.total_cmp(&p.contribution));
    contributions.truncate(options.top);
    Ok(Distance {
        value: v[0],
        error_bound,
        gamma,
        pairs_explored: n,
        pairs: contributions,
        rounds,
    })
}

/// Bisimulation distance between state `s` of `left` and `t` of `right`
/// (given as `(system, state)` pairs)
/// for the observation `observe` (any real; rescaled jointly to `[0, 1]`),
/// with differences `half_life` model-time units ahead counting half.
pub fn bisim_distance<L: Eq + Hash + Clone>(
    (left, s): (&Wlts<L, Real>, StateId),
    (right, t): (&Wlts<L, Real>, StateId),
    observe_left: impl Fn(StateId) -> f64,
    observe_right: impl Fn(StateId) -> f64,
    half_life: f64,
    options: MetricOptions,
) -> Result<Distance, MetricError> {
    if !(half_life > 0.0 && half_life.is_finite()) {
        return Err(MetricError::Invalid(format!(
            "half-life {half_life} must be positive"
        )));
    }
    let q = common_rate(left, right);
    let mut a = uniformise(left, q, observe_left);
    let mut b = uniformise(right, q, observe_right);
    a.initial = s.index();
    b.initial = t.index();
    normalise(&mut a, &mut b);
    distance(&a, &b, discount(q, half_life), options)
}

/// The pairs reachable from the initial pair, the initial pair first.
struct Pairs {
    list: Vec<(usize, usize)>,
    index: HashMap<(usize, usize), usize>,
}

impl Pairs {
    fn explore(a: &Chain, b: &Chain) -> Result<Self, MetricError> {
        let root = (a.initial, b.initial);
        let mut pairs = Self {
            list: vec![root],
            index: HashMap::from([(root, 0)]),
        };
        let mut k = 0;
        while k < pairs.list.len() {
            let (x, y) = pairs.list[k];
            for &(x2, _) in &a.rows[x] {
                for &(y2, _) in &b.rows[y] {
                    if !pairs.index.contains_key(&(x2, y2)) {
                        if pairs.list.len() >= MAX_METRIC_PAIRS {
                            return Err(MetricError::TooManyPairs(MAX_METRIC_PAIRS));
                        }
                        pairs.index.insert((x2, y2), pairs.list.len());
                        pairs.list.push((x2, y2));
                    }
                }
            }
            k += 1;
        }
        Ok(pairs)
    }
}

/// The cheapest coupling of the successor distributions of `x` and `y`
/// against values `v`: `(Σ π·v, π as (pair, mass))`.
fn best_coupling(
    a: &Chain,
    b: &Chain,
    pairs: &Pairs,
    x: usize,
    y: usize,
    v: &[f64],
) -> (f64, Vec<(usize, f64)>) {
    let (ra, rb) = (&a.rows[x], &b.rows[y]);
    if ra.len() == 1 || rb.len() == 1 {
        // A point mass on one side: only one coupling exists.
        let c: Vec<(usize, f64)> = ra
            .iter()
            .flat_map(|&(x2, px)| rb.iter().map(move |&(y2, py)| ((x2, y2), px * py)))
            .map(|(p, m)| (pairs.index[&p], m))
            .collect();
        let value = c.iter().map(|&(j, m)| m * v[j]).sum();
        return (value, c);
    }
    let pa: Vec<f64> = ra.iter().map(|e| e.1).collect();
    let pb: Vec<f64> = rb.iter().map(|e| e.1).collect();
    let cost: Vec<Vec<f64>> = ra
        .iter()
        .map(|&(x2, _)| {
            rb.iter()
                .map(|&(y2, _)| v[pairs.index[&(x2, y2)]])
                .collect()
        })
        .collect();
    // Rows sum to 1 up to rounding; rescale b to a's total.
    let (ta, tb): (f64, f64) = (pa.iter().sum(), pb.iter().sum());
    let pb: Vec<f64> = pb.iter().map(|p| p * ta / tb).collect();
    let (value, plan) = transport(&pa, &pb, &cost);
    let coupling = plan
        .into_iter()
        .map(|(i, j, m)| (pairs.index[&(ra[i].0, rb[j].0)], m))
        .collect();
    (value, coupling)
}

/// Gauss–Seidel evaluation of `V = c + γ Pπ V` to step size `tol`.
fn evaluate(policy: &[Vec<(usize, f64)>], cost: &[f64], gamma: f64, v: &mut [f64], tol: f64) {
    for _ in 0..1_000_000 {
        let mut delta = 0.0f64;
        for k in 0..v.len() {
            let next = cost[k] + gamma * policy[k].iter().map(|&(j, m)| m * v[j]).sum::<f64>();
            delta = delta.max((next - v[k]).abs());
            v[k] = next;
        }
        if delta <= tol {
            return;
        }
    }
}

/// Discounted occupancy `(1 − γ) Σₜ γᵗ Pr[pair at t]` under the couplings.
fn occupancy(policy: &[Vec<(usize, f64)>], gamma: f64, n: usize) -> Vec<f64> {
    let mut mu = vec![0.0; n];
    let mut current = vec![0.0; n];
    current[0] = 1.0 - gamma;
    let mut mass = 1.0 - gamma;
    while mass > 1e-12 {
        let mut next = vec![0.0; n];
        for (k, &p) in current.iter().enumerate() {
            if p == 0.0 {
                continue;
            }
            mu[k] += p;
            for &(j, m) in &policy[k] {
                next[j] += gamma * p * m;
            }
        }
        mass = next.iter().sum();
        current = next;
    }
    mu
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain(rows: Vec<Vec<(usize, f64)>>, observation: Vec<f64>) -> Chain {
        Chain {
            rows,
            observation,
            initial: 0,
        }
    }

    #[test]
    fn two_absorbing_states_with_different_observations() {
        // x stays on 0, y stays on 1: the gap is 1 at every step.
        let a = chain(vec![vec![(0, 1.0)]], vec![0.0]);
        let b = chain(vec![vec![(0, 1.0)]], vec![1.0]);
        let d = distance(&a, &b, 0.9, MetricOptions::default()).unwrap();
        assert!((d.value - 1.0).abs() < 1e-6, "{d:?}");
    }

    #[test]
    fn delayed_difference_is_discounted() {
        // Both start at observation 0; a jumps to 1 after one step, b stays.
        let a = chain(vec![vec![(1, 1.0)], vec![(1, 1.0)]], vec![0.0, 1.0]);
        let b = chain(vec![vec![(0, 1.0)]], vec![0.0]);
        let g = 0.8;
        let d = distance(&a, &b, g, MetricOptions::default()).unwrap();
        // (1 − γ) Σ_{t≥1} γᵗ = γ.
        assert!((d.value - g).abs() < 1e-6, "{d:?}");
        let total: f64 = d.pairs.iter().map(|p| p.contribution).sum();
        assert!((total - d.value).abs() < 1e-6);
    }

    #[test]
    fn coupling_matches_mixtures() {
        // a: fair coin between observation 0 and 1 (absorbing); b the
        // same with the outcomes listed in the other order: distance 0.
        let a = chain(
            vec![vec![(1, 0.5), (2, 0.5)], vec![(1, 1.0)], vec![(2, 1.0)]],
            vec![0.0, 0.0, 1.0],
        );
        let b = chain(
            vec![vec![(1, 0.5), (2, 0.5)], vec![(1, 1.0)], vec![(2, 1.0)]],
            vec![0.0, 1.0, 0.0],
        );
        let d = distance(&a, &b, 0.9, MetricOptions::default()).unwrap();
        assert!(d.value.abs() < 1e-6, "{d:?}");
    }
}
