//! Evidence against a plan, valid whenever you look.
//!
//! The plan is a CTMC. What is observed is its state up to an observation
//! (for example counts per local state) and the label of every firing. If
//! the observation is a Markovian lumping of the plan, every system that is
//! Markovian bisimilar to the plan, on any state space, gives the observed
//! process the same law: the plan's quotient chain. "The system is
//! bisimilar to the plan" is then a single hypothesis about what is
//! observed, and a likelihood ratio against the quotient is a fair bet for
//! all those systems at once.
//!
//! [`PlanEvidence`] keeps that bet up to date as events arrive: a mixture
//! over alternative rates (Gamma priors centred on the plan's rates; see
//! [`Pooling`]) against the plan's quotient. Under the plan its value `E`
//! is a non-negative martingale starting at 1, so by Ville's inequality the
//! probability that `E` *ever* reaches `1/α` is at most `α`. "Off plan" at
//! level `1 − α` is therefore valid at any time you choose to look, and
//! once reached it stays. A move the plan cannot make sets `E = ∞`: the
//! logical verdict is the limit case.
//!
//! What it cannot see: the bet compares *average* rates of observed
//! transitions. A system whose observed rates average out to the plan's
//! (non-exponential holding times, individuals that differ but average
//! out) differs in law but is not detected. [`divergences`] gives the
//! growth rate of the bet against a given alternative, and the growth rate
//! a tester that also saw the internal states could reach.

use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;

use bisim_core::{LabelId, Real, StateId, Wlts};

use crate::Ctmc;
use crate::simulate::ln_gamma;

/// What the bet in [`PlanEvidence`] puts its prior on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Pooling {
    /// One factor per label, shared by every transition with that label:
    /// all rates of an activity may be off by the same factor (the herd
    /// heals faster than planned). Half the stake bets that exactly one
    /// activity is off (shared evenly over the activities), half that any
    /// of them may be. Few unknowns, so such deviations are found quickly.
    PerLabel,
    /// One rate per transition of the quotient: also finds deviations that
    /// depend on the state, at the cost of more unknowns.
    PerTransition,
    /// The average of both bets. It is never worse than the better one by
    /// more than a factor 2 (`ln 2` nats).
    #[default]
    Both,
}

/// Settings of [`PlanEvidence`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvidenceOptions {
    /// Shape of the Gamma prior on every rate (or factor), whose mean is the
    /// plan's. Small values bet on large deviations, large values on small
    /// ones. Default 1 (an exponential prior).
    pub shape: f64,
    /// Confidence level of the verdict: off plan once `E ≥ 1 / (1 − level)`,
    /// so the probability of a false alarm, ever, is at most `1 − level`.
    /// Default 0.95.
    pub level: f64,
    pub pooling: Pooling,
}

impl Default for EvidenceOptions {
    fn default() -> Self {
        Self {
            shape: 1.0,
            level: 0.95,
            pooling: Pooling::Both,
        }
    }
}

/// Why a plan, an observation or an event was refused.
#[derive(Debug, Clone, PartialEq)]
pub enum EvidenceError {
    /// The observation must give one value per state.
    WrongLength { expected: usize, found: usize },
    /// `state` is observed like `representative` but moves differently
    /// between observations: the observed process is not Markov under the
    /// plan. That needs a filter over hidden states, which is not supported.
    NotLumpable {
        state: StateId,
        representative: StateId,
    },
    /// No plan state is observed as `start`.
    UnknownStart(u32),
    /// `shape` must be positive and `level` strictly between 0 and 1.
    InvalidOptions,
    /// Events must arrive in time order.
    TimeOrder { last: f64, got: f64 },
}

impl fmt::Display for EvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongLength { expected, found } => write!(
                f,
                "the observation gives {found} values for {expected} states"
            ),
            Self::NotLumpable {
                state,
                representative,
            } => write!(
                f,
                "{state} is observed like {representative} but moves differently; \
                 the observed process is not Markov under the plan"
            ),
            Self::UnknownStart(b) => write!(f, "no plan state is observed as {b}"),
            Self::InvalidOptions => {
                write!(f, "shape must be positive and level between 0 and 1")
            }
            Self::TimeOrder { last, got } => {
                write!(f, "event at {got} arrives after one at {last}")
            }
        }
    }
}

impl std::error::Error for EvidenceError {}

/// The verdict of [`PlanEvidence`].
#[derive(Debug, Clone, PartialEq)]
pub enum PlanVerdict<L> {
    /// The evidence has not reached `1 / (1 − level)`.
    OnPlan,
    /// The evidence reached `1 / (1 − level)` at `time`. Under the plan this
    /// happens, ever, with probability at most `1 − level`.
    OffPlan { time: f64 },
    /// The plan cannot make this move: off plan with certainty.
    Impossible(ImpossibleMove<L>),
}

/// An observed firing the plan gives rate 0.
#[derive(Debug, Clone, PartialEq)]
pub struct ImpossibleMove<L> {
    pub time: f64,
    pub source: u32,
    pub label: L,
    pub target: u32,
}

/// One observed transition `source —label→ target` and its evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct TransitionEvidence<L> {
    pub source: u32,
    pub label: L,
    pub target: u32,
    /// The plan's rate (per unit time).
    pub plan_rate: f64,
    pub firings: u64,
    /// Time spent observed as `source` so far.
    pub exposure: f64,
    /// Posterior mean of the rate.
    pub estimate: f64,
    /// This transition's share of the per-transition bet, in nats; the
    /// shares add up to its `ln E`.
    pub log_evidence: f64,
}

/// One label (activity) and its evidence: all its transitions together.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelEvidence<L> {
    pub label: L,
    pub firings: u64,
    /// How many firings the plan expected over the same time.
    pub expected: f64,
    /// Posterior mean of the factor between the actual and the planned
    /// rates of this label.
    pub factor: f64,
    /// `ln E` of the bet that only this label is off, in nats. The bet
    /// that all labels may be off is the sum of these.
    pub log_evidence: f64,
}

/// `ln E` of a Gamma(`shape`) mixture over the rate of one Poisson stream
/// after `firings` firings, when the plan expected `expected`:
///
/// `shape·ln shape − ln Γ(shape) + ln Γ(shape + n) − (shape + n)·ln(shape + m) + m`.
///
/// It is the log of `∫ θⁿ e^{−(θ − 1)m} Gamma(θ; shape, shape) dθ`, the
/// likelihood ratio of rate `θ·q` against `q` averaged over the factor `θ`.
/// Real `firings` are allowed, for plug-in predictions.
pub fn log_mixture_evidence(shape: f64, firings: f64, expected: f64) -> f64 {
    shape * shape.ln() - ln_gamma(shape) + ln_gamma(shape + firings)
        - (shape + firings) * (shape + expected).ln()
        + expected
}

/// `ln((eˣ + eʸ) / 2)`.
fn log_mean_exp(x: f64, y: f64) -> f64 {
    let m = x.max(y);
    m + (0.5 * ((x - m).exp() + (y - m).exp())).ln()
}

/// Total rate per `(label, observed target)` from one state.
type ClassSums = HashMap<(LabelId, u32), f64>;

/// The plan's quotient on observations: outgoing `(label, target, rate)`
/// per observed value.
struct Quotient {
    /// Observed values in order of first appearance.
    values: Vec<u32>,
    index: HashMap<u32, usize>,
    /// Per observed value: `(label, target value, rate)`, sorted.
    out: Vec<Vec<(LabelId, u32, f64)>>,
}

impl Quotient {
    fn new<L: Eq + Hash + Clone>(
        plan: &Wlts<L, Real>,
        observe: &[u32],
    ) -> Result<Self, EvidenceError> {
        let n = plan.num_states();
        if observe.len() != n {
            return Err(EvidenceError::WrongLength {
                expected: n,
                found: observe.len(),
            });
        }
        let sums = |s: StateId| {
            let mut m = ClassSums::new();
            for e in plan.successors(s) {
                if e.weight.value() > 0.0 {
                    *m.entry((e.label, observe[e.target.index()])).or_insert(0.0) +=
                        e.weight.value();
                }
            }
            m
        };
        let mut values = Vec::new();
        let mut index = HashMap::new();
        let mut reps: Vec<(StateId, ClassSums)> = Vec::new();
        for s in plan.states() {
            let v = observe[s.index()];
            let mine = sums(s);
            match index.get(&v) {
                None => {
                    index.insert(v, values.len());
                    values.push(v);
                    reps.push((s, mine));
                }
                Some(&b) => {
                    let (rep, theirs) = &reps[b];
                    let close = |x: f64, y: f64| (x - y).abs() <= 1e-9 * x.abs().max(y.abs());
                    let same = mine.len() == theirs.len()
                        && mine
                            .iter()
                            .all(|(k, x)| theirs.get(k).is_some_and(|y| close(*x, *y)));
                    if !same {
                        return Err(EvidenceError::NotLumpable {
                            state: s,
                            representative: *rep,
                        });
                    }
                }
            }
        }
        let out = reps
            .into_iter()
            .map(|(_, m)| {
                let mut row: Vec<(LabelId, u32, f64)> =
                    m.into_iter().map(|((a, t), q)| (a, t, q)).collect();
                row.sort_by_key(|&(a, t, _)| (a, t));
                row
            })
            .collect();
        Ok(Self { values, index, out })
    }

    fn rate(&self, source: u32, label: LabelId, target: u32) -> f64 {
        self.index
            .get(&source)
            .and_then(|&b| {
                self.out[b]
                    .iter()
                    .find(|&&(a, t, _)| a == label && t == target)
            })
            .map_or(0.0, |&(_, _, q)| q)
    }
}

/// One transition of the quotient, with its count so far.
#[derive(Debug, Clone)]
struct Tally {
    label: LabelId,
    target: u32,
    rate: f64,
    firings: u64,
}

/// Evidence against a plan from a stream of observed firings; see the
/// module documentation.
#[derive(Debug, Clone)]
pub struct PlanEvidence<L> {
    label_names: Vec<L>,
    label_ids: HashMap<L, LabelId>,
    values: Vec<u32>,
    index: HashMap<u32, usize>,
    /// Per observed value: its outgoing transitions and their counts.
    tallies: Vec<Vec<Tally>>,
    /// Per observed value: `(label, total planned rate of that label)`.
    label_rates: Vec<Vec<(usize, f64)>>,
    exposure: Vec<f64>,
    /// Per observed value: its share of the per-transition `ln E`.
    transition_shares: Vec<f64>,
    log_transition: f64,
    /// Labels the plan uses somewhere.
    active_labels: Vec<usize>,
    label_firings: Vec<u64>,
    label_expected: Vec<f64>,
    label_shares: Vec<f64>,
    /// `Σ label_shares`: the bet that all labels may be off.
    log_label_product: f64,
    log_e: f64,
    max_log_e: f64,
    crossed: Option<f64>,
    impossible: Option<ImpossibleMove<L>>,
    state: usize,
    now: f64,
    events: u64,
    options: EvidenceOptions,
}

impl<L: Eq + Hash + Clone> PlanEvidence<L> {
    /// Starts observing at time 0, observed as `start`. `observe` gives the
    /// observed value of every plan state and must be a Markovian lumping
    /// of the plan.
    pub fn new(
        plan: &Wlts<L, Real>,
        observe: &[u32],
        start: u32,
        options: EvidenceOptions,
    ) -> Result<Self, EvidenceError> {
        let valid = options.shape.is_finite()
            && options.shape > 0.0
            && options.level > 0.0
            && options.level < 1.0;
        if !valid {
            return Err(EvidenceError::InvalidOptions);
        }
        let q = Quotient::new(plan, observe)?;
        let state = *q
            .index
            .get(&start)
            .ok_or(EvidenceError::UnknownStart(start))?;
        let tallies: Vec<Vec<Tally>> = q
            .out
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&(label, target, rate)| Tally {
                        label,
                        target,
                        rate,
                        firings: 0,
                    })
                    .collect()
            })
            .collect();
        let label_rates = q
            .out
            .iter()
            .map(|row| {
                let mut by_label: Vec<(usize, f64)> = Vec::new();
                for &(a, _, rate) in row {
                    match by_label.iter_mut().find(|(b, _)| *b == a.index()) {
                        Some((_, total)) => *total += rate,
                        None => by_label.push((a.index(), rate)),
                    }
                }
                by_label
            })
            .collect();
        let blocks = q.values.len();
        let label_names = plan.labels().to_vec();
        let label_ids = label_names
            .iter()
            .enumerate()
            .map(|(i, l)| (l.clone(), LabelId(i as u32)))
            .collect();
        let labels = label_names.len();
        let mut active_labels: Vec<usize> =
            q.out.iter().flatten().map(|&(a, _, _)| a.index()).collect();
        active_labels.sort_unstable();
        active_labels.dedup();
        Ok(Self {
            label_names,
            label_ids,
            values: q.values,
            index: q.index,
            tallies,
            label_rates,
            exposure: vec![0.0; blocks],
            transition_shares: vec![0.0; blocks],
            log_transition: 0.0,
            active_labels,
            label_firings: vec![0; labels],
            label_expected: vec![0.0; labels],
            label_shares: vec![0.0; labels],
            log_label_product: 0.0,
            log_e: 0.0,
            max_log_e: 0.0,
            crossed: None,
            impossible: None,
            state,
            now: 0.0,
            events: 0,
            options,
        })
    }

    /// Nothing observable happened until `time`.
    pub fn observe_until(&mut self, time: f64) -> Result<(), EvidenceError> {
        if time.is_nan() || time < self.now {
            return Err(EvidenceError::TimeOrder {
                last: self.now,
                got: time,
            });
        }
        let dt = time - self.now;
        self.now = time;
        if self.impossible.is_some() {
            return Ok(());
        }
        let b = self.state;
        self.exposure[b] += dt;
        for &(a, rate) in &self.label_rates[b] {
            self.label_expected[a] += rate * dt;
        }
        self.update(b);
        Ok(())
    }

    /// The system was seen to fire `label` at `time`, and is now observed as
    /// `target`. After a move the plan cannot make, further events are
    /// accepted and ignored: the verdict is final.
    pub fn fire(&mut self, time: f64, label: &L, target: u32) -> Result<(), EvidenceError> {
        self.observe_until(time)?;
        if self.impossible.is_some() {
            return Ok(());
        }
        self.events += 1;
        let source = self.state;
        let found = self.label_ids.get(label).and_then(|&a| {
            self.tallies[source]
                .iter()
                .position(|t| t.label == a && t.target == target)
        });
        match (found, self.index.get(&target).copied()) {
            (Some(k), Some(next)) => {
                let tally = &mut self.tallies[source][k];
                tally.firings += 1;
                self.label_firings[tally.label.index()] += 1;
                self.update(source);
                self.state = next;
            }
            _ => {
                self.impossible = Some(ImpossibleMove {
                    time,
                    source: self.values[source],
                    label: label.clone(),
                    target,
                });
                self.log_e = f64::INFINITY;
                self.max_log_e = f64::INFINITY;
                self.crossed.get_or_insert(time);
            }
        }
        Ok(())
    }

    /// Recomputes the shares that depend on observed value `b`, `ln E` and
    /// its running maximum.
    fn update(&mut self, b: usize) {
        let shape = self.options.shape;
        let pooling = self.options.pooling;
        if pooling != Pooling::PerLabel {
            let t = self.exposure[b];
            let share: f64 = self.tallies[b]
                .iter()
                .map(|k| log_mixture_evidence(shape, k.firings as f64, k.rate * t))
                .sum();
            self.log_transition += share - self.transition_shares[b];
            self.transition_shares[b] = share;
        }
        if pooling != Pooling::PerTransition {
            for i in 0..self.label_rates[b].len() {
                let a = self.label_rates[b][i].0;
                let share = log_mixture_evidence(
                    shape,
                    self.label_firings[a] as f64,
                    self.label_expected[a],
                );
                self.log_label_product += share - self.label_shares[a];
                self.label_shares[a] = share;
            }
        }
        self.log_e = match pooling {
            Pooling::PerLabel => self.log_label(),
            Pooling::PerTransition => self.log_transition,
            Pooling::Both => log_mean_exp(self.log_label(), self.log_transition),
        };
        if self.log_e > self.max_log_e {
            self.max_log_e = self.log_e;
        }
        if self.crossed.is_none() && self.log_e >= self.threshold() {
            self.crossed = Some(self.now);
        }
    }

    /// `ln E` of the per-label bet: half on exactly one label being off
    /// (the mean of the one-label bets), half on all of them at once.
    fn log_label(&self) -> f64 {
        let shares = self.active_labels.iter().map(|&a| self.label_shares[a]);
        let top = shares.clone().fold(f64::NEG_INFINITY, f64::max);
        if top == f64::NEG_INFINITY {
            return 0.0;
        }
        let sum: f64 = shares.map(|x| (x - top).exp()).sum();
        let one = top + (sum / self.active_labels.len() as f64).ln();
        log_mean_exp(one, self.log_label_product)
    }

    /// `ln(1 / (1 − level))`: the evidence needed for "off plan".
    pub fn threshold(&self) -> f64 {
        -(1.0 - self.options.level).ln()
    }

    /// `ln E` now. It can fall again after a rise; the verdict uses the
    /// running maximum, which Ville's inequality covers.
    pub fn log_evidence(&self) -> f64 {
        self.log_e
    }

    /// The largest `ln E` so far.
    pub fn max_log_evidence(&self) -> f64 {
        self.max_log_e
    }

    pub fn verdict(&self) -> PlanVerdict<L> {
        if let Some(m) = &self.impossible {
            PlanVerdict::Impossible(m.clone())
        } else if let Some(time) = self.crossed {
            PlanVerdict::OffPlan { time }
        } else {
            PlanVerdict::OnPlan
        }
    }

    /// Observed firings so far.
    pub fn events(&self) -> u64 {
        self.events
    }

    /// End of the observation window so far.
    pub fn now(&self) -> f64 {
        self.now
    }

    /// Every activity the plan expected or that fired, largest share of the
    /// evidence first: which activity runs faster or slower than planned,
    /// and by what factor.
    pub fn labels(&self) -> Vec<LabelEvidence<L>> {
        let shape = self.options.shape;
        let mut out: Vec<LabelEvidence<L>> = (0..self.label_names.len())
            .filter(|&a| self.label_expected[a] > 0.0 || self.label_firings[a] > 0)
            .map(|a| {
                let (n, m) = (self.label_firings[a], self.label_expected[a]);
                LabelEvidence {
                    label: self.label_names[a].clone(),
                    firings: n,
                    expected: m,
                    factor: (shape + n as f64) / (shape + m),
                    log_evidence: log_mixture_evidence(shape, n as f64, m),
                }
            })
            .collect();
        out.sort_by(|a, b| b.log_evidence.total_cmp(&a.log_evidence));
        out
    }

    /// Every transition of the plan's quotient from an observed value that
    /// was visited, largest share of the evidence first.
    pub fn transitions(&self) -> Vec<TransitionEvidence<L>> {
        let shape = self.options.shape;
        let mut out: Vec<TransitionEvidence<L>> = self
            .tallies
            .iter()
            .enumerate()
            .filter(|&(b, _)| self.exposure[b] > 0.0)
            .flat_map(|(b, row)| {
                let t = self.exposure[b];
                row.iter().map(move |k| TransitionEvidence {
                    source: self.values[b],
                    label: self.label_names[k.label.index()].clone(),
                    target: k.target,
                    plan_rate: k.rate,
                    firings: k.firings,
                    exposure: t,
                    estimate: k.rate * (shape + k.firings as f64) / (shape + k.rate * t),
                    log_evidence: log_mixture_evidence(shape, k.firings as f64, k.rate * t),
                })
            })
            .collect();
        out.sort_by(|a, b| b.log_evidence.total_cmp(&a.log_evidence));
        out
    }
}

/// How fast evidence against the plan grows when the truth is another
/// system, in nats per unit time (long run, from the stationary
/// distribution of the alternative).
///
/// `per_label ≤ per_transition ≤ (what is observed) ≤ full`: each tester
/// sees less than the next. The rate of what is observed is not computed;
/// it is the relative entropy rate of the observed process, which needs a
/// filter over hidden states when the alternative is not lumpable on the
/// observation.
#[derive(Debug, Clone, PartialEq)]
pub struct Divergences<L> {
    /// Growth rate of the per-label bet: the relative entropy rate between
    /// the alternative's and the plan's firing rates per label.
    pub per_label: f64,
    /// Growth rate of the per-transition bet:
    /// `Σ occupancy · (λ ln(λ/q) − λ + q)` over [`observed`](Self::observed).
    /// [`Pooling::Both`] grows at the larger of the two.
    pub per_transition: f64,
    /// Relative entropy rate from the alternative to the closest system on
    /// its own state space that is observed exactly like the plan: what a
    /// tester that also saw the internal states could reach. Equal to
    /// `per_transition` when the alternative is lumpable on the observation.
    pub full: f64,
    /// Observed firings per unit time under the alternative.
    pub event_rate: f64,
    /// Per observed transition: the alternative's long-run average rate
    /// against the plan's, largest contribution first.
    pub observed: Vec<ObservedRate<L>>,
}

/// One observed transition, plan against alternative.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservedRate<L> {
    pub source: u32,
    pub label: L,
    pub target: u32,
    pub plan_rate: f64,
    /// Firings per unit time spent observed as `source`, in the long run.
    pub average_rate: f64,
    /// Long-run fraction of time observed as `source`.
    pub occupancy: f64,
}

/// `λ ln(λ/q) − λ + q`: the relative entropy rate between Poisson processes
/// of rates `λ` and `q`. Infinite when `q = 0 < λ`.
fn poisson_divergence(lambda: f64, q: f64) -> f64 {
    if lambda <= 0.0 {
        q
    } else if q <= 0.0 {
        f64::INFINITY
    } else {
        lambda * (lambda / q).ln() - lambda + q
    }
}

/// Growth rates of evidence against `plan` when the truth is `alt`, in the
/// long run (stationary distribution of `alt` from its initial state).
/// `observe_alt` gives what each state of `alt` is observed as, in the
/// plan's observed values; firings whose label fails `observed` are not
/// seen (internal steps).
pub fn divergences<L: Eq + Hash + Clone>(
    plan: &Wlts<L, Real>,
    observe_plan: &[u32],
    alt: &Wlts<L, Real>,
    observe_alt: &[u32],
    observed: impl Fn(&L) -> bool,
) -> Result<Divergences<L>, EvidenceError> {
    let q = Quotient::new(plan, observe_plan)?;
    if observe_alt.len() != alt.num_states() {
        return Err(EvidenceError::WrongLength {
            expected: alt.num_states(),
            found: observe_alt.len(),
        });
    }
    let pi = Ctmc::from_wlts(alt).steady_state(1e-12, 100_000);
    let mut full = 0.0;
    let mut event_rate = 0.0;
    // (source, label, target) -> flux; source -> occupancy.
    let mut flux: HashMap<(u32, L, u32), f64> = HashMap::new();
    let mut occupancy: HashMap<u32, f64> = HashMap::new();
    for s in alt.states() {
        let p = pi[s.index()];
        if p <= 0.0 {
            continue;
        }
        let b = observe_alt[s.index()];
        *occupancy.entry(b).or_insert(0.0) += p;
        let mut rates: HashMap<(L, u32), f64> = HashMap::new();
        for e in alt.successors(s) {
            let l = alt.label(e.label);
            if e.weight.value() > 0.0 && observed(l) {
                *rates
                    .entry((l.clone(), observe_alt[e.target.index()]))
                    .or_insert(0.0) += e.weight.value();
            }
        }
        let plan_rate = |l: &L, t: u32| plan.label_id(l).map_or(0.0, |a| q.rate(b, a, t));
        for ((l, t), lambda) in &rates {
            full += p * poisson_divergence(*lambda, plan_rate(l, *t));
            event_rate += p * lambda;
            *flux.entry((b, l.clone(), *t)).or_insert(0.0) += p * lambda;
        }
        // Plan transitions the alternative never makes from `s`.
        if let Some(&bi) = q.index.get(&b) {
            for &(a, t, qr) in &q.out[bi] {
                let l = plan.label(a);
                if !rates.contains_key(&(l.clone(), t)) {
                    full += p * qr;
                    flux.entry((b, l.clone(), t)).or_insert(0.0);
                }
            }
        }
    }
    let mut observed_rates: Vec<ObservedRate<L>> = flux
        .into_iter()
        .map(|((b, l, t), f)| {
            let occ = occupancy[&b];
            let plan_rate = plan.label_id(&l).map_or(0.0, |a| q.rate(b, a, t));
            ObservedRate {
                source: b,
                label: l,
                target: t,
                plan_rate,
                average_rate: f / occ,
                occupancy: occ,
            }
        })
        .collect();
    let contribution =
        |r: &ObservedRate<L>| r.occupancy * poisson_divergence(r.average_rate, r.plan_rate);
    observed_rates.sort_by(|x, y| contribution(y).total_cmp(&contribution(x)));
    let per_transition = observed_rates.iter().map(contribution).sum();
    let mut by_label: HashMap<L, (f64, f64)> = HashMap::new();
    for r in &observed_rates {
        let e = by_label.entry(r.label.clone()).or_insert((0.0, 0.0));
        e.0 += r.occupancy * r.average_rate;
        e.1 += r.occupancy * r.plan_rate;
    }
    let per_label = by_label
        .values()
        .map(|&(actual, planned)| poisson_divergence(actual, planned))
        .sum();
    Ok(Divergences {
        per_label,
        per_transition,
        full,
        event_rate,
        observed: observed_rates,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ln ∫ θⁿ e^{−(θ−1)m} Gamma(θ; a, a) dθ` by the midpoint rule on a log
    /// grid.
    fn by_quadrature(a: f64, n: u32, m: f64) -> f64 {
        let (lo, hi, steps) = (-40.0f64, 6.0f64, 200_000);
        let h = (hi - lo) / steps as f64;
        let mut sum = 0.0;
        for i in 0..steps {
            let x = lo + (i as f64 + 0.5) * h;
            let theta = x.exp();
            let log_prior = a * a.ln() - ln_gamma(a) + (a - 1.0) * theta.ln() - a * theta;
            let log_ratio = n as f64 * theta.ln() - (theta - 1.0) * m;
            // dθ = θ dx.
            sum += (log_prior + log_ratio + theta.ln()).exp() * h;
        }
        sum.ln()
    }

    #[test]
    fn mixture_evidence_matches_quadrature() {
        for &(a, n, m) in &[
            (1.0, 0, 1.5),
            (1.0, 4, 1.5),
            (2.0, 9, 6.0),
            (0.5, 20, 5.0),
            (3.0, 1, 0.2),
        ] {
            let closed = log_mixture_evidence(a, n as f64, m);
            let numeric = by_quadrature(a, n, m);
            assert!(
                (closed - numeric).abs() < 1e-6,
                "a = {a}, n = {n}, m = {m}: {closed} vs {numeric}"
            );
        }
        assert_eq!(log_mixture_evidence(1.0, 0.0, 0.0), 0.0);
    }

    #[test]
    fn poisson_divergence_edges() {
        assert_eq!(poisson_divergence(0.0, 0.3), 0.3);
        assert_eq!(poisson_divergence(0.2, 0.0), f64::INFINITY);
        assert_eq!(poisson_divergence(0.0, 0.0), 0.0);
        assert!(poisson_divergence(0.4, 0.4).abs() < 1e-15);
    }

    #[test]
    fn log_mean_exp_is_the_log_of_the_mean() {
        assert!((log_mean_exp(0.0, 0.0)).abs() < 1e-15);
        assert!((log_mean_exp(1.0, 3.0) - ((1f64.exp() + 3f64.exp()) / 2.0).ln()).abs() < 1e-12);
        assert_eq!(log_mean_exp(f64::NEG_INFINITY, 2.0), 2.0 - 2f64.ln());
    }
}
