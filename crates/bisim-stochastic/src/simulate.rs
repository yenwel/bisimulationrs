//! Gillespie stochastic simulation (SSA direct method) for rate-labelled
//! systems.
//!
//! This is the stochastic race semantics used by the simulators of Bio-PEPA,
//! PRISM (`-simulator`) and Modest (`modes`): in each state, every outgoing
//! activity races as an independent exponential clock; the fastest fires,
//! time advances by an `Exp(total rate)` sample, and the next state is
//! picked proportionally to the rates. Repeated runs give Monte-Carlo
//! estimates of transient behaviour without solving the full CTMC, which is
//! what makes large populations (e.g. `Plant[200]`) tractable.
//!
//! The RNG is a seeded SplitMix64 stream, so runs are reproducible and no
//! extra dependency is needed.

use std::collections::HashMap;
use std::hash::Hash;

use bisim_core::{Real, StateId, Wlts};

/// One fired activity: the time it fired and the transition taken.
#[derive(Debug, Clone, PartialEq)]
pub struct SsaStep<L> {
    /// Simulation time in seconds after this step.
    pub time: f64,
    pub label: L,
    pub source: StateId,
    pub target: StateId,
}

/// A full trajectory: the initial state id plus every fired step.
#[derive(Debug, Clone, PartialEq)]
pub struct Trajectory<L> {
    pub start: StateId,
    pub steps: Vec<SsaStep<L>>,
}

impl<L> Trajectory<L> {
    /// State occupied at the end of the trajectory.
    pub fn final_state(&self) -> StateId {
        self.steps.last().map(|s| s.target).unwrap_or(self.start)
    }

    /// Time of the last step (`0.0` when nothing fired).
    pub fn final_time(&self) -> f64 {
        self.steps.last().map(|s| s.time).unwrap_or(0.0)
    }
}

/// Deterministic seeded RNG (SplitMix64). Good enough for sampling;
/// reproducibility matters more here than cryptographic quality.
#[derive(Debug, Clone)]
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `(0, 1]`: never returns exactly `0`, so `ln` is safe
    /// (and time steps are never negative).
    fn uniform_open(&mut self) -> f64 {
        // 53 random bits -> [0,1); reflect to (0,1].
        let bits = self.next_u64() >> 11;
        1.0 - (bits as f64) / (1u64 << 53) as f64
    }
}

/// Runs one Gillespie trajectory from `start` until `t_max` seconds or
/// `max_steps` firings, whichever comes first.
///
/// Self-loops are kept (they consume time in the race) but do not change
/// the state; zero-rate transitions never fire. Returns the steps in
/// chronological order.
pub fn gillespie<L: Eq + Hash + Clone>(
    sys: &Wlts<L, Real>,
    start: StateId,
    t_max: f64,
    max_steps: usize,
    seed: u64,
) -> Trajectory<L> {
    assert!(t_max >= 0.0, "t_max must be non-negative");
    let mut traj = Trajectory {
        start,
        steps: Vec::new(),
    };
    run(sys, &mut traj, 0.0, t_max, max_steps, seed);
    traj
}

/// Continues `traj` from time `from` (its previous horizon) until `until`,
/// with at most `max_steps` more firings. The race is memoryless, so the
/// result is distributed as one run to `until`; a new `seed` keeps the
/// continuation independent of the first part.
///
/// # Panics
/// If `until < from` or the trajectory already has steps after `from`.
pub fn gillespie_continue<L: Eq + Hash + Clone>(
    sys: &Wlts<L, Real>,
    traj: &mut Trajectory<L>,
    from: f64,
    until: f64,
    max_steps: usize,
    seed: u64,
) {
    assert!(until >= from, "cannot continue backwards in time");
    assert!(
        traj.final_time() <= from,
        "the trajectory already runs past {from}"
    );
    run(sys, traj, from, until, max_steps, seed);
}

/// Appends firings to `traj` from time `now` to `t_max`.
fn run<L: Eq + Hash + Clone>(
    sys: &Wlts<L, Real>,
    traj: &mut Trajectory<L>,
    mut now: f64,
    t_max: f64,
    max_steps: usize,
    seed: u64,
) {
    let mut rng = Rng(seed);
    let mut state = traj.final_state();
    for _ in 0..max_steps {
        let succ = sys.successors(state);
        let total: f64 = succ.iter().map(|e| e.weight.value()).sum();
        if total <= 0.0 {
            break; // absorbing state: no race to run
        }
        now += -rng.uniform_open().ln() / total;
        if now > t_max {
            break;
        }
        // Pick the firing transition proportionally to its rate.
        let mut pick = rng.uniform_open() * total;
        let mut chosen = succ.len() - 1;
        for (i, e) in succ.iter().enumerate() {
            pick -= e.weight.value();
            if pick <= 0.0 {
                chosen = i;
                break;
            }
        }
        let e = &succ[chosen];
        traj.steps.push(SsaStep {
            time: now,
            label: sys.label(e.label).clone(),
            source: state,
            target: e.target,
        });
        state = e.target;
    }
}

/// Runs `runs` independent trajectories and counts how many end in each
/// state at (or before) `t_max`. Returns `(state_id, count)` pairs sorted
/// by state id. Seeds are `seed + run_index`, so the ensemble is
/// reproducible.
///
/// This is the Monte-Carlo counterpart to `Ctmc::transient`: for
/// `runs → ∞` the histogram converges to the transient distribution.
pub fn ensemble<L: Eq + Hash + Clone>(
    sys: &Wlts<L, Real>,
    start: StateId,
    t_max: f64,
    max_steps: usize,
    runs: usize,
    seed: u64,
) -> Vec<(StateId, usize)> {
    let mut counts = vec![0usize; sys.num_states()];
    for run in 0..runs {
        let traj = gillespie(sys, start, t_max, max_steps, seed.wrapping_add(run as u64));
        counts[traj.final_state().index()] += 1;
    }
    counts
        .into_iter()
        .enumerate()
        .map(|(i, c)| (StateId(i as u32), c))
        .collect()
}

/// One event of a trajectory log: entry into `state` at `time`. The
/// initial entry (time 0) has no label; every other entry records the
/// activity that fired.
#[derive(Debug, Clone, PartialEq)]
pub struct LogEvent<L> {
    pub time: f64,
    pub state: StateId,
    pub label: Option<L>,
}

/// Maximum-likelihood rate estimates learnt from one fully observed
/// trajectory: `rate = firings / dwell`. Compare against the model that
/// generated the trajectory to validate a simulator, or fit rates to
/// real event logs with the same transition structure.
#[derive(Debug, Clone, PartialEq)]
pub struct LearnedRate<L> {
    pub source: StateId,
    pub label: L,
    pub target: StateId,
    /// How many times this transition fired.
    pub firings: u64,
    /// Total time spent in `source`.
    pub dwell: f64,
    /// `firings / dwell`, in s⁻¹.
    pub rate: f64,
}

impl<L: Clone> Trajectory<L> {
    /// The trajectory as an event log: the initial entry plus one entry
    /// per fired step, in chronological order.
    pub fn log(&self) -> Vec<LogEvent<L>> {
        let mut out = Vec::with_capacity(self.steps.len() + 1);
        out.push(LogEvent {
            time: 0.0,
            state: self.start,
            label: None,
        });
        for s in &self.steps {
            out.push(LogEvent {
                time: s.time,
                state: s.target,
                label: Some(s.label.clone()),
            });
        }
        out
    }

    /// Forgets labels: for learning from unlabelled sequences, where only
    /// states and times were recorded.
    pub fn without_labels(&self) -> Trajectory<()> {
        Trajectory {
            start: self.start,
            steps: self
                .steps
                .iter()
                .map(|s| SsaStep {
                    time: s.time,
                    label: (),
                    source: s.source,
                    target: s.target,
                })
                .collect(),
        }
    }
}

/// Learns transition rates from a fully observed trajectory over
/// `horizon` seconds: for every observed `(source, label, target)`,
/// `rate = firings / dwell(source)`. Dwells tile `[0, horizon]`, so
/// their sum equals the horizon; unvisited states contribute no rows.
///
/// For independent populations, run [`gillespie`] on one copy's chain:
/// each trajectory is then one individual's event log. For logs that keep
/// growing, use [`OnlineRates`], which this delegates to.
pub fn learn_rates<L: Eq + Hash + Clone>(
    traj: &Trajectory<L>,
    num_states: usize,
    horizon: f64,
) -> Vec<LearnedRate<L>> {
    assert!(horizon >= 0.0, "horizon must be non-negative");
    let mut online = OnlineRates::new(traj.start);
    online.reserve(num_states);
    for s in &traj.steps {
        online.fire(s.time, s.label.clone(), s.target);
    }
    online.observe_until(horizon);
    online.rates()
}

/// Rate estimates kept up to date as an event log grows: each event costs
/// O(1), and [`rates`](Self::rates) gives the same estimates as
/// [`learn_rates`] on the log so far.
#[derive(Debug, Clone)]
pub struct OnlineRates<L> {
    state: StateId,
    now: f64,
    dwell: Vec<f64>,
    firings: HashMap<(usize, L, usize), u64>,
}

impl<L: Eq + Hash + Clone> OnlineRates<L> {
    /// Observation starts at time 0 in `start`.
    pub fn new(start: StateId) -> Self {
        let mut r = Self {
            state: start,
            now: 0.0,
            dwell: Vec::new(),
            firings: HashMap::new(),
        };
        r.reserve(start.index() + 1);
        r
    }

    fn reserve(&mut self, states: usize) {
        if self.dwell.len() < states {
            self.dwell.resize(states, 0.0);
        }
    }

    /// The current state left for `target` through `label` at `time`.
    ///
    /// # Panics
    /// If `time` is earlier than the last event.
    pub fn fire(&mut self, time: f64, label: L, target: StateId) {
        self.observe_until(time);
        self.reserve(target.index() + 1);
        *self
            .firings
            .entry((self.state.index(), label, target.index()))
            .or_insert(0) += 1;
        self.state = target;
    }

    /// Nothing happened until `time` (the observation window grows).
    ///
    /// # Panics
    /// If `time` is earlier than the last event.
    pub fn observe_until(&mut self, time: f64) {
        assert!(time >= self.now, "events must arrive in time order");
        self.dwell[self.state.index()] += time - self.now;
        self.now = time;
    }

    /// End of the observation window so far.
    pub fn now(&self) -> f64 {
        self.now
    }

    /// The maximum-likelihood estimates so far.
    pub fn rates(&self) -> Vec<LearnedRate<L>> {
        let mut out: Vec<LearnedRate<L>> = self
            .firings
            .iter()
            .map(|((source, label, target), &firings)| LearnedRate {
                source: StateId(*source as u32),
                label: label.clone(),
                target: StateId(*target as u32),
                dwell: self.dwell[*source],
                rate: firings as f64 / self.dwell[*source],
                firings,
            })
            .collect();
        out.sort_by_key(|r| (r.source.index(), r.target.index()));
        out
    }

    /// A central credible interval at `level` for every rate, assuming the
    /// transitions fire as Poisson processes (exponential holding times) and
    /// a Gamma(`shape`, `inverse_scale`) prior: the posterior of a rate with
    /// `k` firings over dwell `T` is Gamma(`shape + k`, `inverse_scale + T`)
    /// (Poisson–Gamma conjugacy). `(rate, low, high)` per learned rate, in
    /// the order of [`rates`](Self::rates).
    pub fn intervals(
        &self,
        level: f64,
        shape: f64,
        inverse_scale: f64,
    ) -> Vec<(LearnedRate<L>, f64, f64)> {
        assert!((0.0..1.0).contains(&level), "level must be in [0, 1)");
        assert!(shape > 0.0 && inverse_scale >= 0.0, "invalid Gamma prior");
        let tail = (1.0 - level) / 2.0;
        self.rates()
            .into_iter()
            .map(|r| {
                let a = shape + r.firings as f64;
                let b = inverse_scale + r.dwell;
                let low = gamma_quantile(a, tail) / b;
                let high = gamma_quantile(a, 1.0 - tail) / b;
                (r, low, high)
            })
            .collect()
    }
}

/// Regularised lower incomplete gamma `P(a, x)` (series for `x < a + 1`,
/// continued fraction otherwise; Numerical Recipes §6.2).
fn gamma_p(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let ln_front = a * x.ln() - x - ln_gamma(a);
    if x < a + 1.0 {
        let (mut sum, mut term, mut ap) = (1.0 / a, 1.0 / a, a);
        for _ in 0..10_000 {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if term.abs() < sum.abs() * 1e-15 {
                break;
            }
        }
        (sum.ln() + ln_front).exp()
    } else {
        // Lentz's method for the continued fraction of Q(a, x).
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..10_000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-15 {
                break;
            }
        }
        1.0 - (ln_front + h.ln()).exp()
    }
}

/// `ln Γ(a)` (Lanczos, g = 7, n = 9).
fn ln_gamma(a: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if a < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * a).sin()).ln() - ln_gamma(1.0 - a);
    }
    let a = a - 1.0;
    let mut x = C[0];
    for (i, c) in C.iter().enumerate().skip(1) {
        x += c / (a + i as f64);
    }
    let t = a + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (a + 0.5) * t.ln() - t + x.ln()
}

/// The `p`-quantile of Gamma(`a`, 1), by bisection on `P(a, x)`.
fn gamma_quantile(a: f64, p: f64) -> f64 {
    let (mut lo, mut hi) = (0.0, a.max(1.0));
    while gamma_p(a, hi) < p {
        hi *= 2.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if gamma_p(a, mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo <= 1e-12 * hi.max(1e-300) {
            break;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bisim_core::NamedBuilder;

    fn two_state() -> Wlts<&'static str, Real> {
        // A -a,1-> B -b,2-> A, starting at A.
        let mut b = NamedBuilder::<&str, &str, Real>::new();
        b.initial("A")
            .transition("A", "a", "B", Real(1.0))
            .transition("B", "b", "A", Real(2.0));
        b.build().0
    }

    #[test]
    fn trajectories_are_reproducible_and_ordered() {
        let sys = two_state();
        let t1 = gillespie(&sys, StateId(0), 10.0, 100, 42);
        let t2 = gillespie(&sys, StateId(0), 10.0, 100, 42);
        assert_eq!(t1, t2);
        assert!(!t1.steps.is_empty());
        let mut last = 0.0;
        for s in &t1.steps {
            assert!(s.time >= last && s.time <= 10.0);
            last = s.time;
        }
        // Alternation: A->B->A->B...
        for w in t1.steps.windows(2) {
            assert_eq!(w[0].target, w[1].source);
        }
    }

    #[test]
    fn online_rates_match_batch_learning_at_every_prefix() {
        let sys = two_state();
        let traj = gillespie(&sys, StateId(0), 500.0, 100_000, 5);
        let mut online = OnlineRates::new(traj.start);
        for (i, s) in traj.steps.iter().enumerate() {
            online.fire(s.time, s.label, s.target);
            if i % 37 == 0 {
                let prefix = Trajectory {
                    start: traj.start,
                    steps: traj.steps[..=i].to_vec(),
                };
                assert_eq!(online.rates(), learn_rates(&prefix, 2, s.time));
            }
        }
    }

    #[test]
    fn gamma_intervals_cover_known_quantiles() {
        // Gamma(1, 1) is Exp(1): quantile p is −ln(1 − p).
        for p in [0.025, 0.5, 0.975] {
            let q = gamma_quantile(1.0, p);
            assert!((q + (1.0 - p).ln()).abs() < 1e-9, "{p}: {q}");
        }
        // Gamma(3, 1): P(3, x) = 1 − e^{−x}(1 + x + x²/2).
        let x = 2.5f64;
        let p = 1.0 - (-x).exp() * (1.0 + x + x * x / 2.0);
        assert!((gamma_p(3.0, x) - p).abs() < 1e-12);
        // Intervals shrink as the log grows and contain the true rate.
        let sys = two_state();
        let mut widths = Vec::new();
        for horizon in [50.0, 5_000.0] {
            let traj = gillespie(&sys, StateId(0), horizon, 1_000_000, 11);
            let mut online = OnlineRates::new(traj.start);
            for s in &traj.steps {
                online.fire(s.time, s.label, s.target);
            }
            online.observe_until(horizon);
            let iv = online.intervals(0.95, 1e-3, 0.0);
            let (r, low, high) = &iv[0];
            assert_eq!(r.source, StateId(0));
            assert!(*low <= 1.0 && 1.0 <= *high, "{low} {high}");
            widths.push(high - low);
        }
        assert!(widths[1] < widths[0] / 5.0, "{widths:?}");
    }

    #[test]
    fn continued_runs_extend_the_trajectory() {
        let sys = two_state();
        let mut t = gillespie(&sys, StateId(0), 10.0, 1_000, 3);
        let before = t.steps.len();
        gillespie_continue(&sys, &mut t, 10.0, 20.0, 1_000, 4);
        assert!(t.steps.len() > before);
        assert!(
            t.steps[before..]
                .iter()
                .all(|s| s.time > 10.0 && s.time <= 20.0)
        );
        for w in t.steps.windows(2) {
            assert_eq!(w[0].target, w[1].source);
        }
    }

    #[test]
    fn absorbing_state_yields_no_steps() {
        let (sys, _) = NamedBuilder::from_triples([("x", 'a', "y")]);
        let sys = sys.map_weights(|_| Real(1.0));
        let t = gillespie(&sys, StateId(1), 5.0, 100, 7);
        assert!(t.steps.is_empty());
        assert_eq!(t.final_state(), StateId(1));
    }

    #[test]
    fn ensemble_converges_towards_steady_state() {
        // A<->B with rates 1,2: steady state P(A)=2/3, P(B)=1/3.
        let sys = two_state();
        let counts = ensemble(&sys, StateId(0), 50.0, 10_000, 2_000, 1234);
        let pa = counts[0].1 as f64 / 2000.0;
        assert!((pa - 2.0 / 3.0).abs() < 0.05, "P(A)={pa}");
    }

    #[test]
    fn log_covers_every_entry_and_learning_recovers_rates() {
        let sys = two_state();
        let horizon = 20_000.0;
        let traj = gillespie(&sys, StateId(0), horizon, 1_000_000, 99);
        let log = traj.log();
        assert_eq!(log.len(), traj.steps.len() + 1);
        assert_eq!(
            log[0],
            LogEvent {
                time: 0.0,
                state: StateId(0),
                label: None,
            }
        );
        for w in log.windows(2) {
            assert!(w[1].time >= w[0].time);
        }
        let learned = learn_rates(&traj, sys.num_states(), horizon);
        assert_eq!(learned.len(), 2);
        // Dwells tile [0, horizon]: each source's dwell is recorded once.
        let mut seen = [false; 2];
        let mut tiled = 0.0;
        for r in &learned {
            if !seen[r.source.index()] {
                seen[r.source.index()] = true;
                tiled += r.dwell;
            }
        }
        assert!((tiled - horizon).abs() < 1e-6, "tiled {tiled}");
        let ra = learned
            .iter()
            .find(|r| r.source == StateId(0))
            .expect("A->B");
        let rb = learned
            .iter()
            .find(|r| r.source == StateId(1))
            .expect("B->A");
        assert!((ra.rate - 1.0).abs() < 0.05, "learnt {ra:?}");
        assert!((rb.rate - 2.0).abs() < 0.1, "learnt {rb:?}");
        // Unlabelled sequences group by (source, target).
        let flat = learn_rates(&traj.without_labels(), sys.num_states(), horizon);
        assert_eq!(flat.len(), 2);
    }
}
