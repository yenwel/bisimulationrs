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
    let mut rng = Rng(seed);
    let mut steps = Vec::new();
    let mut now = 0.0;
    let mut state = start;
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
        steps.push(SsaStep {
            time: now,
            label: sys.label(e.label).clone(),
            source: state,
            target: e.target,
        });
        state = e.target;
    }
    Trajectory { start, steps }
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
/// each trajectory is then one individual's event log.
pub fn learn_rates<L: Eq + Hash + Clone>(
    traj: &Trajectory<L>,
    num_states: usize,
    horizon: f64,
) -> Vec<LearnedRate<L>> {
    assert!(horizon >= 0.0, "horizon must be non-negative");
    let mut dwell = vec![0.0; num_states];
    let mut firings: HashMap<(usize, L, usize), u64> = HashMap::new();
    let mut now = 0.0;
    let mut state = traj.start;
    for s in &traj.steps {
        dwell[state.index()] += s.time - now;
        *firings
            .entry((s.source.index(), s.label.clone(), s.target.index()))
            .or_insert(0) += 1;
        now = s.time;
        state = s.target;
    }
    dwell[state.index()] += horizon - now;
    let mut out: Vec<LearnedRate<L>> = firings
        .into_iter()
        .map(|((source, label, target), firings)| LearnedRate {
            source: StateId(source as u32),
            label,
            target: StateId(target as u32),
            dwell: dwell[source],
            rate: firings as f64 / dwell[source],
            firings,
        })
        .collect();
    out.sort_by_key(|r| (r.source.index(), r.target.index()));
    out
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
