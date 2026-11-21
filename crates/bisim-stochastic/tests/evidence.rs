//! Evidence against a plan: false alarms under systems bisimilar to the
//! plan, detection of real deviations, and the blind spot of comparing
//! average rates (hidden sickness phases).
//!
//! The plan is a herd of animals that fall sick, heal or die, and are
//! replaced, observed as counts (healthy, sick, dead). Rates per day.

use std::collections::HashMap;

use bisim_core::{Real, StateId, Wlts, WltsBuilder};
use bisim_stochastic::{
    EvidenceError, EvidenceOptions, PlanEvidence, PlanVerdict, Trajectory, divergences,
    gillespie_continue,
};

type L = &'static str;

#[derive(Debug, Clone, Copy)]
struct Animal {
    sicken: f64,
    heal: f64,
    die: f64,
    replace: f64,
}

const PLAN: Animal = Animal {
    sicken: 0.02,
    heal: 0.1,
    die: 0.01,
    replace: 0.05,
};

/// The count vectors `[healthy, sick, dead]` of `n` animals and their index.
fn count_states(n: usize) -> (Vec<[usize; 3]>, HashMap<[usize; 3], u32>) {
    let mut all = Vec::new();
    for h in (0..=n).rev() {
        for s in (0..=n - h).rev() {
            all.push([h, s, n - h - s]);
        }
    }
    let index = all
        .iter()
        .enumerate()
        .map(|(i, c)| (*c, i as u32))
        .collect();
    (all, index)
}

/// `n` identical animals, counted: the plan. Observed as itself.
fn counted(n: usize, a: Animal) -> (Wlts<L, Real>, Vec<u32>) {
    let (all, index) = count_states(n);
    let mut b = WltsBuilder::new();
    b.ensure_states(all.len());
    for (i, &[h, s, d]) in all.iter().enumerate() {
        let from = StateId(i as u32);
        let mut go = |label: L, to: [usize; 3], rate: f64| {
            if rate > 0.0 {
                b.add_transition(from, label, StateId(index[&to]), Real(rate));
            }
        };
        if h > 0 {
            go("sicken", [h - 1, s + 1, d], h as f64 * a.sicken);
        }
        if s > 0 {
            go("heal", [h + 1, s - 1, d], s as f64 * a.heal);
            go("die", [h, s - 1, d + 1], s as f64 * a.die);
        }
        if d > 0 {
            go("replace", [h + 1, s, d - 1], d as f64 * a.replace);
        }
    }
    b.set_initial(StateId(index[&[n, 0, 0]]));
    (b.build(), (0..all.len() as u32).collect())
}

type Moves = Vec<(usize, L, usize, f64)>;

/// One animal: healthy 0, sick 1, dead 2.
fn simple(a: Animal) -> Moves {
    vec![
        (0, "sicken", 1, a.sicken),
        (1, "heal", 0, a.heal),
        (1, "die", 2, a.die),
        (2, "replace", 0, a.replace),
    ]
}

/// One animal whose sickness has two hidden phases (healthy 0, early 1,
/// late 2, dead 3). The phase change is not observed, and the rates are
/// chosen so that the long-run healing rate per sick day is the plan's.
fn two_phase(a: Animal) -> Moves {
    let fast = 2.0 * a.heal;
    vec![
        (0, "sicken", 1, a.sicken),
        (1, "progress", 2, fast + a.die),
        (2, "heal", 0, fast),
        (1, "die", 3, a.die),
        (2, "die", 3, a.die),
        (3, "replace", 0, a.replace),
    ]
}

/// Every animal on its own; observed as counts. `class` maps a local state
/// to healthy (0), sick (1) or dead (2). All animals start healthy.
fn herd(animals: &[Moves], locals: usize, class: &[usize]) -> (Wlts<L, Real>, Vec<u32>) {
    let n = animals.len();
    let (_, index) = count_states(n);
    let total = locals.pow(n as u32);
    let mut b = WltsBuilder::new();
    b.ensure_states(total);
    let mut observe = vec![0; total];
    for (code, seen) in observe.iter_mut().enumerate() {
        let digit = |i: usize| (code / locals.pow(i as u32)) % locals;
        let mut c = [0; 3];
        for i in 0..n {
            c[class[digit(i)]] += 1;
        }
        *seen = index[&c];
        for (i, moves) in animals.iter().enumerate() {
            let place = locals.pow(i as u32);
            for &(from, label, to, rate) in moves {
                if digit(i) == from {
                    let next = code - from * place + to * place;
                    b.add_transition(
                        StateId(code as u32),
                        label,
                        StateId(next as u32),
                        Real(rate),
                    );
                }
            }
        }
    }
    b.set_initial(StateId(0));
    (b.build(), observe)
}

/// Every state of `sys` split into `copies`, with each rate shared out at
/// random among the copies of its target: a different system, bisimilar to
/// `sys`, observed through the original state.
fn split(sys: &Wlts<L, Real>, copies: usize, seed: u64) -> (Wlts<L, Real>, Vec<u32>) {
    let mut x = seed;
    let mut uniform = || {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64 + 1e-3
    };
    let mut b = WltsBuilder::new();
    b.ensure_states(sys.num_states() * copies);
    for (s, l, t, w) in sys.transitions() {
        for c in 0..copies {
            let shares: Vec<f64> = (0..copies).map(|_| uniform()).collect();
            let sum: f64 = shares.iter().sum();
            for (k, share) in shares.iter().enumerate() {
                b.add_transition(
                    StateId((s.index() * copies + c) as u32),
                    *l,
                    StateId((t.index() * copies + k) as u32),
                    Real(w.value() * share / sum),
                );
            }
        }
    }
    let start = sys.initial().unwrap().index() * copies;
    b.set_initial(StateId(start as u32));
    let observe = (0..sys.num_states() * copies)
        .map(|i| (i / copies) as u32)
        .collect();
    (b.build(), observe)
}

/// Simulates `real` and feeds what is observed to a fresh monitor, until
/// it calls the plan off or has seen `max_events` firings.
fn watch(
    plan: &(Wlts<L, Real>, Vec<u32>),
    real: &(Wlts<L, Real>, Vec<u32>),
    seed: u64,
    max_events: u64,
    options: EvidenceOptions,
) -> PlanEvidence<L> {
    let (sys, observe) = real;
    let mut state = sys.initial().unwrap();
    let mut monitor = PlanEvidence::new(&plan.0, &plan.1, observe[state.index()], options).unwrap();
    let mut time = 0.0;
    for chunk in 0u64.. {
        let mut traj = Trajectory {
            start: state,
            steps: Vec::new(),
        };
        let chunk_seed = seed.wrapping_mul(0x2545_F491_4F6C_DD1D).wrapping_add(chunk);
        gillespie_continue(sys, &mut traj, time, f64::INFINITY, 1024, chunk_seed);
        if traj.steps.is_empty() {
            break;
        }
        for step in &traj.steps {
            if step.label != "progress" {
                monitor
                    .fire(step.time, &step.label, observe[step.target.index()])
                    .unwrap();
            }
            let done =
                monitor.events() >= max_events || !matches!(monitor.verdict(), PlanVerdict::OnPlan);
            if done {
                return monitor;
            }
        }
        state = traj.final_state();
        time = traj.final_time();
    }
    monitor
}

fn alarms(
    plan: &(Wlts<L, Real>, Vec<u32>),
    real: &(Wlts<L, Real>, Vec<u32>),
    runs: u64,
    max_events: u64,
    options: EvidenceOptions,
) -> usize {
    (0..runs)
        .filter(|&r| {
            let m = watch(plan, real, 1000 + r, max_events, options);
            !matches!(m.verdict(), PlanVerdict::OnPlan)
        })
        .count()
}

#[test]
fn observation_must_be_a_lumping() {
    let fast = Animal { heal: 0.2, ..PLAN };
    let (mixed, observe) = herd(&[simple(PLAN), simple(fast)], 3, &[0, 1, 2]);
    let err = PlanEvidence::new(&mixed, &observe, 0, EvidenceOptions::default()).unwrap_err();
    assert!(matches!(err, EvidenceError::NotLumpable { .. }), "{err}");

    let (same, observe) = herd(&[simple(PLAN), simple(PLAN)], 3, &[0, 1, 2]);
    assert!(PlanEvidence::new(&same, &observe, 0, EvidenceOptions::default()).is_ok());
}

/// Ville's inequality: under any system bisimilar to the plan, the chance of
/// ever calling it off plan is at most `1 − level`.
#[test]
fn false_alarms_stay_below_the_level() {
    let options = EvidenceOptions {
        level: 0.9,
        ..EvidenceOptions::default()
    };
    let plan = counted(3, PLAN);
    let individuals = herd(&[simple(PLAN), simple(PLAN), simple(PLAN)], 3, &[0, 1, 2]);
    let shuffled = split(&plan.0, 3, 7);
    for (name, real) in [
        ("plan", &plan),
        ("individuals", &individuals),
        ("split", &shuffled),
    ] {
        let n = alarms(&plan, real, 100, 600, options);
        // At most 10 expected out of 100; 15 leaves room for chance.
        assert!(n <= 15, "{name}: {n} false alarms out of 100");
    }
}

#[test]
fn detects_a_faster_healing_and_names_it() {
    let plan = counted(3, PLAN);
    let fast = counted(3, Animal { heal: 0.2, ..PLAN });
    let options = EvidenceOptions::default();
    let detected = alarms(&plan, &fast, 50, 1500, options);
    assert!(detected >= 48, "detected in {detected} of 50 runs");

    let m = watch(&plan, &fast, 3, 1500, options);
    let top = &m.labels()[0];
    assert_eq!(top.label, "heal", "{:?}", m.labels());
    assert!(top.factor > 1.3, "{top:?}");
    assert_eq!(m.transitions()[0].label, "heal", "{:?}", m.transitions());

    // Both averages the per-label bet (half on one label, half on all) and
    // the per-transition bet (the sum of the transitions' shares).
    let labels = m.labels();
    let one_label = labels.iter().map(|t| t.log_evidence.exp()).sum::<f64>() / 4.0;
    let all_labels = labels.iter().map(|t| t.log_evidence).sum::<f64>().exp();
    let per_transition: f64 = m.transitions().iter().map(|t| t.log_evidence).sum();
    let per_label = 0.5 * (one_label + all_labels);
    let mean = (0.5 * (per_label + per_transition.exp())).ln();
    assert!((mean - m.log_evidence()).abs() < 1e-9 * mean.abs().max(1.0));
}

#[test]
fn impossible_move_is_final() {
    let plan = counted(2, PLAN);
    let (_, index) = count_states(2);
    let start = index[&[2, 0, 0]];
    let mut m = PlanEvidence::new(&plan.0, &plan.1, start, EvidenceOptions::default()).unwrap();
    m.fire(3.0, &"sicken", index[&[1, 1, 0]]).unwrap();
    // A healthy animal cannot die.
    m.fire(5.0, &"die", index[&[0, 1, 1]]).unwrap();
    m.fire(4.0, &"die", index[&[0, 1, 1]]).unwrap_err();
    match m.verdict() {
        PlanVerdict::Impossible(mv) => {
            assert_eq!((mv.time, mv.label), (5.0, "die"));
        }
        v => panic!("{v:?}"),
    }
    m.fire(6.0, &"replace", index[&[2, 0, 0]]).unwrap();
    assert!(matches!(m.verdict(), PlanVerdict::Impossible(_)));
    assert_eq!(m.events(), 2);
}

#[test]
fn divergences_of_lumpable_alternatives_agree() {
    let plan = counted(3, PLAN);
    let same = divergences(&plan.0, &plan.1, &plan.0, &plan.1, |_| true).unwrap();
    assert!(
        same.per_transition.abs() < 1e-12 && same.full.abs() < 1e-12,
        "{same:?}"
    );

    let faster = Animal { heal: 0.12, ..PLAN };
    let counted_faster = counted(3, faster);
    let individuals = herd(
        &[simple(faster), simple(faster), simple(faster)],
        3,
        &[0, 1, 2],
    );
    let a = divergences(
        &plan.0,
        &plan.1,
        &counted_faster.0,
        &counted_faster.1,
        |_| true,
    )
    .unwrap();
    let b = divergences(&plan.0, &plan.1, &individuals.0, &individuals.1, |_| true).unwrap();
    for d in [&a, &b] {
        assert!(d.per_transition > 0.0);
        assert!((d.per_transition - d.full).abs() < 1e-9 * d.full, "{d:?}");
        // A pure change of factor: pooling per label loses nothing.
        assert!(
            (d.per_label - d.per_transition).abs() < 1e-9 * d.full,
            "{d:?}"
        );
    }
    assert!((a.per_transition - b.per_transition).abs() < 1e-9 * a.per_transition);
    assert!((a.event_rate - b.event_rate).abs() < 1e-9 * a.event_rate);
}

/// Hidden sickness phases tuned to the plan's average healing rate: the
/// observed law differs, but the average rates do not.
#[test]
fn hidden_phases_are_invisible_to_average_rates() {
    let plan = counted(2, PLAN);
    let phases = herd(&[two_phase(PLAN), two_phase(PLAN)], 4, &[0, 1, 1, 2]);
    let d = divergences(&plan.0, &plan.1, &phases.0, &phases.1, |l| *l != "progress").unwrap();
    assert!(d.per_transition.abs() < 1e-12, "{d:?}");
    assert!(d.per_label.abs() < 1e-12, "{d:?}");
    assert!(d.full > 1e-3, "{d:?}");

    let options = EvidenceOptions {
        level: 0.9,
        ..EvidenceOptions::default()
    };
    let n = alarms(&plan, &phases, 100, 600, options);
    assert!(n <= 15, "{n} alarms out of 100");
}

/// A fast and a slow healer, tuned so that the herd heals at the plan's
/// average rate per sick day: invisible per label, partly visible per
/// transition (the rate depends on who is sick).
#[test]
fn animals_that_average_out_are_partly_hidden() {
    let plan = counted(2, PLAN);
    let fast = Animal { heal: 0.18, ..PLAN };
    let average_heal = |slow: f64| {
        let mixed = herd(
            &[simple(fast), simple(Animal { heal: slow, ..PLAN })],
            3,
            &[0, 1, 2],
        );
        let d = divergences(&plan.0, &plan.1, &mixed.0, &mixed.1, |_| true).unwrap();
        let heal = d.observed.iter().filter(|r| r.label == "heal");
        let actual: f64 = heal.clone().map(|r| r.occupancy * r.average_rate).sum();
        let planned: f64 = heal.map(|r| r.occupancy * r.plan_rate).sum();
        (actual / planned, d)
    };
    let (mut lo, mut hi) = (0.01, 0.1);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if average_heal(mid).0 < 1.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let (factor, d) = average_heal(0.5 * (lo + hi));
    assert!((factor - 1.0).abs() < 1e-9, "{factor}");
    assert!(d.per_label < 1e-12, "{d:?}");
    assert!(
        d.per_transition > 1e-6 && d.per_transition < 0.5 * d.full,
        "{d:?}"
    );
}
