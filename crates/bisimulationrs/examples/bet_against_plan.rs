//! Bet against the plan: evidence that a herd is off plan, valid whenever
//! you look, for every system that behaves like the plan at once.
//!
//! The plan: animals fall sick (0.02/day), heal (0.1/day), die (0.01/day)
//! and dead animals are replaced (0.05/day). The farmer sees counts
//! (healthy, sick, dead) and every event. All data are simulated.
//!
//! 1. False alarms under systems bisimilar to the plan (the plan itself,
//!    the animals one by one, and a shuffled copy) stay below the level.
//! 2. Healing that is off by a few percent is found after a number of
//!    events close to the prediction `ln(1/α) / divergence`.
//! 3. What the bet cannot see: hidden sickness phases tuned to the plan's
//!    average, and animals that differ but average out.
//!
//! Run (release mode, about a minute):
//! `cargo run --release -p bisimulationrs --example bet_against_plan`

use std::collections::HashMap;

use bisimulationrs::core::{Real, StateId, Wlts, WltsBuilder};
use bisimulationrs::stochastic::{
    Divergences, EvidenceOptions, PlanEvidence, PlanVerdict, Pooling, Trajectory, divergences,
    gillespie_continue, log_mixture_evidence,
};

type L = &'static str;
type System = (Wlts<L, Real>, Vec<u32>);

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

/// `n` identical animals, counted. Observed as itself.
fn counted(n: usize, a: Animal) -> System {
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
/// late 2, dead 3), tuned so that the long-run healing rate per sick day is
/// the plan's. The phase change is not seen.
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

/// Every animal on its own, observed as counts. `class` maps a local state
/// to healthy (0), sick (1) or dead (2). All animals start healthy.
fn herd(animals: &[Moves], locals: usize, class: &[usize]) -> System {
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

/// Every state split into `copies`, each rate shared out at random among
/// the copies of its target: bisimilar to `sys`, on another state space.
fn split(sys: &Wlts<L, Real>, copies: usize, seed: u64) -> System {
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

/// Simulates `real` and hands every observed firing to `see`, until `see`
/// returns `false` or `max_events` firings were seen.
fn simulate(real: &System, seed: u64, max_events: u64, mut see: impl FnMut(f64, L, u32) -> bool) {
    let (sys, observe) = real;
    let mut state = sys.initial().unwrap();
    let (mut time, mut seen) = (0.0, 0);
    for chunk in 0u64.. {
        let mut traj = Trajectory {
            start: state,
            steps: Vec::new(),
        };
        let chunk_seed = seed.wrapping_mul(0x2545_F491_4F6C_DD1D).wrapping_add(chunk);
        gillespie_continue(sys, &mut traj, time, f64::INFINITY, 4096, chunk_seed);
        if traj.steps.is_empty() {
            return;
        }
        for step in &traj.steps {
            if step.label == "progress" {
                continue;
            }
            seen += 1;
            if !see(step.time, step.label, observe[step.target.index()]) || seen >= max_events {
                return;
            }
        }
        state = traj.final_state();
        time = traj.final_time();
    }
}

fn monitor(plan: &System, real: &System, pooling: Pooling) -> PlanEvidence<L> {
    let start = real.1[real.0.initial().unwrap().index()];
    let options = EvidenceOptions {
        pooling,
        ..EvidenceOptions::default()
    };
    PlanEvidence::new(&plan.0, &plan.1, start, options).unwrap()
}

/// The plug-in prediction of `ln E` after time `t`: expected counts in
/// place of the actual ones.
fn predicted_log_e(d: &Divergences<L>, pooling: Pooling, shape: f64, t: f64) -> f64 {
    let per_transition: f64 = d
        .observed
        .iter()
        .map(|r| {
            let exposure = r.occupancy * t;
            log_mixture_evidence(shape, r.average_rate * exposure, r.plan_rate * exposure)
        })
        .sum();
    let mut by_label: HashMap<L, (f64, f64)> = HashMap::new();
    for r in &d.observed {
        let e = by_label.entry(r.label).or_insert((0.0, 0.0));
        e.0 += r.occupancy * r.average_rate * t;
        e.1 += r.occupancy * r.plan_rate * t;
    }
    // Half on exactly one label being off, half on all of them at once.
    let shares: Vec<f64> = by_label
        .values()
        .map(|&(n, m)| log_mixture_evidence(shape, n, m))
        .collect();
    let one = shares.iter().map(|x| x.exp()).sum::<f64>() / shares.len() as f64;
    let all = shares.iter().sum::<f64>().exp();
    let per_label = (0.5 * (one + all)).ln();
    match pooling {
        Pooling::PerLabel => per_label,
        Pooling::PerTransition => per_transition,
        Pooling::Both => (0.5 * (per_label.exp() + per_transition.exp())).ln(),
    }
}

/// Observed events until the plug-in prediction reaches `threshold`.
fn predicted_events(d: &Divergences<L>, pooling: Pooling, threshold: f64) -> f64 {
    let f = |t: f64| predicted_log_e(d, pooling, 1.0, t) - threshold;
    let mut hi = 1.0;
    while f(hi) < 0.0 {
        hi *= 2.0;
        if hi > 1e12 {
            return f64::INFINITY;
        }
    }
    let mut lo = hi / 2.0;
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if f(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi * d.event_rate
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Runs the three pooled bets and an oracle that knows the healing factor
/// on the same simulated paths. Returns the events until each called the
/// plan off (`max_events` when it never did).
fn detection(
    plan: &System,
    real: &System,
    sick: &[usize],
    factor: f64,
    runs: u64,
    max_events: u64,
) -> [Vec<f64>; 4] {
    let threshold = -(0.05f64).ln();
    let mut out: [Vec<f64>; 4] = Default::default();
    for r in 0..runs {
        let mut bets = [
            monitor(plan, real, Pooling::Both),
            monitor(plan, real, Pooling::PerLabel),
            monitor(plan, real, Pooling::PerTransition),
        ];
        let mut found = [None::<u64>; 4];
        let mut events = 0;
        // The oracle's likelihood ratio: N ln θ − (θ − 1)·M for healing, with
        // M the healing the plan expected (sick animals × rate × time).
        let (mut heals, mut expected, mut last, mut state) = (0.0, 0.0, 0.0, real.1[0]);
        simulate(real, 50_000 + r, max_events, |time, label, target| {
            events += 1;
            for (k, bet) in bets.iter_mut().enumerate() {
                if found[k].is_none() {
                    bet.fire(time, &label, target).unwrap();
                    if !matches!(bet.verdict(), PlanVerdict::OnPlan) {
                        found[k] = Some(events);
                    }
                }
            }
            expected += sick[state as usize] as f64 * PLAN.heal * (time - last);
            heals += f64::from(u8::from(label == "heal"));
            (last, state) = (time, target);
            if found[3].is_none() && heals * factor.ln() - (factor - 1.0) * expected >= threshold {
                found[3] = Some(events);
            }
            found.iter().any(Option::is_none)
        });
        for k in 0..4 {
            out[k].push(found[k].unwrap_or(max_events) as f64);
        }
    }
    out
}

/// Relative entropy rate of what is observed for one animal with hidden
/// sickness phases: the observed process is semi-Markov (a sick period
/// always starts in the early phase), so the rate is the divergence of one
/// sick period (its length and how it ends) over the mean cycle length.
fn observed_rate_two_phase(a: Animal) -> f64 {
    let (e, h, d) = (2.0 * a.heal + a.die, 2.0 * a.heal, a.die);
    let early = |t: f64| (-(e + d) * t).exp();
    let late = |t: f64| e / (e - h) * ((-(h + d) * t).exp() - (-(e + d) * t).exp());
    let x = |g: f64, p: f64| if g > 0.0 { g * (g / p).ln() } else { 0.0 };
    let plan_exit = a.heal + a.die;
    let integrand = |t: f64| {
        let survive = (-plan_exit * t).exp();
        x(h * late(t), a.heal * survive) + x(d * (early(t) + late(t)), a.die * survive)
    };
    // Simpson's rule on [0, 600] days; the tails are below e⁻¹²⁰.
    let (steps, end) = (600_000, 600.0);
    let w = end / steps as f64;
    let mut sum = integrand(0.0) + integrand(end);
    for i in 1..steps {
        sum += integrand(i as f64 * w) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    let per_period = sum * w / 3.0;
    let sick_days = 1.0 / (e + d) + e / (e - h) * (1.0 / (h + d) - 1.0 / (e + d));
    let cycle = 1.0 / a.sicken + sick_days + d * sick_days / a.replace;
    per_period / cycle
}

/// The healing rate of the slow animals that makes a herd with `fast`
/// animals heal, per sick day, at the plan's rate.
fn averaging_slow_rate(plan: &System, fast: usize, slow: usize) -> f64 {
    let average = |rate: f64| {
        let mut animals = vec![simple(Animal { heal: 0.18, ..PLAN }); fast];
        animals.extend(vec![simple(Animal { heal: rate, ..PLAN }); slow]);
        let real = herd(&animals, 3, &[0, 1, 2]);
        let d = divergences(&plan.0, &plan.1, &real.0, &real.1, |_| true).unwrap();
        let heal = d.observed.iter().filter(|r| r.label == "heal");
        let actual: f64 = heal.clone().map(|r| r.occupancy * r.average_rate).sum();
        let planned: f64 = heal.map(|r| r.occupancy * r.plan_rate).sum();
        actual / planned
    };
    let (mut lo, mut hi) = (0.001, PLAN.heal);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if average(mid) < 1.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn main() {
    let n = 4;
    let plan = counted(n, PLAN);
    let threshold95 = -(0.05f64).ln();
    let threshold99 = -(0.01f64).ln();
    let sick: Vec<usize> = count_states(n).0.iter().map(|c| c[1]).collect();

    println!("Plan: {n} animals, observed as counts (healthy, sick, dead) and events.\n");

    // 1. False alarms.
    let runs = 2000;
    let events = 3000;
    let individuals = herd(&vec![simple(PLAN); n], 3, &[0, 1, 2]);
    let shuffled = split(&plan.0, 3, 11);
    println!("1. False alarms within {events} events, {runs} runs (Ville: at most 5 % and 1 %)");
    println!(
        "   {:<34} {:>7} {:>9} {:>9}",
        "system (bisimilar to the plan)", "states", "at 95 %", "at 99 %"
    );
    for (i, (name, real)) in [
        ("the plan itself", &plan),
        ("each animal on its own", &individuals),
        ("random split, 3 copies per state", &shuffled),
    ]
    .into_iter()
    .enumerate()
    {
        let (mut a95, mut a99) = (0, 0);
        for r in 0..runs {
            let mut bet = monitor(&plan, real, Pooling::Both);
            simulate(
                real,
                1_000_000 * (i as u64 + 1) + r,
                events,
                |time, label, target| {
                    bet.fire(time, &label, target).unwrap();
                    true
                },
            );
            a95 += usize::from(bet.max_log_evidence() >= threshold95);
            a99 += usize::from(bet.max_log_evidence() >= threshold99);
        }
        println!(
            "   {:<34} {:>7} {:>8.1} % {:>8.1} %",
            name,
            real.0.num_states(),
            100.0 * a95 as f64 / runs as f64,
            100.0 * a99 as f64 / runs as f64
        );
    }

    // 2. Detection of slower or faster healing.
    let runs = 300;
    let cap = 300_000;
    println!(
        "\n2. Events until \"off plan\" at 95 %, all animals heal at a factor of the plan ({runs} runs)"
    );
    println!(
        "   {:>6} {:>10} {:>8} {:>8} | {:>8} {:>9} {:>9} {:>8} | {:>8}",
        "factor",
        "nats/1000",
        "Wald",
        "plug-in",
        "oracle",
        "per label",
        "per trans",
        "both",
        "both med"
    );
    let mut per_day = 0.0;
    for factor in [1.05, 1.10, 1.20, 1.50, 0.90] {
        let real = counted(
            n,
            Animal {
                heal: PLAN.heal * factor,
                ..PLAN
            },
        );
        let d = divergences(&plan.0, &plan.1, &real.0, &real.1, |_| true).unwrap();
        per_day = d.event_rate;
        let per_event = d.per_label / d.event_rate;
        let [both, per_label, per_transition, oracle] =
            detection(&plan, &real, &sick, factor, runs, cap);
        println!(
            "   {:>6.2} {:>10.3} {:>8.0} {:>8.0} | {:>8.0} {:>9.0} {:>9.0} {:>8.0} | {:>8.0}",
            factor,
            1000.0 * per_event,
            threshold95 / per_event,
            predicted_events(&d, Pooling::Both, threshold95),
            mean(&oracle),
            mean(&per_label),
            mean(&per_transition),
            mean(&both),
            median(&both)
        );
    }
    println!(
        "   (means over runs, capped at {cap} events. Wald = ln 20 / divergence per event, the \
         oracle's prediction;\n   plug-in = expected counts in the bet. {n} animals make {per_day:.3} \
         events per day.)"
    );

    // 3. What it cannot see.
    println!("\n3. Alternatives that average out: divergence rates (nats per 1000 events)");
    println!(
        "   {:<34} {:>9} {:>9} {:>9} {:>10} {:>11}",
        "alternative", "per label", "per trans", "full", "found", "full, Wald"
    );
    let slow = averaging_slow_rate(&plan, 2, 2);
    let phases = herd(&vec![two_phase(PLAN); n], 4, &[0, 1, 1, 2]);
    let mut mixed_animals = vec![simple(Animal { heal: 0.18, ..PLAN }); 2];
    mixed_animals.extend(vec![simple(Animal { heal: slow, ..PLAN }); 2]);
    let mixed = herd(&mixed_animals, 3, &[0, 1, 2]);
    let runs = 300;
    let events = 20_000;
    for (name, real) in [
        ("hidden sickness phases", &phases),
        ("2 heal at 0.18/day, 2 at the rest", &mixed),
    ] {
        let d = divergences(&plan.0, &plan.1, &real.0, &real.1, |l| *l != "progress").unwrap();
        let per = |x: f64| 1000.0 * x / d.event_rate;
        let mut when = Vec::new();
        for r in 0..runs {
            let mut bet = monitor(&plan, real, Pooling::Both);
            let mut seen = 0;
            simulate(real, 90_000 + r, events, |time, label, target| {
                seen += 1;
                bet.fire(time, &label, target).unwrap();
                matches!(bet.verdict(), PlanVerdict::OnPlan)
            });
            if !matches!(bet.verdict(), PlanVerdict::OnPlan) {
                when.push(seen as f64);
            }
        }
        println!(
            "   {:<34} {:>9.3} {:>9.3} {:>9.3} {:>8.1} % {:>11.0}",
            name,
            per(d.per_label),
            per(d.per_transition),
            per(d.full),
            100.0 * when.len() as f64 / runs as f64,
            threshold95 / (d.full / d.event_rate)
        );
        if !when.is_empty() {
            println!(
                "   {:<34} found after {:.0} events on average (median {:.0}); plug-in {:.0}",
                "",
                mean(&when),
                median(&when),
                predicted_events(&d, Pooling::Both, threshold95)
            );
        }
    }
    println!(
        "   (slow rate {slow:.4}/day; found = called off plan at 95 % within {events} events, \
         {runs} runs.\n   full, Wald = events a tester that saw every animal's state would need.)"
    );

    let one = counted(1, PLAN);
    let phases1 = herd(&[two_phase(PLAN)], 4, &[0, 1, 1, 2]);
    let d1 = divergences(&one.0, &one.1, &phases1.0, &phases1.1, |l| *l != "progress").unwrap();
    let observed = observed_rate_two_phase(PLAN);
    println!(
        "\n   One animal with hidden phases, nats per 1000 days: per transition {:.4} \
         ≤ observed {:.4} ≤ full {:.4}",
        1000.0 * d1.per_transition,
        1000.0 * observed,
        1000.0 * d1.full
    );
}
