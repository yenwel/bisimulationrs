//! (ε, δ)-bisimulation on seeded random CTMCs: ∼₀,₀ against strong
//! bisimulation by partition refinement, monotonicity, and the paper's
//! reachability bound against exact transient analysis.

use bisim_core::{Partition, Real, StateId, Wlts, WltsBuilder};
use bisim_metric::{epsdelta, min_epsilon, rate_only_bound, reach_bound};
use bisim_refine::refine_fast;
use bisim_stochastic::Ctmc;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Random CTMC whose last state is an absorbing goal.
fn random_ctmc(
    rng: &mut Rng,
    states: u32,
    edges: usize,
    rate: impl Fn(&mut Rng) -> f64,
) -> Wlts<char, Real> {
    let mut b = WltsBuilder::new();
    b.ensure_states(states as usize);
    b.set_initial(StateId(0));
    let goal = states - 1;
    for _ in 0..edges {
        let s = rng.below(goal as u64) as u32;
        let t = rng.below(states as u64) as u32;
        if s != t {
            let r = rate(rng);
            b.add_transition(StateId(s), 'a', StateId(t), Real(r));
        }
    }
    b.build()
}

fn goal_label(n: usize) -> impl Fn(StateId) -> bool + Copy {
    move |s: StateId| s.index() == n - 1
}

/// Strong bisimulation oracle: refinement on the union with self-loops
/// dropped, starting from the labels.
fn strong(left: &Wlts<char, Real>, right: &Wlts<char, Real>) -> Partition {
    let n = left.num_states();
    let mut b = WltsBuilder::new();
    b.ensure_states(n + right.num_states());
    for (sys, offset) in [(left, 0), (right, n)] {
        for (x, _, y, w) in sys.transitions() {
            if x != y {
                b.add_transition(
                    StateId((x.index() + offset) as u32),
                    'τ',
                    StateId((y.index() + offset) as u32),
                    *w,
                );
            }
        }
    }
    let union = b.build();
    let keys = (0..n)
        .map(|i| i == n - 1)
        .chain((0..right.num_states()).map(|i| i == right.num_states() - 1));
    refine_fast(&union, Some(&Partition::from_keys(keys)))
}

#[test]
fn exact_tolerances_give_strong_bisimilarity() {
    let mut rng = Rng(0xE9D_E17A);
    let (mut yes, mut no) = (0, 0);
    for round in 0..200 {
        let n = 3 + (round % 4) as u32;
        let integer = |r: &mut Rng| 1.0 + r.below(2) as f64;
        let left = random_ctmc(&mut rng, n, n as usize * 2, integer);
        let right = random_ctmc(&mut rng, n, n as usize * 2, integer);
        let rel = epsdelta(
            &left,
            &right,
            goal_label(n as usize),
            goal_label(n as usize),
            0.0,
            0.0,
        )
        .unwrap();
        let p = strong(&left, &right);
        for s in 0..n {
            for t in 0..n {
                let want = p.same_block(StateId(s), StateId(n + t));
                let got = rel.related(StateId(s), StateId(t));
                assert_eq!(got, want, "round {round}: {s} vs {t}");
                if want { yes += 1 } else { no += 1 }
            }
        }
    }
    assert!(yes > 100 && no > 100, "{yes} {no}");
}

#[test]
fn relation_grows_with_the_tolerances() {
    let mut rng = Rng(0x40_0707);
    for round in 0..60 {
        let n = 3 + (round % 4) as u32;
        let real = |r: &mut Rng| 0.5 + r.below(1000) as f64 / 400.0;
        let left = random_ctmc(&mut rng, n, n as usize * 2, real);
        let right = random_ctmc(&mut rng, n, n as usize * 2, real);
        let lab = goal_label(n as usize);
        let grid = [(0.0, 0.0), (0.1, 0.1), (0.3, 0.2), (0.6, 0.5), (1.0, 1.0)];
        let rels: Vec<_> = grid
            .iter()
            .map(|&(e, d)| epsdelta(&left, &right, lab, lab, e, d).unwrap())
            .collect();
        for w in rels.windows(2) {
            for s in 0..n {
                for t in 0..n {
                    if w[0].related(StateId(s), StateId(t)) {
                        assert!(w[1].related(StateId(s), StateId(t)), "round {round}");
                    }
                }
            }
        }
    }
}

#[test]
fn reachability_differences_respect_proposition_3() {
    let mut rng = Rng(0xB0_0D5);
    let mut checked = 0;
    for round in 0..150 {
        let n = 3 + (round % 4) as u32;
        let left = random_ctmc(&mut rng, n, n as usize * 2, |r| {
            0.5 + r.below(1000) as f64 / 500.0
        });
        // The right system: the left one with every rate perturbed.
        let mut b = WltsBuilder::new();
        b.ensure_states(n as usize);
        b.set_initial(StateId(0));
        for (s, l, t, w) in left.transitions() {
            let f = 0.85 + rng.below(300) as f64 / 1000.0;
            b.add_transition(s, *l, t, Real(w.value() * f));
        }
        let right = b.build();
        let lab = goal_label(n as usize);
        let delta = 0.3;
        let Some(eps) = min_epsilon(&left, &right, lab, lab, delta).unwrap() else {
            continue;
        };
        let union = left.disjoint_union(&right);
        let ctmc = Ctmc::from_wlts(&union);
        let q = (0..union.num_states())
            .map(|s| {
                union
                    .successors(StateId(s as u32))
                    .iter()
                    .map(|e| e.weight.value())
                    .sum::<f64>()
            })
            .fold(0.0, f64::max);
        let mut targets = vec![false; union.num_states()];
        targets[n as usize - 1] = true;
        targets[2 * n as usize - 1] = true;
        for t in [0.5, 2.0, 5.0] {
            let at = |s: usize| {
                let mut p0 = vec![0.0; union.num_states()];
                p0[s] = 1.0;
                ctmc.reach_bounded(&p0, &targets, t, 1e-10)
            };
            let diff = (at(0) - at(n as usize)).abs();
            let bound = reach_bound(eps, delta, q, t);
            assert!(
                diff <= bound + 1e-6,
                "round {round} t {t}: {diff} > {bound}"
            );
            checked += 1;
        }
    }
    assert!(checked > 100, "{checked}");
}

#[test]
fn rate_only_bound_is_zero_without_tolerance_and_grows_with_it() {
    assert_eq!(rate_only_bound(0.0, 1.0, 3.0), 0.0);
    let small = rate_only_bound(0.05, 1.0, 3.0);
    let large = rate_only_bound(0.3, 1.0, 3.0);
    assert!(
        small > 0.0 && small < large && large <= 1.0,
        "{small} {large}"
    );
    // Sharper than the general bound with ε = 0.
    assert!(large <= reach_bound(0.0, 0.3, 1.0, 3.0));
    // The difference of the Erlang chains E_1 at rates 1 and e^δ:
    // |(1 − e^{−t}) − (1 − e^{−ct})| = e^{−t} − e^{−ct} (N = 1 when t is small).
    let (d, t): (f64, f64) = (0.2, 0.1);
    let n1 = (-t).exp() - (-(d.exp()) * t).exp();
    assert!((rate_only_bound(d, 1.0, t) - n1).abs() < 1e-12);
}
