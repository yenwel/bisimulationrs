//! Property tests on seeded random systems, checked against independent oracles.

use bisim_core::{Lts, NamedBuilder, Real, StateId, Verdict, Wlts, WltsBuilder};
use bisim_refine::{
    GameBudget, bisimilar, bisimilar_states, compare, distinguishing_formula, game_bisimilar,
    minimise, refine, refine_fast, simulation_preorder, simulation_witness,
};

/// xorshift64*: deterministic, dependency-free test randomness.
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

fn random_lts(rng: &mut Rng, states: u32, labels: u8, edges: usize) -> Lts<char> {
    let mut b = WltsBuilder::new();
    b.ensure_states(states as usize);
    b.set_initial(StateId(0));
    for _ in 0..edges {
        let s = StateId(rng.below(states as u64) as u32);
        let t = StateId(rng.below(states as u64) as u32);
        let l = (b'a' + rng.below(labels as u64) as u8) as char;
        b.add_transition(s, l, t, true);
    }
    b.build()
}

fn random_weighted(rng: &mut Rng, states: u32, labels: u8, edges: usize) -> Wlts<char, Real> {
    let mut b = WltsBuilder::new();
    b.ensure_states(states as usize);
    b.set_initial(StateId(0));
    for _ in 0..edges {
        let s = StateId(rng.below(states as u64) as u32);
        let t = StateId(rng.below(states as u64) as u32);
        let l = (b'a' + rng.below(labels as u64) as u8) as char;
        // Small integer rates keep many collisions (hence real splitting).
        b.add_transition(s, l, t, Real(1.0 + rng.below(3) as f64));
    }
    b.build()
}

#[test]
fn fast_refinement_matches_naive() {
    let mut rng = Rng(0xFA57_9EED);
    for round in 0..300 {
        let n = 1 + (round % 12) as u32;
        let sys = random_lts(&mut rng, n, 3, (n as usize) * 2);
        assert_eq!(
            refine_fast(&sys, None),
            refine(&sys, None).partition().clone(),
            "bool round {round}"
        );
        let wsys = random_weighted(&mut rng, n, 3, (n as usize) * 2);
        assert_eq!(
            refine_fast(&wsys, None),
            refine(&wsys, None).partition().clone(),
            "weighted round {round}"
        );
    }
    // With a non-trivial initial partition.
    let mut rng = Rng(7);
    let sys = random_lts(&mut rng, 8, 2, 20);
    let init = bisim_core::Partition::from_keys((0..8).map(|i| i % 3));
    assert_eq!(
        refine_fast(&sys, Some(&init)),
        refine(&sys, Some(&init)).partition().clone()
    );
}

#[test]
fn fast_refinement_matches_naive_on_larger_systems() {
    let mut rng = Rng(0xB16_5EED);
    for round in 0..200 {
        let n = 12 + round % 49;
        let labels = 1 + (round % 3) as u8;
        let edges = (n as usize) * (1 + round as usize % 3) / 2 + 1;
        let sys = random_lts(&mut rng, n, labels, edges);
        let init = if round % 4 == 0 {
            Some(bisim_core::Partition::from_keys(
                (0..n).map(|i| (i * 7 + round) % 5),
            ))
        } else {
            None
        };
        assert_eq!(
            refine_fast(&sys, init.as_ref()),
            refine(&sys, init.as_ref()).partition().clone(),
            "bool round {round}"
        );
        let wsys = random_weighted(&mut rng, n, labels, edges);
        assert_eq!(
            refine_fast(&wsys, init.as_ref()),
            refine(&wsys, init.as_ref()).partition().clone(),
            "weighted round {round}"
        );
    }
}

/// The rounds of `refine` (kept as a split tree) against the rounds
/// computed one at a time.
#[test]
fn split_tree_reproduces_every_round() {
    use bisim_core::{Partition, Signature};
    let mut rng = Rng(0x7EE_5911);
    for round in 0..200 {
        let n = 2 + (round % 25) as u32;
        let sys = random_lts(&mut rng, n, 1 + (round % 3) as u8, (n as usize) * 3 / 2);
        let init = (round % 5 == 0).then(|| Partition::from_keys((0..n).map(|i| i % 3)));
        let refinement = refine(&sys, init.as_ref());
        let mut history = vec![
            init.clone()
                .unwrap_or_else(|| Partition::coarsest(n as usize)),
        ];
        loop {
            let cur = history.last().unwrap();
            let b = cur.as_slice();
            let next =
                Partition::from_keys((0..n).map(|s| (b[s as usize], sys.signature(StateId(s), b))));
            if next.num_blocks() == cur.num_blocks() {
                break;
            }
            history.push(next);
        }
        assert_eq!(refinement.rounds(), history.len() - 1, "round {round}");
        for (k, p) in history.iter().enumerate() {
            assert_eq!(refinement.up_to(k), p, "round {round}, step {k}");
        }
        assert_eq!(refinement.up_to(usize::MAX), history.last().unwrap());
        for s in sys.states() {
            for t in sys.states() {
                let expected = history.iter().position(|p| !p.same_block(s, t));
                assert_eq!(refinement.separation_depth(s, t), expected, "round {round}");
            }
        }
    }
}

/// Bisimilarity as the greatest fixpoint on pairs: an independent oracle.
fn naive_bisim(sys: &Lts<char>) -> Vec<Vec<bool>> {
    let n = sys.num_states();
    let mut r = vec![vec![true; n]; n];
    let matches = |r: &Vec<Vec<bool>>, s: StateId, t: StateId| {
        sys.successors(s).iter().all(|es| {
            sys.successors(t)
                .iter()
                .any(|et| et.label == es.label && r[es.target.index()][et.target.index()])
        })
    };
    loop {
        let mut changed = false;
        for s in sys.states() {
            for t in sys.states() {
                if r[s.index()][t.index()] && !(matches(&r, s, t) && matches(&r, t, s)) {
                    r[s.index()][t.index()] = false;
                    changed = true;
                }
            }
        }
        if !changed {
            return r;
        }
    }
}

#[test]
fn refinement_agrees_with_pairwise_fixpoint() {
    let mut rng = Rng(0x5EED_1234);
    for round in 0..200 {
        let n = 2 + (round % 9) as u32;
        let sys = random_lts(&mut rng, n, 2, (n as usize) * 2);
        let refinement = refine(&sys, None);
        let partition = refinement.partition();
        let oracle = naive_bisim(&sys);

        for s in sys.states() {
            for t in sys.states() {
                let same = partition.same_block(s, t);
                assert_eq!(
                    same,
                    oracle[s.index()][t.index()],
                    "round {round}: {s} vs {t}"
                );

                if !same {
                    let f = distinguishing_formula(&sys, &refinement, s, t)
                        .expect("separated states have a witness");
                    assert!(f.holds(&sys, s), "witness must hold in {s}: {f}");
                    assert!(!f.holds(&sys, t), "witness must fail in {t}: {f}");
                    assert_eq!(
                        Some(f.depth()),
                        refinement.separation_depth(s, t),
                        "witness depth equals separation round"
                    );
                }
            }
        }

        // The result is a fixpoint: refining it again changes nothing.
        let again = refine(&sys, Some(partition));
        assert_eq!(again.partition(), partition);

        // The quotient is minimal and bisimilar to the original.
        let min = minimise(&sys);
        assert_eq!(min.num_states(), partition.num_blocks());
        assert_eq!(bisimilar(&sys, &min), Verdict::Holds);
        assert_eq!(
            refine(&min, None).partition().num_blocks(),
            min.num_states()
        );
    }
}

#[test]
fn bisimilar_implies_mutually_similar_and_witnesses_are_sound() {
    let mut rng = Rng(42);
    for round in 0..200 {
        let n = 2 + (round % 7) as u32;
        let sys = random_lts(&mut rng, n, 2, (n as usize) * 2);
        let pre = simulation_preorder(&sys);
        let bis = refine(&sys, None);
        for s in sys.states() {
            assert!(pre.le(s, s), "simulation is reflexive");
            for t in sys.states() {
                if bis.partition().same_block(s, t) {
                    assert!(pre.equivalent(s, t), "bisimilar ⇒ mutually similar");
                }
                if let Some(f) = simulation_witness(&sys, &pre, s, t) {
                    assert!(f.holds(&sys, s) && !f.holds(&sys, t), "sim witness {f}");
                }
                for u in sys.states() {
                    if pre.le(s, t) && pre.le(t, u) {
                        assert!(pre.le(s, u), "simulation is transitive");
                    }
                }
            }
        }
    }
}

/// Lee & Varaiya (EECS20N): `a.(b + c) + a.b` and `a.(b + c)` simulate each
/// other but are not bisimilar.
#[test]
fn mutual_similarity_is_not_bisimilarity() {
    let (p, _) = NamedBuilder::from_triples([
        ("p", 'a', "p1"),
        ("p", 'a', "p2"),
        ("p1", 'b', "0"),
        ("p1", 'c', "0"),
        ("p2", 'b', "0"),
    ]);
    let (q, _) = NamedBuilder::from_triples([("q", 'a', "q1"), ("q1", 'b', "0"), ("q1", 'c', "0")]);

    let v = compare(&p, &q);
    assert_eq!(v.right_simulates_left, Verdict::Holds);
    assert_eq!(v.mutually_similar, Verdict::Holds);
    let Verdict::Violated { witness } = v.bisimilar else {
        panic!("expected a violation");
    };
    // p can move by `a` into a state that refuses `c`; q cannot.
    assert_eq!(witness.to_string(), "⟨a⟩¬⟨c⟩tt");
}

#[test]
fn simulation_violation_explains_missing_behaviour() {
    // Animal lifecycle: the right model never allows recovery.
    let (left, _) = NamedBuilder::from_triples([
        ("born", "disease", "sick"),
        ("sick", "healing", "healthy"),
        ("sick", "death", "dead"),
    ]);
    let (right, _) =
        NamedBuilder::from_triples([("born", "disease", "sick"), ("sick", "death", "dead")]);

    let v = compare(&left, &right);
    let Verdict::Violated { witness } = &v.right_simulates_left else {
        panic!("right cannot simulate left");
    };
    assert_eq!(witness.to_string(), "⟨disease⟩⟨healing⟩tt");
    assert!(v.mutually_similar.is_violated());
    assert!(v.bisimilar.is_violated());

    // Different state counts are fine when the behaviour is the same.
    let (unfolded, _) = NamedBuilder::from_triples([("x", 't', "y"), ("y", 't', "x")]);
    let (looped, _) = NamedBuilder::from_triples([("z", 't', "z")]);
    assert_eq!(compare(&unfolded, &looped).bisimilar, Verdict::Holds);
}

#[test]
fn markovian_bisimulation_compares_class_sums() {
    // P = (a, 1).Q + (a, 1).R with Q ~ R behaves like (a, 2).Q.
    let mut b = NamedBuilder::<&str, char, Real>::new();
    b.initial("P")
        .transition("P", 'a', "Q", Real(1.0))
        .transition("P", 'a', "R", Real(1.0))
        .transition("Q", 'b', "P", Real(3.0))
        .transition("R", 'b', "P", Real(3.0));
    let (left, _) = b.build();

    let mut b = NamedBuilder::<&str, char, Real>::new();
    b.initial("P")
        .transition("P", 'a', "Q", Real(2.0))
        .transition("Q", 'b', "P", Real(3.0));
    let (right, _) = b.build();
    assert_eq!(bisimilar(&left, &right), Verdict::Holds);

    let mut b = NamedBuilder::<&str, char, Real>::new();
    b.initial("P")
        .transition("P", 'a', "Q", Real(2.5))
        .transition("Q", 'b', "P", Real(3.0));
    let (faster, _): (Wlts<char, Real>, _) = b.build();
    let Verdict::Violated { witness } = bisimilar(&left, &faster) else {
        panic!("rates differ");
    };
    assert!(witness.holds(&left, left.initial().unwrap()));
    assert!(!witness.holds(&faster, faster.initial().unwrap()));
}

/// Same rates into equivalent targets, listed in opposite orders. Float
/// addition is not associative, so naive summation gives class sums that
/// differ in the last bits; those must not split the two states.
#[test]
fn markovian_bisimilarity_ignores_summation_order() {
    let rates = [727511.0, 115478.57142857143, 172873.99999999997];
    let mut b = NamedBuilder::<String, &str, Real>::new();
    b.initial("p".to_owned());
    for (i, r) in rates.iter().enumerate() {
        b.transition("p".to_owned(), "go", format!("t{i}"), Real(*r));
    }
    for (i, r) in rates.iter().rev().enumerate() {
        b.transition("q".to_owned(), "go", format!("t{i}"), Real(*r));
    }
    for i in 0..rates.len() {
        b.transition(format!("t{i}"), "stop", "end".to_owned(), Real(1.0));
    }
    let (sys, names) = b.build();
    let q = StateId(names.iter().position(|n| n == "q").expect("q") as u32);
    assert_eq!(bisimilar_states(&sys, StateId(0), &sys, q), Verdict::Holds);
    // {p, q}, {t0, t1, t2}, {end}.
    assert_eq!(minimise(&sys).num_states(), 3);
}

#[test]
fn class_sums_do_not_depend_on_transition_order() {
    let mut rng = Rng(0x0DD5_0BDE);
    for round in 0..300 {
        let k = 2 + (round % 6) as usize;
        // Rates spanning ten orders of magnitude, 1e-3 to 1e7 s⁻¹.
        let rates: Vec<f64> = (0..k)
            .map(|_| 10f64.powf(rng.below(10_000) as f64 / 1000.0 - 3.0))
            .collect();
        let mut order: Vec<usize> = (0..k).collect();
        for i in (1..k).rev() {
            order.swap(i, rng.below(i as u64 + 1) as usize);
        }
        // p and q reach equivalent targets with the same rates in a
        // different order: distinct targets (class sums) or one target
        // (parallel transitions merged by the builder).
        let one_target = round % 2 == 0;
        let target = |i: usize| StateId(2 + if one_target { 0 } else { i as u32 });
        let end = StateId(2 + k as u32);
        let mut b = WltsBuilder::new();
        for (i, r) in rates.iter().enumerate() {
            b.add_transition(StateId(0), 'a', target(i), Real(*r));
        }
        for (i, &j) in order.iter().enumerate() {
            b.add_transition(StateId(1), 'a', target(i), Real(rates[j]));
        }
        for i in 0..k {
            b.add_transition(target(i), 'b', end, Real(1.0));
        }
        let sys = b.build();
        assert!(
            refine(&sys, None)
                .partition()
                .same_block(StateId(0), StateId(1)),
            "round {round}: rates {rates:?} in order {order:?}"
        );
        assert!(
            refine_fast(&sys, None).same_block(StateId(0), StateId(1)),
            "round {round}: rates {rates:?} in order {order:?}"
        );
    }
}

#[test]
fn depth_limited_game_agrees_with_bounded_refinement() {
    let mut rng = Rng(0xDE97_4CA9);
    for round in 0..300 {
        let n = 2 + (round % 12) as u32;
        let sys = random_lts(&mut rng, n, 2, (n as usize) * 2);
        let refinement = refine(&sys, None);
        let union = sys.disjoint_union(&sys);
        let max_depth = (round % 4) as u32;
        let budget = GameBudget {
            max_pairs: 100_000,
            max_depth,
        };
        for s in sys.states() {
            for t in sys.states() {
                let tt = StateId(t.0 + n);
                match game_bisimilar(&sys, s, &sys, t, budget) {
                    Verdict::Holds => assert!(
                        refinement.partition().same_block(s, t),
                        "round {round}: {s} vs {t} reported bisimilar"
                    ),
                    // Pairs up to depth `max_depth` are expanded, so their
                    // moves are compared: that decides `∼ₖ` for k = depth + 1.
                    Verdict::HoldsUpTo { depth } => {
                        assert_eq!(depth, max_depth);
                        assert!(
                            refinement.up_to(max_depth as usize + 1).same_block(s, t),
                            "round {round}: {s} vs {t} differ within {} steps",
                            max_depth + 1
                        );
                    }
                    Verdict::Violated { witness } => {
                        assert!(witness.holds(&union, s), "round {round}: {witness}");
                        assert!(!witness.holds(&union, tt), "round {round}: {witness}");
                    }
                    other => panic!("round {round}: unexpected {other:?}"),
                }
            }
        }
    }
}

#[test]
fn game_agrees_with_refinement_on_random_systems() {
    let mut rng = Rng(0x6A6D_3E5A);
    // Up to 20 states, dense and with few labels: cycles of matching moves
    // are where coinductive assumptions fail (none showed up below 8
    // states or with 3 sparse labels).
    for round in 0..400 {
        let n = 2 + (round % 19) as u32;
        let (labels, edges) = if round % 2 == 0 { (1, 2) } else { (2, 3) };
        let sys = random_lts(&mut rng, n, labels, (n as usize) * edges);
        let partition = refine(&sys, None).partition().clone();
        for s in sys.states() {
            for t in sys.states() {
                let g = game_bisimilar(&sys, s, &sys, t, GameBudget::default());
                assert_eq!(
                    g.holds(),
                    partition.same_block(s, t),
                    "round {round}: pair {s:?} vs {t:?}"
                );
                if let Verdict::Violated { witness } = g {
                    let union = sys.disjoint_union(&sys);
                    let tt = StateId(t.0 + sys.num_states() as u32);
                    assert!(witness.holds(&union, s));
                    assert!(!witness.holds(&union, tt));
                }
            }
        }
    }
}

#[test]
fn simulation_table_is_indexed_by_bisimulation_classes() {
    // 300 copies of the same two-step chain: 900 states, 3 blocks.
    let mut b = WltsBuilder::<char, bool>::new();
    for i in 0..300u32 {
        b.add_transition(StateId(3 * i), 'a', StateId(3 * i + 1), true);
        b.add_transition(StateId(3 * i + 1), 'b', StateId(3 * i + 2), true);
    }
    b.set_initial(StateId(0));
    let sys = b.build();
    let pre = simulation_preorder(&sys);
    assert_eq!(pre.num_blocks(), 3);
    assert!(pre.equivalent(StateId(0), StateId(300)));
    assert!(pre.le(StateId(2), StateId(0)));
    assert!(!pre.le(StateId(0), StateId(2)));
    let w = simulation_witness(&sys, &pre, StateId(0), StateId(2)).unwrap();
    assert!(
        w.holds(&sys, StateId(0)) && !w.holds(&sys, StateId(2)),
        "{w}"
    );
}

#[test]
fn comparing_without_an_initial_state_is_an_error_not_a_panic() {
    use bisim_refine::{NoInitialState, try_bisimilar, try_compare};
    let mut with = WltsBuilder::<char, bool>::new();
    with.add_transition(StateId(0), 'a', StateId(0), true);
    with.set_initial(StateId(0));
    let with = with.build();
    let mut without = WltsBuilder::<char, bool>::new();
    without.add_transition(StateId(0), 'a', StateId(0), true);
    let without = without.build();
    assert_eq!(try_bisimilar(&with, &without), Err(NoInitialState));
    assert!(try_compare(&without, &with).is_err());
    assert_eq!(try_bisimilar(&with, &with), Ok(Verdict::Holds));
    assert!(try_compare(&with, &with).is_ok());
}

fn verdict_kind<X>(v: &Verdict<X>) -> &'static str {
    match v {
        Verdict::Holds => "holds",
        Verdict::HoldsSoFar => "so-far",
        Verdict::Violated { .. } => "violated",
        _ => "other",
    }
}

/// Prefix verdicts against the full comparison, revealing both systems in a
/// random order: a violation is never retracted (it matches the final
/// answer), "holds" is only claimed when the final answer agrees, and once
/// everything is complete the verdicts equal `compare`.
#[test]
fn prefix_verdicts_are_sound_and_converge_to_the_full_comparison() {
    use bisim_refine::PrefixChecker;
    let mut rng = Rng(0x57AE_AA11);
    let (mut violations, mut so_far) = (0, 0);
    for round in 0..300 {
        let n = 2 + (round % 7) as u32;
        let left = random_lts(&mut rng, n, 2, (n as usize) * 3 / 2);
        let right = random_lts(&mut rng, n, 2, (n as usize) * 3 / 2);
        let full = compare(&left, &right);
        let edges = |sys: &Lts<char>| -> Vec<(u32, char, u32)> {
            sys.transitions()
                .map(|(s, l, t, _)| (s.0, *l, t.0))
                .collect()
        };
        let (mut le, mut re) = (edges(&left), edges(&right));
        // Reveal in random order.
        for list in [&mut le, &mut re] {
            for i in (1..list.len()).rev() {
                list.swap(i, rng.below(i as u64 + 1) as usize);
            }
        }
        let mut checker = PrefixChecker::new(0u32, 0u32);
        let total = le.len() + re.len();
        let (mut seen_l, mut seen_r) = (Vec::new(), Vec::new());
        let complete = |seen: &[(u32, char, u32)], all: &[(u32, char, u32)], s: u32| {
            all.iter().filter(|e| e.0 == s).count() == seen.iter().filter(|e| e.0 == s).count()
        };
        for step in 0..=total {
            if step > 0 {
                // Alternate sides while both have edges left.
                let take_left = (step % 2 == 1 && !le.is_empty()) || re.is_empty();
                if take_left {
                    let e = le.pop().unwrap();
                    seen_l.push(e);
                    checker.observe_left([e]);
                } else {
                    let e = re.pop().unwrap();
                    seen_r.push(e);
                    checker.observe_right([e]);
                }
            }
            let all_l = edges(&left);
            let all_r = edges(&right);
            for s in 0..n {
                if complete(&seen_l, &all_l, s) {
                    checker.complete_left(s);
                }
                if complete(&seen_r, &all_r, s) {
                    checker.complete_right(s);
                }
            }
            let v = checker.verdict();
            for (name, got, want) in [
                (
                    "simulates",
                    &v.right_simulates_left,
                    &full.right_simulates_left,
                ),
                ("similar", &v.mutually_similar, &full.mutually_similar),
                ("bisimilar", &v.bisimilar, &full.bisimilar),
            ] {
                if got.is_violated() {
                    violations += 1;
                    assert!(
                        want.is_violated(),
                        "round {round} step {step}: {name} violated on a prefix but holds in full"
                    );
                }
                if matches!(got, Verdict::Holds) && step < total {
                    // Holds claims every involved state is complete.
                    assert!(!want.is_violated(), "round {round} step {step}: {name}");
                }
                if matches!(got, Verdict::HoldsSoFar) {
                    so_far += 1;
                }
                if step == total {
                    assert_eq!(
                        verdict_kind(got),
                        verdict_kind(want),
                        "round {round}: {name} after everything was seen"
                    );
                }
            }
        }
    }
    // The test must exercise both outcomes.
    assert!(violations > 100 && so_far > 100, "{violations} {so_far}");
}

#[test]
fn prefix_verdict_waits_for_the_missing_move_instead_of_refuting() {
    use bisim_refine::PrefixChecker;
    // The left system can do `a` then `b`. The right has only shown `a`.
    let mut c = PrefixChecker::new("l0", "r0");
    c.observe_left([("l0", 'a', "l1"), ("l1", 'b', "l2")]);
    c.observe_right([("r0", 'a', "r1")]);
    c.complete_left("l0");
    c.complete_left("l1");
    c.complete_left("l2");
    c.complete_right("r0");
    // r1 may still do `b`: nothing refuted yet.
    assert_eq!(verdict_kind(&c.verdict().bisimilar), "so-far");
    // Once r1 is complete without `b`, the difference is final.
    c.complete_right("r1");
    let v = c.verdict();
    let Verdict::Violated { witness } = &v.bisimilar else {
        panic!("expected a violation, got {:?}", v.bisimilar);
    };
    assert_eq!(witness.to_string(), "⟨a⟩⟨b⟩tt");
    // A late `b` repairs an unfinished prefix.
    let mut c = PrefixChecker::new("l0", "r0");
    c.observe_left([("l0", 'a', "l1"), ("l1", 'b', "l2")]);
    c.observe_right([("r0", 'a', "r1")]);
    c.observe_right([("r1", 'b', "r2")]);
    for s in ["l0", "l1", "l2"] {
        c.complete_left(s);
    }
    for s in ["r0", "r1", "r2"] {
        c.complete_right(s);
    }
    assert_eq!(verdict_kind(&c.verdict().bisimilar), "holds");
}
