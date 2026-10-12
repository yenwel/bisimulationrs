//! Property tests on seeded random systems, checked against independent oracles.

use bisim_core::{Lts, NamedBuilder, Real, StateId, Verdict, Wlts, WltsBuilder};
use bisim_refine::{
    bisimilar, compare, distinguishing_formula, minimise, refine, simulation_preorder,
    simulation_witness,
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
