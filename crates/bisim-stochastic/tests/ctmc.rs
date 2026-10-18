use bisim_algebra::pepa::parse_pepa;
use bisim_core::{Partition, Real, StateId};
use bisim_refine::refine;
use bisim_stochastic::Ctmc;

const CYCLE: &str = "r = 1;
    P1 = (start, r).P2;
    P2 = (run, r).P3;
    P3 = (stop, r).P1;";

/// Closed form for the 3-cycle with unit rates, started in P1:
/// P(P1 at t) = (1 + 2 e^{−3t/2} cos(√3 t / 2)) / 3.
fn p1_exact(t: f64) -> f64 {
    (1.0 + 2.0 * (-1.5 * t).exp() * (3f64.sqrt() / 2.0 * t).cos()) / 3.0
}

#[test]
fn transient_matches_closed_form() {
    let single = parse_pepa(&format!("{CYCLE} P1"))
        .unwrap()
        .derive()
        .unwrap();
    let ctmc = Ctmc::from_wlts(&single.lts);
    let p0 = ctmc.initial_distribution();
    for t in [0.0, 0.5, 1.0, 2.0, 7.0, 100.0] {
        let p = ctmc.transient(&p0, t, 1e-12);
        assert!(
            (p[0] - p1_exact(t)).abs() < 1e-9,
            "t = {t} s: {} vs {}",
            p[0],
            p1_exact(t)
        );
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }
}

#[test]
fn two_copies_expected_count_and_steady_state() {
    let model = parse_pepa(&format!("{CYCLE} P1 || P1"))
        .unwrap()
        .derive()
        .unwrap();
    let ctmc = Ctmc::from_wlts(&model.lts);
    let p = ctmc.transient(&ctmc.initial_distribution(), 1.0, 1e-12);

    // Independent copies: expected number in P1 is 2 · P(P1).
    let expected_p1: f64 = model
        .lts
        .states()
        .map(|s| {
            let name = model.state_name(s);
            let copies = name.matches("P1").count() as f64;
            copies * p[s.index()]
        })
        .sum();
    assert!((expected_p1 - 2.0 * p1_exact(1.0)).abs() < 1e-9);

    // Steady state is uniform over the 9 states.
    for pi in ctmc.steady_state(1e-14, 100_000) {
        assert!((pi - 1.0 / 9.0).abs() < 1e-9);
    }
}

#[test]
fn lumped_chain_preserves_aggregated_transients() {
    let model = parse_pepa(&format!("{CYCLE} P1 || P1"))
        .unwrap()
        .derive()
        .unwrap();
    let partition: Partition = refine(&model.lts, None).partition().clone();
    let quotient = model.lts.quotient(&partition);
    assert_eq!(quotient.num_states(), 6);

    let full = Ctmc::from_wlts(&model.lts);
    let lumped = Ctmc::from_wlts(&quotient);
    for t in [0.3, 1.0, 4.0] {
        let p = full.transient(&full.initial_distribution(), t, 1e-12);
        let q = lumped.transient(&lumped.initial_distribution(), t, 1e-12);
        let mut aggregated = [0.0; 6];
        for s in model.lts.states() {
            aggregated[partition.block_of(s) as usize] += p[s.index()];
        }
        for (a, b) in aggregated.iter().zip(&q) {
            assert!((a - b).abs() < 1e-9, "t = {t} s: {a} vs {b}");
        }
    }
    assert_eq!(partition.block_of(StateId(0)), 0);
}

fn chain(n: usize, edges: &[(usize, usize, f64)], initial: usize) -> Ctmc {
    let mut b = bisim_core::WltsBuilder::<char, Real>::new();
    b.ensure_states(n);
    b.set_initial(StateId(initial as u32));
    for &(s, t, r) in edges {
        b.add_transition(StateId(s as u32), 'a', StateId(t as u32), Real(r));
    }
    Ctmc::from_wlts(&b.build())
}

#[test]
fn steady_state_of_a_stiff_chain_is_exact() {
    // 0 <-> 1 at 1000, 1 -> 2 at 0.001, 2 -> 1 at 0.01. Balance gives
    // pi = (1, 1, 0.1) / 2.1 up to the 0.001/0.01 ratio: pi2 = 0.1 * pi1.
    let c = chain(
        3,
        &[(0, 1, 1000.0), (1, 0, 1000.0), (1, 2, 0.001), (2, 1, 0.01)],
        0,
    );
    let pi = c.steady_state(1e-12, 100_000);
    let expected = [1.0 / 2.1, 1.0 / 2.1, 0.1 / 2.1];
    for (p, e) in pi.iter().zip(expected) {
        assert!((p - e).abs() < 1e-9, "{pi:?}");
    }
}

#[test]
fn steady_state_starts_from_the_initial_state() {
    // s0 -> s1 at 1, s0 -> s2 at 3; s1 and s2 absorbing: the chain ends in
    // s1 with probability 1/4, not 1/2 (a uniform start would say 5/12).
    let c = chain(3, &[(0, 1, 1.0), (0, 2, 3.0)], 0);
    let pi = c.steady_state(1e-12, 100_000);
    assert!((pi[0]).abs() < 1e-12);
    assert!((pi[1] - 0.25).abs() < 1e-9, "{pi:?}");
    assert!((pi[2] - 0.75).abs() < 1e-9, "{pi:?}");
}

#[test]
fn steady_state_of_a_large_component_uses_the_iterative_solver() {
    // Birth-death chain on 600 states (above the exact solver's limit):
    // up at 1, down at 2, so pi_k is proportional to 0.5^k.
    let n = 600;
    let mut edges = Vec::new();
    for k in 0..n - 1 {
        edges.push((k, k + 1, 1.0));
        edges.push((k + 1, k, 2.0));
    }
    let pi = chain(n, &edges, 0).steady_state(1e-12, 1_000_000);
    assert!((pi.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    for (k, p) in pi.iter().take(5).enumerate() {
        let expected = 0.5f64.powi(k as i32) * 0.5;
        assert!((p - expected).abs() < 1e-8, "k={k}: {p} vs {expected}");
    }
}

#[test]
fn transient_cost_does_not_jump_with_the_horizon() {
    // Two-state chain, rates 1 and 2: p1(t) = 1/3 (1 - e^{-3t}) from state 0.
    // Slices of a long horizon must stop at a bounded number of terms even
    // when epsilon / slices is far below double precision.
    let c = chain(2, &[(0, 1, 1.0), (1, 0, 2.0)], 0);
    for t in [3.0e6, 3.00000555e6] {
        let start = std::time::Instant::now();
        let p = c.transient(&c.initial_distribution(), t, 1e-9);
        assert!((p[1] - 1.0 / 3.0).abs() < 1e-8, "t={t}: {p:?}");
        assert!(
            start.elapsed().as_secs_f64() < 5.0,
            "t={t} took {:?}",
            start.elapsed()
        );
    }
}

#[test]
fn parallel_transitions_with_different_labels_are_merged() {
    // Two labels from s0 to s1 (rates 1 and 2) act as one rate-3 jump.
    let mut b = bisim_core::WltsBuilder::<char, Real>::new();
    b.set_initial(StateId(0));
    b.add_transition(StateId(0), 'a', StateId(1), Real(1.0));
    b.add_transition(StateId(0), 'b', StateId(1), Real(2.0));
    b.add_transition(StateId(1), 'a', StateId(0), Real(3.0));
    let chain = Ctmc::from_wlts(&b.build());
    let pi = chain.steady_state(1e-12, 100_000);
    assert!(
        (pi[0] - 0.5).abs() < 1e-9 && (pi[1] - 0.5).abs() < 1e-9,
        "{pi:?}"
    );
    // Transient: p1(t) = 1/2 (1 - e^{-6t}).
    let p = chain.transient(&chain.initial_distribution(), 0.3, 1e-12);
    assert!((p[1] - 0.5 * (1.0 - (-1.8f64).exp())).abs() < 1e-9, "{p:?}");
}
