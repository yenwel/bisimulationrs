use bisim_algebra::pepa::parse_pepa;
use bisim_core::{Partition, StateId};
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
