use bisim_api::{
    Options, Severity, analyse, analyse_jani, analyse_prism, check, compare, fluid, query_prism,
    transient,
};

const CYCLE: &str = "r = 1;
    P1 = (start, r).P2;
    P2 = (run, r).P3;
    P3 = (stop, r).P1;";

#[test]
fn check_reports_positions_and_warnings() {
    let d = check("P = (a, r).P;\nP");
    assert_eq!(d.len(), 1);
    assert_eq!(
        (d[0].severity, d[0].line, d[0].column),
        (Severity::Error, 1, 9)
    );

    let d = check("P = (a, 1).P; Q = (b, 1).Q; P <b> Q");
    assert_eq!(d.len(), 1);
    assert_eq!(d[0].severity, Severity::Warning);

    assert!(check(&format!("{CYCLE} P1")).is_empty());
}

#[test]
fn analyse_gives_full_and_lumped_graphs() {
    let a = analyse(&format!("{CYCLE} P1 || P1"), Options::default()).unwrap();
    assert_eq!(a.full.states.len(), 9);
    assert_eq!(a.lumped.states.len(), 6);
    assert_eq!(a.lumped.states.iter().map(|s| s.size).sum::<usize>(), 9);
    assert_eq!(a.leaves, vec!["P1", "P1"]);
    assert_eq!(a.components.len(), 1);
    assert_eq!(a.components[0].states, vec!["P1", "P2", "P3"]);
    assert!(a.steady_state.iter().all(|p| (p - 1.0 / 9.0).abs() < 1e-9));
    assert!(a.full.states[0].initial);

    // Serialises for the web page and the bindings.
    let json = serde_json::to_string(&a).unwrap();
    assert!(json.contains("\"lumped\""));
}

#[test]
fn transient_occupancy_is_expected_count() {
    let t = transient(&format!("{CYCLE} P1[2]"), &[1.0, 0.0], Options::default()).unwrap();
    let p1 = t
        .occupancy
        .iter()
        .find(|s| s.derivative == "P1")
        .expect("P1 series");
    let exact = |t: f64| (1.0 + 2.0 * (-1.5 * t).exp() * (3f64.sqrt() / 2.0 * t).cos()) / 3.0;
    assert!((p1.values[0] - 2.0 * exact(1.0)).abs() < 1e-8);
    assert!(
        (p1.values[1] - 2.0).abs() < 1e-12,
        "times need not be sorted"
    );

    // Single components report a probability per derivative (sums to 1).
    let single = transient(&format!("{CYCLE} P1"), &[0.7], Options::default()).unwrap();
    let total: f64 = single.occupancy.iter().map(|s| s.values[0]).sum();
    assert_eq!(single.occupancy.len(), 3);
    assert!((total - 1.0).abs() < 1e-9);

    assert!(transient(&format!("{CYCLE} P1"), &[-1.0], Options::default()).is_err());
}

#[test]
fn compare_explains_differences() {
    let same = compare(
        &format!("{CYCLE} P1 || P1"),
        &format!("{CYCLE} P1[2]"),
        Options::default(),
    )
    .unwrap();
    assert!(same.markovian_bisimilar.holds);
    assert!(same.bisimilar.holds);

    let faster = "P1 = (start, 2).P2; P2 = (run, 1).P3; P3 = (stop, 1).P1; P1[2]";
    let diff = compare(&format!("{CYCLE} P1[2]"), faster, Options::default()).unwrap();
    assert!(!diff.markovian_bisimilar.holds);
    assert_eq!(
        diff.markovian_bisimilar.formula.as_deref(),
        Some("⟨start⟩=2 tt")
    );
    assert!(diff.markovian_bisimilar.satisfied_in_a.contains(&0));
    assert!(!diff.markovian_bisimilar.satisfied_in_b.contains(&0));
    // Same actions, different rates: untimed behaviour is identical.
    assert!(diff.bisimilar.holds);

    let lifecycle = "Healthy = (disease, 0.02).Sick; Sick = (healing, 0.1).Healthy + (death, 0.01).Dead; Dead = 0; Healthy";
    let no_recovery =
        "Healthy = (disease, 0.02).Sick; Sick = (death, 0.01).Dead; Dead = 0; Healthy";
    let v = compare(lifecycle, no_recovery, Options::default()).unwrap();
    assert!(!v.b_simulates_a.holds);
    assert_eq!(
        v.b_simulates_a.formula.as_deref(),
        Some("⟨disease⟩⟨healing⟩tt")
    );
    assert!(v.b_simulates_a.explanation.contains("initial state of A"));

    let err = compare("P = (a, 1).Q; P", "P", Options::default()).unwrap_err();
    assert_eq!((err.a.len(), err.b.len()), (1, 1));
}

#[test]
fn analyse_prism_derives_lumps_and_reports() {
    let a = analyse_prism(
        "ctmc
         module m
           s : [0..1] init 0;
           [] s=0 -> 1.5 : (s'=1);
           [] s=1 -> 2.5 : (s'=0);
         endmodule",
        Options::default(),
    )
    .unwrap();
    assert_eq!(a.vars, vec!["s"]);
    assert_eq!(a.full.states.len(), 2);
    // Two asymmetric states: no lumping, steady state 2.5/4 vs 1.5/4.
    assert_eq!(a.lumped.states.len(), 2);
    assert!((a.steady_state[0] - 2.5 / 4.0).abs() < 1e-9);
    assert!((a.steady_state[1] - 1.5 / 4.0).abs() < 1e-9);
    assert!(a.note.is_none());

    let err = analyse_prism(
        "mdp module m x : [0..1] init 0; endmodule",
        Options::default(),
    )
    .unwrap_err();
    assert!(err[0].message.contains("mdp"));
}

#[test]
fn query_prism_answers_reachability_and_steady_state() {
    let two_state = "ctmc
        module m
          s : [0..1] init 0;
          [] s=0 -> 1.5 : (s'=1);
          [] s=1 -> 2.5 : (s'=0);
        endmodule";
    // P[F<=t (s=1)] from s=0 is 1 - e^{-1.5 t}.
    for (t, expected) in [
        (0.0, 0.0),
        (1.0, 1.0 - (-1.5f64).exp()),
        (5.0, 1.0 - (-7.5f64).exp()),
    ] {
        let r = query_prism(two_state, &format!("P=? [F<={t} s=1]"), Options::default()).unwrap();
        assert!(
            (r.probability - expected).abs() < 1e-6,
            "t={t}: {}",
            r.probability
        );
        assert_eq!(r.holds, None);
    }
    let r = query_prism(two_state, "P>=0.9 [F<=5 s=1]", Options::default()).unwrap();
    assert_eq!(r.holds, Some(true));
    let r = query_prism(two_state, "S=? [s=0]", Options::default()).unwrap();
    assert!((r.probability - 2.5 / 4.0).abs() < 1e-9);

    let coin = "dtmc
        module coin
          s : [0..2] init 0;
          [] s=0 -> 0.3 : (s'=1) + 0.7 : (s'=2);
          [] s=1 | s=2 -> 1 : (s'=0);
        endmodule";
    // Recurrent: eventual return has probability 1; step-bounded values
    // follow the paths (0->1 direct; 0->2->0 misses twice; ...).
    let r = query_prism(coin, "P=? [F s=1]", Options::default()).unwrap();
    assert!((r.probability - 1.0).abs() < 1e-9);
    let r = query_prism(coin, "P=? [F<=2 s=1]", Options::default()).unwrap();
    assert!((r.probability - 0.3).abs() < 1e-9);
    let r = query_prism(coin, "P=? [F<=3 s=1]", Options::default()).unwrap();
    assert!((r.probability - 0.51).abs() < 1e-9);
    let r = query_prism(coin, "P=? [F<=1 s=1]", Options::default()).unwrap();
    assert!((r.probability - 0.3).abs() < 1e-9);
    assert!(query_prism(coin, "P=? [F<=1.5 s=1]", Options::default()).is_err());
}

#[test]
fn analyse_prism_reports_rewards() {
    let a = analyse_prism(
        "ctmc
         module m
           s : [0..1] init 0;
           [go] s=0 -> 1.5 : (s'=1);
           [back] s=1 -> 2.5 : (s'=0);
         endmodule
         rewards \"r\"
           s=1 : 10;
           [go] s=0 : 2;
         endrewards",
        Options::default(),
    )
    .unwrap();
    assert_eq!(a.rewards.len(), 1);
    let r = &a.rewards[0];
    assert_eq!(r.block, "r");
    // π(s=1) = 1.5/4: state 10 * 0.375; go fires at 1.5 from π(s=0) = 0.625, value 2.
    assert!((r.state - 3.75).abs() < 1e-9, "{}", r.state);
    assert!((r.transition - 1.875).abs() < 1e-9, "{}", r.transition);
}

#[test]
fn fluid_gives_occupancy_series_without_enumeration() {
    let garden = std::fs::read_to_string("../../examples/garden/tomatoes.pepa").unwrap();
    let t = fluid(&garden, &[0.0, 28.0], 0.1, Options::default()).unwrap();
    assert_eq!(t.times, vec![0.0, 28.0]);
    // 24 plants conserved across all derivatives at all times.
    for k in 0..2 {
        let total: f64 = t.occupancy.iter().map(|s| s.values[k]).sum();
        assert!((total - 24.0).abs() < 1e-6, "t={}: {total}", t.times[k]);
    }
    // Day 0: all vegetative; by day 28 most have flowered or moved on.
    let veg0: f64 = t
        .occupancy
        .iter()
        .filter(|s| s.derivative.starts_with("Vegetative"))
        .map(|s| s.values[0])
        .sum();
    assert!((veg0 - 24.0).abs() < 1e-9);
    // Singles have no mean field.
    assert!(fluid("P = (a, 1).Q; Q = 0; P", &[1.0], 0.1, Options::default()).is_err());
    assert!(fluid(&garden, &[1.0], 0.0, Options::default()).is_err());
}

#[test]
fn analyse_jani_matches_prism() {
    let jani = r#"{"jani-version": 1, "name": "m", "type": "ctmc",
        "variables": {"s": {"type": {"kind": "bounded", "lower-bound": 0, "upper-bound": 1}, "initial-value": 0}},
        "automata": [{"name": "m", "locations": [{"name": "l"}], "initial-location": "l",
          "edges": [
            {"location": "l", "guard": {"op": "=", "left": "s", "right": 0},
             "rate": 1.5, "destinations": [{"location": "l", "assignments": [{"ref": "s", "value": 1}]}]},
            {"location": "l", "guard": {"op": "=", "left": "s", "right": 1},
             "rate": 2.5, "destinations": [{"location": "l", "assignments": [{"ref": "s", "value": 0}]}]}
          ]}],
        "system": {"elements": [{"automaton": "m"}]}}"#;
    let a = analyse_jani(jani, Options::default()).unwrap();
    assert_eq!(a.full.states.len(), 2);
    assert!((a.steady_state[0] - 2.5 / 4.0).abs() < 1e-9);
    let p = analyse_prism(
        "ctmc module m s : [0..1] init 0;
         [] s=0 -> 1.5 : (s'=1); [] s=1 -> 2.5 : (s'=0); endmodule",
        Options::default(),
    )
    .unwrap();
    assert_eq!(a.full.states.len(), p.full.states.len());
}
