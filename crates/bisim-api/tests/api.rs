use bisim_api::{Options, Severity, analyse, check, compare, transient};

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
