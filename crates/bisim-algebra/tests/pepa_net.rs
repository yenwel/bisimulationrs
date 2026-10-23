//! The causal (net) semantics of PEPA against the interleaving derivation:
//! forgetting causality, both must describe the same behaviour.

use bisim_algebra::pepa::{PepaError, parse_pepa};
use bisim_core::Verdict;

/// Bundled models with small populations, plus hiding and nested cooperation.
fn models() -> Vec<(&'static str, String)> {
    let small = |s: &str, from: &str, to: &str| s.replace(from, to);
    vec![
        (
            "herd",
            "r = 0.02; Healthy = (disease, r).Sick; Sick = (healing, 0.1).Healthy + (death, 0.01).Dead;
             Dead = 0; Vet = (healing, infty).Vet; Healthy[3] <healing> Vet"
                .to_owned(),
        ),
        (
            "tomatoes",
            small(include_str!("../../../examples/garden/tomatoes.pepa"), "Tomato[24]", "Tomato[2]"),
        ),
        (
            "monoculture",
            small(include_str!("../../../examples/permaculture/monoculture.pepa"), "Healthy[60]", "Healthy[3]"),
        ),
        (
            "credit",
            small(include_str!("../../../examples/credit/credit.pepa"), "[100]", "[2]"),
        ),
        (
            "hiding and nesting",
            "P = (a, 1).(b, 2).P; Q = (b, infty).(c, 1).Q; R = (c, infty).(d, 3).R;
             ((P <b> Q) / {b}) <c> R"
                .to_owned(),
        ),
    ]
}

#[test]
fn net_interleaving_matches_the_derivation() {
    for (name, source) in models() {
        let model = parse_pepa(&source).unwrap_or_else(|e| panic!("{name}: {e}"));
        let net = model.causal_net().unwrap_or_else(|e| panic!("{name}: {e}"));
        let (lts, _) = net.reachability_graph(100_000).unwrap();
        let derived = model
            .derive()
            .unwrap()
            .lts
            .map_weights(|w| !bisim_core::Semiring::is_zero(w));
        let v = bisim_refine::bisimilar(&lts, &derived);
        assert_eq!(v, Verdict::Holds, "{name}: {v:?}");
        // One token per component instance.
        assert_eq!(
            net.initial.len(),
            net.places
                .iter()
                .filter(|p| p.contains('.'))
                .map(|p| p.split('.').next().unwrap())
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            "{name}"
        );
    }
}

#[test]
fn functional_rates_and_huge_arrays_are_refused() {
    let guild = parse_pepa(include_str!("../../../examples/permaculture/guild.pepa")).unwrap();
    let err = guild.causal_net().unwrap_err();
    assert!(matches!(err, PepaError::NoCausalSemantics { .. }), "{err}");
    let credit = parse_pepa(include_str!("../../../examples/credit/credit.pepa")).unwrap();
    assert!(
        credit
            .causal_net()
            .unwrap_err()
            .to_string()
            .contains("component instances")
    );
}
