use std::fmt;

use bisim_algebra::pepa::{Act, PepaError, PepaModel, act, parse_pepa, passive, stop};
use bisim_core::{StateId, Verdict};
use bisim_refine::{bisimilar, minimise};

const LIFECYCLE: &str = "
    r_disease = 0.02;
    Healthy = (disease, r_disease).Sick;
    Sick = (healing, 0.1).Healthy + (death, 0.01).Dead;
    Dead = 0;
    Vet = (healing, infty).Vet;
    Healthy[3] <healing> Vet
";

#[test]
fn builder_parser_and_printer_agree() {
    let mut m = PepaModel::builder();
    let r = m.param("r_disease", 0.02);
    let (healthy, sick, dead, vet) = (m.seq("Healthy"), m.seq("Sick"), m.seq("Dead"), m.seq("Vet"));
    m.define(healthy, act("disease".to_string(), r).then(sick));
    m.define(
        sick,
        act("healing".to_string(), 0.1).then(healthy) + act("death".to_string(), 0.01).then(dead),
    );
    m.define(dead, stop());
    m.define(vet, act("healing".to_string(), passive()).then(vet));
    m.system(healthy.times(3).coop(["healing".to_string()], vet));
    let built = m.build().unwrap();

    let parsed = parse_pepa(LIFECYCLE).unwrap();
    assert_eq!(built, parsed);

    let printed = parsed.to_string();
    assert_eq!(
        printed,
        "r_disease = 0.02;\n\
         Healthy = (disease, r_disease).Sick;\n\
         Sick = (healing, 0.1).Healthy + (death, 0.01).Dead;\n\
         Dead = 0;\n\
         Vet = (healing, infty).Vet;\n\
         Healthy[3] <healing> Vet;\n"
    );
    assert_eq!(parse_pepa(&printed).unwrap(), parsed);
}

#[test]
fn printing_preserves_nesting() {
    let src = "a = 1; b = a * (a + 2) - a / 4;
        P = (x, b).((y, 2 * infty).P + (z, a).(Q + R)) + (w, 1).0;
        Q = (q, 1).P;
        R = (y, 3).R;
        S = (y, 2).S;
        (P / {z} <y> S) || (R <y> (S || S)) / {q}";
    let m = parse_pepa(src).unwrap();
    assert_eq!(parse_pepa(&m.to_string()).unwrap(), m, "printed:\n{m}");
}

/// Hillston, "Fluid flow approximation of PEPA models" (QEST 2005): three
/// derivatives in a cycle, two copies in parallel.
const CYCLE: &str = "r = 1;
    P1 = (start, r).P2;
    P2 = (run, r).P3;
    P3 = (stop, r).P1;";

#[test]
fn hillston_cycle_state_space_and_lumping() {
    let full = parse_pepa(&format!("{CYCLE} P1 || P1"))
        .unwrap()
        .derive()
        .unwrap();
    assert_eq!(full.num_states(), 9);
    assert_eq!(full.state_name(StateId(0)), "(P1, P1)");

    // Swapping the two copies is a Markovian bisimulation: 9 → 6 states.
    let lumped = minimise(&full.lts);
    assert_eq!(lumped.num_states(), 6);

    // The aggregated array form derives the 6-state chain directly.
    let array = parse_pepa(&format!("{CYCLE} P1[2]"))
        .unwrap()
        .derive()
        .unwrap();
    assert_eq!(array.num_states(), 6);
    assert_eq!(array.state_name(StateId(0)), "{P1: 2, P2: 0, P3: 0}");
    assert_eq!(bisimilar(&full.lts, &array.lts), Verdict::Holds);

    // A different rate is detected, with a formula that explains it.
    let faster =
        parse_pepa("r = 1; P1 = (start, 2 * r).P2; P2 = (run, r).P3; P3 = (stop, r).P1; P1[2]")
            .unwrap()
            .derive()
            .unwrap();
    let Verdict::Violated { witness } = bisimilar(&array.lts, &faster.lts) else {
        panic!("different rates must not be bisimilar");
    };
    assert_eq!(witness.depth(), 1);
    // From the initial state, `start` happens at total rate 2 (two copies at r = 1).
    assert_eq!(witness.to_string(), "⟨start⟩=2 tt");
}

#[test]
fn cooperation_uses_apparent_rates_and_hiding_gives_tau() {
    // Two clients (rate 2 each) share a server that serves at rate 3.
    let m = parse_pepa(
        "Client = (req, 2).Wait; Wait = (done, 1).Client;
         Server = (req, 3).Server;
         Client[2] <req> Server",
    )
    .unwrap()
    .derive()
    .unwrap();
    let init = m.lts.initial().unwrap();
    let total: f64 = m
        .lts
        .successors(init)
        .iter()
        .filter(|e| m.lts.label(e.label) == &Act::Act("req".to_string()))
        .map(|e| e.weight.value())
        .sum();
    // min(apparent client rate 4, server rate 3) = 3.
    assert!((total - 3.0).abs() < 1e-12);

    let hidden = parse_pepa("P = (a, 1).(b, 2).P; P / {a}")
        .unwrap()
        .derive()
        .unwrap();
    let labels: Vec<String> = hidden
        .lts
        .labels()
        .iter()
        .map(ToString::to_string)
        .collect();
    assert_eq!(labels, vec!["tau", "b"]);
}

#[test]
fn passive_rates_are_shared_by_weight() {
    // The active partner sets the rate; passive weights split it 1 : 3.
    let m = parse_pepa(
        "Src = (go, 4).Src;
         Sink = (go, infty).A + (go, 3 * infty).B; A = (back, 1).Sink; B = (back, 1).Sink;
         Src <go> Sink",
    )
    .unwrap()
    .derive()
    .unwrap();
    let mut rates: Vec<f64> = m
        .lts
        .successors(StateId(0))
        .iter()
        .map(|e| e.weight.value())
        .collect();
    rates.sort_by(f64::total_cmp);
    assert_eq!(rates, vec![1.0, 3.0]);
}

#[test]
fn validation_errors() {
    assert_eq!(
        parse_pepa("P = (a, infty).P; Q = (b, 1).Q; P <b> Q").unwrap_err(),
        PepaError::UnsynchronisedPassive("a".into())
    );
    assert!(matches!(
        parse_pepa("P = (a, 0).P; P").unwrap_err(),
        PepaError::InvalidRate { .. }
    ));
    assert!(matches!(
        parse_pepa("P = (a, 1).P; P[0]").unwrap_err(),
        PepaError::EmptyArray(_)
    ));
    assert!(matches!(
        parse_pepa("P = (a, 1).P + (a, infty).P; Q = (a, 1).Q; P <a> Q")
            .unwrap()
            .derive(),
        Err(PepaError::MixedRates { .. })
    ));
    let warn = parse_pepa("P = (a, 1).P; Q = (b, 1).Q; P <b> Q")
        .unwrap()
        .warnings();
    assert_eq!(warn.len(), 1, "{warn:?}");
}

/// Actions can be a Rust enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Life {
    Disease,
    Healing,
    Death,
}

impl fmt::Display for Life {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Life::Disease => "disease",
            Life::Healing => "healing",
            Life::Death => "death",
        };
        f.write_str(name)
    }
}

#[test]
fn typed_actions() {
    let mut m = PepaModel::<Life>::builder();
    let (healthy, sick, dead) = (m.seq("Healthy"), m.seq("Sick"), m.seq("Dead"));
    m.define(healthy, act(Life::Disease, 0.02).then(sick));
    m.define(
        sick,
        act(Life::Healing, 0.1).then(healthy) + act(Life::Death, 0.01).then(dead),
    );
    m.define(dead, stop());
    m.system(healthy.comp());
    let typed = m.build().unwrap();

    let parsed = parse_pepa(
        "Healthy = (disease, 0.02).Sick; Sick = (healing, 0.1).Healthy + (death, 0.01).Dead; Dead = 0; Healthy",
    )
    .unwrap();
    let mapped = parsed.map_actions(|a| match a.as_str() {
        "disease" => Life::Disease,
        "healing" => Life::Healing,
        _ => Life::Death,
    });
    assert_eq!(typed, mapped);
    assert_eq!(typed.derive().unwrap().num_states(), 3);
}
