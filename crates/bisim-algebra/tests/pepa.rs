use std::fmt;

use bisim_algebra::pepa::{
    Act, PepaError, PepaModel, RateExpr, act, count, parse_pepa, passive, stop,
};
use bisim_core::{Real, StateId, Verdict, Wlts};
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

/// Total rate of `action`-transitions out of `s`.
fn total_rate(lts: &Wlts<Act<String>, Real>, s: StateId, action: &str) -> f64 {
    lts.successors(s)
        .iter()
        .filter(|e| matches!(lts.label(e.label), Act::Act(a) if a == action))
        .map(|e| e.weight.value())
        .sum()
}

const SIR_FUN: &str = "
    beta = 0.3; gamma = 0.1;
    S = (infect, beta * #I).I;
    I = (recover, gamma).R;
    R = 0;
    S[9] || I[1]
";

#[test]
fn functional_rates_give_mass_action() {
    let m = parse_pepa(SIR_FUN).unwrap();
    let d = m.derive().unwrap();
    // Mass action holds in *every* reachable state: each susceptible is
    // infected at beta per infected, each infected recovers at gamma.
    for s in d.lts.states() {
        let (mut n_s, mut n_i) = (0u32, 0u32);
        for leaf in 0..d.leaves().len() {
            for (deriv, n) in d.local_counts(s, leaf) {
                match deriv.as_str() {
                    "S" => n_s += n,
                    "I" => n_i += n,
                    _ => {}
                }
            }
        }
        let infect = total_rate(&d.lts, s, "infect");
        let recover = total_rate(&d.lts, s, "recover");
        assert!(
            (infect - 0.3 * n_s as f64 * n_i as f64).abs() < 1e-9,
            "state {}: infect {infect} != 0.3*{n_s}*{n_i}",
            d.state_name(s)
        );
        assert!(
            (recover - 0.1 * n_i as f64).abs() < 1e-9,
            "state {}: recover {recover} != 0.1*{n_i}",
            d.state_name(s)
        );
    }
    // Printing round-trips, keeping the `#I` syntax.
    let printed = m.to_string();
    assert!(printed.contains("#I"), "printed:\n{printed}");
    assert_eq!(parse_pepa(&printed).unwrap(), m);
}

#[test]
fn builder_and_macro_support_functional_rates() {
    let mut m = PepaModel::builder();
    let beta = m.param("beta", 0.3);
    let (s, i) = (m.seq("S"), m.seq("I"));
    m.define(
        s,
        act("infect".to_string(), RateExpr::from(beta) * count("I")).then(i),
    );
    m.define(i, stop());
    m.system(s.times(2).par(i));
    let built = m.build().unwrap();
    let derived = built.derive().unwrap();
    // 2 susceptibles, 1 infected: pressure 2 * 0.3 * 1.
    let total = total_rate(
        &derived.lts,
        derived.lts.initial().expect("initial"),
        "infect",
    );
    assert!((total - 0.6).abs() < 1e-12, "got {total}");
}

#[test]
fn functional_rate_errors() {
    // Unknown species, with the available derivatives listed.
    let e = parse_pepa("S = (a, #Nope).S; S[2]")
        .unwrap()
        .derive()
        .unwrap_err();
    assert!(matches!(e, PepaError::UndefinedSpecies { .. }), "got {e:?}");
    assert!(e.to_string().contains("#Nope"));
    // Counts are not allowed in parameters.
    assert!(matches!(
        parse_pepa("r = #S; S = (a, r).S; S[1]").unwrap_err(),
        PepaError::InvalidRate { .. }
    ));
    // Counts are not allowed under `infty`.
    let e = parse_pepa("S = (a, #I * infty).S; I = (a, 1).I; S <a> I").unwrap_err();
    assert!(
        e.to_string().contains("only allowed in prefix rates"),
        "got {e}"
    );
    // A functional rate that goes negative is a derivation error.
    assert!(matches!(
        parse_pepa("S = (a, 1 - #S).S; S[2]")
            .unwrap()
            .derive()
            .unwrap_err(),
        PepaError::InvalidFunctionalRate { .. }
    ));
}

#[test]
fn fluid_drift_matches_hand_computed_flows() {
    // S =(a,2)=> T: dS/dt = -2S, dT/dt = 2S.
    let m = parse_pepa("S = (a, 2).T; T = 0; S[3]").unwrap();
    let f = m.fluid_model().unwrap();
    assert_eq!(
        f.fluid_slots(),
        vec![
            ("S[3]".to_owned(), "S".to_owned()),
            ("S[3]".to_owned(), "T".to_owned())
        ]
    );
    assert_eq!(f.initial_counts(), vec![3.0, 0.0]);
    assert_eq!(f.fluid_drift(&[3.0, 0.0]).unwrap(), vec![-6.0, 6.0]);
    assert_eq!(f.fluid_drift(&[1.5, 1.5]).unwrap(), vec![-3.0, 3.0]);
    // The enumerated model agrees on structure.
    assert_eq!(m.derive().unwrap().num_states(), 4);

    // Functional SIR: dS = -beta*S*I, dI = beta*S*I - gamma*I.
    // Slots are (S,I,R | I,R); initial x = [9,0,0,1,0].
    let sir = parse_pepa(SIR_FUN).unwrap().fluid_model().unwrap();
    let dx = sir.fluid_drift(&[9.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
    assert_eq!(dx.len(), 5);
    assert!((dx[0] + 2.7).abs() < 1e-12, "{dx:?}");
    assert!((dx[1] - 2.7).abs() < 1e-12, "{dx:?}");
    assert!((dx[3] + 0.1).abs() < 1e-12, "{dx:?}");
    assert!((dx[4] - 0.1).abs() < 1e-12, "{dx:?}");
    // Conservation per array: slots (0,1,2) sum to 9, (3,4) to 1.
    assert!((dx[0] + dx[1] + dx[2]).abs() < 1e-12);
    assert!((dx[3] + dx[4]).abs() < 1e-12);

    // Single components have no mean field.
    let e = parse_pepa("S = (a, 1).T; T = 0; S")
        .unwrap()
        .fluid_model()
        .unwrap()
        .fluid_drift(&[1.0]);
    assert!(matches!(e, Err(PepaError::FluidRequiresArrays { .. })));
}

#[test]
fn zero_functional_rate_in_cooperation_is_zero_not_nan() {
    // No infected yet: each susceptible's infect rate is 0.3 * #I = 0, so
    // the apparent rate is 0 and the cooperation share would be 0/0.
    let m = parse_pepa(
        "S = (infect, 0.3 * #I).I; I = (recover, 0.1).S;
         Env = (infect, 2).Env2; Env2 = (reset, 1).Env;
         S[3] <infect> Env[1]",
    )
    .unwrap();
    let d = m.derive().unwrap();
    let init = d.lts.initial().unwrap();
    assert_eq!(total_rate(&d.lts, init, "infect"), 0.0);
    assert!(
        d.lts
            .transitions()
            .all(|(_, _, _, w)| w.value().is_finite())
    );

    let f = m.fluid_model().unwrap();
    let dx = f.fluid_drift(&f.initial_counts()).unwrap();
    assert!(dx.iter().all(|v| v.is_finite()), "drift {dx:?}");
    assert!(dx.iter().all(|v| *v == 0.0), "nothing can move: {dx:?}");
}

#[test]
fn fluid_drift_counts_self_loops_in_cooperation() {
    // `Env` offers `infect` at 2 and every S shares it: the cooperation
    // runs at min(10 * 0.5, 2) = 2 even though Env's own move is a
    // self-loop, so dS = -2 (not 0).
    let m = parse_pepa(
        "S = (infect, 0.5).I; I = (recover, 0.1).S;
         Env = (infect, 2).Env;
         S[10] <infect> Env[1]",
    )
    .unwrap();
    let f = m.fluid_model().unwrap();
    let slots = f.fluid_slots();
    let x = f.initial_counts();
    let dx = f.fluid_drift(&x).unwrap();
    let s = slots.iter().position(|(_, d)| d == "S").unwrap();
    let i = slots.iter().position(|(_, d)| d == "I").unwrap();
    assert!((dx[s] + 2.0).abs() < 1e-12, "{slots:?} {dx:?}");
    assert!((dx[i] - 2.0).abs() < 1e-12, "{slots:?} {dx:?}");
    // The enumerated model agrees: infect fires at total rate 2.
    let d = m.derive().unwrap();
    let rate = total_rate(&d.lts, d.lts.initial().unwrap(), "infect");
    assert!((rate - 2.0).abs() < 1e-12, "{rate}");
}

#[test]
fn zero_rate_moves_do_not_create_states() {
    // No infected, so `infect` has rate 0 in the only state: one state.
    let d = parse_pepa(
        "beta = 0.3;
         S = (infect, beta * #I).I; I = (recover, 0.1).S;
         S[3]",
    )
    .unwrap()
    .derive()
    .unwrap();
    assert_eq!(d.num_states(), 1);
    assert_eq!(d.lts.num_transitions(), 0);
}

#[test]
fn parameters_are_defined_before_use_and_only_once() {
    // A forward reference used to evaluate to 0.
    let e = parse_pepa("a = b + 1; b = 5; P = (x, a).P; P").unwrap_err();
    assert!(e.to_string().contains("unknown rate parameter `b`"), "{e}");
    let m = parse_pepa("b = 5; a = b + 1; P = (x, a).P; P").unwrap();
    assert!(m.to_string().contains("a = 6"), "{m}");
    let e = parse_pepa("r = 1; r = 2; P = (x, r).P; P").unwrap_err();
    assert!(e.to_string().contains("more than once"), "{e}");
    let e = parse_pepa("P = (x, 1).P; P = (y, 1).P; P").unwrap_err();
    assert!(e.to_string().contains("more than once"), "{e}");
}

#[test]
fn species_counts_see_through_aliases() {
    // `P = Q;` is the same derivative as Q: `#P` counts it (it used to
    // count 0, so `inf` never fired).
    let m = parse_pepa("Q = (a, 1).Q; P = Q; S = (inf, #P).S; P[2] || S").unwrap();
    let d = m.derive().unwrap();
    let init = d.lts.initial().unwrap();
    assert!((total_rate(&d.lts, init, "inf") - 2.0).abs() < 1e-12);
    // The same through the fluid drift (counts are continuous there).
    let f = parse_pepa("Q = (a, 1).Q; P = Q; S = (inf, #P).S2; S2 = (back, 1).S; P[2] || S[1]")
        .unwrap()
        .fluid_model()
        .unwrap();
    let slots = f.fluid_slots();
    let dx = f.fluid_drift(&f.initial_counts()).unwrap();
    let s = slots.iter().position(|(_, d)| d == "S").unwrap();
    assert!((dx[s] + 2.0).abs() < 1e-12, "{slots:?} {dx:?}");
}

#[test]
fn huge_arrays_cannot_overflow_counts() {
    assert!(parse_pepa("P = (a, 1).P; P[4294967295]").is_err());
    let m = parse_pepa(
        "P = (a, 1).P; S = (b, #P).S; P[1000000000] || P[1000000000] || P[1000000000] || S",
    )
    .unwrap();
    let f = m.fluid_model().unwrap();
    assert!(f.initial_counts().iter().all(|c| c.is_finite()));
}

#[test]
fn deeply_nested_input_is_an_error_not_a_stack_overflow() {
    let n = 10_000;
    let parens = format!("P = {}0{};\nP", "(".repeat(n), ")".repeat(n));
    let e = parse_pepa(&parens).unwrap_err();
    assert!(e.to_string().contains("nested too deeply"), "{e}");
    let rate = format!("P = (a, {}1{}).P; P", "(".repeat(n), ")".repeat(n));
    assert!(parse_pepa(&rate).is_err());
    let prefixes = format!("P = {}P; P", "(a, 1).".repeat(n));
    assert!(parse_pepa(&prefixes).is_err());
    // Reasonable nesting still parses.
    assert!(parse_pepa("P = (((((0))))); P").is_ok());
}

#[test]
fn long_operator_chains_are_an_error_not_a_stack_overflow() {
    let n = 200_000;
    // A rate `1 + 1 + … + 1` is a left-leaning tree n levels deep.
    let sum = vec!["1"; n].join(" + ");
    let e = parse_pepa(&format!("P = (a, {sum}).P; P")).unwrap_err();
    assert!(e.to_string().contains("too long"), "{e}");
    let coop = vec!["P"; n].join(" || ");
    assert!(parse_pepa(&format!("P = (a, 1).P; {coop}")).is_err());
    // A few hundred terms are fine.
    let ok = vec!["1"; 300].join(" + ");
    assert!(parse_pepa(&format!("P = (a, {ok}).P; P")).is_ok());
}

#[test]
fn negative_parameters_print_in_a_form_that_parses() {
    let m = parse_pepa("d = 0 - 1; P = (a, 2).P; P").unwrap();
    let printed = m.to_string();
    assert!(printed.contains("d = 0 - 1;"), "{printed}");
    assert_eq!(parse_pepa(&printed).unwrap(), m);
}
