use bisimulationrs::algebra::pepa::parse_pepa;
use bisimulationrs::core::Verdict;
use bisimulationrs::pepa;
use bisimulationrs::refine::bisimilar;

#[test]
fn macro_matches_text_parser() {
    let from_macro = pepa! {
        r_disease = 0.02;
        Healthy = (disease, r_disease).Sick;
        Sick = (healing, 0.1).Healthy + (death, 0.01).Dead;
        Dead = 0;
        Vet = (healing, infty).Vet;
        Healthy[3] <healing> Vet
    };
    let from_text = parse_pepa(
        "r_disease = 0.02;
         Healthy = (disease, r_disease).Sick;
         Sick = (healing, 0.1).Healthy + (death, 0.01).Dead;
         Dead = 0;
         Vet = (healing, infty).Vet;
         Healthy[3] <healing> Vet",
    )
    .unwrap();
    assert_eq!(from_macro, from_text);
}

#[test]
fn compare_two_macro_models() {
    let expanded = pepa! {
        P1 = (start, 1).P2; P2 = (run, 1).P3; P3 = (stop, 1).P1;
        P1 || P1
    }
    .derive()
    .unwrap();
    let aggregated = pepa! {
        P1 = (start, 1).P2; P2 = (run, 1).P3; P3 = (stop, 1).P1;
        P1[2]
    }
    .derive()
    .unwrap();
    assert_eq!(bisimilar(&expanded.lts, &aggregated.lts), Verdict::Holds);
}

#[test]
fn macro_supports_functional_rates() {
    // Note: `pepa!` stringifies `#I` as `# I`; the parser accepts both.
    let from_macro = pepa! {
        beta = 0.3;
        S = (infect, beta * #I).I;
        I = (recover, 0.1).R;
        R = 0;
        S[2] || I[1]
    };
    let from_text = parse_pepa(
        "beta = 0.3;
         S = (infect, beta * #I).I;
         I = (recover, 0.1).R;
         R = 0;
         S[2] || I[1]",
    )
    .unwrap();
    assert_eq!(from_macro, from_text);
    assert_eq!(from_macro.derive().unwrap().num_states(), 12);
}

/// The bundled population models stay parseable, derivable and meaningful.
#[test]
fn bundled_domain_models_work() {
    use bisimulationrs::algebra::pepa::parse_pepa;
    for (name, src) in [
        ("coral", include_str!("../../../examples/coral/coral.pepa")),
        (
            "antibiotic",
            include_str!("../../../examples/antibiotic/antibiotic.pepa"),
        ),
        ("bass", include_str!("../../../examples/bass/bass.pepa")),
        (
            "credit",
            include_str!("../../../examples/credit/credit.pepa"),
        ),
    ] {
        let model = parse_pepa(src).unwrap_or_else(|e| panic!("{name}: {e}"));
        let fluid = model
            .fluid_model()
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let drift = fluid.fluid_drift(&fluid.initial_counts()).unwrap();
        assert!(drift.iter().all(|v| v.is_finite()), "{name}: {drift:?}");
        // Populations are conserved: every array's slots sum to zero drift.
        assert!(drift.iter().sum::<f64>().abs() < 1e-9, "{name}: {drift:?}");
    }
    // Bass: the exact chain has one state per number of adopters.
    let bass = parse_pepa(include_str!("../../../examples/bass/bass.pepa")).unwrap();
    assert_eq!(bass.derive().unwrap().num_states(), 1001);
}

#[test]
fn permaculture_guild_beats_the_monoculture() {
    use bisimulationrs::algebra::pepa::parse_pepa;
    use bisimulationrs::stochastic::integrate;
    let lost_and_healthy = |src: &str| {
        let m = parse_pepa(src).unwrap().fluid_model().unwrap();
        let x = integrate(
            |x| m.fluid_drift(x),
            &m.initial_counts(),
            &[4.0, 26.0],
            0.05,
        )
        .unwrap();
        let slots = m.fluid_slots();
        let get = |v: &[f64], name: &str| -> f64 {
            slots
                .iter()
                .zip(v)
                .filter(|((_, d), _)| d == name)
                .map(|(_, c)| c)
                .sum()
        };
        (get(&x[0], "Healthy"), get(&x[1], "Lost"))
    };
    let (mono_healthy, mono_lost) = lost_and_healthy(include_str!(
        "../../../examples/permaculture/monoculture.pepa"
    ));
    // Without opposition healthy plants decay as 60 e^{-0.15 t}.
    assert!(
        (mono_healthy - 60.0 * (-0.15f64 * 4.0).exp()).abs() < 1e-6,
        "{mono_healthy}"
    );
    let (_, guild_lost) =
        lost_and_healthy(include_str!("../../../examples/permaculture/guild.pepa"));
    assert!(guild_lost < mono_lost - 10.0, "{guild_lost} vs {mono_lost}");
}
