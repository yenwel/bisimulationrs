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
