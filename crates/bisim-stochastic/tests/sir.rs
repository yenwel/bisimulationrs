//! The three SIR encodings (`examples/sir/`) agree where they must:
//! same initial infection pressure, same per-state scaling laws, and a
//! PRISM-vs-functional transient cross-check.

use bisim_algebra::pepa::{Act, parse_pepa};
use bisim_algebra::prism::parse_prism;
use bisim_core::{Real, StateId, Wlts};
use bisim_stochastic::Ctmc;

const SIR_COOP: &str = "
    beta = 0.3; gamma = 0.1;
    S = (infect, beta).SI;
    SI = (recover, gamma).R;
    R = 0;
    Carrier = (infect, infty).Carrier;
    S[9] <infect> Carrier
";

const SIR_FUN: &str = "
    beta = 0.3; gamma = 0.1;
    S = (infect, beta * #I).I;
    I = (recover, gamma).R;
    R = 0;
    S[9] || I[1]
";

const SIR_PRISM: &str = "
    ctmc
    const double beta = 0.3;
    const double gamma = 0.1;
    const int N = 10;
    module sir
      s : [0..N] init 9;
      i : [0..N] init 1;
      [infect] s > 0 & i > 0 -> beta*s*i : (s'=s-1) & (i'=i+1);
      [recover] i > 0 -> gamma*i : (i'=i-1);
    endmodule
";

fn total_rate(lts: &Wlts<Act<String>, Real>, s: StateId, action: &str) -> f64 {
    lts.successors(s)
        .iter()
        .filter(|e| matches!(lts.label(e.label), Act::Act(a) if a == action))
        .map(|e| e.weight.value())
        .sum()
}

#[test]
fn three_encodings_agree_on_initial_pressure() {
    let coop = parse_pepa(SIR_COOP).unwrap().derive().unwrap();
    let fun = parse_pepa(SIR_FUN).unwrap().derive().unwrap();
    let prism = parse_prism(SIR_PRISM).unwrap().derive().unwrap();
    assert_eq!(coop.num_states(), 55);
    assert_eq!(fun.num_states(), 110);
    assert_eq!(prism.num_states(), 65);
    for (name, lts, init) in [
        ("coop", &coop.lts, coop.lts.initial().unwrap()),
        ("fun", &fun.lts, fun.lts.initial().unwrap()),
        ("prism", &prism.lts, prism.lts.initial().unwrap()),
    ] {
        // 9 susceptibles, 1 infected: 9 * 0.3 * 1.
        let infect = total_rate(lts, init, "infect");
        assert!(
            (infect - 2.7).abs() < 1e-9,
            "{name}: initial infect pressure {infect}"
        );
    }
}

/// Classic cooperation scales with susceptibles only; the functional and
/// PRISM versions scale with both populations (mass action).
#[test]
fn cooperation_scales_with_s_only() {
    let coop = parse_pepa(SIR_COOP).unwrap().derive().unwrap();
    for s in coop.lts.states() {
        let (mut n_s, mut n_i) = (0u32, 0u32);
        for leaf in 0..coop.leaves().len() {
            for (deriv, n) in coop.local_counts(s, leaf) {
                match deriv.as_str() {
                    "S" => n_s += n,
                    "I" => n_i += n,
                    _ => {}
                }
            }
        }
        let infect = total_rate(&coop.lts, s, "infect");
        assert!(
            (infect - 0.3 * n_s as f64).abs() < 1e-9,
            "state {}: infect {infect} != 0.3*{n_s} (I={n_i})",
            coop.state_name(s)
        );
    }
}

/// The functional PEPA chain and the PRISM chain predict the same
/// expected infected count over 30 days (Monte-Carlo-free: exact
/// transients, aggregated per state).
#[test]
fn functional_and_prism_transients_agree() {
    let fun = parse_pepa(SIR_FUN).unwrap().derive().unwrap();
    let prism = parse_prism(SIR_PRISM).unwrap().derive().unwrap();
    let expected_infected =
        |lts: &Wlts<Act<String>, Real>, infected: &dyn Fn(StateId) -> f64, t: f64| {
            let ctmc = Ctmc::from_wlts(lts);
            ctmc.transient(&ctmc.initial_distribution(), t, 1e-10)
                .into_iter()
                .enumerate()
                .map(|(i, p)| p * infected(StateId(i as u32)))
                .sum::<f64>()
        };
    let fun_i = |s: StateId| {
        (0..fun.leaves().len())
            .flat_map(|leaf| fun.local_counts(s, leaf))
            .filter(|(d, _)| d == "I")
            .map(|(_, n)| n as f64)
            .sum()
    };
    let prism_i = |s: StateId| prism.value(s, "i").expect("i") as f64;
    for t in [1.0, 5.0, 15.0, 30.0] {
        let a = expected_infected(&fun.lts, &fun_i, t);
        let b = expected_infected(&prism.lts, &prism_i, t);
        assert!(
            (a - b).abs() < 1e-6,
            "t = {t}: functional E[I]={a}, prism E[I]={b}"
        );
    }
}

/// The fluid (mean-field) ODE conserves the population and tracks the
/// exact expected infected count qualitatively (mean-field is an
/// approximation, so the band is loose).
#[test]
fn fluid_conserves_and_tracks_exact() {
    use bisim_stochastic::integrate;

    let model = parse_pepa(SIR_FUN).unwrap().fluid_model().unwrap();
    let x0 = model.initial_counts();
    let times: Vec<f64> = vec![0.0, 1.0, 5.0, 15.0, 30.0];
    let snaps = integrate(|x| model.fluid_drift(x), &x0, &times, 0.01).unwrap();
    for (x, t) in snaps.iter().zip(&times) {
        let total: f64 = x.iter().sum();
        assert!((total - 10.0).abs() < 1e-6, "t={t}: total {total}");
        assert!(x.iter().all(|v| *v >= -1e-9), "t={t}: {x:?}");
    }
    // Slots (S,I,R | I,R): fluid E[I] vs exact E[I].
    let derived = parse_pepa(SIR_FUN).unwrap().derive().unwrap();
    let ctmc = Ctmc::from_wlts(&derived.lts);
    for (x, t) in snaps.iter().zip(&times) {
        let fluid_i = x[1] + x[3];
        let p = ctmc.transient(&ctmc.initial_distribution(), *t, 1e-10);
        let mut exact = 0.0;
        for s in derived.lts.states() {
            let n_i: u32 = (0..derived.leaves().len())
                .flat_map(|leaf| derived.local_counts(s, leaf))
                .filter(|(d, _)| d == "I")
                .map(|(_, n)| n)
                .sum();
            exact += p[s.index()] * n_i as f64;
        }
        let band = 0.15 + 0.35 * exact;
        assert!(
            (fluid_i - exact).abs() <= band,
            "t={t}: fluid {fluid_i} vs exact {exact}"
        );
    }
}
