//! Is the herd within tolerance of the plan, and what does that guarantee?
//!
//! A calf is weaned, grows to cull weight, and may fall ill on the way (time
//! unit: days). The "actual" model has rates as they might be fitted from
//! records: each a few percent off the plan. (ε, δ)-bisimulation (Spork et
//! al., CAV 2025) asks for jump probabilities within ε and rates within a
//! factor e^δ, and in return bounds how much the probability of reaching
//! cull weight within a given time can differ. The bound is compared here
//! with the actual difference, computed exactly.
//!
//! Run: `cargo run --release -p bisimulationrs --example tolerance_plan`

use bisimulationrs::algebra::pepa::parse_pepa;
use bisimulationrs::core::StateId;
use bisimulationrs::metric::{common_rate, min_epsilon, reach_bound};
use bisimulationrs::stochastic::Ctmc;

fn herd(wean: f64, gain: f64, ill: f64, recover: f64) -> String {
    format!(
        "wean = {wean}; gain = {gain}; ill = {ill}; recover = {recover};
         Calf = (wean, wean).Grower;
         Grower = (gain, gain).Ready + (fall_ill, ill).Sick;
         Sick = (recover, recover).Grower;
         Ready = 0;
         Calf"
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let plan = parse_pepa(&herd(1.0 / 60.0, 1.0 / 45.0, 0.010, 0.10))?.derive()?;
    let actual = parse_pepa(&herd(1.0 / 57.0, 1.0 / 47.0, 0.011, 0.095))?.derive()?;
    // The label every state carries: has the animal reached cull weight?
    let ready = |m: &bisimulationrs::algebra::pepa::DerivedModel<_>| {
        let names: Vec<bool> = m.lts.states().map(|s| m.state_name(s) == "Ready").collect();
        move |s: StateId| names[s.index()]
    };
    let (lp, la) = (ready(&plan), ready(&actual));
    let q = common_rate(&plan.lts, &actual.lts);

    let reach = |m: &bisimulationrs::algebra::pepa::DerivedModel<_>, days: f64| {
        let ctmc = Ctmc::from_wlts(&m.lts);
        let targets: Vec<bool> = m.lts.states().map(|s| m.state_name(s) == "Ready").collect();
        ctmc.reach_bounded(&ctmc.initial_distribution(), &targets, days, 1e-10)
    };

    for delta in [0.05f64, 0.1, 0.2] {
        let Some(eps) = min_epsilon(&plan.lts, &actual.lts, &lp, &la, delta)? else {
            println!(
                "δ = {delta}: rates differ by more than a factor {:.3}",
                delta.exp()
            );
            continue;
        };
        println!(
            "rates within a factor {:.3} (δ = {delta}): smallest ε = {eps:.4}",
            delta.exp()
        );
        for days in [30.0, 90.0, 180.0] {
            let bound = reach_bound(eps, delta, q, days);
            let diff = (reach(&plan, days) - reach(&actual, days)).abs();
            println!(
                "  P(at cull weight within {days:>3} days): plan {:.3}, actual {:.3}, \
                 difference {diff:.4} ≤ guaranteed {bound:.4}",
                reach(&plan, days),
                reach(&actual, days)
            );
        }
    }
    println!(
        "\nThe guarantee holds for any model within the tolerance, not just this one, which \
         is why it is looser than the actual difference; it also grows with the horizon."
    );
    Ok(())
}
