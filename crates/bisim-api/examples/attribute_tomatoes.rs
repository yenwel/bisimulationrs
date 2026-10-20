//! Which rate explains a bed that is behind plan?
//!
//! The tomato bed from PEPA Studio, with the dated counts from its example.
//! The counts are fitted to the model's growth and loss rates, and the gap
//! in the expected number of ripe plants (plan versus fitted) is split over
//! those rates. Then a what-if: what would shorter vegetative growth do?
//!
//! The split is an attribution *within the model*. It points at the rate the
//! model needs to change to match the counts; whether that rate is what
//! actually holds the bed back needs a control plot or other evidence.
//!
//! Run: `cargo run -p bisim-api --example attribute_tomatoes`

use bisim_api::{
    Intervention, Observation, Options, TimeUnit, Timeline, attribute, fit_parameters, what_if,
};

const PLAN: &str = include_str!("../../../examples/garden/tomatoes.pepa");

fn count(date: &str, state: &str, count: u32) -> Observation {
    Observation {
        date: date.to_owned(),
        component: None,
        state: state.to_owned(),
        count,
    }
}

fn main() {
    let timeline = Timeline {
        start: "15/05/2026".to_owned(),
        unit: TimeUnit::Day,
        level: 0.95,
    };
    let observations = [
        count("05/06/2026", "Vegetative", 17),
        count("05/06/2026", "Flowering", 5),
        count("19/06/2026", "Vegetative", 12),
        count("19/06/2026", "Fruiting", 5),
        count("03/07/2026", "Vegetative", 8),
        count("03/07/2026", "Fruiting", 8),
        count("17/07/2026", "Vegetative", 6),
        count("17/07/2026", "Ripe", 1),
    ];
    let params: Vec<String> = ["veg_days", "flower_days", "fruit_days", "loss"]
        .map(String::from)
        .to_vec();
    let fit =
        fit_parameters(PLAN, &observations, &params, &timeline, Options::default()).expect("fit");
    println!(
        "fitted: {}",
        fit.explanation.split(". ").next().unwrap_or("")
    );

    let a = attribute(
        PLAN,
        &fit.fitted_source,
        None,
        "Ripe",
        70.0,
        &timeline,
        Options::default(),
    )
    .expect("attribution");
    println!(
        "\nexpected ripe plants at day 70: plan {:.1}, fitted {:.1} (gap {:+.1})",
        a.plan, a.actual, a.gap
    );
    println!(
        "{:<12} {:>8} {:>8} {:>10} {:>7}   (effect on ripe plants: shared out, and alone)",
        "rate", "plan", "fitted", "shapley", "alone"
    );
    for c in &a.contributions {
        println!(
            "{:<12} {:>8.3} {:>8.3} {:>+9.1} {:>+7.1}",
            c.name, c.plan, c.actual, c.shapley, c.alone
        );
    }

    println!(
        "\ncaution: eight dated counts pin down four rates only loosely, and the split depends on \
         the model's structure. Treat it as a hypothesis to test, e.g. against a control bed."
    );

    let w = what_if(
        PLAN,
        &[Intervention {
            param: "veg_days".to_owned(),
            value: 21.0,
        }],
        None,
        "Ripe",
        &[70.0],
        &timeline,
        Options::default(),
    )
    .expect("what-if");
    println!(
        "\nwhat if vegetative growth took 21 days instead of 28? ripe at day 70: {:.1} → {:.1}",
        w.baseline.expected[0], w.intervened.expected[0]
    );
}
