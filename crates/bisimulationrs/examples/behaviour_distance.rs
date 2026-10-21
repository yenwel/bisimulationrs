//! How far apart do two systems behave? A number, not just yes or no.
//!
//! A guild bed (companion plants and ladybirds) against a monoculture bed,
//! both from `examples/permaculture/` but with small populations so the
//! exact state spaces stay small. The distance is the optimal-transport
//! bisimulation metric (Calo et al., NeurIPS 2024): run both beds side by
//! side in the way that makes them look most alike, and average how far
//! apart the number of healthy plants is, with differences 4 weeks ahead
//! counting half. It also lists the situations that contribute most.
//!
//! The same monoculture written plant by plant (6 separate components, 729
//! states) against the counted model (28 states) gives 0: how the model is
//! written does not change its behaviour.
//!
//! Run: `cargo run --release -p bisimulationrs --example behaviour_distance`

use bisimulationrs::algebra::pepa::parse_pepa;
use bisimulationrs::core::StateId;
use bisimulationrs::metric::{MetricOptions, bisim_distance};

/// The model files with `Healthy[60]` and `Resting[12]` scaled down.
fn small(source: &str) -> String {
    source
        .replace("Healthy[60]", "Healthy[6]")
        .replace("Resting[12]", "Resting[2]")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let guild = parse_pepa(&small(include_str!(
        "../../../examples/permaculture/guild.pepa"
    )))?
    .derive()?;
    let mono = parse_pepa(&small(include_str!(
        "../../../examples/permaculture/monoculture.pepa"
    )))?
    .derive()?;
    // The observation: healthy plants (leaf 0 is the plant array in both).
    let healthy = |m: &bisimulationrs::algebra::pepa::DerivedModel<_>, s: StateId| {
        m.local_counts(s, 0)
            .into_iter()
            .filter(|(d, _)| d == "Healthy")
            .map(|(_, n)| f64::from(n))
            .sum::<f64>()
    };
    let (g0, m0) = (guild.lts.initial().unwrap(), mono.lts.initial().unwrap());
    println!(
        "guild: {} states, monoculture: {} states (time unit: weeks)",
        guild.lts.num_states(),
        mono.lts.num_states()
    );

    for half_life in [1.0, 4.0, 12.0] {
        let d = bisim_distance(
            (&guild.lts, g0),
            (&mono.lts, m0),
            |s| healthy(&guild, s),
            |s| healthy(&mono, s),
            half_life,
            MetricOptions::default(),
        )?;
        println!(
            "half-life {half_life:>4} weeks: distance {:.3} (± {:.0e}), i.e. {:.1} of 6 plants",
            d.value,
            d.error_bound,
            d.value * 6.0
        );
        if half_life == 4.0 {
            println!("  situations contributing most:");
            for p in d.pairs.iter().take(4) {
                println!(
                    "    guild {:<38} vs mono {:<26} gap {:.2}, time share {:.3}",
                    guild.state_name(StateId(p.left as u32)),
                    mono.state_name(StateId(p.right as u32)),
                    p.gap,
                    p.occupancy
                );
            }
        }
    }

    // The monoculture plant by plant: six separate components.
    let mono_source = include_str!("../../../examples/permaculture/monoculture.pepa");
    let each = mono_source.replace("Healthy[60]", &["Healthy"; 6].join(" || "));
    let each = parse_pepa(&each)?.derive()?;
    let healthy_all = |s: StateId| {
        (0..6)
            .flat_map(|leaf| each.local_counts(s, leaf))
            .filter(|(d, _)| d == "Healthy")
            .map(|(_, n)| f64::from(n))
            .sum::<f64>()
    };
    let d = bisim_distance(
        (&each.lts, each.lts.initial().unwrap()),
        (&mono.lts, m0),
        healthy_all,
        |s| healthy(&mono, s),
        4.0,
        MetricOptions::default(),
    )?;
    println!(
        "\nmonoculture plant by plant ({} states) vs counted ({} states): distance {:.1e}",
        each.lts.num_states(),
        mono.lts.num_states(),
        d.value
    );
    Ok(())
}
