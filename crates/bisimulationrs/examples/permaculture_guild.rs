//! Permaculture: does a guild beat a monoculture against aphids?
//!
//! Two beds of 60 plants. In the monoculture nothing opposes the aphids. In
//! the guild, companion plants repel part of the pressure and ladybirds
//! (which hunt where there is prey) clear infested plants. The model files
//! are in `examples/permaculture/`; both are solved with the fluid
//! (mean-field) approximation and the expected plants lost per season are
//! compared.
//!
//! Run: `cargo run -p bisimulationrs --example permaculture_guild`

use bisimulationrs::algebra::pepa::parse_pepa;
use bisimulationrs::stochastic::integrate;

/// Expected count of each local state at the given times.
fn expected(
    source: &str,
    times: &[f64],
) -> Result<Vec<std::collections::BTreeMap<String, f64>>, Box<dyn std::error::Error>> {
    let model = parse_pepa(source)?.fluid_model()?;
    let snapshots = integrate(
        |x| model.fluid_drift(x),
        &model.initial_counts(),
        times,
        0.05,
    )?;
    let slots = model.fluid_slots();
    Ok(snapshots
        .into_iter()
        .map(|x| {
            let mut counts = std::collections::BTreeMap::new();
            for ((_, state), value) in slots.iter().zip(x) {
                *counts.entry(state.clone()).or_insert(0.0) += value;
            }
            counts
        })
        .collect())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let weeks = [0.0, 4.0, 8.0, 12.0, 18.0, 26.0];
    let mono = expected(
        include_str!("../../../examples/permaculture/monoculture.pepa"),
        &weeks,
    )?;
    let guild = expected(
        include_str!("../../../examples/permaculture/guild.pepa"),
        &weeks,
    )?;

    println!("expected plants out of 60 (and ladybirds hunting out of 12)\n");
    println!("{:>6} | {:>22} | {:>34}", "week", "monoculture", "guild");
    println!(
        "{:>6} | {:>7} {:>7} {:>6} | {:>7} {:>7} {:>6} {:>10}",
        "", "healthy", "infest.", "lost", "healthy", "infest.", "lost", "hunting"
    );
    for (i, week) in weeks.iter().enumerate() {
        let (m, g) = (&mono[i], &guild[i]);
        println!(
            "{week:>6.0} | {:>7.1} {:>7.1} {:>6.1} | {:>7.1} {:>7.1} {:>6.1} {:>10.1}",
            m["Healthy"],
            m["Infested"],
            m["Lost"],
            g["Healthy"],
            g["Infested"],
            g["Lost"],
            g["Hunting"],
        );
    }
    let saved = mono.last().unwrap()["Lost"] - guild.last().unwrap()["Lost"];
    println!(
        "\nplants saved by the guild after {} weeks: {saved:.1}",
        weeks.last().unwrap()
    );
    Ok(())
}
